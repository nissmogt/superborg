/* Loading .borg files: memory map, header parsing, tensor directory, config. */
#include "internal.h"

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

void borg_set_err(char *err, int errlen, const char *fmt, ...) {
    if (!err || errlen <= 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, (size_t)errlen, fmt, ap);
    va_end(ap);
}

double borg_time_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

const char *borg_version(void) { return BORG_VERSION; }

/* ---- file mapping --------------------------------------------------------- */
static int map_file(const char *path, borg_model *m, char *err, int errlen) {
#if defined(_WIN32)
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) { borg_set_err(err, errlen, "cannot open %s", path); return -1; }
    LARGE_INTEGER size;
    GetFileSizeEx(f, &size);
    HANDLE mh = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(f);
    if (!mh) { borg_set_err(err, errlen, "cannot map %s", path); return -1; }
    m->map = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mh);
    if (!m->map) { borg_set_err(err, errlen, "cannot map %s", path); return -1; }
    m->map_size = (size_t)size.QuadPart;
    return 0;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { borg_set_err(err, errlen, "cannot open %s", path); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); borg_set_err(err, errlen, "cannot stat %s", path); return -1; }
    m->map_size = (size_t)st.st_size;
#if !defined(BORG_NO_MMAP)
    void *p = mmap(NULL, m->map_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p != MAP_FAILED) {
        close(fd);
        m->map = p;
#if defined(MADV_WILLNEED)
        madvise(p, m->map_size, MADV_WILLNEED);
#endif
        return 0;
    }
#endif
    /* fallback: read the whole file */
    m->map = malloc(m->map_size);
    if (!m->map) { close(fd); borg_set_err(err, errlen, "out of memory"); return -1; }
    size_t got = 0;
    while (got < m->map_size) {
        ssize_t r = read(fd, (char *)m->map + got, m->map_size - got);
        if (r <= 0) {
            close(fd);
            free(m->map);
            m->map = NULL;
            borg_set_err(err, errlen, "read error on %s", path);
            return -1;
        }
        got += (size_t)r;
    }
    close(fd);
    m->map_is_malloc = 1;
    return 0;
#endif
}

static void unmap_file(borg_model *m) {
    if (!m->map) return;
    if (m->map_is_malloc == 2) return; /* caller's buffer (borg_model_load_memory) */
    if (m->map_is_malloc) { free(m->map); return; }
#if defined(_WIN32)
    UnmapViewOfFile(m->map);
#else
    munmap(m->map, m->map_size);
#endif
}

/* Header numbers come from an untrusted file: convert only integral values within [lo, hi]. */
static int json_int(const json_value *v, double dflt, double lo, double hi, int64_t *out) {
    const double x = json_num(v, dflt);
    if (!(x >= lo && x <= hi) || x != floor(x)) return -1;
    *out = (int64_t)x;
    return 0;
}

static int json_int32(const json_value *v, int dflt, int lo, int *out) {
    int64_t x;
    if (json_int(v, dflt, lo, INT_MAX, &x)) return -1;
    *out = (int)x;
    return 0;
}

/* ---- tensors -------------------------------------------------------------- */
static borg_dtype parse_dtype(const char *s) {
    static const char *names[DT_COUNT] = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "u8", "u32", "i32"};
    for (int i = 0; i < DT_COUNT; i++)
        if (!strcmp(s, names[i])) return (borg_dtype)i;
    return DT_COUNT;
}

const borg_tensor *model_find_tensor(const borg_model *m, const char *name) {
    for (int i = 0; i < m->n_tensors; i++)
        if (!strcmp(m->tensors[i].name, name)) return &m->tensors[i];
    return NULL;
}

/* In 64 bits, so a size that does not fit a 32-bit size_t (wasm32) cannot wrap to a small one;
   UINT64_MAX (matches no tensor) when the shape's element count passes 2^60. */
static uint64_t tensor_expected_bytes(const borg_tensor *t) {
    uint64_t n = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] && n > ((uint64_t)1 << 60) / (uint64_t)t->shape[i]) return UINT64_MAX;
        n *= (uint64_t)t->shape[i];
    }
    switch (t->dtype) {
    case DT_F32: case DT_U32: case DT_I32: return n * 4;
    case DT_F16: case DT_BF16: return n * 2;
    case DT_U8: return n;
    case DT_Q8_0: return n / QK * sizeof(block_q8_0);
    case DT_Q4_0: return n / QK * sizeof(block_q4_0);
    case DT_Q4_1: return n / QK * sizeof(block_q4_1);
    default: return 0;
    }
}

static int get_mat(borg_model *m, const char *name, int rows, int cols, borg_mat *out, char *err, int errlen) {
    const borg_tensor *t = model_find_tensor(m, name);
    if (!t) { borg_set_err(err, errlen, "missing tensor %s", name); return -1; }
    if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols) {
        borg_set_err(err, errlen, "tensor %s has shape [%lld, %lld], expected [%d, %d]", name,
                     (long long)t->shape[0], (long long)(t->ndim > 1 ? t->shape[1] : 0), rows, cols);
        return -1;
    }
    if (t->dtype > DT_Q4_1) { borg_set_err(err, errlen, "tensor %s has a non-float dtype", name); return -1; }
    if ((t->dtype == DT_Q8_0 || t->dtype == DT_Q4_0 || t->dtype == DT_Q4_1) && cols % QK) {
        borg_set_err(err, errlen, "tensor %s: quantized rows must be a multiple of %d", name, QK);
        return -1;
    }
    out->dtype = t->dtype;
    out->rows = rows;
    out->cols = cols;
    out->data = t->data;
    out->row_bytes = dtype_row_bytes(t->dtype, cols);
    return 0;
}

static const float *get_vec(borg_model *m, const char *name, int n, int optional, char *err, int errlen) {
    const borg_tensor *t = model_find_tensor(m, name);
    if (!t) {
        if (!optional) borg_set_err(err, errlen, "missing tensor %s", name);
        return NULL;
    }
    if (t->dtype != DT_F32 || t->ndim != 1 || t->shape[0] != n) {
        borg_set_err(err, errlen, "tensor %s must be f32 [%d]", name, n);
        return NULL;
    }
    return (const float *)t->data;
}

static const void *get_table(borg_model *m, const char *name, borg_dtype dt, int64_t n, char *err, int errlen) {
    const borg_tensor *t = model_find_tensor(m, name);
    if (!t || t->dtype != dt) { borg_set_err(err, errlen, "missing or mistyped tokenizer table %s", name); return NULL; }
    int64_t count = 1;
    for (int i = 0; i < t->ndim; i++) count *= t->shape[i];
    if (n >= 0 && count != n) { borg_set_err(err, errlen, "tokenizer table %s has %lld entries, expected %lld",
                                             name, (long long)count, (long long)n); return NULL; }
    return t->data;
}

static int load_tokenizer(borg_model *m, const json_value *tj, char *err, int errlen) {
    borg_tokenizer *t = &m->tok;
    const borg_tensor *off = model_find_tensor(m, "tokenizer.vocab_offsets");
    if (!off) { borg_set_err(err, errlen, "missing tokenizer tables"); return -1; }
    const borg_tensor *mg = model_find_tensor(m, "tokenizer.merges");
    if (off->ndim != 1 || off->shape[0] < 2 || off->shape[0] > (1 << 28) ||
        (mg && (mg->ndim != 2 || mg->shape[1] != 3 || mg->shape[0] > (1 << 28)))) {
        borg_set_err(err, errlen, "invalid tokenizer table shapes");
        return -1;
    }
    t->n_vocab = (int)off->shape[0] - 1;
    t->n_merges = mg ? (int)mg->shape[0] : 0;
    t->vocab_off = (const uint32_t *)get_table(m, "tokenizer.vocab_offsets", DT_U32, t->n_vocab + 1, err, errlen);
    t->token_type = (const uint8_t *)get_table(m, "tokenizer.token_type", DT_U8, t->n_vocab, err, errlen);
    t->vocab = (const uint8_t *)get_table(m, "tokenizer.vocab", DT_U8, -1, err, errlen);
    if (!t->vocab_off || !t->token_type || !t->vocab) return -1;
    const borg_tensor *vt = model_find_tensor(m, "tokenizer.vocab");
    if ((size_t)t->vocab_off[t->n_vocab] > vt->nbytes) { borg_set_err(err, errlen, "tokenizer offsets out of range"); return -1; }
    for (int i = 0; i < t->n_vocab; i++)
        if (t->vocab_off[i] > t->vocab_off[i + 1]) { borg_set_err(err, errlen, "tokenizer offsets not monotonic"); return -1; }
    if (t->n_merges) {
        t->merges = (const uint32_t *)get_table(m, "tokenizer.merges", DT_U32, (int64_t)t->n_merges * 3, err, errlen);
        if (!t->merges) return -1;
        for (int64_t i = 0; i < (int64_t)t->n_merges * 3; i++)
            if (t->merges[i] >= (uint32_t)t->n_vocab) { borg_set_err(err, errlen, "merge references unknown token"); return -1; }
    }
    t->byte_level = json_bool(json_get(tj, "byte_level"), 1);
    const char *pt = json_str(json_get(tj, "pretokenizer"), "none");
    if (!strcmp(pt, "gpt2")) t->pretok = PRETOK_GPT2;
    else if (!strcmp(pt, "llama3")) t->pretok = PRETOK_LLAMA3;
    else if (!strcmp(pt, "qwen2")) t->pretok = PRETOK_QWEN2;
    else if (!strcmp(pt, "smollm")) t->pretok = PRETOK_SMOLLM;
    else if (!strcmp(pt, "none")) t->pretok = PRETOK_NONE;
    else { borg_set_err(err, errlen, "unknown pre-tokenizer %s", pt); return -1; }
    t->nfc = json_bool(json_get(tj, "nfc"), 0);
    t->spm_add_prefix_space = json_bool(json_get(tj, "spm_add_prefix_space"), 0);
    t->byte_fallback = json_bool(json_get(tj, "byte_fallback"), 0);
    t->ignore_merges = json_bool(json_get(tj, "ignore_merges"), 0);
    t->add_bos = json_bool(json_get(tj, "add_bos"), 0);
    t->add_eos = json_bool(json_get(tj, "add_eos"), 0);
    if (json_int32(json_get(tj, "bos_id"), -1, -1, &t->bos_id) || json_int32(json_get(tj, "eos_id"), -1, -1, &t->eos_id) ||
        json_int32(json_get(tj, "pad_id"), -1, -1, &t->pad_id) || json_int32(json_get(tj, "unk_id"), -1, -1, &t->unk_id)) {
        borg_set_err(err, errlen, "invalid tokenizer special token id");
        return -1;
    }
    if (t->bos_id >= t->n_vocab) t->bos_id = -1;
    if (t->eos_id >= t->n_vocab) t->eos_id = -1;
    if (t->pad_id >= t->n_vocab) t->pad_id = -1;
    if (t->unk_id >= t->n_vocab) t->unk_id = -1;
    const json_value *eos = json_get(tj, "eos_ids");
    if (eos && eos->type == JS_ARR)
        for (int i = 0; i < eos->len && t->n_eos_ids < 16; i++) {
            int id;
            if (!json_int32(&eos->items[i], -1, 0, &id) && id < t->n_vocab) t->eos_ids[t->n_eos_ids++] = id;
        }
    if (t->n_eos_ids == 0 && t->eos_id >= 0) t->eos_ids[t->n_eos_ids++] = t->eos_id;
    const json_value *added = json_get(tj, "added_tokens");
    if (added && added->type == JS_ARR && added->len > 0) {
        t->added = (int32_t *)malloc(sizeof(int32_t) * (size_t)added->len);
        if (!t->added) { borg_set_err(err, errlen, "out of memory"); return -1; }
        for (int i = 0; i < added->len; i++) {
            int id;
            if (!json_int32(&added->items[i], -1, 0, &id) && id < t->n_vocab) t->added[t->n_added++] = id;
        }
    }
    return tokenizer_init(t, err, errlen);
}

void borg_model_free(borg_model *m) {
    if (!m) return;
    tokenizer_free(&m->tok);
    free(m->tok.added);
    free(m->layers);
    if (m->tensors)
        for (int i = 0; i < m->n_tensors; i++) free((void *)m->tensors[i].name);
    free(m->tensors);
    free(m->header_json);
    unmap_file(m);
    free(m);
}

/* Parse and validate a model whose bytes are already at m->map; frees m on failure. */
static borg_model *load_mapped(borg_model *m, const char *path, char *err, int errlen) {
    json_value *root = NULL;
    const uint8_t *base = (const uint8_t *)m->map;
    if (m->map_size < 16 || memcmp(base, "BORG", 4) != 0) { borg_set_err(err, errlen, "%s is not a .borg file", path); goto fail; }
    uint32_t version;
    uint64_t hlen;
    memcpy(&version, base + 4, 4);
    memcpy(&hlen, base + 8, 8);
    if (version != 1) { borg_set_err(err, errlen, "unsupported .borg version %u", version); goto fail; }
    if (hlen > m->map_size - 16) { borg_set_err(err, errlen, "truncated header"); goto fail; }
    m->header_json = (char *)malloc(hlen + 1);
    if (!m->header_json) { borg_set_err(err, errlen, "out of memory"); goto fail; }
    memcpy(m->header_json, base + 16, hlen);
    m->header_json[hlen] = 0;
    root = json_parse(m->header_json, hlen, err, errlen);
    if (!root) goto fail;
    const uint64_t data_start = (16 + hlen + 63) / 64 * 64;

    /* tensors */
    const json_value *tens = json_get(root, "tensors");
    if (!tens || tens->type != JS_OBJ) { borg_set_err(err, errlen, "header has no tensors"); goto fail; }
    m->tensors = (borg_tensor *)calloc((size_t)tens->len, sizeof(borg_tensor));
    if (!m->tensors) { borg_set_err(err, errlen, "out of memory"); goto fail; }
    for (int i = 0; i < tens->len; i++) {
        const json_value *tv = &tens->items[i];
        borg_tensor *t = &m->tensors[m->n_tensors];
        t->name = strdup(tens->keys[i]);
        m->n_tensors++;
        t->dtype = parse_dtype(json_str(json_get(tv, "dtype"), ""));
        if (t->dtype == DT_COUNT) { borg_set_err(err, errlen, "tensor %s: unknown dtype", t->name); goto fail; }
        const json_value *shape = json_get(tv, "shape");
        if (!shape || shape->type != JS_ARR || shape->len < 1 || shape->len > 4) {
            borg_set_err(err, errlen, "tensor %s: bad shape", t->name); goto fail;
        }
        t->ndim = shape->len;
        for (int d = 0; d < t->ndim; d++) /* element counts are checked by tensor_expected_bytes */
            if (json_int(&shape->items[d], -1, 0, (double)INT_MAX, &t->shape[d])) {
                borg_set_err(err, errlen, "tensor %s: bad shape", t->name); goto fail;
            }
        int64_t off64, nb64; /* each at most map_size: the 64-bit sum cannot wrap */
        if (json_int(json_get(tv, "offset"), -1, 0, (double)m->map_size, &off64) ||
            json_int(json_get(tv, "nbytes"), -1, 0, (double)m->map_size, &nb64) ||
            data_start + (uint64_t)off64 + (uint64_t)nb64 > (uint64_t)m->map_size) {
            borg_set_err(err, errlen, "tensor %s lies outside the file", t->name); goto fail;
        }
        const size_t off = (size_t)off64, nb = (size_t)nb64;
        if (off % 64) { /* the format guarantees it; kernels rely on aligned rows */
            borg_set_err(err, errlen, "tensor %s is not 64-byte aligned", t->name); goto fail;
        }
        t->nbytes = nb;
        if (tensor_expected_bytes(t) != (uint64_t)nb64) {
            borg_set_err(err, errlen, "tensor %s: size %zu does not match its shape", t->name, t->nbytes); goto fail;
        }
        t->data = base + (size_t)data_start + off;
    }

    /* config */
    const json_value *cj = json_get(root, "config");
    if (!cj) { borg_set_err(err, errlen, "header has no config"); goto fail; }
    borg_config *c = &m->cfg;
    /* bounds keep every derived product (n_heads * head_dim, n_batch * width, ...) inside int */
    if (json_int32(json_get(cj, "dim"), 0, 1, &c->dim) || json_int32(json_get(cj, "n_layers"), 0, 1, &c->n_layers) ||
        json_int32(json_get(cj, "n_heads"), 0, 1, &c->n_heads) ||
        json_int32(json_get(cj, "n_kv_heads"), c->n_heads, 1, &c->n_kv_heads) ||
        json_int32(json_get(cj, "head_dim"), c->n_heads ? c->dim / c->n_heads : 0, 1, &c->head_dim) ||
        json_int32(json_get(cj, "hidden_dim"), 0, 1, &c->hidden_dim) ||
        json_int32(json_get(cj, "vocab_size"), 0, 1, &c->vocab_size) ||
        json_int32(json_get(cj, "max_seq_len"), 2048, 1, &c->max_seq_len) || c->dim > (1 << 20) ||
        c->hidden_dim > (1 << 22) || c->n_layers > 4096 || c->n_heads > 4096 || c->vocab_size > (1 << 24) ||
        (int64_t)c->n_heads * c->head_dim > (1 << 20)) {
        borg_set_err(err, errlen, "invalid model config"); goto fail;
    }
    c->norm_eps = (float)json_num(json_get(cj, "norm_eps"), 1e-5);
    c->rope_theta = (float)json_num(json_get(cj, "rope_theta"), 10000.0);
    c->attn_bias = json_bool(json_get(cj, "attn_bias"), 0);
    c->qk_norm = json_bool(json_get(cj, "qk_norm"), 0);
    c->tie_embeddings = json_bool(json_get(cj, "tie_embeddings"), 0);
    if (c->dim <= 0 || c->n_layers <= 0 || c->n_heads <= 0 || c->n_kv_heads <= 0 || c->head_dim <= 0 ||
        c->hidden_dim <= 0 || c->vocab_size <= 0 || c->n_heads % c->n_kv_heads || c->head_dim % 2 ||
        c->head_dim > 1024) {
        borg_set_err(err, errlen, "invalid model config"); goto fail;
    }
    const char *act = json_str(json_get(cj, "act"), "silu");
    if (!strcmp(act, "silu")) m->act = ACT_SILU;
    else if (!strcmp(act, "gelu_tanh")) m->act = ACT_GELU_TANH;
    else { borg_set_err(err, errlen, "unsupported activation %s", act); goto fail; }
    const json_value *rs = json_get(cj, "rope_scaling");
    if (rs && rs->type == JS_OBJ) {
        const char *rt = json_str(json_get(rs, "type"), "");
        m->rope_factor = (float)json_num(json_get(rs, "factor"), 1.0);
        if (!strcmp(rt, "linear")) m->rope_scale_type = ROPE_SCALE_LINEAR;
        else if (!strcmp(rt, "llama3")) {
            m->rope_scale_type = ROPE_SCALE_LLAMA3;
            m->rope_low_freq_factor = (float)json_num(json_get(rs, "low_freq_factor"), 1.0);
            m->rope_high_freq_factor = (float)json_num(json_get(rs, "high_freq_factor"), 4.0);
            if (json_int32(json_get(rs, "original_max_position_embeddings"), 8192, 1, &m->rope_orig_max_pos)) {
                borg_set_err(err, errlen, "invalid rope scaling"); goto fail;
            }
        } else { borg_set_err(err, errlen, "unsupported rope scaling %s", rt); goto fail; }
    }

    /* weights */
    int D = c->dim, HD = c->head_dim, NQ = c->n_heads * HD, NKV = c->n_kv_heads * HD, F = c->hidden_dim;
    if (get_mat(m, "tok_embd", c->vocab_size, D, &m->tok_embd, err, errlen)) goto fail;
    if (c->tie_embeddings) m->output = m->tok_embd;
    else if (get_mat(m, "output", c->vocab_size, D, &m->output, err, errlen)) goto fail;
    if (!(m->output_norm = get_vec(m, "output_norm", D, 0, err, errlen))) goto fail;
    m->layers = (borg_layer *)calloc((size_t)c->n_layers, sizeof(borg_layer));
    if (!m->layers) { borg_set_err(err, errlen, "out of memory"); goto fail; }
    for (int l = 0; l < c->n_layers; l++) {
        borg_layer *L = &m->layers[l];
        char name[64];
#define MAT(field, suffix, r, cl) \
    snprintf(name, sizeof name, "blk.%d." suffix, l); \
    if (get_mat(m, name, r, cl, &L->field, err, errlen)) goto fail;
#define VEC(field, suffix, n, opt) \
    snprintf(name, sizeof name, "blk.%d." suffix, l); \
    L->field = get_vec(m, name, n, opt, err, errlen); \
    if (!L->field && (!(opt) || (err && err[0]))) goto fail;
        MAT(wq, "attn_q", NQ, D)
        MAT(wk, "attn_k", NKV, D)
        MAT(wv, "attn_v", NKV, D)
        MAT(wo, "attn_o", D, NQ)
        MAT(w_gate, "ffn_gate", F, D)
        MAT(w_up, "ffn_up", F, D)
        MAT(w_down, "ffn_down", D, F)
        VEC(attn_norm, "attn_norm", D, 0)
        VEC(ffn_norm, "ffn_norm", D, 0)
        if (c->attn_bias) {
            VEC(bq, "attn_q.bias", NQ, 0)
            VEC(bk, "attn_k.bias", NKV, 0)
            VEC(bv, "attn_v.bias", NKV, 0)
        }
        if (c->qk_norm) {
            VEC(q_norm, "attn_q_norm", HD, 0)
            VEC(k_norm, "attn_k_norm", HD, 0)
        }
#undef MAT
#undef VEC
    }
    for (int i = 0; i < m->n_tensors; i++)
        if (strncmp(m->tensors[i].name, "tokenizer.", 10)) c->weight_bytes += m->tensors[i].nbytes;

    /* tokenizer */
    const json_value *tj = json_get(root, "tokenizer");
    if (!tj) { borg_set_err(err, errlen, "header has no tokenizer"); goto fail; }
    if (load_tokenizer(m, tj, err, errlen)) goto fail;
    c->n_vocab = m->tok.n_vocab < c->vocab_size ? m->tok.n_vocab : c->vocab_size;
    c->bos_id = m->tok.bos_id;
    c->eos_id = m->tok.eos_id;
    c->add_bos = m->tok.add_bos;

    json_free(root);
    return m;
fail:
    if (err && errlen > 0 && !err[0]) borg_set_err(err, errlen, "failed to load %s", path);
    json_free(root);
    borg_model_free(m);
    return NULL;
}

borg_model *borg_model_load(const char *path, char *err, int errlen) {
    if (err && errlen > 0) err[0] = 0;
    borg_model *m = (borg_model *)calloc(1, sizeof(borg_model));
    if (!m) { borg_set_err(err, errlen, "out of memory"); return NULL; }
    if (map_file(path, m, err, errlen)) {
        borg_model_free(m);
        return NULL;
    }
    return load_mapped(m, path, err, errlen);
}

borg_model *borg_model_load_memory(const void *data, size_t size, char *err, int errlen) {
    if (err && errlen > 0) err[0] = 0;
    borg_model *m = (borg_model *)calloc(1, sizeof(borg_model));
    if (!m) { borg_set_err(err, errlen, "out of memory"); return NULL; }
    m->map = (void *)data;
    m->map_size = size;
    m->map_is_malloc = 2;
    return load_mapped(m, "<memory>", err, errlen);
}

void borg_model_config(const borg_model *m, borg_config *out) { *out = m->cfg; }
const char *borg_model_header(const borg_model *m) { return m->header_json; }
