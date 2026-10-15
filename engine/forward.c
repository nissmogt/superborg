/* The transformer forward pass, KV cache and context management.
 *
 * Per layer (pre-norm decoder block, as in Llama / Qwen2 / Qwen3):
 *   h  = RMSNorm(x) * w_attn_norm
 *   q, k, v = Wq h (+bq), Wk h (+bk), Wv h (+bv)
 *   q, k = RMSNorm_per_head(q, k) * w_q_norm, w_k_norm          (Qwen3 only)
 *   q, k = RoPE(q, k, pos)                                       (rotate-half form)
 *   x += Wo softmax(q k^T / sqrt(head_dim) + causal mask) v      (grouped-query attention)
 *   h  = RMSNorm(x) * w_ffn_norm
 *   x += Wdown (act(Wgate h) * Wup h)
 * logits = Wout (RMSNorm(x) * w_out_norm)
 *
 * A chunk of up to n_batch tokens goes through every layer together, so each weight
 * row is read from memory once per chunk instead of once per token.
 */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_JOBS 3
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- context lifecycle ---------------------------------------------------- */
borg_ctx_params borg_ctx_default_params(void) {
    borg_ctx_params p = {0, 0, 0};
    return p;
}

static void compute_inv_freq(const borg_model *m, float *inv_freq) {
    const int hd = m->cfg.head_dim;
    const float base = m->cfg.rope_theta;
    for (int i = 0; i < hd / 2; i++) {
        /* same float32 arithmetic as transformers' default RoPE init */
        float e = (float)(2 * i) / (float)hd;
        float f = 1.0f / (float)pow((double)base, (double)e);
        if (m->rope_scale_type == ROPE_SCALE_LINEAR) {
            f /= m->rope_factor;
        } else if (m->rope_scale_type == ROPE_SCALE_LLAMA3) {
            const float factor = m->rope_factor;
            const float old_len = (float)m->rope_orig_max_pos;
            const float low_wl = old_len / m->rope_low_freq_factor;
            const float high_wl = old_len / m->rope_high_freq_factor;
            const float wl = 2.0f * (float)M_PI / f;
            float g = wl > low_wl ? f / factor : f;
            if (!(wl < high_wl) && !(wl > low_wl)) {
                float smooth = (old_len / wl - m->rope_low_freq_factor) /
                               (m->rope_high_freq_factor - m->rope_low_freq_factor);
                g = (1.0f - smooth) * g / factor + smooth * g;
            }
            f = g;
        }
        inv_freq[i] = f;
    }
}

void borg_ctx_free(borg_ctx *c) {
    if (!c) return;
    free(c->tokens);
    free(c->k_cache);
    free(c->v_cache);
    free(c->inv_freq);
    free(c->x);
    free(c->xb);
    free(c->xb2);
    free(c->q);
    free(c->k);
    free(c->v);
    free(c->hb);
    free(c->hb2);
    free(c->xq);
    free(c->att);
    free(c->embd);
    free(c->stats);
    free(c);
}

borg_ctx *borg_ctx_new(borg_model *m, borg_pool *pool, const borg_ctx_params *params) {
    borg_ctx_params p = params ? *params : borg_ctx_default_params();
    const borg_config *cfg = &m->cfg;
    borg_ctx *c = (borg_ctx *)calloc(1, sizeof(borg_ctx));
    if (!c) return NULL;
    c->model = m;
    c->pool = pool;
    c->n_ctx = p.n_ctx > 0 ? p.n_ctx : (cfg->max_seq_len < 4096 ? cfg->max_seq_len : 4096);
    c->n_batch = p.n_batch > 0 ? p.n_batch : 64;
    if (c->n_batch > c->n_ctx) c->n_batch = c->n_ctx;
    c->kv_f16 = p.kv_f16 ? 1 : 0;

    const int B = c->n_batch, D = cfg->dim, HD = cfg->head_dim;
    const int NQ = cfg->n_heads * HD, NKV = cfg->n_kv_heads * HD, F = cfg->hidden_dim;
    const int nthreads = borg_pool_size(pool);
    int max_cols = D;
    if (F > max_cols) max_cols = F;
    if (NQ > max_cols) max_cols = NQ;
    c->max_cols = (max_cols + QK - 1) / QK * QK;
    const size_t kv_elems = (size_t)cfg->n_layers * NKV * (size_t)c->n_ctx;
    const size_t esz = c->kv_f16 ? 2 : 4;

    c->tokens = (int32_t *)calloc((size_t)c->n_ctx, sizeof(int32_t));
    c->k_cache = calloc(kv_elems, esz);
    c->v_cache = calloc(kv_elems, esz);
    c->inv_freq = (float *)calloc((size_t)HD / 2, sizeof(float));
    c->x = (float *)calloc((size_t)B * D, sizeof(float));
    c->xb = (float *)calloc((size_t)B * (NQ > D ? NQ : D), sizeof(float));
    c->xb2 = (float *)calloc((size_t)B * D, sizeof(float));
    c->q = (float *)calloc((size_t)B * NQ, sizeof(float));
    c->k = (float *)calloc((size_t)B * NKV, sizeof(float));
    c->v = (float *)calloc((size_t)B * NKV, sizeof(float));
    c->hb = (float *)calloc((size_t)B * F, sizeof(float));
    c->hb2 = (float *)calloc((size_t)B * F, sizeof(float));
    c->xq = (block_q8a *)calloc((size_t)B * (c->max_cols / QK), sizeof(block_q8a));
    c->att = (float *)calloc((size_t)nthreads * c->n_ctx, sizeof(float));
    c->embd = (float *)calloc((size_t)D, sizeof(float));
    if (!c->tokens || !c->k_cache || !c->v_cache || !c->inv_freq || !c->x || !c->xb || !c->xb2 || !c->q ||
        !c->k || !c->v || !c->hb || !c->hb2 || !c->xq || !c->att || !c->embd) {
        borg_ctx_free(c);
        return NULL;
    }
    compute_inv_freq(m, c->inv_freq);
    return c;
}

int borg_ctx_enable_stats(borg_ctx *c, int enable) {
    if (!enable) {
        free(c->stats);
        c->stats = NULL;
        c->stats_tokens = 0;
        return 0;
    }
    if (c->stats) return 0;
    c->stats = (double *)calloc((size_t)c->model->cfg.n_layers * SITE_COUNT * c->max_cols, sizeof(double));
    c->stats_tokens = 0;
    return c->stats ? 0 : -1;
}

static int site_cols(const borg_config *cfg, int site) {
    switch (site) {
    case SITE_ATTN_OUT: return cfg->n_heads * cfg->head_dim;
    case SITE_FFN_MID: return cfg->hidden_dim;
    default: return cfg->dim;
    }
}

const double *borg_ctx_stats(const borg_ctx *c, int layer, int site, int *n_cols) {
    const borg_config *cfg = &c->model->cfg;
    if (!c->stats || layer < 0 || layer >= cfg->n_layers || site < 0 || site >= SITE_COUNT) return NULL;
    if (n_cols) *n_cols = site_cols(cfg, site);
    return c->stats + ((size_t)layer * SITE_COUNT + site) * c->max_cols;
}

int64_t borg_ctx_stats_tokens(const borg_ctx *c) { return c->stats_tokens; }

static void record_stats(borg_ctx *c, int layer, int site, const float *x, int T, int ldx) {
    if (!c->stats) return;
    const int n = site_cols(&c->model->cfg, site);
    double *s = c->stats + ((size_t)layer * SITE_COUNT + site) * c->max_cols;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++) {
            const double v = x[(size_t)t * ldx + i];
            s[i] += v * v;
        }
}

int borg_ctx_n_ctx(const borg_ctx *c) { return c->n_ctx; }
int borg_ctx_n_past(const borg_ctx *c) { return c->n_past; }
const int32_t *borg_ctx_tokens(const borg_ctx *c) { return c->tokens; }
const float *borg_ctx_embedding(const borg_ctx *c) { return c->embd; }

void borg_ctx_truncate(borg_ctx *c, int n_keep) {
    if (n_keep < 0) n_keep = 0;
    if (n_keep < c->n_past) c->n_past = n_keep;
}

/* ---- small vector ops ----------------------------------------------------- */
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    const float ss = vec_dot_f32(x, x, n) / (float)n;
    const float scale = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * scale);
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* rotate-half RoPE on one head */
static void rope_head(float *v, int hd, const float *cosv, const float *sinv) {
    const int half = hd / 2;
    for (int i = 0; i < half; i++) {
        const float a = v[i], b = v[i + half];
        v[i] = a * cosv[i] - b * sinv[i];
        v[i + half] = b * cosv[i] + a * sinv[i];
    }
}

/* ---- parallel matrix multiply --------------------------------------------- */
typedef struct {
    const borg_mat *m[MAX_JOBS];
    float *out[MAX_JOBS];
    int ldo[MAX_JOBS];
    int start[MAX_JOBS + 1]; /* prefix sums of row counts */
    int n_jobs;
    int T;
    const float *x;
    int ldx;
    const block_q8a *xq;
    int ldxq;
} mm_task;

static void mm_range(void *arg, int r0, int r1, int ith) {
    (void)ith;
    mm_task *t = (mm_task *)arg;
    for (int j = 0; j < t->n_jobs; j++) {
        const int a = r0 > t->start[j] ? r0 : t->start[j];
        const int b = r1 < t->start[j + 1] ? r1 : t->start[j + 1];
        if (a < b)
            mat_mul_rows(t->m[j], a - t->start[j], b - t->start[j], t->T, t->x, t->ldx, t->xq, t->ldxq, t->out[j],
                         t->ldo[j]);
    }
}

typedef struct {
    const float *x;
    int ldx, cols;
    block_q8a *xq;
    int ldxq;
} quant_task;

static void quant_range(void *arg, int b, int e, int ith) {
    (void)ith;
    quant_task *t = (quant_task *)arg;
    for (int i = b; i < e; i++) quantize_row_q8a(t->x + (size_t)i * t->ldx, t->xq + (size_t)i * t->ldxq, t->cols);
}

/* out_j[t, :] = M_j x[t, :] for every job j; all matrices share the input width. */
static void matmul(borg_ctx *c, int n_jobs, const borg_mat **ms, float **outs, const int *ldos, int T,
                   const float *x, int ldx) {
    mm_task t;
    memset(&t, 0, sizeof t);
    t.n_jobs = n_jobs;
    int need_q = 0;
    for (int j = 0; j < n_jobs; j++) {
        t.m[j] = ms[j];
        t.out[j] = outs[j];
        t.ldo[j] = ldos[j];
        t.start[j + 1] = t.start[j] + ms[j]->rows;
        need_q |= mat_wants_q8a(ms[j]);
    }
    t.T = T;
    t.x = x;
    t.ldx = ldx;
    if (need_q) {
        quant_task qt = {x, ldx, ms[0]->cols, c->xq, c->max_cols / QK};
        pool_for(c->pool, T, 1, quant_range, &qt);
        t.xq = c->xq;
        t.ldxq = qt.ldxq;
    }
    const int nth = borg_pool_size(c->pool);
    const int total = t.start[n_jobs];
    /* ~16 chunks per thread balances uneven cores and late workers */
    int chunk = total / (nth * 16);
    if (chunk < 4) chunk = 4;
    pool_for(c->pool, total, chunk, mm_range, &t);
}

static void matmul1(borg_ctx *c, const borg_mat *m, float *out, int ldo, int T, const float *x, int ldx) {
    const borg_mat *ms[1] = {m};
    float *outs[1] = {out};
    int ldos[1] = {ldo};
    matmul(c, 1, ms, outs, ldos, T, x, ldx);
}

/* ---- per-token work ------------------------------------------------------- */
typedef struct {
    borg_ctx *c;
    int layer, T, pos0;
} tok_task;

/* bias, q/k norm, RoPE, and store k/v into the cache */
static void qkv_post(borg_ctx *c, int l, int t, int pos) {
    const borg_model *m = c->model;
    const borg_config *cfg = &m->cfg;
    const borg_layer *L = &m->layers[l];
    const int HD = cfg->head_dim, NQ = cfg->n_heads * HD, NKV = cfg->n_kv_heads * HD;
    float *q = c->q + (size_t)t * NQ;
    float *k = c->k + (size_t)t * NKV;
    float *v = c->v + (size_t)t * NKV;
    if (L->bq) {
        for (int i = 0; i < NQ; i++) q[i] += L->bq[i];
        for (int i = 0; i < NKV; i++) k[i] += L->bk[i];
        for (int i = 0; i < NKV; i++) v[i] += L->bv[i];
    }
    if (L->q_norm) {
        for (int h = 0; h < cfg->n_heads; h++) rmsnorm(q + h * HD, q + h * HD, L->q_norm, HD, cfg->norm_eps);
        for (int h = 0; h < cfg->n_kv_heads; h++) rmsnorm(k + h * HD, k + h * HD, L->k_norm, HD, cfg->norm_eps);
    }
    float cosv[512], sinv[512];
    for (int i = 0; i < HD / 2; i++) {
        const float ang = (float)pos * c->inv_freq[i];
        cosv[i] = cosf(ang);
        sinv[i] = sinf(ang);
    }
    for (int h = 0; h < cfg->n_heads; h++) rope_head(q + h * HD, HD, cosv, sinv);
    for (int h = 0; h < cfg->n_kv_heads; h++) rope_head(k + h * HD, HD, cosv, sinv);
    const size_t layer_off = (size_t)l * NKV * c->n_ctx;
    for (int h = 0; h < cfg->n_kv_heads; h++) {
        const size_t off = layer_off + ((size_t)h * c->n_ctx + (size_t)pos) * HD;
        if (c->kv_f16) {
            uint16_t *kc = (uint16_t *)c->k_cache + off, *vc = (uint16_t *)c->v_cache + off;
            for (int i = 0; i < HD; i++) {
                kc[i] = fp32_to_fp16(k[h * HD + i]);
                vc[i] = fp32_to_fp16(v[h * HD + i]);
            }
        } else {
            memcpy((float *)c->k_cache + off, k + h * HD, (size_t)HD * sizeof(float));
            memcpy((float *)c->v_cache + off, v + h * HD, (size_t)HD * sizeof(float));
        }
    }
}

static void qkv_post_range(void *arg, int b, int e, int ith) {
    (void)ith;
    tok_task *t = (tok_task *)arg;
    for (int i = b; i < e; i++) qkv_post(t->c, t->layer, i, t->pos0 + i);
}

static void attention_range(void *arg, int b, int e, int ith) {
    tok_task *t = (tok_task *)arg;
    borg_ctx *c = t->c;
    const borg_config *cfg = &c->model->cfg;
    const int HD = cfg->head_dim, NQ = cfg->n_heads * HD, NKV = cfg->n_kv_heads * HD;
    const int group = cfg->n_heads / cfg->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)HD);
    float *att = c->att + (size_t)ith * c->n_ctx;
    const size_t layer_off = (size_t)t->layer * NKV * c->n_ctx;
    for (int item = b; item < e; item++) {
        const int tok = item / cfg->n_heads, h = item % cfg->n_heads;
        const int pos = t->pos0 + tok;
        const int kvh = h / group;
        const float *q = c->q + (size_t)tok * NQ + (size_t)h * HD;
        float *out = c->xb + (size_t)tok * NQ + (size_t)h * HD;
        const size_t base = layer_off + (size_t)kvh * c->n_ctx * HD;
        if (c->kv_f16) {
            const uint16_t *K = (const uint16_t *)c->k_cache + base;
            for (int j = 0; j <= pos; j++) att[j] = vec_dot_f16_f32(K + (size_t)j * HD, q, HD) * scale;
        } else {
            const float *K = (const float *)c->k_cache + base;
            for (int j = 0; j <= pos; j++) att[j] = vec_dot_f32(K + (size_t)j * HD, q, HD) * scale;
        }
        softmax(att, pos + 1);
        memset(out, 0, (size_t)HD * sizeof(float));
        if (c->kv_f16) {
            const uint16_t *V = (const uint16_t *)c->v_cache + base;
            for (int j = 0; j <= pos; j++) vec_axpy_f16(out, att[j], V + (size_t)j * HD, HD);
        } else {
            const float *V = (const float *)c->v_cache + base;
            for (int j = 0; j <= pos; j++) vec_axpy_f32(out, att[j], V + (size_t)j * HD, HD);
        }
    }
}

static void act_range(void *arg, int b, int e, int ith) {
    (void)ith;
    borg_ctx *c = (borg_ctx *)arg;
    float *g = c->hb, *u = c->hb2;
    if (c->model->act == ACT_SILU) {
        for (int i = b; i < e; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
    } else {
        for (int i = b; i < e; i++) {
            const float x = g[i];
            g[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))) * u[i];
        }
    }
}

typedef struct {
    const float *w;
    float *out;
    const float *in;
    int n;
    float eps;
} norm_task;

static void norm_range(void *arg, int b, int e, int ith) {
    (void)ith;
    norm_task *t = (norm_task *)arg;
    for (int i = b; i < e; i++) rmsnorm(t->out + (size_t)i * t->n, t->in + (size_t)i * t->n, t->w, t->n, t->eps);
}

static void run_norm(borg_ctx *c, float *out, const float *in, const float *w, int T, int n) {
    norm_task t = {w, out, in, n, c->model->cfg.norm_eps};
    pool_for(c->pool, T, 1, norm_range, &t);
}

/* ---- the forward pass ----------------------------------------------------- */
static void forward_chunk(borg_ctx *c, const int32_t *toks, int T, int pos0, float *logits, int logits_mode) {
    borg_model *m = c->model;
    const borg_config *cfg = &m->cfg;
    const int D = cfg->dim, HD = cfg->head_dim;
    const int NQ = cfg->n_heads * HD, NKV = cfg->n_kv_heads * HD, F = cfg->hidden_dim;

    for (int t = 0; t < T; t++) mat_get_row(&m->tok_embd, toks[t], c->x + (size_t)t * D);

    for (int l = 0; l < cfg->n_layers; l++) {
        const borg_layer *L = &m->layers[l];
        /* attention */
        run_norm(c, c->xb, c->x, L->attn_norm, T, D);
        record_stats(c, l, SITE_ATTN_IN, c->xb, T, D);
        {
            const borg_mat *ms[3] = {&L->wq, &L->wk, &L->wv};
            float *outs[3] = {c->q, c->k, c->v};
            int ldos[3] = {NQ, NKV, NKV};
            matmul(c, 3, ms, outs, ldos, T, c->xb, D);
        }
        tok_task tt = {c, l, T, pos0};
        pool_for(c->pool, T, 1, qkv_post_range, &tt);
        pool_for(c->pool, T * cfg->n_heads, 1, attention_range, &tt);
        record_stats(c, l, SITE_ATTN_OUT, c->xb, T, NQ);
        matmul1(c, &L->wo, c->xb2, D, T, c->xb, NQ);
        for (size_t i = 0; i < (size_t)T * D; i++) c->x[i] += c->xb2[i];

        /* feed-forward */
        run_norm(c, c->xb, c->x, L->ffn_norm, T, D);
        record_stats(c, l, SITE_FFN_IN, c->xb, T, D);
        {
            const borg_mat *ms[2] = {&L->w_gate, &L->w_up};
            float *outs[2] = {c->hb, c->hb2};
            int ldos[2] = {F, F};
            matmul(c, 2, ms, outs, ldos, T, c->xb, D);
        }
        {
            const int nth = borg_pool_size(c->pool);
            int chunk = T * F / (nth * 4);
            if (chunk < 512) chunk = 512;
            pool_for(c->pool, T * F, chunk, act_range, c);
        }
        record_stats(c, l, SITE_FFN_MID, c->hb, T, F);
        matmul1(c, &L->w_down, c->xb2, D, T, c->hb, F);
        for (size_t i = 0; i < (size_t)T * D; i++) c->x[i] += c->xb2[i];
    }

    /* final norm and output projection */
    if (logits_mode == 2) {
        run_norm(c, c->xb, c->x, m->output_norm, T, D);
        matmul1(c, &m->output, logits, cfg->vocab_size, T, c->xb, D);
        memcpy(c->embd, c->xb + (size_t)(T - 1) * D, (size_t)D * sizeof(float));
    } else {
        rmsnorm(c->embd, c->x + (size_t)(T - 1) * D, m->output_norm, D, cfg->norm_eps);
        if (logits_mode == 1) matmul1(c, &m->output, logits, cfg->vocab_size, 1, c->embd, D);
    }
}

int borg_decode(borg_ctx *c, const int32_t *tokens, int n, float *logits, int logits_mode) {
    if (n <= 0) return 0;
    if (c->n_past + n > c->n_ctx) return -1;
    if (logits_mode && !logits) return -2;
    const int V = c->model->cfg.vocab_size;
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= V) return -3;
    for (int done = 0; done < n;) {
        const int T = n - done < c->n_batch ? n - done : c->n_batch;
        const int last = done + T == n;
        int mode = 0;
        float *out = NULL;
        if (logits_mode == 2) {
            mode = 2;
            out = logits + (size_t)done * V;
        } else if (logits_mode == 1 && last) {
            mode = 1;
            out = logits;
        }
        forward_chunk(c, tokens + done, T, c->n_past, out, mode);
        if (c->stats) c->stats_tokens += T;
        memcpy(c->tokens + c->n_past, tokens + done, (size_t)T * sizeof(int32_t));
        c->n_past += T;
        done += T;
    }
    return 0;
}
