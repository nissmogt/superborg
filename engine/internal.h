/* Internal structures shared by the engine's translation units. Not part of the API. */
#ifndef BORG_INTERNAL_H
#define BORG_INTERNAL_H

#include "borg.h"

#include <stdint.h>
#include <stdio.h>

#define BORG_VERSION "0.1.0"
#define QK 32 /* quantization block size */

/* ---- data types ----------------------------------------------------------- */
typedef enum { DT_F32 = 0, DT_F16, DT_BF16, DT_Q8_0, DT_Q4_0, DT_Q4_1, DT_U8, DT_U32, DT_I32, DT_COUNT } borg_dtype;

typedef struct {
    uint16_t d;
    int8_t qs[QK];
} block_q8_0; /* 34 bytes */

typedef struct {
    uint16_t d;
    uint8_t qs[QK / 2];
} block_q4_0; /* 18 bytes */

typedef struct {
    uint16_t d;
    uint16_t m;
    uint8_t qs[QK / 2];
} block_q4_1; /* 20 bytes: x = d * q + m, q in 0..15 */

/* Activations quantized to 8 bits for integer dot products: scale as float plus the
 * block sum (d * sum(qs)) used by formats with an offset. */
typedef struct {
    float d;
    float s;
    int8_t qs[QK];
} block_q8a; /* 40 bytes */

typedef struct {
    const char *name;
    borg_dtype dtype;
    int ndim;
    int64_t shape[4];
    const void *data;
    size_t nbytes;
} borg_tensor;

/* A weight matrix [rows, cols] in any storage type. */
typedef struct {
    borg_dtype dtype;
    int rows, cols;
    const void *data;
    size_t row_bytes;
} borg_mat;

/* ---- tokenizer ------------------------------------------------------------ */
enum { PRETOK_NONE = 0, PRETOK_GPT2, PRETOK_LLAMA3, PRETOK_QWEN2, PRETOK_SMOLLM };
enum { TT_NORMAL = 0, TT_BYTE = 1, TT_SPECIAL = 2, TT_ADDED = 3, TT_UNUSED = 4 };

typedef struct borg_tok_impl borg_tok_impl;

typedef struct {
    /* tables (point into the mapped file) */
    int n_vocab, n_merges;
    const uint8_t *vocab;       /* concatenated token bytes */
    const uint32_t *vocab_off;  /* [n_vocab + 1] */
    const uint8_t *token_type;  /* [n_vocab] */
    const uint32_t *merges;     /* [n_merges * 3]: left, right, merged */
    /* settings from the header */
    int byte_level, pretok, nfc, spm_add_prefix_space, byte_fallback, ignore_merges;
    int add_bos, add_eos, bos_id, eos_id, pad_id, unk_id;
    int n_eos_ids;
    int32_t eos_ids[16];
    int n_added;
    int32_t *added; /* ids matched verbatim when parse_special (owned) */
    borg_tok_impl *impl; /* hash tables built by tokenizer_init (owned) */
} borg_tokenizer;

/* Build lookup tables. Returns 0 on success. */
int tokenizer_init(borg_tokenizer *t, char *err, int errlen);
void tokenizer_free(borg_tokenizer *t);

/* ---- model ---------------------------------------------------------------- */
typedef struct {
    borg_mat wq, wk, wv, wo, w_gate, w_up, w_down;
    const float *attn_norm, *ffn_norm;
    const float *bq, *bk, *bv;          /* NULL without attn_bias */
    const float *q_norm, *k_norm;       /* NULL without qk_norm */
} borg_layer;

enum { ACT_SILU = 0, ACT_GELU_TANH = 1 };
enum { ROPE_SCALE_NONE = 0, ROPE_SCALE_LINEAR, ROPE_SCALE_LLAMA3 };

struct borg_model {
    borg_config cfg;
    int act;
    int rope_scale_type;
    float rope_factor, rope_low_freq_factor, rope_high_freq_factor;
    int rope_orig_max_pos;
    borg_mat tok_embd, output;
    const float *output_norm;
    borg_layer *layers;
    borg_tokenizer tok;
    /* file mapping */
    void *map;
    size_t map_size;
    int map_is_malloc;
    char *header_json;
    int n_tensors;
    borg_tensor *tensors;
};

const borg_tensor *model_find_tensor(const borg_model *m, const char *name);

/* ---- context -------------------------------------------------------------- */
struct borg_ctx {
    borg_model *model;
    borg_pool *pool;
    int n_ctx, n_batch, kv_f16;
    int n_past;
    int32_t *tokens;     /* [n_ctx] tokens whose keys/values are cached */
    void *k_cache;       /* [n_layers][n_kv_heads][n_ctx][head_dim], f32 or f16 */
    void *v_cache;
    float *inv_freq;     /* [head_dim / 2] RoPE frequencies */
    /* scratch for up to n_batch tokens */
    float *x;            /* [B, dim] residual stream */
    float *xb;           /* [B, max(dim, n_heads*head_dim)] normalized input / attention output */
    float *xb2;          /* [B, dim] */
    float *q;            /* [B, n_heads*head_dim] */
    float *k, *v;        /* [B, n_kv_heads*head_dim] */
    float *hb, *hb2;     /* [B, hidden_dim] */
    block_q8a *xq;       /* [B, max(dim, hidden_dim, n_heads*head_dim) / QK] */
    float *att;          /* [n_threads, n_ctx] attention scores */
    float *embd;         /* [dim] final hidden state of the last token */
    borg_ctx **row_ctx;  /* [n_batch] cache owner of each row in the current chunk */
    int *row_pos;        /* [n_batch] position of each row */
    double *stats;       /* optional [n_layers][SITE_COUNT][max_cols] sums of x^2 per input column */
    int64_t stats_tokens;
    int max_cols;        /* columns of xq rows */
};

/* ---- JSON (json.c) -------------------------------------------------------- */
typedef enum { JS_NULL = 0, JS_BOOL, JS_NUM, JS_STR, JS_ARR, JS_OBJ } json_type;

typedef struct json_value json_value;
struct json_value {
    json_type type;
    double num;           /* JS_NUM, JS_BOOL (0/1) */
    char *str;            /* JS_STR (decoded, NUL-terminated); object keys are in keys[] */
    int len;              /* string length in bytes, or element count */
    json_value *items;    /* JS_ARR / JS_OBJ elements */
    char **keys;          /* JS_OBJ keys */
};

json_value *json_parse(const char *text, size_t len, char *err, int errlen);
void json_free(json_value *v);
const json_value *json_get(const json_value *obj, const char *key); /* NULL if missing */
double json_num(const json_value *v, double dflt);
int json_bool(const json_value *v, int dflt);
const char *json_str(const json_value *v, const char *dflt);

/* ---- threads (pool.c) ----------------------------------------------------- */
/* fn(ctx, begin, end, ith) processes items [begin, end); ith identifies the calling thread
 * (0 = caller, unique among threads running at the same time) for per-thread scratch. */
typedef void (*borg_range_fn)(void *ctx, int begin, int end, int ith);
/* Run fn over [0, n_items) in chunks of `chunk` items on the pool; returns when all are done. */
void pool_for(borg_pool *pool, int n_items, int chunk, borg_range_fn fn, void *ctx);

/* ---- numeric kernels (kernels.c) ----------------------------------------- */
float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
float bf16_to_fp32(uint16_t h);

size_t dtype_row_bytes(borg_dtype dt, int cols);
const char *dtype_name(borg_dtype dt);

/* Activation statistics for importance-weighted quantization (see borg_ctx_stats). */
enum { SITE_ATTN_IN = 0, SITE_ATTN_OUT, SITE_FFN_IN, SITE_FFN_MID, SITE_COUNT };

/* Quantize n floats (n % QK == 0) into n/QK activation blocks. */
void quantize_row_q8a(const float *x, block_q8a *y, int n);
/* Dequantize row r of a matrix into out[cols]. */
void mat_get_row(const borg_mat *m, int r, float *out);
/* Dot product of matrix row r with an activation vector given in both float and q8a form. */
float mat_dot_row(const borg_mat *m, int r, const float *xf, const block_q8a *xq);
/* out[t * ldo + r] = dot(W[r], x[t]) for r in [r0, r1) and t in [0, T). */
void mat_mul_rows(const borg_mat *m, int r0, int r1, int T, const float *x, int ldx, const block_q8a *xq,
                  int ldxq, float *out, int ldo);
int mat_wants_q8a(const borg_mat *m); /* does this matrix consume q8a activations? */

float vec_dot_f32(const float *a, const float *b, int n);
void vec_axpy_f32(float *y, float a, const float *x, int n); /* y += a * x */
float vec_dot_f16_f32(const uint16_t *a, const float *b, int n);
void vec_axpy_f16(float *y, float a, const uint16_t *x, int n);

/* ---- sampling helpers (sample.c) ----------------------------------------- */
int32_t sampler_draw(borg_sampler *s, const float *probs, int n); /* draw from unnormalized probs */
float sampler_uniform(borg_sampler *s);

/* ---- utilities ------------------------------------------------------------ */
double borg_time_ms(void);
void borg_set_err(char *err, int errlen, const char *fmt, ...);

#endif /* BORG_INTERNAL_H */
