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
/* calloc of a * b elements of esz bytes, or NULL when that does not fit size_t: on 32-bit targets
   (wasm32) the product of n_ctx or n_batch and model sizes would otherwise wrap to a small buffer */
static void *calloc2(uint64_t a, uint64_t b, size_t esz) {
    if (b && a > SIZE_MAX / esz / b) return NULL;
    return calloc((size_t)(a * b), esz);
}

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
    free(c->row_ctx);
    free(c->row_pos);
    free(c->stats);
    free(c->tr_resid);
    free(c->tr_attn);
    free(c->tr_mlp);
    free(c->tr_norms);
    free(c->tr_vnorm);
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
    const uint64_t kv_rows = (uint64_t)cfg->n_layers * NKV; /* times n_ctx elements */
    const size_t esz = c->kv_f16 ? 2 : 4;

    c->tokens = (int32_t *)calloc((size_t)c->n_ctx, sizeof(int32_t));
    c->k_cache = calloc2(kv_rows, (uint64_t)c->n_ctx, esz);
    c->v_cache = calloc2(kv_rows, (uint64_t)c->n_ctx, esz);
    c->inv_freq = (float *)calloc((size_t)HD / 2, sizeof(float));
    c->x = (float *)calloc2(B, D, sizeof(float));
    c->xb = (float *)calloc2(B, NQ > D ? NQ : D, sizeof(float));
    c->xb2 = (float *)calloc2(B, D, sizeof(float));
    c->q = (float *)calloc2(B, NQ, sizeof(float));
    c->k = (float *)calloc2(B, NKV, sizeof(float));
    c->v = (float *)calloc2(B, NKV, sizeof(float));
    c->hb = (float *)calloc2(B, F, sizeof(float));
    c->hb2 = (float *)calloc2(B, F, sizeof(float));
    c->xq = (block_q8a *)calloc2(B, c->max_cols / QK, sizeof(block_q8a));
    c->att = (float *)calloc2(nthreads, c->n_ctx, sizeof(float));
    c->embd = (float *)calloc((size_t)D, sizeof(float));
    c->row_ctx = (borg_ctx **)calloc((size_t)B, sizeof(borg_ctx *));
    c->row_pos = (int *)calloc((size_t)B, sizeof(int));
    if (!c->row_ctx || !c->row_pos || !c->tokens || !c->k_cache || !c->v_cache || !c->inv_freq || !c->x || !c->xb || !c->xb2 || !c->q ||
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
    c->stats = (double *)calloc2((uint64_t)c->model->cfg.n_layers * SITE_COUNT, c->max_cols, sizeof(double));
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

/* ---- activation trace ----------------------------------------------------- */
int borg_ctx_enable_trace(borg_ctx *c, int enable) {
    if (!enable) {
        c->trace_on = 0;
        return 0;
    }
    const borg_config *cfg = &c->model->cfg;
    if (!c->tr_resid) {
        c->tr_resid = (float *)calloc2(cfg->n_layers + 1, cfg->dim, sizeof(float));
        c->tr_attn = (float *)calloc2((uint64_t)cfg->n_layers * cfg->n_heads, c->n_ctx, sizeof(float));
        c->tr_mlp = (float *)calloc2(cfg->n_layers, cfg->hidden_dim, sizeof(float));
        c->tr_norms = (float *)calloc((size_t)cfg->n_layers * 3, sizeof(float));
        c->tr_vnorm = (float *)calloc2((uint64_t)cfg->n_layers * cfg->n_kv_heads, c->n_ctx, sizeof(float));
        if (!c->tr_resid || !c->tr_attn || !c->tr_mlp || !c->tr_norms || !c->tr_vnorm) {
            free(c->tr_resid);
            free(c->tr_attn);
            free(c->tr_mlp);
            free(c->tr_norms);
            free(c->tr_vnorm);
            c->tr_resid = c->tr_attn = c->tr_mlp = c->tr_norms = c->tr_vnorm = NULL;
            return -1;
        }
        c->trace_pos = -1;
    }
    c->trace_on = 1;
    return 0;
}

/* |v| of the cached positions [0, n) of every layer and kv head, from the cache itself: for
 * positions that arrive by borg_ctx_copy or borg_ctx_load rather than through qkv_post. */
static void vnorm_from_cache(borg_ctx *c, int n) {
    const borg_config *cfg = &c->model->cfg;
    const int HD = cfg->head_dim, NKV = cfg->n_kv_heads * HD;
    for (int l = 0; l < cfg->n_layers; l++)
        for (int h = 0; h < cfg->n_kv_heads; h++) {
            const size_t base = (size_t)l * NKV * c->n_ctx + (size_t)h * c->n_ctx * HD;
            float *out = c->tr_vnorm + ((size_t)l * cfg->n_kv_heads + h) * c->n_ctx;
            for (int j = 0; j < n; j++) {
                float ss = 0.0f;
                if (c->kv_f16) {
                    const uint16_t *v = (const uint16_t *)c->v_cache + base + (size_t)j * HD;
                    for (int i = 0; i < HD; i++) {
                        const float x = fp16_to_fp32(v[i]);
                        ss += x * x;
                    }
                } else {
                    const float *v = (const float *)c->v_cache + base + (size_t)j * HD;
                    ss = vec_dot_f32(v, v, HD);
                }
                out[j] = sqrtf(ss);
            }
        }
}

int borg_ctx_trace_pos(const borg_ctx *c) { return c->tr_resid ? c->trace_pos : -1; }

const float *borg_ctx_trace_resid(const borg_ctx *c, int layer) {
    if (!c->tr_resid || c->trace_pos < 0 || layer < 0 || layer > c->model->cfg.n_layers) return NULL;
    return c->tr_resid + (size_t)layer * c->model->cfg.dim;
}

const float *borg_ctx_trace_attn(const borg_ctx *c, int layer, int head, int *n_pos) {
    const borg_config *cfg = &c->model->cfg;
    if (!c->tr_attn || c->trace_pos < 0 || layer < 0 || layer >= cfg->n_layers || head < 0 || head >= cfg->n_heads)
        return NULL;
    if (n_pos) *n_pos = c->trace_pos + 1;
    return c->tr_attn + ((size_t)layer * cfg->n_heads + head) * c->n_ctx;
}

const float *borg_ctx_trace_mlp(const borg_ctx *c, int layer) {
    if (!c->tr_mlp || c->trace_pos < 0 || layer < 0 || layer >= c->model->cfg.n_layers) return NULL;
    return c->tr_mlp + (size_t)layer * c->model->cfg.hidden_dim;
}

const float *borg_ctx_trace_norms(const borg_ctx *c) {
    return c->tr_norms && c->trace_pos >= 0 ? c->tr_norms : NULL;
}

const float *borg_ctx_trace_vnorm(const borg_ctx *c, int layer, int kv_head) {
    const borg_config *cfg = &c->model->cfg;
    if (!c->tr_vnorm || layer < 0 || layer >= cfg->n_layers || kv_head < 0 || kv_head >= cfg->n_kv_heads) return NULL;
    return c->tr_vnorm + ((size_t)layer * cfg->n_kv_heads + kv_head) * c->n_ctx;
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
    borg_ctx *c;  /* owner of the scratch buffers */
    int layer, T;
} tok_task;

/* bias, q/k norm, RoPE, and store k/v of row t into the cache of kc at pos */
static void qkv_post(borg_ctx *c, borg_ctx *kc, int l, int t, int pos) {
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
    const size_t layer_off = (size_t)l * NKV * kc->n_ctx;
    for (int h = 0; h < cfg->n_kv_heads; h++) {
        const size_t off = layer_off + ((size_t)h * kc->n_ctx + (size_t)pos) * HD;
        /* |v| of this position for the effective-attention view (rows are distinct per item: no race);
         * 0 while tracing is off, so a slot re-evaluated after a truncate or reset does not keep a
         * stale norm from an earlier traced evaluation */
        if (kc->tr_vnorm)
            kc->tr_vnorm[((size_t)l * cfg->n_kv_heads + h) * kc->n_ctx + pos] =
                kc->trace_on ? sqrtf(vec_dot_f32(v + h * HD, v + h * HD, HD)) : 0.0f;
        if (kc->kv_f16) {
            uint16_t *kd = (uint16_t *)kc->k_cache + off, *vd = (uint16_t *)kc->v_cache + off;
            for (int i = 0; i < HD; i++) {
                kd[i] = fp32_to_fp16(k[h * HD + i]);
                vd[i] = fp32_to_fp16(v[h * HD + i]);
            }
        } else {
            memcpy((float *)kc->k_cache + off, k + h * HD, (size_t)HD * sizeof(float));
            memcpy((float *)kc->v_cache + off, v + h * HD, (size_t)HD * sizeof(float));
        }
    }
}

static void qkv_post_range(void *arg, int b, int e, int ith) {
    (void)ith;
    tok_task *t = (tok_task *)arg;
    for (int i = b; i < e; i++) qkv_post(t->c, t->c->row_ctx[i], t->layer, i, t->c->row_pos[i]);
}

static void attention_range(void *arg, int b, int e, int ith) {
    tok_task *t = (tok_task *)arg;
    borg_ctx *c = t->c;
    const borg_config *cfg = &c->model->cfg;
    const int HD = cfg->head_dim, NQ = cfg->n_heads * HD, NKV = cfg->n_kv_heads * HD;
    const int group = cfg->n_heads / cfg->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)HD);
    float *att = c->att + (size_t)ith * c->n_ctx;
    for (int item = b; item < e; item++) {
        const int tok = item / cfg->n_heads, h = item % cfg->n_heads;
        const borg_ctx *kc = c->row_ctx[tok];
        const int pos = c->row_pos[tok];
        const int kvh = h / group;
        const float *q = c->q + (size_t)tok * NQ + (size_t)h * HD;
        float *out = c->xb + (size_t)tok * NQ + (size_t)h * HD;
        const size_t base = (size_t)t->layer * NKV * kc->n_ctx + (size_t)kvh * kc->n_ctx * HD;
        if (kc->kv_f16) {
            const uint16_t *K = (const uint16_t *)kc->k_cache + base;
            for (int j = 0; j <= pos; j++) att[j] = vec_dot_f16_f32(K + (size_t)j * HD, q, HD) * scale;
        } else {
            const float *K = (const float *)kc->k_cache + base;
            for (int j = 0; j <= pos; j++) att[j] = vec_dot_f32(K + (size_t)j * HD, q, HD) * scale;
        }
        softmax(att, pos + 1);
        if (c->trace_on && kc == c && tok == t->T - 1) /* rows are distinct per item: no race */
            memcpy(c->tr_attn + ((size_t)t->layer * cfg->n_heads + h) * c->n_ctx, att, (size_t)(pos + 1) * sizeof(float));
        memset(out, 0, (size_t)HD * sizeof(float));
        if (kc->kv_f16) {
            const uint16_t *V = (const uint16_t *)kc->v_cache + base;
            for (int j = 0; j <= pos; j++) vec_axpy_f16(out, att[j], V + (size_t)j * HD, HD);
        } else {
            const float *V = (const float *)kc->v_cache + base;
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
/* Rows of the chunk may belong to different caches: c->row_ctx[t] / c->row_pos[t] say where
 * row t's keys and values go. c provides the scratch buffers. */
static void forward_chunk(borg_ctx *c, const int32_t *toks, int T, float *logits, int logits_mode) {
    borg_model *m = c->model;
    const borg_config *cfg = &m->cfg;
    const int D = cfg->dim, HD = cfg->head_dim;
    const int NQ = cfg->n_heads * HD, NKV = cfg->n_kv_heads * HD, F = cfg->hidden_dim;

    for (int t = 0; t < T; t++) mat_get_row(&m->tok_embd, toks[t], c->x + (size_t)t * D);
    /* trace the last row when it belongs to this context (not other sequences of a multi batch) */
    const int tr = c->trace_on && c->row_ctx[T - 1] == c;
    const size_t last = (size_t)(T - 1) * D;
    if (tr) {
        c->trace_pos = c->row_pos[T - 1];
        memcpy(c->tr_resid, c->x + last, (size_t)D * sizeof(float));
    }

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
        tok_task tt = {c, l, T};
        pool_for(c->pool, T, 1, qkv_post_range, &tt);
        pool_for(c->pool, T * cfg->n_heads, 1, attention_range, &tt);
        record_stats(c, l, SITE_ATTN_OUT, c->xb, T, NQ);
        matmul1(c, &L->wo, c->xb2, D, T, c->xb, NQ);
        for (size_t i = 0; i < (size_t)T * D; i++) c->x[i] += c->xb2[i];
        if (tr) c->tr_norms[l * 3] = sqrtf(vec_dot_f32(c->xb2 + last, c->xb2 + last, D));

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
        if (tr) memcpy(c->tr_mlp + (size_t)l * F, c->hb + (size_t)(T - 1) * F, (size_t)F * sizeof(float));
        matmul1(c, &L->w_down, c->xb2, D, T, c->hb, F);
        for (size_t i = 0; i < (size_t)T * D; i++) c->x[i] += c->xb2[i];
        if (tr) {
            c->tr_norms[l * 3 + 1] = sqrtf(vec_dot_f32(c->xb2 + last, c->xb2 + last, D));
            c->tr_norms[l * 3 + 2] = sqrtf(vec_dot_f32(c->x + last, c->x + last, D));
            memcpy(c->tr_resid + (size_t)(l + 1) * D, c->x + last, (size_t)D * sizeof(float));
        }
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
        for (int t = 0; t < T; t++) {
            c->row_ctx[t] = c;
            c->row_pos[t] = c->n_past + t;
        }
        forward_chunk(c, tokens + done, T, out, mode);
        if (c->stats) c->stats_tokens += T;
        memcpy(c->tokens + c->n_past, tokens + done, (size_t)T * sizeof(int32_t));
        c->n_past += T;
        done += T;
    }
    return 0;
}

int decode_multi_with(borg_ctx *lead, borg_ctx **ctxs, int n, const int32_t *tokens, float *logits) {
    if (n <= 0) return 0;
    if (n > lead->n_batch) return -4;
    const int V = lead->model->cfg.vocab_size;
    for (int i = 0; i < n; i++) {
        const borg_ctx *c = ctxs[i];
        if (c->model != lead->model || c->n_ctx != lead->n_ctx) return -5;
        if (c->n_past + 1 > c->n_ctx) return -1;
        if (tokens[i] < 0 || tokens[i] >= V) return -3;
        for (int j = 0; j < i; j++)
            if (ctxs[j] == c) return -6;
    }
    for (int i = 0; i < n; i++) {
        lead->row_ctx[i] = ctxs[i];
        lead->row_pos[i] = ctxs[i]->n_past;
    }
    forward_chunk(lead, tokens, n, logits, logits ? 2 : 0);
    const borg_model *m = lead->model;
    for (int i = 0; i < n; i++) {
        borg_ctx *c = ctxs[i];
        c->tokens[c->n_past++] = tokens[i];
        /* each sequence's final hidden state, as borg_ctx_embedding promises */
        rmsnorm(c->embd, lead->x + (size_t)i * m->cfg.dim, m->output_norm, m->cfg.dim, m->cfg.norm_eps);
    }
    if (lead->stats) lead->stats_tokens += n; /* statistics of a batch accumulate in its scratch owner */
    return 0;
}

int borg_decode_multi(borg_ctx **ctxs, int n, const int32_t *tokens, float *logits) {
    if (n <= 0) return 0;
    return decode_multi_with(ctxs[0], ctxs, n, tokens, logits);
}

int borg_ctx_copy(borg_ctx *dst, const borg_ctx *src) {
    if (dst == src) return 0;
    if (dst->model != src->model || dst->kv_f16 != src->kv_f16 || src->n_past > dst->n_ctx) return -1;
    const borg_config *cfg = &src->model->cfg;
    const size_t esz = src->kv_f16 ? 2 : 4, HD = (size_t)cfg->head_dim, n = (size_t)src->n_past;
    for (int l = 0; l < cfg->n_layers; l++)
        for (int h = 0; h < cfg->n_kv_heads; h++) {
            const size_t so = (((size_t)l * cfg->n_kv_heads + h) * src->n_ctx) * HD * esz;
            const size_t d_o = (((size_t)l * cfg->n_kv_heads + h) * dst->n_ctx) * HD * esz;
            memcpy((char *)dst->k_cache + d_o, (const char *)src->k_cache + so, n * HD * esz);
            memcpy((char *)dst->v_cache + d_o, (const char *)src->v_cache + so, n * HD * esz);
        }
    memcpy(dst->tokens, src->tokens, n * sizeof(int32_t));
    dst->n_past = src->n_past;
    if (dst->tr_vnorm) vnorm_from_cache(dst, (int)n);
    return 0;
}

/* KV cache file: "BKV1", u32 kv_f16, n_layers, n_kv_heads, head_dim, vocab_size, n_past, then the
 * 64-bit FNV-1a hash of the model header (so a cache never loads into a different model of the same
 * shape), i32 tokens[n_past], then per layer and kv head: K[n_past][head_dim], V[n_past][head_dim]. */
static uint64_t header_hash(const borg_model *m) {
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char *p = (const unsigned char *)m->header_json; *p; p++) h = (h ^ *p) * 1099511628211ull;
    return h;
}

int borg_ctx_save(const borg_ctx *c, const char *path) {
    const borg_config *cfg = &c->model->cfg;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const uint64_t hh = header_hash(c->model);
    const uint32_t hdr[8] = {(uint32_t)c->kv_f16, (uint32_t)cfg->n_layers, (uint32_t)cfg->n_kv_heads,
                             (uint32_t)cfg->head_dim, (uint32_t)cfg->vocab_size, (uint32_t)c->n_past,
                             (uint32_t)hh, (uint32_t)(hh >> 32)};
    const size_t esz = c->kv_f16 ? 2 : 4, row = (size_t)cfg->head_dim * esz * (size_t)c->n_past;
    int ok = fwrite("BKV1", 1, 4, f) == 4 && fwrite(hdr, sizeof hdr, 1, f) == 1 &&
             fwrite(c->tokens, sizeof(int32_t), (size_t)c->n_past, f) == (size_t)c->n_past;
    for (int l = 0; ok && l < cfg->n_layers; l++)
        for (int h = 0; ok && h < cfg->n_kv_heads; h++) {
            const size_t off = (((size_t)l * cfg->n_kv_heads + h) * c->n_ctx) * cfg->head_dim * esz;
            ok = (!row || fwrite((const char *)c->k_cache + off, 1, row, f) == row) &&
                 (!row || fwrite((const char *)c->v_cache + off, 1, row, f) == row);
        }
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

int borg_ctx_load(borg_ctx *c, const char *path) {
    const borg_config *cfg = &c->model->cfg;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    uint32_t hdr[8];
    const uint64_t hh = header_hash(c->model);
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BKV1", 4) || fread(hdr, sizeof hdr, 1, f) != 1 ||
        hdr[0] != (uint32_t)c->kv_f16 || hdr[1] != (uint32_t)cfg->n_layers || hdr[2] != (uint32_t)cfg->n_kv_heads ||
        hdr[3] != (uint32_t)cfg->head_dim || hdr[4] != (uint32_t)cfg->vocab_size || hdr[5] > (uint32_t)c->n_ctx ||
        hdr[6] != (uint32_t)hh || hdr[7] != (uint32_t)(hh >> 32)) {
        fclose(f);
        return -2;
    }
    const int n = (int)hdr[5];
    const size_t esz = c->kv_f16 ? 2 : 4, row = (size_t)cfg->head_dim * esz * (size_t)n;
    c->n_past = 0;
    int ok = fread(c->tokens, sizeof(int32_t), (size_t)n, f) == (size_t)n;
    for (int i = 0; ok && i < n; i++) ok = c->tokens[i] >= 0 && c->tokens[i] < cfg->vocab_size;
    for (int l = 0; ok && l < cfg->n_layers; l++)
        for (int h = 0; ok && h < cfg->n_kv_heads; h++) {
            const size_t off = (((size_t)l * cfg->n_kv_heads + h) * c->n_ctx) * cfg->head_dim * esz;
            ok = (!row || fread((char *)c->k_cache + off, 1, row, f) == row) &&
                 (!row || fread((char *)c->v_cache + off, 1, row, f) == row);
        }
    fclose(f);
    if (!ok) return -3;
    c->n_past = n;
    if (c->tr_vnorm) vnorm_from_cache(c, n);
    return 0;
}

int borg_ctx_lens(borg_ctx *c, const int *layers, int n, float *logits) {
    const borg_model *m = c->model;
    const borg_config *cfg = &m->cfg;
    if (!c->tr_resid || c->trace_pos < 0) return -1;
    if (n <= 0) return 0;
    if (!logits) return -2;
    for (int i = 0; i < n; i++)
        if (layers[i] < 0 || layers[i] > cfg->n_layers) return -3;
    const int D = cfg->dim, V = cfg->vocab_size;
    /* all requested layers go through the output matrix together: its weights are read once */
    for (int done = 0; done < n;) {
        const int T = n - done < c->n_batch ? n - done : c->n_batch;
        for (int t = 0; t < T; t++)
            rmsnorm(c->xb + (size_t)t * D, c->tr_resid + (size_t)layers[done + t] * D, m->output_norm, D, cfg->norm_eps);
        matmul1(c, &m->output, logits + (size_t)done * V, V, T, c->xb, D);
        done += T;
    }
    return 0;
}

/* ---- fused logit lens ----------------------------------------------------- */
/* borg_ctx_lens_summary multiplies the output matrix in blocks of LENS_BLOCK rows. A block leaves,
 * per lens row, its own partial summary (the top list, the max, the sums of exp(z - max) and of
 * (z - max) exp(z - max), the counts above each tracked logit); the partials are merged in block
 * order afterwards, so the result is the same whichever threads ran which blocks. */
#define LENS_BLOCK 512

typedef struct {
    double max, sum, wsum; /* the block's largest logit, sum of exp(z - max), sum of (z - max) exp(z - max) */
    int32_t n_top;         /* entries filled in the block's top list */
} lens_part;

typedef struct {
    borg_ctx *c;
    int T, n_vocab, k, n_track;
    const float *track_logit; /* [T][n_track] logits of the tracked tokens */
    float *scratch;           /* [n_threads][T][LENS_BLOCK] logits of the block in flight */
    lens_part *parts;         /* [n_blocks][T] */
    int32_t *above;           /* [n_blocks][T][n_track] logits strictly greater than each tracked one */
    int32_t *ids;             /* [n_blocks][T][k] top ids of each block, best first */
    float *vals;              /* [n_blocks][T][k] their logits */
} lens_task;

/* exp(z - mx) for floats z <= mx without a libm call, so that a loop over a block vectorizes:
 * x = n ln2 + r with |r| <= ln2/2 (ln2 split Cody-Waite style, so n ln2 is subtracted exactly),
 * e^r from its Taylor series to degree 12 (truncation error 2e-16) and n placed in the exponent
 * bits. Within a few ulp of exp(). x is held at -700 and above (exp(-700) is 1e-304, nothing
 * next to the max term's 1) by a comparison of the bit patterns: gcc does not if-convert a
 * floating-point compare, and for non-positive doubles (x is -0.0 when z equals mx) the larger
 * signed integer pattern is the more negative number. */
static inline double lens_exp(float z, float mx) {
    const double lo = -700.0;
    double x = -((double)mx - z);
    int64_t xb, lob;
    memcpy(&xb, &x, sizeof xb);
    memcpy(&lob, &lo, sizeof lob);
    xb = xb > lob ? lob : xb;
    memcpy(&x, &xb, sizeof x);
    const int32_t n = (int32_t)(x * 1.4426950408889634 - 0.5); /* nearest integer: x <= 0, so truncation rounds */
    const double r = (x - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;
    double p = 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    const int64_t bits = (int64_t)(n + 1023) << 52;
    double s;
    memcpy(&s, &bits, sizeof s);
    return p * s;
}

/* Put (id, v) into a list of n entries sorted by logit descending then id ascending. Ids arrive in
 * increasing order, so an equal logit stays behind the entries already there. When the list is
 * full (n == k) the caller has checked v > vals[k - 1], and the last entry drops out. */
static void top_insert(int32_t *ids, float *vals, int *n, int k, int32_t id, float v) {
    int j = *n < k ? (*n)++ : k - 1;
    while (j > 0 && vals[j - 1] < v) {
        ids[j] = ids[j - 1];
        vals[j] = vals[j - 1];
        j--;
    }
    ids[j] = id;
    vals[j] = v;
}

static void lens_range(void *arg, int b0, int b1, int ith) {
    lens_task *t = (lens_task *)arg;
    borg_ctx *c = t->c;
    const borg_mat *w = &c->model->output;
    const int T = t->T, k = t->k, nt = t->n_track, D = c->model->cfg.dim;
    float *z = t->scratch + (size_t)ith * T * LENS_BLOCK;
    double e[LENS_BLOCK];
    for (int b = b0; b < b1; b++) {
        const int r0 = b * LENS_BLOCK, r1 = r0 + LENS_BLOCK < t->n_vocab ? r0 + LENS_BLOCK : t->n_vocab;
        const int nr = r1 - r0;
        /* the kernel of borg_ctx_lens on the same normalized and quantized rows, so each logit is
         * the same number; row r0 of the matrix lands in column 0 of z */
        mat_mul_rows_off(w, r0, r1, T, c->xb, D, c->xq, c->max_cols / QK, z, LENS_BLOCK, r0);
        for (int tt = 0; tt < T; tt++) {
            const float *row = z + (size_t)tt * LENS_BLOCK;
            lens_part *p = &t->parts[(size_t)b * T + tt];
            int32_t *ids = t->ids + ((size_t)b * T + tt) * k;
            float *vals = t->vals + ((size_t)b * T + tt) * k;
            int32_t *above = t->above + ((size_t)b * T + tt) * nt;
            int n = 0;
            for (int j = 0; j < nr; j++)
                if (n < k || row[j] > vals[k - 1]) top_insert(ids, vals, &n, k, r0 + j, row[j]);
            const float mx = vals[0]; /* the block's largest logit */
            for (int i = 0; i < nt; i++) {
                const float zt = t->track_logit[(size_t)tt * nt + i];
                int32_t cnt = 0;
                for (int j = 0; j < nr; j++) cnt += row[j] > zt;
                above[i] = cnt;
            }
            for (int j = 0; j < nr; j++) e[j] = lens_exp(row[j], mx);
            double s = 0.0, ws = 0.0;
            for (int j = 0; j < nr; j++) s += e[j];
            for (int j = 0; j < nr; j++) ws += ((double)row[j] - (double)mx) * e[j];
            p->max = mx;
            p->sum = s;
            p->wsum = ws;
            p->n_top = n;
        }
    }
}

int borg_ctx_lens_summary(borg_ctx *c, const int *layers, int n, int n_vocab, int k, const int32_t *track, int n_track,
                          int32_t *top_ids, float *top_logits, float *track_logit, int32_t *track_rank, double *lse,
                          double *entropy) {
    const borg_model *m = c->model;
    const borg_config *cfg = &m->cfg;
    if (!c->tr_resid || c->trace_pos < 0) return -1;
    if (n <= 0) return 0;
    if (!top_ids || !top_logits || !lse) return -2;
    if (n_track > 0 && (!track || !track_logit || !track_rank)) return -2;
    for (int i = 0; i < n; i++)
        if (layers[i] < 0 || layers[i] > cfg->n_layers) return -3;
    if (n_vocab == 0) n_vocab = cfg->vocab_size;
    if (n_vocab < 1 || n_vocab > cfg->vocab_size) return -6;
    if (k < 1 || k > n_vocab) return -4;
    if (k > BORG_LENS_MAX_K) return -8;
    if (n_track < 0) return -5;
    for (int i = 0; i < n_track; i++)
        if (track[i] < 0 || track[i] >= n_vocab) return -5;
    const int D = cfg->dim, nth = borg_pool_size(c->pool), ldxq = c->max_cols / QK;
    const int n_blocks = (n_vocab + LENS_BLOCK - 1) / LENS_BLOCK;
    const int TB = n < c->n_batch ? n : c->n_batch; /* lens rows per pass, as borg_ctx_lens */
    const int nt1 = n_track > 0 ? n_track : 1;         /* allocations stay non-empty with nothing tracked */
    lens_task t;
    memset(&t, 0, sizeof t);
    t.c = c;
    t.n_vocab = n_vocab;
    t.k = k;
    t.n_track = n_track;
    t.scratch = (float *)calloc2((uint64_t)nth * TB, LENS_BLOCK, sizeof(float));
    t.parts = (lens_part *)calloc2((uint64_t)n_blocks, TB, sizeof(lens_part));
    t.above = (int32_t *)calloc2((uint64_t)n_blocks * TB, nt1, sizeof(int32_t));
    t.ids = (int32_t *)calloc2((uint64_t)n_blocks * TB, k, sizeof(int32_t));
    t.vals = (float *)calloc2((uint64_t)n_blocks * TB, k, sizeof(float));
    float *zt = (float *)calloc2((uint64_t)TB, nt1, sizeof(float));
    int rc = 0;
    if (!t.scratch || !t.parts || !t.above || !t.ids || !t.vals || !zt) {
        rc = -7;
        goto done;
    }
    for (int done = 0; done < n; done += TB) {
        const int T = n - done < TB ? n - done : TB;
        for (int i = 0; i < T; i++)
            rmsnorm(c->xb + (size_t)i * D, c->tr_resid + (size_t)layers[done + i] * D, m->output_norm, D, cfg->norm_eps);
        if (mat_wants_q8a(&m->output)) {
            quant_task qt = {c->xb, D, D, c->xq, ldxq};
            pool_for(c->pool, T, 1, quant_range, &qt);
        }
        /* the tracked tokens' logits first: the blocks count the logits above each */
        for (int i = 0; i < T; i++)
            for (int j = 0; j < n_track; j++)
                zt[(size_t)i * n_track + j] =
                    mat_dot_row(&m->output, track[j], c->xb + (size_t)i * D, c->xq + (size_t)i * ldxq);
        t.T = T;
        t.track_logit = zt;
        pool_for(c->pool, n_blocks, 1, lens_range, &t);
        /* merge the blocks' partials, in block order */
        for (int i = 0; i < T; i++) {
            int32_t *ids = top_ids + (size_t)(done + i) * k;
            float *vals = top_logits + (size_t)(done + i) * k;
            int n_top = 0;
            double mx = t.parts[i].max, s = 0.0, ws = 0.0;
            for (int b = 0; b < n_blocks; b++) {
                const lens_part *p = &t.parts[(size_t)b * T + i];
                const int32_t *bi = t.ids + ((size_t)b * T + i) * k;
                const float *bv = t.vals + ((size_t)b * T + i) * k;
                for (int j = 0; j < p->n_top; j++) { /* sorted: once one is out, the rest are too */
                    if (n_top == k && !(bv[j] > vals[k - 1])) break;
                    top_insert(ids, vals, &n_top, k, bi[j], bv[j]);
                }
                if (p->max > mx) mx = p->max;
            }
            /* sum (z - mx) e^(z - mx) over a block is e^(max_b - mx) (wsum_b + (max_b - mx) sum_b) */
            for (int b = 0; b < n_blocks; b++) {
                const lens_part *p = &t.parts[(size_t)b * T + i];
                const double f = exp(p->max - mx);
                s += p->sum * f;
                ws += (p->wsum + (p->max - mx) * p->sum) * f;
            }
            lse[done + i] = mx + log(s);
            if (entropy) entropy[done + i] = log(s) - ws / s; /* H = lse - E[z] */
            for (int j = 0; j < n_track; j++) {
                int32_t above = 0;
                for (int b = 0; b < n_blocks; b++) above += t.above[((size_t)b * T + i) * n_track + j];
                track_logit[(size_t)(done + i) * n_track + j] = zt[(size_t)i * n_track + j];
                track_rank[(size_t)(done + i) * n_track + j] = above;
            }
        }
    }
done:
    free(t.scratch);
    free(t.parts);
    free(t.above);
    free(t.ids);
    free(t.vals);
    free(zt);
    return rc;
}
