/* borg: a dependency-free inference engine for decoder-only transformers.
 *
 * Public C API. Everything here is safe to call through FFI (Python ctypes).
 * A model is read-only after loading and can be shared by several contexts.
 * A context holds the KV cache and scratch memory for one sequence.
 * A pool holds worker threads; contexts that run one after another (for example
 * a draft and a target model during speculative decoding) can share one pool,
 * but a pool must never be driven from two OS threads at the same time.
 */
#ifndef BORG_H
#define BORG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define BORG_API __declspec(dllexport)
#else
#define BORG_API __attribute__((visibility("default")))
#endif

typedef struct borg_model borg_model;
typedef struct borg_ctx borg_ctx;
typedef struct borg_pool borg_pool;
typedef struct borg_sampler borg_sampler;

typedef struct {
    int dim, n_layers, n_heads, n_kv_heads, head_dim, hidden_dim, vocab_size, max_seq_len;
    float norm_eps, rope_theta;
    int attn_bias, qk_norm, tie_embeddings;
    int n_vocab;      /* tokenizer vocabulary size (<= vocab_size) */
    int bos_id, eos_id;
    int add_bos;
    size_t weight_bytes; /* bytes of weights mapped from the file */
} borg_config;

BORG_API const char *borg_version(void);
BORG_API const char *borg_simd(void); /* kernels compiled in: "avx512", "avx2", "neon", "scalar" */

/* ---- threads -------------------------------------------------------------- */
BORG_API borg_pool *borg_pool_new(int n_threads); /* n_threads <= 0: one per physical-ish core */
BORG_API void borg_pool_free(borg_pool *pool);
BORG_API int borg_pool_size(const borg_pool *pool);

/* ---- model ---------------------------------------------------------------- */
/* Returns NULL on failure and writes a message into err (if err != NULL). */
BORG_API borg_model *borg_model_load(const char *path, char *err, int errlen);
BORG_API void borg_model_free(borg_model *model);
BORG_API void borg_model_config(const borg_model *model, borg_config *out);
BORG_API const char *borg_model_header(const borg_model *model); /* raw JSON header, NUL-terminated */

/* ---- tokenizer ------------------------------------------------------------ */
/* Tokenize UTF-8 text. add_bos: -1 = use the model default, 0 = no, 1 = yes.
 * parse_special: match added tokens (e.g. "<|im_start|>") verbatim in the text.
 * Returns the token count. If it exceeds max_out, returns -count and writes nothing useful. */
BORG_API int borg_tokenize(const borg_model *model, const char *text, int text_len, int add_bos,
                           int parse_special, int32_t *out, int max_out);
/* Raw bytes of one token (SentencePiece "▁" becomes a space, byte tokens become the byte).
 * Returns the length; if it exceeds buflen the output is truncated. Special tokens return
 * their literal text. Ids outside the vocabulary return 0. */
BORG_API int borg_token_bytes(const borg_model *model, int32_t tok, char *buf, int buflen);
/* Decode a token sequence into buf (not NUL-terminated). skip_special drops special/control
 * tokens. Returns the full length needed (may exceed buflen). */
BORG_API int borg_detokenize(const borg_model *model, const int32_t *toks, int n, int skip_special,
                             char *buf, int buflen);
BORG_API int borg_token_type(const borg_model *model, int32_t tok); /* see docs/FORMAT.md */
BORG_API int borg_is_eog(const borg_model *model, int32_t tok);    /* end-of-generation token? */

/* ---- context -------------------------------------------------------------- */
typedef struct {
    int n_ctx;   /* maximum sequence length held in the KV cache (0 = min(model max, 4096)) */
    int n_batch; /* tokens processed per forward chunk during prefill (0 = 64) */
    int kv_f16;  /* store the KV cache as float16 (halves its memory) */
} borg_ctx_params;

BORG_API borg_ctx_params borg_ctx_default_params(void);
BORG_API borg_ctx *borg_ctx_new(borg_model *model, borg_pool *pool, const borg_ctx_params *params);
BORG_API void borg_ctx_free(borg_ctx *ctx);
BORG_API int borg_ctx_n_ctx(const borg_ctx *ctx);
BORG_API int borg_ctx_n_past(const borg_ctx *ctx); /* tokens currently held in the KV cache */
BORG_API const int32_t *borg_ctx_tokens(const borg_ctx *ctx);
/* Drop cached tokens at positions >= n_keep. */
BORG_API void borg_ctx_truncate(borg_ctx *ctx, int n_keep);

/* Run the model on tokens[0..n) placed at positions [n_past, n_past + n) where n_past is
 * the current cache length, and append them to the cache.
 * logits_mode 0: write nothing; 1: last token's logits (vocab_size floats);
 * 2: logits of every token (n * vocab_size floats). Returns 0 on success. */
BORG_API int borg_decode(borg_ctx *ctx, const int32_t *tokens, int n, float *logits, int logits_mode);
/* Hidden state after the final norm for the last decoded token (dim floats). */
BORG_API const float *borg_ctx_embedding(const borg_ctx *ctx);

/* Activation statistics for importance-weighted quantization. When enabled, every
 * borg_decode adds x^2 of each matrix input column. Sites: 0 attention input (q/k/v),
 * 1 attention output (o), 2 feed-forward input (gate/up), 3 feed-forward hidden (down).
 * borg_ctx_stats returns the sums (n_cols values via *n_cols) or NULL; divide by
 * borg_ctx_stats_tokens for the mean. */
BORG_API int borg_ctx_enable_stats(borg_ctx *ctx, int enable);
BORG_API const double *borg_ctx_stats(const borg_ctx *ctx, int layer, int site, int *n_cols);
BORG_API int64_t borg_ctx_stats_tokens(const borg_ctx *ctx);

/* ---- sampling ------------------------------------------------------------- */
typedef struct {
    float temperature;    /* <= 0: greedy */
    int top_k;            /* <= 0: off */
    float top_p;          /* >= 1: off */
    float min_p;          /* <= 0: off */
    float repeat_penalty; /* 1: off */
    int repeat_last_n;    /* window for the repetition penalty */
    uint64_t seed;
} borg_sampling;

BORG_API borg_sampling borg_sampling_default(void);
BORG_API borg_sampler *borg_sampler_new(const borg_sampling *params, int vocab_size);
BORG_API void borg_sampler_free(borg_sampler *s);
BORG_API void borg_sampler_reset(borg_sampler *s, uint64_t seed);
/* Sample one token. logits is modified in place (penalties, temperature, softmax).
 * history: recent tokens for the repetition penalty (may be NULL). */
BORG_API int32_t borg_sampler_sample(borg_sampler *s, float *logits, const int32_t *history, int n_history);
/* Turn logits into the sampling distribution in place (after penalties, temperature,
 * top-k/top-p/min-p). Greedy produces a one-hot vector. */
BORG_API void borg_sampler_probs(borg_sampler *s, float *logits, const int32_t *history, int n_history);

/* ---- generation ----------------------------------------------------------- */
/* Called for every emitted token with its text bytes. Return nonzero to stop. */
typedef int (*borg_token_cb)(int32_t tok, const char *piece, int piece_len, void *user);

typedef struct {
    int max_tokens;
    borg_sampling sampling;
    borg_ctx *draft;  /* optional draft-model context sharing the vocabulary */
    int n_draft;      /* tokens proposed per speculative step (draft model or lookup) */
    int lookup_ngram; /* >0: propose tokens by matching the last n-gram earlier in the
                         context (prompt-lookup decoding) when no draft model is set */
    int ignore_eos;
    const int32_t *stop_tokens; /* extra tokens that end generation */
    int n_stop_tokens;
} borg_gen_params;

typedef struct {
    int n_prompt;        /* prompt tokens evaluated (after reusing the cached prefix) */
    int n_reused;        /* prompt tokens served from the KV cache */
    int n_gen;           /* tokens generated */
    int n_drafted;       /* tokens proposed by the draft source */
    int n_accepted;      /* proposed tokens accepted by the target model */
    int n_target_evals;  /* target forward passes during generation */
    double t_prompt_ms, t_gen_ms;
    int stop_reason;     /* 0 max_tokens, 1 end-of-generation token, 2 callback, 3 context full */
} borg_gen_stats;

BORG_API borg_gen_params borg_gen_default_params(void);
/* Generate a continuation of prompt. The cached prefix shared with the context's current
 * tokens is reused, the rest of the prompt is evaluated, then up to max_tokens tokens are
 * produced. Generated tokens are written to out (if non-NULL, up to max_out).
 * Returns the number of generated tokens, or a negative value on error. */
BORG_API int borg_generate(borg_ctx *ctx, const int32_t *prompt, int n_prompt, const borg_gen_params *params,
                           borg_token_cb cb, void *user, int32_t *out, int max_out, borg_gen_stats *stats);

#ifdef __cplusplus
}
#endif
#endif /* BORG_H */
