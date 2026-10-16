/* borg command-line interface: one-shot generation, interactive chat, benchmarks. */
#include "internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *model, *draft, *prompt, *prompt_file, *system;
    int n_predict, threads, n_ctx, n_draft, lookup, chat, interactive, bench, kv_f16, raw, verbose;
    int bench_prompt, bench_gen, ignore_eos, json;
    borg_sampling s;
} options;

static void usage(void) {
    fprintf(stderr,
            "usage: borg -m MODEL.borg [options]\n"
            "  -p TEXT            prompt\n"
            "  -f FILE            read the prompt from a file\n"
            "  -n N               tokens to generate (default 256)\n"
            "  -t N               threads (default: all cores)\n"
            "  -c N               context length (default: min(model, 4096))\n"
            "  --chat             wrap the prompt in the model's chat template\n"
            "  --system TEXT      system message for --chat / -i\n"
            "  -i                 interactive chat\n"
            "  --temp T           temperature (default 0 = greedy)\n"
            "  --top-k K  --top-p P  --min-p P  --repeat-penalty R  --seed S\n"
            "  --draft MODEL      draft model for speculative decoding\n"
            "  --n-draft K        tokens drafted per step (default 6)\n"
            "  --lookup N         prompt-lookup speculative decoding with N-gram matching\n"
            "  --kv-f16           store the KV cache in float16\n"
            "  --json             constrain the output to valid JSON\n"
            "  --ignore-eos       keep generating past end-of-sequence tokens\n"
            "  --raw              do not parse special tokens in the prompt\n"
            "  --bench [P G]      benchmark prefill of P tokens and generation of G tokens\n"
            "  -v                 print timing statistics\n");
}

static int parse_args(int argc, char **argv, options *o) {
    memset(o, 0, sizeof *o);
    o->n_predict = 256;
    o->n_draft = 6;
    o->s = borg_sampling_default();
    o->bench_prompt = 256;
    o->bench_gen = 64;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (usage(), exit(1), (char *)NULL))
        if (!strcmp(a, "-m")) o->model = NEXT();
        else if (!strcmp(a, "-p")) o->prompt = NEXT();
        else if (!strcmp(a, "-f")) o->prompt_file = NEXT();
        else if (!strcmp(a, "-n")) o->n_predict = atoi(NEXT());
        else if (!strcmp(a, "-t")) o->threads = atoi(NEXT());
        else if (!strcmp(a, "-c")) o->n_ctx = atoi(NEXT());
        else if (!strcmp(a, "--chat")) o->chat = 1;
        else if (!strcmp(a, "--system")) o->system = NEXT();
        else if (!strcmp(a, "-i")) o->interactive = o->chat = 1;
        else if (!strcmp(a, "--temp")) o->s.temperature = (float)atof(NEXT());
        else if (!strcmp(a, "--top-k")) o->s.top_k = atoi(NEXT());
        else if (!strcmp(a, "--top-p")) o->s.top_p = (float)atof(NEXT());
        else if (!strcmp(a, "--min-p")) o->s.min_p = (float)atof(NEXT());
        else if (!strcmp(a, "--repeat-penalty")) o->s.repeat_penalty = (float)atof(NEXT());
        else if (!strcmp(a, "--seed")) o->s.seed = (uint64_t)strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--draft")) o->draft = NEXT();
        else if (!strcmp(a, "--n-draft")) o->n_draft = atoi(NEXT());
        else if (!strcmp(a, "--lookup")) o->lookup = atoi(NEXT());
        else if (!strcmp(a, "--kv-f16")) o->kv_f16 = 1;
        else if (!strcmp(a, "--ignore-eos")) o->ignore_eos = 1;
        else if (!strcmp(a, "--json")) o->json = 1;
        else if (!strcmp(a, "--raw")) o->raw = 1;
        else if (!strcmp(a, "-v")) o->verbose = 1;
        else if (!strcmp(a, "--bench")) {
            o->bench = 1;
            if (i + 2 < argc && argv[i + 1][0] != '-' && argv[i + 2][0] != '-') {
                o->bench_prompt = atoi(argv[++i]);
                o->bench_gen = atoi(argv[++i]);
            }
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); exit(0); }
        else { fprintf(stderr, "unknown option %s\n", a); usage(); return -1; }
#undef NEXT
    }
    if (!o->model) { usage(); return -1; }
    if (o->bench && (o->bench_prompt < 1 || o->bench_gen < 0)) {
        fprintf(stderr, "--bench needs at least one prompt token\n");
        return -1;
    }
    return 0;
}

/* Read a whole file into a NUL-terminated buffer. Reads until EOF instead of trusting the
 * file size, so pipes, /dev/stdin and process substitution work. */
static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 4096, len = 0;
    char *s = (char *)malloc(cap);
    while (s) {
        if (len + 1 >= cap) {
            char *t = (char *)realloc(s, cap * 2);
            if (!t) {
                free(s);
                s = NULL;
                break;
            }
            s = t;
            cap *= 2;
        }
        const size_t k = fread(s + len, 1, cap - 1 - len, f);
        len += k;
        if (k == 0) {
            if (ferror(f)) {
                free(s);
                s = NULL;
            }
            break;
        }
    }
    if (s) s[len] = 0;
    fclose(f);
    return s;
}

/* ---- chat templates (built-in fallbacks; the Python layer renders the model's own Jinja template) */
typedef struct {
    char *buf;
    size_t len, cap;
} sbuf;

static void sb_add(sbuf *b, const char *s) {
    size_t n = strlen(s);
    if (b->len + n + 1 > b->cap) {
        b->cap = (b->len + n + 1) * 2;
        b->buf = (char *)realloc(b->buf, b->cap);
    }
    memcpy(b->buf + b->len, s, n + 1);
    b->len += n;
}

static const char *chat_format(const borg_model *m) {
    static char fmt[32] = "plain";
    json_value *root = json_parse(borg_model_header(m), strlen(borg_model_header(m)), NULL, 0);
    const char *f = json_str(json_get(json_get(root, "tokenizer"), "chat_format"), "plain");
    snprintf(fmt, sizeof fmt, "%s", f);
    json_free(root);
    return fmt;
}

static void chat_turn(sbuf *b, const char *fmt, const char *role, const char *content) {
    if (!strcmp(fmt, "chatml") || !strcmp(fmt, "qwen3")) {
        sb_add(b, "<|im_start|>"); sb_add(b, role); sb_add(b, "\n"); sb_add(b, content); sb_add(b, "<|im_end|>\n");
    } else if (!strcmp(fmt, "llama3")) {
        sb_add(b, "<|start_header_id|>"); sb_add(b, role); sb_add(b, "<|end_header_id|>\n\n");
        sb_add(b, content); sb_add(b, "<|eot_id|>");
    } else if (!strcmp(fmt, "zephyr")) {
        sb_add(b, "<|"); sb_add(b, role); sb_add(b, "|>\n"); sb_add(b, content); sb_add(b, "</s>\n");
    } else if (!strcmp(fmt, "mistral")) {
        if (!strcmp(role, "user")) { sb_add(b, "[INST] "); sb_add(b, content); sb_add(b, " [/INST]"); }
        else if (!strcmp(role, "assistant")) { sb_add(b, content); sb_add(b, "</s>"); }
        else { sb_add(b, content); sb_add(b, "\n\n"); }
    } else {
        sb_add(b, role); sb_add(b, ": "); sb_add(b, content); sb_add(b, "\n");
    }
}

static void chat_open_assistant(sbuf *b, const char *fmt) {
    if (!strcmp(fmt, "chatml") || !strcmp(fmt, "qwen3")) sb_add(b, "<|im_start|>assistant\n");
    else if (!strcmp(fmt, "llama3")) sb_add(b, "<|start_header_id|>assistant<|end_header_id|>\n\n");
    else if (!strcmp(fmt, "zephyr")) sb_add(b, "<|assistant|>\n");
    else if (!strcmp(fmt, "plain")) sb_add(b, "assistant:");
}

/* ---- streaming output ------------------------------------------------------ */
typedef struct {
    sbuf *transcript;
} stream_state;

static int on_token(int32_t tok, const char *piece, int len, void *user) {
    (void)tok;
    stream_state *s = (stream_state *)user;
    fwrite(piece, 1, (size_t)len, stdout);
    fflush(stdout);
    if (s && s->transcript) {
        char tmp[4097];
        int n = len < 4096 ? len : 4096;
        memcpy(tmp, piece, (size_t)n);
        tmp[n] = 0;
        sb_add(s->transcript, tmp);
    }
    return 0;
}

static int32_t *tokenize_alloc(const borg_model *m, const char *text, int add_bos, int special, int *n) {
    int len = (int)strlen(text);
    int cap = len + 16;
    int32_t *toks = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    *n = borg_tokenize(m, text, len, add_bos, special, toks, cap);
    if (*n < 0) {
        cap = -*n;
        toks = (int32_t *)realloc(toks, sizeof(int32_t) * (size_t)cap);
        *n = borg_tokenize(m, text, len, add_bos, special, toks, cap);
    }
    return toks;
}

static void print_stats(const borg_gen_stats *st) {
    fprintf(stderr, "\n[prompt: %d tokens (%d reused) in %.1f ms = %.1f tok/s | gen: %d tokens in %.1f ms = %.1f tok/s",
            st->n_prompt, st->n_reused, st->t_prompt_ms, st->n_prompt ? st->n_prompt * 1000.0 / st->t_prompt_ms : 0.0,
            st->n_gen, st->t_gen_ms, st->n_gen ? st->n_gen * 1000.0 / st->t_gen_ms : 0.0);
    if (st->n_drafted)
        fprintf(stderr, " | drafted %d, accepted %d (%.0f%%), %d target passes", st->n_drafted, st->n_accepted,
                100.0 * st->n_accepted / st->n_drafted, st->n_target_evals);
    fprintf(stderr, "]\n");
}

static int run_bench(borg_model *m, borg_pool *pool, const options *o) {
    borg_config cfg;
    borg_model_config(m, &cfg);
    borg_ctx_params cp = borg_ctx_default_params();
    cp.n_ctx = o->bench_prompt + o->bench_gen + 8;
    cp.kv_f16 = o->kv_f16;
    borg_ctx *c = borg_ctx_new(m, pool, &cp);
    if (!c) { fprintf(stderr, "cannot allocate context\n"); return 1; }
    int32_t *toks = (int32_t *)malloc(sizeof(int32_t) * (size_t)o->bench_prompt);
    for (int i = 0; i < o->bench_prompt; i++) toks[i] = (int32_t)((i * 7919 + 13) % cfg.n_vocab);
    float *logits = (float *)malloc(sizeof(float) * (size_t)cfg.vocab_size);
    /* warm-up: page in the weights */
    borg_decode(c, toks, 8 < o->bench_prompt ? 8 : o->bench_prompt, logits, 1);
    borg_ctx_truncate(c, 0);
    double t0 = borg_time_ms();
    borg_decode(c, toks, o->bench_prompt, logits, 1);
    double t1 = borg_time_ms();
    int32_t tok = 0;
    for (int i = 0; i < o->bench_gen; i++) {
        int best = 0;
        for (int j = 1; j < cfg.n_vocab; j++)
            if (logits[j] > logits[best]) best = j;
        tok = best;
        borg_decode(c, &tok, 1, logits, 1);
    }
    double t2 = borg_time_ms();
    printf("model: %s | simd: %s | threads: %d | weights: %.1f MB\n", o->model, borg_simd(), borg_pool_size(pool),
           cfg.weight_bytes / 1e6);
    printf("prefill: %d tokens in %.1f ms = %.1f tok/s\n", o->bench_prompt, t1 - t0, o->bench_prompt * 1000.0 / (t1 - t0));
    printf("decode:  %d tokens in %.1f ms = %.1f tok/s\n", o->bench_gen, t2 - t1, o->bench_gen * 1000.0 / (t2 - t1));
    free(toks);
    free(logits);
    borg_ctx_free(c);
    return 0;
}

int main(int argc, char **argv) {
    options o;
    if (parse_args(argc, argv, &o)) return 1;
    char err[512];
    borg_model *m = borg_model_load(o.model, err, sizeof err);
    if (!m) { fprintf(stderr, "error: %s\n", err); return 1; }
    borg_pool *pool = borg_pool_new(o.threads);
    if (o.bench) {
        int rc = run_bench(m, pool, &o);
        borg_pool_free(pool);
        borg_model_free(m);
        return rc;
    }
    borg_model *dm = NULL;
    borg_ctx *dctx = NULL;
    borg_ctx_params cp = borg_ctx_default_params();
    cp.n_ctx = o.n_ctx;
    cp.kv_f16 = o.kv_f16;
    borg_ctx *c = borg_ctx_new(m, pool, &cp);
    if (!c) { fprintf(stderr, "error: cannot allocate context\n"); return 1; }
    if (o.draft) {
        dm = borg_model_load(o.draft, err, sizeof err);
        if (!dm) { fprintf(stderr, "error: draft: %s\n", err); return 1; }
        borg_ctx_params dp = cp;
        dp.n_ctx = borg_ctx_n_ctx(c);
        dctx = borg_ctx_new(dm, pool, &dp);
    }
    borg_gen_params gp = borg_gen_default_params();
    gp.max_tokens = o.n_predict;
    gp.sampling = o.s;
    gp.draft = dctx;
    gp.n_draft = o.n_draft;
    gp.lookup_ngram = o.lookup;
    gp.ignore_eos = o.ignore_eos;
    gp.json_mode = o.json;
    const int n_ctx = borg_ctx_n_ctx(c);
    int32_t *out = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_ctx);
    const char *fmt = chat_format(m);
    borg_gen_stats st;

    char *file_prompt = o.prompt_file ? read_file(o.prompt_file) : NULL;
    if (o.prompt_file && !file_prompt) { fprintf(stderr, "error: cannot read %s\n", o.prompt_file); return 1; }
    const char *prompt = file_prompt ? file_prompt : (o.prompt ? o.prompt : "");

    if (!o.interactive) {
        sbuf b = {0};
        sb_add(&b, "");
        if (o.chat) {
            if (o.system) chat_turn(&b, fmt, "system", o.system);
            chat_turn(&b, fmt, "user", prompt);
            chat_open_assistant(&b, fmt);
        } else {
            sb_add(&b, prompt);
            if (!o.verbose) printf("%s", prompt);
        }
        int n;
        int32_t *toks = tokenize_alloc(m, b.buf, -1, !o.raw, &n);
        if (n <= 0) { fprintf(stderr, "error: empty prompt\n"); return 1; }
        if (n >= n_ctx) { fprintf(stderr, "error: prompt (%d tokens) exceeds context %d\n", n, n_ctx); return 1; }
        stream_state ss = {NULL};
        int rc = borg_generate(c, toks, n, &gp, on_token, &ss, out, n_ctx, &st);
        printf("\n");
        if (rc < 0) fprintf(stderr, "error: generation failed (%d)\n", rc);
        if (o.verbose) print_stats(&st);
        free(toks);
        free(b.buf);
    } else {
        sbuf conv = {0};
        sb_add(&conv, "");
        if (o.system) chat_turn(&conv, fmt, "system", o.system);
        char line[8192];
        for (;;) {
            printf("\n> ");
            fflush(stdout);
            if (!fgets(line, sizeof line, stdin)) break;
            size_t L = strlen(line);
            while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            if (!L) continue;
            if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
            chat_turn(&conv, fmt, "user", line);
            sbuf full = {0};
            sb_add(&full, conv.buf);
            chat_open_assistant(&full, fmt);
            int n;
            int32_t *toks = tokenize_alloc(m, full.buf, -1, 1, &n);
            if (n >= n_ctx - 16) { fprintf(stderr, "[context full]\n"); free(toks); free(full.buf); break; }
            sbuf reply = {0};
            sb_add(&reply, "");
            stream_state ss = {&reply};
            borg_generate(c, toks, n, &gp, on_token, &ss, out, n_ctx, &st);
            if (o.verbose) print_stats(&st);
            chat_turn(&conv, fmt, "assistant", reply.buf);
            free(reply.buf);
            free(full.buf);
            free(toks);
        }
        free(conv.buf);
    }
    free(file_prompt);
    free(out);
    borg_ctx_free(c);
    borg_ctx_free(dctx);
    borg_model_free(dm);
    borg_pool_free(pool);
    borg_model_free(m);
    return 0;
}
