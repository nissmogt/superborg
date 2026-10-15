/* Text generation with KV-cache prefix reuse and speculative decoding.
 *
 * Speculative decoding: a cheap proposer guesses the next K tokens, the target model
 * scores all of them in one batched forward pass, and the longest prefix the target
 * agrees with is kept, plus one token from the target itself. Proposers:
 *   - a small draft model with the same tokenizer, or
 *   - prompt lookup: the continuation of the most recent earlier occurrence of the
 *     last n tokens (no extra model; strong on summarization, code edits, RAG).
 * With temperature > 0 acceptance uses the rejection rule of Leviathan et al. (2023)
 * and Chen et al. (2023), so the output distribution equals plain sampling from the
 * target model. With greedy decoding the output is identical to plain greedy decoding.
 */
#include "internal.h"

#include <stdlib.h>
#include <string.h>

borg_gen_params borg_gen_default_params(void) {
    borg_gen_params p;
    memset(&p, 0, sizeof p);
    p.max_tokens = 256;
    p.sampling = borg_sampling_default();
    p.n_draft = 6;
    return p;
}

/* Make the cache hold exactly toks[0..n), reusing the shared prefix. With logits != NULL the
 * last token is always evaluated so its logits are available. */
static int sync_ctx(borg_ctx *c, const int32_t *toks, int n, float *logits, int *n_reused) {
    int common = 0;
    const int lim = c->n_past < n ? c->n_past : n;
    while (common < lim && c->tokens[common] == toks[common]) common++;
    if (logits && common == n) common = n - 1;
    borg_ctx_truncate(c, common);
    if (n_reused) *n_reused = common;
    if (n > common) return borg_decode(c, toks + common, n - common, logits, logits ? 1 : 0);
    return 0;
}

typedef struct {
    borg_ctx *c;
    const borg_gen_params *p;
    borg_token_cb cb;
    void *user;
    int32_t *seq; /* prompt + generated tokens */
    int len;
    int n_gen;
    int32_t *out;
    int max_out;
    int stop;     /* -1 running, else stop reason */
    char *piece;
    int piece_cap;
} gen_state;

static int is_stop_token(const gen_state *g, int32_t tok) {
    if (!g->p->ignore_eos && borg_is_eog(g->c->model, tok)) return 1;
    for (int i = 0; i < g->p->n_stop_tokens; i++)
        if (g->p->stop_tokens[i] == tok) return 1;
    return 0;
}

/* Emit one token. Returns 1 if generation must stop afterwards. */
static int emit(gen_state *g, int32_t tok) {
    if (is_stop_token(g, tok)) {
        g->stop = 1;
        return 1;
    }
    g->seq[g->len++] = tok;
    if (g->out && g->n_gen < g->max_out) g->out[g->n_gen] = tok;
    g->n_gen++;
    if (g->cb) {
        int n = borg_token_bytes(g->c->model, tok, g->piece, g->piece_cap);
        if (n > g->piece_cap) n = g->piece_cap;
        if (g->cb(tok, g->piece, n, g->user)) {
            g->stop = 2;
            return 1;
        }
    }
    if (g->n_gen >= g->p->max_tokens) {
        g->stop = 0;
        return 1;
    }
    return 0;
}

/* Prompt lookup: continuation of the latest earlier match of the trailing n-gram. */
static int lookup_propose(const int32_t *seq, int len, int ngram, int k, int32_t *draft) {
    for (int g = ngram; g >= 1; g--) {
        if (len < g + 1) continue;
        const int32_t *pat = seq + len - g;
        for (int i = len - g - 1; i >= 0; i--) {
            if (memcmp(seq + i, pat, (size_t)g * sizeof(int32_t)) != 0) continue;
            int n = 0;
            while (n < k && i + g + n < len) {
                draft[n] = seq[i + g + n];
                n++;
            }
            if (n > 0) return n;
        }
    }
    return 0;
}

int borg_generate(borg_ctx *c, const int32_t *prompt, int n_prompt, const borg_gen_params *params,
                  borg_token_cb cb, void *user, int32_t *out, int max_out, borg_gen_stats *stats) {
    borg_gen_params p = params ? *params : borg_gen_default_params();
    borg_gen_stats st;
    memset(&st, 0, sizeof st);
    if (n_prompt <= 0 || n_prompt > c->n_ctx) return -1;
    const borg_model *m = c->model;
    const int Vt = m->cfg.vocab_size;
    int Vs = m->cfg.n_vocab; /* ids the sampler may produce */
    borg_ctx *dc = p.draft;
    if (dc) {
        if (dc->model->cfg.n_vocab != m->cfg.n_vocab) return -2;
        if (dc->n_ctx < c->n_ctx) return -3;
    }
    int K = p.n_draft;
    if (!dc && p.lookup_ngram <= 0) K = 0;
    if (p.json_mode) K = 0;
    if (K < 0) K = 0;
    if (K > 32) K = 32;
    const int greedy = p.sampling.temperature <= 0.0f;

    gen_state g;
    memset(&g, 0, sizeof g);
    g.c = c;
    g.p = &p;
    g.cb = cb;
    g.user = user;
    g.out = out;
    g.max_out = out ? max_out : 0;
    g.stop = -1;
    g.piece_cap = 4096;
    const int seq_cap = c->n_ctx + 64;
    g.seq = (int32_t *)malloc(sizeof(int32_t) * (size_t)seq_cap);
    g.piece = (char *)malloc((size_t)g.piece_cap);
    float *logits = (float *)malloc(sizeof(float) * (size_t)Vt * (size_t)(K + 1));
    float *L0 = (float *)malloc(sizeof(float) * (size_t)Vt);
    float *qd = K && dc ? (float *)malloc(sizeof(float) * (size_t)Vs * (size_t)K) : NULL;
    float *dlog = dc ? (float *)malloc(sizeof(float) * (size_t)dc->model->cfg.vocab_size) : NULL;
    int32_t draft[33];
    borg_sampler *smp = borg_sampler_new(&p.sampling, Vs);
    json_state js;
    json_state_init(&js);
    const int json_want = p.sampling.temperature <= 0.0f ? 1 : (p.sampling.top_k > 0 ? p.sampling.top_k : 64);
    int32_t *jidx = p.json_mode ? (int32_t *)malloc(sizeof(int32_t) * (size_t)Vs) : NULL;
    float *jval = p.json_mode ? (float *)malloc(sizeof(float) * (size_t)Vs) : NULL;
    uint8_t *jkeep = p.json_mode ? (uint8_t *)malloc((size_t)Vs) : NULL;
    int rc = 0;
    if (!g.seq || !g.piece || !logits || !L0 || (K && dc && !qd) || (dc && !dlog) || !smp ||
        (p.json_mode && (!jidx || !jval || !jkeep))) {
        rc = -4;
        goto done;
    }
    memcpy(g.seq, prompt, sizeof(int32_t) * (size_t)n_prompt);
    g.len = n_prompt;

    double t0 = borg_time_ms();
    if (sync_ctx(c, prompt, n_prompt, L0, &st.n_reused)) {
        rc = -5;
        goto done;
    }
    st.n_prompt = n_prompt - st.n_reused;
    double t1 = borg_time_ms();
    st.t_prompt_ms = t1 - t0;
    if (p.max_tokens <= 0) {
        g.stop = 0;
        goto finish;
    }

    int pending = -1; /* last emitted token, not yet in the cache */
    while (g.stop < 0) {
        const int room = c->n_ctx - c->n_past - (pending >= 0 ? 1 : 0);
        if (room < 0) {
            g.stop = 3;
            break;
        }
        /* propose */
        int k = 0;
        int kmax = K;
        if (kmax > room) kmax = room;
        if (kmax > p.max_tokens - g.n_gen - 1) kmax = p.max_tokens - g.n_gen - 1;
        if (kmax > 0) {
            if (dc) {
                if (sync_ctx(dc, g.seq, g.len, dlog, NULL)) {
                    rc = -6;
                    break;
                }
                for (; k < kmax; k++) {
                    float *q = qd + (size_t)k * Vs;
                    memcpy(q, dlog, sizeof(float) * (size_t)Vs);
                    borg_sampler_probs(smp, q, g.seq, g.len + k); /* one-hot when greedy */
                    draft[k] = sampler_draw(smp, q, Vs);
                    g.seq[g.len + k] = draft[k]; /* scratch, overwritten by emit */
                    if (k + 1 < kmax && borg_decode(dc, &draft[k], 1, dlog, 1)) {
                        rc = -6;
                        break;
                    }
                }
                if (rc) break;
            } else {
                k = lookup_propose(g.seq, g.len, p.lookup_ngram, kmax, draft);
            }
        }
        st.n_drafted += k;

        /* score pending + drafts with the target in one pass */
        int32_t batch[34];
        int nb = 0;
        if (pending >= 0) batch[nb++] = pending;
        for (int i = 0; i < k; i++) batch[nb++] = draft[i];
        const float *Lrow[34]; /* Lrow[i]: target logits for the choice after i accepted drafts */
        if (nb > 0) {
            if (borg_decode(c, batch, nb, logits, nb > 1 ? 2 : 1)) {
                rc = -5;
                break;
            }
            st.n_target_evals++;
        }
        if (pending >= 0) {
            for (int i = 0; i <= k; i++) Lrow[i] = logits + (size_t)i * Vt;
        } else {
            Lrow[0] = L0;
            for (int i = 1; i <= k; i++) Lrow[i] = logits + (size_t)(i - 1) * Vt;
        }
        const int cache_base = c->n_past - nb + (pending >= 0 ? 1 : 0); /* cache length incl. pending */

        /* accept */
        int acc = 0;
        int32_t next = -1;
        for (int i = 0; i <= k; i++) {
            float *Li = (float *)Lrow[i];
            if (p.json_mode) json_constrain(m, &js, Li, Vs, jidx, jval, jkeep, json_want);
            if (greedy) {
                int32_t y = borg_sampler_sample(smp, Li, g.seq, g.len);
                if (i < k && y == draft[i]) {
                    acc++;
                    if (emit(&g, y)) break;
                    continue;
                }
                next = y;
                break;
            }
            borg_sampler_probs(smp, Li, g.seq, g.len);
            if (i == k) {
                next = sampler_draw(smp, Li, Vs);
                break;
            }
            const int32_t d = draft[i];
            const float pd = Li[d];
            const float qdv = dc ? qd[(size_t)i * Vs + d] : 1.0f;
            if (qdv > 0.0f && sampler_uniform(smp) * qdv < pd) {
                acc++;
                if (emit(&g, d)) break;
                continue;
            }
            /* rejected: sample from the residual max(0, p - q) */
            if (dc) {
                const float *q = qd + (size_t)i * Vs;
                for (int j = 0; j < Vs; j++) {
                    float r = Li[j] - q[j];
                    Li[j] = r > 0.0f ? r : 0.0f;
                }
            } else {
                Li[d] = 0.0f;
            }
            next = sampler_draw(smp, Li, Vs);
            break;
        }
        st.n_accepted += acc;
        /* the cache holds pending + all drafts; keep pending + the accepted ones */
        borg_ctx_truncate(c, cache_base + acc);
        pending = -1;
        if (g.stop < 0 && next >= 0) {
            if (p.json_mode) json_state_feed_token(m, &js, next);
            if (!emit(&g, next)) pending = next;
        }
        if (g.stop < 0 && pending < 0) {
            /* nothing to continue from (cannot happen) */
            g.stop = 0;
        }
    }

finish:
    st.n_gen = g.n_gen;
    st.t_gen_ms = borg_time_ms() - t1;
    st.stop_reason = g.stop < 0 ? 0 : g.stop;
done:
    if (stats) *stats = st;
    borg_sampler_free(smp);
    free(g.seq);
    free(g.piece);
    free(logits);
    free(L0);
    free(qd);
    free(dlog);
    free(jidx);
    free(jval);
    free(jkeep);
    return rc ? rc : g.n_gen;
}

int borg_generate_multi(borg_ctx **ctxs, int n_seq, const int32_t *prompt, int n_prompt,
                        const borg_gen_params *params, borg_multi_cb cb, void *user, int *n_generated,
                        int *stop_reasons, borg_gen_stats *stats) {
    borg_gen_params p = params ? *params : borg_gen_default_params();
    borg_gen_stats st;
    memset(&st, 0, sizeof st);
    if (n_seq <= 0 || n_prompt <= 0) return -1;
    borg_ctx *lead = ctxs[0];
    if (n_seq > lead->n_batch) return -4;
    if (n_prompt >= lead->n_ctx) return -1;
    const borg_model *m = lead->model;
    const int Vt = m->cfg.vocab_size, Vs = m->cfg.n_vocab;
    const int hist_cap = lead->n_ctx + 1;

    int rc = 0;
    float *logits = (float *)malloc(sizeof(float) * (size_t)Vt * (size_t)n_seq);
    float **row = (float **)calloc((size_t)n_seq, sizeof(float *));
    int32_t *hist = (int32_t *)malloc(sizeof(int32_t) * (size_t)hist_cap * (size_t)n_seq);
    int *len = (int *)calloc((size_t)n_seq, sizeof(int));
    int *ngen = (int *)calloc((size_t)n_seq, sizeof(int));
    int *reason = (int *)calloc((size_t)n_seq, sizeof(int));
    int *active = (int *)calloc((size_t)n_seq, sizeof(int));
    borg_ctx **act_ctx = (borg_ctx **)calloc((size_t)n_seq, sizeof(borg_ctx *));
    int32_t *act_tok = (int32_t *)calloc((size_t)n_seq, sizeof(int32_t));
    int *act_idx = (int *)calloc((size_t)n_seq, sizeof(int));
    borg_sampler **smp = (borg_sampler **)calloc((size_t)n_seq, sizeof(borg_sampler *));
    json_state *js = (json_state *)calloc((size_t)n_seq, sizeof(json_state));
    const int json_want = p.sampling.temperature <= 0.0f ? 1 : (p.sampling.top_k > 0 ? p.sampling.top_k : 64);
    int32_t *jidx = p.json_mode ? (int32_t *)malloc(sizeof(int32_t) * (size_t)Vs) : NULL;
    float *jval = p.json_mode ? (float *)malloc(sizeof(float) * (size_t)Vs) : NULL;
    uint8_t *jkeep = p.json_mode ? (uint8_t *)malloc((size_t)Vs) : NULL;
    char piece[4096];
    if (!logits || !row || !hist || !len || !ngen || !reason || !active || !act_ctx || !act_tok || !act_idx || !smp ||
        !js || (p.json_mode && (!jidx || !jval || !jkeep))) {
        rc = -4;
        goto done;
    }
    for (int i = 0; i < n_seq; i++) {
        borg_sampling sp = p.sampling;
        sp.seed = p.sampling.seed + (uint64_t)i;
        smp[i] = borg_sampler_new(&sp, Vs);
        if (!smp[i]) { rc = -4; goto done; }
        json_state_init(&js[i]);
    }

    double t0 = borg_time_ms();
    if (sync_ctx(lead, prompt, n_prompt, logits, &st.n_reused)) { rc = -5; goto done; }
    st.n_prompt = n_prompt - st.n_reused;
    for (int i = 0; i < n_seq; i++) {
        if (i > 0) {
            if (borg_ctx_copy(ctxs[i], lead)) { rc = -5; goto done; }
            memcpy(logits + (size_t)i * Vt, logits, sizeof(float) * (size_t)Vt);
        }
        row[i] = logits + (size_t)i * Vt;
        memcpy(hist + (size_t)i * hist_cap, prompt, sizeof(int32_t) * (size_t)n_prompt);
        len[i] = n_prompt;
        active[i] = p.max_tokens > 0;
    }
    double t1 = borg_time_ms();
    st.t_prompt_ms = t1 - t0;

    float *step = (float *)malloc(sizeof(float) * (size_t)Vt * (size_t)n_seq);
    if (!step) { rc = -4; goto done; }
    for (;;) {
        int na = 0;
        for (int i = 0; i < n_seq; i++) {
            if (!active[i]) continue;
            int32_t *h = hist + (size_t)i * hist_cap;
            if (p.json_mode) json_constrain(m, &js[i], row[i], Vs, jidx, jval, jkeep, json_want);
            const int32_t tok = borg_sampler_sample(smp[i], row[i], h, len[i]);
            if (p.json_mode) json_state_feed_token(m, &js[i], tok);
            int stop = !p.ignore_eos && borg_is_eog(m, tok);
            for (int k = 0; k < p.n_stop_tokens && !stop; k++) stop = p.stop_tokens[k] == tok;
            if (stop) { active[i] = 0; reason[i] = 1; continue; }
            h[len[i]++] = tok;
            ngen[i]++;
            st.n_gen++;
            if (cb) {
                int n = borg_token_bytes(m, tok, piece, (int)sizeof piece);
                if (n > (int)sizeof piece) n = (int)sizeof piece;
                if (cb(i, tok, piece, n, user)) { active[i] = 0; reason[i] = 2; continue; }
            }
            if (ngen[i] >= p.max_tokens) { active[i] = 0; reason[i] = 0; continue; }
            if (ctxs[i]->n_past + 1 >= ctxs[i]->n_ctx) { active[i] = 0; reason[i] = 3; continue; }
            act_ctx[na] = ctxs[i];
            act_tok[na] = tok;
            act_idx[na] = i;
            na++;
        }
        if (na == 0) break;
        if (borg_decode_multi(act_ctx, na, act_tok, step)) { rc = -5; break; }
        st.n_target_evals++;
        for (int k = 0; k < na; k++) memcpy(row[act_idx[k]], step + (size_t)k * Vt, sizeof(float) * (size_t)Vt);
    }
    free(step);
    st.t_gen_ms = borg_time_ms() - t1;
    if (n_generated) memcpy(n_generated, ngen, sizeof(int) * (size_t)n_seq);
    if (stop_reasons) memcpy(stop_reasons, reason, sizeof(int) * (size_t)n_seq);
done:
    if (stats) *stats = st;
    if (smp)
        for (int i = 0; i < n_seq; i++) borg_sampler_free(smp[i]);
    free(smp);
    free(logits);
    free(row);
    free(hist);
    free(len);
    free(ngen);
    free(reason);
    free(active);
    free(act_ctx);
    free(act_tok);
    free(act_idx);
    free(js);
    free(jidx);
    free(jval);
    free(jkeep);
    return rc ? rc : st.n_gen;
}
