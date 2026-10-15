/* Token sampling: repetition penalty, temperature, top-k, top-p (nucleus), min-p.
 * The order follows Hugging Face generate(): penalty, temperature, top-k, top-p, min-p. */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct borg_sampler {
    borg_sampling p;
    int vocab;
    uint64_t rng;
    int32_t *idx;   /* [vocab] candidate ids */
    float *val;     /* [vocab] candidate values */
    uint32_t *mark; /* [vocab] stamp per token for de-duplicating the penalty window */
    uint32_t stamp;
};

borg_sampling borg_sampling_default(void) {
    borg_sampling s;
    s.temperature = 0.0f;
    s.top_k = 0;
    s.top_p = 1.0f;
    s.min_p = 0.0f;
    s.repeat_penalty = 1.0f;
    s.repeat_last_n = 64;
    s.seed = 0x5EEDu;
    return s;
}

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float rand_unit(borg_sampler *s) { return (float)(splitmix64(&s->rng) >> 40) * (1.0f / 16777216.0f); }

borg_sampler *borg_sampler_new(const borg_sampling *params, int vocab) {
    borg_sampler *s = (borg_sampler *)calloc(1, sizeof(borg_sampler));
    if (!s) return NULL;
    s->p = params ? *params : borg_sampling_default();
    s->vocab = vocab;
    s->rng = s->p.seed;
    s->idx = (int32_t *)malloc(sizeof(int32_t) * (size_t)vocab);
    s->val = (float *)malloc(sizeof(float) * (size_t)vocab);
    s->mark = (uint32_t *)calloc((size_t)vocab, sizeof(uint32_t));
    if (!s->idx || !s->val || !s->mark) {
        borg_sampler_free(s);
        return NULL;
    }
    return s;
}

void borg_sampler_free(borg_sampler *s) {
    if (!s) return;
    free(s->idx);
    free(s->val);
    free(s->mark);
    free(s);
}

void borg_sampler_reset(borg_sampler *s, uint64_t seed) { s->rng = seed; }

static void apply_penalty(borg_sampler *s, float *logits, const int32_t *hist, int n_hist) {
    const float pen = s->p.repeat_penalty;
    if (pen == 1.0f || !hist || n_hist <= 0) return;
    int start = 0;
    if (s->p.repeat_last_n > 0 && n_hist > s->p.repeat_last_n) start = n_hist - s->p.repeat_last_n;
    if (++s->stamp == 0) {
        memset(s->mark, 0, sizeof(uint32_t) * (size_t)s->vocab);
        s->stamp = 1;
    }
    for (int i = start; i < n_hist; i++) {
        const int32_t t = hist[i];
        if (t < 0 || t >= s->vocab || s->mark[t] == s->stamp) continue;
        s->mark[t] = s->stamp;
        logits[t] = logits[t] < 0.0f ? logits[t] * pen : logits[t] / pen;
    }
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* Partially order val/idx[0..n) so that the k largest values come first (unordered). */
static void select_top(float *val, int32_t *idx, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const float pivot = val[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (val[i] > pivot) i++;
            while (val[j] < pivot) j--;
            if (i <= j) {
                float tv = val[i]; val[i] = val[j]; val[j] = tv;
                int32_t ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
                i++;
                j--;
            }
        }
        if (k - 1 <= j) hi = j;
        else if (k - 1 >= i) lo = i;
        else break;
    }
}

static void sort_desc(float *val, int32_t *idx, int n) {
    /* insertion sort for short arrays, otherwise heap sort (no recursion, no allocation) */
    if (n < 64) {
        for (int i = 1; i < n; i++) {
            float v = val[i];
            int32_t id = idx[i];
            int j = i - 1;
            while (j >= 0 && val[j] < v) {
                val[j + 1] = val[j];
                idx[j + 1] = idx[j];
                j--;
            }
            val[j + 1] = v;
            idx[j + 1] = id;
        }
        return;
    }
    /* min-heap so the final array is descending */
    for (int start = n / 2 - 1; start >= 0; start--) {
        int root = start;
        for (;;) {
            int child = 2 * root + 1;
            if (child >= n) break;
            if (child + 1 < n && val[child + 1] < val[child]) child++;
            if (val[child] < val[root]) {
                float tv = val[root]; val[root] = val[child]; val[child] = tv;
                int32_t ti = idx[root]; idx[root] = idx[child]; idx[child] = ti;
                root = child;
            } else break;
        }
    }
    for (int end = n - 1; end > 0; end--) {
        float tv = val[0]; val[0] = val[end]; val[end] = tv;
        int32_t ti = idx[0]; idx[0] = idx[end]; idx[end] = ti;
        int root = 0;
        for (;;) {
            int child = 2 * root + 1;
            if (child >= end) break;
            if (child + 1 < end && val[child + 1] < val[child]) child++;
            if (val[child] < val[root]) {
                float tv2 = val[root]; val[root] = val[child]; val[child] = tv2;
                int32_t ti2 = idx[root]; idx[root] = idx[child]; idx[child] = ti2;
                root = child;
            } else break;
        }
    }
}

/* Build the candidate list (idx/val with probabilities, sorted descending when needed).
 * Returns the number of candidates. Requires temperature > 0. */
static int build_candidates(borg_sampler *s, const float *logits) {
    const int V = s->vocab;
    const float inv_t = 1.0f / s->p.temperature;
    int n = V;
    for (int i = 0; i < V; i++) {
        s->idx[i] = i;
        s->val[i] = logits[i] * inv_t;
    }
    if (s->p.top_k > 0 && s->p.top_k < n) {
        select_top(s->val, s->idx, n, s->p.top_k);
        n = s->p.top_k;
    }
    /* softmax over the candidates */
    float mx = s->val[0];
    for (int i = 1; i < n; i++)
        if (s->val[i] > mx) mx = s->val[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        s->val[i] = expf(s->val[i] - mx);
        sum += s->val[i];
    }
    const float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) s->val[i] *= inv;

    if (s->p.top_p < 1.0f) {
        /* sorting 150k entries per token is slow; tokens below 1e-9 cannot change the nucleus
         * unless the kept mass falls short, which is checked below */
        int m = n;
        if (n > 1024) {
            m = 0;
            for (int i = 0; i < n; i++)
                if (s->val[i] >= 1e-9f) {
                    float tv = s->val[m]; s->val[m] = s->val[i]; s->val[i] = tv;
                    int32_t ti = s->idx[m]; s->idx[m] = s->idx[i]; s->idx[i] = ti;
                    m++;
                }
        }
        sort_desc(s->val, s->idx, m);
        double cum = 0.0;
        int keep = m;
        for (int i = 0; i < m; i++) {
            cum += s->val[i];
            if (cum >= s->p.top_p) {
                keep = i + 1;
                break;
            }
        }
        if (cum < s->p.top_p && m < n) {
            sort_desc(s->val, s->idx, n);
            keep = n;
            cum = 0.0;
            for (int i = 0; i < n; i++) {
                cum += s->val[i];
                if (cum >= s->p.top_p) {
                    keep = i + 1;
                    break;
                }
            }
        }
        n = keep;
    }
    if (s->p.min_p > 0.0f && n > 1) {
        float pmax = 0.0f;
        for (int i = 0; i < n; i++)
            if (s->val[i] > pmax) pmax = s->val[i];
        const float thr = s->p.min_p * pmax;
        int m = 0;
        for (int i = 0; i < n; i++)
            if (s->val[i] >= thr) {
                s->val[m] = s->val[i];
                s->idx[m] = s->idx[i];
                m++;
            }
        n = m;
    }
    double tot = 0.0;
    for (int i = 0; i < n; i++) tot += s->val[i];
    const float r = (float)(1.0 / tot);
    for (int i = 0; i < n; i++) s->val[i] *= r;
    return n;
}

int32_t borg_sampler_sample(borg_sampler *s, float *logits, const int32_t *history, int n_history) {
    apply_penalty(s, logits, history, n_history);
    if (s->p.temperature <= 0.0f) return argmax(logits, s->vocab);
    const int n = build_candidates(s, logits);
    const float r = rand_unit(s);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += s->val[i];
        if (r < cum) return s->idx[i];
    }
    return s->idx[n - 1];
}

void borg_sampler_probs(borg_sampler *s, float *logits, const int32_t *history, int n_history) {
    apply_penalty(s, logits, history, n_history);
    if (s->p.temperature <= 0.0f) {
        const int best = argmax(logits, s->vocab);
        memset(logits, 0, sizeof(float) * (size_t)s->vocab);
        logits[best] = 1.0f;
        return;
    }
    const int n = build_candidates(s, logits);
    memset(logits, 0, sizeof(float) * (size_t)s->vocab);
    for (int i = 0; i < n; i++) logits[s->idx[i]] = s->val[i];
}

/* draw from an explicit distribution (used by speculative sampling) */
int32_t sampler_draw(borg_sampler *s, const float *probs, int n) {
    double tot = 0.0;
    for (int i = 0; i < n; i++) tot += probs[i];
    const double r = (double)rand_unit(s) * tot;
    double cum = 0.0;
    int last = 0;
    for (int i = 0; i < n; i++) {
        if (probs[i] <= 0.0f) continue;
        cum += probs[i];
        last = i;
        if (r < cum) return i;
    }
    return last;
}

float sampler_uniform(borg_sampler *s) { return rand_unit(s); }
