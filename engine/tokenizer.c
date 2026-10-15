/* Tokenizer: byte-level BPE (GPT-2 family) and SentencePiece-style BPE, written to give the
 * same tokens as Hugging Face `tokenizers` for the same tokenizer.json.
 *
 * Pipeline, in HF's order:
 *   1. added tokens ("<|im_start|>") are cut out of the raw text, leftmost-longest match;
 *   2. every segment between them is pre-tokenized: byte-level models split it with one of
 *      the published regexes (implemented below as hand-written matchers, no regex engine);
 *      SentencePiece models prepend "▁", turn every space into "▁" and keep it whole;
 *   3. every piece goes through BPE: repeatedly merge the adjacent pair with the lowest merge
 *      rank, the leftmost one on ties (HF's priority-queue algorithm, reproduced exactly).
 *
 * All state built here is read-only afterwards; a tokenize call allocates its own scratch,
 * so one model can be used from several threads at once. */
#include "internal.h"
#include "unicode_tables.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define CP_INVALID 0xFFFFFFFFu /* an undecodable byte: its own "other" character */

/* ---- hash tables ---------------------------------------------------------- */
typedef struct {
    uint32_t hash;
    int32_t id; /* -1: empty */
} vocab_slot;

typedef struct {
    uint64_t key; /* left << 32 | right; UINT64_MAX: empty */
    int32_t rank, id;
} merge_slot;

typedef struct {
    int32_t child, sibling; /* first child / next sibling, -1 = none */
    int32_t tok;            /* token ending at this node, or -1 */
    uint8_t byte;
} trie_node;

struct borg_tok_impl {
    vocab_slot *vtab;
    uint32_t vmask;
    merge_slot *mtab;
    uint32_t mmask;
    /* byte-level: token of each single byte; SentencePiece: the <0xXX> byte tokens */
    int32_t byte_tok[256];
    int spm_split; /* SentencePiece: pieces can be cut before every word-initial "▁" */
    /* added tokens, matched verbatim when parse_special */
    trie_node *trie;
    int n_trie;
    int32_t root[256];
};

static inline uint32_t hash_bytes(const uint8_t *p, int n) {
    uint32_t h = 2166136261u; /* FNV-1a */
    for (int i = 0; i < n; i++) h = (h ^ p[i]) * 16777619u;
    return h;
}

static inline uint32_t hash_pair(uint64_t key) {
    key *= 0x9E3779B97F4A7C15ull;
    return (uint32_t)(key >> 32);
}

static uint32_t pow2_at_least(uint64_t n) {
    uint32_t c = 16;
    while (c < n && c < (1u << 30)) c <<= 1;
    return c;
}

static inline const uint8_t *tok_str(const borg_tokenizer *t, int32_t id, int *len) {
    *len = (int)(t->vocab_off[id + 1] - t->vocab_off[id]);
    return t->vocab + t->vocab_off[id];
}

/* Id of the normal (model-vocabulary) token with exactly these bytes, or -1. */
static int32_t vocab_find(const borg_tokenizer *t, const uint8_t *p, int n) {
    const borg_tok_impl *im = t->impl;
    uint32_t h = hash_bytes(p, n);
    for (uint32_t i = h & im->vmask;; i = (i + 1) & im->vmask) {
        const vocab_slot *s = &im->vtab[i];
        if (s->id < 0) return -1;
        if (s->hash != h) continue;
        int len;
        const uint8_t *q = tok_str(t, s->id, &len);
        if (len == n && !memcmp(p, q, (size_t)n)) return s->id;
    }
}

static inline const merge_slot *merge_find(const borg_tok_impl *im, int32_t left, int32_t right) {
    uint64_t key = (uint64_t)(uint32_t)left << 32 | (uint32_t)right;
    for (uint32_t i = hash_pair(key) & im->mmask;; i = (i + 1) & im->mmask) {
        const merge_slot *s = &im->mtab[i];
        if (s->key == key) return s;
        if (s->key == UINT64_MAX) return NULL;
    }
}

/* ---- init ----------------------------------------------------------------- */
void tokenizer_free(borg_tokenizer *t) {
    if (!t || !t->impl) return;
    free(t->impl->vtab);
    free(t->impl->mtab);
    free(t->impl->trie);
    free(t->impl);
    t->impl = NULL;
}

int tokenizer_init(borg_tokenizer *t, char *err, int errlen) {
    if (t->n_vocab <= 0 || !t->vocab || !t->vocab_off || !t->token_type) {
        borg_set_err(err, errlen, "tokenizer has no vocabulary");
        return -1;
    }
    if (t->n_vocab > (1 << 28) || t->n_merges > (1 << 28) || t->n_merges < 0) {
        borg_set_err(err, errlen, "tokenizer tables too large");
        return -1;
    }
    borg_tok_impl *im = (borg_tok_impl *)calloc(1, sizeof(borg_tok_impl));
    if (!im) goto oom;
    t->impl = im;

    /* token bytes -> id. Only normal tokens: byte tokens are reached through byte_tok and
     * added tokens through the trie, as HF keeps them apart from the BPE vocabulary. */
    uint32_t vcap = pow2_at_least((uint64_t)t->n_vocab * 2);
    im->vmask = vcap - 1;
    im->vtab = (vocab_slot *)malloc(sizeof(vocab_slot) * vcap);
    if (!im->vtab) goto oom;
    for (uint32_t i = 0; i < vcap; i++) im->vtab[i].id = -1;
    for (int32_t id = 0; id < t->n_vocab; id++) {
        int len;
        const uint8_t *p = tok_str(t, id, &len);
        if (t->token_type[id] != TT_NORMAL || len == 0) continue;
        uint32_t h = hash_bytes(p, len), i = h & im->vmask;
        int dup = 0;
        for (; im->vtab[i].id >= 0; i = (i + 1) & im->vmask) {
            int l2;
            const uint8_t *q = tok_str(t, im->vtab[i].id, &l2);
            if (im->vtab[i].hash == h && l2 == len && !memcmp(p, q, (size_t)len)) { dup = 1; break; }
        }
        if (dup) continue; /* first id wins */
        im->vtab[i].hash = h;
        im->vtab[i].id = id;
    }

    for (int b = 0; b < 256; b++) im->byte_tok[b] = -1;
    if (t->byte_level) {
        for (int b = 0; b < 256; b++) {
            uint8_t c = (uint8_t)b;
            im->byte_tok[b] = vocab_find(t, &c, 1);
        }
    } else {
        for (int32_t id = 0; id < t->n_vocab; id++) {
            int len;
            const uint8_t *p = tok_str(t, id, &len);
            if (t->token_type[id] == TT_BYTE && len == 1 && im->byte_tok[p[0]] < 0) im->byte_tok[p[0]] = id;
        }
    }
    if (!t->byte_level) {
        /* Cutting SentencePiece text before word-initial "▁" (see segment_spm) is exact when no
         * token a merge can produce holds "▁" right after another character, and "▁" itself is
         * a token (so it ends a run of unknown characters, as a cut does). */
        static const uint8_t meta[3] = {0xE2, 0x96, 0x81};
        im->spm_split = vocab_find(t, meta, 3) >= 0;
        for (int32_t id = 0; id < t->n_vocab && im->spm_split; id++) {
            int len;
            const uint8_t *p = tok_str(t, id, &len);
            if (t->token_type[id] == TT_BYTE) continue;
            for (int i = 1; i + 3 <= len; i++)
                if (p[i] == 0xE2 && p[i + 1] == 0x96 && p[i + 2] == 0x81 &&
                    !(i >= 3 && p[i - 3] == 0xE2 && p[i - 2] == 0x96 && p[i - 1] == 0x81)) {
                    im->spm_split = 0;
                    break;
                }
        }
    }

    /* (left, right) -> (rank, merged). A repeated pair keeps its last rank, like HF's
     * HashMap built from the merge list. */
    uint32_t mcap = pow2_at_least((uint64_t)(t->n_merges > 0 ? t->n_merges : 1) * 2);
    im->mmask = mcap - 1;
    im->mtab = (merge_slot *)malloc(sizeof(merge_slot) * mcap);
    if (!im->mtab) goto oom;
    for (uint32_t i = 0; i < mcap; i++) im->mtab[i].key = UINT64_MAX;
    for (int r = 0; r < t->n_merges; r++) {
        const uint32_t *m = t->merges + (size_t)r * 3;
        uint64_t key = (uint64_t)m[0] << 32 | m[1];
        uint32_t i = hash_pair(key) & im->mmask;
        while (im->mtab[i].key != UINT64_MAX && im->mtab[i].key != key) i = (i + 1) & im->mmask;
        im->mtab[i].key = key;
        im->mtab[i].rank = r;
        im->mtab[i].id = (int32_t)m[2];
    }

    /* trie of added tokens */
    for (int b = 0; b < 256; b++) im->root[b] = -1;
    size_t total = 0;
    for (int k = 0; k < t->n_added; k++) {
        int len;
        tok_str(t, t->added[k], &len);
        total += (size_t)len;
    }
    if (total) {
        im->trie = (trie_node *)malloc(sizeof(trie_node) * total);
        if (!im->trie) goto oom;
    }
    for (int k = 0; k < t->n_added; k++) {
        int len;
        const uint8_t *p = tok_str(t, t->added[k], &len);
        if (len == 0) continue;
        int32_t *link = &im->root[p[0]], node = -1;
        for (int d = 0; d < len; d++) {
            if (d > 0) {
                link = &im->trie[node].child;
                while (*link >= 0 && im->trie[*link].byte != p[d]) link = &im->trie[*link].sibling;
            }
            if (*link < 0) {
                trie_node *nd = &im->trie[im->n_trie];
                nd->child = nd->sibling = nd->tok = -1;
                nd->byte = p[d];
                *link = im->n_trie++;
            }
            node = *link;
        }
        if (im->trie[node].tok < 0) im->trie[node].tok = t->added[k]; /* first id wins */
    }
    return 0;
oom:
    tokenizer_free(t);
    borg_set_err(err, errlen, "out of memory");
    return -1;
}

/* ---- UTF-8 and character classes ----------------------------------------- */
/* Decode the character at p[0..n), n >= 1. Invalid or truncated sequences (which a Rust
 * string can never hold) yield CP_INVALID for a single byte. */
static inline int utf8_next(const uint8_t *p, int n, uint32_t *cp) {
    uint32_t c = p[0];
    if (c < 0x80) { *cp = c; return 1; }
    if (c >= 0xC2 && c <= 0xDF) {
        if (n >= 2 && (p[1] & 0xC0) == 0x80) { *cp = (c & 0x1F) << 6 | (p[1] & 0x3F); return 2; }
    } else if (c >= 0xE0 && c <= 0xEF) {
        if (n >= 3 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
            uint32_t v = (c & 0x0F) << 12 | (uint32_t)(p[1] & 0x3F) << 6 | (p[2] & 0x3F);
            if (v >= 0x800 && (v < 0xD800 || v > 0xDFFF)) { *cp = v; return 3; }
        }
    } else if (c >= 0xF0 && c <= 0xF4) {
        if (n >= 4 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
            uint32_t v = (c & 0x07) << 18 | (uint32_t)(p[1] & 0x3F) << 12 | (uint32_t)(p[2] & 0x3F) << 6 | (p[3] & 0x3F);
            if (v >= 0x10000 && v <= 0x10FFFF) { *cp = v; return 4; }
        }
    }
    *cp = CP_INVALID;
    return 1;
}

/* \p{L}, \p{N}, \s or other, as Oniguruma sees them (see tools/gen_unicode_tables.py). */
static inline int uc_class(uint32_t c) {
    if (c < 0x80) {
        if ((c | 0x20) - 'a' < 26u) return UC_L;
        if (c - '0' < 10u) return UC_N;
        if (c == ' ' || c - 9u < 5u) return UC_S;
        return UC_O;
    }
    int lo = 0, hi = BORG_UC_NCLASS - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (c < borg_uc_class[mid].lo) hi = mid - 1;
        else if (c > borg_uc_class[mid].hi) lo = mid + 1;
        else return borg_uc_class[mid].cls;
    }
    return UC_O;
}

/* Rust char::is_numeric (HF's Digits pre-tokenizer). */
static int uc_numeric(uint32_t c) {
    if (c < 0x80) return c - '0' < 10u;
    int lo = 0, hi = BORG_UC_NNUMERIC - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (c < borg_uc_numeric[mid][0]) hi = mid - 1;
        else if (c > borg_uc_numeric[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

/* Class of the character at byte i of p[0..n); its length goes to *len, its code to *cp. */
static inline int class_at(const uint8_t *p, int n, int i, int *len, uint32_t *cp) {
    uint32_t c;
    *len = utf8_next(p + i, n - i, &c);
    if (cp) *cp = c;
    return c == CP_INVALID ? UC_O : uc_class(c);
}

/* End of the run of class cls starting at byte i. */
static inline int run_end(const uint8_t *p, int n, int i, int cls) {
    while (i < n) {
        int len;
        if (class_at(p, n, i, &len, NULL) != cls) break;
        i += len;
    }
    return i;
}

/* ---- pre-tokenizer regexes ------------------------------------------------ */
/* Each matcher returns the end (byte offset) of the regex match that starts at byte i of the
 * segment p[0..n). Every character class is covered by some alternative, so matches tile
 * the segment and the "Isolated" split behaviour reduces to cutting at match boundaries. */

/* 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ */
static int match_gpt2(const uint8_t *p, int n, int i) {
    if (p[i] == '\'' && i + 1 < n) {
        uint8_t a = p[i + 1];
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if (i + 2 < n) {
            uint8_t b = p[i + 2];
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
        }
    }
    int len, cls = class_at(p, n, i, &len, NULL);
    if (p[i] == ' ' && i + 1 < n) {
        int l2, c2 = class_at(p, n, i + 1, &l2, NULL);
        if (c2 != UC_S) return run_end(p, n, i + 1 + l2, c2); /* " ?X+" with the space */
    }
    if (cls != UC_S) return run_end(p, n, i + len, cls);
    /* whitespace: \s+(?!\S) leaves the last space for the next word, unless the run is a
     * single character (then \s+ takes it) or reaches the end */
    int j = i + len, last = i;
    while (j < n) {
        int l2;
        if (class_at(p, n, j, &l2, NULL) != UC_S) break;
        last = j;
        j += l2;
    }
    if (j == n || last == i) return j;
    return last;
}

static inline int is_nl(uint8_t c) { return c == '\r' || c == '\n'; }
static inline uint32_t ascii_lower(uint32_t c) { return c - 'A' < 26u ? c + 32 : c; }

/* (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,max_digits}
 * | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * (qwen2: max_digits 1; llama3: 3) */
static int match_qwen2(const uint8_t *p, int n, int i, int max_digits) {
    int len;
    uint32_t c;
    int cls = class_at(p, n, i, &len, &c);
    if (c == '\'' && i + 1 < n) {
        /* Oniguruma's case folding: besides ASCII, only U+017F (long s) folds to 's' */
        int la;
        uint32_t a;
        class_at(p, n, i + 1, &la, &a);
        a = a == 0x17F ? 's' : ascii_lower(a);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 1 + la;
        if (i + 2 < n && la == 1) {
            uint32_t b = ascii_lower(p[i + 2]);
            if (((a == 'r' || a == 'v') && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
        }
    }
    if (cls == UC_L) return run_end(p, n, i + len, UC_L);
    if (i + len < n) {
        int l2, c2 = class_at(p, n, i + len, &l2, NULL);
        if (cls != UC_N && !is_nl(p[i]) && c2 == UC_L) return run_end(p, n, i + len + l2, UC_L);
    }
    if (cls == UC_N) {
        int j = i + len, k = 1;
        while (j < n && k < max_digits) {
            int l2;
            if (class_at(p, n, j, &l2, NULL) != UC_N) break;
            j += l2;
            k++;
        }
        return j;
    }
    int start = -1;
    if (cls == UC_O) start = i + len;
    else if (c == ' ' && i + 1 < n) {
        int l2;
        if (class_at(p, n, i + 1, &l2, NULL) == UC_O) start = i + 1 + l2;
    }
    if (start >= 0) {
        int j = run_end(p, n, start, UC_O);
        while (j < n && is_nl(p[j])) j++;
        return j;
    }
    /* whitespace run [i, j): \s*[\r\n]+ ends after its last CR/LF; otherwise as in gpt2 */
    int j = i + len, last = i, nl_end = is_nl(p[i]) ? i + 1 : -1;
    while (j < n) {
        int l2;
        if (class_at(p, n, j, &l2, NULL) != UC_S) break;
        last = j;
        j += l2;
        if (is_nl(p[last])) nl_end = j;
    }
    if (nl_end >= 0) return nl_end;
    if (j == n || last == i) return j;
    return last;
}

/* ---- per-call state ------------------------------------------------------- */
typedef struct {
    int32_t id, prev, next; /* id -1: merged into its left neighbour */
} bpe_sym;

typedef struct {
    int32_t rank, pos, id;
} bpe_merge;

typedef struct {
    uint32_t hash;
    int32_t key_off, key_len; /* piece bytes in cache_keys; key_len 0 = empty slot */
    int32_t tok_off, n_tok;   /* its tokens in cache_toks */
} cache_slot;

typedef struct {
    const borg_tokenizer *t;
    int32_t *out;
    int max_out;
    int n; /* tokens produced, may exceed max_out */
    int oom;
    bpe_sym *sym;
    bpe_merge *heap;
    int32_t *res;
    int cap_sym;
    uint8_t *norm; /* SentencePiece normalized segment */
    size_t cap_norm;
    /* piece -> tokens cache, used on long texts */
    cache_slot *cache;
    uint32_t cache_mask, cache_used;
    int32_t *cache_toks;
    int cache_ntok, cache_cap;
    uint8_t *cache_keys;
    int cache_nkey, cache_kcap;
} tok_run;

static inline void emit(tok_run *r, int32_t id) {
    if (r->n < r->max_out) r->out[r->n] = id;
    if (r->n < INT_MAX) r->n++;
}

static int reserve_sym(tok_run *r, int n) {
    if (n <= r->cap_sym) return 0;
    int cap = r->cap_sym ? r->cap_sym : 64;
    while (cap < n) cap = cap > INT_MAX / 6 ? n : cap * 2;
    bpe_sym *s = (bpe_sym *)realloc(r->sym, sizeof(bpe_sym) * (size_t)cap);
    if (s) r->sym = s;
    bpe_merge *h = (bpe_merge *)realloc(r->heap, sizeof(bpe_merge) * (size_t)cap * 3);
    if (h) r->heap = h;
    int32_t *o = (int32_t *)realloc(r->res, sizeof(int32_t) * (size_t)cap);
    if (o) r->res = o;
    if (!s || !h || !o) { r->oom = 1; return -1; }
    r->cap_sym = cap;
    return 0;
}

/* ---- BPE ------------------------------------------------------------------ */
/* Min-heap on (rank, pos): the lowest rank first, the leftmost position on ties. */
static inline int merge_less(const bpe_merge *a, const bpe_merge *b) {
    return a->rank != b->rank ? a->rank < b->rank : a->pos < b->pos;
}

static void heap_push(bpe_merge *h, int *n, bpe_merge m) {
    int i = (*n)++;
    while (i > 0) {
        int parent = (i - 1) >> 1;
        if (!merge_less(&m, &h[parent])) break;
        h[i] = h[parent];
        i = parent;
    }
    h[i] = m;
}

static bpe_merge heap_pop(bpe_merge *h, int *n) {
    bpe_merge top = h[0], last = h[--(*n)];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= *n) break;
        if (c + 1 < *n && merge_less(&h[c + 1], &h[c])) c++;
        if (!merge_less(&h[c], &last)) break;
        h[i] = h[c];
        i = c;
    }
    h[i] = last;
    return top;
}

/* Merge r->sym[0..ns) and write the resulting ids to r->res. Returns their count.
 * Mirrors tokenizers' Word::merge_all: stale queue entries are recognised by the pair at
 * their position no longer producing the same merged token. */
static int bpe_merge_syms(tok_run *r, int ns) {
    const borg_tok_impl *im = r->t->impl;
    bpe_sym *s = r->sym;
    if (ns <= 1) {
        if (ns == 1) r->res[0] = s[0].id;
        return ns;
    }
    bpe_merge *heap = r->heap;
    int nh = 0;
    for (int i = 0; i < ns; i++) {
        s[i].prev = i - 1;
        s[i].next = i + 1 < ns ? i + 1 : -1;
    }
    for (int i = 0; i + 1 < ns; i++) {
        const merge_slot *m = merge_find(im, s[i].id, s[i + 1].id);
        if (m) heap_push(heap, &nh, (bpe_merge){m->rank, i, m->id});
    }
    while (nh > 0) {
        bpe_merge top = heap_pop(heap, &nh);
        bpe_sym *a = &s[top.pos];
        if (a->id < 0 || a->next < 0) continue;
        bpe_sym *b = &s[a->next];
        const merge_slot *m = merge_find(im, a->id, b->id);
        if (!m || m->id != top.id) continue;
        a->id = top.id;
        b->id = -1;
        a->next = b->next;
        if (a->next >= 0) s[a->next].prev = top.pos;
        if (a->prev >= 0) {
            m = merge_find(im, s[a->prev].id, a->id);
            if (m) heap_push(heap, &nh, (bpe_merge){m->rank, a->prev, m->id});
        }
        if (a->next >= 0) {
            m = merge_find(im, a->id, s[a->next].id);
            if (m) heap_push(heap, &nh, (bpe_merge){m->rank, top.pos, m->id});
        }
    }
    int k = 0;
    for (int i = 0; i >= 0; i = s[i].next) r->res[k++] = s[i].id; /* s[0] is never merged away */
    return k;
}

/* Symbol building shared by both vocabularies, following tokenizers' BPE::merge_word:
 * a character missing from the vocabulary becomes byte tokens (byte fallback) or the
 * unknown token, consecutive unknowns fused into one; without either it is dropped. */
typedef struct {
    int ns;
    int unk; /* an unknown token is pending */
} sym_builder;

static inline void sb_known(tok_run *r, sym_builder *b, int32_t id) {
    if (b->unk) { r->sym[b->ns++].id = r->t->unk_id; b->unk = 0; }
    r->sym[b->ns++].id = id;
}

static inline void sb_unknown(tok_run *r, sym_builder *b, const uint8_t *p, int len) {
    const borg_tokenizer *t = r->t;
    if (t->byte_fallback) {
        int ok = 1;
        for (int k = 0; k < len; k++)
            if (t->impl->byte_tok[p[k]] < 0) ok = 0;
        if (ok) { /* HF adds these without flushing a pending unknown */
            for (int k = 0; k < len; k++) r->sym[b->ns++].id = t->impl->byte_tok[p[k]];
            return;
        }
    }
    if (t->unk_id >= 0) b->unk = 1;
}

static inline void sb_finish(tok_run *r, sym_builder *b) {
    if (b->unk) { r->sym[b->ns++].id = r->t->unk_id; b->unk = 0; }
}

static int grow(void **p, int *cap, int need, size_t elem) {
    if (need <= *cap) return 0;
    int c = *cap ? *cap : 4096;
    while (c < need) c *= 2;
    void *q = realloc(*p, elem * (size_t)c);
    if (!q) return -1;
    *p = q;
    *cap = c;
    return 0;
}

/* Tokenize one piece (p[0..n), n > 0) and emit its tokens. Byte-level pieces start with one
 * symbol per byte, SentencePiece pieces with one per character. whole_ok: with ignore_merges,
 * a piece that is itself a vocabulary entry is emitted as is. */
static void bpe_piece(tok_run *r, const uint8_t *p, int n, int whole_ok) {
    const borg_tokenizer *t = r->t;
    cache_slot *slot = NULL;
    uint32_t h = 0;
    if (r->cache && n <= 64) {
        h = hash_bytes(p, n);
        /* Bounded probing: FNV-1a is easy to collide on purpose, and an unbounded cluster
         * of crafted words would make every lookup linear in the cache size. A piece that
         * finds no slot within the bound is simply not cached (it is never inserted further
         * out, so lookups need not look further either). */
        uint32_t i = h & r->cache_mask;
        for (int probe = 0; probe < 32; probe++, i = (i + 1) & r->cache_mask) {
            cache_slot *c = &r->cache[i];
            if (!c->key_len) { slot = c; break; }
            if (c->hash == h && c->key_len == n && !memcmp(r->cache_keys + c->key_off, p, (size_t)n)) {
                for (int k = 0; k < c->n_tok; k++) emit(r, r->cache_toks[c->tok_off + k]);
                return;
            }
        }
    }
    if (n > INT_MAX / 8) { r->oom = 1; return; } /* keeps the scratch sizes in range */
    int k;
    int32_t whole = whole_ok && t->ignore_merges ? vocab_find(t, p, n) : -1;
    if (whole >= 0) {
        if (reserve_sym(r, 1)) return;
        r->res[0] = whole;
        k = 1;
    } else {
        if (reserve_sym(r, n + 1)) return;
        sym_builder b = {0, 0};
        if (t->byte_level) {
            for (int i = 0; i < n; i++) {
                int32_t id = t->impl->byte_tok[p[i]];
                if (id >= 0) sb_known(r, &b, id);
                else sb_unknown(r, &b, p + i, 1);
            }
        } else {
            for (int i = 0; i < n;) {
                uint32_t c;
                int len = utf8_next(p + i, n - i, &c);
                int32_t id = vocab_find(t, p + i, len);
                if (id >= 0) sb_known(r, &b, id);
                else sb_unknown(r, &b, p + i, len);
                i += len;
            }
        }
        sb_finish(r, &b);
        k = bpe_merge_syms(r, b.ns);
    }
    for (int i = 0; i < k; i++) emit(r, r->res[i]);
    /* remember the piece while the cache is at most half full (it only saves work, so a
     * failed allocation just skips this) */
    if (slot && r->cache_used * 2 < r->cache_mask && k <= 0xFFFF && r->cache_ntok < INT_MAX / 2 &&
        !grow((void **)&r->cache_toks, &r->cache_cap, r->cache_ntok + k, sizeof(int32_t)) &&
        !grow((void **)&r->cache_keys, &r->cache_kcap, r->cache_nkey + n, 1)) {
        /* k can be 0 (every byte dropped) while cache_toks is still NULL: memcpy(NULL, .., 0) is UB */
        if (k) memcpy(r->cache_toks + r->cache_ntok, r->res, sizeof(int32_t) * (size_t)k);
        memcpy(r->cache_keys + r->cache_nkey, p, (size_t)n);
        slot->hash = h;
        slot->key_off = r->cache_nkey;
        slot->key_len = n;
        slot->tok_off = r->cache_ntok;
        slot->n_tok = k;
        r->cache_ntok += k;
        r->cache_nkey += n;
        r->cache_used++;
    }
}

/* Pre-tokenize a byte-level segment and tokenize every piece. */
static void segment_byte_level(tok_run *r, const uint8_t *p, int n) {
    int pretok = r->t->pretok;
    if (pretok == PRETOK_NONE) {
        if (n > 0) bpe_piece(r, p, n, 1);
        return;
    }
    if (pretok == PRETOK_SMOLLM) {
        /* Digits(individual_digits): every numeric character is a piece of its own; the
         * gpt2 regex then runs on each stretch between them separately */
        int i = 0;
        while (i < n && !r->oom) {
            int len;
            uint32_t c;
            len = utf8_next(p + i, n - i, &c);
            if (c != CP_INVALID && uc_numeric(c)) {
                bpe_piece(r, p + i, len, 1);
                i += len;
                continue;
            }
            int e = i + len;
            while (e < n) {
                int l2 = utf8_next(p + e, n - e, &c);
                if (c != CP_INVALID && uc_numeric(c)) break;
                e += l2;
            }
            while (i < e && !r->oom) {
                int j = match_gpt2(p, e, i);
                bpe_piece(r, p + i, j - i, 1);
                i = j;
            }
        }
        return;
    }
    int i = 0;
    while (i < n && !r->oom) {
        int j = pretok == PRETOK_GPT2 ? match_gpt2(p, n, i) : match_qwen2(p, n, i, pretok == PRETOK_LLAMA3 ? 3 : 1);
        bpe_piece(r, p + i, j - i, 1);
        i = j;
    }
}

static inline int is_meta(const uint8_t *q) { return q[0] == 0xE2 && q[1] == 0x96 && q[2] == 0x81; }

/* SentencePiece-style segment: optional "▁" prefix, spaces become "▁", then BPE over the
 * characters of the whole segment (there is no pre-tokenizer). */
static void segment_spm(tok_run *r, const uint8_t *p, int n) {
    const borg_tokenizer *t = r->t;
    if (n == 0) return; /* HF's Prepend normalizer leaves empty segments empty */
    if ((size_t)n > ((size_t)INT_MAX - 3) / 3) { r->oom = 1; return; }
    size_t need = (size_t)n * 3 + 3;
    if (need > r->cap_norm) {
        uint8_t *nb = (uint8_t *)realloc(r->norm, need);
        if (!nb) { r->oom = 1; return; }
        r->norm = nb;
        r->cap_norm = need;
    }
    uint8_t *q = r->norm;
    int m = 0;
    if (t->spm_add_prefix_space) { q[m++] = 0xE2; q[m++] = 0x96; q[m++] = 0x81; }
    for (int i = 0; i < n; i++) {
        if (p[i] == ' ') { q[m++] = 0xE2; q[m++] = 0x96; q[m++] = 0x81; }
        else q[m++] = p[i];
    }
    if (t->ignore_merges) {
        int32_t whole = vocab_find(t, q, m);
        if (whole >= 0) { emit(r, whole); return; }
    }
    if (!t->impl->spm_split) {
        bpe_piece(r, q, m, 0);
        return;
    }
    /* No token holds "▁" after another character, so no merge can join the text before a
     * "▁" that follows a non-"▁" character with the text after it: BPE on the pieces cut
     * there gives the same tokens as on the whole segment, at a fraction of the cost. */
    int start = 0;
    for (int i = 3; i + 3 <= m; i++) {
        if (is_meta(q + i) && !is_meta(q + i - 3)) {
            bpe_piece(r, q + start, i - start, 0);
            start = i;
            i += 2;
        }
        if (r->oom) return;
    }
    bpe_piece(r, q + start, m - start, 0);
}

static void tokenize_segment(tok_run *r, const uint8_t *p, int n) {
    if (r->t->byte_level) segment_byte_level(r, p, n);
    else segment_spm(r, p, n);
}

/* Longest added token starting at p[0..n); returns its length (0 if none) and id. */
static int match_added(const borg_tok_impl *im, const uint8_t *p, int n, int32_t *id) {
    int best = 0;
    int32_t node = im->root[p[0]];
    for (int d = 1; node >= 0; d++) {
        if (im->trie[node].tok >= 0) { best = d; *id = im->trie[node].tok; }
        if (d >= n) break;
        node = im->trie[node].child;
        while (node >= 0 && im->trie[node].byte != p[d]) node = im->trie[node].sibling;
    }
    return best;
}

BORG_API int borg_tokenize(const borg_model *model, const char *text, int text_len, int add_bos,
                           int parse_special, int32_t *out, int max_out) {
    if (!model || !model->tok.impl) return 0;
    const borg_tokenizer *t = &model->tok;
    if (!text) { /* treated as empty; "" also keeps the segment pointers off NULL (NULL + 0 is UB) */
        text = "";
        text_len = 0;
    } else if (text_len < 0) {
        size_t l = strlen(text);
        text_len = l > (size_t)INT_MAX ? INT_MAX : (int)l;
    }
    const uint8_t *s = (const uint8_t *)text;
    tok_run r;
    memset(&r, 0, sizeof r);
    r.t = t;
    r.out = out;
    r.max_out = out && max_out > 0 ? max_out : 0;
    if (text_len > 4096) {
        /* repeated words are common in long texts; the cache only saves work */
        uint32_t slots = pow2_at_least((uint64_t)text_len / 4);
        if (slots > (1u << 18)) slots = 1u << 18;
        r.cache = (cache_slot *)calloc(slots, sizeof(cache_slot));
        r.cache_mask = slots - 1;
    }

    if (add_bos < 0) add_bos = t->add_bos;
    if (add_bos && t->bos_id >= 0) emit(&r, t->bos_id);
    if (parse_special && t->impl->n_trie) {
        int seg = 0;
        for (int i = 0; i < text_len && !r.oom;) {
            int32_t id = -1;
            int len = t->impl->root[s[i]] >= 0 ? match_added(t->impl, s + i, text_len - i, &id) : 0;
            if (!len) { i++; continue; }
            tokenize_segment(&r, s + seg, i - seg);
            emit(&r, id);
            i += len;
            seg = i;
        }
        if (!r.oom) tokenize_segment(&r, s + seg, text_len - seg);
    } else {
        tokenize_segment(&r, s, text_len);
    }

    free(r.sym);
    free(r.heap);
    free(r.res);
    free(r.norm);
    free(r.cache);
    free(r.cache_toks);
    free(r.cache_keys);
    if (r.oom) return 0; /* out of memory: no tokens rather than a bogus size request */
    return r.n <= r.max_out ? r.n : -r.n;
}

/* ---- decoding ------------------------------------------------------------- */
typedef struct {
    char *buf;
    int buflen, pos;
    int strip; /* drop the next byte if it is a space */
} dec_out;

static inline void put(dec_out *o, char c) {
    if (o->strip) {
        o->strip = 0;
        if (c == ' ') return;
    }
    if (o->pos < o->buflen) o->buf[o->pos] = c;
    if (o->pos < INT_MAX) o->pos++;
}

/* Bytes of one token as they appear in text: SentencePiece pieces turn "▁" into a space
 * (byte tokens are left alone, as HF's decoder replaces "▁" before byte fallback). */
static void token_text(const borg_tokenizer *t, int32_t tok, dec_out *o) {
    int len;
    const uint8_t *p = tok_str(t, tok, &len);
    if (t->byte_level || t->token_type[tok] == TT_BYTE) {
        for (int k = 0; k < len; k++) put(o, (char)p[k]);
        return;
    }
    for (int k = 0; k < len;) {
        if (k + 2 < len && p[k] == 0xE2 && p[k + 1] == 0x96 && p[k + 2] == 0x81) { put(o, ' '); k += 3; }
        else put(o, (char)p[k++]);
    }
}

BORG_API int borg_token_bytes(const borg_model *model, int32_t tok, char *buf, int buflen) {
    if (!model || tok < 0 || tok >= model->tok.n_vocab) return 0;
    dec_out o = {buf, buf ? buflen : 0, 0, 0};
    token_text(&model->tok, tok, &o);
    return o.pos;
}

BORG_API int borg_detokenize(const borg_model *model, const int32_t *toks, int n, int skip_special, char *buf,
                             int buflen) {
    if (!model || !toks) return 0;
    const borg_tokenizer *t = &model->tok;
    /* SentencePiece with a "▁" prefix: drop the one leading space it becomes (HF's Strip
     * decoder runs on the fused output, so this is the first byte of the whole text) */
    dec_out o = {buf, buf ? buflen : 0, 0, !t->byte_level && t->spm_add_prefix_space};
    for (int i = 0; i < n; i++) {
        int32_t tok = toks[i];
        if (tok < 0 || tok >= t->n_vocab) continue;
        if (skip_special && t->token_type[tok] == TT_SPECIAL) continue;
        token_text(t, tok, &o);
    }
    return o.pos;
}

BORG_API int borg_token_type(const borg_model *model, int32_t tok) {
    if (!model || tok < 0 || tok >= model->tok.n_vocab) return TT_UNUSED;
    return model->tok.token_type[tok];
}

BORG_API int borg_is_eog(const borg_model *model, int32_t tok) {
    if (!model || tok < 0) return 0;
    const borg_tokenizer *t = &model->tok;
    if (tok == t->eos_id) return 1;
    for (int i = 0; i < t->n_eos_ids; i++)
        if (t->eos_ids[i] == tok) return 1;
    return 0;
}
