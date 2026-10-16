/* JSON-constrained decoding.
 *
 * A byte-level pushdown recognizer for JSON (RFC 8259) tracks the text generated so far.
 * Before each sampling step, candidate tokens are tested by feeding their bytes to a copy
 * of the recognizer; tokens that would make the text invalid get logit -inf. Only the
 * highest-scoring candidates are tested (most steps need a handful of checks), falling
 * back to the whole vocabulary when none of them fits. Once the top-level value is
 * complete, only end-of-generation tokens remain allowed.
 */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum {
    J_VALUE = 0,  /* expecting a value */
    J_OBJ_KEY,    /* after '{' or ',' in an object: expecting '"' (or '}' right after '{') */
    J_COLON,      /* after a key */
    J_AFTER,      /* after a value: ',' or closing bracket, or end at top level */
    J_STRING,     /* inside a string */
    J_ESCAPE,     /* after a backslash */
    J_UNICODE,    /* inside \uXXXX (count in aux) */
    J_NUMBER,     /* inside a number (sub-state in aux) */
    J_LITERAL,    /* inside true/false/null (literal index and position in aux) */
    J_DONE,       /* top-level value complete */
    J_ERROR
};

/* number sub-states */
enum { N_SIGN, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP };

static const char *LITERALS[3] = {"true", "false", "null"};

void json_state_init(json_state *s) {
    memset(s, 0, sizeof *s);
    s->mode = J_VALUE;
}

static int top(const json_state *s) { return s->depth ? s->stack[s->depth - 1] : 0; }

static int is_ws(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

/* after a complete value */
static void value_done(json_state *s) { s->mode = s->depth ? J_AFTER : J_DONE; }

static int start_value(json_state *s, unsigned char c) {
    if (c == '"') { s->mode = J_STRING; s->in_key = 0; return 1; }
    if (c == '{' || c == '[') {
        if (s->depth >= JSON_MAX_DEPTH) return 0;
        s->stack[s->depth++] = c;
        s->mode = c == '{' ? J_OBJ_KEY : J_VALUE;
        s->just_opened = 1;
        return 1;
    }
    if (c == '-') { s->mode = J_NUMBER; s->aux = N_SIGN; return 1; }
    if (c == '0') { s->mode = J_NUMBER; s->aux = N_ZERO; return 1; }
    if (c >= '1' && c <= '9') { s->mode = J_NUMBER; s->aux = N_INT; return 1; }
    for (int i = 0; i < 3; i++)
        if (c == (unsigned char)LITERALS[i][0]) { s->mode = J_LITERAL; s->aux = i * 16 + 1; return 1; }
    return 0;
}

static int number_can_end(int sub) { return sub == N_ZERO || sub == N_INT || sub == N_FRAC || sub == N_EXP; }

/* Feed one byte; returns 0 if the text becomes invalid. */
int json_state_feed(json_state *s, unsigned char c) {
    for (;;) { /* loop only when a number ends and the byte must be re-read */
        switch (s->mode) {
        case J_DONE:
            return is_ws(c);
        case J_VALUE:
            if (is_ws(c)) return 1;
            if (c == ']' && s->just_opened && top(s) == '[') {
                s->depth--;
                s->just_opened = 0;
                value_done(s);
                return 1;
            }
            s->just_opened = 0;
            return start_value(s, c);
        case J_OBJ_KEY:
            if (is_ws(c)) return 1;
            if (c == '}' && s->just_opened) {
                s->depth--;
                s->just_opened = 0;
                value_done(s);
                return 1;
            }
            s->just_opened = 0;
            if (c != '"') return 0;
            s->mode = J_STRING;
            s->in_key = 1;
            return 1;
        case J_COLON:
            if (is_ws(c)) return 1;
            if (c != ':') return 0;
            s->mode = J_VALUE;
            return 1;
        case J_AFTER:
            if (is_ws(c)) return 1;
            if (c == ',') {
                s->mode = top(s) == '{' ? J_OBJ_KEY : J_VALUE;
                return 1;
            }
            if ((c == '}' && top(s) == '{') || (c == ']' && top(s) == '[')) {
                s->depth--;
                value_done(s);
                return 1;
            }
            return 0;
        case J_STRING:
            /* RFC 8259 text is UTF-8: validate multi-byte sequences (no overlongs, surrogates
             * or code points above U+10FFFF), since byte-level tokens can split characters */
            if (s->u8_need) {
                if (c < s->u8_lo || c > s->u8_hi) return 0;
                s->u8_need--;
                s->u8_lo = 0x80;
                s->u8_hi = 0xBF;
                return 1;
            }
            if (c >= 0x80) {
                s->u8_lo = 0x80;
                s->u8_hi = 0xBF;
                if (c >= 0xC2 && c <= 0xDF) s->u8_need = 1;
                else if (c >= 0xE0 && c <= 0xEF) {
                    s->u8_need = 2;
                    if (c == 0xE0) s->u8_lo = 0xA0;
                    if (c == 0xED) s->u8_hi = 0x9F;
                } else if (c >= 0xF0 && c <= 0xF4) {
                    s->u8_need = 3;
                    if (c == 0xF0) s->u8_lo = 0x90;
                    if (c == 0xF4) s->u8_hi = 0x8F;
                } else {
                    return 0;
                }
                return 1;
            }
            if (c == '"') {
                if (s->in_key) s->mode = J_COLON;
                else value_done(s);
                return 1;
            }
            if (c == '\\') { s->mode = J_ESCAPE; return 1; }
            return c >= 0x20; /* control characters must be escaped */
        case J_ESCAPE:
            if (c == 'u') { s->mode = J_UNICODE; s->aux = 0; return 1; }
            if (strchr("\"\\/bfnrt", c) && c) { s->mode = J_STRING; return 1; }
            return 0;
        case J_UNICODE:
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return 0;
            if (++s->aux == 4) s->mode = J_STRING;
            return 1;
        case J_LITERAL: {
            const char *lit = LITERALS[s->aux / 16];
            const int pos = s->aux % 16;
            if (c != (unsigned char)lit[pos]) return 0;
            s->aux++;
            if (!lit[pos + 1]) value_done(s);
            return 1;
        }
        case J_NUMBER: {
            const int sub = s->aux;
            const int digit = c >= '0' && c <= '9';
            switch (sub) {
            case N_SIGN:
                if (c == '0') { s->aux = N_ZERO; return 1; }
                if (digit) { s->aux = N_INT; return 1; }
                return 0;
            case N_ZERO:
            case N_INT:
                if (digit && sub == N_INT) return 1;
                if (c == '.') { s->aux = N_DOT; return 1; }
                if (c == 'e' || c == 'E') { s->aux = N_E; return 1; }
                break;
            case N_DOT:
                if (digit) { s->aux = N_FRAC; return 1; }
                return 0;
            case N_FRAC:
                if (digit) return 1;
                if (c == 'e' || c == 'E') { s->aux = N_E; return 1; }
                break;
            case N_E:
                if (c == '+' || c == '-') { s->aux = N_ESIGN; return 1; }
                if (digit) { s->aux = N_EXP; return 1; }
                return 0;
            case N_ESIGN:
                if (digit) { s->aux = N_EXP; return 1; }
                return 0;
            case N_EXP:
                if (digit) return 1;
                break;
            }
            if (!number_can_end(sub)) return 0;
            value_done(s);
            if (s->mode == J_DONE) return is_ws(c); /* a bare top-level number ends at whitespace */
            continue; /* re-read the byte after the number */
        }
        default:
            return 0;
        }
    }
}

/* Can the text end here? (a top-level number may still be open) */
static int json_state_complete(const json_state *s) {
    return s->mode == J_DONE || (s->mode == J_NUMBER && s->depth == 0 && number_can_end(s->aux));
}

static int token_fits(const borg_model *m, const json_state *s, int32_t tok) {
    const borg_tokenizer *t = &m->tok;
    if (tok < 0 || tok >= t->n_vocab) return 0;
    if (borg_is_eog(m, tok)) return json_state_complete(s);
    if (t->token_type[tok] == TT_SPECIAL || t->token_type[tok] == TT_UNUSED) return 0;
    if (s->mode == J_DONE) return 0; /* nothing but end-of-generation after the value */
    char buf[256];
    int n = borg_token_bytes(m, tok, buf, (int)sizeof buf);
    if (n <= 0 || n > (int)sizeof buf) return 0;
    json_state tmp = *s;
    for (int i = 0; i < n; i++)
        if (!json_state_feed(&tmp, (unsigned char)buf[i])) return 0;
    return 1;
}

/* Keep only candidates that fit: the best `want` valid tokens among the 512 highest logits,
 * or every valid token when none of those fits. All other logits become -inf. */
void json_constrain(const borg_model *m, const json_state *s, float *logits, int n_vocab, int32_t *idx, float *val,
                    uint8_t *keep, int want) {
    memset(keep, 0, (size_t)n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        idx[i] = i;
        val[i] = logits[i];
    }
    const int k = n_vocab < 512 ? n_vocab : 512;
    sampler_select_top(val, idx, n_vocab, k);
    sampler_sort_desc(val, idx, k);
    int found = 0;
    for (int i = 0; i < k && found < want; i++)
        if (isfinite(val[i]) && token_fits(m, s, idx[i])) {
            keep[idx[i]] = 1;
            found++;
        }
    if (!found)
        for (int i = 0; i < n_vocab; i++)
            if (token_fits(m, s, i)) {
                keep[i] = 1;
                found++;
            }
    if (!found) /* nothing can continue the text (e.g. the recognizer failed): let it end */
        for (int i = 0; i < n_vocab; i++)
            if (borg_is_eog(m, i)) keep[i] = 1;
    for (int i = 0; i < n_vocab; i++)
        if (!keep[i]) logits[i] = -INFINITY;
}

/* Advance the recognizer over an emitted token. */
void json_state_feed_token(const borg_model *m, json_state *s, int32_t tok) {
    if (borg_is_eog(m, tok)) return; /* end-of-generation markers are not part of the JSON text */
    char buf[256];
    int n = borg_token_bytes(m, tok, buf, (int)sizeof buf);
    if (n > (int)sizeof buf) n = (int)sizeof buf;
    for (int i = 0; i < n; i++)
        if (!json_state_feed(s, (unsigned char)buf[i])) {
            s->mode = J_ERROR;
            return;
        }
}
