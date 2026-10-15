/* Minimal JSON parser for the .borg header. Strings are decoded to UTF-8. */
#include "internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p, *end;
    char *err;
    int errlen;
    int depth;
} jparser;

static int jfail(jparser *ps, const char *msg) {
    if (ps->err && !ps->err[0]) borg_set_err(ps->err, ps->errlen, "json: %s", msg);
    return -1;
}

static void skip_ws(jparser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static int put_utf8(char *out, uint32_t cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

static int hex4(const char *s, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

static int parse_string(jparser *ps, char **out, int *outlen) {
    if (ps->p >= ps->end || *ps->p != '"') return jfail(ps, "expected string");
    ps->p++;
    const char *start = ps->p;
    size_t cap = 0;
    while (ps->p < ps->end && *ps->p != '"') {
        if (*ps->p == '\\') ps->p++;
        ps->p++;
        cap++;
    }
    if (ps->p >= ps->end) return jfail(ps, "unterminated string");
    char *s = (char *)malloc(cap * 4 + 1);
    if (!s) return jfail(ps, "out of memory");
    int n = 0;
    const char *q = start;
    while (q < ps->p) {
        char c = *q++;
        if (c != '\\') { s[n++] = c; continue; }
        c = *q++;
        switch (c) {
        case '"': s[n++] = '"'; break;
        case '\\': s[n++] = '\\'; break;
        case '/': s[n++] = '/'; break;
        case 'b': s[n++] = '\b'; break;
        case 'f': s[n++] = '\f'; break;
        case 'n': s[n++] = '\n'; break;
        case 'r': s[n++] = '\r'; break;
        case 't': s[n++] = '\t'; break;
        case 'u': {
            uint32_t cp;
            if (ps->p - q < 4 || hex4(q, &cp)) { free(s); return jfail(ps, "bad \\u escape"); }
            q += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && ps->p - q >= 6 && q[0] == '\\' && q[1] == 'u') {
                uint32_t lo;
                if (!hex4(q + 2, &lo) && lo >= 0xDC00 && lo < 0xE000) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    q += 6;
                }
            }
            n += put_utf8(s + n, cp);
            break;
        }
        default: free(s); return jfail(ps, "bad escape");
        }
    }
    s[n] = 0;
    ps->p++; /* closing quote */
    *out = s;
    *outlen = n;
    return 0;
}

static int parse_value(jparser *ps, json_value *v);

static int push_item(json_value *v, int *cap) {
    if (v->len == *cap) {
        int nc = *cap ? *cap * 2 : 8;
        json_value *ni = (json_value *)realloc(v->items, (size_t)nc * sizeof(json_value));
        if (!ni) return -1;
        v->items = ni;
        if (v->type == JS_OBJ) {
            char **nk = (char **)realloc(v->keys, (size_t)nc * sizeof(char *));
            if (!nk) return -1;
            v->keys = nk;
        }
        *cap = nc;
    }
    memset(&v->items[v->len], 0, sizeof(json_value));
    return 0;
}

static int parse_value(jparser *ps, json_value *v) {
    memset(v, 0, sizeof(*v));
    if (++ps->depth > 64) return jfail(ps, "nesting too deep");
    skip_ws(ps);
    if (ps->p >= ps->end) return jfail(ps, "unexpected end");
    char c = *ps->p;
    int rc = 0;
    if (c == '{' || c == '[') {
        int obj = c == '{';
        v->type = obj ? JS_OBJ : JS_ARR;
        ps->p++;
        int cap = 0;
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == (obj ? '}' : ']')) { ps->p++; ps->depth--; return 0; }
        for (;;) {
            if (push_item(v, &cap)) return jfail(ps, "out of memory");
            if (obj) {
                skip_ws(ps);
                char *key; int klen;
                if (parse_string(ps, &key, &klen)) return -1;
                v->keys[v->len] = key;
                skip_ws(ps);
                if (ps->p >= ps->end || *ps->p != ':') { v->len++; return jfail(ps, "expected ':'"); }
                ps->p++;
            }
            rc = parse_value(ps, &v->items[v->len]);
            v->len++;
            if (rc) return rc;
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == (obj ? '}' : ']')) { ps->p++; break; }
            return jfail(ps, "expected ',' or closing bracket");
        }
    } else if (c == '"') {
        v->type = JS_STR;
        rc = parse_string(ps, &v->str, &v->len);
    } else if (c == 't' && ps->end - ps->p >= 4 && !memcmp(ps->p, "true", 4)) {
        v->type = JS_BOOL; v->num = 1; ps->p += 4;
    } else if (c == 'f' && ps->end - ps->p >= 5 && !memcmp(ps->p, "false", 5)) {
        v->type = JS_BOOL; v->num = 0; ps->p += 5;
    } else if (c == 'n' && ps->end - ps->p >= 4 && !memcmp(ps->p, "null", 4)) {
        v->type = JS_NULL; ps->p += 4;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        char buf[64];
        int n = 0;
        while (ps->p < ps->end && n < 63 && strchr("+-0123456789.eE", *ps->p)) buf[n++] = *ps->p++;
        buf[n] = 0;
        char *endp;
        v->type = JS_NUM;
        v->num = strtod(buf, &endp);
        if (*endp) return jfail(ps, "bad number");
    } else {
        return jfail(ps, "unexpected character");
    }
    ps->depth--;
    return rc;
}

static void json_free_inner(json_value *v) {
    if (!v) return;
    free(v->str);
    if (v->type == JS_ARR || v->type == JS_OBJ) {
        for (int i = 0; i < v->len; i++) {
            json_free_inner(&v->items[i]);
            if (v->keys) free(v->keys[i]);
        }
        free(v->items);
        free(v->keys);
    }
}

void json_free(json_value *v) {
    json_free_inner(v);
    free(v);
}

json_value *json_parse(const char *text, size_t len, char *err, int errlen) {
    jparser ps = {text, text + len, err, errlen, 0};
    if (err && errlen > 0) err[0] = 0;
    json_value *v = (json_value *)calloc(1, sizeof(json_value));
    if (!v) return NULL;
    if (parse_value(&ps, v)) { json_free(v); return NULL; }
    skip_ws(&ps);
    if (ps.p != ps.end) { jfail(&ps, "trailing characters"); json_free(v); return NULL; }
    return v;
}

const json_value *json_get(const json_value *obj, const char *key) {
    if (!obj || obj->type != JS_OBJ) return NULL;
    for (int i = 0; i < obj->len; i++)
        if (!strcmp(obj->keys[i], key)) return &obj->items[i];
    return NULL;
}

double json_num(const json_value *v, double dflt) {
    if (!v || (v->type != JS_NUM && v->type != JS_BOOL)) return dflt;
    return v->num;
}

int json_bool(const json_value *v, int dflt) {
    if (!v || (v->type != JS_BOOL && v->type != JS_NUM)) return dflt;
    return v->num != 0;
}

const char *json_str(const json_value *v, const char *dflt) {
    if (!v || v->type != JS_STR) return dflt;
    return v->str;
}
