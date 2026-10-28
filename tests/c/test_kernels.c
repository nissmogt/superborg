/* Kernel tests: every SIMD dot product against a double-precision reference on the
 * dequantized operands, the interleaved-layout kernels against their scalar contract (bit for
 * bit, for every token count), the repack round trip, plus fp16 round trips. Run: make test-c */
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 88172645463325252ull;
static float frand(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (float)((double)(rng >> 11) / 9007199254740992.0 * 2.0 - 1.0);
}

static int failures = 0;
#define CHECK(cond, ...)                 \
    do {                                 \
        if (!(cond)) {                   \
            printf("FAIL: " __VA_ARGS__); \
            printf("\n");                \
            failures++;                  \
        }                                \
    } while (0)

static void quantize_q8_0(const float *x, block_q8_0 *b, int n) {
    for (int i = 0; i < n / QK; i++) {
        float amax = 0;
        for (int j = 0; j < QK; j++) amax = fmaxf(amax, fabsf(x[i * QK + j]));
        float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        b[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK; j++) b[i].qs[j] = (int8_t)lrintf(x[i * QK + j] * id);
    }
}

static void quantize_q4_0(const float *x, block_q4_0 *b, int n) {
    for (int i = 0; i < n / QK; i++) {
        float amax = 0, mx = 0;
        for (int j = 0; j < QK; j++)
            if (fabsf(x[i * QK + j]) > amax) { amax = fabsf(x[i * QK + j]); mx = x[i * QK + j]; }
        float d = mx / -8.0f, id = d ? 1.0f / d : 0.0f;
        b[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK / 2; j++) {
            int lo = (int)(x[i * QK + j] * id + 8.5f), hi = (int)(x[i * QK + j + 16] * id + 8.5f);
            lo = lo > 15 ? 15 : (lo < 0 ? 0 : lo);
            hi = hi > 15 ? 15 : (hi < 0 ? 0 : hi);
            b[i].qs[j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

static void quantize_q4_1(const float *x, block_q4_1 *b, int n) {
    for (int i = 0; i < n / QK; i++) {
        float mn = x[i * QK], mx = x[i * QK];
        for (int j = 1; j < QK; j++) { mn = fminf(mn, x[i * QK + j]); mx = fmaxf(mx, x[i * QK + j]); }
        float d = (mx - mn) / 15.0f, id = d ? 1.0f / d : 0.0f;
        b[i].d = fp32_to_fp16(d);
        b[i].m = fp32_to_fp16(mn);
        for (int j = 0; j < QK / 2; j++) {
            int lo = (int)((x[i * QK + j] - mn) * id + 0.5f), hi = (int)((x[i * QK + j + 16] - mn) * id + 0.5f);
            lo = lo > 15 ? 15 : (lo < 0 ? 0 : lo);
            hi = hi > 15 ? 15 : (hi < 0 ? 0 : hi);
            b[i].qs[j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

static void test_fp16(void) {
    for (int i = 0; i < 65536; i++) {
        uint16_t h = (uint16_t)i;
        float f = fp16_to_fp32(h);
        if (isnan(f)) continue;
        CHECK(fp32_to_fp16(f) == h, "fp16 round trip %04x", i);
    }
    CHECK(fp32_to_fp16(65520.0f) == 0x7C00, "fp16 overflow to inf");
    CHECK(fp16_to_fp32(0x0001) == 5.9604644775390625e-08f, "fp16 subnormal");
}

static void test_dot(borg_dtype dt, int cols) {
    const int rows = 7;
    float *w = malloc(sizeof(float) * rows * cols), *x = malloc(sizeof(float) * cols);
    float *wd = malloc(sizeof(float) * cols);
    for (int i = 0; i < rows * cols; i++) w[i] = frand() * (i % 97 == 0 ? 8.0f : 1.0f);
    for (int i = 0; i < cols; i++) x[i] = frand() * (i % 31 == 0 ? 5.0f : 1.0f);
    size_t rb = dtype_row_bytes(dt, cols);
    uint8_t *data = malloc(rb * rows);
    for (int r = 0; r < rows; r++) {
        uint8_t *row = data + rb * r;
        const float *src = w + (size_t)r * cols;
        if (dt == DT_F32) memcpy(row, src, sizeof(float) * cols);
        else if (dt == DT_F16) for (int i = 0; i < cols; i++) ((uint16_t *)row)[i] = fp32_to_fp16(src[i]);
        else if (dt == DT_BF16) for (int i = 0; i < cols; i++) { uint32_t u; memcpy(&u, &src[i], 4); ((uint16_t *)row)[i] = (uint16_t)(u >> 16); }
        else if (dt == DT_Q8_0) quantize_q8_0(src, (block_q8_0 *)row, cols);
        else if (dt == DT_Q4_0) quantize_q4_0(src, (block_q4_0 *)row, cols);
        else if (dt == DT_Q4_1) quantize_q4_1(src, (block_q4_1 *)row, cols);
    }
    borg_mat m = {dt, rows, cols, data, rb, MAT_PLAIN, NULL};
    block_q8a *xq = malloc(sizeof(block_q8a) * (cols / QK + 1));
    float *xdq = malloc(sizeof(float) * cols);
    if (cols % QK == 0) {
        quantize_row_q8a(x, xq, cols);
        for (int i = 0; i < cols / QK; i++) {
            int s = 0;
            for (int j = 0; j < QK; j++) { xdq[i * QK + j] = xq[i].d * xq[i].qs[j]; s += xq[i].qs[j]; }
            CHECK(fabsf(xq[i].s - xq[i].d * s) <= 1e-5f * fabsf(xq[i].d * s) + 1e-7f, "q8a block sum");
            CHECK(xq[i].sum == s, "q8a code sum");
        }
    }
    enum { NT = 6 }; /* exercises the 4-token kernels plus a remainder */
    static float out[7 * NT];
    static float xs[NT * 4096];
    for (int t = 0; t < NT; t++) memcpy(xs + t * cols, x, sizeof(float) * cols);
    static block_q8a xqs[NT * 4096 / QK];
    if (cols % QK == 0) for (int t = 0; t < NT; t++) memcpy(xqs + t * (cols / QK), xq, sizeof(block_q8a) * (cols / QK));
    mat_mul_rows(&m, 0, rows, NT, xs, cols, xqs, cols / QK, out, rows);
    /* batched results must equal token-by-token results bit for bit, with distinct tokens */
    static float xs2[NT * 4096], out2[7 * NT];
    static block_q8a xqs2[NT * 4096 / QK];
    for (int i = 0; i < NT * cols; i++) xs2[i] = frand();
    if (cols % QK == 0) for (int t = 0; t < NT; t++) quantize_row_q8a(xs2 + t * cols, xqs2 + t * (cols / QK), cols);
    mat_mul_rows(&m, 0, rows, NT, xs2, cols, xqs2, cols / QK, out2, rows);
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < rows; r++) {
            float one[7]; /* mat_mul_rows writes column r of the output row */
            mat_mul_rows(&m, r, r + 1, 1, xs2 + t * cols, cols, xqs2 + t * (cols / QK), cols / QK, one, rows);
            CHECK(one[r] == out2[t * rows + r], "%s cols=%d batched != single (token %d row %d)", dtype_name(dt), cols, t, r);
        }
    for (int r = 0; r < rows; r++) {
        mat_get_row(&m, r, wd);
        const float *xv = mat_wants_q8a(&m) ? xdq : x;
        double ref = 0, mag = 0;
        for (int i = 0; i < cols; i++) { ref += (double)wd[i] * xv[i]; mag += fabs((double)wd[i] * xv[i]); }
        float got = mat_dot_row(&m, r, x, xq);
        CHECK(fabs(got - ref) <= 1e-5 * mag + 1e-6, "%s cols=%d row %d: got %.7g want %.7g", dtype_name(dt), cols, r, got, ref);
        for (int t = 0; t < NT; t++) CHECK(fabsf(out[t * rows + r] - got) <= 1e-5f * (float)mag + 1e-6f, "%s mat_mul_rows token %d row %d", dtype_name(dt), t, r);
    }
    free(w); free(x); free(wd); free(data); free(xq); free(xdq);
}

/* The interleaved-layout contract written out from the plain blocks, with volatile intermediates
 * so that no compiler fuses the product and the sum: the engine's scalar kernel must give these
 * exact bits, and every SIMD kernel must give the scalar kernel's. */
static float contract_ref(const borg_mat *plain, int r, const block_q8a *x) {
    const uint8_t *row = (const uint8_t *)plain->data + (size_t)r * plain->row_bytes;
    float acc = 0.0f;
    for (int b = 0; b < plain->cols / QK; b++) {
        int32_t dot = 0;
        float d, mn = 0.0f;
        if (plain->dtype == DT_Q8_0) {
            const block_q8_0 *w = (const block_q8_0 *)row + b;
            for (int j = 0; j < QK; j++) dot += (int)w->qs[j] * x[b].qs[j];
            d = fp16_to_fp32(w->d);
        } else if (plain->dtype == DT_Q4_0) {
            const block_q4_0 *w = (const block_q4_0 *)row + b;
            for (int j = 0; j < QK / 2; j++) dot += ((w->qs[j] & 0x0F) - 8) * x[b].qs[j] + ((w->qs[j] >> 4) - 8) * x[b].qs[j + QK / 2];
            d = fp16_to_fp32(w->d);
        } else {
            const block_q4_1 *w = (const block_q4_1 *)row + b;
            for (int j = 0; j < QK / 2; j++) dot += (w->qs[j] & 0x0F) * x[b].qs[j] + (w->qs[j] >> 4) * x[b].qs[j + QK / 2];
            d = fp16_to_fp32(w->d);
            mn = fp16_to_fp32(w->m);
        }
        volatile float s = d * x[b].d;
        volatile float prod = s * (float)dot;
        acc = acc + prod;
        if (plain->dtype == DT_Q4_1) {
            volatile float mterm = mn * x[b].s;
            acc = acc + mterm;
        }
    }
    return acc;
}

static void test_x8(borg_dtype dt, int cols) {
    const int rows = 37, nb = cols / QK, TMAX = 64; /* 37 rows: 4 whole groups and a partial one */
    float *w = malloc(sizeof(float) * rows * cols);
    for (int i = 0; i < rows * cols; i++) w[i] = frand() * (i % 97 == 0 ? 8.0f : 1.0f);
    const size_t rb = dtype_row_bytes(dt, cols);
    uint8_t *data = malloc(rb * rows);
    for (int r = 0; r < rows; r++) {
        if (dt == DT_Q8_0) quantize_q8_0(w + (size_t)r * cols, (block_q8_0 *)(data + rb * r), cols);
        else if (dt == DT_Q4_0) quantize_q4_0(w + (size_t)r * cols, (block_q4_0 *)(data + rb * r), cols);
        else quantize_q4_1(w + (size_t)r * cols, (block_q4_1 *)(data + rb * r), cols);
    }
    const borg_mat plain = {dt, rows, cols, data, rb, MAT_PLAIN, NULL};
    uint8_t *x8 = malloc(mat_x8_bytes(&plain));
    CHECK(mat_x8_bytes(&plain) == rb * 40, "%s x8 bytes", dtype_name(dt));
    CHECK(mat_repack_x8(&plain, x8) == 0, "%s repack", dtype_name(dt));
    borg_mat m = plain;
    m.data = x8;
    m.layout = MAT_X8;
    { /* the in-place conversion of whole groups gives the same bytes as the copy */
        uint8_t *copy = malloc(rb * 32);
        memcpy(copy, data, rb * 32);
        borg_mat p32 = plain;
        p32.rows = 32;
        p32.data = copy;
        CHECK(mat_repack_x8(&p32, copy) == 0, "%s in-place repack", dtype_name(dt));
        CHECK(memcmp(copy, x8, rb * 32) == 0, "%s cols=%d in-place repack differs", dtype_name(dt), cols);
        free(copy);
    }
    float *a = malloc(sizeof(float) * cols), *b = malloc(sizeof(float) * cols);
    for (int r = 0; r < rows; r++) { /* round trip: every row dequantizes to the same floats */
        mat_get_row(&plain, r, a);
        mat_get_row(&m, r, b);
        CHECK(memcmp(a, b, sizeof(float) * cols) == 0, "%s cols=%d row %d dequantizes differently", dtype_name(dt), cols, r);
    }
    float *xs = malloc(sizeof(float) * TMAX * cols);
    block_q8a *xq = malloc(sizeof(block_q8a) * TMAX * nb);
    for (int i = 0; i < TMAX * cols; i++) xs[i] = frand() * (i % 53 == 0 ? 6.0f : 1.0f);
    for (int t = 0; t < TMAX; t++) quantize_row_q8a(xs + (size_t)t * cols, xq + (size_t)t * nb, cols);
    float *ref = malloc(sizeof(float) * TMAX * rows), *out = malloc(sizeof(float) * TMAX * rows);
    mat_mul_rows_scalar(&m, 0, rows, TMAX, xq, nb, ref, rows, 0);
    for (int t = 0; t < TMAX; t++)
        for (int r = 0; r < rows; r++) {
            const float c = contract_ref(&plain, r, xq + (size_t)t * nb);
            CHECK(memcmp(&c, &ref[t * rows + r], 4) == 0, "%s cols=%d scalar kernel != contract (token %d row %d: %.9g vs %.9g)",
                  dtype_name(dt), cols, t, r, ref[t * rows + r], c);
            if (t < 4) { /* and the contract is the dot product, to rounding */
                mat_get_row(&plain, r, a);
                double d = 0, mag = 0;
                for (int i = 0; i < cols; i++) {
                    const double xv = xq[(size_t)t * nb + i / QK].d * xq[(size_t)t * nb + i / QK].qs[i % QK];
                    d += a[i] * xv;
                    mag += fabs(a[i] * xv);
                }
                CHECK(fabs(c - d) <= 1e-5 * mag + 1e-6, "%s cols=%d contract vs dot (token %d row %d: %.7g vs %.7g)", dtype_name(dt), cols, t, r, c, d);
            }
        }
    /* every token count goes through its own kernel instantiation: all must give the scalar bits */
    const int Ts[] = {1, 2, 3, 4, 5, 8, 16, 17, 64};
    for (size_t k = 0; k < sizeof Ts / sizeof Ts[0]; k++) {
        const int T = Ts[k];
        memset(out, 0xFF, sizeof(float) * TMAX * rows);
        mat_mul_rows(&m, 0, rows, T, NULL, 0, xq, nb, out, rows);
        CHECK(memcmp(out, ref, sizeof(float) * T * rows) == 0, "%s cols=%d T=%d differs from the scalar kernel", dtype_name(dt), cols, T);
    }
    for (int t = 0; t < 5; t++)
        for (int r = 0; r < rows; r++) {
            const float one = mat_dot_row(&m, r, NULL, xq + (size_t)t * nb);
            CHECK(memcmp(&one, &ref[t * rows + r], 4) == 0, "%s cols=%d mat_dot_row token %d row %d", dtype_name(dt), cols, t, r);
        }
    /* row ranges of any alignment with an output offset: the lanes outside stay untouched */
    const int ranges[][3] = {{3, 29, 3}, {0, 37, 0}, {8, 24, 8}, {15, 17, 15}, {30, 37, 2}, {16, 33, 0}};
    for (size_t k = 0; k < sizeof ranges / sizeof ranges[0]; k++) {
        const int r0 = ranges[k][0], r1 = ranges[k][1], ro = ranges[k][2], T = 5;
        memset(out, 0xFF, sizeof(float) * TMAX * rows);
        mat_mul_rows_off(&m, r0, r1, T, NULL, 0, xq, nb, out, rows, ro);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < rows; c++) {
                const int r = c + ro;
                uint32_t bits;
                memcpy(&bits, &out[t * rows + c], 4);
                if (r >= r0 && r < r1) CHECK(memcmp(&out[t * rows + c], &ref[t * rows + r], 4) == 0, "%s rows [%d,%d) ro=%d token %d row %d", dtype_name(dt), r0, r1, ro, t, r);
                else CHECK(bits == 0xFFFFFFFFu, "%s rows [%d,%d) ro=%d wrote outside at token %d column %d", dtype_name(dt), r0, r1, ro, t, c);
            }
    }
    free(w); free(data); free(x8); free(a); free(b); free(xs); free(xq); free(ref); free(out);
}

static void test_vec(void) {
    for (int n = 1; n < 300; n += 7) {
        float a[300], b[300], y[300], y2[300];
        uint16_t h[300];
        double ref = 0, refh = 0;
        for (int i = 0; i < n; i++) { a[i] = frand(); b[i] = frand(); h[i] = fp32_to_fp16(a[i]); y[i] = y2[i] = frand();
            ref += (double)a[i] * b[i]; refh += (double)fp16_to_fp32(h[i]) * b[i]; }
        CHECK(fabs(vec_dot_f32(a, b, n) - ref) < 1e-4, "vec_dot_f32 n=%d", n);
        CHECK(fabs(vec_dot_f16_f32(h, b, n) - refh) < 1e-4, "vec_dot_f16_f32 n=%d", n);
        vec_axpy_f32(y, 0.5f, a, n);
        vec_axpy_f16(y2, 0.5f, h, n);
        for (int i = 0; i < n; i++) {
            CHECK(fabsf(y[i] - (y2[i] - 0.5f * fp16_to_fp32(h[i]) + 0.5f * a[i])) < 1e-5f, "axpy n=%d i=%d", n, i);
        }
    }
}

int main(void) {
    printf("simd: %s\n", borg_simd());
    test_fp16();
    test_vec();
    const int sizes[] = {32, 64, 96, 576, 896, 1024, 2048, 4096};
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
        test_dot(DT_F32, sizes[s]);
        test_dot(DT_F16, sizes[s]);
        test_dot(DT_BF16, sizes[s]);
        test_dot(DT_Q8_0, sizes[s]);
        test_dot(DT_Q4_0, sizes[s]);
        test_dot(DT_Q4_1, sizes[s]);
        test_x8(DT_Q8_0, sizes[s]);
        test_x8(DT_Q4_0, sizes[s]);
        test_x8(DT_Q4_1, sizes[s]);
    }
    test_dot(DT_F32, 37);
    test_dot(DT_F16, 45);
    printf(failures ? "%d FAILURES\n" : "all kernel tests passed\n", failures);
    return failures != 0;
}
