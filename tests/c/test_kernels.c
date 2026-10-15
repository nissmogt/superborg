/* Kernel tests: every SIMD dot product against a double-precision reference on the
 * dequantized operands, plus fp16 round trips. Run: make test-c */
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
    }
    borg_mat m = {dt, rows, cols, data, rb};
    block_q8a *xq = malloc(sizeof(block_q8a) * (cols / QK + 1));
    float *xdq = malloc(sizeof(float) * cols);
    if (cols % QK == 0) {
        quantize_row_q8a(x, xq, cols);
        for (int i = 0; i < cols / QK; i++) {
            int s = 0;
            for (int j = 0; j < QK; j++) { xdq[i * QK + j] = xq[i].d * xq[i].qs[j]; s += xq[i].qs[j]; }
            CHECK(fabsf(xq[i].s - xq[i].d * s) <= 1e-5f * fabsf(xq[i].d * s) + 1e-7f, "q8a block sum");
        }
    }
    float out[7 * 3];
    float xs[3 * 4096];
    for (int t = 0; t < 3; t++) memcpy(xs + t * cols, x, sizeof(float) * cols);
    block_q8a xqs[3 * 4096 / QK];
    if (cols % QK == 0) for (int t = 0; t < 3; t++) memcpy(xqs + t * (cols / QK), xq, sizeof(block_q8a) * (cols / QK));
    mat_mul_rows(&m, 0, rows, 3, xs, cols, xqs, cols / QK, out, rows);
    for (int r = 0; r < rows; r++) {
        mat_get_row(&m, r, wd);
        const float *xv = mat_wants_q8a(&m) ? xdq : x;
        double ref = 0, mag = 0;
        for (int i = 0; i < cols; i++) { ref += (double)wd[i] * xv[i]; mag += fabs((double)wd[i] * xv[i]); }
        float got = mat_dot_row(&m, r, x, xq);
        CHECK(fabs(got - ref) <= 1e-5 * mag + 1e-6, "%s cols=%d row %d: got %.7g want %.7g", dtype_name(dt), cols, r, got, ref);
        for (int t = 0; t < 3; t++) CHECK(out[t * rows + r] == got, "%s mat_mul_rows token %d row %d", dtype_name(dt), t, r);
    }
    free(w); free(x); free(wd); free(data); free(xq); free(xdq);
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
    }
    test_dot(DT_F32, 37);
    test_dot(DT_F16, 45);
    printf(failures ? "%d FAILURES\n" : "all kernel tests passed\n", failures);
    return failures != 0;
}
