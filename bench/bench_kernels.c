/* Single-thread matmul microbenchmark: GMAC/s for each weight type and batch size.
 * Reports the best of several repetitions so background load matters less.
 * Build: cc -O3 -march=native -Iengine bench/bench_kernels.c engine/kernels.c -lm */
#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 2048, cols = argc > 2 ? atoi(argv[2]) : 2048;
    const borg_dtype types[] = {DT_Q8_0, DT_Q4_0, DT_F16, DT_F32};
    const int batches[] = {1, 4, 16, 64};
    printf("simd %s, %dx%d matrix\n", borg_simd(), rows, cols);
    float *x = malloc(sizeof(float) * 64 * cols), *out = malloc(sizeof(float) * 64 * rows);
    block_q8a *xq = malloc(sizeof(block_q8a) * 64 * (cols / QK));
    for (int i = 0; i < 64 * cols; i++) x[i] = (float)((i * 2654435761u) % 1000) / 500.0f - 1.0f;
    for (int t = 0; t < 64; t++) quantize_row_q8a(x + (size_t)t * cols, xq + (size_t)t * (cols / QK), cols);
    for (size_t ti = 0; ti < sizeof types / sizeof types[0]; ti++) {
        borg_dtype dt = types[ti];
        size_t rb = dtype_row_bytes(dt, cols);
        uint8_t *w = malloc(rb * rows);
        for (size_t i = 0; i < rb * rows; i++) w[i] = (uint8_t)(i * 2654435761u >> 13);
        if (dt == DT_Q8_0 || dt == DT_Q4_0)
            for (int r = 0; r < rows * (cols / QK); r++) {
                uint16_t *d = (uint16_t *)(w + (size_t)r * (dt == DT_Q8_0 ? sizeof(block_q8_0) : sizeof(block_q4_0)));
                *d = fp32_to_fp16(0.01f);
            }
        if (dt == DT_F16) for (size_t i = 0; i < (size_t)rows * cols; i++) ((uint16_t *)w)[i] = fp32_to_fp16(0.01f);
        if (dt == DT_F32) for (size_t i = 0; i < (size_t)rows * cols; i++) ((float *)w)[i] = 0.01f;
        borg_mat m = {dt, rows, cols, w, rb};
        for (size_t bi = 0; bi < sizeof batches / sizeof batches[0]; bi++) {
            const int T = batches[bi];
            double best = 1e30;
            for (int rep = 0; rep < 7; rep++) {
                double t0 = now_ms();
                mat_mul_rows(&m, 0, rows, T, x, cols, xq, cols / QK, out, rows);
                double t1 = now_ms();
                if (t1 - t0 < best) best = t1 - t0;
            }
            printf("%-5s T=%-3d %8.3f ms  %7.2f GMAC/s\n", dtype_name(dt), T, best,
                   (double)rows * cols * T / best / 1e6);
        }
        free(w);
    }
    return 0;
}
