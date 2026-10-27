/* Numeric kernels: float16 conversion, activation quantization, dot products and the matrix
 * products on the interleaved weight layout.
 *
 * Each kernel has a portable scalar version. SIMD versions are compiled in when the
 * target supports them: AVX2 (+FMA, F16C), AVX-512 VNNI on x86, NEON (+dotprod) on ARM.
 */
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The interleaved-layout kernels promise the same float result on every target; WebAssembly has
 * no fused multiply-add, so their products and sums must not be contracted into one here either
 * (gcc fuses mul + add intrinsics by default; the Makefile also passes -ffp-contract=off). */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define BORG_AVX2 1
#include <immintrin.h>
#endif
#if defined(BORG_AVX2) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && \
    !defined(BORG_NO_VNNI)
#define BORG_VNNI 1
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define BORG_NEON 1
#include <arm_neon.h>
#endif
#if defined(__wasm_simd128__)
#define BORG_WASM 1
#include <wasm_simd128.h>
#endif

const char *borg_simd(void) {
#if defined(BORG_VNNI)
    return "avx512vnni";
#elif defined(BORG_AVX2)
    return "avx2";
#elif defined(BORG_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return "neon-dotprod";
#elif defined(BORG_NEON)
    return "neon";
#elif defined(BORG_WASM)
    return "wasm-simd128";
#else
    return "scalar";
#endif
}

/* ---- float16 / bfloat16 --------------------------------------------------- */
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

float fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#elif defined(__aarch64__)
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
#else
    /* exact conversion including subnormals, inf and nan */
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const float exp_scale = 0x1.0p-112f;
    const float normalized = u2f((two_w >> 4) + (0xE0u << 23)) * exp_scale;
    const float denormalized = u2f((two_w >> 17) | (126u << 23)) - 0.5f;
    const uint32_t result = sign | (two_w < (1u << 27) ? f2u(denormalized) : f2u(normalized));
    return u2f(result);
#endif
}

uint16_t fp32_to_fp16(float f) {
#if defined(__F16C__)
    return _cvtss_sh(f, 0);
#elif defined(__aarch64__)
    __fp16 v = (__fp16)f;
    uint16_t h;
    memcpy(&h, &v, 2);
    return h;
#else
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    const uint32_t w = f2u(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    base = u2f((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = f2u(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
#endif
}

float bf16_to_fp32(uint16_t h) { return u2f((uint32_t)h << 16); }

size_t dtype_row_bytes(borg_dtype dt, int cols) {
    switch (dt) {
    case DT_F32: return (size_t)cols * 4;
    case DT_F16: case DT_BF16: return (size_t)cols * 2;
    case DT_Q8_0: return (size_t)(cols / QK) * sizeof(block_q8_0);
    case DT_Q4_0: return (size_t)(cols / QK) * sizeof(block_q4_0);
    case DT_Q4_1: return (size_t)(cols / QK) * sizeof(block_q4_1);
    default: return 0;
    }
}

const char *dtype_name(borg_dtype dt) {
    static const char *names[DT_COUNT] = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "u8", "u32", "i32"};
    return dt < DT_COUNT ? names[dt] : "?";
}

/* ---- activation quantization --------------------------------------------- */
void quantize_row_q8a(const float *x, block_q8a *y, int n) {
    const int nb = n / QK;
    for (int i = 0; i < nb; i++) {
        const float *xb = x + i * QK;
        float amax = 0.0f;
#if defined(BORG_AVX2)
        __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 m = _mm256_setzero_ps();
        for (int j = 0; j < QK; j += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(xb + j)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        amax = _mm_cvtss_f32(m4);
#else
        for (int j = 0; j < QK; j++) {
            float a = fabsf(xb[j]);
            if (a > amax) amax = a;
        }
#endif
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y[i].d = d;
        int sum = 0;
#if defined(BORG_AVX2)
        __m256 vid = _mm256_set1_ps(id);
        __m256i q[4];
        for (int j = 0; j < 4; j++)
            q[j] = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xb + 8 * j), vid),
                                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i s = _mm256_add_epi32(_mm256_add_epi32(q[0], q[1]), _mm256_add_epi32(q[2], q[3]));
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        s4 = _mm_hadd_epi32(s4, s4);
        s4 = _mm_hadd_epi32(s4, s4);
        sum = _mm_cvtsi128_si32(s4);
        __m256i p01 = _mm256_packs_epi32(q[0], q[1]);
        __m256i p23 = _mm256_packs_epi32(q[2], q[3]);
        __m256i p = _mm256_packs_epi16(p01, p23);
        /* undo the in-lane interleaving of the packs */
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *)y[i].qs, p);
#else
        for (int j = 0; j < QK; j++) {
            int v = (int)lrintf(xb[j] * id);
            v = v > 127 ? 127 : (v < -127 ? -127 : v);
            y[i].qs[j] = (int8_t)v;
            sum += v;
        }
#endif
        y[i].s = d * (float)sum;
        y[i].sum = sum;
    }
}

/* ---- float dot products --------------------------------------------------- */
#if defined(BORG_AVX2)
static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
#endif

float vec_dot_f32(const float *a, const float *b, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(BORG_AVX2)
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    sum = hsum256(_mm256_add_ps(acc0, acc1));
#elif defined(BORG_NEON)
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#elif defined(BORG_WASM)
    v128_t acc0 = wasm_f32x4_splat(0.0f), acc1 = wasm_f32x4_splat(0.0f);
    for (; i + 8 <= n; i += 8) {
        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
    }
    acc0 = wasm_f32x4_add(acc0, acc1);
    sum = wasm_f32x4_extract_lane(acc0, 0) + wasm_f32x4_extract_lane(acc0, 1) + wasm_f32x4_extract_lane(acc0, 2) +
          wasm_f32x4_extract_lane(acc0, 3);
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void vec_axpy_f32(float *y, float a, const float *x, int n) {
    int i = 0;
#if defined(BORG_AVX2)
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
#elif defined(BORG_NEON)
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
#elif defined(BORG_WASM)
    v128_t va = wasm_f32x4_splat(a);
    for (; i + 4 <= n; i += 4)
        wasm_v128_store(y + i, wasm_f32x4_add(wasm_v128_load(y + i), wasm_f32x4_mul(va, wasm_v128_load(x + i))));
#endif
    for (; i < n; i++) y[i] += a * x[i];
}

float vec_dot_f16_f32(const uint16_t *a, const float *b, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(BORG_AVX2) && defined(__F16C__)
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i))), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i + 8))), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i))), _mm256_loadu_ps(b + i), acc0);
    sum = hsum256(_mm256_add_ps(acc0, acc1));
#elif defined(BORG_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
    float32x4_t acc = vdupq_n_f32(0);
    for (; i + 4 <= n; i += 4) acc = vfmaq_f32(acc, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a + i))), vld1q_f32(b + i));
    sum = vaddvq_f32(acc);
#endif
    for (; i < n; i++) sum += fp16_to_fp32(a[i]) * b[i];
    return sum;
}

void vec_axpy_f16(float *y, float a, const uint16_t *x, int n) {
    int i = 0;
#if defined(BORG_AVX2) && defined(__F16C__)
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))), _mm256_loadu_ps(y + i)));
#elif defined(BORG_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x + i)))));
#endif
    for (; i < n; i++) y[i] += a * fp16_to_fp32(x[i]);
}

static float vec_dot_bf16_f32(const uint16_t *a, const float *b, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(BORG_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(a + i))), 16);
        acc = _mm256_fmadd_ps(_mm256_castsi256_ps(w), _mm256_loadu_ps(b + i), acc);
    }
    sum = hsum256(acc);
#endif
    for (; i < n; i++) sum += bf16_to_fp32(a[i]) * b[i];
    return sum;
}

/* ---- quantized dot products ---------------------------------------------- */
#if defined(BORG_AVX2)
/* sum of int8 products of 32 lanes as 8 int32 lanes */
static inline __m256i dot_i8_i8(__m256i x, __m256i y) {
#if defined(BORG_VNNI)
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
#else
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot16 = _mm256_maddubs_epi16(ax, sy);
    return _mm256_madd_epi16(dot16, _mm256_set1_epi16(1));
#endif
}
/* unsigned (0..15) times signed int8 */
static inline __m256i dot_u8_i8(__m256i u, __m256i s) {
#if defined(BORG_VNNI)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1));
#endif
}
#endif

#if defined(BORG_WASM)
/* sum of 16 int8 products as 4 int32 lanes */
static inline v128_t wasm_dot_i8x16(v128_t a, v128_t b) {
    return wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(a), wasm_i16x8_extend_low_i8x16(b)),
                          wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(a), wasm_i16x8_extend_high_i8x16(b)));
}
static inline float wasm_hsum_f32(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) +
           wasm_f32x4_extract_lane(v, 3);
}
#endif

static float vec_dot_q8_0_q8a(const block_q8_0 *w, const block_q8a *x, int nb) {
    int i = 0;
    float sum = 0.0f;
#if defined(BORG_AVX2)
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (; i + 2 <= nb; i += 2) {
        __m256i qw0 = _mm256_loadu_si256((const __m256i *)w[i].qs);
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[i].qs);
        __m256i qw1 = _mm256_loadu_si256((const __m256i *)w[i + 1].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[i + 1].qs);
        __m256 d0 = _mm256_set1_ps(fp16_to_fp32(w[i].d) * x[i].d);
        __m256 d1 = _mm256_set1_ps(fp16_to_fp32(w[i + 1].d) * x[i + 1].d);
        acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(dot_i8_i8(qw0, qx0)), acc0);
        acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(dot_i8_i8(qw1, qx1)), acc1);
    }
    for (; i < nb; i++) {
        __m256i qw = _mm256_loadu_si256((const __m256i *)w[i].qs);
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[i].qs);
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(w[i].d) * x[i].d), _mm256_cvtepi32_ps(dot_i8_i8(qw, qx)), acc0);
    }
    sum = hsum256(_mm256_add_ps(acc0, acc1));
#elif defined(BORG_NEON)
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    for (; i < nb; i++) {
        int8x16_t w0 = vld1q_s8(w[i].qs), w1 = vld1q_s8(w[i].qs + 16);
        int8x16_t x0 = vld1q_s8(x[i].qs), x1 = vld1q_s8(x[i].qs + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
#else
        int16x8_t a0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
        int16x8_t a1 = vmull_s8(vget_high_s8(w0), vget_high_s8(x0));
        int16x8_t a2 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
        int16x8_t a3 = vmull_s8(vget_high_s8(w1), vget_high_s8(x1));
        int32x4_t p = vaddq_s32(vaddq_s32(vpaddlq_s16(a0), vpaddlq_s16(a1)), vaddq_s32(vpaddlq_s16(a2), vpaddlq_s16(a3)));
#endif
        float d = fp16_to_fp32(w[i].d) * x[i].d;
        if (i & 1) acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(p), d);
        else acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p), d);
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#elif defined(BORG_WASM)
    v128_t acc = wasm_f32x4_splat(0.0f);
    for (; i < nb; i++) {
        const v128_t p = wasm_i32x4_add(wasm_dot_i8x16(wasm_v128_load(w[i].qs), wasm_v128_load(x[i].qs)),
                                        wasm_dot_i8x16(wasm_v128_load(w[i].qs + 16), wasm_v128_load(x[i].qs + 16)));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(p), wasm_f32x4_splat(fp16_to_fp32(w[i].d) * x[i].d)));
    }
    sum = wasm_hsum_f32(acc);
#endif
    for (; i < nb; i++) {
        int s = 0;
        for (int j = 0; j < QK; j++) s += (int)w[i].qs[j] * (int)x[i].qs[j];
        sum += fp16_to_fp32(w[i].d) * x[i].d * (float)s;
    }
    return sum;
}

static float vec_dot_q4_0_q8a(const block_q4_0 *w, const block_q8a *x, int nb) {
    /* sum_j d_w * (q_j - 8) * d_x * x_j = d_w * (d_x * sum_j q_j x_j - 8 * s_x) */
    int i = 0;
    float sum = 0.0f;
#if defined(BORG_AVX2)
    const __m256i lowmask = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.0f;
    for (; i < nb; i++) {
        __m128i raw = _mm_loadu_si128((const __m128i *)w[i].qs);
        __m256i q = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), lowmask);
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[i].qs);
        const float dw = fp16_to_fp32(w[i].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[i].d), _mm256_cvtepi32_ps(dot_u8_i8(q, qx)), acc);
        corr += dw * x[i].s;
    }
    sum = hsum256(acc) - 8.0f * corr;
#elif defined(BORG_NEON)
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t s8 = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0);
    for (; i < nb; i++) {
        uint8x16_t raw = vld1q_u8(w[i].qs);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m4)), s8);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), s8);
        int8x16_t x0 = vld1q_s8(x[i].qs), x1 = vld1q_s8(x[i].qs + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, x0), hi, x1);
#else
        int16x8_t a0 = vmull_s8(vget_low_s8(lo), vget_low_s8(x0));
        int16x8_t a1 = vmull_s8(vget_high_s8(lo), vget_high_s8(x0));
        int16x8_t a2 = vmull_s8(vget_low_s8(hi), vget_low_s8(x1));
        int16x8_t a3 = vmull_s8(vget_high_s8(hi), vget_high_s8(x1));
        int32x4_t p = vaddq_s32(vaddq_s32(vpaddlq_s16(a0), vpaddlq_s16(a1)), vaddq_s32(vpaddlq_s16(a2), vpaddlq_s16(a3)));
#endif
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), fp16_to_fp32(w[i].d) * x[i].d);
    }
    sum = vaddvq_f32(acc);
#elif defined(BORG_WASM)
    const v128_t m4 = wasm_i8x16_splat(0x0F), s8 = wasm_i8x16_splat(8);
    v128_t acc = wasm_f32x4_splat(0.0f);
    for (; i < nb; i++) {
        const v128_t raw = wasm_v128_load(w[i].qs);
        const v128_t lo = wasm_i8x16_sub(wasm_v128_and(raw, m4), s8);
        const v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), s8);
        const v128_t p = wasm_i32x4_add(wasm_dot_i8x16(lo, wasm_v128_load(x[i].qs)),
                                        wasm_dot_i8x16(hi, wasm_v128_load(x[i].qs + 16)));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(p), wasm_f32x4_splat(fp16_to_fp32(w[i].d) * x[i].d)));
    }
    sum = wasm_hsum_f32(acc);
#endif
    for (; i < nb; i++) {
        int s = 0;
        for (int j = 0; j < QK / 2; j++) {
            s += ((int)(w[i].qs[j] & 0x0F) - 8) * (int)x[i].qs[j];
            s += ((int)(w[i].qs[j] >> 4) - 8) * (int)x[i].qs[j + QK / 2];
        }
        sum += fp16_to_fp32(w[i].d) * x[i].d * (float)s;
    }
    return sum;
}

static float vec_dot_q4_1_q8a(const block_q4_1 *w, const block_q8a *x, int nb) {
    /* sum_j (d_w q_j + m_w) d_x x_j = d_w d_x sum_j q_j x_j + m_w s_x */
    int i = 0;
    float sum = 0.0f;
#if defined(BORG_AVX2)
    const __m256i lowmask = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.0f;
    for (; i < nb; i++) {
        __m128i raw = _mm_loadu_si128((const __m128i *)w[i].qs);
        __m256i q = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), lowmask);
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[i].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(w[i].d) * x[i].d), _mm256_cvtepi32_ps(dot_u8_i8(q, qx)), acc);
        corr += fp16_to_fp32(w[i].m) * x[i].s;
    }
    sum = hsum256(acc) + corr;
#elif defined(BORG_NEON)
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float32x4_t acc = vdupq_n_f32(0);
    float corr = 0.0f;
    for (; i < nb; i++) {
        uint8x16_t raw = vld1q_u8(w[i].qs);
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, m4));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
        int8x16_t x0 = vld1q_s8(x[i].qs), x1 = vld1q_s8(x[i].qs + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, x0), hi, x1);
#else
        int16x8_t a0 = vmull_s8(vget_low_s8(lo), vget_low_s8(x0));
        int16x8_t a1 = vmull_s8(vget_high_s8(lo), vget_high_s8(x0));
        int16x8_t a2 = vmull_s8(vget_low_s8(hi), vget_low_s8(x1));
        int16x8_t a3 = vmull_s8(vget_high_s8(hi), vget_high_s8(x1));
        int32x4_t p = vaddq_s32(vaddq_s32(vpaddlq_s16(a0), vpaddlq_s16(a1)), vaddq_s32(vpaddlq_s16(a2), vpaddlq_s16(a3)));
#endif
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), fp16_to_fp32(w[i].d) * x[i].d);
        corr += fp16_to_fp32(w[i].m) * x[i].s;
    }
    sum = vaddvq_f32(acc) + corr;
#elif defined(BORG_WASM)
    const v128_t m4 = wasm_i8x16_splat(0x0F);
    v128_t acc = wasm_f32x4_splat(0.0f);
    float corr = 0.0f;
    for (; i < nb; i++) {
        const v128_t raw = wasm_v128_load(w[i].qs);
        const v128_t p = wasm_i32x4_add(wasm_dot_i8x16(wasm_v128_and(raw, m4), wasm_v128_load(x[i].qs)),
                                        wasm_dot_i8x16(wasm_u8x16_shr(raw, 4), wasm_v128_load(x[i].qs + 16)));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(p), wasm_f32x4_splat(fp16_to_fp32(w[i].d) * x[i].d)));
        corr += fp16_to_fp32(w[i].m) * x[i].s;
    }
    sum = wasm_hsum_f32(acc) + corr;
#endif
    for (; i < nb; i++) {
        int s = 0;
        for (int j = 0; j < QK / 2; j++) {
            s += (int)(w[i].qs[j] & 0x0F) * (int)x[i].qs[j];
            s += (int)(w[i].qs[j] >> 4) * (int)x[i].qs[j + QK / 2];
        }
        sum += fp16_to_fp32(w[i].d) * x[i].d * (float)s + fp16_to_fp32(w[i].m) * x[i].s;
    }
    return sum;
}

/* ---- interleaved weight layout (MAT_X8) ----------------------------------
 * The batched kernels read quantized matrices in groups of 8 rows (mat_repack_x8, applied when a
 * model is loaded). Per 32-element block a group stores the 8 rows' scales (8 fp16 d, then 8 fp16
 * m for q4_1) followed by the rows' quant bytes interleaved 4 bytes at a time: chunk j holds
 * bytes 4j..4j+3 of row 0, then of row 1, ... of row 7, so one 32-byte load is 8 columns x 4
 * elements and a token's 4 activation bytes are broadcast against it. q8_0 bytes are stored as
 * w + 127 (unsigned, for the u8 x s8 dot instructions); q4 nibbles stay unsigned. A group takes
 * the 8 * row_bytes bytes its rows took; rows past the end of a matrix (rows % 8 != 0) are zero.
 *
 * Numerical contract, the same for every kernel and token count: per (column, token) and block,
 * the exact int32 dot of the codes (the weight offsets 127 and 8 come out as 127 * sum(x) and
 * 8 * sum(x), also in int32), then acc = acc + (d_w * d_x) * float(dot) and, for q4_1,
 * acc = acc + m_w * s_x, in block order, each operation rounded to float. Single-token and
 * batched evaluation therefore agree bit for bit, and so do the scalar, AVX2, AVX-512, NEON and
 * WebAssembly kernels (tests/c/test_kernels.c checks both). */
static int x8_hdr(borg_dtype dt) { return dt == DT_Q4_1 ? 32 : 16; }    /* scale bytes per group block */
static int x8_qb(borg_dtype dt) { return dt == DT_Q8_0 ? QK : QK / 2; } /* quant bytes per row and block */
static int x8_comp(borg_dtype dt) { return dt == DT_Q8_0 ? 127 : dt == DT_Q4_0 ? 8 : 0; } /* weight offset */

static inline uint16_t x8_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
#if defined(BORG_AVX2)
static inline int32_t x8_dword(const int8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
#endif

size_t mat_x8_bytes(const borg_mat *m) { return (size_t)((m->rows + 7) / 8) * 8 * m->row_bytes; }

int mat_repack_x8(const borg_mat *m, void *dst) {
    const borg_dtype dt = m->dtype;
    const int nb = m->cols / QK, hdr = x8_hdr(dt), qb = x8_qb(dt), gb = hdr + 8 * qb;
    const int bb = (int)(m->row_bytes / (size_t)nb); /* bytes of a plain block */
    const int ng = (m->rows + 7) / 8;
    const size_t gbytes = 8 * m->row_bytes;
    const int in_place = dst == m->data;
    uint8_t *tmp = NULL;
    if (in_place && !(tmp = (uint8_t *)malloc(gbytes))) return -1;
    for (int g = 0; g < ng; g++) {
        const uint8_t *src = (const uint8_t *)m->data + (size_t)g * gbytes;
        uint8_t *out = (uint8_t *)dst + (size_t)g * gbytes;
        if (in_place) { /* the group's rows are overwritten by its own output */
            memcpy(tmp, src, gbytes);
            src = tmp;
        }
        if (g * 8 + 8 > m->rows) memset(out, 0, gbytes);
        for (int r = 0; r < 8 && g * 8 + r < m->rows; r++) {
            const uint8_t *row = src + (size_t)r * m->row_bytes;
            for (int b = 0; b < nb; b++) {
                const uint8_t *blk = row + (size_t)b * bb;
                uint8_t *ob = out + (size_t)b * gb;
                memcpy(ob + 2 * r, blk, 2);
                if (dt == DT_Q4_1) memcpy(ob + 16 + 2 * r, blk + 2, 2);
                const uint8_t *q = blk + (dt == DT_Q4_1 ? 4 : 2);
                uint8_t *oq = ob + hdr + 4 * r;
                for (int j = 0; j < qb / 4; j++) {
                    uint32_t v;
                    memcpy(&v, q + 4 * j, 4);
                    /* q8_0: 127 added to each byte modulo 256, with no carry between bytes */
                    if (dt == DT_Q8_0) v = ((v & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) ^ (v & 0x80808080u);
                    memcpy(oq + 32 * j, &v, 4);
                }
            }
        }
    }
    free(tmp);
    return 0;
}

/* Dequantize row r of an interleaved matrix (the same arithmetic as the plain layout). */
static void x8_get_row(const borg_mat *m, int r, float *out) {
    const borg_dtype dt = m->dtype;
    const int nb = m->cols / QK, hdr = x8_hdr(dt), gb = hdr + 8 * x8_qb(dt), lane = r % 8;
    const uint8_t *grp = (const uint8_t *)m->data + (size_t)(r / 8) * 8 * m->row_bytes;
    for (int b = 0; b < nb; b++) {
        const uint8_t *p = grp + (size_t)b * gb, *q = p + hdr + 4 * lane;
        const float d = fp16_to_fp32(x8_u16(p + 2 * lane));
        float *o = out + b * QK;
        if (dt == DT_Q8_0) {
            for (int j = 0; j < 8; j++)
                for (int k = 0; k < 4; k++) o[4 * j + k] = d * (float)((int)q[32 * j + k] - 127);
        } else if (dt == DT_Q4_0) {
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++) {
                    o[4 * j + k] = d * (float)((int)(q[32 * j + k] & 0x0F) - 8);
                    o[16 + 4 * j + k] = d * (float)((int)(q[32 * j + k] >> 4) - 8);
                }
        } else {
            const float mn = fp16_to_fp32(x8_u16(p + 16 + 2 * lane));
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++) {
                    o[4 * j + k] = d * (float)(q[32 * j + k] & 0x0F) + mn;
                    o[16 + 4 * j + k] = d * (float)(q[32 * j + k] >> 4) + mn;
                }
        }
    }
}

/* The contract spelled out: the reference for the SIMD kernels and the kernel of builds without
 * SIMD. The product and the sum are separate statements so that no compiler fuses them. */
void mat_mul_rows_scalar(const borg_mat *m, int r0, int r1, int T, const block_q8a *xq, int ldxq, float *out, int ldo,
                         int ro) {
    const borg_dtype dt = m->dtype;
    const int nb = m->cols / QK, hdr = x8_hdr(dt), gb = hdr + 8 * x8_qb(dt), comp = x8_comp(dt);
    for (int r = r0; r < r1; r++) {
        const uint8_t *grp = (const uint8_t *)m->data + (size_t)(r / 8) * 8 * m->row_bytes;
        const int lane = r % 8;
        for (int t = 0; t < T; t++) {
            const block_q8a *x = xq + (size_t)t * ldxq;
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                const uint8_t *p = grp + (size_t)b * gb, *q = p + hdr + 4 * lane;
                int32_t dot = 0;
                if (dt == DT_Q8_0) {
                    for (int j = 0; j < 8; j++)
                        for (int k = 0; k < 4; k++) dot += (int)q[32 * j + k] * x[b].qs[4 * j + k];
                } else {
                    for (int j = 0; j < 4; j++)
                        for (int k = 0; k < 4; k++) {
                            dot += (int)(q[32 * j + k] & 0x0F) * x[b].qs[4 * j + k];
                            dot += (int)(q[32 * j + k] >> 4) * x[b].qs[16 + 4 * j + k];
                        }
                }
                dot -= comp * x[b].sum;
                const float s = fp16_to_fp32(x8_u16(p + 2 * lane)) * x[b].d;
                const float prod = s * (float)dot;
                acc = acc + prod;
                if (dt == DT_Q4_1) {
                    const float mterm = fp16_to_fp32(x8_u16(p + 16 + 2 * lane)) * x[b].s;
                    acc = acc + mterm;
                }
            }
            out[(size_t)t * ldo + (size_t)(r - ro)] = acc;
        }
    }
}

/* X8_EACH(N, M) expands M(0) ... M(N-1): the SIMD kernels unroll their token loops this way so
 * that every accumulator is a register (left to the optimizer, a rolled loop keeps them in memory
 * and runs about 3x slower). */
#define X8_T1(M) M(0)
#define X8_T2(M) X8_T1(M) M(1)
#define X8_T3(M) X8_T2(M) M(2)
#define X8_T4(M) X8_T3(M) M(3)
#define X8_T5(M) X8_T4(M) M(4)
#define X8_T6(M) X8_T5(M) M(5)
#define X8_T7(M) X8_T6(M) M(6)
#define X8_T8(M) X8_T7(M) M(7)
#define X8_EACH_(N, M) X8_T##N(M)
#define X8_EACH(N, M) X8_EACH_(N, M)

/* A SIMD kernel computes 8 columns of row group g0 (16 on AVX-512: groups g0 and g1) against the
 * tokens xs[0..NT-1] and stores lanes [lo, hi) of each token's result at out + t * ldo. */
typedef void (*x8_kernel)(const uint8_t *g0, const uint8_t *g1, int nb, const block_q8a *const *xs, float *out,
                          int ldo, int lo, int hi);

#if defined(BORG_VNNI)
/* AVX-512: 16 columns x up to 8 tokens. The two groups' chunks share one zmm (8 columns x 4
 * elements each) and each token's 4 activation bytes are a broadcast dword, 8 dpbusd per token
 * and block into an int32 vector, then one convert, multiply and add. With 4 tokens or fewer
 * the even and odd chunks use separate int32 accumulators (ia, ib) so the dot chains are
 * shorter; the sum is the same. */
static inline __m512i z_load2(const uint8_t *a, const uint8_t *b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a)),
                              _mm256_loadu_si256((const __m256i *)b), 1);
}
static inline __m512 z_scales(const uint8_t *a, const uint8_t *b) {
    return _mm512_cvtph_ps(_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)a)),
                                                   _mm_loadu_si128((const __m128i *)b), 1));
}
static inline void z_store(float *o, int lo, int hi, __m512 v) {
    if (lo == 0 && hi == 16) _mm512_storeu_ps(o, v);
    else _mm512_mask_compressstoreu_ps(o, (__mmask16)(((1u << hi) - 1) & ~((1u << lo) - 1)), v);
}
#define Z_DECL(t)                                                                          \
    const block_q8a *x##t = xs[t];                                                         \
    __m512i ia##t = _mm512_setzero_si512(), ib##t = _mm512_setzero_si512();                \
    __m512 acc##t = _mm512_setzero_ps();
#define Z_ZERO(t) ia##t = _mm512_setzero_si512(); ib##t = _mm512_setzero_si512();
#define Z_DOTA(t) ia##t = _mm512_dpbusd_epi32(ia##t, w, _mm512_set1_epi32(x8_dword(x##t[b].qs + xo)));
#define Z_DOTB(t) ib##t = _mm512_dpbusd_epi32(ib##t, w, _mm512_set1_epi32(x8_dword(x##t[b].qs + xo)));
#define Z_FIN(t)                                                                                          \
    {                                                                                                     \
        const __m512i d = _mm512_sub_epi32(_mm512_add_epi32(ia##t, ib##t), _mm512_set1_epi32(comp * x##t[b].sum)); \
        acc##t = _mm512_add_ps(acc##t, _mm512_mul_ps(_mm512_mul_ps(dw, _mm512_set1_ps(x##t[b].d)), _mm512_cvtepi32_ps(d))); \
    }
#define Z_FIN41(t) acc##t = _mm512_add_ps(acc##t, _mm512_mul_ps(mw, _mm512_set1_ps(x##t[b].s)));
#define Z_STORE(t) z_store(out + (size_t)t * ldo, lo, hi, acc##t);
#define Z_BLOCK_Q8(NT, ODD)                                                                                    \
    for (int j = 0; j < 8; j += 2) {                                                                           \
        { const __m512i w = z_load2(p0 + 16 + 32 * j, p1 + 16 + 32 * j); const int xo = 4 * j; X8_EACH(NT, Z_DOTA) } \
        { const __m512i w = z_load2(p0 + 48 + 32 * j, p1 + 48 + 32 * j); const int xo = 4 * j + 4; X8_EACH(NT, ODD) } \
    }
/* low nibbles of chunk j pair with activation bytes 4j.., high nibbles with 16 + 4j.. */
#define Z_BLOCK_Q4(NT, ODD)                                                                                    \
    for (int j = 0; j < 4; j++) {                                                                              \
        const __m512i raw = z_load2(p0 + hdr + 32 * j, p1 + hdr + 32 * j);                                     \
        { const __m512i w = _mm512_and_si512(raw, m4); const int xo = 4 * j; X8_EACH(NT, Z_DOTA) }            \
        { const __m512i w = _mm512_and_si512(_mm512_srli_epi16(raw, 4), m4); const int xo = 16 + 4 * j; X8_EACH(NT, ODD) } \
    }
#define Z_M41(NT) { const __m512 mw = z_scales(p0 + 16, p1 + 16); X8_EACH(NT, Z_FIN41) }
#define Z_KERNEL(NAME, NT, ODD, BLOCK, HDR, GB, COMP, M41)                                                    \
    static void NAME(const uint8_t *g0, const uint8_t *g1, int nb, const block_q8a *const *xs, float *out, int ldo, \
                     int lo, int hi) {                                                                         \
        const int hdr = HDR, comp = COMP;                                                                      \
        const __m512i m4 = _mm512_set1_epi8(0x0F);                                                             \
        (void)hdr; (void)m4;                                                                                   \
        X8_EACH(NT, Z_DECL)                                                                                    \
        for (int b = 0; b < nb; b++) {                                                                         \
            const uint8_t *p0 = g0 + (size_t)b * GB, *p1 = g1 + (size_t)b * GB;                                \
            X8_EACH(NT, Z_ZERO)                                                                                \
            BLOCK(NT, ODD)                                                                                     \
            const __m512 dw = z_scales(p0, p1);                                                                \
            X8_EACH(NT, Z_FIN)                                                                                 \
            M41                                                                                                \
        }                                                                                                      \
        X8_EACH(NT, Z_STORE)                                                                                   \
    }
Z_KERNEL(z_q8_0_t1, 1, Z_DOTB, Z_BLOCK_Q8, 16, 272, 127, )
Z_KERNEL(z_q8_0_t2, 2, Z_DOTB, Z_BLOCK_Q8, 16, 272, 127, )
Z_KERNEL(z_q8_0_t3, 3, Z_DOTB, Z_BLOCK_Q8, 16, 272, 127, )
Z_KERNEL(z_q8_0_t4, 4, Z_DOTB, Z_BLOCK_Q8, 16, 272, 127, )
Z_KERNEL(z_q8_0_t5, 5, Z_DOTA, Z_BLOCK_Q8, 16, 272, 127, )
Z_KERNEL(z_q8_0_t6, 6, Z_DOTA, Z_BLOCK_Q8, 16, 272, 127, )
Z_KERNEL(z_q8_0_t7, 7, Z_DOTA, Z_BLOCK_Q8, 16, 272, 127, )
Z_KERNEL(z_q8_0_t8, 8, Z_DOTA, Z_BLOCK_Q8, 16, 272, 127, )
Z_KERNEL(z_q4_0_t1, 1, Z_DOTB, Z_BLOCK_Q4, 16, 144, 8, )
Z_KERNEL(z_q4_0_t2, 2, Z_DOTB, Z_BLOCK_Q4, 16, 144, 8, )
Z_KERNEL(z_q4_0_t3, 3, Z_DOTB, Z_BLOCK_Q4, 16, 144, 8, )
Z_KERNEL(z_q4_0_t4, 4, Z_DOTB, Z_BLOCK_Q4, 16, 144, 8, )
Z_KERNEL(z_q4_0_t5, 5, Z_DOTA, Z_BLOCK_Q4, 16, 144, 8, )
Z_KERNEL(z_q4_0_t6, 6, Z_DOTA, Z_BLOCK_Q4, 16, 144, 8, )
Z_KERNEL(z_q4_0_t7, 7, Z_DOTA, Z_BLOCK_Q4, 16, 144, 8, )
Z_KERNEL(z_q4_0_t8, 8, Z_DOTA, Z_BLOCK_Q4, 16, 144, 8, )
Z_KERNEL(z_q4_1_t1, 1, Z_DOTB, Z_BLOCK_Q4, 32, 160, 0, Z_M41(1))
Z_KERNEL(z_q4_1_t2, 2, Z_DOTB, Z_BLOCK_Q4, 32, 160, 0, Z_M41(2))
Z_KERNEL(z_q4_1_t3, 3, Z_DOTB, Z_BLOCK_Q4, 32, 160, 0, Z_M41(3))
Z_KERNEL(z_q4_1_t4, 4, Z_DOTB, Z_BLOCK_Q4, 32, 160, 0, Z_M41(4))
Z_KERNEL(z_q4_1_t5, 5, Z_DOTA, Z_BLOCK_Q4, 32, 160, 0, Z_M41(5))
Z_KERNEL(z_q4_1_t6, 6, Z_DOTA, Z_BLOCK_Q4, 32, 160, 0, Z_M41(6))
Z_KERNEL(z_q4_1_t7, 7, Z_DOTA, Z_BLOCK_Q4, 32, 160, 0, Z_M41(7))
Z_KERNEL(z_q4_1_t8, 8, Z_DOTA, Z_BLOCK_Q4, 32, 160, 0, Z_M41(8))
static const x8_kernel z_kernels[3][9] = {
    {NULL, z_q8_0_t1, z_q8_0_t2, z_q8_0_t3, z_q8_0_t4, z_q8_0_t5, z_q8_0_t6, z_q8_0_t7, z_q8_0_t8},
    {NULL, z_q4_0_t1, z_q4_0_t2, z_q4_0_t3, z_q4_0_t4, z_q4_0_t5, z_q4_0_t6, z_q4_0_t7, z_q4_0_t8},
    {NULL, z_q4_1_t1, z_q4_1_t2, z_q4_1_t3, z_q4_1_t4, z_q4_1_t5, z_q4_1_t6, z_q4_1_t7, z_q4_1_t8},
};
#endif /* BORG_VNNI */

#if defined(BORG_AVX2)
/* AVX2: 8 columns x up to 4 tokens in ymm registers (on AVX-512 machines these serve a group
 * without a partner). With a u8 x s8 dot instruction (AVX-512 VL or AVX-VNNI) the weight bytes
 * are used as stored. The maddubs fallback saturates its int16 pairs, so there q8_0 goes
 * through signed weights, |w| * sign(x, w), whose pairs stay below 2^15 and need no offset
 * correction; q4 nibbles are small enough as they are. */
#if defined(BORG_VNNI) || defined(__AVXVNNI__)
#define Y_U8 1
#define Y_Q8_COMP 127
static inline __m256i y_dpbusd(__m256i acc, __m256i w, __m256i x) {
#if defined(BORG_VNNI)
    return _mm256_dpbusd_epi32(acc, w, x);
#else
    return _mm256_dpbusd_avx_epi32(acc, w, x);
#endif
}
#else
#define Y_U8 0
#define Y_Q8_COMP 0
static inline __m256i y_dpbusd(__m256i acc, __m256i w, __m256i x) {
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1)));
}
#endif
static inline __m256 y_scales(const uint8_t *p) {
#if defined(__F16C__)
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
#else
    float f[8];
    for (int i = 0; i < 8; i++) f[i] = fp16_to_fp32(x8_u16(p + 2 * i));
    return _mm256_loadu_ps(f);
#endif
}
static inline void y_store(float *o, int lo, int hi, __m256 v) {
    if (lo == 0 && hi == 8) {
        _mm256_storeu_ps(o, v);
    } else {
        float tmp[8];
        _mm256_storeu_ps(tmp, v);
        memcpy(o, tmp + lo, (size_t)(hi - lo) * sizeof(float));
    }
}
#define Y_DECL(t)                                                                          \
    const block_q8a *x##t = xs[t];                                                         \
    __m256i ia##t = _mm256_setzero_si256(), ib##t = _mm256_setzero_si256();                \
    __m256 acc##t = _mm256_setzero_ps();
#define Y_ZERO(t) ia##t = _mm256_setzero_si256(); ib##t = _mm256_setzero_si256();
#define Y_DOTA(t) ia##t = y_dpbusd(ia##t, w, _mm256_set1_epi32(x8_dword(x##t[b].qs + xo)));
#define Y_DOTB(t) ib##t = y_dpbusd(ib##t, w, _mm256_set1_epi32(x8_dword(x##t[b].qs + xo)));
#define Y_DOTSA(t) ia##t = y_dpbusd(ia##t, w, _mm256_sign_epi8(_mm256_set1_epi32(x8_dword(x##t[b].qs + xo)), ws));
#define Y_DOTSB(t) ib##t = y_dpbusd(ib##t, w, _mm256_sign_epi8(_mm256_set1_epi32(x8_dword(x##t[b].qs + xo)), ws));
#define Y_FIN(t)                                                                                          \
    {                                                                                                     \
        const __m256i d = _mm256_sub_epi32(_mm256_add_epi32(ia##t, ib##t), _mm256_set1_epi32(comp * x##t[b].sum)); \
        acc##t = _mm256_add_ps(acc##t, _mm256_mul_ps(_mm256_mul_ps(dw, _mm256_set1_ps(x##t[b].d)), _mm256_cvtepi32_ps(d))); \
    }
#define Y_FIN41(t) acc##t = _mm256_add_ps(acc##t, _mm256_mul_ps(mw, _mm256_set1_ps(x##t[b].s)));
#define Y_STORE(t) y_store(out + (size_t)t * ldo, lo, hi, acc##t);
#if Y_U8
#define Y_BLOCK_Q8(NT, ODD)                                                                                    \
    for (int j = 0; j < 8; j += 2) {                                                                           \
        { const __m256i w = _mm256_loadu_si256((const __m256i *)(p0 + 16 + 32 * j)); const int xo = 4 * j; X8_EACH(NT, Y_DOTA) } \
        { const __m256i w = _mm256_loadu_si256((const __m256i *)(p0 + 48 + 32 * j)); const int xo = 4 * j + 4; X8_EACH(NT, ODD) } \
    }
#else
#define Y_BLOCK_Q8(NT, ODD)                                                                                    \
    for (int j = 0; j < 8; j += 2) {                                                                           \
        { const __m256i ws = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(p0 + 16 + 32 * j)), _mm256_set1_epi8(127)); \
          const __m256i w = _mm256_sign_epi8(ws, ws); const int xo = 4 * j; X8_EACH(NT, Y_DOTSA) }            \
        { const __m256i ws = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(p0 + 48 + 32 * j)), _mm256_set1_epi8(127)); \
          const __m256i w = _mm256_sign_epi8(ws, ws); const int xo = 4 * j + 4; X8_EACH(NT, Y_DOTSB) }        \
    }
#endif
#define Y_BLOCK_Q4(NT, ODD)                                                                                    \
    for (int j = 0; j < 4; j++) {                                                                              \
        const __m256i raw = _mm256_loadu_si256((const __m256i *)(p0 + hdr + 32 * j));                          \
        { const __m256i w = _mm256_and_si256(raw, m4); const int xo = 4 * j; X8_EACH(NT, Y_DOTA) }            \
        { const __m256i w = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4); const int xo = 16 + 4 * j; X8_EACH(NT, ODD) } \
    }
#define Y_M41(NT) { const __m256 mw = y_scales(p0 + 16); X8_EACH(NT, Y_FIN41) }
#define Y_KERNEL(NAME, NT, ODD, BLOCK, HDR, GB, COMP, M41)                                                    \
    static void NAME(const uint8_t *g0, const uint8_t *g1, int nb, const block_q8a *const *xs, float *out, int ldo, \
                     int lo, int hi) {                                                                         \
        const int hdr = HDR, comp = COMP;                                                                      \
        const __m256i m4 = _mm256_set1_epi8(0x0F);                                                             \
        (void)g1; (void)hdr; (void)m4;                                                                         \
        X8_EACH(NT, Y_DECL)                                                                                    \
        for (int b = 0; b < nb; b++) {                                                                         \
            const uint8_t *p0 = g0 + (size_t)b * GB;                                                           \
            X8_EACH(NT, Y_ZERO)                                                                                \
            BLOCK(NT, ODD)                                                                                     \
            const __m256 dw = y_scales(p0);                                                                    \
            X8_EACH(NT, Y_FIN)                                                                                 \
            M41                                                                                                \
        }                                                                                                      \
        X8_EACH(NT, Y_STORE)                                                                                   \
    }
Y_KERNEL(y_q8_0_t1, 1, Y_DOTB, Y_BLOCK_Q8, 16, 272, Y_Q8_COMP, )
Y_KERNEL(y_q8_0_t2, 2, Y_DOTB, Y_BLOCK_Q8, 16, 272, Y_Q8_COMP, )
Y_KERNEL(y_q8_0_t3, 3, Y_DOTB, Y_BLOCK_Q8, 16, 272, Y_Q8_COMP, )
Y_KERNEL(y_q8_0_t4, 4, Y_DOTB, Y_BLOCK_Q8, 16, 272, Y_Q8_COMP, )
Y_KERNEL(y_q4_0_t1, 1, Y_DOTB, Y_BLOCK_Q4, 16, 144, 8, )
Y_KERNEL(y_q4_0_t2, 2, Y_DOTB, Y_BLOCK_Q4, 16, 144, 8, )
Y_KERNEL(y_q4_0_t3, 3, Y_DOTB, Y_BLOCK_Q4, 16, 144, 8, )
Y_KERNEL(y_q4_0_t4, 4, Y_DOTB, Y_BLOCK_Q4, 16, 144, 8, )
Y_KERNEL(y_q4_1_t1, 1, Y_DOTB, Y_BLOCK_Q4, 32, 160, 0, Y_M41(1))
Y_KERNEL(y_q4_1_t2, 2, Y_DOTB, Y_BLOCK_Q4, 32, 160, 0, Y_M41(2))
Y_KERNEL(y_q4_1_t3, 3, Y_DOTB, Y_BLOCK_Q4, 32, 160, 0, Y_M41(3))
Y_KERNEL(y_q4_1_t4, 4, Y_DOTB, Y_BLOCK_Q4, 32, 160, 0, Y_M41(4))
#define X8_NT8 4
static const x8_kernel x8_kernels8[3][5] = {
    {NULL, y_q8_0_t1, y_q8_0_t2, y_q8_0_t3, y_q8_0_t4},
    {NULL, y_q4_0_t1, y_q4_0_t2, y_q4_0_t3, y_q4_0_t4},
    {NULL, y_q4_1_t1, y_q4_1_t2, y_q4_1_t3, y_q4_1_t4},
};
#endif /* BORG_AVX2 */

#if defined(BORG_NEON)
/* NEON: 8 columns x up to 4 tokens; a 16-byte chunk load is 4 columns x 4 elements. With dotprod,
 * vdot against lane j of the token's 16 activation bytes; without it, widening multiplies whose
 * pair sums are accumulated and folded to column sums once per block (exact integer arithmetic
 * either way). q8_0 bytes are made signed again (w + 127 - 127) for the signed-by-signed
 * instructions, so no offset is subtracted; q4 nibbles are used as they are. */
static inline int8x16_t n_signed(const uint8_t *p) { return vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(p), vdupq_n_u8(127))); }
static inline float32x4_t n_scales(const uint8_t *p) {
#if defined(__ARM_FP16_FORMAT_IEEE)
    uint16x4_t h = vld1_u16((const uint16_t *)(const void *)p);
    return vcvt_f32_f16(vreinterpret_f16_u16(h));
#else
    float f[4];
    for (int i = 0; i < 4; i++) f[i] = fp16_to_fp32(x8_u16(p + 2 * i));
    return vld1q_f32(f);
#endif
}
static inline int32x4_t n_pairsum(int32x4_t a, int32x4_t b) { /* [a0+a1, a2+a3, b0+b1, b2+b3] */
#if defined(__aarch64__)
    return vpaddq_s32(a, b);
#else
    return vcombine_s32(vpadd_s32(vget_low_s32(a), vget_high_s32(a)), vpadd_s32(vget_low_s32(b), vget_high_s32(b)));
#endif
}
static inline void n_store(float *o, int lo, int hi, float32x4_t a, float32x4_t b) {
    if (lo == 0 && hi == 8) {
        vst1q_f32(o, a);
        vst1q_f32(o + 4, b);
    } else {
        float tmp[8];
        vst1q_f32(tmp, a);
        vst1q_f32(tmp + 4, b);
        memcpy(o, tmp + lo, (size_t)(hi - lo) * sizeof(float));
    }
}
#if defined(__ARM_FEATURE_DOTPROD)
#define N_DECL(t)                                                                          \
    const block_q8a *x##t = xs[t];                                                         \
    int32x4_t ia##t = vdupq_n_s32(0), ib##t = vdupq_n_s32(0);                              \
    float32x4_t acc##t = vdupq_n_f32(0), acd##t = vdupq_n_f32(0);                          \
    int8x16_t xa##t, xb##t;
#define N_ZERO(t) ia##t = vdupq_n_s32(0); ib##t = vdupq_n_s32(0); xa##t = vld1q_s8(x##t[b].qs); xb##t = vld1q_s8(x##t[b].qs + 16);
#define N_DOT(t, XV, L) ia##t = vdotq_laneq_s32(ia##t, w0, XV##t, L); ib##t = vdotq_laneq_s32(ib##t, w1, XV##t, L);
#define N_FOLD(t)
#else
#define N_DECL(t)                                                                          \
    const block_q8a *x##t = xs[t];                                                         \
    int32x4_t ia##t, ib##t, pa##t = vdupq_n_s32(0), pb##t = vdupq_n_s32(0), pc##t = vdupq_n_s32(0), pd##t = vdupq_n_s32(0); \
    float32x4_t acc##t = vdupq_n_f32(0), acd##t = vdupq_n_f32(0);                          \
    int8x16_t xa##t, xb##t;
#define N_ZERO(t) pa##t = pb##t = pc##t = pd##t = vdupq_n_s32(0); xa##t = vld1q_s8(x##t[b].qs); xb##t = vld1q_s8(x##t[b].qs + 16);
#define N_DOT(t, XV, L)                                                                                     \
    {                                                                                                       \
        const int8x16_t xr = vreinterpretq_s8_s32(vdupq_n_s32(vgetq_lane_s32(vreinterpretq_s32_s8(XV##t), L))); \
        pa##t = vpadalq_s16(pa##t, vmull_s8(vget_low_s8(w0), vget_low_s8(xr)));                                 \
        pb##t = vpadalq_s16(pb##t, vmull_s8(vget_high_s8(w0), vget_high_s8(xr)));                               \
        pc##t = vpadalq_s16(pc##t, vmull_s8(vget_low_s8(w1), vget_low_s8(xr)));                                 \
        pd##t = vpadalq_s16(pd##t, vmull_s8(vget_high_s8(w1), vget_high_s8(xr)));                               \
    }
#define N_FOLD(t) ia##t = n_pairsum(pa##t, pb##t); ib##t = n_pairsum(pc##t, pd##t);
#endif
#define N_DA0(t) N_DOT(t, xa, 0)
#define N_DA1(t) N_DOT(t, xa, 1)
#define N_DA2(t) N_DOT(t, xa, 2)
#define N_DA3(t) N_DOT(t, xa, 3)
#define N_DB0(t) N_DOT(t, xb, 0)
#define N_DB1(t) N_DOT(t, xb, 1)
#define N_DB2(t) N_DOT(t, xb, 2)
#define N_DB3(t) N_DOT(t, xb, 3)
#define N_FIN(t)                                                                                          \
    {                                                                                                     \
        N_FOLD(t)                                                                                         \
        const int32x4_t c = vdupq_n_s32(comp * x##t[b].sum);                                              \
        const float32x4_t s0 = vmulq_n_f32(dw0, x##t[b].d), s1 = vmulq_n_f32(dw1, x##t[b].d);              \
        acc##t = vaddq_f32(acc##t, vmulq_f32(s0, vcvtq_f32_s32(vsubq_s32(ia##t, c))));                     \
        acd##t = vaddq_f32(acd##t, vmulq_f32(s1, vcvtq_f32_s32(vsubq_s32(ib##t, c))));                     \
    }
#define N_FIN41(t) acc##t = vaddq_f32(acc##t, vmulq_n_f32(mw0, x##t[b].s)); acd##t = vaddq_f32(acd##t, vmulq_n_f32(mw1, x##t[b].s));
#define N_STORE(t) n_store(out + (size_t)t * ldo, lo, hi, acc##t, acd##t);
#define N_Q8_STEP(NT, J, D) { const int8x16_t w0 = n_signed(p0 + 16 + 32 * (J)), w1 = n_signed(p0 + 32 + 32 * (J)); X8_EACH(NT, D) }
#define N_BLOCK_Q8(NT, ODD)                                                                                    \
    N_Q8_STEP(NT, 0, N_DA0) N_Q8_STEP(NT, 1, N_DA1) N_Q8_STEP(NT, 2, N_DA2) N_Q8_STEP(NT, 3, N_DA3)            \
    N_Q8_STEP(NT, 4, N_DB0) N_Q8_STEP(NT, 5, N_DB1) N_Q8_STEP(NT, 6, N_DB2) N_Q8_STEP(NT, 7, N_DB3)
#define N_Q4_STEP(NT, J, DLO, DHI)                                                                             \
    {                                                                                                          \
        const uint8x16_t raw0 = vld1q_u8(p0 + hdr + 32 * (J)), raw1 = vld1q_u8(p0 + hdr + 16 + 32 * (J));      \
        { const int8x16_t w0 = vreinterpretq_s8_u8(vandq_u8(raw0, m4)), w1 = vreinterpretq_s8_u8(vandq_u8(raw1, m4)); X8_EACH(NT, DLO) } \
        { const int8x16_t w0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4)), w1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4)); X8_EACH(NT, DHI) } \
    }
#define N_BLOCK_Q4(NT, ODD)                                                                                    \
    N_Q4_STEP(NT, 0, N_DA0, N_DB0) N_Q4_STEP(NT, 1, N_DA1, N_DB1) N_Q4_STEP(NT, 2, N_DA2, N_DB2) N_Q4_STEP(NT, 3, N_DA3, N_DB3)
#define N_M41(NT) { const float32x4_t mw0 = n_scales(p0 + 16), mw1 = n_scales(p0 + 24); X8_EACH(NT, N_FIN41) }
#define N_KERNEL(NAME, NT, ODD, BLOCK, HDR, GB, COMP, M41)                                                    \
    static void NAME(const uint8_t *g0, const uint8_t *g1, int nb, const block_q8a *const *xs, float *out, int ldo, \
                     int lo, int hi) {                                                                         \
        const int hdr = HDR, comp = COMP;                                                                      \
        const uint8x16_t m4 = vdupq_n_u8(0x0F);                                                                \
        (void)g1; (void)hdr; (void)m4;                                                                         \
        X8_EACH(NT, N_DECL)                                                                                    \
        for (int b = 0; b < nb; b++) {                                                                         \
            const uint8_t *p0 = g0 + (size_t)b * GB;                                                           \
            X8_EACH(NT, N_ZERO)                                                                                \
            BLOCK(NT, ODD)                                                                                     \
            const float32x4_t dw0 = n_scales(p0), dw1 = n_scales(p0 + 8);                                      \
            X8_EACH(NT, N_FIN)                                                                                 \
            M41                                                                                                \
        }                                                                                                      \
        X8_EACH(NT, N_STORE)                                                                                   \
    }
N_KERNEL(n_q8_0_t1, 1, 0, N_BLOCK_Q8, 16, 272, 0, )
N_KERNEL(n_q8_0_t2, 2, 0, N_BLOCK_Q8, 16, 272, 0, )
N_KERNEL(n_q8_0_t3, 3, 0, N_BLOCK_Q8, 16, 272, 0, )
N_KERNEL(n_q8_0_t4, 4, 0, N_BLOCK_Q8, 16, 272, 0, )
N_KERNEL(n_q4_0_t1, 1, 0, N_BLOCK_Q4, 16, 144, 8, )
N_KERNEL(n_q4_0_t2, 2, 0, N_BLOCK_Q4, 16, 144, 8, )
N_KERNEL(n_q4_0_t3, 3, 0, N_BLOCK_Q4, 16, 144, 8, )
N_KERNEL(n_q4_0_t4, 4, 0, N_BLOCK_Q4, 16, 144, 8, )
N_KERNEL(n_q4_1_t1, 1, 0, N_BLOCK_Q4, 32, 160, 0, N_M41(1))
N_KERNEL(n_q4_1_t2, 2, 0, N_BLOCK_Q4, 32, 160, 0, N_M41(2))
N_KERNEL(n_q4_1_t3, 3, 0, N_BLOCK_Q4, 32, 160, 0, N_M41(3))
N_KERNEL(n_q4_1_t4, 4, 0, N_BLOCK_Q4, 32, 160, 0, N_M41(4))
#define X8_NT8 4
static const x8_kernel x8_kernels8[3][5] = {
    {NULL, n_q8_0_t1, n_q8_0_t2, n_q8_0_t3, n_q8_0_t4},
    {NULL, n_q4_0_t1, n_q4_0_t2, n_q4_0_t3, n_q4_0_t4},
    {NULL, n_q4_1_t1, n_q4_1_t2, n_q4_1_t3, n_q4_1_t4},
};
#endif /* BORG_NEON */

#if defined(BORG_WASM)
/* WebAssembly SIMD128: 8 columns x up to 4 tokens. There is no byte dot product, so each chunk
 * is widened to int16 once (4 columns x 4 elements per vector), a token's 4 activation bytes
 * are splat and widened, and i32x4.dot_i16x8 leaves two partial sums per column that are
 * accumulated over the chunks and folded to column sums once per block (exact integers). */
static inline v128_t w_scales(const uint8_t *p) {
    return wasm_f32x4_make(fp16_to_fp32(x8_u16(p)), fp16_to_fp32(x8_u16(p + 2)), fp16_to_fp32(x8_u16(p + 4)),
                           fp16_to_fp32(x8_u16(p + 6)));
}
static inline v128_t w_fold(v128_t a, v128_t b) { /* [a0+a1, a2+a3, b0+b1, b2+b3] */
    return wasm_i32x4_add(wasm_i32x4_shuffle(a, b, 0, 2, 4, 6), wasm_i32x4_shuffle(a, b, 1, 3, 5, 7));
}
static inline void w_store(float *o, int lo, int hi, v128_t a, v128_t b) {
    if (lo == 0 && hi == 8) {
        wasm_v128_store(o, a);
        wasm_v128_store(o + 4, b);
    } else {
        float tmp[8];
        wasm_v128_store(tmp, a);
        wasm_v128_store(tmp + 4, b);
        memcpy(o, tmp + lo, (size_t)(hi - lo) * sizeof(float));
    }
}
#define W_DECL(t)                                                                          \
    const block_q8a *x##t = xs[t];                                                         \
    v128_t pa##t = wasm_i32x4_splat(0), pb##t = wasm_i32x4_splat(0), pc##t = wasm_i32x4_splat(0), pd##t = wasm_i32x4_splat(0); \
    v128_t acc##t = wasm_f32x4_splat(0.0f), acd##t = wasm_f32x4_splat(0.0f);
#define W_ZERO(t) pa##t = pb##t = pc##t = pd##t = wasm_i32x4_splat(0);
#define W_DOT(t)                                                                                          \
    {                                                                                                     \
        const v128_t xw = wasm_i16x8_extend_low_i8x16(wasm_v128_load32_splat(x##t[b].qs + xo));           \
        pa##t = wasm_i32x4_add(pa##t, wasm_i32x4_dot_i16x8(w0l, xw));                                     \
        pb##t = wasm_i32x4_add(pb##t, wasm_i32x4_dot_i16x8(w0h, xw));                                     \
        pc##t = wasm_i32x4_add(pc##t, wasm_i32x4_dot_i16x8(w1l, xw));                                     \
        pd##t = wasm_i32x4_add(pd##t, wasm_i32x4_dot_i16x8(w1h, xw));                                     \
    }
#define W_FIN(t)                                                                                          \
    {                                                                                                     \
        const v128_t c = wasm_i32x4_splat(comp * x##t[b].sum);                                            \
        const v128_t ia = wasm_i32x4_sub(w_fold(pa##t, pb##t), c), ib = wasm_i32x4_sub(w_fold(pc##t, pd##t), c); \
        const v128_t xd = wasm_f32x4_splat(x##t[b].d);                                                    \
        acc##t = wasm_f32x4_add(acc##t, wasm_f32x4_mul(wasm_f32x4_mul(dw0, xd), wasm_f32x4_convert_i32x4(ia))); \
        acd##t = wasm_f32x4_add(acd##t, wasm_f32x4_mul(wasm_f32x4_mul(dw1, xd), wasm_f32x4_convert_i32x4(ib))); \
    }
#define W_FIN41(t)                                                                                        \
    acc##t = wasm_f32x4_add(acc##t, wasm_f32x4_mul(mw0, wasm_f32x4_splat(x##t[b].s)));                    \
    acd##t = wasm_f32x4_add(acd##t, wasm_f32x4_mul(mw1, wasm_f32x4_splat(x##t[b].s)));
#define W_STORE(t) w_store(out + (size_t)t * ldo, lo, hi, acc##t, acd##t);
/* the chunk widened: w0l/w0h are columns 0,1 / 2,3 and w1l/w1h columns 4,5 / 6,7 */
#define W_WIDEN(c0, c1)                                                                                   \
    const v128_t w0l = wasm_u16x8_extend_low_u8x16(c0), w0h = wasm_u16x8_extend_high_u8x16(c0);           \
    const v128_t w1l = wasm_u16x8_extend_low_u8x16(c1), w1h = wasm_u16x8_extend_high_u8x16(c1);
#define W_BLOCK_Q8(NT, ODD)                                                                               \
    for (int j = 0; j < 8; j++) {                                                                         \
        W_WIDEN(wasm_v128_load(p0 + 16 + 32 * j), wasm_v128_load(p0 + 32 + 32 * j))                       \
        const int xo = 4 * j;                                                                             \
        X8_EACH(NT, W_DOT)                                                                                \
    }
#define W_BLOCK_Q4(NT, ODD)                                                                               \
    for (int j = 0; j < 4; j++) {                                                                         \
        const v128_t raw0 = wasm_v128_load(p0 + hdr + 32 * j), raw1 = wasm_v128_load(p0 + hdr + 16 + 32 * j); \
        { W_WIDEN(wasm_v128_and(raw0, m4), wasm_v128_and(raw1, m4)) const int xo = 4 * j; X8_EACH(NT, W_DOT) } \
        { W_WIDEN(wasm_u8x16_shr(raw0, 4), wasm_u8x16_shr(raw1, 4)) const int xo = 16 + 4 * j; X8_EACH(NT, W_DOT) } \
    }
#define W_M41(NT) { const v128_t mw0 = w_scales(p0 + 16), mw1 = w_scales(p0 + 24); X8_EACH(NT, W_FIN41) }
#define W_KERNEL(NAME, NT, ODD, BLOCK, HDR, GB, COMP, M41)                                                    \
    static void NAME(const uint8_t *g0, const uint8_t *g1, int nb, const block_q8a *const *xs, float *out, int ldo, \
                     int lo, int hi) {                                                                         \
        const int hdr = HDR, comp = COMP;                                                                      \
        const v128_t m4 = wasm_i8x16_splat(0x0F);                                                              \
        (void)g1; (void)hdr; (void)m4;                                                                         \
        X8_EACH(NT, W_DECL)                                                                                    \
        for (int b = 0; b < nb; b++) {                                                                         \
            const uint8_t *p0 = g0 + (size_t)b * GB;                                                           \
            X8_EACH(NT, W_ZERO)                                                                                \
            BLOCK(NT, ODD)                                                                                     \
            const v128_t dw0 = w_scales(p0), dw1 = w_scales(p0 + 8);                                           \
            X8_EACH(NT, W_FIN)                                                                                 \
            M41                                                                                                \
        }                                                                                                      \
        X8_EACH(NT, W_STORE)                                                                                   \
    }
W_KERNEL(w_q8_0_t1, 1, 0, W_BLOCK_Q8, 16, 272, 127, )
W_KERNEL(w_q8_0_t2, 2, 0, W_BLOCK_Q8, 16, 272, 127, )
W_KERNEL(w_q8_0_t3, 3, 0, W_BLOCK_Q8, 16, 272, 127, )
W_KERNEL(w_q8_0_t4, 4, 0, W_BLOCK_Q8, 16, 272, 127, )
W_KERNEL(w_q4_0_t1, 1, 0, W_BLOCK_Q4, 16, 144, 8, )
W_KERNEL(w_q4_0_t2, 2, 0, W_BLOCK_Q4, 16, 144, 8, )
W_KERNEL(w_q4_0_t3, 3, 0, W_BLOCK_Q4, 16, 144, 8, )
W_KERNEL(w_q4_0_t4, 4, 0, W_BLOCK_Q4, 16, 144, 8, )
W_KERNEL(w_q4_1_t1, 1, 0, W_BLOCK_Q4, 32, 160, 0, W_M41(1))
W_KERNEL(w_q4_1_t2, 2, 0, W_BLOCK_Q4, 32, 160, 0, W_M41(2))
W_KERNEL(w_q4_1_t3, 3, 0, W_BLOCK_Q4, 32, 160, 0, W_M41(3))
W_KERNEL(w_q4_1_t4, 4, 0, W_BLOCK_Q4, 32, 160, 0, W_M41(4))
#define X8_NT8 4
static const x8_kernel x8_kernels8[3][5] = {
    {NULL, w_q8_0_t1, w_q8_0_t2, w_q8_0_t3, w_q8_0_t4},
    {NULL, w_q4_0_t1, w_q4_0_t2, w_q4_0_t3, w_q4_0_t4},
    {NULL, w_q4_1_t1, w_q4_1_t2, w_q4_1_t3, w_q4_1_t4},
};
#endif /* BORG_WASM */

#if defined(X8_NT8)
/* Run a kernel set over the tokens in tiles of nt_max. */
static void x8_run(const x8_kernel *k, int nt_max, const uint8_t *g0, const uint8_t *g1, int nb, int T,
                   const block_q8a *xq, int ldxq, float *o, int ldo, int lo, int hi) {
    for (int t0 = 0; t0 < T; t0 += nt_max) {
        const int nt = T - t0 < nt_max ? T - t0 : nt_max;
        const block_q8a *xs[8];
        for (int t = 0; t < nt; t++) xs[t] = xq + (size_t)(t0 + t) * ldxq;
        k[nt](g0, g1, nb, xs, o + (size_t)t0 * ldo, ldo, lo, hi);
    }
}
#endif

/* mat_mul_rows_off for an interleaved matrix: whole row groups go through the kernels and only
 * the lanes inside [r0, r1) are stored, so any row range works (the partial groups at the ends
 * of a range cost a full group each). Groups are the outer loop so that a group's weights stay
 * in cache while the activations of all tokens stream past. */
static void x8_mul(const borg_mat *m, int r0, int r1, int T, const block_q8a *xq, int ldxq, float *out, int ldo, int ro) {
#if defined(X8_NT8)
    const int nb = m->cols / QK, ng = (m->rows + 7) / 8, k = m->dtype - DT_Q8_0;
    const size_t gbytes = 8 * m->row_bytes;
    for (int c0 = r0 & ~7; c0 < r1;) {
        const int g = c0 / 8;
        const uint8_t *g0 = (const uint8_t *)m->data + (size_t)g * gbytes;
        int cols = 8;
#if defined(BORG_VNNI)
        if (g + 1 < ng && c0 + 8 < r1) cols = 16; /* a pair of groups for the 16-column kernels */
#else
        (void)ng;
#endif
        const int lo = (r0 > c0 ? r0 : c0) - c0, hi = (r1 < c0 + cols ? r1 : c0 + cols) - c0;
        float *o = out + (size_t)(c0 + lo - ro);
#if defined(BORG_VNNI)
        if (cols == 16) x8_run(z_kernels[k], 8, g0, g0 + gbytes, nb, T, xq, ldxq, o, ldo, lo, hi);
        else
#endif
            x8_run(x8_kernels8[k], X8_NT8, g0, g0, nb, T, xq, ldxq, o, ldo, lo, hi);
        c0 += cols;
    }
#else
    mat_mul_rows_scalar(m, r0, r1, T, xq, ldxq, out, ldo, ro);
#endif
}

/* ---- matrix helpers ------------------------------------------------------- */
int mat_wants_q8a(const borg_mat *m) { return m->dtype == DT_Q8_0 || m->dtype == DT_Q4_0 || m->dtype == DT_Q4_1; }

void mat_get_row(const borg_mat *m, int r, float *out) {
    if (m->layout == MAT_X8) {
        x8_get_row(m, r, out);
        return;
    }
    const uint8_t *row = (const uint8_t *)m->data + (size_t)r * m->row_bytes;
    const int n = m->cols;
    switch (m->dtype) {
    case DT_F32: memcpy(out, row, (size_t)n * 4); break;
    case DT_F16: for (int i = 0; i < n; i++) out[i] = fp16_to_fp32(((const uint16_t *)row)[i]); break;
    case DT_BF16: for (int i = 0; i < n; i++) out[i] = bf16_to_fp32(((const uint16_t *)row)[i]); break;
    case DT_Q8_0: {
        const block_q8_0 *b = (const block_q8_0 *)row;
        for (int i = 0; i < n / QK; i++) {
            const float d = fp16_to_fp32(b[i].d);
            for (int j = 0; j < QK; j++) out[i * QK + j] = d * (float)b[i].qs[j];
        }
        break;
    }
    case DT_Q4_0: {
        const block_q4_0 *b = (const block_q4_0 *)row;
        for (int i = 0; i < n / QK; i++) {
            const float d = fp16_to_fp32(b[i].d);
            for (int j = 0; j < QK / 2; j++) {
                out[i * QK + j] = d * (float)((int)(b[i].qs[j] & 0x0F) - 8);
                out[i * QK + j + QK / 2] = d * (float)((int)(b[i].qs[j] >> 4) - 8);
            }
        }
        break;
    }
    case DT_Q4_1: {
        const block_q4_1 *b = (const block_q4_1 *)row;
        for (int i = 0; i < n / QK; i++) {
            const float d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].m);
            for (int j = 0; j < QK / 2; j++) {
                out[i * QK + j] = d * (float)(b[i].qs[j] & 0x0F) + mn;
                out[i * QK + j + QK / 2] = d * (float)(b[i].qs[j] >> 4) + mn;
            }
        }
        break;
    }
    default: memset(out, 0, (size_t)n * 4);
    }
}

float mat_dot_row(const borg_mat *m, int r, const float *xf, const block_q8a *xq) {
    if (m->layout == MAT_X8) { /* the batched kernel on one row, so the number is the same */
        float v;
        x8_mul(m, r, r + 1, 1, xq, 0, &v, 1, r);
        return v;
    }
    const uint8_t *row = (const uint8_t *)m->data + (size_t)r * m->row_bytes;
    switch (m->dtype) {
    case DT_F32: return vec_dot_f32((const float *)row, xf, m->cols);
    case DT_F16: return vec_dot_f16_f32((const uint16_t *)row, xf, m->cols);
    case DT_BF16: return vec_dot_bf16_f32((const uint16_t *)row, xf, m->cols);
    case DT_Q8_0: return vec_dot_q8_0_q8a((const block_q8_0 *)row, xq, m->cols / QK);
    case DT_Q4_0: return vec_dot_q4_0_q8a((const block_q4_0 *)row, xq, m->cols / QK);
    case DT_Q4_1: return vec_dot_q4_1_q8a((const block_q4_1 *)row, xq, m->cols / QK);
    default: return 0.0f;
    }
}

/* ---- multi-token kernels --------------------------------------------------
 * Batched evaluation (prompt prefill, speculative verification) multiplies each weight
 * row with several activation rows. These kernels load and unpack every weight block
 * once for 4 tokens instead of once per token. */
#if defined(BORG_AVX2)
static void dot_q8_0_q8a_x4(const block_q8_0 *w, const block_q8a *const x[4], int nb, float out[4]) {
    /* Same summation order as vec_dot_q8_0_q8a (even blocks, odd blocks, then the tail in the
     * even accumulator), so batched and token-by-token evaluation give bit-identical results:
     * speculative decoding relies on that to reproduce plain greedy decoding exactly. */
    __m256 a[4][2];
    for (int c = 0; c < 4; c++) a[c][0] = a[c][1] = _mm256_setzero_ps();
    const int nb2 = nb & ~1;
    for (int i = 0; i < nb; i++) {
        const __m256i qw = _mm256_loadu_si256((const __m256i *)w[i].qs);
        const __m256i aw = _mm256_sign_epi8(qw, qw);
        const float dw = fp16_to_fp32(w[i].d);
        const int k = i < nb2 ? (i & 1) : 0;
        for (int c = 0; c < 4; c++) {
            const __m256i sx = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x[c][i].qs), qw);
            a[c][k] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[c][i].d), _mm256_cvtepi32_ps(dot_u8_i8(aw, sx)), a[c][k]);
        }
    }
    for (int c = 0; c < 4; c++) out[c] = hsum256(_mm256_add_ps(a[c][0], a[c][1]));
}

static void dot_q4_0_q8a_x4(const block_q4_0 *w, const block_q8a *const x[4], int nb, float out[4]) {
    const __m256i lowmask = _mm256_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    float corr[4] = {0, 0, 0, 0};
    for (int i = 0; i < nb; i++) {
        const __m128i raw = _mm_loadu_si128((const __m128i *)w[i].qs);
        const __m256i q = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), lowmask);
        const float dw = fp16_to_fp32(w[i].d);
#define STEP(c, acc)                                                                          \
    {                                                                                         \
        const __m256i qx = _mm256_loadu_si256((const __m256i *)x[c][i].qs);                   \
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[c][i].d), _mm256_cvtepi32_ps(dot_u8_i8(q, qx)), acc); \
        corr[c] += dw * x[c][i].s;                                                            \
    }
        STEP(0, acc0) STEP(1, acc1) STEP(2, acc2) STEP(3, acc3)
#undef STEP
    }
    out[0] = hsum256(acc0) - 8.0f * corr[0];
    out[1] = hsum256(acc1) - 8.0f * corr[1];
    out[2] = hsum256(acc2) - 8.0f * corr[2];
    out[3] = hsum256(acc3) - 8.0f * corr[3];
}

static void dot_q4_1_q8a_x4(const block_q4_1 *w, const block_q8a *const x[4], int nb, float out[4]) {
    const __m256i lowmask = _mm256_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    float corr[4] = {0, 0, 0, 0};
    for (int i = 0; i < nb; i++) {
        const __m128i raw = _mm_loadu_si128((const __m128i *)w[i].qs);
        const __m256i q = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), lowmask);
        const float dw = fp16_to_fp32(w[i].d), mw = fp16_to_fp32(w[i].m);
#define STEP(c, acc)                                                                          \
    {                                                                                         \
        const __m256i qx = _mm256_loadu_si256((const __m256i *)x[c][i].qs);                   \
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * x[c][i].d), _mm256_cvtepi32_ps(dot_u8_i8(q, qx)), acc); \
        corr[c] += mw * x[c][i].s;                                                            \
    }
        STEP(0, acc0) STEP(1, acc1) STEP(2, acc2) STEP(3, acc3)
#undef STEP
    }
    out[0] = hsum256(acc0) + corr[0];
    out[1] = hsum256(acc1) + corr[1];
    out[2] = hsum256(acc2) + corr[2];
    out[3] = hsum256(acc3) + corr[3];
}
#endif

#if defined(BORG_WASM)
/* WebAssembly versions: the weight block is widened to int16 once and reused for 4 tokens.
 * Each token's sum is formed exactly as in the 1-token kernels (one accumulator, block order). */
typedef struct { v128_t l0, h0, l1, h1; } wasm_w16;

static inline wasm_w16 wasm_widen(v128_t a, v128_t b) {
    wasm_w16 r = {wasm_i16x8_extend_low_i8x16(a), wasm_i16x8_extend_high_i8x16(a), wasm_i16x8_extend_low_i8x16(b),
                  wasm_i16x8_extend_high_i8x16(b)};
    return r;
}
static inline v128_t wasm_dot_w16(const wasm_w16 *w, const int8_t *qs) {
    const v128_t x0 = wasm_v128_load(qs), x1 = wasm_v128_load(qs + 16);
    const v128_t d0 = wasm_i32x4_add(wasm_i32x4_dot_i16x8(w->l0, wasm_i16x8_extend_low_i8x16(x0)),
                                     wasm_i32x4_dot_i16x8(w->h0, wasm_i16x8_extend_high_i8x16(x0)));
    const v128_t d1 = wasm_i32x4_add(wasm_i32x4_dot_i16x8(w->l1, wasm_i16x8_extend_low_i8x16(x1)),
                                     wasm_i32x4_dot_i16x8(w->h1, wasm_i16x8_extend_high_i8x16(x1)));
    return wasm_i32x4_add(d0, d1);
}

static void dot_q8_0_q8a_x4(const block_q8_0 *w, const block_q8a *const x[4], int nb, float out[4]) {
    v128_t acc[4] = {wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f)};
    for (int i = 0; i < nb; i++) {
        const wasm_w16 q = wasm_widen(wasm_v128_load(w[i].qs), wasm_v128_load(w[i].qs + 16));
        const float dw = fp16_to_fp32(w[i].d);
        for (int c = 0; c < 4; c++)
            acc[c] = wasm_f32x4_add(acc[c], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_dot_w16(&q, x[c][i].qs)),
                                                           wasm_f32x4_splat(dw * x[c][i].d)));
    }
    for (int c = 0; c < 4; c++) out[c] = wasm_hsum_f32(acc[c]);
}

static void dot_q4_0_q8a_x4(const block_q4_0 *w, const block_q8a *const x[4], int nb, float out[4]) {
    const v128_t m4 = wasm_i8x16_splat(0x0F), s8 = wasm_i8x16_splat(8);
    v128_t acc[4] = {wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f)};
    for (int i = 0; i < nb; i++) {
        const v128_t raw = wasm_v128_load(w[i].qs);
        const wasm_w16 q = wasm_widen(wasm_i8x16_sub(wasm_v128_and(raw, m4), s8), wasm_i8x16_sub(wasm_u8x16_shr(raw, 4), s8));
        const float dw = fp16_to_fp32(w[i].d);
        for (int c = 0; c < 4; c++)
            acc[c] = wasm_f32x4_add(acc[c], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_dot_w16(&q, x[c][i].qs)),
                                                           wasm_f32x4_splat(dw * x[c][i].d)));
    }
    for (int c = 0; c < 4; c++) out[c] = wasm_hsum_f32(acc[c]);
}

static void dot_q4_1_q8a_x4(const block_q4_1 *w, const block_q8a *const x[4], int nb, float out[4]) {
    const v128_t m4 = wasm_i8x16_splat(0x0F);
    v128_t acc[4] = {wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f), wasm_f32x4_splat(0.0f)};
    float corr[4] = {0, 0, 0, 0};
    for (int i = 0; i < nb; i++) {
        const v128_t raw = wasm_v128_load(w[i].qs);
        const wasm_w16 q = wasm_widen(wasm_v128_and(raw, m4), wasm_u8x16_shr(raw, 4));
        const float dw = fp16_to_fp32(w[i].d), mw = fp16_to_fp32(w[i].m);
        for (int c = 0; c < 4; c++) {
            acc[c] = wasm_f32x4_add(acc[c], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_dot_w16(&q, x[c][i].qs)),
                                                           wasm_f32x4_splat(dw * x[c][i].d)));
            corr[c] += mw * x[c][i].s;
        }
    }
    for (int c = 0; c < 4; c++) out[c] = wasm_hsum_f32(acc[c]) + corr[c];
}
#endif

void mat_mul_rows_off(const borg_mat *m, int r0, int r1, int T, const float *x, int ldx, const block_q8a *xq,
                      int ldxq, float *out, int ldo, int ro) {
    if (r0 >= r1 || T <= 0) return;
    if (m->layout == MAT_X8) {
        x8_mul(m, r0, r1, T, xq, ldxq, out, ldo, ro);
        return;
    }
#if defined(BORG_AVX2) || defined(BORG_WASM)
    if ((m->dtype == DT_Q8_0 || m->dtype == DT_Q4_0 || m->dtype == DT_Q4_1) && T >= 4) {
        /* Plain layout (BORG_NO_REPACK=1): tile 16 weight rows x 4 tokens, so the rows stay in L1
         * while every group of 4 activation rows is read once per tile instead of once per weight
         * row. Each dot product is computed exactly as by mat_dot_row. */
        enum { RB = 16 };
        const int nb = m->cols / QK;
        for (int rb = r0; rb < r1; rb += RB) {
            const int re = rb + RB < r1 ? rb + RB : r1;
            int t = 0;
            for (; t + 4 <= T; t += 4) {
                const block_q8a *xs[4] = {xq + (size_t)t * ldxq, xq + (size_t)(t + 1) * ldxq,
                                          xq + (size_t)(t + 2) * ldxq, xq + (size_t)(t + 3) * ldxq};
                for (int r = rb; r < re; r++) {
                    const uint8_t *row = (const uint8_t *)m->data + (size_t)r * m->row_bytes;
                    float o[4];
                    if (m->dtype == DT_Q8_0) dot_q8_0_q8a_x4((const block_q8_0 *)row, xs, nb, o);
                    else if (m->dtype == DT_Q4_0) dot_q4_0_q8a_x4((const block_q4_0 *)row, xs, nb, o);
                    else dot_q4_1_q8a_x4((const block_q4_1 *)row, xs, nb, o);
                    for (int c = 0; c < 4; c++) out[(size_t)(t + c) * ldo + (size_t)(r - ro)] = o[c];
                }
            }
            for (; t < T; t++)
                for (int r = rb; r < re; r++)
                    out[(size_t)t * ldo + (size_t)(r - ro)] = mat_dot_row(m, r, NULL, xq + (size_t)t * ldxq);
        }
        return;
    }
#endif
    for (int r = r0; r < r1; r++)
        for (int t = 0; t < T; t++)
            out[(size_t)t * ldo + (size_t)(r - ro)] =
                mat_dot_row(m, r, x ? x + (size_t)t * ldx : NULL, xq ? xq + (size_t)t * ldxq : NULL);
}

void mat_mul_rows(const borg_mat *m, int r0, int r1, int T, const float *x, int ldx, const block_q8a *xq,
                  int ldxq, float *out, int ldo) {
    mat_mul_rows_off(m, r0, r1, T, x, ldx, xq, ldxq, out, ldo, 0);
}
