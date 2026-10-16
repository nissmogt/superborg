/* Numeric kernels: float16 conversion, activation quantization, dot products.
 *
 * Each kernel has a portable scalar version. SIMD versions are compiled in when the
 * target supports them: AVX2 (+FMA, F16C), AVX-512 VNNI on x86, NEON (+dotprod) on ARM.
 */
#include "internal.h"

#include <math.h>
#include <string.h>

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

const char *borg_simd(void) {
#if defined(BORG_VNNI)
    return "avx512vnni";
#elif defined(BORG_AVX2)
    return "avx2";
#elif defined(BORG_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return "neon-dotprod";
#elif defined(BORG_NEON)
    return "neon";
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

/* ---- matrix helpers ------------------------------------------------------- */
int mat_wants_q8a(const borg_mat *m) { return m->dtype == DT_Q8_0 || m->dtype == DT_Q4_0 || m->dtype == DT_Q4_1; }

void mat_get_row(const borg_mat *m, int r, float *out) {
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

void mat_mul_rows(const borg_mat *m, int r0, int r1, int T, const float *x, int ldx, const block_q8a *xq,
                  int ldxq, float *out, int ldo) {
#if defined(BORG_AVX2)
    if ((m->dtype == DT_Q8_0 || m->dtype == DT_Q4_0 || m->dtype == DT_Q4_1) && T >= 4) {
        const int nb = m->cols / QK;
        for (int r = r0; r < r1; r++) {
            const uint8_t *row = (const uint8_t *)m->data + (size_t)r * m->row_bytes;
            int t = 0;
            for (; t + 4 <= T; t += 4) {
                const block_q8a *xs[4] = {xq + (size_t)t * ldxq, xq + (size_t)(t + 1) * ldxq,
                                          xq + (size_t)(t + 2) * ldxq, xq + (size_t)(t + 3) * ldxq};
                float o[4];
                if (m->dtype == DT_Q8_0) dot_q8_0_q8a_x4((const block_q8_0 *)row, xs, nb, o);
                else if (m->dtype == DT_Q4_0) dot_q4_0_q8a_x4((const block_q4_0 *)row, xs, nb, o);
                else dot_q4_1_q8a_x4((const block_q4_1 *)row, xs, nb, o);
                for (int c = 0; c < 4; c++) out[(size_t)(t + c) * ldo + r] = o[c];
            }
            for (; t < T; t++) out[(size_t)t * ldo + r] = mat_dot_row(m, r, NULL, xq + (size_t)t * ldxq);
        }
        return;
    }
#endif
    for (int r = r0; r < r1; r++)
        for (int t = 0; t < T; t++)
            out[(size_t)t * ldo + r] = mat_dot_row(m, r, x ? x + (size_t)t * ldx : NULL, xq ? xq + (size_t)t * ldxq : NULL);
}
