/*
 * gen_gemm.c — 범용 엔진(gen_step.c)용 가중치 형식별 행렬곱 (논문 비교군 속도·전력 측정, §v56).
 *
 * 모든 형식이 **같은 구조**로 짜여 있다 — 비교에서 달라지는 것은 가중치 형식(=읽는 바이트·연산 종류)뿐이어야 한다.
 *   생성(M=1, GEMV)  : 출력 행 N 을 OpenMP 로 나누고 행마다 K 를 SIMD 내적 (메모리 대역 한계)
 *   prefill(M≥6)     : 가중치 4행 × 토큰 6개 레지스터 블록, 누산기 24개 이름 붙은 변수 (tern_gemm.c 와 같은 꼴)
 *
 *   bf16 : W bf16(N×K) · 활성 bf16(짝수 반올림) → fp32 누산  (_mm512_dpbf16_ps, 32쌍/명령)  — torch bf16 추론과 같은 정밀도
 *   f32  : W fp32 · 활성 fp32 → fp32 FMA (16/명령)
 *   i8   : W int8(N×K, 행별 스케일) · 활성 int8(토큰별 absmax) → int32 정확 누산 (vpdpbusd, 64쌍/명령)
 *          y = (Σ xq·wq − 128·Σwq) · sw[n] / sx[m]   (삼진 커널과 같은 u8 보정 · 같은 출력 식)
 * 삼진은 tern_gemm.c 를 그대로 쓴다.
 *
 * 빌드: build_gen.sh (gcc -O3 -march=native -fopenmp -ffp-contract=off)
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>

#if !defined(__AVX512F__) || !defined(__AVX512BF16__) || !defined(__AVX512VNNI__)
#error "gen_gemm.c 는 AVX-512 F/BF16/VNNI 가 필요하다 (-march=native, Zen4 · Sapphire Rapids 이상)"
#endif

/* ── bf16 ──────────────────────────────────────────────────────────────── */

static inline uint16_t f2bf(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);         /* 짝수 반올림 (torch 의 .to(bfloat16) 와 같다) */
    return (uint16_t)(u >> 16);
}

void gen_to_bf16(const float *x, size_t n, uint16_t *y)
{
    #pragma omp parallel for schedule(static) if (n > 65536)
    for (size_t i = 0; i < n; i++) y[i] = f2bf(x[i]);
}

#define BH(p) ((__m512bh)_mm512_loadu_si512((const void *)(p)))
#define DB(a, x, w) a = _mm512_dpbf16_ps(a, x, w)
#define BROW(t, a0, a1, a2, a3) { __m512bh xv = BH(xb + (size_t)(t) * K + o); \
    DB(a0, xv, w0); DB(a1, xv, w1); DB(a2, xv, w2); DB(a3, xv, w3); }
#define FOUT(t, a0, a1, a2, a3) { float *yr = Y + (size_t)(t) * N; \
    yr[0] = _mm512_reduce_add_ps(a0) + (bias ? bias[0] : 0.f); yr[1] = _mm512_reduce_add_ps(a1) + (bias ? bias[1] : 0.f); \
    yr[2] = _mm512_reduce_add_ps(a2) + (bias ? bias[2] : 0.f); yr[3] = _mm512_reduce_add_ps(a3) + (bias ? bias[3] : 0.f); }

static void bblock4x6(const uint16_t *xb, int K, const uint16_t *wb, const float *bias, int N, float *Y)
{
    __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512 a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    __m512 a40 = a00, a41 = a00, a42 = a00, a43 = a00, a50 = a00, a51 = a00, a52 = a00, a53 = a00;
    for (int o = 0; o < K; o += 32) {
        __m512bh w0 = BH(wb + o), w1 = BH(wb + (size_t)K + o), w2 = BH(wb + (size_t)2 * K + o), w3 = BH(wb + (size_t)3 * K + o);
        BROW(0, a00, a01, a02, a03) BROW(1, a10, a11, a12, a13) BROW(2, a20, a21, a22, a23)
        BROW(3, a30, a31, a32, a33) BROW(4, a40, a41, a42, a43) BROW(5, a50, a51, a52, a53)
    }
    FOUT(0, a00, a01, a02, a03) FOUT(1, a10, a11, a12, a13) FOUT(2, a20, a21, a22, a23)
    FOUT(3, a30, a31, a32, a33) FOUT(4, a40, a41, a42, a43) FOUT(5, a50, a51, a52, a53)
}

/* X: M×K fp32 (bf16 로 바꿔 쓴다), W: N×K bf16, bias: N 또는 NULL, Y: M×N.  K % 32 == 0. */
void gen_gemm_bf16(const float *X, int M, int K, const uint16_t *W, const float *bias, int N, float *Y)
{
    uint16_t *xb = (uint16_t *)aligned_alloc(64, ((size_t)M * K * 2 + 63) / 64 * 64);
    gen_to_bf16(X, (size_t)M * K, xb);
    int m_big = (M >= 6 && N % 4 == 0) ? (M / 6) * 6 : 0;
    if (m_big) {
        #pragma omp parallel for schedule(static)
        for (int n0 = 0; n0 < N; n0 += 4)
            for (int m0 = 0; m0 < m_big; m0 += 6)
                bblock4x6(xb + (size_t)m0 * K, K, W + (size_t)n0 * K, bias ? bias + n0 : NULL, N,
                          Y + (size_t)m0 * N + n0);
    }
    if (m_big < M) {
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const uint16_t *wr = W + (size_t)n * K;
            for (int m = m_big; m < M; m++) {
                const uint16_t *xr = xb + (size_t)m * K;
                __m512 a0 = _mm512_setzero_ps(), a1 = a0;
                int o = 0;
                for (; o + 64 <= K; o += 64) { DB(a0, BH(xr + o), BH(wr + o)); DB(a1, BH(xr + o + 32), BH(wr + o + 32)); }
                for (; o < K; o += 32) DB(a0, BH(xr + o), BH(wr + o));
                Y[(size_t)m * N + n] = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) + (bias ? bias[n] : 0.f);
            }
        }
    }
    free(xb);
}

/* ── fp32 ──────────────────────────────────────────────────────────────── */

#define FM(a, x, w) a = _mm512_fmadd_ps(x, w, a)
#define FROW(t, a0, a1, a2, a3) { __m512 xv = _mm512_loadu_ps(X + (size_t)(t) * K + o); \
    FM(a0, xv, w0); FM(a1, xv, w1); FM(a2, xv, w2); FM(a3, xv, w3); }

static void fblock4x6(const float *X, int K, const float *wb, const float *bias, int N, float *Y)
{
    __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512 a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    __m512 a40 = a00, a41 = a00, a42 = a00, a43 = a00, a50 = a00, a51 = a00, a52 = a00, a53 = a00;
    for (int o = 0; o < K; o += 16) {
        __m512 w0 = _mm512_loadu_ps(wb + o), w1 = _mm512_loadu_ps(wb + (size_t)K + o);
        __m512 w2 = _mm512_loadu_ps(wb + (size_t)2 * K + o), w3 = _mm512_loadu_ps(wb + (size_t)3 * K + o);
        FROW(0, a00, a01, a02, a03) FROW(1, a10, a11, a12, a13) FROW(2, a20, a21, a22, a23)
        FROW(3, a30, a31, a32, a33) FROW(4, a40, a41, a42, a43) FROW(5, a50, a51, a52, a53)
    }
    FOUT(0, a00, a01, a02, a03) FOUT(1, a10, a11, a12, a13) FOUT(2, a20, a21, a22, a23)
    FOUT(3, a30, a31, a32, a33) FOUT(4, a40, a41, a42, a43) FOUT(5, a50, a51, a52, a53)
}

/* X: M×K, W: N×K fp32.  K % 16 == 0. */
void gen_gemm_f32(const float *X, int M, int K, const float *W, const float *bias, int N, float *Y)
{
    int m_big = (M >= 6 && N % 4 == 0) ? (M / 6) * 6 : 0;
    if (m_big) {
        #pragma omp parallel for schedule(static)
        for (int n0 = 0; n0 < N; n0 += 4)
            for (int m0 = 0; m0 < m_big; m0 += 6)
                fblock4x6(X + (size_t)m0 * K, K, W + (size_t)n0 * K, bias ? bias + n0 : NULL, N,
                          Y + (size_t)m0 * N + n0);
    }
    if (m_big < M) {
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const float *wr = W + (size_t)n * K;
            for (int m = m_big; m < M; m++) {
                const float *xr = X + (size_t)m * K;
                __m512 a0 = _mm512_setzero_ps(), a1 = a0;
                int o = 0;
                for (; o + 32 <= K; o += 32) {
                    FM(a0, _mm512_loadu_ps(xr + o), _mm512_loadu_ps(wr + o));
                    FM(a1, _mm512_loadu_ps(xr + o + 16), _mm512_loadu_ps(wr + o + 16));
                }
                for (; o < K; o += 16) FM(a0, _mm512_loadu_ps(xr + o), _mm512_loadu_ps(wr + o));
                Y[(size_t)m * N + n] = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) + (bias ? bias[n] : 0.f);
            }
        }
    }
}

/* ── int8 × int8 (W8A8) ───────────────────────────────────────────────── */

#define DP(a, x, w) a = _mm512_dpbusd_epi32(a, x, w)
#define IROW(t, a0, a1, a2, a3) { __m512i xv = _mm512_loadu_si512(xu + (size_t)(t) * K + o); \
    DP(a0, xv, w0); DP(a1, xv, w1); DP(a2, xv, w2); DP(a3, xv, w3); }
#define IOUT1(t, j, a) Y[(size_t)(t) * N + j] = (float)(_mm512_reduce_add_epi32(a) - 128 * ws[j]) * g[j] / s[t] + (bias ? bias[j] : 0.f);
#define IOUT(t, a0, a1, a2, a3) { IOUT1(t, 0, a0) IOUT1(t, 1, a1) IOUT1(t, 2, a2) IOUT1(t, 3, a3) }

static void iblock4x6(const uint8_t *xu, int K, const int8_t *wb, const float *s, const int32_t *ws,
                      const float *g, const float *bias, int N, float *Y)
{
    __m512i a00 = _mm512_setzero_si512(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512i a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    __m512i a40 = a00, a41 = a00, a42 = a00, a43 = a00, a50 = a00, a51 = a00, a52 = a00, a53 = a00;
    for (int o = 0; o < K; o += 64) {
        __m512i w0 = _mm512_loadu_si512(wb + o), w1 = _mm512_loadu_si512(wb + (size_t)K + o);
        __m512i w2 = _mm512_loadu_si512(wb + (size_t)2 * K + o), w3 = _mm512_loadu_si512(wb + (size_t)3 * K + o);
        IROW(0, a00, a01, a02, a03) IROW(1, a10, a11, a12, a13) IROW(2, a20, a21, a22, a23)
        IROW(3, a30, a31, a32, a33) IROW(4, a40, a41, a42, a43) IROW(5, a50, a51, a52, a53)
    }
    IOUT(0, a00, a01, a02, a03) IOUT(1, a10, a11, a12, a13) IOUT(2, a20, a21, a22, a23)
    IOUT(3, a30, a31, a32, a33) IOUT(4, a40, a41, a42, a43) IOUT(5, a50, a51, a52, a53)
}

/* xq: M×K int8 (tern_quant_rows 결과), s: M, W: N×K int8, wsum: N, g: N (가중치 스케일), bias: N|NULL.  K % 64 == 0. */
void gen_gemm_i8(const int8_t *xq, const float *s, int M, int K, const int8_t *W, const int32_t *wsum,
                 const float *g, const float *bias, int N, float *Y)
{
    uint8_t *xu = (uint8_t *)aligned_alloc(64, ((size_t)M * K + 63) / 64 * 64);
    for (size_t i = 0; i < (size_t)M * K; i++) xu[i] = (uint8_t)(xq[i] ^ 0x80);
    int m_big = (M >= 6 && N % 4 == 0) ? (M / 6) * 6 : 0;
    if (m_big) {
        #pragma omp parallel for schedule(static)
        for (int n0 = 0; n0 < N; n0 += 4)
            for (int m0 = 0; m0 < m_big; m0 += 6)
                iblock4x6(xu + (size_t)m0 * K, K, W + (size_t)n0 * K, s + m0, wsum + n0, g + n0,
                          bias ? bias + n0 : NULL, N, Y + (size_t)m0 * N + n0);
    }
    if (m_big < M) {
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const int8_t *wr = W + (size_t)n * K;
            for (int m = m_big; m < M; m++) {
                const uint8_t *xr = xu + (size_t)m * K;
                __m512i a0 = _mm512_setzero_si512(), a1 = a0;
                int o = 0;
                for (; o + 128 <= K; o += 128) {
                    DP(a0, _mm512_loadu_si512(xr + o), _mm512_loadu_si512(wr + o));
                    DP(a1, _mm512_loadu_si512(xr + o + 64), _mm512_loadu_si512(wr + o + 64));
                }
                for (; o < K; o += 64) DP(a0, _mm512_loadu_si512(xr + o), _mm512_loadu_si512(wr + o));
                int32_t z = _mm512_reduce_add_epi32(_mm512_add_epi32(a0, a1)) - 128 * wsum[n];
                Y[(size_t)m * N + n] = (float)z * g[n] / s[m] + (bias ? bias[n] : 0.f);
            }
        }
    }
    free(xu);
}

/* ── 원소별 보조 ──────────────────────────────────────────────────────── */

/* GELU (erf, torch F.gelu 기본값) 제자리 */
void gen_gelu(float *x, size_t n)
{
    #pragma omp parallel for schedule(static) if (n > 65536)
    for (size_t i = 0; i < n; i++) x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
}
