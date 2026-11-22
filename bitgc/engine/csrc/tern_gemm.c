/*
 * tern_gemm.c — 삼진 가중치({-1,0,1}) × int8 활성 행렬곱 (BitNet b1.58 식, 하이브리드 1B 엔진용)
 *
 * 가중치는 64개씩 두 비트마스크(pos, neg)로 저장한다 → 2 bit/weight (1.09B 파라미터 = 273MB).
 * 내적 Σ x_k·w_k 는 곱셈 없는 덧셈·뺄셈이지만, x86 에서는 AVX-512 VNNI(vpdpbusd)가
 * 64개 int8 을 한 명령에 누산하므로 비트마스크를 바이트 {-1,0,1} 로 펼쳐 넣는다:
 *     w_bytes = movm(neg) − movm(pos)          (movm: 비트 → 0xFF/0x00)
 *     vpdpbusd 는 u8×s8 이라 x 를 x+128 (= x ^ 0x80) 로 바꾸고 128·Σw 를 뺀다.
 * 누산은 int32 로 **정확한 정수** — torch `_TernaryAct8Int8` 의 torch._int_mm 과 같은 값.
 *
 *   y[m][n] = (float)(Σ_k xq[m][k]·wq[n][k]) · γ[n] / s[m]
 *   xq = round(x·s).clamp(−128,127),  s = 127 / max|x| (행별),  round = 짝수 반올림(torch.round)
 *
 * 빌드 (WSL):  gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC \
 *                  -o engine_int/libtern.so engine_int/tern_gemm.c
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#define TERN_VNNI 1
#else
#define TERN_VNNI 0
#endif

int tern_has_vnni(void) { return TERN_VNNI; }

int tern_set_threads(int n)
{
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
    return omp_get_max_threads();
#else
    (void)n;
    return 1;
#endif
}

/* 행별 absmax int8 양자화. torch: s = 127/amax.clamp(1e-5); round(x*s).clamp(-128,127). */
void tern_quant_rows(const float *x, int M, int K, int8_t *xq, float *s_out)
{
    #pragma omp parallel for schedule(static) if (M > 4)
    for (int m = 0; m < M; m++) {
        const float *r = x + (size_t)m * K;
        float amax = 0.f;
        #pragma omp simd reduction(max:amax)
        for (int k = 0; k < K; k++) {
            float a = fabsf(r[k]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-5f) amax = 1e-5f;
        float s = 127.0f / amax;
        s_out[m] = s;
        int8_t *q = xq + (size_t)m * K;
        #pragma omp simd
        for (int k = 0; k < K; k++) {
            float v = nearbyintf(r[k] * s);          /* 기본 반올림 모드 = 짝수 반올림 */
            if (v > 127.f) v = 127.f;
            if (v < -128.f) v = -128.f;
            q[k] = (int8_t)v;
        }
    }
}

/* 가중치 행 n 의 Σw (= popcnt(pos) − popcnt(neg)) — u8 보정용. export 때 미리 계산해 넘긴다. */

#if TERN_VNNI
/* 4행 × 6토큰 마이크로커널. 누산기 24개를 **이름 붙은 지역 변수**로 둔다 — 배열(acc[6][4])로 두면
 * GCC 가 매 반복 스택에 내렸다 올려(실측 504 GMAC/s) 레지스터 블로킹이 무의미해졌다. */
#define DP(a, x, w) a = _mm512_dpbusd_epi32(a, x, w)
#define ROW(t, a0, a1, a2, a3) { __m512i xv = _mm512_loadu_si512(xp + (size_t)(t) * K + o); \
    DP(a0, xv, w0); DP(a1, xv, w1); DP(a2, xv, w2); DP(a3, xv, w3); }
#define OUT(t, a0, a1, a2, a3) { \
    Y[(size_t)(t) * N + 0] = (float)(_mm512_reduce_add_epi32(a0) - 128 * ws[0]) * g[0] / s[t]; \
    Y[(size_t)(t) * N + 1] = (float)(_mm512_reduce_add_epi32(a1) - 128 * ws[1]) * g[1] / s[t]; \
    Y[(size_t)(t) * N + 2] = (float)(_mm512_reduce_add_epi32(a2) - 128 * ws[2]) * g[2] / s[t]; \
    Y[(size_t)(t) * N + 3] = (float)(_mm512_reduce_add_epi32(a3) - 128 * ws[3]) * g[3] / s[t]; }

static void block4x6(const uint8_t *xp, int K, const int8_t *wb, const float *s,
                     const int32_t *ws, const float *g, int N, float *Y)
{
    __m512i a00 = _mm512_setzero_si512(), a01 = a00, a02 = a00, a03 = a00;
    __m512i a10 = a00, a11 = a00, a12 = a00, a13 = a00, a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    __m512i a30 = a00, a31 = a00, a32 = a00, a33 = a00, a40 = a00, a41 = a00, a42 = a00, a43 = a00;
    __m512i a50 = a00, a51 = a00, a52 = a00, a53 = a00;
    for (int o = 0; o < K; o += 64) {
        __m512i w0 = _mm512_loadu_si512(wb + o), w1 = _mm512_loadu_si512(wb + (size_t)K + o);
        __m512i w2 = _mm512_loadu_si512(wb + (size_t)2 * K + o), w3 = _mm512_loadu_si512(wb + (size_t)3 * K + o);
        ROW(0, a00, a01, a02, a03) ROW(1, a10, a11, a12, a13) ROW(2, a20, a21, a22, a23)
        ROW(3, a30, a31, a32, a33) ROW(4, a40, a41, a42, a43) ROW(5, a50, a51, a52, a53)
    }
    OUT(0, a00, a01, a02, a03) OUT(1, a10, a11, a12, a13) OUT(2, a20, a21, a22, a23)
    OUT(3, a30, a31, a32, a33) OUT(4, a40, a41, a42, a43) OUT(5, a50, a51, a52, a53)
}
#endif

/*
 * xq: M×K int8,  s: M,  W: N×(K/64)×2 uint64 (pos, neg),  wsum: N int32,  g: N float
 * Y:  M×N float (행 우선)
 */
void tern_gemm(const int8_t *xq, const float *s, int M, int K,
               const uint64_t *W, const int32_t *wsum, const float *g, int N, float *Y)
{
    const int KB = K / 64;
#if TERN_VNNI
    /* x+128 (u8) 로 한 번 바꿔 둔다 */
    uint8_t *xu = (uint8_t *)aligned_alloc(64, ((size_t)M * K + 63) / 64 * 64);
    for (size_t i = 0; i < (size_t)M * K; i++) xu[i] = (uint8_t)(xq[i] ^ 0x80);
    const int MT = 8;
    /* ── 큰 M (prefill): 4행 × 6토큰 레지스터 블록. 가중치 4행을 바이트로 한 번 펼쳐 두고(스레드별 버퍼)
     *    토큰 6개씩 돌며 24개 누산기에 vpdpbusd — x·w 를 레지스터에서 재사용해 계산 한계로 만든다.
     *    (행마다 토큰 8개씩 펼치던 방식은 x 를 N 번 다시 읽어 0.44 TMAC/s 에 그쳤다) */
    int m_big = (M >= 6 && N % 4 == 0) ? (M / 6) * 6 : 0;
    if (m_big) {
        #pragma omp parallel
        {
            int8_t *wb = (int8_t *)aligned_alloc(64, (size_t)4 * K);
            #pragma omp for schedule(static)
            for (int n0 = 0; n0 < N; n0 += 4) {
                for (int r = 0; r < 4; r++) {
                    const uint64_t *wr = W + (size_t)(n0 + r) * KB * 2;
                    for (int kb = 0; kb < KB; kb++)
                        _mm512_storeu_si512(wb + (size_t)r * K + kb * 64,
                            _mm512_sub_epi8(_mm512_movm_epi8((__mmask64)wr[2 * kb + 1]),
                                            _mm512_movm_epi8((__mmask64)wr[2 * kb])));
                }
                for (int m0 = 0; m0 < m_big; m0 += 6)
                    block4x6(xu + (size_t)m0 * K, K, wb, s + m0, wsum + n0, g + n0, N,
                             Y + (size_t)m0 * N + n0);
            }
            free(wb);
        }
    }
    if (m_big == M) { free(xu); return; }
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint64_t *wr = W + (size_t)n * KB * 2;
        const float gn = g[n];
        const int32_t corr = 128 * wsum[n];
        for (int m0 = m_big; m0 < M; m0 += MT) {
            int mt = M - m0 < MT ? M - m0 : MT;
            __m512i acc[8];
            for (int t = 0; t < 8; t++) acc[t] = _mm512_setzero_si512();
            for (int kb = 0; kb < KB; kb++) {
                __m512i wv = _mm512_sub_epi8(_mm512_movm_epi8((__mmask64)wr[2 * kb + 1]),
                                             _mm512_movm_epi8((__mmask64)wr[2 * kb]));
                const uint8_t *xp = xu + (size_t)m0 * K + (size_t)kb * 64;
                #pragma GCC unroll 8
                for (int t = 0; t < 8; t++) {
                    if (t < mt)
                        acc[t] = _mm512_dpbusd_epi32(acc[t], _mm512_loadu_si512(xp + (size_t)t * K), wv);
                }
            }
            for (int t = 0; t < mt; t++) {
                int32_t z = _mm512_reduce_add_epi32(acc[t]) - corr;
                Y[(size_t)(m0 + t) * N + n] = (float)z * gn / s[m0 + t];
            }
        }
    }
    free(xu);
#else
    (void)wsum;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint64_t *wr = W + (size_t)n * KB * 2;
        for (int m = 0; m < M; m++) {
            const int8_t *x = xq + (size_t)m * K;
            int32_t z = 0;
            for (int kb = 0; kb < KB; kb++) {
                uint64_t p = wr[2 * kb], q = wr[2 * kb + 1];
                for (int j = 0; j < 64; j++) {
                    int32_t v = x[kb * 64 + j];
                    if ((p >> j) & 1) z += v;
                    else if ((q >> j) & 1) z -= v;
                }
            }
            Y[(size_t)m * N + n] = (float)z * g[n] / s[m];
        }
    }
#endif
}

/* ── 0 열 건너뛰기 (2026-11-03, E1c 의 ReLU² FFN 용) ──────────────────────────────────────────────
 * fc2 입력은 int8 양자화 뒤 ~96% 가 정확히 0 이다. 행 단위(출력 n 마다 K 비트) 저장으로는 0 인 입력이 있어도
 * 같은 64비트 낱말을 읽어야 하므로 아무것도 안 줄어든다. 그래서 가중치를 **열 단위**로 둔다:
 *     Wc[k][w] = {pos, neg} 64비트 (출력 n = 64w..64w+63), = pack_ternary(wq.T)
 * 입력 k 가 0 이면 그 열(N 비트 ×2)을 읽지 않는다. 0 아닌 입력마다 ±xq[k] 를 int32 누산기에 마스크 덧셈·뺄셈.
 * 정수 합이 행 단위 커널과 정확히 같으므로 출력도 같은 식 (float)z·γ[n]/s[m] 로 같다.
 * 스레드는 출력 256개(누산기 16개 = 열 한 칸 64바이트 = 캐시 줄 하나)씩 나눈다. N % 256 == 0 이어야 한다. */
static long long g_nz = 0, g_tot = 0;
void tern_sparse_stats(long long *out, int reset)
{
    out[0] = g_nz; out[1] = g_tot;
    if (reset) g_nz = g_tot = 0;
}

void tern_gemm_cols(const int8_t *xq, const float *s, int M, int K,
                    const uint64_t *Wc, const float *g, int N, float *Y)
{
    const int NW = N / 64, NC = N / 256;
    int *cnt = (int *)malloc(sizeof(int) * M);
    int *idx = (int *)malloc(sizeof(int) * (size_t)M * K);
    long long nz = 0;
    #pragma omp parallel for schedule(static) reduction(+:nz) if (M > 4)
    for (int m = 0; m < M; m++) {
        const int8_t *x = xq + (size_t)m * K;
        int *ix = idx + (size_t)m * K, c = 0;
        for (int k = 0; k < K; k++) if (x[k]) ix[c++] = k;
        cnt[m] = c;
        nz += c;
    }
    g_nz += nz; g_tot += (long long)M * K;
#if TERN_VNNI
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m++) {
        for (int c = 0; c < NC; c++) {
            __m512i acc[16];
            for (int j = 0; j < 16; j++) acc[j] = _mm512_setzero_si512();
            const int8_t *x = xq + (size_t)m * K;
            const int *ix = idx + (size_t)m * K;
            for (int i = 0; i < cnt[m]; i++) {
                const int k = ix[i];
                const __m512i bv = _mm512_set1_epi32((int)x[k]);
                const uint64_t *wp = Wc + ((size_t)k * NW + (size_t)c * 4) * 2;
                #pragma GCC unroll 4
                for (int w = 0; w < 4; w++) {
                    const uint64_t p = wp[2 * w], q = wp[2 * w + 1];
                    #pragma GCC unroll 4
                    for (int h = 0; h < 4; h++) {
                        const int j = w * 4 + h;
                        acc[j] = _mm512_mask_add_epi32(acc[j], (__mmask16)(p >> (16 * h)), acc[j], bv);
                        acc[j] = _mm512_mask_sub_epi32(acc[j], (__mmask16)(q >> (16 * h)), acc[j], bv);
                    }
                }
            }
            float *y = Y + (size_t)m * N + (size_t)c * 256;
            const float *gc = g + (size_t)c * 256;
            const float sm = s[m];
            for (int j = 0; j < 16; j++) {
                int32_t z[16];
                _mm512_storeu_si512(z, acc[j]);
                for (int e = 0; e < 16; e++) y[j * 16 + e] = (float)z[e] * gc[j * 16 + e] / sm;
            }
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t *x = xq + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            int32_t z = 0;
            for (int i = 0; i < cnt[m]; i++) {
                const int k = idx[(size_t)m * K + i];
                const uint64_t *wp = Wc + ((size_t)k * NW + n / 64) * 2;
                if ((wp[0] >> (n % 64)) & 1) z += x[k];
                else if ((wp[1] >> (n % 64)) & 1) z -= x[k];
            }
            Y[(size_t)m * N + n] = (float)z * g[n] / s[m];
        }
    }
    (void)NC;
#endif
    free(idx);
    free(cnt);
}
