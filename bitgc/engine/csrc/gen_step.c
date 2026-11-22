/*
 * gen_step.c — 범용 1B 엔진의 한 토큰 진행 (논문 비교군 속도·전력 측정, §v56). hyb_step.c 를 일반화한 것.
 *
 * 같은 코드 경로가 네 모델을 돌린다 — 다른 것은 층 구성 표와 가중치 형식뿐:
 *   우리 하이브리드 (삼진 × int8)          gconv/attention · hardtanh · pre_o/pre_fc2 LN · 잔차 스케일 · bias 없음
 *   (B) float 하이브리드 (bf16 | fp32)     위와 같은 배선
 *   (A) 표준 Transformer (bf16 | fp32)     전 층 attention · GELU · pre_o/pre_fc2 없음 · 스케일 없음 · bias 있음
 *   (A) PTQ w8a8 (int8 × int8)            (A) 배선 + int8 가중치
 *   BitNet 계열 E1·E1b·E1c (삼진 × int8)   표준 배선 + BitLinear: 선형층마다 양자화 직전 정규화(pre_a·pre_o·pre_f1·pre_fc2),
 *                                          gconv/attention · GELU | ReLU² · bias 없음 (2026-11-03)
 * 선택 항목은 포인터가 NULL 이면 건너뛴다 (LN 없음 · 스케일 1 · bias 0). 정규화 칸은 bias 포인터가 NULL 이면 RMSNorm.
 *
 * 포인터 표 P: 전역 G_NGLOBAL 칸 + 층마다 PER_LAYER 칸. 정수 표 I:
 *   [d, h, ff, L, k, seq, V, fmt(0 삼진·1 bf16·2 fp32·3 int8), act(0 hardtanh·1 gelu·2 relu²), mixer_0..L-1, dil_0..L-1]
 * 배치는 gen_engine.py 의 _ctable() 과 같아야 한다.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>

void tern_quant_rows(const float *x, int M, int K, int8_t *xq, float *s_out);
void tern_gemm(const int8_t *xq, const float *s, int M, int K,
               const uint64_t *W, const int32_t *wsum, const float *g, int N, float *Y);
void tern_gemm_cols(const int8_t *xq, const float *s, int M, int K,
                    const uint64_t *Wc, const float *g, int N, float *Y);
void gen_gemm_bf16(const float *X, int M, int K, const uint16_t *W, const float *bias, int N, float *Y);
void gen_gemm_f32(const float *X, int M, int K, const float *W, const float *bias, int N, float *Y);
void gen_gemm_i8(const int8_t *xq, const float *s, int M, int K, const int8_t *W, const int32_t *wsum,
                 const float *g, const float *bias, int N, float *Y);

enum { FMT_TERN, FMT_BF16, FMT_F32, FMT_I8 };
enum { G_TOK, G_POS, G_NFW, G_NFB, G_LMBF, G_NGLOBAL };
/* 선형층 하나 = 4칸: W, A1, A2, BIAS  (삼진: bits·wsum·gamma / int8: W·wsum·scale / bf16·fp32: W 만)
 * 삼진에서 A1(wsum) 이 NULL 이면 W 는 열 단위 저장(cbits) → 0 열 건너뛰기 커널 tern_gemm_cols */
enum { L_N1W, L_N1B, L_N2W, L_N2B, L_POW, L_POB, L_PFW, L_PFB, L_ASC, L_FSC, L_PAW, L_PAB, L_P1W, L_P1B,
       L_WA, L_WO = L_WA + 4, L_F1 = L_WO + 4, L_F2 = L_F1 + 4,
       L_MIX = L_F2 + 4, L_GATE, L_RING, L_KC, L_VC, PER_LAYER };

static void layernorm(const float *x, int n, const float *w, const float *b, float *y)
{
    float mu = 0.f;
    #pragma omp simd reduction(+:mu)
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= (float)n;
    float var = 0.f;
    #pragma omp simd reduction(+:var)
    for (int i = 0; i < n; i++) { float dd = x[i] - mu; var += dd * dd; }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    #pragma omp simd
    for (int i = 0; i < n; i++) y[i] = (x[i] - mu) * inv * w[i] + b[i];
}

/* torch nn.RMSNorm(eps=None) 는 입력 dtype 의 머신 엡실론을 쓴다 (fp32 1.1920929e-07) */
static void rmsnorm(const float *x, int n, const float *w, float *y)
{
    float ss = 0.f;
    #pragma omp simd reduction(+:ss)
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + 1.1920929e-07f);
    #pragma omp simd
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * w[i];
}

/* 양자화 직전 정규화 칸: bias 가 있으면 LayerNorm, 없으면 RMSNorm (제자리 가능) */
static void prenorm(const float *x, int n, const float *w, const float *b, float *y)
{
    if (b) layernorm(x, n, w, b, y);
    else rmsnorm(x, n, w, y);
}

void gen_rms_rows(const float *x, int M, int n, const float *w, float *y)
{
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) rmsnorm(x + (size_t)m * n, n, w, y + (size_t)m * n);
}

static double g_prof[4];      /* [0] 가중치 행렬곱 · [1] 혼합(conv 탭/attention) · [2] lm_head · [3] 전체 */
void gen_prof(double *out, int reset)
{
    for (int i = 0; i < 4; i++) { out[i] = g_prof[i]; if (reset) g_prof[i] = 0; }
}
static inline double now(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return 0.0;
#endif
}

static int8_t *g_q = NULL;
static int g_qcap = 0;

/* 여러 행(M)·한 행 공용 선형층 — Python prefill 도 이걸 부른다 */
void gen_linear(int fmt, const float *x, int M, int K, const void *const *S, int N, float *y)
{
    double t0 = now();
    const float *bias = (const float *)S[3];
    if (fmt == FMT_BF16) {
        gen_gemm_bf16(x, M, K, (const uint16_t *)S[0], bias, N, y);
    } else if (fmt == FMT_F32) {
        gen_gemm_f32(x, M, K, (const float *)S[0], bias, N, y);
    } else {
        int need = M * K + 64;
        int8_t *q = g_q;
        float sbuf[64], *s = M <= 64 ? sbuf : (float *)malloc(sizeof(float) * M);
        if (M > 1) q = (int8_t *)aligned_alloc(64, ((size_t)need + 63) / 64 * 64);
        else if (g_qcap < need) { free(g_q); g_q = q = (int8_t *)aligned_alloc(64, ((size_t)need + 63) / 64 * 64); g_qcap = need; }
        tern_quant_rows(x, M, K, q, s);
        if (fmt == FMT_TERN && !S[1]) {
            tern_gemm_cols(q, s, M, K, (const uint64_t *)S[0], (const float *)S[2], N, y);
            if (bias) for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) y[(size_t)m * N + n] += bias[n];
        } else if (fmt == FMT_TERN) {
            tern_gemm(q, s, M, K, (const uint64_t *)S[0], (const int32_t *)S[1], (const float *)S[2], N, y);
            if (bias) for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) y[(size_t)m * N + n] += bias[n];
        } else {
            gen_gemm_i8(q, s, M, K, (const int8_t *)S[0], (const int32_t *)S[1], (const float *)S[2], bias, N, y);
        }
        if (M > 1) free(q);
        if (s != sbuf) free(s);
    }
    g_prof[0] += now() - t0;
}

static void act_inplace(int act, float *f, int n)
{
    if (act == 0) {
        for (int i = 0; i < n; i++) f[i] = f[i] < -1.f ? -1.f : (f[i] > 1.f ? 1.f : f[i]);
    } else if (act == 2) {
        #pragma omp simd
        for (int i = 0; i < n; i++) { float r = f[i] > 0.f ? f[i] : 0.f; f[i] = r * r; }
    } else {
        /* erff 는 벡터화되지 않아(libmvec 은 -ffast-math 에서만) 직렬이면 1B 에서 토큰당 ~2.5 ms — 스레드로 나눈다.
           원소별 같은 erff 라 출력은 직렬과 비트 단위로 같다 */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) f[i] = 0.5f * f[i] * (1.0f + erff(f[i] * 0.70710678118654752f));
    }
}

void gen_step(const void *const *P, const int *I, int t, int token, float *logits, float *work)
{
    const int d = I[0], h = I[1], ff = I[2], L = I[3], k = I[4], seq = I[5], V = I[6], fmt = I[7], act = I[8];
    const int *mixer = I + 9, *dil = I + 9 + L;
    /* 게이트 (11-15, bitgc3g): 층 I[9+2L] 의 혼합 출력을 입력 토큰이 I[10+2L] 일 때 0 으로 (tests/echo_gate.py allbot, α=0).
     * 없으면 -1. K·V 캐시·gconv 이력은 그대로 쓰고 잔차에 더하는 값만 막는다 (torch 와 같음). */
    const int gate_l = I[9 + 2 * L], gate_tok = I[10 + 2 * L];
    const int dk = d / h;
    float *x = work, *hn = x + d, *a = hn + d, *ctx = a + d, *qkv = ctx + d;
    float *f1 = qkv + 3 * d, *f1n = f1 + ff, *sc = f1n + ff;

    const double T0 = now();
    const float *tok = (const float *)P[G_TOK], *pos = (const float *)P[G_POS];
    for (int i = 0; i < d; i++) x[i] = tok[(size_t)token * d + i] + pos[(size_t)t * d + i];

    for (int l = 0; l < L; l++) {
        const void *const *Q = P + G_NGLOBAL + (size_t)l * PER_LAYER;
        layernorm(x, d, (const float *)Q[L_N1W], (const float *)Q[L_N1B], hn);
        if (Q[L_PAW]) prenorm(hn, d, (const float *)Q[L_PAW], (const float *)Q[L_PAB], hn);
        if (mixer[l] == 0) {                              /* gconv */
            gen_linear(fmt, hn, 1, d, Q + L_WA, d, a);
            const double tm = now();
            const int D = dil[l], Pn = (k - 1) * D + 1;
            float *ring = (float *)Q[L_RING];
            memcpy(ring + (size_t)(t % Pn) * d, a, sizeof(float) * d);
            const float *mix = (const float *)Q[L_MIX], *gate = (const float *)Q[L_GATE];
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < d; c++) {
                float hm = 0.f, hg = 0.f;
                for (int j = 0; j < k; j++) {
                    int src = t - (k - 1 - j) * D;
                    float v = src >= 0 ? ring[(size_t)(src % Pn) * d + c] : 0.f;
                    hm += v * mix[c * k + j];
                    hg += v * gate[c * k + j];
                }
                ctx[c] = hm * (1.0f / (1.0f + expf(-hg)));
            }
            g_prof[1] += now() - tm;
        } else {                                          /* attention */
            gen_linear(fmt, hn, 1, d, Q + L_WA, 3 * d, qkv);
            const double tm = now();
            float *kc = (float *)Q[L_KC], *vc = (float *)Q[L_VC];
            const float scale = 1.0f / sqrtf((float)dk);
            #pragma omp parallel for schedule(static)
            for (int hh = 0; hh < h; hh++) {
                const float *q = qkv + hh * dk;
                float *K = kc + (size_t)hh * seq * dk, *Vv = vc + (size_t)hh * seq * dk;
                memcpy(K + (size_t)t * dk, qkv + d + hh * dk, sizeof(float) * dk);
                memcpy(Vv + (size_t)t * dk, qkv + 2 * d + hh * dk, sizeof(float) * dk);
                float *s = sc + (size_t)hh * seq, mx = -INFINITY;
                for (int j = 0; j <= t; j++) {
                    const float *kk = K + (size_t)j * dk;
                    float z = 0.f;
                    for (int e = 0; e < dk; e++) z += q[e] * kk[e];
                    s[j] = z * scale;
                    if (s[j] > mx) mx = s[j];
                }
                float sum = 0.f;
                for (int j = 0; j <= t; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
                float *o = ctx + hh * dk;
                for (int e = 0; e < dk; e++) o[e] = 0.f;
                for (int j = 0; j <= t; j++) {
                    const float p = s[j] / sum, *vv = Vv + (size_t)j * dk;
                    for (int e = 0; e < dk; e++) o[e] += p * vv[e];
                }
            }
            g_prof[1] += now() - tm;
        }
        const float *ow = ctx;
        if (Q[L_POW]) { prenorm(ctx, d, (const float *)Q[L_POW], (const float *)Q[L_POB], hn); ow = hn; }
        gen_linear(fmt, ow, 1, d, Q + L_WO, d, a);
        const float as = Q[L_ASC] ? *(const float *)Q[L_ASC] : 1.f, fs = Q[L_FSC] ? *(const float *)Q[L_FSC] : 1.f;
        if (!(l == gate_l && token == gate_tok))
            for (int i = 0; i < d; i++) x[i] += as * a[i];
        layernorm(x, d, (const float *)Q[L_N2W], (const float *)Q[L_N2B], hn);
        if (Q[L_P1W]) prenorm(hn, d, (const float *)Q[L_P1W], (const float *)Q[L_P1B], hn);
        gen_linear(fmt, hn, 1, d, Q + L_F1, ff, f1);
        act_inplace(act, f1, ff);
        const float *fi = f1;
        if (Q[L_PFW]) { prenorm(f1, ff, (const float *)Q[L_PFW], (const float *)Q[L_PFB], f1n); fi = f1n; }
        gen_linear(fmt, fi, 1, ff, Q + L_F2, d, a);
        for (int i = 0; i < d; i++) x[i] += fs * a[i];
    }
    layernorm(x, d, (const float *)P[G_NFW], (const float *)P[G_NFB], hn);
    const double TL = now();
    const uint16_t *bf = (const uint16_t *)P[G_LMBF];
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < V; v++) {
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        int i = 0;
        if (bf) {
            const uint16_t *r = bf + (size_t)v * d;
            for (; i + 32 <= d; i += 32) {
                __m512 w0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256((const __m256i *)(r + i))), 16));
                __m512 w1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256((const __m256i *)(r + i + 16))), 16));
                acc0 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(hn + i), acc0);
                acc1 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(hn + i + 16), acc1);
            }
        } else {
            const float *r = tok + (size_t)v * d;
            for (; i + 32 <= d; i += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(r + i), _mm512_loadu_ps(hn + i), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(r + i + 16), _mm512_loadu_ps(hn + i + 16), acc1);
            }
        }
        logits[v] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
    g_prof[2] += now() - TL;
    g_prof[3] += now() - T0;
}
