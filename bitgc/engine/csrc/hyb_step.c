/*
 * hyb_step.c — 하이브리드 1B 의 한 토큰 진행(생성 경로)을 통째로 C 에서 (tern_gemm.c 와 함께 libtern.so).
 *
 * numpy 로 한 토큰을 돌리면 층마다 작은 연산 수십 개(LN·양자화·탭 모으기·sigmoid …)의 파이썬·할당
 * 오버헤드가 가중치 읽기(273MB/토큰)보다 컸다 (24층 ≈ 19ms 중 가중치 몫 ~5ms). 여기서는
 *   LN → 삼진×int8 GEMV(OpenMP) → conv 탭(고리 버퍼)·게이트 / attention(K·V 캐시) → LN → W_o
 *   → 잔차 → LN → fc1 → hardtanh → LN → fc2 → 잔차, 마지막 LN → lm_head(fp32, OpenMP)
 * 를 한 함수로 한다. 모든 상태(고리 버퍼·K/V 캐시)는 파이썬(numpy)이 소유한 배열이라 prefill(numpy 경로)과
 * 생성(이 함수)이 같은 상태를 이어 쓴다.
 *
 * 포인터 표 P (층 i 마다 PER_LAYER 칸, 앞에 전역 칸) · 정수 표 I 의 배치는 hyb_engine.py 의 _ctable() 과 같다.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

void tern_quant_rows(const float *x, int M, int K, int8_t *xq, float *s_out);
void tern_gemm(const int8_t *xq, const float *s, int M, int K,
               const uint64_t *W, const int32_t *wsum, const float *g, int N, float *Y);

/* 전역 칸 */
enum { G_TOK, G_POS, G_NFW, G_NFB, G_LMBF, G_NGLOBAL };   /* G_LMBF: lm_head bf16 (NULL 이면 fp32) */
/* 층 칸 */
enum { L_N1W, L_N1B, L_N2W, L_N2B, L_POW, L_POB, L_PFW, L_PFB, L_ASC, L_FSC,
       L_WA, L_WAS, L_WAG,          /* gconv: W_v · attn: W_qkv */
       L_WO, L_WOS, L_WOG, L_F1, L_F1S, L_F1G, L_F2, L_F2S, L_F2G,
       L_MIX, L_GATE, L_RING, L_KC, L_VC, PER_LAYER };

static void layernorm(const float *x, int n, const float *w, const float *b, float *y)
{
    float mu = 0.f;
    #pragma omp simd reduction(+:mu)
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= (float)n;
    float var = 0.f;
    #pragma omp simd reduction(+:var)
    for (int i = 0; i < n; i++) { float d = x[i] - mu; var += d * d; }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    #pragma omp simd
    for (int i = 0; i < n; i++) y[i] = (x[i] - mu) * inv * w[i] + b[i];
}

/* 구간별 누적 시간 (초): [0] 삼진 GEMV · [1] 혼합(conv 탭/attention) · [2] lm_head · [3] 전체 */
static double g_prof[4];
void hyb_prof(double *out, int reset)
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

static void tlin(const float *x, int K, const void *const *P, int base, int N, float *y,
                 int8_t *qbuf)
{
    double t0 = now();
    float s;
    tern_quant_rows(x, 1, K, qbuf, &s);
    tern_gemm(qbuf, &s, 1, K, (const uint64_t *)P[base], (const int32_t *)P[base + 1],
              (const float *)P[base + 2], N, y);
    g_prof[0] += now() - t0;
}

/*
 * I: [d, h, ff, L, k, seq, V, mixer_0..L-1 (0=gconv,1=attn), dil_0..L-1]
 * work: float 작업 공간 (최소 d*8 + ff*2 + 3*d + seq*h)
 */
void hyb_step(const void *const *P, const int *I, int t, int token, float *logits, float *work)
{
    const int d = I[0], h = I[1], ff = I[2], L = I[3], k = I[4], seq = I[5], V = I[6];
    const int *mixer = I + 7, *dil = I + 7 + L;
    const int dk = d / h;
    float *x = work, *hn = x + d, *a = hn + d, *ctx = a + d, *qkv = ctx + d;   /* qkv: 3d */
    float *f1 = qkv + 3 * d, *f1n = f1 + ff, *sc = f1n + ff;                 /* sc: h*seq */
    static int8_t *qbuf = NULL;
    static int qcap = 0;
    if (qcap < ff + 64) { free(qbuf); qbuf = (int8_t *)aligned_alloc(64, ff + 64); qcap = ff + 64; }

    const double T0 = now();
    const float *tok = (const float *)P[G_TOK], *pos = (const float *)P[G_POS];
    for (int i = 0; i < d; i++) x[i] = tok[(size_t)token * d + i] + pos[(size_t)t * d + i];

    for (int l = 0; l < L; l++) {
        const void *const *Q = P + G_NGLOBAL + (size_t)l * PER_LAYER;
        layernorm(x, d, (const float *)Q[L_N1W], (const float *)Q[L_N1B], hn);
        if (mixer[l] == 0) {                              /* gconv */
            tlin(hn, d, Q, L_WA, d, a, qbuf);              /* v */
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
            tlin(hn, d, Q, L_WA, 3 * d, qkv, qbuf);
            const double tm = now();
            float *kc = (float *)Q[L_KC], *vc = (float *)Q[L_VC];   /* (h, seq, dk) */
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
        layernorm(ctx, d, (const float *)Q[L_POW], (const float *)Q[L_POB], hn);
        tlin(hn, d, Q, L_WO, d, a, qbuf);
        const float as = *(const float *)Q[L_ASC], fs = *(const float *)Q[L_FSC];
        for (int i = 0; i < d; i++) x[i] += as * a[i];
        layernorm(x, d, (const float *)Q[L_N2W], (const float *)Q[L_N2B], hn);
        tlin(hn, d, Q, L_F1, ff, f1, qbuf);
        for (int i = 0; i < ff; i++) f1[i] = f1[i] < -1.f ? -1.f : (f1[i] > 1.f ? 1.f : f1[i]);
        layernorm(f1, ff, (const float *)Q[L_PFW], (const float *)Q[L_PFB], f1n);
        tlin(f1n, ff, Q, L_F2, d, a, qbuf);
        for (int i = 0; i < d; i++) x[i] += fs * a[i];
    }
    layernorm(x, d, (const float *)P[G_NFW], (const float *)P[G_NFB], hn);
    const double TL = now();
    /* lm_head: logits[v] = tok[v]·hn (weight tying). bf16 표가 있으면 그걸 읽는다 — 134MB → 67MB,
     * 생성 한 토큰에서 가중치 다음으로 큰 메모리 읽기라 대역 한계에서 그대로 시간이 준다. */
    const uint16_t *bf = (const uint16_t *)P[G_LMBF];
    if (bf) {
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < V; v++) {
            const uint16_t *r = bf + (size_t)v * d;
#if defined(__AVX512F__)
            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
            int i = 0;
            for (; i + 32 <= d; i += 32) {
                __m512 w0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256((const __m256i *)(r + i))), 16));
                __m512 w1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256((const __m256i *)(r + i + 16))), 16));
                acc0 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(hn + i), acc0);
                acc1 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(hn + i + 16), acc1);
            }
            float z = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
#else
            float z = 0.f;
            int i = 0;
#endif
            for (; i < d; i++) {
                uint32_t u = (uint32_t)r[i] << 16;
                float w;
                memcpy(&w, &u, 4);
                z += w * hn[i];
            }
            logits[v] = z;
        }
        g_prof[2] += now() - TL;
        g_prof[3] += now() - T0;
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < V; v++) {
        const float *r = tok + (size_t)v * d;
#if defined(__AVX512F__)
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 32 <= d; i += 32) {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(r + i), _mm512_loadu_ps(hn + i), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(r + i + 16), _mm512_loadu_ps(hn + i + 16), acc1);
        }
        float z = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
        for (; i < d; i++) z += r[i] * hn[i];
#else
        float z = 0.f;
        for (int i = 0; i < d; i++) z += r[i] * hn[i];
#endif
        logits[v] = z;
    }
    g_prof[2] += now() - TL;
    g_prof[3] += now() - T0;
}

/* ── prefill(여러 토큰) 보조 커널 — numpy 의 단일 스레드 LN·conv·attention 을 대신한다 ─────────── */

/* 행별 LayerNorm (M×n) */
void hyb_ln_rows(const float *x, int M, int n, const float *w, const float *b, float *y)
{
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) layernorm(x + (size_t)m * n, n, w, b, y + (size_t)m * n);
}

/* gconv 여러 토큰: v (T×d) 를 고리 버퍼(Pn×d, 시각 t 의 칸 t mod Pn)에 쓰며 시각 t0..t0+T-1 의 출력.
 * 고리 버퍼가 Pn 칸뿐이라 T 가 크면 먼저 쓴 값이 덮인다 → 이력은 고리에서, 이번 구간은 v 에서 읽는다. */
void hyb_gconv_seq(const float *v, int T, int d, float *ring, int Pn, int t0, int D, int k,
                   const float *mix, const float *gate, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < d; c++) {
        for (int t = 0; t < T; t++) {
            float hm = 0.f, hg = 0.f;
            for (int j = 0; j < k; j++) {
                int src = t0 + t - (k - 1 - j) * D;
                float val;
                if (src < 0) val = 0.f;
                else if (src >= t0) val = v[(size_t)(src - t0) * d + c];
                else val = ring[(size_t)(src % Pn) * d + c];
                hm += val * mix[c * k + j];
                hg += val * gate[c * k + j];
            }
            out[(size_t)t * d + c] = hm * (1.0f / (1.0f + expf(-hg)));
        }
    }
    /* 마지막 Pn 개를 고리에 */
    int from = T > Pn ? T - Pn : 0;
    for (int t = from; t < T; t++)
        memcpy(ring + (size_t)((t0 + t) % Pn) * d, v + (size_t)t * d, sizeof(float) * d);
}

/* 인과 attention 여러 토큰: qkv (T×3d), 캐시 (h, seq, dk) 에 k·v 를 쓰고 ctx (T×d) */
void hyb_attn_seq(const float *qkv, int T, int t0, int h, int dk, int seq, float *kc, float *vc, float *ctx)
{
    const int d = h * dk;
    for (int t = 0; t < T; t++)
        for (int hh = 0; hh < h; hh++) {
            memcpy(kc + ((size_t)hh * seq + t0 + t) * dk, qkv + (size_t)t * 3 * d + d + hh * dk, sizeof(float) * dk);
            memcpy(vc + ((size_t)hh * seq + t0 + t) * dk, qkv + (size_t)t * 3 * d + 2 * d + hh * dk, sizeof(float) * dk);
        }
    const float scale = 1.0f / sqrtf((float)dk);
    #pragma omp parallel
    {
        float *s = (float *)malloc(sizeof(float) * (t0 + T));
        #pragma omp for schedule(dynamic, 4) collapse(2)
        for (int hh = 0; hh < h; hh++)
            for (int t = 0; t < T; t++) {
                const float *q = qkv + (size_t)t * 3 * d + hh * dk;
                const float *K = kc + (size_t)hh * seq * dk, *V = vc + (size_t)hh * seq * dk;
                const int n = t0 + t + 1;
                float mx = -INFINITY;
                for (int j = 0; j < n; j++) {
                    float z = 0.f;
                    #pragma omp simd reduction(+:z)
                    for (int e = 0; e < dk; e++) z += q[e] * K[(size_t)j * dk + e];
                    s[j] = z * scale;
                    if (s[j] > mx) mx = s[j];
                }
                float sum = 0.f;
                for (int j = 0; j < n; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
                float *o = ctx + (size_t)t * d + hh * dk;
                for (int e = 0; e < dk; e++) o[e] = 0.f;
                for (int j = 0; j < n; j++) {
                    const float p = s[j] / sum, *vv = V + (size_t)j * dk;
                    #pragma omp simd
                    for (int e = 0; e < dk; e++) o[e] += p * vv[e];
                }
            }
        free(s);
    }
}
