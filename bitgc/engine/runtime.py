"""정수 CPU 엔진 런타임 — 삼진 가중치 × 8bit 활성 행렬곱을 곱셈 없이 (C, OpenMP, AVX2 / AVX-512 VNNI).

원 연구 logiformer `engine_int/gen_engine.py` 를 옮긴 것이다 (논문의 속도·에너지 측정에 쓴 엔진). 같은 C 코드가 원 연구의
여러 비교군 형식(bf16 · f32 · int8)도 돌리지만, 이 저장소의 BitGC 모델은 삼진(`tern`) 하나만 쓴다.

  eng = GenLM("model.npz");  lg = eng.forward(ids, last_only=True);  lg = eng.step(tok);  eng.reset()
  prefill(여러 토큰) = gen_linear(M=T) + 보조 커널 · 생성(한 토큰) = gen_step (C 함수 하나, KV 캐시·conv 링 버퍼 포함)

자산은 `scripts/export_engine.py` 가 만든다. 라이브러리는 `python -m bitgc.engine.build` (Linux · WSL · macOS, gcc/clang + OpenMP).
환경 변수: GEN_THREADS_PREFILL (기본 코어 전부) · GEN_THREADS_DECODE (기본 코어 절반 = 물리 코어) · GEN_FAST_HEAD (기본 1: lm_head bf16)
           BITGC_ENGINE_LIB (라이브러리 경로 직접 지정)
"""
from __future__ import annotations

import ctypes
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_P, _I = ctypes.c_void_p, ctypes.c_int
FMTS = {"tern": 0, "bf16": 1, "f32": 2, "i8": 3}


LIB_NAME = "libbitgc_engine.dylib" if __import__("sys").platform == "darwin" else "libbitgc_engine.so"


def _load():
    p = os.environ.get("BITGC_ENGINE_LIB") or os.path.join(_HERE, LIB_NAME)
    if not os.path.exists(p):
        raise RuntimeError(f"엔진 라이브러리가 없다: {p} — 먼저 빌드: python -m bitgc.engine.build")
    lib = ctypes.CDLL(p)
    lib.tern_set_threads.argtypes = [_I]
    lib.tern_set_threads.restype = _I
    lib.gen_linear.argtypes = [_I, _P, _I, _I, _P, _I, _P]
    lib.gen_linear.restype = None
    lib.gen_step.argtypes = [_P, _P, _I, _I, _P, _P]
    lib.gen_step.restype = None
    lib.gen_prof.argtypes = [_P, _I]
    lib.gen_prof.restype = None
    lib.gen_gelu.argtypes = [_P, ctypes.c_size_t]
    lib.gen_gelu.restype = None
    lib.hyb_ln_rows.argtypes = [_P, _I, _I, _P, _P, _P]
    lib.hyb_ln_rows.restype = None
    lib.gen_rms_rows.argtypes = [_P, _I, _I, _P, _P]
    lib.gen_rms_rows.restype = None
    lib.tern_sparse_stats.argtypes = [_P, _I]
    lib.tern_sparse_stats.restype = None
    lib.hyb_gconv_seq.argtypes = [_P, _I, _I, _P, _I, _I, _I, _I, _P, _P, _P]
    lib.hyb_gconv_seq.restype = None
    lib.hyb_attn_seq.argtypes = [_P, _I, _I, _I, _I, _I, _P, _P, _P]
    lib.hyb_attn_seq.restype = None
    return lib


LIB = _load()


class GenLM:
    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.z = {k: z[k] for k in z.files}
        self.meta = m = json.loads(bytes(self.z["meta"]).decode("utf-8"))
        self.fmt = FMTS[m["fmt"]]
        self.d, self.h, self.L, self.ff = m["d_model"], m["heads"], m["layers"], m["d_ff"]
        self.dk = self.d // self.h
        self.k = m.get("kernel", 9)
        self.mixers, self.dil = m["mixers"], m["dilations"]
        self.act = {"hardtanh": 0, "gelu": 1, "relu2": 2}[m["act"]]
        self.seq_len, self.V = m["seq_len"], m["vocab"]
        self.tok, self.pos = self.z["tok_emb"], self.z["pos_emb"]
        g = m.get("gate")            # {"layer": 11, "token": 5} — 그 층 혼합 출력을 입력 토큰이 그 토큰일 때 잔차에 더하지 않는다
        self.gate_l, self.gate_tok = (int(g["layer"]), int(g["token"])) if g else (-1, -1)
        ncpu = os.cpu_count() or 8
        self.threads_prefill = int(os.environ.get("GEN_THREADS_PREFILL", ncpu))
        self.threads_decode = int(os.environ.get("GEN_THREADS_DECODE", max(1, ncpu // 2)))
        self.fast_head = os.environ.get("GEN_FAST_HEAD", "1") == "1"
        self._keep = []
        self.reset()

    # ── 상태 ──
    def reset(self):
        if getattr(self, "_cP", None) is not None:
            for a in self.kc + self.vc + self.hist:
                if a is not None:
                    a[...] = 0.0
            self.t = 0
            return
        self._cP, self.t = None, 0
        self.kc, self.vc, self.hist = [None] * self.L, [None] * self.L, [None] * self.L
        for i, mx in enumerate(self.mixers):
            if mx == "attn":
                self.kc[i] = np.zeros((self.h, self.seq_len, self.dk), np.float32)
                self.vc[i] = np.zeros((self.h, self.seq_len, self.dk), np.float32)
            else:
                self.hist[i] = np.zeros(((self.k - 1) * self.dil[i] + 1, self.d), np.float32)

    # ── 선형층 슬롯 (C 의 4칸: W, A1, A2, BIAS) ──
    def _slots(self, tag):
        z, f = self.z, self.meta["fmt"]
        if f == "tern" and f"{tag}.cbits" in z:        # 열 단위 저장 → 0 열 건너뛰기 (wsum 칸 NULL 이 신호)
            arrs = [z[f"{tag}.cbits"], None, z[f"{tag}.gamma"]]
        elif f == "tern":
            arrs = [z[f"{tag}.bits"], z[f"{tag}.wsum"], z[f"{tag}.gamma"]]
        elif f == "i8":
            arrs = [z[f"{tag}.w8"], z[f"{tag}.wsum"], z[f"{tag}.scale"]]
        else:
            arrs = [z[f"{tag}.w"], None, None]
        arrs.append(z.get(f"{tag}.bias"))
        return arrs

    def _lin(self, x, tag, N):
        x = np.ascontiguousarray(x, np.float32)
        M, K = x.shape
        s = self._slots(tag)
        S = (ctypes.c_void_p * 4)(*[None if a is None else a.ctypes.data for a in s])
        y = np.empty((M, N), np.float32)
        LIB.gen_linear(self.fmt, x.ctypes.data, M, K, S, N, y.ctypes.data)
        return y

    def _norm(self, x, tag):
        """tag.w 가 없으면 그대로, tag.b 가 없으면 RMSNorm (C 의 prenorm 과 같은 규약)."""
        if f"{tag}.w" not in self.z:
            return x
        x = np.ascontiguousarray(x, np.float32)
        y = np.empty_like(x)
        if f"{tag}.b" in self.z:
            LIB.hyb_ln_rows(x.ctypes.data, x.shape[0], x.shape[1], self.z[f"{tag}.w"].ctypes.data,
                            self.z[f"{tag}.b"].ctypes.data, y.ctypes.data)
        else:
            LIB.gen_rms_rows(x.ctypes.data, x.shape[0], x.shape[1], self.z[f"{tag}.w"].ctypes.data, y.ctypes.data)
        return y

    def _scale(self, tag):
        return float(np.asarray(self.z[tag]).reshape(-1)[0]) if tag in self.z else 1.0

    def forward(self, ids, last_only: bool = False):
        """prefill (여러 토큰, 캐시 이어 씀). 반환: (T, V) 또는 마지막 (V,)."""
        ids = list(ids)
        T = len(ids)
        assert self.t + T <= self.seq_len
        LIB.tern_set_threads(self.threads_prefill)
        x = (self.tok[ids] + self.pos[self.t:self.t + T]).astype(np.float32)
        d = self.d
        for i in range(self.L):
            h = self._norm(self._norm(x, f"L{i}.n1"), f"L{i}.pre_a")
            if self.mixers[i] == "attn":
                qkv = self._lin(h, f"L{i}.WA", 3 * d)
                ctx = np.empty((T, d), np.float32)
                LIB.hyb_attn_seq(qkv.ctypes.data, T, self.t, self.h, self.dk, self.seq_len,
                                 self.kc[i].ctypes.data, self.vc[i].ctypes.data, ctx.ctypes.data)
            else:
                v = self._lin(h, f"L{i}.WA", d)
                ctx = np.empty((T, d), np.float32)
                P = (self.k - 1) * self.dil[i]
                LIB.hyb_gconv_seq(v.ctypes.data, T, d, self.hist[i].ctypes.data, P + 1, self.t, self.dil[i], self.k,
                                  self.z[f"L{i}.mix"].ctypes.data, self.z[f"L{i}.gate"].ctypes.data, ctx.ctypes.data)
            o = self._lin(self._norm(ctx, f"L{i}.pre_o"), f"L{i}.WO", d)
            if i == self.gate_l:
                o[np.asarray(ids) == self.gate_tok] = 0.0
            x = x + self._scale(f"L{i}.attn_scale") * o
            f1 = self._lin(self._norm(self._norm(x, f"L{i}.n2"), f"L{i}.pre_f1"), f"L{i}.F1", self.ff)
            if self.act == 1:
                LIB.gen_gelu(f1.ctypes.data, f1.size)
            elif self.act == 2:
                np.maximum(f1, 0.0, out=f1)
                np.multiply(f1, f1, out=f1)
            else:
                np.clip(f1, -1.0, 1.0, out=f1)
            x = x + self._scale(f"L{i}.ffn_scale") * self._lin(self._norm(f1, f"L{i}.pre_fc2"), f"L{i}.F2", d)
        self.t += T
        x = self._norm(x[-1:] if last_only else x, "normf")
        lg = x @ self.tok.T
        return lg[-1] if last_only else lg

    # ── 생성 (C 한 함수) ──
    _LN = ["n1.w", "n1.b", "n2.w", "n2.b", "pre_o.w", "pre_o.b", "pre_fc2.w", "pre_fc2.b", "attn_scale", "ffn_scale",
           "pre_a.w", "pre_a.b", "pre_f1.w", "pre_f1.b"]     # C 의 L_N1W..L_P1B 순서

    def _ctable(self):
        keep, P = [], []

        def ptr(a):
            if a is None:
                P.append(None)
                return
            a = np.ascontiguousarray(a)
            keep.append(a)
            P.append(a.ctypes.data)
        z = self.z
        for k in ("tok_emb", "pos_emb", "normf.w", "normf.b"):
            ptr(z[k])
        ptr(self._lm_bf16() if self.fast_head else None)
        for i in range(self.L):
            for k in self._LN:
                a = z.get(f"L{i}.{k}")
                ptr(None if a is None else np.asarray(a, np.float32).reshape(-1))
            for name in ("WA", "WO", "F1", "F2"):
                for a in self._slots(f"L{i}.{name}"):
                    ptr(a)
            ptr(z.get(f"L{i}.mix"))
            ptr(z.get(f"L{i}.gate"))
            ptr(self.hist[i])
            ptr(self.kc[i])
            ptr(self.vc[i])
        I = [self.d, self.h, self.ff, self.L, self.k, self.seq_len, self.V, self.fmt, self.act]
        I += [1 if mx == "attn" else 0 for mx in self.mixers] + list(self.dil) + [self.gate_l, self.gate_tok]
        self._cP = (ctypes.c_void_p * len(P))(*P)
        self._cI = (ctypes.c_int * len(I))(*I)
        self._ckeep = keep
        self._work = np.zeros(8 * self.d + 2 * self.ff + self.h * self.seq_len + 64, np.float32)
        self._logits = np.zeros(self.V, np.float32)

    def _lm_bf16(self):
        if getattr(self, "_bf", None) is None:
            u = self.tok.view(np.uint32).astype(np.uint64)
            self._bf = np.ascontiguousarray(((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16))
        return self._bf

    def step(self, token_id: int):
        assert self.t < self.seq_len
        if self._cP is None:
            self._ctable()
        LIB.tern_set_threads(self.threads_decode)
        LIB.gen_step(self._cP, self._cI, self.t, int(token_id), self._logits.ctypes.data, self._work.ctypes.data)
        self.t += 1
        return self._logits.copy()

    def prof(self, reset=True):
        out = np.zeros(4, np.float64)
        LIB.gen_prof(out.ctypes.data, 1 if reset else 0)
        return out

    def weight_bytes(self):
        """블록 선형층 가중치 바이트 (생성 한 토큰에 읽는 양의 대부분 — 0 열 건너뛰기 전)."""
        f = self.meta["fmt"]
        keys = {"tern": (".bits", ".cbits"), "i8": (".w8",)}.get(f, (".w",))
        return sum(v.nbytes for k, v in self.z.items() if k.startswith("L") and k.endswith(keys)
                   and any(t in k for t in (".WA", ".WO", ".F1", ".F2")))

    def sparse_stats(self, reset=True):
        """0 열 건너뛰기 커널이 본 입력 수 (0 아닌 것, 전체)."""
        out = np.zeros(2, np.int64)
        LIB.tern_sparse_stats(out.ctypes.data, 1 if reset else 0)
        return int(out[0]), int(out[1])
