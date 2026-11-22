"""엔진 속도 측정 — 배치 1 생성 ms/토큰 (문맥 길이별) · prefill tok/s.

  python scripts/bench_engine.py --engine bitgc_1b.npz [--ctx 64,448] [--n 64] [--repeat 3]

원 연구 표의 측정 방식과 같다: 문맥을 prefill 한 뒤 무작위 토큰을 n 개 한 토큰씩 넣고, 준비(첫 호출) 를 뺀 평균.
다른 프로그램이 CPU 를 쓰고 있으면 크게 느려진다 (스레드 경합).
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # numpy BLAS 스레드가 엔진의 OpenMP 와 경합하면 생성이 ~1.5배 느려진다

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from bitgc.engine.runtime import GenLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ctx", default="64,448")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    eng = GenLM(a.engine)
    rng = np.random.default_rng(0)
    print(f"스레드 prefill {eng.threads_prefill} · 생성 {eng.threads_decode} | 가중치 {eng.weight_bytes()/1e6:.0f}MB", flush=True)
    eng.reset()
    eng.forward(rng.integers(100, eng.V, 8).tolist())
    eng.step(100)                                   # 준비: C 표 · lm_head bf16
    for c in [int(v) for v in a.ctx.split(",")]:
        dec, pre = [], []
        for _ in range(a.repeat):
            eng.reset()
            ids = rng.integers(100, eng.V, c).tolist()
            t0 = time.perf_counter()
            eng.forward(ids, last_only=True)
            t1 = time.perf_counter()
            for t in rng.integers(100, eng.V, a.n).tolist():
                eng.step(t)
            t2 = time.perf_counter()
            pre.append(c / (t1 - t0))
            dec.append(1000 * (t2 - t1) / a.n)
        print(f"문맥 {c:4d}: 생성 {np.median(dec):6.2f} ms/토큰 (반복 {a.repeat}회 중앙값, 범위 {min(dec):.2f}~{max(dec):.2f}) · "
              f"prefill {np.median(pre):6.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
