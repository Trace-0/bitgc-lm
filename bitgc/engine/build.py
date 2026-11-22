"""엔진 라이브러리 빌드:  python -m bitgc.engine.build  [--cc gcc] [--no-native]

csrc/ 의 C 네 개를 OpenMP 와 함께 공유 라이브러리로 묶는다. -march=native 로 이 CPU 의 명령어(AVX2 · AVX-512 VNNI)를 쓴다
(다른 CPU 로 옮길 라이브러리면 --no-native). Windows 는 WSL 에서 빌드·실행한다.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ["tern_gemm.c", "hyb_step.c", "gen_gemm.c", "gen_step.c"]


def build(cc: str = "", native: bool = True) -> str:
    mac = sys.platform == "darwin"
    out = os.path.join(_HERE, "libbitgc_engine.dylib" if mac else "libbitgc_engine.so")
    cc = cc or os.environ.get("CC", "clang" if mac else "gcc")
    omp = ["-Xpreprocessor", "-fopenmp", "-lomp"] if mac else ["-fopenmp"]
    cmd = [cc, "-O3", *(["-march=native"] if native else []), *omp, "-ffp-contract=off", "-shared", "-fPIC",
           "-o", out, *[os.path.join(_HERE, "csrc", s) for s in SRC], "-lm"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"빌드 완료: {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default="")
    ap.add_argument("--no-native", action="store_true")
    a = ap.parse_args()
    build(a.cc, not a.no_native)
