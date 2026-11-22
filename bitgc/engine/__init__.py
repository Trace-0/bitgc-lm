"""정수 CPU 엔진 — 삼진 가중치 행렬곱을 곱셈 없이 (원 연구의 속도·에너지 측정 엔진).

  python -m bitgc.engine.build                                   # 라이브러리 빌드 (한 번)
  python scripts/export_engine.py --ckpt hf:trace-000/bitgc-lm-1b --out bitgc_1b.npz --parity
  python chat.py --engine bitgc_1b.npz --tokenizer tokenizer.model

`runtime.GenLM` 은 import 할 때 라이브러리를 읽는다 — 이 패키지 자체는 빌드 없이도 import 된다.
"""
