"""bitgc-v1 .pt → 배포 폴더 (model.safetensors + config.json + tokenizer.model [+ README.md]).

  python scripts/export_pretrained.py --ckpt runs/1b/instr.pt --tokenizer tokenizer/my_spm.model \
      --out export/bitgc-lm-1b [--card model_card.md]

폴더는 chat.py / sft.py / probe_recall.py 의 --ckpt 로 그대로 쓸 수 있고 (--tokenizer 생략 가능),
Hugging Face 에 올리면 --ckpt hf:사용자/저장소 로 받아 쓴다. 내보낸 뒤 원본과 로짓이 같은지 확인한다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from bitgc.checkpoint import load_checkpoint, save_pretrained  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", default="", help="모델 카드 (README.md 로 복사)")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    save_pretrained(ck, a.out, a.tokenizer)
    del ck
    if a.card:
        shutil.copyfile(a.card, os.path.join(a.out, "README.md"))
    torch.manual_seed(0)
    n1, _ = load_checkpoint(a.ckpt)               # 한 번에 모델 하나만 메모리에 (1B fp32 ≈ 4.4GB)
    n1.eval()
    x = torch.randint(7, n1.cfg.vocab_size, (1, 64))
    if n1.gate_token is not None:
        x[0, 40] = n1.gate_token
    with torch.no_grad():
        ref = n1(x)
    del n1
    n2, ck2 = load_checkpoint(a.out)
    n2.eval()
    with torch.no_grad():
        d = (ref - n2(x)).abs().max().item()
    assert d == 0.0, f"내보낸 폴더의 로짓이 원본과 다르다 (max|Δ| {d})"
    print(f"저장 {a.out} — 로짓 원본과 같음 · 게이트 {list(n2.gate_layers)} · "
          f"{os.path.getsize(os.path.join(a.out, 'model.safetensors'))/1e9:.2f}GB")


if __name__ == "__main__":
    main()
