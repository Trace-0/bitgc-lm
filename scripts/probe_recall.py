"""체크포인트의 회상 층을 다시 재고, 원하면 그 결과를 체크포인트에 기록한다.

  python scripts/probe_recall.py --ckpt runs/1b/best.pt --tokenizer tokenizer/korean_spm_v2.model
  python scripts/probe_recall.py --ckpt … --tokenizer … --write            # recall 기록을 측정 결과로 바꾼다
  python scripts/probe_recall.py --ckpt … --tokenizer … --K 4,8,16 --method induction

SFT 를 거친 체크포인트에서도 잴 수 있다 (게이트는 측정 동안 끈다 — 측정 문항에는 <|bot|> 이 없다).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from bitgc.checkpoint import atomic_save, load_checkpoint, tokenizer_info, tokenizer_path  # noqa: E402
from bitgc.recall_probe import RecallProbe, RecallTracker  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="", help="비우면 체크포인트 폴더의 tokenizer.model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--method", choices=["auto", "assoc", "induction"], default="auto")
    ap.add_argument("--K", default="4", help="연상 회상 줄 수 (쉼표로 여러 개 — 기록에는 첫 값)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    import sentencepiece as spm
    a.tokenizer = a.tokenizer or tokenizer_path(a.ckpt)
    assert a.tokenizer, "--tokenizer 가 필요하다 (체크포인트 폴더에 tokenizer.model 없음)"
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    net, ck = load_checkpoint(a.ckpt, a.device, 0.0)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if a.device.startswith("cuda") else contextlib.nullcontext
    tok = tokenizer_info(sp, a.tokenizer)
    old = ck.get("recall") or {}
    first = None
    for K in [int(k) for k in a.K.split(",")]:
        pr = RecallProbe(sp, n_items=a.n, K=K, method=a.method)
        res = pr.run(net, a.device, ac)
        tr = RecallTracker(gate_token_id=tok["bot_id"])
        tr.update(-1, res)
        print(f"K={K} {tr.summary(res)}")
        first = first or tr
    print(f"체크포인트 기록: 회상 층 b{old.get('layer')} · 게이트 {old.get('gate_layers')}")
    if a.write:
        rec = first.record()
        rec["history"] = list(old.get("history") or []) + rec["history"]
        ck["recall"] = rec
        ck["tokenizer"] = ck.get("tokenizer") or tok
        if os.path.isdir(a.ckpt):            # save_pretrained 폴더 — config.json 만 고친다
            import json
            cfg = json.load(open(os.path.join(a.ckpt, "config.json"), encoding="utf-8"))
            cfg["recall"] = rec
            with open(os.path.join(a.ckpt, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=1, default=str)
        else:
            atomic_save(ck, a.ckpt)
        print(f"기록함: 회상 층 b{rec['layer']} → 게이트 {rec['gate_layers']}")


if __name__ == "__main__":
    main()
