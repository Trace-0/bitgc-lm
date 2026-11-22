"""추론 — 체크포인트에 남은 회상 층을 응답 첫 토큰 위치에서 막고 생성한다.

  python chat.py --ckpt bitgc_1b_chat.pt --tokenizer tokenizer/korean_spm_v2.model            # 대화 (여러 턴)
  python chat.py --ckpt … --tokenizer … --prompt "프랑스의 수도는 어디야?"                     # 한 번
  python chat.py --ckpt … --tokenizer … --prompt "…" --compare-gate                           # 게이트 있음/없음 나란히
  python chat.py --ckpt … --tokenizer … --text "옛날 옛적에"                                   # 대화 형식 없이 이어 쓰기

대화 명령: /reset (대화 비우기) · /gate on|off · /quit
"""
from __future__ import annotations

import argparse
import sys

import torch

from bitgc.checkpoint import load_checkpoint, tokenizer_path
from bitgc.generate import Generator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="", help="비우면 체크포인트 폴더의 tokenizer.model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true", help="GPU 에서 bf16 autocast (기본 fp32 — 평가와 같은 경로)")
    ap.add_argument("--prompt", default="", help="한 번만 묻고 끝낸다")
    ap.add_argument("--text", default="", help="대화 형식 없이 이어 쓰기")
    ap.add_argument("--compare-gate", action="store_true", help="게이트 있음/없음 두 답을 나란히")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (평가 설정)")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--rep", type=float, default=1.2, help="반복 벌점")
    ap.add_argument("--no-repeat", type=int, default=6, help="같은 n-gram 금지")
    ap.add_argument("--format", choices=["eval", "sft"], default="eval", help="대화 토큰열 형식 (bitgc/generate.py)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import sentencepiece as spm
    a.tokenizer = a.tokenizer or tokenizer_path(a.ckpt)
    assert a.tokenizer, "--tokenizer 가 필요하다 (체크포인트 폴더에 tokenizer.model 없음)"
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    net, ck = load_checkpoint(a.ckpt, a.device, 0.0, apply_gate=not a.no_gate)
    gen = Generator(net, sp, a.device, torch.bfloat16 if a.bf16 else None)
    rec = ck.get("recall") or {}
    print(f"[모델] {net.cfg.n_layers}층 · attention {net.attn_ids} · 회상 층 기록 b{rec.get('layer')} "
          f"({rec.get('method')}) → 게이트 {list(net.gate_layers) or '없음'}", file=sys.stderr, flush=True)
    kw = dict(max_new=a.max_new, temperature=a.temperature, top_k=a.top_k, rep=a.rep, no_repeat=a.no_repeat,
              seed=a.seed)

    if a.text:
        out = gen.generate([sp.eos_id()] + sp.encode(a.text), chat_rules=False, **kw)
        print(a.text + sp.decode(out))
        return
    if a.prompt:
        if a.compare_gate:
            for on in (True, False):
                net.gate_on = on
                print(f"[게이트 {'켬' if on else '끔'}] {gen.reply([(a.prompt, None)], a.format, **kw)}")
        else:
            print(gen.reply([(a.prompt, None)], a.format, **kw))
        return

    turns = []
    print("대화를 시작합니다 (/reset · /gate on|off · /quit)", file=sys.stderr)
    while True:
        try:
            u = input("사용자> ").strip()
        except EOFError:
            break
        if not u:
            continue
        if u == "/quit":
            break
        if u == "/reset":
            turns = []
            continue
        if u.startswith("/gate"):
            net.gate_on = u.endswith("on")
            print(f"(게이트 {'켬' if net.gate_on else '끔'})")
            continue
        r = gen.reply(turns + [(u, None)], a.format, **kw)
        print(f"모델> {r}")
        turns.append((u, r))


if __name__ == "__main__":
    main()
