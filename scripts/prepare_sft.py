"""대화 jsonl → SFT 텐서 파일 (.pt). 원 연구 `sft_data_v2.py` 와 같은 포장 규칙.

  python scripts/prepare_sft.py --input chat.jsonl --tokenizer tokenizer/korean_spm_v2.model \
      --out data/sft_chat.pt --seq-len 512 [--val-input heldout.jsonl]

입력 한 줄: {"turns": [["user", "질문"], ["bot", "답"], …]}  (역할은 user/bot, dict {"role","text"} 도 받는다)
토큰열:    [bos] <|user|> 질문 <|end|> <|bot|> 답 <|end|> …   (대화를 seq_len 까지 이어 붙인다)
학습 신호: bot 턴의 본문과 <|end|> 만 (user 턴·패딩은 -100). <|bot|> 위치의 정답 = 답의 첫 토큰 —
          회상 층 게이트가 막는 바로 그 위치다.
⚠ val 은 사전학습 코퍼스에 들어가지 않은 대화로 (--val-input). 사전학습이 본 대화로 재면 SFT 전 ppl 이 낮게 나온다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random

import torch

IGNORE = -100


def read(path):
    out = []
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        try:
            turns = json.loads(ln).get("turns") or []
        except json.JSONDecodeError:
            continue
        tt = []
        for t in turns:
            who, text = (t[0], t[1]) if isinstance(t, (list, tuple)) else (t.get("role"), t.get("text"))
            who = "user" if who in ("user", "human", "사용자") else "bot"
            if text:
                tt.append((who, text))
        if len(tt) >= 2:
            out.append(tt)
    return out


def pack(rows, sp, seq_len, u, b, e, bos, pad):
    X, Y = [], []
    cx, cy = [bos], [IGNORE]
    n_drop = 0

    def flush():
        nonlocal cx, cy
        if len(cx) <= 1:
            return
        xs, ys = cx[:seq_len], cy[1:seq_len + 1]
        X.append(xs + [pad] * (seq_len - len(xs)))
        Y.append(ys + [IGNORE] * (seq_len - len(ys)))
        cx, cy = [bos], [IGNORE]

    for turns in rows:
        sx, sy = [], []
        for who, text in turns:
            ids = sp.encode(text)
            if who == "user":
                sx += [u] + ids + [e]
                sy += [IGNORE] * (len(ids) + 2)
            else:
                sx += [b] + ids + [e]
                sy += [IGNORE] + ids + [e]
        if len(sx) > seq_len:
            sx, sy = sx[:seq_len], sy[:seq_len]
            if all(t == IGNORE for t in sy):
                n_drop += 1
                continue
        if len(cx) + len(sx) > seq_len + 1:
            flush()
        cx += sx
        cy += sy
    flush()
    return torch.tensor(X, dtype=torch.int16 if sp.get_piece_size() < 32768 else torch.int32), \
        torch.tensor(Y, dtype=torch.int16 if sp.get_piece_size() < 32768 else torch.int32), n_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--val-input", default="", help="val 로만 쓸 대화 (없으면 끝 5% 또는 --n-val)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    pid = sp.piece_to_id
    u, b, e = pid("<|user|>"), pid("<|bot|>"), pid("<|end|>")
    assert min(u, b, e) != sp.unk_id(), "토크나이저에 <|user|> <|bot|> <|end|> 가 없다 (scripts/train_tokenizer.py)"
    rows = [t for p in a.input for t in read(p)]
    random.Random(a.seed).shuffle(rows)
    x, y, nd = pack(rows, sp, a.seq_len, u, b, e, sp.bos_id(), sp.pad_id())
    if a.val_input:
        vx, vy, _ = pack(read(a.val_input), sp, a.seq_len, u, b, e, sp.bos_id(), sp.pad_id())
        vx, vy = vx[:a.n_val], vy[:a.n_val]
        tx, ty = x, y
    else:
        nv = min(a.n_val, max(1, len(x) // 20))
        tx, ty, vx, vy = x[:-nv], y[:-nv], x[-nv:], y[-nv:]
    n_sup = int((ty != IGNORE).sum())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"train_x": tx, "train_y": ty, "val_x": vx, "val_y": vy, "vocab_size": sp.get_piece_size(),
                "markers": {"user": u, "bot": b, "end": e}, "meta": {"n_supervised": n_sup, "seq_len": a.seq_len,
                                                                      "tokenizer": os.path.basename(a.tokenizer)}},
               a.out)
    print(f"저장 {a.out} — 대화 {len(rows):,} → train {len(tx):,} / val {len(vx):,} 시퀀스 · 학습 신호 "
          f"{n_sup/1e6:.2f}M 토큰 (너무 길어 버림 {nd})")


if __name__ == "__main__":
    main()
