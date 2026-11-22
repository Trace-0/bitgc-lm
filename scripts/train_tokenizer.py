"""SentencePiece 토크나이저 학습 — 원 연구 `korean_spm_v2` 와 같은 설정.

  python scripts/train_tokenizer.py --input corpus1.txt corpus2.jsonl --out tokenizer/my_spm --vocab 16384

입력: .txt (한 줄 = 한 문장/문단) 또는 .jsonl ({"text": …} 또는 {"turns": [[역할, 말], …]}).
설정: BPE · byte_fallback (UNK 없음) · nfkc (줄바꿈 보존) · 대화 표지 <|user|> <|bot|> <|end|> ·
      unk 1 · bos 2 · eos 3 · pad 0.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random


def lines(paths, max_lines, seed):
    rng = random.Random(seed)
    out = []
    for p in paths:
        for ln in io.open(p, encoding="utf-8", errors="replace"):
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            if p.endswith(".jsonl"):
                try:
                    o = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "text" in o:
                    out.append(o["text"])
                for t in o.get("turns") or []:
                    out.append(t[1] if isinstance(t, (list, tuple)) else t.get("text", ""))
            else:
                out.append(ln)
    rng.shuffle(out)
    return out[:max_lines] if max_lines else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="출력 접두 (…/name → name.model · name.vocab)")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--max-lines", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import sentencepiece as spm
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + "_corpus.txt"
    with io.open(tmp, "w", encoding="utf-8") as f:
        for t in lines(a.input, a.max_lines, a.seed):
            f.write(t.replace("\n", " ") + "\n")
    spm.SentencePieceTrainer.train(
        input=tmp, model_prefix=a.out, vocab_size=a.vocab, model_type="bpe",
        character_coverage=0.9995, byte_fallback=True,
        remove_extra_whitespaces=False, normalization_rule_name="nfkc",   # 줄바꿈을 지우지 않는다
        unk_id=1, bos_id=2, eos_id=3, pad_id=0,
        user_defined_symbols=["<|user|>", "<|bot|>", "<|end|>"],
        input_sentence_size=2_000_000, shuffle_input_sentence=True, num_threads=os.cpu_count() or 8)
    os.remove(tmp)
    sp = spm.SentencePieceProcessor(model_file=a.out + ".model")
    s = "<|user|>안녕?<|end|><|bot|>네, 안녕하세요.\n두 번째 줄<|end|>"
    assert sp.decode(sp.encode(s)) == s, "왕복 불일치"
    print(f"저장 {a.out}.model (vocab {sp.get_piece_size()}) · 왕복 검사 통과")


if __name__ == "__main__":
    main()
