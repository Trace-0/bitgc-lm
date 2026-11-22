"""사전학습 코퍼스 → uint16 토큰 파일 (.bin + .meta.json). 문서마다 끝에 eos 를 붙인다.

  python scripts/prepare_pretrain.py --input wiki.jsonl docs/*.txt --tokenizer tokenizer/korean_spm_v2.model \
      --out data/corpus.bin [--shuffle-docs]

입력: .jsonl ({"text": …} 한 줄 = 한 문서) 또는 .txt (빈 줄로 문서 구분).
⚠ train.py 는 파일 끝 2% (최대 8000 시퀀스) 를 val 로 쓴다 — 소스별로 이어 붙이면 val 이 한 소스뿐이 되므로
  여러 소스는 --shuffle-docs 로 섞을 것 (원 연구 v2 코퍼스에서 val 이 전부 대화였던 사고).
"""
from __future__ import annotations

import argparse
import io
import json
import random

import numpy as np


def docs(paths):
    for p in paths:
        if p.endswith(".jsonl"):
            for ln in io.open(p, encoding="utf-8", errors="replace"):
                try:
                    t = json.loads(ln).get("text", "")
                except json.JSONDecodeError:
                    continue
                if t.strip():
                    yield t
        else:
            buf = []
            for ln in io.open(p, encoding="utf-8", errors="replace"):
                if ln.strip():
                    buf.append(ln.rstrip("\n"))
                elif buf:
                    yield "\n".join(buf)
                    buf = []
            if buf:
                yield "\n".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle-docs", action="store_true", help="문서 순서를 섞는다 (전부 메모리에 올린다)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    assert sp.get_piece_size() <= 65536, "uint16 에 안 들어가는 vocab"
    eos = sp.eos_id()
    it = docs(a.input)
    if a.shuffle_docs:
        it = list(it)
        random.Random(a.seed).shuffle(it)
    n_tok = n_doc = 0
    with io.open(a.out, "wb") as w:
        buf, chunk = [], []
        for d in it:
            buf.append(d)
            if len(buf) >= 1000:
                for ids in sp.encode(buf):
                    chunk.extend(ids + [eos])
                n_doc += len(buf)
                buf = []
            if len(chunk) >= 4_000_000:
                np.asarray(chunk, dtype=np.uint16).tofile(w)
                n_tok += len(chunk)
                chunk = []
        for ids in sp.encode(buf):
            chunk.extend(ids + [eos])
        n_doc += len(buf)
        np.asarray(chunk, dtype=np.uint16).tofile(w)
        n_tok += len(chunk)
    meta = {"vocab_size": sp.get_piece_size(), "n_tokens": n_tok, "n_docs": n_doc, "tokenizer": a.tokenizer}
    io.open(a.out.replace(".bin", ".meta.json"), "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False))
    print(f"저장 {a.out} — 문서 {n_doc:,} · {n_tok/1e6:.1f}M 토큰")


if __name__ == "__main__":
    main()
