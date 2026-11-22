"""데이터 — 사전학습 토큰 파일(.bin + .meta.json) · SFT 텐서 파일(.pt).

사전학습: uint16 토큰 열 (문서마다 끝에 eos). `scripts/prepare_pretrain.py` 가 만든다.
  i 번째 시퀀스 x = ids[i·T : (i+1)·T], y = ids[i·T+1 : (i+1)·T+1]. 끝 2% (최대 8000 시퀀스) 가 val.
SFT: {"train_x","train_y","val_x","val_y","vocab_size","meta"} — y 는 학습하지 않는 자리가 -100.
  `scripts/prepare_sft.py` 가 만든다.
"""
from __future__ import annotations

import json

import numpy as np
import torch

IGNORE = -100


class BinSeqDataset(torch.utils.data.Dataset):
    """memmap 위에서 요청한 시퀀스만 잘라 준다 (2B 토큰도 RAM 에 안 올린다)."""

    def __init__(self, arr, first: int, count: int, seq_len: int):
        self.arr, self.first, self.count, self.T = arr, first, count, seq_len

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        o = (self.first + i) * self.T
        w = torch.from_numpy(np.asarray(self.arr[o:o + self.T + 1], dtype=np.int64))
        return w[:-1], w[1:]


def load_pretrain(path: str, seq_len: int, max_seqs: int = 0, val_frac: float = 0.02, max_val: int = 8000):
    meta = json.load(open(path.replace(".bin", ".meta.json"), encoding="utf-8"))
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    n_use = arr.size if max_seqs <= 0 else min(arr.size, (max_seqs + 1) * seq_len + 1)
    n_seq = (n_use - 1) // seq_len
    n_val = max(1, min(max_val, int(n_seq * val_frac)))
    tr = BinSeqDataset(arr, 0, n_seq - n_val, seq_len)
    va = BinSeqDataset(arr, n_seq - n_val, n_val, seq_len)
    return tr, va, meta


def load_sft(path: str):
    p = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("train_x", "train_y", "val_x", "val_y", "vocab_size"):
        assert k in p, f"{path}: SFT 파일에 '{k}' 가 없다"
    return p
