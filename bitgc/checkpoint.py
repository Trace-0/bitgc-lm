"""체크포인트 형식 — 가중치 + 모델 설정 + 회상 층 기록 + 토크나이저 정보.

  {
    "format": "bitgc-v1",
    "config": ModelConfig.to_dict(),
    "state_dict": {...},
    "recall": {              # bitgc.recall_probe.RecallTracker.record() — 없으면 게이트 없음
        "layer": 11, "gate_layers": [11], "token": "<|bot|>", "token_id": 5,
        "method": "assoc", "decided_step": 8000, "history": [...], ...
    },
    "tokenizer": {"file": "korean_spm_v2.model", "eos_id": 3, "bos_id": 2, "pad_id": 0,
                  "user_id": 4, "bot_id": 5, "end_id": 6},
    "meta": {...}            # 학습 단계·val ppl 등 (자유 형식)
  }

불러올 때 recall 기록이 있으면 게이트를 자동으로 건다 (`apply_gate=False` 로 끌 수 있다).
"""
from __future__ import annotations

import os

import torch

from .model import BitGCLM, ModelConfig

FORMAT = "bitgc-v1"


def atomic_save(obj, path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def cpu_state(net) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}


def make_ckpt(net: BitGCLM, state_dict=None, recall=None, tokenizer=None, meta=None) -> dict:
    return {"format": FORMAT, "config": net.cfg.to_dict(),
            "state_dict": state_dict if state_dict is not None else cpu_state(net),
            "recall": recall, "tokenizer": tokenizer or {}, "meta": meta or {}}


def save_pretrained(ck: dict, out_dir: str, tokenizer_file: str = "") -> None:
    """배포용 폴더: model.safetensors (fp32, lm_head 는 tok_emb 와 공유라 빼고 저장) + config.json
    (+ tokenizer.model). pickle 이 없어 Hugging Face 등에 올리기 안전하다. `load_checkpoint` 가 폴더도 읽는다."""
    import json
    import shutil

    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().to(torch.float32).contiguous() for k, v in ck["state_dict"].items() if k != "lm_head.weight"}
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": FORMAT})
    cfg = {k: ck.get(k) for k in ("format", "config", "recall", "tokenizer", "meta")}
    if tokenizer_file:
        shutil.copyfile(tokenizer_file, os.path.join(out_dir, "tokenizer.model"))
        cfg["tokenizer"] = dict(cfg.get("tokenizer") or {}, file="tokenizer.model")
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1, default=str)


def resolve(path: str) -> str:
    """'hf:사용자/저장소' 면 Hugging Face 에서 받아 로컬 폴더 경로를 돌려준다."""
    if path.startswith("hf:"):
        from huggingface_hub import snapshot_download
        return snapshot_download(path[3:])
    return path


def tokenizer_path(path: str) -> str:
    """체크포인트 폴더(또는 hf:)에 tokenizer.model 이 있으면 그 경로."""
    p = os.path.join(resolve(path), "tokenizer.model")
    return p if os.path.isdir(resolve(path)) and os.path.exists(p) else ""


def load_checkpoint(path: str, device="cpu", dropout: float = 0.0, apply_gate: bool = True):
    """→ (net, ck). path = .pt 파일 · save_pretrained 폴더 · 'hf:사용자/저장소'.
    net 은 eval 모드가 아니다 — 필요한 쪽에서 .eval()."""
    path = resolve(path)
    if os.path.isdir(path):
        import json

        from safetensors.torch import load_file
        ck = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
        sd = load_file(os.path.join(path, "model.safetensors"))
        sd["lm_head.weight"] = sd["tok_emb.weight"]
        ck["state_dict"] = sd
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck.get("format") == FORMAT, (
        f"{path}: bitgc 체크포인트가 아니다 (원 연구 코드 체크포인트는 scripts/convert_logiformer.py 로 변환)")
    cfg = ModelConfig.from_dict(dict(ck["config"], dropout=dropout))
    net = BitGCLM(cfg)
    net.load_state_dict(ck["state_dict"], strict=True)
    net.to(device)
    if apply_gate:
        apply_recall_gate(net, ck)
    return net, ck


def apply_recall_gate(net: BitGCLM, ck: dict) -> bool:
    """체크포인트의 회상 층 기록대로 게이트를 건다. 건 경우 True."""
    rec = ck.get("recall") or {}
    layers = rec.get("gate_layers")
    tok = rec.get("token_id")
    if tok is None:
        tok = (ck.get("tokenizer") or {}).get("bot_id")
    if layers and tok is not None:
        net.set_recall_gate(layers, tok)
        return True
    net.set_recall_gate((), None)
    return False


def tokenizer_info(sp, file: str = "") -> dict:
    def pid(p):
        i = sp.piece_to_id(p)
        return None if i == sp.unk_id() else int(i)
    return {"file": os.path.basename(file) if file else "",
            "vocab_size": int(sp.get_piece_size()),
            "eos_id": int(sp.eos_id()), "bos_id": int(sp.bos_id()), "pad_id": int(sp.pad_id()),
            "user_id": pid("<|user|>"), "bot_id": pid("<|bot|>"), "end_id": pid("<|end|>")}
