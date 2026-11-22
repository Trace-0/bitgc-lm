"""원 연구 코드(logiformer, `xnor_lm_test.py --model bitnet --mixer gconv …` / `sft_1b.py`)의 체크포인트를
이 저장소 형식(bitgc-v1)으로 바꾼다. 가중치 키가 같아서 그대로 옮기고, 설정·토크나이저 정보·회상 층 기록을 붙인다.

  python scripts/convert_logiformer.py --src instr2_bitgc3g.pt --tokenizer tokenizer/korean_spm_v2.model \
      --out bitgc_1b_chat.pt --recall-layer 11          # 이미 아는 회상 층을 기록 (원 연구 §26.1)
  python scripts/convert_logiformer.py --src 1b_best.pt --tokenizer … --out bitgc_1b_base.pt --probe
                                                         # 변환하면서 회상 층을 측정해 기록

지원 범위: StdGPT 계열 중 bitnet(γ=tensor) · 삼진 · gconv 하이브리드 · ReLU²/GELU · rscale (wire 의 다른 항목은 없음).
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from bitgc.checkpoint import FORMAT, atomic_save, tokenizer_info  # noqa: E402
from bitgc.model import BitGCLM, ModelConfig  # noqa: E402


def config_from_logiformer(arch: dict, vocab: int) -> ModelConfig:
    assert arch.get("family") == "gpt" and arch.get("bitnet") == "tensor", \
        f"지원하지 않는 원 연구 체크포인트: family={arch.get('family')} bitnet={arch.get('bitnet')}"
    wire = {w.strip() for w in (arch.get("std_wire") or "").split(",") if w.strip()}
    assert wire <= {"rscale"}, f"wire {wire} 미지원 (rscale 만)"
    assert (arch.get("wquant") or "tern") == "tern" and not arch.get("dilate_cap"), "삼진·팽창 상한 없음만 지원"
    mixer = arch.get("mixer") or "attn"
    attn = list(arch.get("attn_layers") or []) if mixer == "gconv" else list(range(arch["layers"]))
    act = arch.get("ffn_act") or "gelu"
    assert act in ("relu2", "gelu"), act
    return ModelConfig(vocab_size=vocab, d_model=arch["d_model"], n_heads=arch["heads"], n_layers=arch["layers"],
                       d_ff=arch["d_ff"], seq_len=arch["seq_len"], attn_layers=tuple(attn),
                       conv_kernel=arch.get("conv_kernel") or 9, conv_dilate=bool(arch.get("conv_dilate")),
                       ffn_act=act, rscale="rscale" in wire, dropout=0.0)


def load_logiformer(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict") or ck.get("best_state") or ck["model"]
    vocab = sd["tok_emb.weight"].shape[0]
    cfg = config_from_logiformer(ck["arch"], vocab)
    net = BitGCLM(cfg)
    net.load_state_dict(sd, strict=True)
    return net, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recall-layer", type=int, default=-1, help="이미 아는 회상 층 (측정 없이 기록)")
    ap.add_argument("--probe", action="store_true", help="변환하면서 회상 층을 측정해 기록")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    net, ck = load_logiformer(a.src)
    tok = tokenizer_info(sp, a.tokenizer)
    recall = None
    if a.probe:
        from bitgc.recall_probe import RecallProbe, RecallTracker
        net.to(a.device)
        tr = RecallTracker(gate_token_id=tok["bot_id"])
        res = RecallProbe(sp).run(net, a.device)
        tr.update(-1, res)
        print(tr.summary(res))
        recall = tr.record()
        net.to("cpu")
    if a.recall_layer >= 0:
        assert a.recall_layer in net.attn_ids, f"{a.recall_layer} 는 attention 층이 아니다 ({net.attn_ids})"
        if recall and recall["layer"] != a.recall_layer:
            print(f"⚠ 측정한 회상 층 b{recall['layer']} 대신 지정한 b{a.recall_layer} 를 기록한다")
        recall = dict(recall or {}, layer=a.recall_layer, gate_layers=[a.recall_layer], token="<|bot|>",
                      token_id=tok["bot_id"], method=(recall or {}).get("method", "given"))
    meta = {"converted_from": os.path.basename(a.src),
            "src_best_val_ppl": ck.get("best_val_ppl"), "src_sft": bool(ck.get("sft"))}
    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    atomic_save({"format": FORMAT, "config": net.cfg.to_dict(), "state_dict": sd, "recall": recall,
                 "tokenizer": tok, "meta": meta}, a.out)
    print(f"저장 {a.out} | {net.cfg.n_layers}층 attention {net.attn_ids} | 회상 층 기록 "
          f"{(recall or {}).get('gate_layers')}")


if __name__ == "__main__":
    main()
