"""BitGC 체크포인트 → 정수 엔진 자산 (.npz) + 정합 검사.

  python -m bitgc.engine.build
  python scripts/export_engine.py --ckpt hf:trace-000/bitgc-lm-1b --out bitgc_1b.npz --parity

담는 것: 삼진 가중치를 64개씩 비트마스크 두 장(+1 / −1)으로 묶은 것 · γ · 정규화 · conv 커널 · 잔차 스칼라 · 임베딩(fp32).
fc2 는 열 단위로 저장한다 → ReLU² 출력이 0 인 입력의 열은 읽지 않는다 (토큰당 메모리 읽기 −35%).
회상 층 게이트는 체크포인트의 recall 기록에서 가져와 meta 에 넣는다 (엔진은 한 층만 지원).
제약: d_model · d_ff 가 256 의 배수 (configs/ 의 세 설정은 모두 맞다).

--parity: 같은 입력을 torch(BitGCLM, fp32)와 엔진(prefill · 한 토큰씩 둘 다)에 넣어 top-1 일치와 NLL 을 비교한다.
  8bit 활성 양자화의 반올림 경계 때문에 완전히 같지는 않다 (원 연구 1B: top-1 97~98%, NLL 차 < 0.01).
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # numpy BLAS 스레드가 엔진의 OpenMP 와 경합하면 생성이 ~1.5배 느려진다

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bitgc.checkpoint import load_checkpoint, tokenizer_path  # noqa: E402

PARITY_TEXT = [("내 이름은 민수야. 기억해 줄래?", "네, 민수 님. 기억할게요."),
               ("오늘 날씨가 좋아서 공원에 산책을 다녀왔어.", "좋네요! 산책하면 기분도 상쾌해지죠."),
               ("세종대왕은 어떤 업적을 남겼어?", "세종대왕은 훈민정음을 만들고 과학 기술을 발전시켰습니다."),
               ("내 이름이 뭐라고 했지?", None)]


def pack_ternary(wq: np.ndarray):
    """(N, K) {-1,0,1} → (N, K/64, 2) uint64 [양, 음] 비트마스크 + Σw (int32). 비트 j = 원소 64·kb + j."""
    N, K = wq.shape
    assert K % 64 == 0, "K 는 64 의 배수여야 한다"
    w = wq.reshape(N, K // 64, 64)
    bitw = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    pos = ((w == 1).astype(np.uint64) * bitw).sum(axis=2, dtype=np.uint64)
    neg = ((w == -1).astype(np.uint64) * bitw).sum(axis=2, dtype=np.uint64)
    wsum = (wq == 1).sum(1).astype(np.int32) - (wq == -1).sum(1).astype(np.int32)
    return np.ascontiguousarray(np.stack([pos, neg], axis=2)), wsum


def _f32(t):
    return np.ascontiguousarray(t.detach().cpu().float().numpy().astype(np.float32))


def put_linear(store, tag, lin, cols=False):
    """BitLinear → 삼진 비트 + γ (행렬 하나에 γ 하나, 행마다 같은 값으로 펼친다) + 양자화 직전 정규화."""
    w = lin.weight.detach().cpu().float()
    g = (w.abs().mean() * torch.ones(w.shape[0], 1)).clamp(min=1e-5)
    wq = (w / g).round().clamp(-1, 1).to(torch.int8).numpy()
    if cols:
        store[f"{tag}.cbits"] = pack_ternary(np.ascontiguousarray(wq.T))[0]
    else:
        store[f"{tag}.bits"], store[f"{tag}.wsum"] = pack_ternary(wq)
    store[f"{tag}.gamma"] = g[:, 0].numpy().astype(np.float32)


def _norm(mod, store, tag):
    store[f"{tag}.w"] = _f32(mod.weight)
    if getattr(mod, "bias", None) is not None:      # 없으면 엔진이 RMSNorm 으로 본다
        store[f"{tag}.b"] = _f32(mod.bias)


def export(net, out, gate=None, source=""):
    cfg = net.cfg
    # 삼진 커널은 입력 차원을 256 단위로 묶어 계산한다 — 아니면 틀린 값이 나온다 (d 192·320 에서 실측, 256·512·2048 은 torch 와 일치)
    assert cfg.d_model % 256 == 0 and cfg.d_ff % 256 == 0,         f"엔진은 d_model·d_ff 가 256 의 배수여야 한다 (지금 {cfg.d_model}·{cfg.d_ff})"
    store = {"tok_emb": _f32(net.tok_emb.weight), "pos_emb": _f32(net.pos_emb.weight)}
    _norm(net.norm, store, "normf")
    mixers, dils = [], []
    for i, layer in enumerate(net.layers):
        _norm(layer.norm1, store, f"L{i}.n1")
        _norm(layer.norm2, store, f"L{i}.n2")
        at = layer.attn
        if layer.is_attn:
            mixers.append("attn")
            dils.append(1)
            wa = at.qkv
        else:
            mixers.append("gconv")
            dils.append(int(at.mix.d))
            wa = at.W_v
            store[f"L{i}.mix"] = _f32(at.mix.weight).reshape(at.mix.weight.shape[0], -1)
            store[f"L{i}.gate"] = _f32(at.gate.weight).reshape(at.gate.weight.shape[0], -1)
        for name, lin, pre in (("WA", wa, "pre_a"), ("WO", at.W_o, "pre_o"),
                               ("F1", layer.ffn.fc1, "pre_f1"), ("F2", layer.ffn.fc2, "pre_fc2")):
            put_linear(store, f"L{i}.{name}", lin, cols=(name == "F2" and cfg.ffn_act == "relu2"))
            _norm(lin.norm, store, f"L{i}.{pre}")
        if layer.attn_scale is not None:
            store[f"L{i}.attn_scale"] = _f32(layer.attn_scale)
            store[f"L{i}.ffn_scale"] = _f32(layer.ffn_scale)
    meta = {"fmt": "tern", "arch": "bitnet", "act": cfg.ffn_act, "vocab": cfg.vocab_size, "d_model": cfg.d_model,
            "layers": cfg.n_layers, "heads": cfg.n_heads, "d_ff": cfg.d_ff, "seq_len": cfg.seq_len,
            "kernel": cfg.conv_kernel, "mixers": mixers, "dilations": dils, "source": source,
            "skip_fc2": cfg.ffn_act == "relu2", "bitgc_config": cfg.to_dict()}
    if gate:
        meta["gate"] = gate
    store["meta"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, **store)
    print(f"저장 {out}  {os.path.getsize(out)/1e6:.0f}MB | {cfg.n_layers}층 · 게이트 {gate}", flush=True)


@torch.no_grad()
def parity(net, npz, ids):
    from bitgc.engine.runtime import GenLM
    eng = GenLM(npz)
    x = ids[:-1]
    ref = net(torch.tensor([x])).float().numpy()[0]
    t1 = time.time()
    got = eng.forward(x)
    t2 = time.time()
    eng.reset()
    inc = np.stack([eng.step(t) for t in x])
    t3 = time.time()

    def nll(lg):
        lg = lg - lg.max(1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        return float(-lp[np.arange(len(x)), ids[1:]].mean())
    gpos = [i for i, t in enumerate(x) if t == eng.gate_tok]
    for name, g in (("prefill", got), ("한 토큰씩", inc)):
        top = 100 * (ref.argmax(1) == g.argmax(1)).mean()
        gtop = 100 * (ref[gpos].argmax(1) == g[gpos].argmax(1)).mean() if gpos else float("nan")
        print(f"정합 [{name}] top-1 {top:.1f}% (게이트 위치 {len(gpos)}개 {gtop:.1f}%) | NLL torch {nll(ref):.4f} "
              f"엔진 {nll(g):.4f}", flush=True)
    print(f"  엔진 prefill {len(x)/(t2-t1):.0f} tok/s · 한 토큰씩 {1000*(t3-t2)/len(x):.1f} ms/tok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".pt · save_pretrained 폴더 · hf:사용자/저장소")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--no-gate", action="store_true", help="회상 층 기록이 있어도 게이트를 넣지 않는다")
    ap.add_argument("--parity", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    net, ck = load_checkpoint(a.ckpt, "cpu", 0.0, apply_gate=not a.no_gate)
    net.eval()
    gate = None
    if net.gate_layers:
        assert len(net.gate_layers) == 1, "엔진 게이트는 한 층만 지원한다"
        gate = {"layer": int(net.gate_layers[0]), "token": int(net.gate_token)}
    export(net, a.out, gate, os.path.basename(a.ckpt.rstrip("/")))
    if a.parity:
        import sentencepiece as spm
        tk = a.tokenizer or tokenizer_path(a.ckpt)
        assert tk, "--parity 에는 --tokenizer 가 필요하다"
        from bitgc.generate import Generator
        sp = spm.SentencePieceProcessor(model_file=tk)
        ids = Generator(net, sp).chat_ids(PARITY_TEXT)
        ids = ids + sp.encode("민수라고 하셨어요.")
        parity(net, a.out, ids)


if __name__ == "__main__":
    main()
