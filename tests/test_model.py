"""작은 무작위 모델로 확인 (CPU, 수 초): python -m pytest tests -q   또는   python tests/test_model.py

1. 증분 생성 (KV 캐시 + conv 상태) 의 로짓 = 전체 문맥 한 번에 계산한 로짓
2. 게이트는 응답 표지 위치의 출력만 바꾼다 (그 앞 위치는 정확히 같다), 게이트 아닌 층 지정은 거부
3. knockout 은 지정 위치·층만 바꾼다
4. 저장 → 불러오기 왕복, 게이트 기록이 따라온다
5. 회상 측정·추적기가 돈다 (무작위 모델이라 값 자체는 의미 없음)
"""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from bitgc.checkpoint import load_checkpoint, make_ckpt  # noqa: E402
from bitgc.model import BitGCLM, ModelConfig  # noqa: E402

BOT = 5


def tiny(dilate=False):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=6, d_ff=64, seq_len=48,
                      attn_layers=(1, 4), conv_kernel=3, conv_dilate=dilate, dropout=0.0)
    return BitGCLM(cfg).eval()


def test_incremental_matches_full():
    for dil in (False, True):
        net = tiny(dil)
        net.set_recall_gate([4], BOT)
        x = torch.randint(7, 64, (1, 30))
        x[0, 10] = x[0, 20] = BOT
        with torch.no_grad():
            full = net(x)
            caches = net.new_caches()
            steps = [net(x[:, :12], caches=caches, pos0=0)]
            for t in range(12, 30):
                steps.append(net(x[:, t:t + 1], caches=caches, pos0=t))
        inc = torch.cat(steps, 1)
        assert torch.allclose(full, inc, atol=1e-4), (dil, (full - inc).abs().max())


def test_gate_only_changes_bot_positions():
    net = tiny()
    x = torch.randint(7, 64, (2, 40))
    x[:, 25] = BOT
    with torch.no_grad():
        a = net(x)
        net.set_recall_gate([4], BOT)
        b = net(x)
        net.gate_on = False
        c = net(x)
    assert torch.equal(a[:, :25], b[:, :25]), "게이트 앞 위치가 바뀌었다"
    assert (a[:, 25] - b[:, 25]).abs().max() > 1e-4, "게이트가 응답 표지 위치를 안 바꿨다"
    assert torch.equal(a, c), "gate_on=False 가 원래와 다르다"
    try:
        net.set_recall_gate([0], BOT)        # 0 은 gconv 층
        raise AssertionError("gconv 층 게이트를 받아들였다")
    except AssertionError as e:
        assert "attention" in str(e)


def test_knockout():
    net = tiny()
    x = torch.randint(7, 64, (2, 20))
    m = torch.zeros(2, 20, dtype=torch.bool)
    m[:, 15] = True
    with torch.no_grad():
        a, b = net(x), net(x, knockout={1: m})
    assert torch.equal(a[:, :15], b[:, :15])
    assert (a[:, 15] - b[:, 15]).abs().max() > 1e-5


def test_save_load_roundtrip():
    net = tiny()
    rec = {"layer": 4, "gate_layers": [4], "token": "<|bot|>", "token_id": BOT, "method": "assoc"}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        torch.save(make_ckpt(net, recall=rec, tokenizer={"bot_id": BOT}), p)
        net2, ck = load_checkpoint(p)
    net2.eval()
    assert net2.gate_layers == (4,) and net2.gate_token == BOT
    x = torch.randint(7, 64, (1, 20))
    with torch.no_grad():
        net.set_recall_gate([4], BOT)
        assert torch.equal(net(x), net2(x))


def test_probe_runs():
    import sentencepiece as spm
    # 토크나이저는 저장소에 없다 — Hugging Face 모델 저장소의 tokenizer.model 을 tokenizer/ 에 받아 두거나 경로를 준다
    tk = os.environ.get("BITGC_TOKENIZER", os.path.join(_ROOT, "tokenizer", "tokenizer.model"))
    if not os.path.exists(tk):
        tk = os.path.join(_ROOT, "tokenizer", "korean_spm_v2.model")
    if not os.path.exists(tk):
        print("  (토크나이저 없음 — 회상 측정 시험 건너뜀)")
        return
    from bitgc.recall_probe import RecallProbe, RecallTracker
    sp = spm.SentencePieceProcessor(model_file=tk)
    cfg = ModelConfig(vocab_size=sp.get_piece_size(), d_model=32, n_heads=4, n_layers=4, d_ff=64, seq_len=64,
                      attn_layers=(1, 3), dropout=0.0)
    net = BitGCLM(cfg)
    tr = RecallTracker(gate_token_id=sp.piece_to_id("<|bot|>"))
    for m in ("assoc", "induction"):
        res = RecallProbe(sp, n_items=16, method=m).run(net)
        assert set(res["layers"]) == {1, 3} and res["leader"] in (1, 3)
        tr.update(0, res)
    rec = tr.record()
    assert rec["gate_layers"] == [rec["layer"]] and len(rec["history"]) == 2


def test_engine_matches_torch():
    """정수 엔진 (빌드돼 있을 때만): 게이트 포함 내보내기 → prefill·한 토큰씩 모두 torch 와 top-1 대부분 일치."""
    from bitgc.engine.runtime import LIB_NAME
    if not os.path.exists(os.path.join(_ROOT, "bitgc", "engine", LIB_NAME)) and not os.environ.get("BITGC_ENGINE_LIB"):
        print("  (엔진 라이브러리 없음 — python -m bitgc.engine.build 뒤에 시험)")
        return
    import numpy as np
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    from export_engine import export
    from bitgc.engine.runtime import GenLM
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=512, d_model=256, n_heads=4, n_layers=4, d_ff=512, seq_len=64,
                      attn_layers=(1, 3), dropout=0.0)          # 엔진은 d_model·d_ff 가 256 의 배수
    net = BitGCLM(cfg).eval()
    net.set_recall_gate([3], BOT)
    x = torch.randint(7, 512, (1, 40))
    x[0, 20] = x[0, 30] = BOT
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.npz")
        export(net, p, {"layer": 3, "token": BOT})
        eng = GenLM(p)
        with torch.no_grad():
            ref = net(x)[0].numpy()
        pre = eng.forward(x[0].tolist())
        eng.reset()
        inc = np.stack([eng.step(int(t)) for t in x[0]])
    for g in (pre, inc):
        agree = (ref.argmax(1) == g.argmax(1)).mean()
        assert agree > 0.95 and np.abs(ref - g).max() < 0.1, (agree, np.abs(ref - g).max())
    # 게이트가 엔진에서도 걸렸는지: 게이트 없는 torch 와는 응답 표지 위치에서 달라야 한다
    net.gate_on = False
    with torch.no_grad():
        nog = net(x)[0].numpy()
    assert np.abs(nog[30] - pre[30]).max() > 10 * np.abs(ref[30] - pre[30]).max()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
