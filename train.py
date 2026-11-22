"""사전학습 — 학습 중간에 회상 층을 자동으로 재고, 그 위치를 체크포인트에 남긴다.

  python train.py --data data/corpus.bin --tokenizer tokenizer/korean_spm_v2.model \
      --config configs/bitgc_1b.json --out-dir runs/1b --lr 1.5e-4 --batch-size 16 \
      --warmup 3000 --amp bf16 --compile --grad-ckpt --eval-every 8000 --ckpt-every 4000 --probe-every 500

산출 (--out-dir):
  best.pt     val 이 가장 좋은 가중치 + 설정 + **회상 층 기록 (recall)** + 토크나이저 정보 (bitgc-v1)
  last.pt     마지막 가중치 (같은 형식)
  resume.pt   옵티마이저·RNG·데이터 위치 포함 — 같은 명령을 다시 실행하면 이어진다
  recall.jsonl 회상 측정 기록 (측정마다 한 줄)
  train.log   로그

회상 층 측정 (`bitgc/recall_probe.py`): --probe-every step 마다, 그리고 평가·저장·끝에서.
  어느 attention 층이 문맥 회상을 맡는지는 학습 초반(원 연구 30M: step 1000 ≈ 전체의 8%, 1B: step 8000 이전)에
  정해지고 그 뒤로 바뀌지 않았다.
  기록은 언제나 **마지막 측정** 을 따른다 — best.pt 를 저장할 때 그 자리에서 다시 재서 그 가중치에 맞춘다.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from bitgc.checkpoint import FORMAT, atomic_save, cpu_state, tokenizer_info
from bitgc.data import load_pretrain
from bitgc.model import BitGCLM, ModelConfig, compile_layers, n_params
from bitgc.recall_probe import RecallProbe, RecallTracker

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_config(a, vocab: int) -> ModelConfig:
    d = {}
    if a.config:
        d = json.load(open(a.config, encoding="utf-8"))
    for k in ("d_model", "n_heads", "n_layers", "d_ff", "seq_len", "conv_kernel", "ffn_act"):
        v = getattr(a, k)
        if v is not None:
            d[k] = v
    if a.attn_layers is not None:
        d["attn_layers"] = [int(v) for v in a.attn_layers.split(",") if v.strip()]
    if a.conv_dilate:
        d["conv_dilate"] = True
    if a.no_rscale:
        d["rscale"] = False
    d["vocab_size"] = vocab
    d["dropout"] = a.dropout
    return ModelConfig.from_dict(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="uint16 토큰 .bin (옆에 .meta.json)")
    ap.add_argument("--tokenizer", required=True, help="SentencePiece .model (회상 측정 문항·응답 표지)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="", help="모델 설정 json (configs/). 아래 인자가 덮어쓴다")
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--n-heads", type=int)
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--d-ff", type=int)
    ap.add_argument("--seq-len", type=int)
    ap.add_argument("--attn-layers", help="쉼표 구분 attention 층 (예: 3,7,11,15,19,23)")
    ap.add_argument("--conv-kernel", type=int)
    ap.add_argument("--conv-dilate", action="store_true")
    ap.add_argument("--ffn-act", choices=["relu2", "gelu"])
    ap.add_argument("--no-rscale", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)
    # 학습
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-seqs", type=int, default=0, help="코퍼스 앞 N 시퀀스만 (0 = 전량)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--final-lr-frac", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    ap.add_argument("--compile", action="store_true", help="층마다 torch.compile")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (1B 를 24GB 에)")
    ap.add_argument("--vram-gb", type=float, default=0.0, help="VRAM 하드 상한 (0 = 없음)")
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=8000)
    ap.add_argument("--ckpt-every", type=int, default=4000)
    ap.add_argument("--max-hours", type=float, default=0.0, help="이 시간이 지나면 저장하고 멈춘다")
    ap.add_argument("--stop-file", default="", help="이 파일이 생기면 저장하고 멈춘다")
    ap.add_argument("--stop-at-step", type=int, default=0,
                    help="이 전역 step 에서 저장하고 멈춘다 (lr 일정은 전체 길이 그대로 — 짧은 시험·회상 층 미리 보기)")
    # 회상 층 측정
    ap.add_argument("--probe-every", type=int, default=500, help="회상 층 측정 주기 (step, 0 = 평가 때만)")
    ap.add_argument("--probe-items", type=int, default=256)
    ap.add_argument("--probe-K", type=int, default=4, help="연상 회상 문항의 줄 수")
    ap.add_argument("--probe-method", choices=["auto", "assoc", "induction"], default="auto")
    ap.add_argument("--probe-patience", type=int, default=3, help="같은 층이 이만큼 연속이면 '확정'")
    ap.add_argument("--probe-min-drop", type=float, default=0.5, help="확정에 필요한 최소 Δlogp (nats)")
    ap.add_argument("--gate-token", default="<|bot|>", help="응답 표지 — 추론·SFT 에서 이 위치의 회상 층을 막는다")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    P = lambda n: os.path.join(a.out_dir, n)                      # noqa: E731
    log_f = open(P("train.log"), "a", encoding="utf-8", buffering=1)

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        log_f.write(msg + "\n")

    log(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작: {' '.join(sys.argv)}")
    if a.stop_file and os.path.exists(a.stop_file):
        os.remove(a.stop_file)
    stop = {"on": False, "why": ""}

    def _sig(_s, _f):
        if stop["on"]:
            raise KeyboardInterrupt
        stop.update(on=True, why="중지 신호")
        print("\n[중지 요청] 현재 step 을 마치고 저장합니다 (한 번 더 = 즉시 종료)", flush=True)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    t_start = time.time()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    tok = tokenizer_info(sp, a.tokenizer)
    gate_id = sp.piece_to_id(a.gate_token)
    if gate_id == sp.unk_id():
        log(f"⚠ 토크나이저에 응답 표지 {a.gate_token} 가 없다 — 회상 층은 기록하지만 게이트 토큰은 비워 둔다")
        gate_id = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if a.vram_gb > 0:
            tot = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / tot))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    use_amp = a.amp == "bf16" and device.type == "cuda"
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext

    seq_len_cfg = build_config(a, sp.get_piece_size()).seq_len
    tr_ds, va_ds, meta = load_pretrain(a.data, seq_len_cfg, a.max_seqs)
    vocab = int(meta["vocab_size"])
    assert vocab == sp.get_piece_size(), f"코퍼스 vocab {vocab} ≠ 토크나이저 {sp.get_piece_size()}"
    cfg = build_config(a, vocab)
    log(f"데이터 train {len(tr_ds):,} / val {len(va_ds):,} 시퀀스 × {cfg.seq_len}")

    net = BitGCLM(cfg).to(device)
    net.grad_ckpt = a.grad_ckpt
    log(f"모델 d{cfg.d_model} L{cfg.n_layers} H{cfg.n_heads} ff{cfg.d_ff} | attention 층 {list(cfg.attn_layers)} | "
        f"{n_params(net)/1e6:.2f}M | {device}")
    if len(cfg.attn_layers) < 2:
        log("⚠ attention 층이 1개 — 회상 층은 그 층이 될 수밖에 없다 (측정은 그대로 한다)")
    if a.compile:
        compile_layers(net)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay,
                            fused=device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    probe = RecallProbe(sp, n_items=a.probe_items, K=a.probe_K, method=a.probe_method, seed=a.seed)
    tracker = RecallTracker(patience=a.probe_patience, min_drop=a.probe_min_drop,
                            gate_token=a.gate_token, gate_token_id=gate_id)
    log(f"회상 측정: {probe.method} · 문항 {len(probe.items)} · 주기 {a.probe_every or '평가 때만'} step")

    n_steps = len(tr_ds) // a.batch_size
    total = a.epochs * n_steps

    def loader(ep, skip):
        g = torch.Generator().manual_seed(a.seed * 100003 + ep)
        perm = torch.randperm(len(tr_ds), generator=g).tolist()[:n_steps * a.batch_size]
        return DataLoader(Subset(tr_ds, perm[skip * a.batch_size:]), batch_size=a.batch_size,
                          shuffle=False, num_workers=(0 if os.name == "nt" else 2),
                          pin_memory=device.type == "cuda", drop_last=True)

    va = DataLoader(va_ds, batch_size=a.batch_size)

    def set_lr(step):
        if a.warmup and step < a.warmup:
            lr = a.lr * (step + 1) / a.warmup
        else:
            prog = min(1.0, max(0.0, (step - a.warmup) / max(1, total - a.warmup)))
            lr = a.lr * (a.final_lr_frac + (1 - a.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate():
        net.eval()
        s = n = 0
        for x, y in va:
            x, y = x.to(device), y.to(device)
            with ac():
                lg = net(x)
            s += float(ce(lg.float().reshape(-1, vocab), y.reshape(-1))) * y.numel()
            n += y.numel()
        net.train()
        return s / n

    def run_probe(step, why=""):
        res = probe.run(net, device, ac)
        tracker.update(step, res)
        with open(P("recall.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "why": why, **res}, ensure_ascii=False) + "\n")
        log(f"  step {step} {tracker.summary(res)}")

    best, best_state, best_recall = float("inf"), None, None
    ep0, i0 = 1, 0
    if os.path.exists(P("resume.pt")):
        rk = torch.load(P("resume.pt"), map_location="cpu", weights_only=False)
        assert rk["config"] == cfg.to_dict(), "resume.pt 의 모델 설정이 지금 인자와 다르다"
        net.load_state_dict(rk["model"])
        opt.load_state_dict(rk["opt"])
        ep0, i0, best = rk["epoch"], rk["step"], rk["best"]
        best_state, best_recall = rk.get("best_state"), rk.get("best_recall")
        tracker.history = list(rk.get("recall_history") or [])
        torch.set_rng_state(rk["rng_cpu"])
        if device.type == "cuda" and rk.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(rk["rng_cuda"])
        del rk
        log(f"재개: ep{ep0} step {i0}/{n_steps} | best val ppl {math.exp(best):.2f} | 회상 기록 {len(tracker.history)}개")

    def ckpt_obj(state, recall, val):
        return {"format": FORMAT, "config": cfg.to_dict(), "state_dict": state, "recall": recall,
                "tokenizer": tok, "meta": {"stage": "pretrain", "val_loss": val,
                                           "val_ppl": math.exp(val) if math.isfinite(val) else None,
                                           "data": os.path.basename(a.data), "args": vars(a)}}

    def save_resume(ep, step):
        atomic_save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step,
                     "best": best, "best_state": best_state, "best_recall": best_recall,
                     "recall_history": tracker.record()["history"] if tracker.history else [],
                     "config": cfg.to_dict(), "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, P("resume.pt"))

    def on_eval(ep, step):
        nonlocal best, best_state, best_recall
        v = evaluate()
        run_probe((ep - 1) * n_steps + step, "eval")   # 평가한 바로 그 가중치에서 잰다 (전역 step)
        mark = ""
        if v < best:
            best, best_state, best_recall = v, cpu_state(net), tracker.record()
            atomic_save(ckpt_obj(best_state, best_recall, best), P("best.pt"))
            mark = "  ★ 최고 (best.pt)"
        log(f"  ep{ep} step {step} val {v:.4f} (ppl {math.exp(v):.2f}){mark}")
        save_resume(ep, step)

    if ep0 == 1 and i0 == 0 and not tracker.history:
        run_probe(0, "init")
    net.train()
    bad = 0
    try:
        for ep in range(ep0, a.epochs + 1):
            skip = i0 if ep == ep0 else 0
            t0, s, n = time.time(), 0.0, 0
            for j, (x, y) in enumerate(loader(ep, skip)):
                i = skip + j
                gstep = (ep - 1) * n_steps + i
                lr = set_lr(gstep)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with ac():
                    lg = net(x)
                loss = ce(lg.float().reshape(-1, vocab), y.reshape(-1))
                loss.backward()
                gn = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                lv, gv = float(loss.detach()), float(gn)
                if math.isfinite(lv) and math.isfinite(gv):
                    opt.step()
                    bad = 0
                else:                            # nan/inf 갱신은 건너뛴다 (50 연속이면 멈춤)
                    bad += 1
                    log(f"  ⚠ loss/기울기 비정상 step {i+1} (loss {lv}, grad {gv}) — 건너뜀 ({bad})")
                    if bad >= 50:
                        stop.update(on=True, why="loss/기울기 비정상 50 연속")
                opt.zero_grad(set_to_none=True)
                if math.isfinite(lv):
                    s, n = s + lv, n + 1
                done = i + 1
                if a.log_every and done % a.log_every == 0:
                    el = time.time() - t0
                    eta = el / max(1, done - skip) * (total - gstep - 1) / 3600
                    log(f"  ep{ep} step {done}/{n_steps} loss {s/max(1,n):.4f} (ppl {math.exp(s/max(1,n)):.1f}) "
                        f"lr {lr:.2e} gn {gv:.2f} | {el/60:.0f}분 · 남은 {eta:.1f}h")
                    s = n = 0
                ev = a.eval_every and done % a.eval_every == 0
                if ev:
                    on_eval(ep, done)
                elif a.probe_every and done % a.probe_every == 0:
                    run_probe(gstep + 1)
                if a.ckpt_every and done % a.ckpt_every == 0 and not ev:
                    save_resume(ep, done)
                if a.stop_file and done % 20 == 0 and os.path.exists(a.stop_file):
                    stop.update(on=True, why="정지 파일")
                if a.max_hours and (time.time() - t_start) / 3600 >= a.max_hours:
                    stop.update(on=True, why=f"시간 한도 {a.max_hours}h")
                if a.stop_at_step and gstep + 1 >= a.stop_at_step:
                    stop.update(on=True, why=f"--stop-at-step {a.stop_at_step}")
                if stop["on"]:
                    save_resume(ep, done)
                    if a.probe_every and done % a.probe_every:
                        run_probe(gstep + 1, "stop")
                    rec = tracker.record()
                    atomic_save(ckpt_obj(cpu_state(net), rec, float("nan")), P("last.pt"))
                    log(f"[안전 종료] {stop['why']} — ep{ep} step {done} 저장 (resume.pt · last.pt, 회상 층 "
                        f"b{rec['layer'] if rec else '?'}). 같은 명령으로 이어진다")
                    return
            i0 = 0
            on_eval(ep, n_steps)
            save_resume(ep + 1, 0)
    except KeyboardInterrupt:
        log("[즉시 종료] 저장하지 않았다 — 마지막 resume.pt 에서 이어진다")
        return

    # 끝: 마지막 가중치도 남긴다 (회상 기록은 마지막 측정 = 이 가중치)
    rec = tracker.record()
    atomic_save(ckpt_obj(cpu_state(net), rec, evaluate()), P("last.pt"))
    log(f"== 완료: best val ppl {math.exp(best):.2f} | 회상 층 b{rec['layer']} "
        f"({'확정' if rec['decided'] else '미확정'}, step {rec['decided_step']}~) → 게이트 {rec['gate_layers']} @ "
        f"{a.gate_token} | best.pt · last.pt")
    if best_recall and best_recall["layer"] != rec["layer"]:
        log(f"⚠ best.pt 의 회상 층 b{best_recall['layer']} 와 마지막 측정 b{rec['layer']} 가 다르다 — "
            f"scripts/probe_recall.py 로 다시 재서 기록할 것")


if __name__ == "__main__":
    main()
