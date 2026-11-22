"""SFT (대화·지시 미세조정) — 체크포인트에 남은 회상 층을 응답 첫 토큰 위치에서 막은 채 학습한다.

  python sft.py --ckpt runs/1b/best.pt --data data/sft_chat.pt --out runs/1b/chat.pt \
      --replay data/replay.bin --replay-per-batch 4 --epochs 1 --epoch-frac 0.5 --lr 3e-5 --compile

왜 학습 때부터 막는가 (원 연구 §26.16~26.18, 1B 최종 모델):
  - 추론 때만 회상 층을 막으면 되풀이가 사라지는 대신 '빈 답' 이 된다 — 그 층이 답 첫 토큰의 주 경로라
    다른 층에 대신할 경로가 없다 (자유 대화 답의 81% 가 '.' 한 글자).
  - 막은 채 SFT 하면 다른 층이 답 시작을 넘겨받는다: 지시 126 문항 30 → 64 (E1c 기준 52), 시드 3개 평균 63.7.
  - 막고 배운 모델도 게이트를 떼면 다시 무너진다 (22) → 게이트는 배포 구조의 일부다. 그래서 회상 층 기록을
    산출 체크포인트에 그대로 넘겨, 다음 단계 SFT·추론이 같은 게이트를 쓰게 한다.
  - 사전학습부터 막는 것은 소용없었다 — 4000 step 안에 다른 층이 같은 베끼기를 다시 만들었다 (§26.21).

평가 때 '첫 답 베끼기' 도 잰다: 응답 첫 위치의 argmax 가 직전 사용자 턴의 첫 토큰인 비율 (그 토큰이 정답이 아닌 자리만 = 틀린 베끼기).
여러 단계(대화 → 글 읽기 → 지시)는 이 스크립트를 이어서 실행한다 (--ckpt 에 앞 단계 산출).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from bitgc.checkpoint import atomic_save, cpu_state, load_checkpoint, tokenizer_info, tokenizer_path
from bitgc.data import IGNORE, load_sft
from bitgc.model import compile_layers

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@torch.no_grad()
def echo_rate(net, vx, vy, B, device, ac, user_id, bot_id, max_rows=512):
    """응답 첫 위치(<|bot|>)에서 argmax 가 직전 사용자 턴 첫 토큰인 비율 → (베끼기 %, 틀린 베끼기 %, 첫 토큰 정답 %)."""
    was = net.training
    net.eval()
    cp = wr = ok = n = 0
    for i in range(0, min(vx.shape[0], max_rows) - B + 1, B):
        x, y = vx[i:i + B].to(device).long(), vy[i:i + B].to(device).long()
        with ac():
            lg = net(x)
        pred = lg.argmax(-1)
        for r in range(x.shape[0]):
            row = x[r].tolist()
            last_user = None
            for t, tok in enumerate(row):
                if tok == user_id and t + 1 < len(row):
                    last_user = row[t + 1]
                elif tok == bot_id and last_user is not None and int(y[r, t]) != IGNORE:
                    p, g = int(pred[r, t]), int(y[r, t])
                    n += 1
                    cp += p == last_user
                    wr += (p == last_user) and (g != last_user)
                    ok += p == g
    net.train(was)
    return (100 * cp / max(1, n), 100 * wr / max(1, n), 100 * ok / max(1, n), n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="bitgc-v1 체크포인트 (사전학습 또는 앞 단계 SFT)")
    ap.add_argument("--data", required=True, help="scripts/prepare_sft.py 산출 .pt")
    ap.add_argument("--tokenizer", default="", help="비우면 체크포인트 폴더의 tokenizer.model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="", help="재개 파일 경로 (비우면 재개 파일 없이)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--epoch-frac", type=float, default=1.0, help="epoch 당 이 비율만 본다")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--vram-gb", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=6000)
    ap.add_argument("--ckpt-every", type=int, default=3000)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--stop-file", default="")
    # 망각 대응: 배치 일부를 사전학습 코퍼스(.bin, prepare_pretrain.py 산출)에서 뽑는다
    ap.add_argument("--replay", default="")
    ap.add_argument("--replay-per-batch", type=int, default=0)
    ap.add_argument("--replay-weight", type=float, default=0.5)
    # 게이트
    ap.add_argument("--no-gate", action="store_true", help="회상 층 게이트 없이 (비교용)")
    ap.add_argument("--gate-layers", default="", help="기록 대신 이 층을 막는다 (쉼표 구분)")
    a = ap.parse_args()

    if a.stop_file and os.path.exists(a.stop_file):
        os.remove(a.stop_file)
    stop = {"on": False, "why": ""}

    def _sig(_s, _f):
        if stop["on"]:
            raise KeyboardInterrupt
        stop.update(on=True, why="중지 신호")
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if a.vram_gb > 0:
            tot = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / tot))
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else contextlib.nullcontext

    import sentencepiece as spm
    a.tokenizer = a.tokenizer or tokenizer_path(a.ckpt)
    assert a.tokenizer, "--tokenizer 가 필요하다 (체크포인트 폴더에 tokenizer.model 없음)"
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    tok = tokenizer_info(sp, a.tokenizer)
    p = load_sft(a.data)
    tx, ty, vx, vy = p["train_x"], p["train_y"], p["val_x"], p["val_y"]
    net, ck = load_checkpoint(a.ckpt, device, a.dropout, apply_gate=not a.no_gate)
    vocab = net.cfg.vocab_size
    assert p["vocab_size"] == vocab, "SFT 데이터와 모델의 vocab 이 다르다"
    recall = dict(ck.get("recall") or {})
    if a.gate_layers and not a.no_gate:
        gl = [int(v) for v in a.gate_layers.split(",")]
        net.set_recall_gate(gl, tok["bot_id"])
        recall.update(gate_layers=gl, token_id=tok["bot_id"], gate_override=True)
    if net.gate_layers:
        print(f"게이트: 블록 {list(net.gate_layers)} 혼합 출력 × 0 @ 모든 <|bot|> (id {net.gate_token}) — "
              f"회상 측정 {recall.get('method')} · 확정 {recall.get('decided')} (step {recall.get('decided_step')}~)",
              flush=True)
    else:
        print("게이트 없음" + (" (--no-gate)" if a.no_gate else " — 체크포인트에 회상 층 기록이 없다"), flush=True)

    net.grad_ckpt = not a.no_grad_ckpt
    if a.compile:
        compile_layers(net)
    params = list(net.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=device.type == "cuda")
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    B = a.batch_size
    RB = a.replay_per_batch if a.replay else 0
    SB = B - RB
    assert SB > 0
    n_tr = int(tx.shape[0] // SB * a.epoch_frac)
    total = a.epochs * n_tr
    rep = None
    T = tx.shape[1]
    if RB:                                   # 사전학습 .bin 을 그대로 — 행 r = 토큰 [r·T, r·T+T+1)
        rep = np.memmap(a.replay, dtype=np.uint16, mode="r")
        n_rep = (rep.size - 1) // T
    print(f"SFT train {tx.shape[0]:,} / val {vx.shape[0]:,} 시퀀스 | step {n_tr}/epoch × {a.epochs}"
          + (f" | replay {n_rep:,} 행, 배치 {SB}+{RB}" if RB else ""), flush=True)

    def set_lr(step):
        lr = a.lr * (step + 1) / a.warmup if step < a.warmup else \
            a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - a.warmup) / max(1, total - a.warmup))))
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate():
        net.eval()
        s = n = 0
        for i in range(0, vx.shape[0] - B + 1, B):
            x, y = vx[i:i + B].to(device).long(), vy[i:i + B].to(device).long()
            with ac():
                lg = net(x)
            s += float(nn.functional.cross_entropy(lg.float().reshape(-1, vocab), y.reshape(-1),
                                                   ignore_index=IGNORE, reduction="sum"))
            n += int((y != IGNORE).sum())
        net.train()
        return s / max(1, n)

    def echo_line():
        on = echo_rate(net, vx, vy, B, device, ac, tok["user_id"], tok["bot_id"])
        line = f"첫 답 베끼기 {on[0]:.1f}% (틀린 {on[1]:.1f}%) · 첫 토큰 정답 {on[2]:.1f}% [{on[3]}자리]"
        if net.gate_layers:
            net.gate_on = False
            off = echo_rate(net, vx, vy, B, device, ac, tok["user_id"], tok["bot_id"])
            net.gate_on = True
            line += f" | 게이트 뗌: 베끼기 {off[0]:.1f}% (틀린 {off[1]:.1f}%) · 정답 {off[2]:.1f}%"
        return line

    ep0, i0, best, base = 1, 0, float("inf"), None
    if a.resume and os.path.exists(a.resume):
        rk = torch.load(a.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(rk["model"])
        opt.load_state_dict(rk["opt"])
        ep0, i0, best, base = rk["epoch"], rk["step"], rk["best"], rk["base"]
        torch.set_rng_state(rk["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rk["rng_cuda"])
        del rk
        print(f"재개: ep{ep0} step {i0}/{n_tr}", flush=True)
    else:
        base = evaluate()
        print(f"SFT 전 val ppl {math.exp(base):.2f} | {echo_line()}", flush=True)

    def save_resume(ep, step):
        if a.resume:
            atomic_save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step,
                         "best": best, "base": base, "rng_cpu": torch.get_rng_state(),
                         "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, a.resume)

    def save_best(v):
        meta = dict(ck.get("meta") or {})
        stages = list(meta.get("sft_stages") or []) + [{"data": os.path.basename(a.data), "val_loss": v,
                                                         "gate": list(net.gate_layers), "lr": a.lr}]
        out = {"format": ck["format"], "config": net.cfg.to_dict(), "state_dict": cpu_state(net),
               "recall": (recall if not a.no_gate else dict(recall, gate_layers=[], gate_disabled=True)),
               "tokenizer": tok, "meta": dict(meta, stage="sft", sft_stages=stages, base_ckpt=os.path.basename(a.ckpt))}
        atomic_save(out, a.out)

    def check_eval(ep, step):
        nonlocal best
        v = evaluate()
        mark = ""
        if v < best:
            best, mark = v, "  ★ 최고 (저장)"
            save_best(v)
        print(f"  ep{ep} step {step} val {v:.4f} (ppl {math.exp(v):.2f}){mark} | {echo_line()}", flush=True)

    t_start, bad = time.time(), 0
    net.train()
    for ep in range(ep0, a.epochs + 1):
        g = torch.Generator().manual_seed(a.seed * 1000 + ep)
        perm = torch.randperm(tx.shape[0], generator=g)
        start = i0 if ep == ep0 else 0
        t0, s, n = time.time(), 0.0, 0
        for i in range(start, n_tr):
            gstep = (ep - 1) * n_tr + i
            lr = set_lr(gstep)
            idx = perm[i * SB:(i + 1) * SB]
            x, y = tx[idx].to(device).long(), ty[idx].to(device).long()
            if RB:
                ri = np.random.default_rng(a.seed * 1_000_003 + gstep).integers(0, n_rep, RB)
                rr = torch.from_numpy(np.stack([np.asarray(rep[r * T:r * T + T + 1], dtype=np.int64)
                                                for r in ri])).to(device)
                x = torch.cat([x, rr[:, :-1]])
            with ac():
                lg = net(x)
            loss = sft_loss = ce(lg[:SB].float().reshape(-1, vocab), y.reshape(-1))
            if RB:
                rl = nn.functional.cross_entropy(lg[SB:].float().reshape(-1, vocab), rr[:, 1:].reshape(-1))
                loss = (1 - a.replay_weight) * sft_loss + a.replay_weight * rl
            loss.backward()
            gn = nn.utils.clip_grad_norm_(params, 1.0)
            lv, gv = float(loss.detach()), float(gn)
            if math.isfinite(lv) and math.isfinite(gv):
                opt.step()
                bad = 0
            else:
                bad += 1
                if bad >= 50:
                    stop.update(on=True, why="loss/기울기 비정상 50 연속")
            opt.zero_grad(set_to_none=True)
            sv = float(sft_loss.detach())
            if math.isfinite(sv):
                s, n = s + sv, n + 1
            done = i + 1
            if a.log_every and done % a.log_every == 0:
                el = time.time() - t0
                eta = el / max(1, done - start) * (total - gstep - 1) / 3600
                print(f"  ep{ep} step {done}/{n_tr} SFT loss {s/max(1,n):.4f} lr {lr:.2e} gn {gv:.2f} | "
                      f"{el/60:.0f}분 · 남은 {eta:.1f}h", flush=True)
                s = n = 0
            if a.eval_every and done % a.eval_every == 0 and done < n_tr:
                check_eval(ep, done)
                save_resume(ep, done)
            elif a.ckpt_every and done % a.ckpt_every == 0:
                save_resume(ep, done)
            if a.stop_file and done % 20 == 0 and os.path.exists(a.stop_file):
                stop.update(on=True, why="정지 파일")
            if a.max_hours and (time.time() - t_start) / 3600 >= a.max_hours:
                stop.update(on=True, why=f"시간 한도 {a.max_hours}h")
            if stop["on"]:
                save_resume(ep, done)
                print(f"[안전 종료] {stop['why']} — ep{ep} step {done}"
                      + (" 저장. 같은 명령으로 이어진다" if a.resume else " (재개 파일 없음)"), flush=True)
                return
        check_eval(ep, n_tr)
        save_resume(ep + 1, 0)
    print(f"== SFT 완료: val ppl {math.exp(base):.2f} → {math.exp(best):.2f} — {a.out} "
          f"(게이트 {list(net.gate_layers) or '없음'} 기록 유지)", flush=True)


if __name__ == "__main__":
    main()
