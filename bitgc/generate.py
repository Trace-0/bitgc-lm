"""생성 — 증분 디코딩(KV 캐시 + conv 상태)과 대화 형식.

대화 형식 (SFT 데이터와 같다):  [eos] <|user|>질문<|end|><|bot|>답<|end|><|user|>…<|end|><|bot|>
마지막 <|bot|> 위치의 출력이 답의 첫 토큰이다. 게이트가 걸린 모델은 그 위치에서 회상 층을 막는다.

디코딩 기본값 = 원 연구의 최종 평가 설정 (지시 126 문항 · 멀티턴 평가, `tests/instr_eval.py` 와 같음):
  greedy · 반복 벌점 1.2 · 같은 6-gram 금지 · 첫 답 토큰으로 맨 '▁' 금지 · 답 안에서 <|bot|>·<|user|> 금지.
  ('맨 ▁ 금지' 는 게이트 모델의 첫 답 분포가 평평할 때 greedy 가 공백 하나를 골라 빈 답이 되는 것을 막는다 — 원 연구 §26.18)
"""
from __future__ import annotations

import numpy as np
import torch


def banned_ngram(seq, n):
    if n <= 0 or len(seq) < n - 1:
        return ()
    tail = tuple(seq[-(n - 1):])
    return {seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i:i + n - 1]) == tail}


class Generator:
    def __init__(self, net, sp, device="cpu", dtype=None):
        self.net, self.sp, self.device = net.eval(), sp, device
        self.dtype = dtype                       # 예: torch.bfloat16 (GPU). None 이면 fp32
        self.T = net.cfg.seq_len
        pid = sp.piece_to_id
        self.user, self.bot, self.end = pid("<|user|>"), pid("<|bot|>"), pid("<|end|>")
        self.bare = pid("▁")

    def _ac(self):
        import contextlib
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device if isinstance(self.device, str) else self.device.type, dtype=self.dtype)

    @torch.no_grad()
    def logits(self, ids) -> torch.Tensor:
        """전체 문맥을 한 번에 (검증용). 마지막 위치 로짓."""
        x = torch.tensor([ids[-self.T:]], device=self.device)
        with self._ac():
            return self.net(x)[0, -1].float()

    def chat_ids(self, turns, fmt: str = "eval") -> list[int]:
        """turns = [(user, bot or None), …] — 마지막 bot 이 None 이면 그 답을 생성한다.

        fmt="eval": 원 연구의 평가와 같은 토큰열 — 턴 문자열을 통째로 인코딩한다
                    ([eos] ▁ <|user|> 질문 … <|end|> ▁ <|bot|>). 보고한 수치는 전부 이 형식이다.
        fmt="sft" : SFT 데이터와 같은 토큰열 — 표지는 id 로, 본문은 따로 인코딩 ([eos] <|user|> ▁질문 … <|end|> <|bot|>).
        """
        sp, ids = self.sp, [self.sp.eos_id()]
        for u, b in turns:
            if fmt == "sft":
                ids += [self.user] + sp.encode(u) + [self.end]
                if b is not None:
                    ids += [self.bot] + sp.encode(b) + [self.end]
            else:
                ids += sp.encode("<|user|>" + u + "<|end|>")
                if b is not None:
                    ids += sp.encode("<|bot|>" + b + "<|end|>")
        return ids + ([self.bot] if fmt == "sft" else sp.encode("<|bot|>"))

    @torch.no_grad()
    def generate(self, ids, max_new=128, temperature=0.0, top_k=0, rep=1.2, no_repeat=6,
                 chat_rules=True, stop_ids=None, seed=0, stream=None) -> list[int]:
        stop = set(stop_ids or ()) | {self.sp.eos_id(), self.end}
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = list(ids)
        out: list[int] = []
        lg = self._prefill(ids)
        for _ in range(max_new):
            v = lg.clone()
            if rep != 1.0 and out:
                u = torch.tensor(sorted(set(out)), device=v.device)
                v[u] = torch.where(v[u] > 0, v[u] / rep, v[u] * rep)
            for t in banned_ngram(out, no_repeat):
                v[t] = float("-inf")
            if chat_rules:
                v[self.bot] = v[self.user] = float("-inf")
                if not out:
                    v[self.bare] = float("-inf")
            if temperature <= 0:
                nx = int(torch.argmax(v))
            else:
                v = v / temperature
                if top_k:
                    kv, ki = torch.topk(v, top_k)
                    nx = int(ki[torch.multinomial(torch.softmax(kv, -1).cpu(), 1, generator=g)])
                else:
                    nx = int(torch.multinomial(torch.softmax(v, -1).cpu(), 1, generator=g))
            if nx in stop:
                break
            out.append(nx)
            if stream is not None:
                stream(nx)
            ids.append(nx)
            if self.pos >= self.T:               # 위치 임베딩 끝 — 뒤쪽 문맥으로 다시 채운다
                lg = self._prefill(ids)
                continue
            lg = self._step(nx)
        return out

    # ── 백엔드 (torch). EngineGenerator 는 이 두 개만 바꾼다 ──
    @torch.no_grad()
    def _prefill(self, seq) -> torch.Tensor:
        seq = seq[-(self.T - 1):]                # 생성할 자리를 하나 남긴다
        self.caches = self.net.new_caches()
        x = torch.tensor([seq], device=self.device)
        with self._ac():
            lg = self.net(x, caches=self.caches, pos0=0)[0, -1].float()
        self.pos = len(seq)
        return lg

    @torch.no_grad()
    def _step(self, tok) -> torch.Tensor:
        x = torch.tensor([[tok]], device=self.device)
        with self._ac():
            lg = self.net(x, caches=self.caches, pos0=self.pos)[0, -1].float()
        self.pos += 1
        return lg

    def reply(self, turns, fmt: str = "eval", **kw) -> str:
        return self.sp.decode(self.generate(self.chat_ids(turns, fmt), **kw))


class EngineGenerator(Generator):
    """같은 디코딩 규칙을 정수 엔진(bitgc.engine.runtime.GenLM)으로. net 은 설정(seq_len)만 쓴다 — 가중치는 엔진 자산에."""

    def __init__(self, eng, sp):
        self.eng, self.sp, self.device, self.dtype = eng, sp, "cpu", None
        self.T = eng.seq_len
        pid = sp.piece_to_id
        self.user, self.bot, self.end = pid("<|user|>"), pid("<|bot|>"), pid("<|end|>")
        self.bare = pid("▁")
        self.net = None

    def logits(self, ids):
        self.eng.reset()
        return torch.from_numpy(self.eng.forward(ids[-self.T:], last_only=True).copy())

    def _prefill(self, seq):
        seq = seq[-(self.T - 1):]
        self.eng.reset()
        lg = self.eng.forward(seq, last_only=True)
        self.pos = len(seq)
        return torch.from_numpy(np.asarray(lg, np.float32).copy())

    def _step(self, tok):
        lg = self.eng.step(int(tok))
        self.pos += 1
        return torch.from_numpy(lg)
