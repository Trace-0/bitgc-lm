"""회상 층 측정 — 학습 중간에 자동으로 돌려 '어느 attention 층이 문맥에서 값을 찾아 옮기는가' 를 찾는다.

왜 필요한가 (원 연구 logiformer `bitnet_comparison.md` §24.4·§26.1·§26.11):
  - 팽창 없는 gconv 하이브리드에서 회상을 맡는 attention 층은 학습 초반(전체의 2~3%)의 '경주' 로 정해진다.
    먼저 값을 가리키기 시작한 층이 이기고, 진 층은 잔차 스칼라가 줄어 돌아오지 않는다 (14/14).
    → 층 수·시드·데이터·학습률 일정이 바뀌면 회상 층 위치가 바뀐다 (30M 시드 8개 중 블록 3 이 6, 블록 7 이 2).
  - 1B 최종 모델은 블록 11 하나가 회상을 맡았고, 그 층이 SFT 뒤 '질문 첫머리 되풀이' 를 만들었다.
    응답 첫 토큰 위치(<|bot|>)에서 그 층을 막고 SFT 한 것이 최종 처방이다 (§26.18).

측정 (기본 `assoc`, 원 연구 `tests/assoc_circuit.py` 와 같은 문항):
  문맥   "\\n사과: 23\\n바다: 71\\n구름: 45\\n연필: 12\\n바다:"   → 정답 다음 토큰 '▁71'
  기준   정답 토큰의 log p 와 argmax 정답률
  끄기   attention 층 b 의 혼합층 출력을 **질의 위치에서만** 0 (잔차 더하기 직전, 깨끗한 제거)
  점수   drop_b = 기준 log p − 끄기 log p   (nats, 클수록 그 층이 답을 실어 나른다)
  회상 층 = drop 이 가장 큰 attention 층.

  `induction` (대안, 데이터·토크나이저 무관): 무작위 토큰 열 r 을 두 번 이어 붙이고 두 번째 r 의 다음 토큰을
  맞히게 한다. 두 번째 r 위치 전부에서 층 b 를 끈다.
  `auto`: 토크나이저가 assoc 문항을 만들 수 있으면 assoc, 아니면 induction.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

KO_WORDS = ["사과", "바다", "구름", "연필", "기차", "고양이", "나무", "의자", "시계", "우산", "거울", "편지", "모자", "피아노",
            "호수", "다리", "창문", "열쇠", "가방", "등대", "사막", "섬", "숲", "강", "별", "달", "꽃", "새", "배", "산",
            "공원", "도서관", "병원", "학교", "시장", "공항", "항구", "농장", "성당", "극장", "박물관", "운동장", "정원",
            "부엌", "다락", "지붕", "계단", "엘리베이터", "자전거", "비행기", "망원경", "주전자", "냉장고", "세탁기",
            "책상", "침대", "담요", "베개", "양초", "촛대", "나침반", "지도", "깃발", "북", "종", "피리", "바이올린", "첼로"]
EN_WORDS = ["apple", "ocean", "cloud", "pencil", "train", "cat", "tree", "chair", "clock", "umbrella", "mirror", "letter",
            "hat", "piano", "lake", "bridge", "window", "key", "bag", "lighthouse", "desert", "island", "forest", "river",
            "star", "moon", "flower", "bird", "ship", "mountain", "park", "library", "hospital", "school", "market",
            "airport", "harbor", "farm", "church", "theater", "museum", "garden", "kitchen", "roof", "stairs", "bicycle",
            "airplane", "telescope", "kettle", "fridge", "desk", "bed", "blanket", "pillow", "candle", "compass", "map",
            "flag", "drum", "bell", "flute", "violin", "cello"]


class RecallProbe:
    def __init__(self, sp, n_items: int = 256, K: int = 4, method: str = "auto", seed: int = 0,
                 words=None, ind_len: int = 24, batch_size: int = 64):
        self.sp, self.n, self.K, self.bs = sp, n_items, K, batch_size
        self.seed, self.ind_len = seed, ind_len
        self.words = list(words) if words else None
        if method == "auto":
            method = "assoc" if self._assoc_ok() else "induction"
        assert method in ("assoc", "induction"), method
        self.method = method
        self.items = self._assoc_items() if method == "assoc" else self._induction_items()

    # ── 문항 ──
    def _vals(self):
        sp, out = self.sp, {}
        for v in range(10, 100):
            t = sp.encode(f"가방: {v}") if self._lang() == "ko" else sp.encode(f"bag: {v}")
            if sp.id_to_piece(t[-1]).lstrip("▁") == str(v):
                out[v] = t[-1]
        return out

    def _lang(self):
        if self.words:
            return "custom"
        ko = self.sp.encode("사과")
        return "ko" if all(i != self.sp.unk_id() for i in ko) and len(ko) <= 2 else "en"

    def _word_list(self):
        if self.words:
            return self.words
        return KO_WORDS if self._lang() == "ko" else EN_WORDS

    def _assoc_ok(self) -> bool:
        try:
            return len(self._vals()) >= 60 and len(self._word_list()) >= 2 * self.K
        except Exception:        # noqa: BLE001
            return False

    def _assoc_items(self):
        sp, rng = self.sp, random.Random(500 + self.K + self.seed)
        vt = self._vals()
        vals_ok = sorted(vt)
        words = self._word_list()
        items = []
        while len(items) < self.n:
            ws = rng.sample(words, self.K)
            vs = rng.sample(vals_ok, self.K)
            qi = rng.randrange(self.K)
            lines = [f"{w}: {v}" for w, v in zip(ws, vs)]
            ids = [sp.eos_id()] + sp.encode("\n" + "\n".join(lines) + f"\n{ws[qi]}:")
            # 값 토큰이 문맥 안에 그대로 있는지 (토크나이저가 숫자를 앞 글자와 합치면 버린다)
            if vt[vs[qi]] not in ids:
                continue
            items.append((ids, [len(ids) - 1], [vt[vs[qi]]]))
        return items

    def _induction_items(self):
        sp, rng = self.sp, random.Random(900 + self.seed)
        pool = [i for i in range(sp.get_piece_size())
                if not (sp.is_control(i) or sp.is_unknown(i) or sp.is_byte(i) or sp.is_unused(i))]
        pool = pool[len(pool) // 10: len(pool) * 9 // 10]          # 아주 흔하거나 아주 드문 조각은 뺀다
        L, items = self.ind_len, []
        for _ in range(self.n):
            r = rng.sample(pool, L)
            ids = [sp.eos_id()] + r + r
            qpos = list(range(L + 1, 2 * L))                        # 두 번째 r 의 위치 (마지막 제외)
            tgt = [ids[p + 1] for p in qpos]
            items.append((ids, qpos, tgt))
        return items

    # ── 측정 ──
    def _batches(self, device):
        for b in range(0, len(self.items), self.bs):
            ch = self.items[b:b + self.bs]
            L = max(len(it[0]) for it in ch)
            x = torch.zeros(len(ch), L, dtype=torch.long)
            qm = torch.zeros(len(ch), L, dtype=torch.bool)
            tg = torch.full((len(ch), L), -1, dtype=torch.long)
            for r, (ids, qpos, tgt) in enumerate(ch):
                x[r, :len(ids)] = torch.tensor(ids)
                for p, t in zip(qpos, tgt):
                    qm[r, p] = True
                    tg[r, p] = t
            yield x.to(device), qm.to(device), tg.to(device)

    @torch.no_grad()
    def _score(self, net, device, autocast, ko_layer=None):
        lp_sum = acc = n = 0.0
        for x, qm, tg in self._batches(device):
            ko = None if ko_layer is None else {ko_layer: qm}
            with autocast():
                lg = net(x, knockout=ko)
            lg = lg.float()[qm]                                     # (Nq, V)
            t = tg[qm]
            lp_sum += float(F.log_softmax(lg, -1).gather(1, t[:, None]).sum())
            acc += float((lg.argmax(-1) == t).sum())
            n += t.numel()
        return lp_sum / n, 100.0 * acc / n

    def run(self, net, device=None, autocast=None) -> dict:
        """→ {"method", "base_lp", "base_acc", "layers": {b: {"drop", "ko_acc"}}, "leader", "split", "scales"}"""
        import contextlib
        device = device or next(net.parameters()).device
        autocast = autocast or contextlib.nullcontext
        was_training, gate_on = net.training, net.gate_on
        net.eval()
        net.gate_on = False                     # 측정은 게이트 없는 망 자체로 (질의에 <|bot|> 없음 — 안전장치)
        try:
            base_lp, base_acc = self._score(net, device, autocast)
            layers = {}
            for b in net.attn_ids:
                lp, a = self._score(net, device, autocast, ko_layer=b)
                layers[b] = {"drop": base_lp - lp, "ko_acc": a}
        finally:
            net.gate_on = gate_on
            net.train(was_training)
        order = sorted(layers, key=lambda b: -layers[b]["drop"])
        lead = order[0]
        second = order[1] if len(order) > 1 else None
        d1 = layers[lead]["drop"]
        d2 = layers[second]["drop"] if second is not None else 0.0
        scales = {b: (float(net.layers[b].attn_scale.detach().float()) if net.layers[b].attn_scale is not None
                      else None) for b in net.attn_ids}
        return {"method": self.method, "base_lp": base_lp, "base_acc": base_acc,
                "layers": {int(b): v for b, v in layers.items()}, "leader": int(lead),
                "second": (None if second is None else int(second)),
                "margin": d1 - d2, "split": bool(d1 > 0 and d2 >= 0.5 * d1), "scales": scales}


class RecallTracker:
    """측정 결과를 쌓아 회상 층을 판정한다.

    leader 가 `patience` 번 연속 같고 그 층의 Δlogp 가 `min_drop` 이상이면 '확정' (decided).
    확정 뒤에 바뀌면 다시 센다 — 기록은 항상 마지막 측정의 leader 를 따른다 (게이트는 최종 가중치에 맞아야 하므로).

    기본값 근거 (원 연구 30M 시드 8개 · 스냅샷 step 500~4000, README 의 검증 표):
      step 500~750 에는 모든 층의 Δlogp 가 0.5 nats 아래이고 leader 가 시드의 최종 회상 층과 자주 다르다 (잡음).
      step 1000 (전체 11,963 step 의 ~8%, 워밍업 150) 부터는 8/8 시드에서 leader 가 최종 회상 층과 같고
      끝까지 바뀌지 않았다. 1B 최종 모델은 step 8000 (전체의 2.8%) 에 이미 블록 11 로 끝나 있었다.
    """

    def __init__(self, patience: int = 3, min_drop: float = 0.5, gate_token: str = "<|bot|>",
                 gate_token_id: int | None = None, history=None):
        self.patience, self.min_drop = patience, min_drop
        self.gate_token, self.gate_token_id = gate_token, gate_token_id
        self.history = list(history or [])

    def update(self, step: int, res: dict) -> dict:
        row = {"step": int(step), **res}
        self.history.append(row)
        return row

    def streak(self) -> tuple[int | None, int, int | None]:
        """(현재 leader, 연속 횟수, 연속이 시작된 step)"""
        if not self.history:
            return None, 0, None
        lead = self.history[-1]["leader"]
        n, start = 0, None
        for r in reversed(self.history):
            if r["leader"] != lead or _drop(r, lead) < self.min_drop:
                break
            n, start = n + 1, r["step"]
        return lead, n, start

    def record(self) -> dict | None:
        if not self.history:
            return None
        lead, n, start = self.streak()
        last = self.history[-1]
        return {"layer": lead, "gate_layers": [lead], "token": self.gate_token, "token_id": self.gate_token_id,
                "method": last["method"], "decided": n >= self.patience, "streak": n, "decided_step": start,
                "split": last["split"], "second": last["second"], "probe_step": last["step"],
                "last": _slim(last), "history": [_slim(r) for r in self.history]}

    def summary(self, res: dict | None = None) -> str:
        r = res or self.history[-1]
        lead, n, start = self.streak()
        cells = " ".join(f"b{b}:{v['drop']:+.2f}/{v['ko_acc']:.0f}%" for b, v in r["layers"].items())
        flag = " 나뉨" if r["split"] else ""
        state = f"확정(step {start}~, {n}회)" if n >= self.patience else f"후보 {n}/{self.patience}"
        return (f"[회상 측정 {r['method']}] 정답 {r['base_acc']:.1f}% logp {r['base_lp']:.2f} | "
                f"층별 끄기 Δlogp/정답 {cells} | 회상 층 b{r['leader']}{flag} ({state})")


def _drop(r: dict, b: int) -> float:
    if "layers" in r:
        return float(r["layers"][b]["drop"])
    return float(r["drop"][b])


def _slim(r: dict) -> dict:
    if "layers" not in r:            # 이미 줄인 기록 (재개)
        return r
    return {"step": r["step"], "leader": r["leader"], "method": r["method"], "second": r["second"],
            "base_acc": round(r["base_acc"], 2),
            "base_lp": round(r["base_lp"], 4), "split": r["split"],
            "drop": {int(b): round(v["drop"], 4) for b, v in r["layers"].items()},
            "ko_acc": {int(b): round(v["ko_acc"], 2) for b, v in r["layers"].items()}}
