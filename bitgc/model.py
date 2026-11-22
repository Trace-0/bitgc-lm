"""BitGC 언어 모델 — 삼진 가중치(BitNet b1.58) + 게이트 conv / attention 하이브리드.

구성 (최종 1B 설정, `configs/bitgc_1b.json`):
  - 블록의 모든 Linear = BitLinear: 입력 정규화 → 토큰별 absmax 8bit 활성 → 삼진 absmean 가중치 (STE)
  - 혼합층: `attn_layers` 의 층만 causal self-attention, 나머지는 게이트 conv
        v = W_v(x);  out = W_o( conv_mix(v) * sigmoid(conv_gate(v)) )   (인과 depthwise conv, 커널 9)
  - FFN 활성 ReLU² (fc2 입력에 정확한 0 → 배치 1 생성에서 그 열을 읽지 않아도 된다).
    fc2 의 입력 정규화만 RMSNorm (LayerNorm 은 평균을 빼서 0 을 없앤다)
  - 잔차 스칼라 (rscale): 혼합층·FFN 출력에 학습되는 스칼라, 초깃값 1/√(2L)
  - 학습 위치 임베딩 · tok_emb ↔ lm_head 가중치 공유

회상 층 게이트 (`set_recall_gate`):
  학습 중 측정으로 찾은 '회상 층'(문맥에서 값을 찾아 옮기는 attention 층)의 혼합 출력을
  입력 토큰이 응답 표지(<|bot|>)인 위치에서만 잔차에 더하지 않는다. 그 위치의 출력이
  응답의 첫 토큰을 고르므로 "응답 첫 토큰에서 회상 층을 막는다" 와 같다.
  KV 캐시는 그대로 채운다 (뒤 위치는 그 위치를 정상적으로 본다).

state_dict 키는 원 연구 코드(logiformer `transformer/std_gpt.py`, bitnet+gconv+rscale+relu2)와 같다 —
그쪽 체크포인트를 그대로 불러올 수 있다 (`scripts/convert_logiformer.py`).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ───────────────────────────── 설정 ─────────────────────────────
@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 2048
    n_heads: int = 32
    n_layers: int = 24
    d_ff: int = 8192
    seq_len: int = 512
    attn_layers: tuple = (3, 7, 11, 15, 19, 23)
    conv_kernel: int = 9
    conv_dilate: bool = False      # 층 i 팽창 2^(i mod 주기). 최종 모델은 끔 (원 연구 §24.10~11)
    ffn_act: str = "relu2"         # relu2 | gelu
    rscale: bool = True
    dropout: float = 0.1

    def __post_init__(self):
        self.attn_layers = tuple(int(i) for i in self.attn_layers)
        assert self.ffn_act in ("relu2", "gelu"), self.ffn_act
        assert all(0 <= i < self.n_layers for i in self.attn_layers), "attn_layers 범위 밖"
        assert self.d_model % self.n_heads == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["attn_layers"] = list(self.attn_layers)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


# ───────────────────────────── 부품 ─────────────────────────────
class BitLinear(nn.Linear):
    """BitNet b1.58 BitLinear: 정규화 → 토큰별 absmax int8 → 삼진 {−γ, 0, +γ} 가중치 (γ = 행렬 평균 |W|).
    양자화는 straight-through estimator 로 학습한다. bias 없음."""

    def __init__(self, d_in: int, d_out: int, norm: str = "ln"):
        super().__init__(d_in, d_out, bias=False)
        self.norm = nn.RMSNorm(d_in) if norm == "rms" else nn.LayerNorm(d_in)

    def forward(self, x):
        x = self.norm(x)
        sc = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x = x + ((x * sc).round().clamp(-128, 127) / sc - x).detach()
        w = self.weight
        g = w.abs().mean().clamp(min=1e-5)
        w = w + ((w / g).round().clamp(-1, 1) * g - w).detach()
        return F.linear(x, w)


class CausalDepthwiseConv(nn.Module):
    """채널별 인과 conv (왼쪽만 패딩). 파라미터 d_model×k."""

    def __init__(self, d_model: int, kernel: int = 9, dilation: int = 1):
        super().__init__()
        self.k, self.d = kernel, dilation
        self.weight = nn.Parameter(torch.randn(d_model, 1, kernel) / kernel)

    @property
    def span(self) -> int:              # 현재 위치 앞에서 필요한 과거 길이
        return (self.k - 1) * self.d

    def forward(self, x, past=None):    # x (B,T,d) · past (B,span,d) 또는 None(=0 패딩)
        h = x.transpose(1, 2)
        if past is None:
            h = F.pad(h, (self.span, 0))
        else:
            h = torch.cat([past.transpose(1, 2), h], dim=2)
        h = F.conv1d(h, self.weight, groups=x.shape[-1], dilation=self.d)
        return h.transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = BitLinear(d_model, 3 * d_model)
        self.W_o = BitLinear(d_model, d_model)
        self.p = dropout

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in (q, k, v))
        if cache is not None:            # 증분 생성: 이전 K·V 에 이어 붙인다
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        Tk = k.shape[2]
        if T == Tk:
            ctx = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        else:                            # 새 토큰 T 개가 과거 Tk−T 개 뒤에 온다
            i = torch.arange(T, device=x.device)[:, None] + (Tk - T)
            j = torch.arange(Tk, device=x.device)[None, :]
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=j <= i)
        return self.W_o(ctx.transpose(1, 2).reshape(B, T, D))


class GConvMixer(nn.Module):
    """게이트 conv 혼합: W_v → 인과 depthwise conv × sigmoid(게이트 conv) → W_o."""

    def __init__(self, d_model: int, kernel: int = 9, dilation: int = 1):
        super().__init__()
        self.W_v = BitLinear(d_model, d_model)
        self.W_o = BitLinear(d_model, d_model)
        self.mix = CausalDepthwiseConv(d_model, kernel, dilation)
        self.gate = CausalDepthwiseConv(d_model, kernel, dilation)

    def forward(self, x, cache=None):
        v = self.W_v(x)
        past = None
        if cache is not None:
            past = cache.get("v")
            if past is None:
                past = v.new_zeros(v.shape[0], self.mix.span, v.shape[2])
            cache["v"] = torch.cat([past, v], dim=1)[:, -self.mix.span:] if self.mix.span else past
        return self.W_o(self.mix(v, past) * torch.sigmoid(self.gate(v, past)))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, act: str = "relu2"):
        super().__init__()
        self.act = act
        self.fc1 = BitLinear(d_model, d_ff)
        self.fc2 = BitLinear(d_ff, d_model, norm=("rms" if act == "relu2" else "ln"))

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(F.relu(h).square() if self.act == "relu2" else F.gelu(h))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int):
        super().__init__()
        self.is_attn = i in cfg.attn_layers
        self.norm1 = nn.LayerNorm(cfg.d_model)
        if self.is_attn:
            self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
        else:
            dil = 2 ** (i % _dilate_period(cfg)) if cfg.conv_dilate else 1
            self.attn = GConvMixer(cfg.d_model, cfg.conv_kernel, dil)   # 이름 'attn' 은 원 코드와 키를 맞추려고
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg.d_model, cfg.d_ff, cfg.ffn_act)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.rscale:
            init = 1.0 / math.sqrt(2.0 * cfg.n_layers)
            self.attn_scale = nn.Parameter(torch.full((1,), init))
            self.ffn_scale = nn.Parameter(torch.full((1,), init))
        else:
            self.attn_scale = self.ffn_scale = None

    def forward(self, x, keep=None, cache=None):
        """keep: (B,T,1) 혼합층 출력에 곱할 값 (게이트·끄기), None 이면 그대로."""
        a = self.drop(self.attn(self.norm1(x), cache))
        if self.attn_scale is not None:
            a = self.attn_scale * a
        if keep is not None:
            a = a * keep
        x = x + a
        f = self.drop(self.ffn(self.norm2(x)))
        return x + (f if self.ffn_scale is None else self.ffn_scale * f)


def _dilate_period(cfg: ModelConfig) -> int:
    n = 1
    while (cfg.conv_kernel - 1) * (2 ** n) <= max(1, cfg.seq_len - 1):
        n += 1
    return n


# ───────────────────────────── 모델 ─────────────────────────────
class BitGCLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grad_ckpt = False
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        res_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for b in self.layers:
            nn.init.normal_(b.attn.W_o.weight, std=res_std)
            nn.init.normal_(b.ffn.fc2.weight, std=res_std)
        # 회상 층 게이트 — 가중치가 아니라 설정이다 (체크포인트의 'recall' 기록에서 온다)
        self.gate_layers: tuple = ()
        self.gate_token: int | None = None
        self.gate_on = True

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def attn_ids(self) -> list[int]:
        return [i for i, l in enumerate(self.layers) if l.is_attn]

    def set_recall_gate(self, layers, token_id: int | None):
        """layers 의 혼합 출력을 입력 토큰 == token_id 인 위치에서 0 으로. layers=() 면 끈다."""
        layers = tuple(int(i) for i in (layers or ()))
        assert all(self.layers[i].is_attn for i in layers), f"게이트 층은 attention 층이어야 한다: {layers}"
        self.gate_layers, self.gate_token = layers, (None if token_id is None else int(token_id))

    def _keeps(self, x, knockout, dtype):
        gm = None
        if self.gate_on and self.gate_layers and self.gate_token is not None:
            gm = (x != self.gate_token).unsqueeze(-1).to(dtype)        # 1 = 통과, 0 = 막음
        out = []
        for li in range(len(self.layers)):
            k = gm if (gm is not None and li in self.gate_layers) else None
            if knockout and li in knockout:
                ko = (1.0 - knockout[li].unsqueeze(-1).to(dtype))
                k = ko if k is None else k * ko
            out.append(k)
        return out

    def forward(self, x, knockout: dict | None = None, caches: list | None = None, pos0: int = 0):
        """x (B,T) 토큰. knockout = {층: (B,T) bool} 그 위치의 혼합 출력을 0 (회상 측정용).
        caches/pos0 는 증분 생성용 (`generate.py`)."""
        B, T = x.shape
        pos = torch.arange(pos0, pos0 + T, device=x.device).unsqueeze(0)
        h = self.emb_drop(self.tok_emb(x) + self.pos_emb(pos))
        keeps = self._keeps(x, knockout, h.dtype)
        for li, layer in enumerate(self.layers):
            c = None if caches is None else caches[li]
            if self.grad_ckpt and self.training and c is None:
                h = checkpoint(layer, h, keeps[li], use_reentrant=False)
            else:
                h = layer(h, keeps[li], c)
        return self.lm_head(self.norm(h))

    def new_caches(self) -> list:
        return [dict() for _ in self.layers]


def compile_layers(net: BitGCLM) -> None:
    """층마다 torch.compile (원 연구의 학습 경로와 같다). 수식은 같고 연산 퓨전만 다르다."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    for layer in net.layers:
        layer.forward = torch.compile(layer.forward, dynamic=False)


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in {id(p): p for p in net.parameters()}.values())
