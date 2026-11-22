---
language:
- ko
license: mit
library_name: pytorch
tags:
- bitnet
- ternary
- 1.58-bit
- korean
- gated-convolution
pipeline_tag: text-generation
---

# bitgc-lm-1b-base (사전학습 모델, 논문의 E1d)

삼진 가중치(BitNet b1.58) + 게이트 conv / attention 하이브리드 한국어 언어 모델 1.09B 의 **사전학습본**입니다 (대화·지시 미세조정 전). 대화용은 bitgc-lm-1b.
코드: https://github.com/Trace-0/bitgc-lm

- 블록의 모든 선형층이 삼진 {−γ, 0, +γ} 가중치 · 8bit 활성 (BitLinear), FFN ReLU², 잔차 스칼라
- 24층 중 attention 은 블록 3·7·11·15·19·23 의 6층, 나머지 18층은 게이트 conv (커널 9)
- **회상 층 게이트**: 사전학습 중 자동 측정으로 찾은 회상 층(블록 11)의 출력을 응답 첫 토큰 위치(`<|bot|>`)에서 막는다.
  `config.json` 의 `recall` 에 측정 기록이 있고, 이 모델로 `sft.py` 를 돌리면 그 게이트를 건 채 학습한다.
  (사전학습 텍스트에는 `<|bot|>` 이 없어 이 모델의 일반 생성에는 게이트가 영향이 없다.)

## 사용

```bash
git clone https://github.com/Trace-0/bitgc-lm && cd bitgc-lm && pip install -r requirements.txt
python chat.py --ckpt hf:trace-000/bitgc-lm-1b-base --text "세종대왕은" --temperature 0.7 --top-k 40     # 이어 쓰기
python scripts/probe_recall.py --ckpt hf:trace-000/bitgc-lm-1b-base --K 4,8,16                           # 회상 층 다시 재기
python sft.py --ckpt hf:trace-000/bitgc-lm-1b-base --data my_sft.pt --out my_chat.pt …                   # 회상 층 게이트를 건 SFT
```

## 학습

AI Hub 본문 + 한국어 위키백과 혼합 2.375B 토큰 · 289,418 step · 배치 16×512 · lr 1.5e-4 (워밍업 3000, 코사인) · bf16 · A100 ~33h.
val ppl 9.02 (같은 val 에서 이전 설계 E1c 9.25).

## 출처 표기 (학습 데이터)
 (학습 데이터)

이 모델은 **AI 허브 (https://www.aihub.or.kr)** 의 다음 데이터를 활용해 만들었다.

- 009. 전문분야_기술과학_한국어 멀티세션 데이터
- 010. 전문분야_사회과학_한국어 멀티세션 데이터
- 045. 지식검색 대화
- 048. 일반상식 문장 생성 데이터
- 049. 일반상식 문장 생성 평가 데이터
- 121. 한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터

그 밖에 **한국어 위키백과** (https://ko.wikipedia.org, CC BY-SA 4.0) 문서를 사전학습에 썼다.
토크나이저(`tokenizer.model`, SentencePiece BPE 16384)는 위 AI 허브 데이터로 학습했다.
