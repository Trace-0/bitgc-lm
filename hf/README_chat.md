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

# bitgc-lm-1b (대화·지시 모델, 논문의 E1d-G)

삼진 가중치(BitNet b1.58) + 게이트 conv / attention 하이브리드 한국어 언어 모델 1.09B 의 **대화·지시 미세조정본**입니다.
코드: https://github.com/Trace-0/bitgc-lm

- 블록의 모든 선형층이 삼진 {−γ, 0, +γ} 가중치 · 8bit 활성 (BitLinear), FFN ReLU², 잔차 스칼라
- 24층 중 attention 은 블록 3·7·11·15·19·23 의 6층, 나머지 18층은 게이트 conv (커널 9)
- **회상 층 게이트**: 사전학습 중 자동 측정으로 찾은 회상 층(블록 11)의 출력을 응답 첫 토큰 위치(`<|bot|>`)에서 막는다.
  `config.json` 의 `recall` 에 기록돼 있고, 코드가 불러올 때 자동으로 건다. 대화 SFT 부터 이 게이트를 건 채 학습했다.

## 사용

```bash
git clone https://github.com/Trace-0/bitgc-lm && cd bitgc-lm && pip install -r requirements.txt
python chat.py --ckpt hf:trace-000/bitgc-lm-1b                                    # 여러 턴 대화
python chat.py --ckpt hf:trace-000/bitgc-lm-1b --prompt "프랑스의 수도는 어디야?" --compare-gate
```

CPU fp32 로 돌릴 수 있다 (가중치 4.4GB, 증분 생성). 문맥 길이 512 토큰.

## 학습

| 단계 | 데이터 | 설정 |
|---|---|---|
| 사전학습 (E1d) | AI Hub 본문 + 한국어 위키백과 혼합 2.375B 토큰 | 289,418 step · 배치 16×512 · lr 1.5e-4 코사인 · bf16 · A100 ~33h |
| 대화 SFT | AI Hub 대화 (아래 009·010·045·121) | 0.5 epoch · lr 3e-5 · 사전학습 코퍼스 replay 4/8 · 게이트 블록 11 |
| 글 읽기 SFT | 위키 문서를 붙인 질문 → 글 속 정답 (위키 기반 자동 생성) + 대화 | 1 epoch · lr 3e-5 · 게이트 |
| 지시 SFT | 형식 지시 문항 (자동 생성, 평가와 표현·낱말 분리) + 대화 | 1 epoch · lr 3e-5 · 게이트 |

## 평가 (원 연구의 자체 평가, E1c = 같은 크기·데이터의 이전 설계: 팽창 conv · 잔차 스칼라 없음 · 게이트 없음)

| | 이 모델 | E1c |
|---|---|---|
| 사전학습 val ppl (같은 val) | 9.02 | 9.25 |
| 지시 따르기 126 문항 (디코딩 규칙, 지시 SFT 시드 3개 평균) | **75.0** | 55.7 |
| 같은 문항, 디코딩 규칙 없음 · 시드 0: 게이트 켬 → 뗌 | 64 → 22 | 52 |

자체 제작 평가이며 외부 벤치마크가 아니다. 사실 지식·긴 문맥·자유 대화는 약하고, 틀린 내용을 지어낼 수 있다.

## 출처 표기 (학습 데이터)

이 모델은 **AI 허브 (https://www.aihub.or.kr)** 의 다음 데이터를 활용해 만들었다.

- 009. 전문분야_기술과학_한국어 멀티세션 데이터
- 010. 전문분야_사회과학_한국어 멀티세션 데이터
- 045. 지식검색 대화
- 048. 일반상식 문장 생성 데이터
- 049. 일반상식 문장 생성 평가 데이터
- 121. 한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터

그 밖에 **한국어 위키백과** (https://ko.wikipedia.org, CC BY-SA 4.0) 문서를 사전학습과 글 읽기·지시 SFT 문항 생성에 썼다.
토크나이저(`tokenizer.model`, SentencePiece BPE 16384)는 위 AI 허브 데이터로 학습했다.
