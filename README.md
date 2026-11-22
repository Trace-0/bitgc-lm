# BitGC LM — 삼진 가중치 하이브리드 언어 모델과 회상 층 게이트

한국어 1B 언어 모델(삼진 가중치 BitNet b1.58 + 게이트 conv/attention 하이브리드)의 **학습·추론 코드**입니다.
졸업 논문 「비트 연산 언어 모델」의 최종 모델(E1d-G)을 만든 절차를 정리했고, 원 연구에서 사람이 손으로 하던
**회상 층 찾기 → 응답 첫 토큰에서 막기** 를 학습·추론 파이프라인 안으로 옮겼습니다.

*English summary: training and inference code for a 1B Korean LM with ternary (BitNet b1.58) weights and a
gated-conv / attention hybrid. During pretraining an automatic probe finds the "recall layer" (the attention layer
that copies values from context) and stores it in the checkpoint; SFT and inference then mask that layer's output
at the response-start token, which removed question-echoing in the final model.*

## 무엇이 문제였고, 무엇을 자동화했나

1. 팽창 없는 gconv 하이브리드에서는 attention 층 중 **하나가 문맥 회상을 도맡는다** (끄면 연상 회상 정답이 0 근처).
   어느 층이 맡는지는 학습 초반의 '경주'로 정해진다 — 30M 시드 8개 중 블록 3 이 6개, 블록 7 이 2개.
   그래서 **층 수·시드·데이터가 바뀌면 회상 층이 어디에 잡힐지 미리 알 수 없다.**
2. 1B 최종 사전학습 모델은 블록 11 이 회상 층이었고, SFT 뒤 그 층이 답의 첫 토큰을 질문 첫머리에서 베껴
   **질문을 되풀이**했다. 응답 첫 토큰 위치(`<|bot|>`)에서 그 층의 출력을 0 으로 막고 SFT 한 모델이 최종 모델이다.
   - 추론 때만 막으면 되풀이가 '빈 답'으로 바뀔 뿐이다 → **SFT 부터 막아야** 다른 층이 답 시작을 넘겨받는다.
   - 막고 배운 모델도 게이트를 떼면 다시 무너진다 → **게이트는 배포 구조의 일부**다.
   - 사전학습부터 막으면 4000 step 안에 다른 층이 같은 베끼기를 다시 만든다 → 사전학습에는 걸지 않는다.

이 저장소는 이것을 자동으로 한다.

| 단계 | 하는 일 | 코드 |
|---|---|---|
| 사전학습 | `--probe-every` step 마다 attention 층을 하나씩 질의 위치에서 끄고 연상 회상 정답 log p 가 얼마나 떨어지는지 잰다. 가장 크게 떨어뜨리는 층 = 회상 층. 같은 층이 3번 연속(Δlogp ≥ 0.5)이면 '확정'. 결과를 `best.pt`·`last.pt` 의 `recall` 기록으로 남긴다 | `train.py`, `bitgc/recall_probe.py` |
| SFT | 체크포인트의 `recall.gate_layers` 를 읽어 모든 `<|bot|>` 위치에서 그 층의 혼합 출력을 0 으로 막은 채 학습. 기록을 산출물에 그대로 넘긴다. 평가 때 '첫 답 베끼기' 비율을 게이트 있음/없음으로 같이 보고 | `sft.py` |
| 추론 | 불러올 때 기록대로 게이트를 건다 (`--no-gate`, `/gate off` 로 비교 가능) | `chat.py`, `bitgc/generate.py` |

## 검증 — 자동 측정이 원 연구의 회상 층 판정을 재현하는가

원 연구의 체크포인트를 변환해 `RecallProbe` (기본 `assoc`) 로 다시 쟀다. 변환한 모델의 로짓은 원 코드와 차이 0 (최대 |Δ| = 0.0).

**30M (d512 L8, attention 블록 3·7), 시드 8개 — 끝 모델** (원 판정은 질의 위치 끄기·가리킴 기준, 원 연구 §24.4)

| 시드 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 원 판정 | 7 (블록 3 몫 남음) | 3 | 3 | 3 | 7 | 3 | 3 | 3 |
| 자동 측정 (Δlogp 블록 3 / 7) | **7** (0.56 / 2.81) | **3** (2.88 / 0.51) | **3** | **3** | **7** (0.02 / 5.42) | **3** | **3** | **3** |

**학습 초반 스냅샷** (같은 8 시드): step 500·750 은 모든 층의 Δlogp 가 대부분 0.5 nats 아래이고 leader 가 자주 틀린다.
**step 1000 (전체의 ~8%, 워밍업 150) 부터 8/8 시드에서 leader = 끝 회상 층이고 step 4000·끝까지 한 번도 바뀌지 않았다.**
→ 기본값 `--probe-every 500 · --probe-patience 3 · --probe-min-drop 0.5`.

**150M (d1024 L12, 블록 3·7·11)**: 원 판정 블록 7 (s1·s2·팽창 있음 s0) → 자동 측정 7·7·7. 블록 3·7 에 나뉜 s0 도 필수 층인 7 을 고른다 (Δlogp 0.99 / 2.42).

**1B** (원 판정: 최종 모델의 사전학습본 E1d = 블록 11 단독, 블록 15 보조 · 이전 설계 E1c = 블록 7):

| 1B | Δlogp 블록 3 / 7 / 11 / 15 / 19 / 23 | 자동 측정 |
|---|---|---|
| E1d (최종 모델의 사전학습본) | −0.04 / 0.07 / **2.19** / 1.10 / 0.08 / 0.02 | **블록 11** (블록 15 가 절반 → `split` 표시) |
| E1c (팽창 conv, 이전 설계) | −0.11 / **1.12** / 0.14 / 0.76 / 0.14 / 0.06 | **블록 7** |

공개 가중치의 `recall` 기록은 블록 11 이다.

**학습 중 자동 측정 — 이 저장소의 `train.py` 로 30M 을 다시 학습** (원 연구 경주와 같은 인자·시드, 3000 step 에서 멈춤):

| 시드 | 원 판정 | step 750 | 1000 | 1250 | 1500 | 3000 | 확정 | `last.pt` 기록 |
|---|---|---|---|---|---|---|---|---|
| 1 | 블록 3 | b7 (0.05/0.49) | b3 나뉨 (0.93/0.83) | b3 (1.43/0.68) | b3 (1.76/0.77) | b3 (1.94/0.92) | step 1500 (1000~) | `gate_layers: [3]` |
| 4 | 블록 7 | b7 (0.00/1.48) | b7 (−0.08/3.05) | b7 (−0.04/3.33) | b7 | b7 (0.00/4.13) | step 1250 (750~) | `gate_layers: [7]` |

(괄호 = 블록 3 / 7 을 끈 Δlogp.) 이어서 `sft.py` 로 짧게 대화 SFT 를 하면 기록된 층이 자동으로 게이트가 되고,
응답 첫 위치의 질문 첫 토큰 베끼기가 게이트 있음 0.3% / 게이트 뗌 68.5% 로 갈린다 (시드 4, SFT 740 step).

`induction` (무작위 토큰 열 반복, 토크나이저·언어 무관) 도 30M 끝 모델·step 1000 이후 8/8 을 맞히지만, 나뉜 회로인
150M s0 에서는 블록 3 을 고른다 (6.83 / 6.18) — 원 연구의 '질의 위치 끄기' 정의에 가까운 `assoc` 을 기본으로 두고,
토크나이저가 assoc 문항을 못 만들 때만 `induction` 으로 넘어간다 (`--probe-method auto`).

## 설치

```bash
pip install -r requirements.txt      # torch ≥ 2.4 (bf16 · torch.compile), sentencepiece, numpy
python tests/test_model.py           # CPU 수 초 — 증분 생성 = 전체 계산, 게이트 위치, 저장·복원
```

## 사용법

### 1. 토크나이저와 데이터

```bash
python scripts/train_tokenizer.py --input corpus.txt chat.jsonl --out tokenizer/my_spm --vocab 16384
python scripts/prepare_pretrain.py --input wiki.jsonl docs.txt --tokenizer tokenizer/my_spm.model \
    --out data/corpus.bin --shuffle-docs
python scripts/prepare_sft.py --input chat.jsonl --tokenizer tokenizer/my_spm.model --out data/sft_chat.pt
```

토크나이저에는 대화 표지 `<|user|> <|bot|> <|end|>` 가 있어야 한다 (`train_tokenizer.py` 가 넣는다).
공개 모델과 같은 토크나이저를 쓰려면 Hugging Face 모델 저장소에서 받는다 (이 저장소에는 없다):
`huggingface-cli download trace-000/bitgc-lm-1b tokenizer.model --local-dir tokenizer`
SFT jsonl 한 줄: `{"turns": [["user", "질문"], ["bot", "답"], …]}`.

### 2. 사전학습 (회상 층 자동 측정)

```bash
# 1B — 최종 모델과 같은 인자 (원 연구: A100 40GB ~33h, 2.375B 토큰 = 289,418 step)
python train.py --data data/corpus.bin --tokenizer tokenizer/my_spm.model --config configs/bitgc_1b.json \
    --out-dir runs/1b --batch-size 16 --lr 1.5e-4 --warmup 3000 --amp bf16 --compile --grad-ckpt \
    --eval-every 8000 --ckpt-every 4000 --probe-every 500 --max-hours 22.5

# 30M 로 회상 층만 미리 보기 (lr 일정은 전체 길이 그대로, 3000 step 에서 멈춤)
python train.py --data data/corpus.bin --tokenizer tokenizer/my_spm.model --config configs/bitgc_30m.json \
    --out-dir runs/30m --lr 1.2e-3 --warmup 150 --compile --stop-at-step 3000 --probe-every 250
```

로그 예:
```
  step 1000 [회상 측정 assoc] 정답 2.8% logp -6.08 | 층별 끄기 Δlogp/정답 b3:+1.05/1% b7:+0.63/2% | 회상 층 b3 (후보 1/3)
  step 2000 [회상 측정 assoc] … | 회상 층 b3 (확정(step 1000~, 3회))
```
측정은 매번 `recall.jsonl` 에 한 줄씩 남는다. 같은 명령을 다시 실행하면 `resume.pt` 에서 이어진다.
층 수를 바꿀 때는 `--n-layers 12 --attn-layers 3,7,11` 처럼 인자로 덮어쓴다.

### 3. SFT (기록된 회상 층을 응답 첫 토큰에서 막고)

최종 모델은 세 단계를 이어서 학습했다 — 각 단계의 산출이 다음 단계의 `--ckpt`.

```bash
python sft.py --ckpt runs/1b/best.pt --data data/sft_chat.pt --tokenizer tokenizer/my_spm.model --out runs/1b/chat.pt \
    --resume runs/1b/sft_resume.pt --epochs 1 --epoch-frac 0.5 --lr 3e-5 --warmup 200 \
    --replay data/corpus.bin --replay-per-batch 4 --replay-weight 0.5 --compile           # 대화 (원 연구 ~7h, 3090)
python sft.py --ckpt runs/1b/chat.pt --data data/sft_rag.pt   … --out runs/1b/rag.pt   --lr 3e-5 --warmup 30 --replay …
python sft.py --ckpt runs/1b/rag.pt  --data data/sft_instr.pt … --out runs/1b/instr.pt --lr 3e-5 --warmup 30 --replay …
```

로그 예 (30M 시연, 시드 4):
```
게이트: 블록 [7] 혼합 출력 × 0 @ 모든 <|bot|> (id 5) — 회상 측정 assoc · 확정 True (step 750~)
  ep1 step 740 val 3.6392 (ppl 38.06)  ★ 최고 (저장) | 첫 답 베끼기 0.3% (틀린 0.1%) · 첫 토큰 정답 2.5% [734자리] | 게이트 뗌: 베끼기 68.5% (틀린 13.1%) · 정답 57.2%
```
(파이프라인 확인용 740 step 짧은 SFT 라 게이트 쪽 첫 토큰 정답은 아직 2.5% 로 낮다 — 다른 층이 답 시작을 아직 넘겨받지 못했다.
원 연구 1B 에서도 지시 SFT 10분만 게이트로 학습하면 열린 대화의 56% 가 빈 답이었고, 대화 SFT 전체(~7h)를 게이트로 학습해서야
풀렸다. 게이트 뗀 쪽 정답이 높은 것은 질문 첫 토큰이 실제 정답인 자리가 많기 때문이다 (원 연구 대화 데이터 27.7%).)

`--no-gate` 로 게이트 없이, `--gate-layers 7,11` 로 기록 대신 다른 층을 막을 수 있다 (비교용).

### 4. 추론

```bash
python chat.py --ckpt runs/1b/instr.pt --tokenizer tokenizer/my_spm.model                 # 여러 턴 대화
python chat.py --ckpt … --tokenizer … --prompt "프랑스의 수도는 어디야?" --compare-gate     # 게이트 켬/끔 나란히
python chat.py --ckpt … --tokenizer … --text "옛날 옛적에" --temperature 0.7 --top-k 40      # 이어 쓰기
```

디코딩 기본값은 원 연구의 최종 평가 설정이다: greedy · 반복 벌점 1.2 · 6-gram 금지 · 첫 답 토큰으로 맨 공백(`▁`) 금지 ·
답 안에서 `<|bot|>`·`<|user|>` 금지. 증분 생성(KV 캐시 + conv 상태)이라 CPU 에서도 1B 를 돌릴 수 있다 (fp32 ~4.4GB).

### 5. 회상 층 다시 재기 · 원 연구 체크포인트 변환

```bash
python scripts/probe_recall.py --ckpt runs/1b/best.pt --tokenizer … --K 4,8,16           # 재기만
python scripts/probe_recall.py --ckpt runs/1b/best.pt --tokenizer … --write              # 기록 갱신
python scripts/convert_logiformer.py --src 1b_best.pt --tokenizer … --out base.pt --probe  # 원 연구 형식 → bitgc-v1
```

## 모델 구조

| | 1B (configs/bitgc_1b.json) |
|---|---|
| 형상 | d2048 · 24층 · 32헤드 · FFN 8192 · 문맥 512 · vocab 16384 · 1092.9M |
| 혼합층 | attention 블록 3·7·11·15·19·23, 나머지 18층은 게이트 conv (커널 9, 팽창 없음) |
| 선형층 | 전부 BitLinear: LayerNorm(fc2 는 RMSNorm) → 토큰별 absmax 8bit → 삼진 absmean 가중치 |
| FFN | ReLU² |
| 잔차 | 혼합층·FFN 출력에 학습 스칼라 (초깃값 1/√48) |
| 그 밖 | 학습 위치 임베딩 · 임베딩/출력 가중치 공유 · dropout 0.1 · AdamW(0.9, 0.95, wd 0.01) · 기울기 자르기 1.0 |

체크포인트(`bitgc-v1`)는 가중치 + `config` + `recall` (회상 층·게이트 층·응답 표지 id·측정 기록) + `tokenizer` 정보를 담는다.
`bitgc/checkpoint.py` 머리말 참조.

## 공개 가중치

| 모델 | 내용 |
|---|---|
| [bitgc-lm-1b](https://huggingface.co/trace-000/bitgc-lm-1b) | 최종 대화·지시 모델 (E1d-G) — 회상 층 블록 11 게이트 포함 |
| [bitgc-lm-1b-base](https://huggingface.co/trace-000/bitgc-lm-1b-base) | 사전학습 모델 (E1d) — 회상 층 측정 기록 포함, 직접 SFT 할 때 |

```bash
python chat.py --ckpt hf:trace-000/bitgc-lm-1b                      # 토크나이저는 저장소에 함께 있다
```

`scripts/export_pretrained.py` 가 `.pt` 를 이 형식(model.safetensors + config.json + tokenizer.model)으로 바꾼다.

## 출처 표기

공개 가중치와 토크나이저는 **AI 허브 (https://www.aihub.or.kr)** 의 다음 데이터를 활용해 만들었다:
009. 전문분야_기술과학_한국어 멀티세션 데이터 · 010. 전문분야_사회과학_한국어 멀티세션 데이터 · 045. 지식검색 대화 ·
048. 일반상식 문장 생성 데이터 · 049. 일반상식 문장 생성 평가 데이터 · 121. 한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터.
그 밖에 한국어 위키백과 (https://ko.wikipedia.org, CC BY-SA 4.0) 를 썼다. 데이터 자체는 이 저장소에 포함하지 않는다.

## 한계

- 회상 층 = 되풀이를 만드는 층이라는 대응은 **1B 모델 하나**에서 확인했다 (연상 회상·비밀번호 회상·되풀이 모두 블록 11).
  다른 설정에서는 `sft.py` 의 '첫 답 베끼기 · 게이트 뗌' 비교로 확인할 것.
- 게이트는 기본으로 **한 층**만 막는다. 회로가 두 층에 나뉜 모델(측정 결과 `split: true`)에서는 한 층만 막아도 되는지 검증하지 않았다.
- 측정 문항(`assoc`)은 한국어 낱말·두 자리 숫자 형식이다. 다른 언어 토크나이저는 영어 낱말 목록을 쓰고, 숫자가 한 토큰이 아니면 `induction` 으로 넘어간다.
- 평가 수치는 원 연구의 자체 평가다 (외부 벤치마크 아님). 1B 모델은 사실 지식·긴 문맥·열린 대화가 약하다.
