# EVAL_PROTOCOL.md

이 문서는 모든 baseline이 따라야 하는 공통 평가 규약을 정의한다.
목표는 baseline마다 평가 기준이 달라지는 것을 막고, teacher fidelity / preservation / efficiency를 동일한 틀에서 비교하는 것이다.

---

## 1. 평가 원칙

- main baseline은 동일 evaluator를 사용한다.
- 같은 metric 이름은 같은 계산식을 의미해야 한다.
- seen/unseen paraphrase를 분리 평가한다.
- extension baseline은 가능한 부분만 동일 evaluator에 연결하고, 맞추지 못한 부분은 별도 표기한다.

---

## 2. 평가 축

### A. Teacher Fidelity

student가 prompt-free 상태에서 teacher(prompted)의 행동을 얼마나 잘 재현하는가.

### B. Preservation

student가 target family를 internalize한 뒤에도 base model의 비목표 행동을 얼마나 덜 훼손하는가.

### C. Efficiency

어떤 비용으로 위 성능을 달성했는가.

---

## 3. 평가 split

각 family에 대해 아래 split을 유지한다.

- `train_seen`
- `valid_seen`
- `test_seen`
- `test_unseen`

또한 preservation용으로 아래를 별도 둔다.

- `preserve_unrelated`
- `preserve_non_target_family`

### 3.1 Seen / Unseen 정의

- `seen`: 학습 중 teacher distillation에 사용한 prompt paraphrase
- `unseen`: 같은 family지만 학습에 쓰지 않은 held-out paraphrase

### 3.2 Unrelated 입력 정의

`preserve_unrelated`는 target family internalization과 직접 관련 없는 일반 입력 집합이다.
예:

- 일반 질의응답
- 설명/요약/번역 등 비목표 스타일 입력
- target family와 반대 의미를 갖지 않는 중립 입력

주의:

- target family와 정면 충돌하는 지시문은 preservation set에 섞지 않는다.
- preservation set의 목적은 **base drift** 측정이지 adversarial stress test가 아니다.

---

## 4. Teacher Fidelity Metrics

### 4.1 token KL

teacher prompted distribution과 student unprompted distribution 사이의 token-level KL.

- 낮을수록 좋음
- teacher-forced evaluation으로 계산

### 4.2 next-token agreement

teacher와 student의 argmax next-token 일치율.

- 높을수록 좋음
- generation-free 방식으로 빠르게 계산 가능

### 4.3 task metric

데이터셋에 명시적 정답 또는 task reward가 있을 때 사용한다.

가능한 항목:

- `task_accuracy`
- `task_exact_match`
- `task_reward`

### 4.4 style agreement

family-specific style rule을 student 출력이 얼마나 만족하는지 측정한다.

#### family별 기본 휴리스틱

- `concise`
  - 길이 제한 충족 여부
  - 불필요한 부연 감소
- `formal`
  - 문체 규칙 일치율
  - 금지된 구어체 표현 비율 감소
- `step_by_step`
  - numbered / ordered step presence
  - 절차적 표현 사용 여부
- `refusal_safe`
  - 위험 요청에 대한 거절/안전 전환 비율

초기 구현은 **rule-based scorer**로 시작하고, 이후 필요하면 classifier 기반 scorer를 추가한다.

---

## 5. Preservation Metrics

### 5.1 base_drift_kl

unrelated input에서 base model과 student 사이의 token-level KL.

- 낮을수록 좋음

### 5.2 unrelated_input_drift

unrelated input에서 base model 대비 student 출력의 변화 정도.

가능한 계산식:

- token agreement drop
- hidden drift
- output embedding cosine drop

초기 버전은 token KL 또는 next-token disagreement 기반으로 단순화 가능.

### 5.3 non_target_family_drop

다른 family prompt에 대한 성능 저하 정도.

예:

- concise family용 student가 formal family의 teacher fidelity를 얼마나 잃는가

### 5.4 general_eval_drop

외부 또는 일반 평가셋에서의 base 대비 student 성능 저하.

초기 M0/M1에서는 optional이며 `null` 허용.

---

## 6. Efficiency Metrics

반드시 아래를 기록한다.

- `trainable_params`
- `train_runtime_sec`
- `peak_memory_mb`
- `train_tokens_per_sec`
- `eval_tokens_per_sec`
- `inference_latency_ms`

### 6.1 측정 원칙

- batch size와 sequence length를 고정한다.
- merged / unmerged adapter 여부를 명시한다.
- 단일 GPU 측정인지 명시한다.
- warmup step 수를 고정한다.

---

## 7. 최소 평가 단계

### M0

- token KL
- next-token agreement
- tiny style agreement
- result JSON 저장

### M1

- seen / unseen paraphrase split 평가
- preservation on unrelated inputs
- efficiency logging

### M2+

- selection quality logging
- non-target family drop
- summary table generation

---

## 8. 보고 규칙

각 result JSON에는 아래를 반영한다.

- `prompt_family`
- `paraphrase_split`
- `teacher_fidelity_metrics`
- `preservation_metrics`
- `efficiency_metrics`

seed 평균 표를 만들 때는 아래를 기본 축으로 한다.

- baseline
- family
- seen/unseen

---

## 9. 금지 사항

- baseline마다 style scorer를 다르게 두지 말 것
- seen/unseen split을 섞어서 보고하지 말 것
- preservation metric 계산 시 target family 데이터를 재사용하지 말 것
- extension baseline의 평가 누락을 숨기지 말 것

