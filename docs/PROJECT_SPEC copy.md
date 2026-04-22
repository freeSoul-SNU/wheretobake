# PROJECT_SPEC.md

## 프로젝트 이름

Where to Bake: Mechanism-Guided Selective Prompt Baking

---

## 1. 문제 정의

긴 고정 prompt가 반복적으로 동일한 행동을 유도하는 실제 응용에서는, 매 inference마다 prompt를 그대로 다시 넣는 방식이 비효율적이다.
기존 prompt baking / context distillation 계열은 보통 **prompt-conditioned teacher**의 출력을 **prompt-free student**가 모사하도록 학습하지만,
대개 모든 layer 또는 넓은 범위의 target module을 균일하게 업데이트한다.

본 프로젝트는 아래 가설을 검증한다.

> prompt가 유도하는 내부 변화는 일부 안정적이고 family-specific한 모듈에 집중된다. 
> 따라서 그 모듈에만 LoRA를 배치하고 teacher-relative delta를 distill하면,
> 더 적은 trainable parameter로 teacher fidelity를 유지하면서 base behavior degradation을 줄일 수 있다.

---

## 2. 기본 설정

- Base model: Hugging Face decoder-only causal LM
- Teacher: `M_base(prompt, x)`
- Student: `M_base + LoRA(x)`
- Student inference: **prompt 없음**
- Train objective: prompted teacher vs unprompted student alignment

### 2.1 단위 작업 정의

한 training/eval example은 아래 필드를 가진다.

- `example_id`
- `input_text`
- `prompt_family`
- `prompt_id`
- `split` (`train`, `valid`, `test`)
- `paraphrase_split` (`seen`, `unseen`)
- optional: `task_label`, `style_label`, `metadata`

---

## 3. Prompt Family

초기 논문 범위는 아래 family로 제한한다.

- `concise`: 짧고 핵심만 답하기
- `formal`: 격식 있고 정돈된 어조
- `step_by_step`: 단계적/절차적 설명
- `refusal_safe` (선택): 위험 요청에 더 보수적으로 거절

각 family는 paraphrase 10~30개를 가진다.

예시:

- concise-1: "Answer briefly and only include the key point."
- concise-2: "Be concise. Avoid unnecessary explanation."
- concise-3: "Respond with the minimum explanation needed."

### 3.1 Family 정의 원칙

- prompt family는 **표면 문장 유사성**이 아니라 **의도된 행동 변화의 동등성**으로 정의한다.
- 단어가 많이 달라도 같은 행동을 유도하면 같은 family가 될 수 있다.
- 단어가 비슷해도 행동 목표가 반대면 다른 family다.

### 3.2 초기 버전의 구성 방식

- family assignment는 **수동(manual)** 으로 수행한다.
- automatic clustering / semantic grouping은 초기 범위에서 제외한다.
- 각 family는 lexical overlap이 높은 paraphrase와 낮은 paraphrase를 모두 포함한다.
- 각 family는 seen paraphrase와 unseen paraphrase를 명시적으로 분리 저장한다.

### 3.3 QC 규칙

1. 반대 지시문은 같은 family에 넣지 않는다.
2. topical similarity만으로 묶지 않는다.
3. family membership은 사람이 검증한다.
4. family 외부 prompt와 family 내부 paraphrase를 명확히 분리 저장한다.
5. 평가에는 반드시 unseen paraphrase transfer를 포함한다.

상세 형식은 `docs/PROMPT_FAMILIES.md`를 따른다.

---

## 4. 알고리즘 개요

### Stage A. Localization Dataset 생성

입력 `x`, prompt family `F`, family 내 paraphrase `p_j`에 대해 아래를 수집한다.

- base hidden / module output
- teacher hidden / module output
- teacher logits

즉, `(x, F, p_j)` 단위로 base 대비 teacher의 내부 변화를 기록한다.

### Stage B. Prompt-Induced Module Localization

모듈 `m`에 대해 base-relative change를 정의한다.

- `Delta_m(x, p) = h_m(x, p) - h_m(x, null)`

stability는 **같은 family 안에서 일관되고, 다른 family와는 구분되는가**로 정의한다.

#### 4.1 점수 정의

1. `within_family_consistency[m, F]`
   - 같은 family `F` 내부 paraphrase 쌍 `(p_i, p_j)`에 대해 `Delta_m(x, p_i)`와 `Delta_m(x, p_j)`가 얼마나 유사한가
2. `across_family_similarity[m, F]`
   - `F`의 prompt와 다른 family `G != F`의 prompt가 동일 모듈에서 얼마나 비슷하게 반응하는가
3. `stability_score[m, F]`
   - `within_family_consistency[m, F] - alpha * across_family_similarity[m, F]`
4. `causal_score[m, F]`
   - module patching / ablation 시 teacher behavior 복원 또는 붕괴에 미치는 영향
5. `combined_score[m, F]`
   - 기본값: `stability_score[m, F] * causal_score[m, F]`

#### 4.2 수식적 정의

입력 `x`, family `F`, prompt `p`에 대해

- `Delta_m(x, p) = h_m(x, p) - h_m(x, null)`

그러면,

- `WithinFamilySim[m, F]`
  - 같은 family `F` 내부의 서로 다른 paraphrase 쌍 `(p_i, p_j)`에 대해
  - `cos(Delta_m(x, p_i), Delta_m(x, p_j))` 평균

- `AcrossFamilySim[m, F]`
  - `p in F`, `q in G`, `G != F` 쌍에 대해
  - `cos(Delta_m(x, p), Delta_m(x, q))` 평균

- `stability_score[m, F] = WithinFamilySim[m, F] - alpha * AcrossFamilySim[m, F]`

선택 모듈 집합 `M_F`는 family별로 얻는다.

### Stage C. Selective LoRA Placement

`M_F`에만 LoRA를 삽입한다.

초기 granularity:

- transformer block output
- attention output projection block
- MLP block

후속 granularity:

- q/k/v/o projection
- neuron/head subset

### Stage D. Distillation

#### 4.3 Token KL

teacher의 prompted next-token distribution과 student의 unprompted distribution을 맞춘다.

#### 4.4 Delta Loss

선택 모듈 `m`에 대해

- `d_T[m] = norm(h_T[m] - h_B[m])`
- `d_S[m] = norm(h_S[m] - h_B[m])`
- `delta_loss = sum_m || d_T[m] - d_S[m] ||^2`

초기 norm 후보:

- layernorm-style standardization
- cosine distance on normalized residuals
- simple L2 on normalized residuals

#### 4.5 Preserve Loss

prompt와 무관한 base behavior 보존을 위한 regularizer

- `KL(base || student)` on unrelated inputs
- optional: hidden drift penalty on unrelated inputs

### 4.6 전체 Loss

`L = L_kl + lambda_delta * L_delta + lambda_preserve * L_preserve`

반드시 아래 ablation이 가능해야 한다.

- KL only
- KL + delta
- KL + preserve
- KL + delta + preserve

---

## 5. Baselines

상세 정의는 `docs/BASELINES.md`를 source of truth로 따른다.

### 5.1 Main Baselines

동일 teacher/student 구조와 동일 benchmark protocol 안에서 직접 비교 가능한 방법:

1. Prompt Baking style KL-only LoRA
2. Full target-module LoRA + KL only
3. All-layer LoRA + KL only
4. Random module subset + KL only
5. Magnitude-based top-k selection
6. Gradient-based selection
7. Ours: mechanism-guided + delta + preserve

### 5.2 Extension Baselines

문제 설정이 완전히 같지 않을 수 있으므로 appendix/extension table로 분리하는 방법:

1. GenPI-style internalization
2. OPCD-style refinement

### 5.3 공통 Benchmark 조건

가능한 모든 main baseline은 아래 조건을 공유한다.

- 동일 base model
- 동일 tokenizer
- 동일 prompt family 정의
- 동일 train/valid/test split
- 동일 seen/unseen paraphrase split
- 동일 result JSON schema
- 동일 evaluator
- 동일 random seed set

---

## 6. 평가 항목

### 6.1 Teacher Fidelity

- task accuracy / task reward / exact match (가능한 경우)
- style agreement score
- token KL / next-token agreement

### 6.2 Preservation

- base model general benchmark drop
- unrelated prompt behavior drift
- family 내부 unseen paraphrase transfer
- non-target family degradation

### 6.3 Efficiency

- trainable parameter count
- wall-clock training time
- peak GPU memory
- inference latency
- optional: merged LoRA vs unmerged latency

상세 평가는 `docs/EVAL_PROTOCOL.md`를 따른다.

---

## 7. 데이터 및 저장 계약

### 7.1 Localization Cache 최소 필드

- `example_id`
- `prompt_family`
- `prompt_id`
- `module_name`
- `base_output`
- `teacher_output`
- `teacher_logits`

### 7.2 Result JSON

모든 baseline 실행 결과는 `docs/RESULT_SCHEMA.md`를 따른다.

### 7.3 Config 계약

모든 학습/평가 설정은 `docs/CONFIG_SPEC.md`를 따른다.

---

## 8. 구현 Milestone

### M0

- HF + PEFT 기반 KL-only baking baseline 재현
- smoke train + tiny eval + result JSON 저장

### M1

- baseline registry + common evaluator 구축
- module hooks 및 delta extraction
- human-defined prompt family loader

### M2

- localization score 계산
- selective LoRA placement
- family별 module selection 저장

### M3

- delta loss / preserve loss 추가
- ablation + seen/unseen transfer + preservation 평가

### M4

- extension baseline 연결
- summary table 자동 생성
- optional on-policy refinement

---

## 9. 초기 성공 기준

- KL-only all-layer LoRA 대비, 비슷한 fidelity에서 trainable params 감소
- random selection 대비 일관된 향상
- preserve loss 추가 시 unrelated/base degradation 감소
- family 내부 prompt paraphrase transfer 확인
- unseen paraphrase에도 family-level transfer 확인

---

## 10. 위험 요소

1. localization noise가 커서 stable module이 잘 안 잡힐 수 있다.
2. delta loss가 KL optimization을 방해할 수 있다.
3. selective placement의 이득이 random top-k와 큰 차이가 안 날 수 있다.
4. family 범위가 너무 넓으면 결론이 흐려진다.
5. 잘못된 family assignment가 stability score를 왜곡할 수 있다.
6. generic module이 높은 stability를 보이는 것처럼 보일 수 있다.

위 리스크에 대한 대응은 `docs/EXECPLAN.md`와 `docs/DECISIONS.md`에 누적 기록한다.

