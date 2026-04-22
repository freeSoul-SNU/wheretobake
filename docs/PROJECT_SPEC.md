# PROJECT_SPEC.md

## 프로젝트 이름

Where to Bake: Mechanism-Guided Selective Prompt Baking (working title)

## 1. 문제 정의

긴 고정 prompt가 유도하는 행동을 매 inference마다 다시 넣는 것은 비효율적이다. 기존 prompt baking / context distillation은 prompt-conditioned teacher의 출력을 prompt-free student가 모사하게 학습하지만, 보통 모든 layer 또는 넓은 범위의 파라미터를 균일하게 업데이트한다.

우리는 다음 가설을 검증한다.

> prompt가 유도하는 내부 변화는 일부 안정적인 모듈에 집중된다. 따라서 그 모듈에만 LoRA를 배치하고 teacher-relative delta를 distill하면, 더 적은 파라미터로 teacher fidelity를 유지하면서 base behavior degradation을 줄일 수 있다.

## 2. 기본 설정

- Base model: Hugging Face decoder-only causal LM
- Teacher: `M_base(prompt, x)`
- Student: `M_base + LoRA(x)`
- Inference at student: **prompt 없음**

## 3. Prompt family

초기 범위는 좁게 잡는다.

- concise / brief answering
- formal style
- step-by-step reasoning style
- refusal/safe style (선택)

각 family는 paraphrase 10~30개를 가진다.

예시:

- concise-1: "Answer briefly and only include the key point."
- concise-2: "Be concise. Avoid unnecessary explanation."
- concise-3: "Respond with the minimum explanation needed."

### 3.1 Prompt family 정의 원칙

- prompt family는 **표면 문장 유사성**이 아니라 **의도된 행동 변화의 동등성(behavioral equivalence)**으로 정의한다.
- 즉, 단어가 많이 달라도 같은 행동을 유도하면 같은 family가 될 수 있다.
- 반대로 단어가 비슷해도 의미가 반대면 다른 family다.

예:

- 같은 family가 되어야 하는 경우
  - "Answer briefly."
  - "Be concise."
  - "Keep the response short and only include the key point."
- 같은 family가 되면 안 되는 경우
  - "Do not be concise."
  - "Give a detailed explanation."

위 두 문장은 concise prompt와 단어가 일부 겹칠 수 있어도, 행동 목표가 반대이므로 같은 family에 넣지 않는다.

### 3.2 초기 논문 버전의 family 구성 방식

- family assignment는 **수동(manual)** 으로 수행한다.
- 자동 semantic clustering이나 automatic paraphrase grouping은 초기 논문 범위에서 제외한다.
- 각 family는 다음을 모두 포함하도록 구성한다.
  1. 표면적으로 매우 유사한 paraphrase
  2. 단어는 꽤 다르지만 의미는 유사한 paraphrase
- 실험 시 family 외부의 prompt와 family 내부 paraphrase를 명확히 구분해 저장한다.

### 3.3 Paraphrase quality control

Paraphrase는 단순히 topical similarity가 아니라 **behavior preservation**을 만족해야 한다.

규칙:

1. 반대 지시문(opposite directive)은 같은 family에 넣지 않는다.
2. 키워드가 비슷해도 행동 목표가 다르면 분리한다.
3. family membership은 초기 버전에서 사람이 검증한다.
4. 각 family에는 lexical overlap이 높은 예시와 낮은 예시를 모두 포함한다.
5. 평가는 unseen paraphrase transfer를 포함해야 한다.
6. 평가 시 seen paraphrase와 unseen paraphrase를 분리한다.

## 4. 알고리즘 개요

### Stage A. Localization dataset 생성

입력 `x`, prompt family `F`, family 내 paraphrase `p_j`에 대해 다음을 수집한다.

- base hidden / module output
- teacher hidden / module output
- teacher logits
  을 저장한다.

즉, `(x, F, p_j)` 단위로 base와 teacher의 내부 변화를 기록한다.

기본 저장 단위:

- `input_id`
- `prompt_family`
- `prompt_id`
- `module_name`
- `base_output`
- `teacher_output`
- `teacher_logits`

### Stage B. Prompt-induced module localization

모듈 `m`에 대해 base-relative change를 정의한다.

- `Delta_m(x, p) = h_m(x, p) - h_m(x, null)`

여기서 stability는 "paraphrase가 비슷한 단어를 쓰는가"가 아니라,
**같은 behavioral family 안에서 일관되게 나타나고, 다른 family와는 구분되는가**로 정의한다.

#### 4.1 기본 점수 정의

1. `within_family_consistency[m, F]`

   - 동일 family `F` 안의 paraphrase 쌍 `(p_i, p_j)`에 대해
   - `Delta_m(x, p_i)`와 `Delta_m(x, p_j)`가 얼마나 일관되는지

2. `across_family_similarity[m, F]`

   - `F`의 prompt와 다른 family `G != F`의 prompt를 비교했을 때
   - 동일한 방식으로 반응하는 정도

3. `stability_score[m, F]`

   - `within_family_consistency[m, F] - alpha * across_family_similarity[m, F]`
   - 높은 stability는 다음을 뜻한다.
     1. 같은 family 내부에서는 일관되게 반응한다.
     2. 다른 family에는 덜 일반적으로 반응한다.

4. `causal_score[m, F]`

   - module patching 또는 ablation 시 teacher behavior 복원/붕괴 정도

5. `combined_score[m, F]`
   - 기본은 `stability_score[m, F] * causal_score[m, F]`

선택 모듈 집합 `M_F`를 family별로 얻는다.

#### 4.2 Stability 수식적 정의

입력 x, family F, prompt p에 대해

- `Delta_m(x, p) = h_m(x, p) - h_m(x, null)`

그러면,

- `WithinFamilySim[m, F]`

  - 같은 family F 내부의 서로 다른 paraphrase 쌍 `(p_i, p_j)` 에 대해
  - `cos(Delta_m(x, p_i), Delta_m(x, p_j))` 의 평균

- `AcrossFamilySim[m, F]`
  - `p in F`, `q in G`, `G != F` 인 쌍에 대해
  - `cos(Delta_m(x, p), Delta_m(x, q))` 의 평균

최종적으로,

- `stability_score[m, F] = WithinFamilySim[m, F] - alpha * AcrossFamilySim[m, F]`

해석:

- `WithinFamilySim`이 높으면 같은 행동을 의도한 paraphrase들에 대해 모듈 반응이 안정적이라는 뜻이다.
- `AcrossFamilySim`이 낮을수록 그 모듈은 generic한 반응이 아니라 family-specific한 반응을 가진다.

### Stage C. Selective LoRA placement

`M_F`에만 LoRA를 삽입한다.

초기 granularity:

- transformer block output
- attention output projection block
- MLP block

후속 granularity:

- q/k/v/o projection 단위
- neuron/head subset

### Stage D. Distillation

#### 4.3 Token KL

teacher prompt 조건 분포와 student 무프롬프트 분포를 맞춘다.

#### 4.4 Delta loss

선택 모듈 `m`에 대해

- `d_T[m] = norm(h_T[m] - h_B[m])`
- `d_S[m] = norm(h_S[m] - h_B[m])`
- `delta_loss = sum_m || d_T[m] - d_S[m] ||^2`

초기 norm 후보:

- layernorm-style standardization
- cosine similarity loss
- simple L2 on normalized residuals

#### 4.5 Preserve loss

prompt와 무관한 base behavior 보존용 regularizer

- `KL(base || student)` on unrelated inputs

### 전체 loss

`L = L_kl + lambda_delta * L_delta + lambda_preserve * L_preserve`

## 5. Baselines

상세 비교 원칙과 외부 레포 링크는 `docs/BASELINES.md`를 따른다.

### 구현/출발점 baselines

1. Prompt Baking style KL-only LoRA
2. GenPI / prompt internalization style prompt-free distillation
3. OPCD-style on-policy refinement (후속 stage)

### 실험 baselines

1. Prompted teacher
2. Full target-module LoRA + KL only
3. All-layer LoRA + KL only
4. Random module subset + KL only
5. Magnitude-based top-k selection
6. Gradient-based selection
7. Ours: mechanism-guided + delta + preserve

### 5.1 Baseline 비교 계층

본 연구의 baseline은 두 계층으로 나눈다.

#### Main baselines

우리와 동일한 teacher/student 구조 및 동일 benchmark protocol 안에서 직접 비교 가능한 방법:

1. Prompt Baking style KL-only LoRA
2. Full target-module LoRA + KL only
3. All-layer LoRA + KL only
4. Random module subset + KL only
5. Magnitude-based top-k module selection
6. Gradient-based module selection
7. Ours: mechanism-guided + delta + preserve

#### Extension baselines

문제 설정이 완전히 같지는 않지만, prompt internalization 계열의 참고 비교:

1. GenPI-style internalization
2. OPCD-style refinement

주의:

- Extension baselines는 main table과 별도 표로 보고할 수 있다.
- 문제 설정 차이가 큰 경우, 정성 비교 또는 appendix 비교로 제한한다.

### 5.2 공통 benchmark protocol

가능한 모든 main baseline은 아래 공통 조건에서 비교한다.

- 동일 base model
- 동일 tokenizer
- 동일 teacher prompt family
- 동일 train/valid/test split
- 동일 seen/unseen paraphrase split
- 동일 batch size 또는 유사한 effective token budget
- 동일 평가 metric
- 동일 random seed set
- 동일 result schema

평가 항목:

- teacher fidelity
- preservation
- efficiency
- seen paraphrase transfer
- unseen paraphrase transfer

baseline 간 차이는 알고리즘 자체에만 두고,
데이터/모델/평가 파이프라인의 차이는 최소화한다.

### 5.3 GenPI 비교 원칙

GenPI는 본 연구의 main baseline이라기보다 prompt internalization 계열의 extension baseline으로 취급한다.

이유:

- teacher/student 세팅과 데이터 생성 방식이 동일하지 않을 수 있다.
- long system prompt internalization이나 agent-style prompt setting에 더 가깝다.

따라서 GenPI 비교는 아래 중 하나로 제한한다.

1. 비교 가능한 하위 설정으로 재구현한 `GenPI-lite` baseline
2. 동일 task/data/model에서 맞춘 부분만 appendix 또는 extension table로 보고

GenPI official repo 결과를 그대로 main table의 직접 비교값으로 사용하지 않는다.

## 6. 평가 지표

### Teacher fidelity

- task accuracy
- style agreement score
- token KL / next-token agreement

### Preservation

- base model general benchmark drop
- unrelated prompt behavior drift
- paraphrase transfer
- unseen paraphrase transfer within the same family

### Efficiency

- trainable parameter count
- wall clock training time
- peak GPU memory
- inference latency (merged LoRA vs unmerged optional)

## 7. 구현 milestone

### M0

- base HF + PEFT 학습 loop
- KL-only baking 재현

### M1

- module hooks 및 delta extraction
- localization score 계산
- human-defined prompt family loader 추가

### M2

- selective LoRA placement
- delta loss 추가

### M3

- preservation regularizer
- full ablation

### M4

- on-policy refinement (optional)

## 8. 위험 요소

1. localization noise가 커서 stable module이 잘 안 잡힐 수 있다.
2. delta loss가 KL을 방해할 수 있다.
3. selective placement의 이득이 random top-k와 큰 차이가 안 날 수 있다.
4. prompt family 범위가 너무 넓으면 결론이 흐려진다.
5. lexical similarity가 높은 prompt를 잘못 같은 family로 묶으면 stability score가 왜곡될 수 있다.
6. generic module이 높은 stability를 보이는 것처럼 보일 수 있다.

## 9. 초기 성공 기준

- KL-only all-layer LoRA 대비, 비슷한 fidelity에서 trainable params 감소
- random selection 대비 일관된 향상
- preserve loss 추가 시 base degradation 감소
- prompt paraphrase transfer 확인
- unseen paraphrase에도 family-level transfer 확인
