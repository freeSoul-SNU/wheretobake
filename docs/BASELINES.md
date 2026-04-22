# BASELINES.md

이 문서는 baseline을 **main baseline**과 **extension baseline**으로 나누고,
각 baseline의 목적, 구현 방식, 비교 시 주의점, 외부 논문/코드 출처를 명확히 한다.

문서의 목적은 두 가지다.

1. Codex가 baseline을 임의로 단순화하거나 우리에게 유리하게 변형하지 않도록 한다.
2. 논문 작성 시 어떤 baseline을 main table에 넣고, 어떤 baseline을 appendix/extension table로 분리할지 명확히 한다.

---

## 1. baseline 계층

### 1.1 Main baselines

다음 baseline은 **동일 teacher/student 구조**, **동일 prompt family**, **동일 evaluator**, **동일 result schema** 안에서 직접 비교 가능해야 한다.

1. Prompt Baking style KL-only LoRA
2. Full target-module LoRA + KL only
3. All-layer LoRA + KL only
4. Random module subset + KL only
5. Magnitude-based top-k selection
6. Gradient-based selection
7. Ours: mechanism-guided + delta + preserve

### 1.2 Extension baselines

다음 baseline은 문제 설정이 완전히 같지 않을 수 있으므로, main table이 아니라 appendix 또는 extension table로 분리한다.

1. GenPI-style internalization
2. OPCD-style refinement

### 1.3 운영 원칙

- main baseline은 **같은 runner와 같은 metric schema**로 직접 비교한다.
- extension baseline은
  - 비교 가능한 하위 설정으로 재구현하거나,
  - 외부 repo wrapper로 동일 evaluator에만 연결한다.
- 세팅 차이가 큰 경우, 정성 비교 또는 appendix 표로 제한한다.

---

## 2. 공통 구현 원칙

### 2.1 절대로 하지 말 것

- baseline을 우리 방법에 유리하게 임의 단순화하지 말 것
- 외부 논문의 보고 수치를 그대로 main table의 직접 비교값으로 쓰지 말 것
- teacher prompt, split, metric을 baseline마다 다르게 두지 말 것
- baseline마다 다른 post-processing을 몰래 넣지 말 것
- extension baseline을 main baseline인 것처럼 보고하지 말 것

### 2.2 공통 조건

가능한 모든 main baseline은 아래 조건을 공유한다.

- 동일 base model
- 동일 tokenizer
- 동일 prompt family 정의
- 동일 train/valid/test split
- 동일 seen/unseen paraphrase split
- 동일 evaluator
- 동일 result JSON schema
- 동일 random seed set

### 2.3 공통 result schema

모든 baseline 결과는 최소한 아래 필드를 포함한다.

- `baseline_name`
- `model_name`
- `seed`
- `prompt_family`
- `trainable_params`
- `train_runtime_sec`
- `peak_memory_mb`
- `teacher_fidelity_metrics`
- `preservation_metrics`
- `efficiency_metrics`
- `notes`

상세 포맷은 `docs/RESULT_SCHEMA.md`를 따른다.

---

## 3. baseline별 정의

## 3.1 Prompt Baking style KL-only LoRA

### 역할

가장 기본이 되는 direct baseline.

### 목적

- prompt-conditioned teacher와 prompt-free student를 KL distillation으로 맞추는 가장 단순한 baking baseline을 제공한다.
- 우리 방법이 이 baseline보다 실제로 나은지 검증한다.

### 구현 원칙

- student는 prompt 없이 입력만 받는다.
- teacher는 prompt + 입력을 받는다.
- loss는 token-level KL만 사용한다.
- module selection, delta loss, preserve loss는 넣지 않는다.

### 구현 방식

- **internal reimplementation**
- baseline name: `promptbake_kl`

### 참고 출처

- paper: Prompt Baking
- code: `AlexanderDetkov/Prompt-Baking`

### 구현 메모

- 공식 저장소는 아이디어 검증용 구조일 수 있으므로, 본 프로젝트에서는 HF/PEFT 기반 내부 재구현을 기본으로 한다.

---

## 3.2 Full target-module LoRA + KL only

### 역할

우리가 선택적으로 LoRA를 넣지 않았을 때의 직접 비교 baseline.

### 목적

- 같은 target module family 전체에 LoRA를 넣었을 때와 우리 selective placement를 비교한다.

### 구현 원칙

- target module 집합은 config에서 고정한다.
- selection은 하지 않는다.
- loss는 KL only.

### 구현 방식

- **internal baseline**
- baseline name: `full_target_lora_kl`

---

## 3.3 All-layer LoRA + KL only

### 역할

더 넓은 적응 범위를 주는 upper-style baseline.

### 목적

- 더 많은 trainable params를 허용했을 때의 Pareto 비교 기준을 제공한다.

### 구현 원칙

- 가능한 모든 지원 layer에 LoRA를 넣는다.
- loss는 KL only.
- delta/preserve는 사용하지 않는다.

### 코드 출처

- LoRA 공식 구현 참고 가능
- 실제 프로젝트 구현은 HF PEFT 기반으로 통일한다.

### 구현 방식

- **internal baseline**
- baseline name: `all_layer_lora_kl`

---

## 3.4 Random module subset + KL only

### 역할

selection 자체의 효과를 보기 위한 최소 대조군.

### 목적

- 우리 selection이 단순한 sparsity 효과가 아니라는 점을 보이기 위한 baseline.

### 구현 원칙

- ours와 동일한 trainable parameter budget 또는 동일한 module 수를 맞춘다.
- module은 random seed로 선택한다.
- loss는 KL only.

### 구현 방식

- **internal baseline**
- baseline name: `random_subset_kl`

---

## 3.5 Magnitude-based top-k selection

### 역할

mechanistic selection이 아니라 단순 변화량 기반 selection과 비교하기 위한 baseline.

### 목적

- family-aware stability/causal selection이 단순 magnitude heuristic보다 나은지 확인한다.

### 구현 원칙

- module별 delta magnitude 또는 activation shift magnitude를 집계한다.
- 상위 k개 module만 선택한다.
- 동일 parameter budget을 맞춘다.
- 초기 버전은 KL only, 이후 ablation으로 `+delta` 가능.

### 구현 방식

- **internal heuristic baseline**
- baseline name: `magnitude_topk`

---

## 3.6 Gradient-based selection

### 역할

gradient saliency 기반 module selection 비교 baseline.

### 목적

- stability/causal selection이 단순 gradient importance보다 나은지 검증한다.

### 구현 원칙

- teacher-student KL에 대한 module별 gradient norm 또는 Fisher-style proxy를 사용한다.
- 상위 k개 module을 선택한다.
- ours와 동일 parameter budget을 맞춘다.

### 구현 방식

- **internal heuristic baseline**
- baseline name: `gradient_topk`

### 주의

- gradient 정의 방식이 여러 개 가능하므로, 초기 논문에서는 가장 단순한 `gradient norm top-k`를 기본값으로 한다.
- 세부 정의는 config와 코드에 명확히 남긴다.

---

## 3.7 Ours: mechanism-guided + delta + preserve

### 역할

주 proposed method.

### 구현 원칙

- human-defined prompt family 사용
- `stability = within_family_consistency - alpha * across_family_similarity`
- causal score 계산
- selected modules에만 LoRA 삽입
- KL + delta + preserve 조합 사용

### 구현 방식

- baseline name: `ours_selective`

---

## 3.8 GenPI-style internalization

### 역할

prompt internalization 계열의 참고 비교 baseline.

### 목적

- prompt를 내부화하는 다른 계열의 방법과 qualitative/extension 비교를 제공한다.

### 구현 원칙

- main baseline처럼 직접 비교하지 않는다.
- 가능하면 `GenPI-lite` 형태로 비교 가능한 하위 설정을 만든다.
- prompt generation / reason generation / pseudo conversation 등 GenPI 고유 요소는 명시적으로 켜고 끈다.

### 구현 방식

- **extension baseline**
- baseline name: `genpi_lite`
- 가능하면 wrapper 또는 최소 재구현

### 주의

- GenPI는 long prompt internalization, pseudo conversation, reason generation 등 setting 차이가 있을 수 있다.
- 따라서 같은 표에 넣을 때는 무엇을 맞췄고 무엇을 생략했는지 반드시 표기한다.

---

## 3.9 OPCD-style refinement

### 역할

후속 고도화 baseline 또는 2-stage training baseline.

### 목적

- teacher-forced KL-only baseline 이후, student trajectory mismatch를 줄이는 refinement 단계로 사용할 수 있다.

### 구현 원칙

- 초기 논문에서는 optional stage로 둔다.
- main baseline이 아니라 extension baseline 또는 후속 실험으로 둔다.
- reverse-KL / on-policy sampling 도입 여부를 명시한다.

### 구현 방식

- **extension baseline**
- baseline name: `opcd_refine`

---

## 4. 외부 논문 / 코드 링크

### Prompt Baking

- paper: `https://arxiv.org/abs/2409.13697`
- code: `https://github.com/AlexanderDetkov/Prompt-Baking`

### GenPI

- paper: `https://arxiv.org/abs/2411.15927`
- code: `https://github.com/kaistAI/GenPI`

### OPCD

- paper: `https://arxiv.org/abs/2602.12275`
- official code: 공개 저장소가 불명확하면 논문만 우선 참조하고, 코드 공개 여부를 별도 기록한다.

### LoRA

- paper/info: `https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/`
- code: `https://github.com/microsoft/LoRA`

---

## 5. Baseline 추가 시 필수 작업

1. `docs/BASELINES.md`에서 main/extension 여부를 먼저 확인한다.
2. `src/baselines/<name>.py`에 builder/wrapper를 추가한다.
3. `configs/baselines/<name>.yaml`을 만든다.
4. `tests/test_baseline_registry.py`에 registry load test를 추가한다.
5. smoke train + tiny eval을 돌린다.
6. 공통 result JSON이 생성되는지 검증한다.
7. 차별점과 한계를 `docs/DECISIONS.md` 또는 실험 기록에 남긴다.

---

## 6. 금지 사항

- baseline 이름만 같고 실제 구현이 다른 상태로 보고하지 말 것
- 외부 repo 코드를 일부 바꿔 놓고 동일 비교라고 주장하지 말 것
- extension baseline의 공식 수치를 main table의 직접 비교값으로 사용하지 말 것
- prompt family / split / metric 차이를 숨기지 말 것
