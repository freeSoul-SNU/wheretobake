# CONFIG_SPEC.md

이 문서는 모든 학습/평가 YAML config의 계약을 정의한다.
실험 값의 source of truth는 코드가 아니라 `configs/*.yaml`이다.

---

## 1. 기본 원칙

- 모든 magic number는 config로 이동한다.
- baseline별 차이는 가능한 한 config로 제어한다.
- config는 사람이 읽기 쉬워야 하며, baseline 비교에 필요한 핵심 값이 명시되어야 한다.
- 실행 시 실제로 사용된 resolved config를 outputs에 저장한다.

---

## 2. 권장 파일 분리

```text
configs/
├── base_smoke.yaml
├── models/
│   └── llama3_8b_instruct.yaml
├── dataset/
│   └── prompt_family_v1.yaml
└── baselines/
    ├── promptbake_kl.yaml
    ├── full_target_lora_kl.yaml
    ├── random_subset_kl.yaml
    └── ours_selective.yaml
```

상속/병합 방식을 쓰더라도 최종 resolved config는 한 파일로 저장 가능해야 한다.

---

## 3. 최상위 필수 섹션

모든 run config는 아래 최상위 키를 가진다.

- `run`
- `model`
- `data`
- `prompting`
- `baseline`
- `lora`
- `train`
- `eval`
- `logging`
- `output`

선택적 섹션:

- `localization`
- `selection`
- `loss`
- `preservation`
- `efficiency`

---

## 4. 섹션별 필수 필드

### 4.1 `run`

```yaml
run:
  run_name: promptbake_kl_smoke
  seed: 42
  device: cuda
  dtype: bfloat16
  mode: train_eval   # train | eval | train_eval | localization_only
```

필수 필드:

- `run_name`
- `seed`
- `mode`

---

### 4.2 `model`

```yaml
model:
  base_model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
  tokenizer_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
  trust_remote_code: false
  use_flash_attention: false
  gradient_checkpointing: true
```

필수 필드:

- `base_model_name_or_path`
- `tokenizer_name_or_path`

---

### 4.3 `data`

```yaml
data:
  train_path: data/train.jsonl
  valid_path: data/valid.jsonl
  test_path: data/test.jsonl
  max_source_length: 1024
  max_target_length: 256
  prompt_family_spec_path: docs/PROMPT_FAMILIES.md
  input_template_name: default
```

필수 필드:

- `train_path`
- `valid_path`
- `test_path`
- `max_source_length`
- `max_target_length`

---

### 4.4 `prompting`

```yaml
prompting:
  teacher_uses_prompt: true
  student_uses_prompt: false
  prompt_field: prompt_text
  input_field: input_text
  target_field: target_text
```

필수 필드:

- `teacher_uses_prompt`
- `student_uses_prompt`
- `prompt_field`
- `input_field`

---

### 4.5 `baseline`

```yaml
baseline:
  name: promptbake_kl
  family_scope: all
  extension: false
```

필수 필드:

- `name`

`name`은 `docs/BASELINES.md`에 정의된 baseline registry 이름과 일치해야 한다.

---

### 4.6 `lora`

```yaml
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
  bias: none
  task_type: CAUSAL_LM
```

필수 필드:

- `enabled`
- `r`
- `alpha`
- `dropout`
- `target_modules`

주의:

- selective baseline의 경우 `target_modules`는 selection output으로 override될 수 있다.
- random / magnitude / gradient / ours는 `selection` 섹션과 함께 해석된다.

---

### 4.7 `train`

```yaml
train:
  num_epochs: 1
  max_steps: 20
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  weight_decay: 0.0
  warmup_ratio: 0.0
  max_grad_norm: 1.0
  teacher_forward_mode: online   # online | cached
```

필수 필드:

- `per_device_train_batch_size`
- `learning_rate`
- `teacher_forward_mode`

---

### 4.8 `loss`

```yaml
loss:
  kl_weight: 1.0
  delta_weight: 0.0
  preserve_weight: 0.0
  temperature: 1.0
  reduction: mean
```

필수 필드:

- `kl_weight`

Ablation 규칙:

- KL only → `delta_weight: 0.0`, `preserve_weight: 0.0`
- KL + delta → `delta_weight > 0`, `preserve_weight: 0.0`
- KL + preserve → `delta_weight: 0.0`, `preserve_weight > 0`
- KL + delta + preserve → 둘 다 `> 0`

---

### 4.9 `localization`

```yaml
localization:
  enabled: false
  module_granularity: block_output   # block_output | attn_output | mlp_output
  alpha: 0.5
  cache_dir: outputs/localization_cache
  topk_modules: 8
```

필수 필드(활성화 시):

- `enabled`
- `module_granularity`
- `alpha`
- `topk_modules`

---

### 4.10 `selection`

```yaml
selection:
  enabled: false
  strategy: none   # none | random | magnitude_topk | gradient_topk | ours_combined
  selected_modules_path: null
  parameter_budget: null
  random_seed: 42
```

필수 필드:

- `enabled`
- `strategy`

---

### 4.11 `eval`

```yaml
eval:
  do_tiny_eval: true
  do_full_eval: false
  metrics:
    - token_kl
    - next_token_agreement
    - style_agreement
  seen_unseen_split: true
  preservation_eval: true
```

필수 필드:

- `metrics`

---

### 4.12 `logging`

```yaml
logging:
  log_every_steps: 1
  save_resolved_config: true
  save_result_json: true
  save_predictions: false
```

필수 필드:

- `save_resolved_config`
- `save_result_json`

---

### 4.13 `output`

```yaml
output:
  output_dir: outputs/promptbake_kl_smoke
  checkpoint_dir: outputs/promptbake_kl_smoke/checkpoints
  result_json_path: outputs/promptbake_kl_smoke/result.json
```

필수 필드:

- `output_dir`
- `result_json_path`

---

## 5. 검증 규칙

모든 config는 아래를 통과해야 한다.

1. YAML load 가능
2. required key 존재
3. baseline name이 registry에 존재
4. result JSON path가 지정됨
5. train/eval 모드와 필수 섹션이 호환됨

---

## 6. M0용 최소 예시

```yaml
run:
  run_name: promptbake_kl_smoke
  seed: 42
  mode: train_eval

model:
  base_model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
  tokenizer_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

baseline:
  name: promptbake_kl

prompting:
  teacher_uses_prompt: true
  student_uses_prompt: false
  prompt_field: prompt_text
  input_field: input_text
  target_field: target_text

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj]

train:
  max_steps: 20
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  teacher_forward_mode: online

loss:
  kl_weight: 1.0
  delta_weight: 0.0
  preserve_weight: 0.0
  temperature: 1.0

output:
  output_dir: outputs/promptbake_kl_smoke
  result_json_path: outputs/promptbake_kl_smoke/result.json
```

