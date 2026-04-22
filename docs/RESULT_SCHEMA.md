# RESULT_SCHEMA.md

이 문서는 baseline 실행 결과를 저장하는 공통 JSON schema를 정의한다.
모든 baseline은 동일한 결과 키를 사용해야 하며, baseline별 차이는 `notes` 또는 하위 metric 필드에만 둔다.

---

## 1. 목적

- baseline 간 직접 비교 가능성 확보
- 실험 병합/집계 자동화 용이성 확보
- 세션이 바뀌어도 결과를 재해석할 수 있게 함

---

## 2. 최상위 필수 필드

모든 result JSON은 아래 필드를 포함해야 한다.

- `run_name`
- `timestamp`
- `git_commit`
- `baseline_name`
- `model_name`
- `seed`
- `prompt_family`
- `paraphrase_split`
- `trainable_params`
- `train_runtime_sec`
- `peak_memory_mb`
- `teacher_fidelity_metrics`
- `preservation_metrics`
- `efficiency_metrics`
- `config_path`
- `resolved_config_path`
- `notes`

---

## 3. 필드 설명

### 3.1 기본 메타데이터

- `run_name`: 사람이 읽을 수 있는 실행 이름
- `timestamp`: ISO-8601 timestamp
- `git_commit`: 실행 시점 commit hash
- `baseline_name`: `docs/BASELINES.md`의 registry 이름
- `model_name`: base model name or path
- `seed`: 정수 시드
- `prompt_family`: family 이름 또는 `all`
- `paraphrase_split`: `seen`, `unseen`, `all`

### 3.2 자원 사용량

- `trainable_params`: adapter 또는 업데이트되는 파라미터 수
- `train_runtime_sec`: 전체 학습 시간(초)
- `peak_memory_mb`: 학습 중 peak GPU memory

### 3.3 teacher fidelity metrics

`teacher_fidelity_metrics`는 dict이며 최소 아래 key 중 가능한 것을 포함한다.

- `token_kl`
- `next_token_agreement`
- `task_accuracy`
- `task_reward`
- `style_agreement`
- `teacher_match_rate`

없거나 해당되지 않는 항목은 `null` 허용.

### 3.4 preservation metrics

`preservation_metrics`는 dict이며 최소 아래 key를 포함한다.

- `base_drift_kl`
- `unrelated_input_drift`
- `non_target_family_drop`
- `general_eval_drop`

초기 M0/M1에서는 일부가 `null`이어도 된다. 단, 키는 유지한다.

### 3.5 efficiency metrics

`efficiency_metrics`는 dict이며 최소 아래 key를 포함한다.

- `trainable_params`
- `estimated_adapter_bytes`
- `train_tokens_per_sec`
- `eval_tokens_per_sec`
- `inference_latency_ms`

`trainable_params`는 최상위 필드와 중복되지만, downstream 집계를 위해 허용한다.

### 3.6 config 경로

- `config_path`: 사용자가 지정한 run config
- `resolved_config_path`: 상속/병합 후 실제 사용된 config snapshot

---

## 4. 권장 선택 필드

- `selected_modules`
- `selection_strategy`
- `selection_budget`
- `loss_weights`
- `dataset_summary`
- `eval_summary_path`
- `prediction_path`
- `error_count`
- `warnings`

---

## 5. 예시 JSON

```json
{
  "run_name": "promptbake_kl_smoke",
  "timestamp": "2026-04-22T15:10:00+09:00",
  "git_commit": "abc1234",
  "baseline_name": "promptbake_kl",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "seed": 42,
  "prompt_family": "concise",
  "paraphrase_split": "seen",
  "trainable_params": 16777216,
  "train_runtime_sec": 48.3,
  "peak_memory_mb": 18234,
  "teacher_fidelity_metrics": {
    "token_kl": 0.91,
    "next_token_agreement": 0.64,
    "task_accuracy": null,
    "task_reward": null,
    "style_agreement": 0.87,
    "teacher_match_rate": 0.59
  },
  "preservation_metrics": {
    "base_drift_kl": 0.08,
    "unrelated_input_drift": 0.11,
    "non_target_family_drop": null,
    "general_eval_drop": null
  },
  "efficiency_metrics": {
    "trainable_params": 16777216,
    "estimated_adapter_bytes": 67108864,
    "train_tokens_per_sec": 532.1,
    "eval_tokens_per_sec": 801.4,
    "inference_latency_ms": 42.6
  },
  "config_path": "configs/baselines/promptbake_kl.yaml",
  "resolved_config_path": "outputs/promptbake_kl_smoke/resolved_config.yaml",
  "selected_modules": null,
  "selection_strategy": null,
  "selection_budget": null,
  "loss_weights": {
    "kl_weight": 1.0,
    "delta_weight": 0.0,
    "preserve_weight": 0.0
  },
  "dataset_summary": {
    "train_examples": 128,
    "valid_examples": 32,
    "test_examples": 32
  },
  "notes": "M0 smoke run"
}
```

---

## 6. 검증 규칙

result JSON 저장 시 아래를 검사한다.

1. 필수 키 존재 여부
2. `baseline_name`이 registry에 존재하는지
3. metric dict가 null이 아닌 dict인지
4. 숫자형 값이 NaN / inf가 아닌지
5. `config_path`와 `resolved_config_path`가 비어 있지 않은지

---

## 7. 집계 규칙

summary script는 아래 키를 기본 그룹 키로 사용한다.

- `baseline_name`
- `model_name`
- `prompt_family`
- `paraphrase_split`
- `seed`

평균/표준편차 집계 시 seed 단위를 기본으로 한다.

