# Where to Bake: Mechanism-Guided Selective Prompt Baking

이 저장소는 **긴 고정 prompt가 유도하는 행동을, inference 시 prompt 없이 재현**하는 연구용 코드베이스를 위한 문서 패키지다.
핵심 아이디어는 **prompt family별로 안정적이고 인과적으로 중요한 모듈만 선택하여 LoRA를 삽입**하고,
teacher-relative behavior를 KL + internal delta + preserve loss로 distill하는 것이다.

이 문서는 사람이 보는 실행 엔트리이며, Codex/에이전트용 상세 규칙은 `AGENTS.md`를 따른다.
세부 연구 정의와 구현 계약은 `docs/` 아래 문서를 source of truth로 사용한다.

---

## 문서 우선순위

1. `docs/PROJECT_SPEC.md` — 연구 가설, 알고리즘, loss, 데이터 계약
2. `docs/BASELINES.md` — main/extension baseline 정의와 비교 원칙
3. `docs/EXECPLAN.md` — 현재 milestone, 체크리스트, 완료 기준
4. `docs/CONFIG_SPEC.md` — YAML config 계약
5. `docs/RESULT_SCHEMA.md` — 결과 JSON 저장 스키마
6. `docs/EVAL_PROTOCOL.md` — fidelity / preservation / efficiency 평가 규약
7. `docs/PROMPT_FAMILIES.md` — prompt family 정의 방식 및 QC
8. `docs/DECISIONS.md` — 설계 결정 기록
9. `docs/EXPERIMENT_LOG_TEMPLATE.md` — 실험 기록 템플릿
10. `docs/REPO_PLAN.md` — 추천 디렉터리 구조와 운영 가이드

문서와 코드가 충돌할 때 우선순위는 아래와 같다.

- 실험 값: `configs/*.yaml`
- baseline 비교 규칙: `docs/BASELINES.md`
- 알고리즘 의도: `docs/PROJECT_SPEC.md`
- 저장 포맷: `docs/RESULT_SCHEMA.md`
- 나머지 구현 세부: 실제 코드

---

## 최소 디렉터리 구조

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt / pyproject.toml
├── configs/
│   ├── base_smoke.yaml
│   ├── dataset/
│   ├── models/
│   └── baselines/
├── docs/
├── scripts/
├── src/
│   ├── baselines/
│   ├── data/
│   ├── eval/
│   ├── localization/
│   ├── models/
│   ├── train/
│   └── utils/
├── tests/
└── outputs/
```

---

## 구현 범위

### 초기 범위에 포함

- Hugging Face decoder-only causal LM
- PEFT LoRA 기반 selective adaptation
- teacher prompted / student unprompted distillation
- prompt family 기반 module localization
- KL / delta / preserve loss 조합
- 공통 baseline registry와 evaluator
- config 기반 실행, smoke train, tiny eval, result JSON 저장

### 초기 범위에서 제외

- multimodal 설정
- RLHF / DPO / preference optimization
- neuron/head-level full tracing
- production serving 최적화
- encrypted/coded prompt hybrid
- automatic prompt family clustering
- full GenPI reproduction / on-policy refinement full pipeline

---

## 빠른 시작 권장 순서

1. `docs/EXECPLAN.md`에서 현재 milestone 확인
2. `docs/CONFIG_SPEC.md`에서 필수 config key 확인
3. `docs/BASELINES.md`에서 baseline 계층과 비교 금지 사항 확인
4. `docs/PROJECT_SPEC.md`로 알고리즘 의도 확인
5. baseline 1종에 대해 smoke train + tiny eval 실행
6. `docs/RESULT_SCHEMA.md`에 맞는 결과 JSON 저장 여부 확인
7. `docs/EXPERIMENT_LOG_TEMPLATE.md`에 실험 기록 남기기

### 최소 실행 예시

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/promptbake_kl.yaml
```

`transformers`의 최신 보안 정책 때문에, 이 저장소의 기본 smoke config는 모델을 `safetensors` 형식으로만 로드하도록 설정되어 있다.
만약 특정 모델이 `safetensors`를 제공하지 않으면, `torch>=2.6`으로 올리거나 `safetensors`가 있는 모델로 바꾸는 편이 안전하다.

### Long-Form Dataset 생성

```bash
PYTHONPATH=src python3 scripts/generate_longform_dataset.py
```

생성 결과는 `data/datasets/longform_v1/`에 저장된다.
이 데이터는 기존 smoke용 단문 dataset보다 긴 입력과 family별 요약 target을 포함하므로 더 넓은 실험용 예시로 사용할 수 있다.

### Long-Form Baseline 실행

```bash
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/promptbake_kl_longform.yaml
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/random_subset_kl_longform.yaml
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/all_layer_lora_kl_longform.yaml
```

GPU가 있으면:

```bash
PYTHONPATH=src python3 -m where_to_bake.run --config configs/baselines/promptbake_kl_longform_gpu.yaml
```

### 결과 요약

```bash
PYTHONPATH=src python3 scripts/summarize_results.py --root-dir outputs --output-prefix outputs/summary/results_summary
```

추가 데이터 설계 원칙은 `docs/DATASET_GUIDE.md`를 따른다.
현재 구현 범위와 미구현 범위는 `docs/IMPLEMENTATION_STATUS.md`를,
코드 동작 흐름은 `docs/CODE_WALKTHROUGH.md`를 보면 된다.

### Prompt Similarity 확인

같은 family prompt와 다른 family prompt가 model 내부 delta 기준으로 얼마나 비슷한지 보려면:

```bash
PYTHONPATH=src python3 scripts/generate_longform_dataset.py
PYTHONPATH=src python3 scripts/run_prompt_similarity.py --config configs/baselines/prompt_similarity_longform.yaml
```

출력:

- `outputs/localization/prompt_similarity_longform.json`
- `outputs/localization/prompt_similarity_longform.csv`

### Stage 실행 스크립트

각 milestone별 대표 실행 스크립트:

```bash
bash scripts/run_stage_m0.sh
bash scripts/run_stage_m1.sh
bash scripts/run_stage_m4.sh
```

단일 dispatcher:

```bash
bash scripts/run_stage.sh m0
bash scripts/run_stage.sh m1
bash scripts/run_stage.sh m2
bash scripts/run_stage.sh m3
bash scripts/run_stage.sh m4
```

주의:

- `m0`: 최소 smoke baseline 실행
- `m1`: long-form dataset 생성, main baseline 일부 비교 실행, summary 생성
- `m2`, `m3`: 아직 미구현 stage라 안내 메시지와 함께 종료
- `m4`: 현재 가능한 result summary 자동화만 수행

---

## 첫 구현 대상

초기 구현은 반드시 아래 baseline부터 시작한다.

- baseline name: `promptbake_kl`
- 구조: teacher(prompted) vs student(unprompted)
- loss: token-level KL only
- adapter: LoRA
- 검증: single-GPU 10~20 step smoke train + tiny eval + result JSON 저장

이 baseline이 안정적으로 동작하기 전에는 localization, selective placement, delta/preserve 확장을 본격적으로 진행하지 않는다.

---

## 산출물 규칙

모든 실행은 최소한 아래를 남겨야 한다.

- 실행 config path
- git commit hash
- baseline name
- model name
- seed
- prompt family / seen-unseen split
- trainable params
- runtime / peak memory
- 주요 metric
- result JSON path

자세한 형식은 `docs/RESULT_SCHEMA.md`와 `docs/EXPERIMENT_LOG_TEMPLATE.md`를 따른다.

---

## 권장 운영 방식

- 큰 작업은 먼저 `docs/EXECPLAN.md` 체크리스트를 갱신한다.
- baseline 추가 전 반드시 `docs/BASELINES.md`에서 main/extension 여부를 확인한다.
- prompt family를 추가할 때는 `docs/PROMPT_FAMILIES.md`의 QC 규칙을 먼저 통과시킨다.
- 실험 설정은 코드에 하드코딩하지 말고 config에 둔다.
- 실험 결과 해석은 `docs/DECISIONS.md`에 축적한다.
