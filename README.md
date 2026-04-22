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
