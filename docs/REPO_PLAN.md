# REPO_PLAN.md

## 추천 파일 트리

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── base_smoke.yaml
│   ├── models/
│   ├── dataset/
│   └── baselines/
│       ├── promptbake_kl.yaml
│       ├── full_target_lora_kl.yaml
│       ├── random_subset_kl.yaml
│       └── ours_selective.yaml
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── BASELINES.md
│   ├── EXECPLAN.md
│   ├── REPO_PLAN.md
│   ├── CONFIG_SPEC.md
│   ├── RESULT_SCHEMA.md
│   ├── EVAL_PROTOCOL.md
│   ├── PROMPT_FAMILIES.md
│   ├── EXPERIMENT_LOG_TEMPLATE.md
│   └── DECISIONS.md
├── scripts/
│   ├── run_smoke.sh
│   ├── run_baseline.sh
│   └── run_eval.sh
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

## 꼭 필요한 파일

- `AGENTS.md`: Codex용 짧은 실행 규칙
- `README.md`: 사람이 보는 설치/실행법
- `docs/PROJECT_SPEC.md`: 알고리즘 정의
- `docs/BASELINES.md`: baseline 정의와 비교 원칙
- `docs/EXECPLAN.md`: multi-step 작업 계획
- `docs/CONFIG_SPEC.md`: config source of truth
- `docs/RESULT_SCHEMA.md`: 저장 포맷 source of truth
- `configs/*.yaml`: 실험 설정 source of truth

---

## 있으면 좋은 파일

- `docs/EXPERIMENT_LOG_TEMPLATE.md`: 결과 기록 형식 통일
- `docs/DECISIONS.md`: 중간 설계 선택 근거
- `docs/PROMPT_FAMILIES.md`: family assignment / QC 규칙
- `.codex/agents/*.toml`: explorer / implementer 같은 custom subagent 분리

---

## 운영 원칙

### 1. 문서와 코드의 역할 분리

- 문서는 계약(contract)을 정의한다.
- 코드는 그 계약을 구현한다.
- config는 실제 실험 값을 정의한다.

### 2. source of truth

- 알고리즘 의도: `docs/PROJECT_SPEC.md`
- baseline 비교 규칙: `docs/BASELINES.md`
- 현재 우선순위: `docs/EXECPLAN.md`
- 실행 값: `configs/*.yaml`
- 결과 저장 형식: `docs/RESULT_SCHEMA.md`

### 3. 구조 변경 규칙

아래 변경은 반드시 `docs/EXECPLAN.md`와 `docs/DECISIONS.md`에 기록한다.

- 디렉터리 구조 변경
- baseline 추가
- config schema 변경
- evaluator 저장 형식 변경
- prompt family schema 변경

---

## Codex 활용 팁

- 탐색용 세션과 구현용 세션을 분리한다.
- 큰 작업은 항상 `EXECPLAN.md`를 먼저 갱신하게 시킨다.
- baseline 추가 전 `BASELINES.md`를 먼저 읽게 한다.
- config 변경 전 `CONFIG_SPEC.md`를 먼저 읽게 한다.

### 요청 예시

- "AGENTS.md와 PROJECT_SPEC.md를 읽고, M0 구현 계획을 5줄로 적은 뒤 시작해라."
- "이번 작업은 multi-file 변경이다. EXECPLAN.md 체크리스트를 갱신하고 smoke test까지 완료해라."
- "random subset baseline과 ours config를 추가하고, 공통 중복 코드는 리팩터링해라."
