# AGENTS.md

이 저장소는 **Where to Bake: Mechanism-Guided Selective Prompt Baking** 연구용 코드베이스다.
Codex는 이 파일을 "백과사전"처럼 쓰지 말고, **짧은 실행 규칙 + 문서 인덱스**로 사용한다.
세부 설계는 `docs/PROJECT_SPEC.md`, baseline 정의와 외부 레포 참조는 `docs/BASELINES.md`, 장기 작업 계획은 `docs/EXECPLAN.md`를 우선 참고한다.

## 1) 연구 목표

- 목표: 긴 고정 prompt가 유도하는 행동을, prompt 없이도 재현하는 **선택적 LoRA baking** 알고리즘을 구현한다.
- 핵심 아이디어:
  1. **행동적으로 동등한 prompt family** 안에서 안정적이고(cross-paraphrase stable), 다른 family와 구분되며(across-family discriminative), teacher behavior에 인과적으로 중요한(causal) 모듈을 찾는다.
  2. 찾은 모듈에만 LoRA를 배치한다.
  3. teacher-relative internal delta와 output KL을 함께 distill한다.
- 기본 teacher: `base model + system/user prompt`
- 기본 student: `base model + selective LoRA`, **inference 시 prompt 없음**
- 중요한 원칙: **prompt의 표면 형태(surface form) 유사성**이 아니라, **동일한 행동 변화를 유도하는 behavioral equivalence**를 기준으로 모듈을 찾는다.

## 2) Codex의 기본 작업 원칙

- 작은 변경으로 시작하고, 매 단계마다 실행 가능한 상태를 유지한다.
- 임의의 대규모 리팩터링을 하지 않는다.
- 새 파일을 추가할 때는 왜 필요한지 commit message/summary에 설명한다.
- 구현 전에는 항상 아래 순서를 따른다.
  1. 관련 문서 읽기
  2. 영향받는 파일 파악
  3. 구현 계획 3~7줄 작성
  4. 구현
  5. 최소 검증 실행
  6. 결과 및 한계 보고

## 3) 문서 우선순위

1. `docs/PROJECT_SPEC.md` — 알고리즘, loss, 실험 가설, baseline 정의, 평가 프로토콜
2. `docs/BASELINES.md` — baseline 계층(main/extension), 외부 레포/논문 링크, 구현 원칙
3. `docs/EXECPLAN.md` — 현재 우선순위와 milestone
4. `README.md` — 설치/실행법 (없다면 추가 가능)
5. `configs/*.yaml` — 실험 설정의 source of truth

문서와 코드가 충돌하면:

- 실험 설정은 `configs/*.yaml`
- baseline 비교 원칙은 `docs/BASELINES.md`
- 알고리즘 의도는 `docs/PROJECT_SPEC.md`
- 구현 세부는 실제 코드

충돌을 해결하지 못하면, 추정하지 말고 TODO를 남기고 보고한다.

## 4) 구현 범위

초기 구현 범위는 다음만 포함한다.

- decoder-only HF causal LM
- PEFT LoRA 기반 selective adaptation
- teacher-forced token-level KL distillation
- 선택된 block/module에 대한 internal delta distillation
- 평가: task fidelity / base preservation / efficiency
- baseline registry + 공통 eval harness

아직 하지 말 것:

- 멀티모달
- RLHF / DPO 통합
- full mechanistic tracing at neuron level
- production serving 최적화
- encrypted/coded prompt hybrid
- prompt family의 자동 군집화/자동 의미판별

## 5) 권장 디렉터리 구조

- `src/data/` : prompt family, distillation dataset, collator
- `src/models/` : model wrapper, hooks, selective LoRA injection
- `src/localization/` : stability score, causal score, module selection
- `src/train/` : distillation loop, loss, trainer
- `src/eval/` : fidelity / preservation / efficiency eval
- `src/baselines/` : baseline registry, baseline builders, wrapper
- `configs/` : 모델/실험 설정 YAML
- `configs/baselines/` : baseline별 config
- `scripts/` : 재현 가능한 실행 스크립트
- `tests/` : 빠른 unit/smoke tests
- `outputs/` : 로컬 결과물 (git ignore 권장)

## 6) 구현 세부 규칙

- Python 3.11 기준으로 작성한다.
- 가능하면 `transformers`, `peft`, `accelerate`, `datasets`, `torch`를 우선 사용한다.
- 학습 루프는 먼저 단순하게 만들고, 성능 최적화는 나중에 한다.
- 함수는 가능하면 작게 유지한다. 한 함수는 한 책임만 갖게 한다.
- public 함수/클래스에는 docstring을 달고, tensor shape를 적는다.
- 실험에 쓰이는 모든 magic number는 config로 이동한다.
- module 이름은 HF model의 실제 layer path와 대응되게 유지한다.
- prompt family label은 초기 논문 버전에서는 **수동 정의(manual assignment)** 한다.
- baseline 추가 시, baseline 고유 로직은 `src/baselines/` 아래로 격리한다.

## 7) localization 규칙

- magnitude만으로 모듈을 고르지 않는다.
- **단어 중복률, 토큰 overlap, 문장 표면 유사도**를 stability의 주 기준으로 쓰지 않는다.
- prompt family는 처음에 사람이 직접 정의한다.
  - prompt family란, 표현은 달라도 **같은 행동 변화를 의도한 prompt들의 집합**이다.
  - 예: concise / formal / step-by-step / refusal
- 최소한 다음 점수들을 분리 구현한다.
  - `within_family_consistency`: 같은 family 내부 paraphrase들에 대해 module delta가 얼마나 일관되는가
  - `across_family_similarity`: 다른 family prompt들과도 같은 식으로 반응하는가
  - `causal_score`: ablate/patch 시 teacher behavior에 미치는 영향
- `stability_score`는 아래 개념을 따른다.
  - **같은 behavioral family 내부에서는 높고**
  - **다른 family와는 구분될수록 높다**
- 기본 형태:
  - `stability_score = within_family_consistency - alpha * across_family_similarity`
- 최종 선택은 config에서 `stability_only`, `causal_only`, `combined`를 바꿀 수 있게 한다.
- `combined` 기본값은 다음을 우선 고려한다.
  - `select_score = stability_score * causal_score`
- 초기에 neuron/head 단위까지 가지 말고, **block output / attn output / mlp output** 단위부터 구현한다.
- 요약 원칙:
  - **Stability is defined over behavioral equivalence classes of prompts, not over surface-form similarity.**

## 8) distillation 규칙

기본 loss는 아래 3개다.

- `kl_loss`: teacher(prompted) vs student(unprompted)
- `delta_loss`: selected module의 base-relative delta matching
- `preserve_loss`: unrelated/base behavior drift 억제

반드시 ablation 가능하게 구현한다.

- KL only
- KL + delta
- KL + preserve
- KL + delta + preserve

## 9) baseline 통합 및 공통 평가 규칙

- baseline은 가능하면 하나의 공통 runner 인터페이스로 실행한다.
- baseline마다 학습 방식이 달라도, 평가는 동일한 benchmark protocol을 따른다.
- baseline은 **main baselines**와 **extension baselines**로 구분한다.
  - main baselines:
    - Prompt Baking style KL-only LoRA
    - Full/all-layer LoRA
    - Random subset
    - Magnitude-based selection
    - Gradient-based selection
    - Ours
  - extension baselines:
    - GenPI-style internalization
    - OPCD-style refinement
- external baseline을 추가할 때는 다음 중 하나를 선택한다.
  1. 동일 저장소 안에서 비교 가능한 형태로 재구현
  2. 외부 repo wrapper를 만들고, 동일 eval harness로 결과만 수집
- 외부 방법을 우리에게 유리하게 단순화하지 말 것
- 비교 가능성이 낮은 baseline은 main table에 넣지 말고 appendix/extension table로 분리한다.

## 10) 실험 규칙

새 실험을 추가할 때마다 반드시 다음을 기록한다.

- 목적
- 사용 config 경로
- seed
- trainable params 수
- teacher 모델 / student 모델
- prompt family
- 평가 명령

빠른 smoke test 없이 큰 학습 job을 돌리지 않는다.

## 11) 테스트/검증 규칙

최소 검증 순서:

1. import test
2. baseline registry load test
3. single batch forward
4. loss scalar finite 확인
5. 10~20 step smoke train
6. tiny eval run
7. result json schema validation

baseline 추가 시 반드시:

- baseline 이름이 registry에 등록되는지
- config로 실행 가능한지
- 공통 evaluator가 동작하는지
- 결과가 공통 JSON 형식으로 저장되는지
  를 검증한다.

코드 수정 후 가능한 경우 다음을 우선 실행한다.

- unit tests
- smoke training
- config lint / YAML load check

## 12) ExecPlan 사용 규칙

작업이 아래 중 하나면 `docs/EXECPLAN.md`를 갱신한다.

- 1시간 이상 걸릴 구현
- 여러 파일에 걸친 변경
- baseline 추가
- 실험 설계 변경
- 디렉터리 구조 변경

갱신 내용:

- 배경
- 목표
- 단계별 체크리스트
- 완료 기준
- 리스크

## 13) Codex에게 기대하는 출력 형식

작업 완료 후에는 항상 아래를 보고한다.

- 변경 파일 목록
- 핵심 변경점
- 실행한 검증 명령
- 아직 남은 리스크 / TODO
- 다음 권장 작업 1~3개

## 14) 금지 사항

- 검증하지 않은 성능 향상을 사실처럼 쓰지 말 것
- 실험 결과를 config 없이 하드코딩하지 말 것
- baseline 구현을 우리 방법에 유리하게 임의 수정하지 말 것
- 무거운 dependency를 사용자 동의 없이 추가하지 말 것
- 재현 불가능한 notebook-only 구현으로 끝내지 말 것
- lexical similarity가 높은 prompt를 자동으로 같은 family로 간주하지 말 것
- 반대 의미의 prompt를 같은 family에 넣지 말 것
