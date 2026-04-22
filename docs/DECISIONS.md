# DECISIONS.md

이 문서는 설계 결정과 그 근거를 누적 기록한다.
ADR(Architecture Decision Record)처럼 간단히 유지한다.

---

## D-001. 초기 구현은 HF + PEFT 기반 내부 재구현으로 통일

- 상태: accepted
- 이유:
  - Prompt Baking, LoRA, 여러 baseline을 같은 runner에서 비교하기 쉽다.
  - 외부 repo 의존도를 줄이고 result schema를 통일할 수 있다.
- 영향:
  - baseline 구현 비용은 다소 늘어나지만 비교 공정성은 좋아진다.

---

## D-002. Prompt family는 manual assignment를 기본으로 한다

- 상태: accepted
- 이유:
  - 초기 논문에서는 behavioral equivalence를 사람이 보증하는 편이 더 신뢰 가능하다.
  - automatic clustering은 범위를 크게 늘린다.
- 영향:
  - family QC 문서와 기록이 필요하다.

---

## D-003. Extension baseline은 main table과 분리한다

- 상태: accepted
- 이유:
  - GenPI, OPCD는 문제 설정과 훈련 절차 차이가 크다.
  - 직접 비교 가능성이 낮은 baseline을 main table에 넣으면 해석이 흐려진다.
- 영향:
  - appendix / extension table 구조가 필요하다.

---

## D-004. Localization granularity는 block / attn output / mlp output부터 시작한다

- 상태: accepted
- 이유:
  - neuron/head 단위는 구현과 해석 비용이 크다.
  - 초기 단계에서는 coarse module selection만으로도 가설 검증이 가능하다.
- 영향:
  - module registry adapter가 필요하다.

---

## D-005. Result JSON schema를 먼저 고정한다

- 상태: accepted
- 이유:
  - baseline을 늘릴수록 결과 저장 형식이 빨리 흔들린다.
  - 집계 자동화와 smoke test 품질을 위해 M0에서 먼저 고정하는 편이 낫다.
- 영향:
  - `docs/RESULT_SCHEMA.md`와 validator가 필요하다.

---

## D-006. M0 smoke baseline은 tiny GPT-2와 synthetic prompt-family data로 시작한다

- 상태: accepted
- 이유:
  - 저장소에 실행 코드가 없는 상태에서 가장 먼저 검증해야 할 것은 runner, config, result schema, teacher/student wiring이다.
  - 작은 공개 모델과 synthetic data면 single-device smoke path를 빠르게 확인할 수 있다.
- 영향:
  - 초기 metric은 연구 결과라기보다 파이프라인 검증용이며, 실제 실험 전 더 큰 모델과 실데이터로 교체해야 한다.

---

## 템플릿

```text
## D-XXX. 제목
- 상태: proposed | accepted | deprecated
- 배경:
- 결정:
- 이유:
- 대안:
- 영향:
- 후속 작업:
```
