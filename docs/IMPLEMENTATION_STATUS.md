# IMPLEMENTATION_STATUS.md

이 문서는 현재 저장소가 **어디까지 구현되었고**, **무엇이 아직 미구현인지**를 한눈에 보기 위한 체크리스트다.

상태 표기:

- `[x]` 구현 완료
- `[-]` 부분 구현
- `[ ]` 미구현

---

## Stage M0. Minimal Baseline

- [x] config loader
- [x] HF causal LM loader
- [x] tokenizer loader
- [x] teacher prompted / student unprompted formatting
- [x] token-level KL loss
- [x] teacher no-grad / student grad graph separation
- [x] shifted causal-LM KL alignment on target prediction positions
- [x] smoke trainer
- [x] smoke-train non-zero LoRA grad norm test
- [x] tiny evaluator
- [x] result JSON 저장
- [x] unused delta/preserve loss status reporting
- [x] `promptbake_kl` smoke run

현재 의미:

- 최소 baseline 파이프라인이 실행된다.
- 연구 결과를 주장할 단계는 아니다.

---

## Stage M1. Baseline Harness

- [x] baseline registry
- [x] baseline별 config 분리
- [x] 공통 runner
- [x] heuristic selection scaffolding
- [x] `random_subset_kl` selection logic
- [x] `all_layer_lora_kl` selection logic
- [x] `magnitude_topk` heuristic selection logic
- [x] `gradient_topk` heuristic selection logic
- [x] selection debug artifact 저장
- [x] result summary script
- [-] 여러 baseline의 end-to-end result 축적

현재 의미:

- baseline 구조는 비교 가능한 형태로 정리되어 있다.
- 하지만 summary에 실제로 쌓인 run은 아직 충분하지 않다.

---

## Stage M2. Localization

- [-] localization cache
- [-] module delta dump
- [-] within_family_consistency
- [-] across_family_similarity
- [-] stability_score
- [-] causal_score
- [-] family별 top-k localization output
- [-] normalized selection score

부분 구현:

- [x] prompt-induced module delta 수집
- [x] within-family vs across-family similarity inspection
- [x] response-region 기반 module delta pooling
- [x] stability score preview report
- [x] sequence-level response-region KL causal proxy
- [x] family 내 정규화 기반 `selection_score`
- [ ] selection loop 재주입

현재 의미:

- **연구의 핵심 localization stage는 아직 구현되지 않았다.**
- 따라서 "어디에 LoRA를 붙여야 하는가"를 mechanism-guided 방식으로 최종 결론 낼 수는 없다.
- 다만, 같은 family prompt와 다른 family prompt가 응답 구간 내부 delta 기준으로 얼마나 비슷한지, 그리고 module ablation이 응답 구간 logits를 얼마나 흔드는지 보는 **관찰 도구**는 이제 있다.

---

## Stage M3. Selective Method

- [ ] selective LoRA placement from localization output
- [ ] delta loss
- [ ] preserve loss
- [ ] KL + delta ablation
- [ ] KL + preserve ablation
- [ ] KL + delta + preserve ablation
- [-] `ours_selective` registry/config placeholder

현재 의미:

- 제안 방법의 이름과 자리만 있고, 본체는 아직 아니다.

---

## Stage M4. Reporting

- [x] result JSON schema
- [x] summary JSON/CSV export
- [ ] paper-style main table generation
- [ ] extension table generation
- [ ] markdown report generation

---

## 지금 당장 가능한 것

- `promptbake_kl` smoke run
- long-form dataset 생성
- cross-function dataset 생성
- conda 기반 테스트 환경 재현
- 일부 heuristic baseline에 대해 selection artifact 확인
- outputs 아래 result 요약
- prompt similarity 결과 자동 분석
- summary-style family와 cross-function family를 각각 M2로 비교

## 지금 당장 불가능한 것

- mechanism-guided localization 주장
- stability/causal 근거로 LoRA 위치 결정
- 제안 방법(`ours_selective`)의 실험 결과 해석
