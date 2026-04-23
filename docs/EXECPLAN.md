# EXECPLAN.md

## 목적

이 문서는 Codex가 여러 세션에 걸쳐 **동일한 구현 우선순위, 산출물, 완료 기준**을 유지하도록 하는 실행 계획 문서다.
핵심 원칙은 아래와 같다.

> **작동하는 baseline을 먼저 만들고, 그 위에 선택 모듈 탐색과 제안 방법을 점진적으로 얹는다.**

---

## 활성 작업 로그

### 2026-04-22 M0 bootstrap

- 배경:
  - 현재 저장소는 문서 중심 상태이며, 실행 코드와 smoke baseline이 없다.
- 목표:
  - `promptbake_kl` baseline이 config 기반으로 학습/평가/result 저장까지 한 번에 연결되도록 만든다.
- 단계별 체크리스트:
  - [x] `src/`, `configs/`, `scripts/`, `tests/`, `data/` 최소 구조 생성
  - [x] baseline registry와 config loader 추가
  - [x] synthetic prompt-family dataset 및 prompt family YAML 추가
  - [x] teacher prompted / student unprompted KL distillation 루프 추가
  - [x] tiny eval 및 result JSON writer 추가
  - [x] HF dependency 설치 후 end-to-end smoke run 재검증
- 완료 기준:
  - `promptbake_kl` config 하나로 train/eval/result 저장 경로가 결정된다.
- 리스크:
  - 현재 환경의 Python은 user-site가 기본 비활성화라서, 검증 시에는 user-site 경로를 `PYTHONPATH`에 명시했다. 일반 사용 환경에서는 venv 또는 표준 site-packages 설치를 권장한다.

### 2026-04-22 M1 harness expansion

- 배경:
  - 현재 구현은 `promptbake_kl` 단일 baseline과 tiny synthetic smoke에 머물러 있어 baseline 비교와 폭넓은 검증 요구를 만족하지 못한다.
- 목표:
  - 최소 3개 이상의 main baseline이 같은 runner에서 train/eval/result 저장까지 동작하게 하고, synthetic 검증 범위도 seen/unseen 및 여러 family를 포함하도록 넓힌다.
- 단계별 체크리스트:
  - [ ] baseline별 selection/build 로직 추가
  - [ ] main baseline config 3종 이상 추가
  - [ ] broader synthetic dataset 추가
  - [ ] 다중 baseline smoke/integration test 추가
  - [ ] summary runner 또는 smoke script 추가
- 완료 기준:
  - baseline 이름만 바꿔도 적어도 `promptbake_kl`, `all_layer_lora_kl`, `random_subset_kl`가 실행된다.
  - 더 넓은 synthetic split에서 결과 JSON이 baseline별로 저장된다.
- 리스크:
  - heuristic baseline의 selection 정의는 초기 버전이라 논문용 최종 정의와 다를 수 있으므로 config와 notes에 명확히 남겨야 한다.

### 2026-04-23 long-form experiment tooling

- 배경:
  - 기존 synthetic dataset은 smoke 확인에는 충분했지만, long-form summarization이나 broader baseline 비교를 하기에는 너무 작고 짧다.
- 목표:
  - seed corpus 기반으로 long-form dataset을 재생성할 수 있게 하고, GPU/longform config 및 result summary utility를 추가한다.
- 단계별 체크리스트:
  - [x] long-form seed corpus와 prompt family spec 추가
  - [x] dataset generator 추가
  - [x] long-form experiment config 추가
  - [x] result summary script 추가
  - [x] dataset generation / summary test 추가
- 완료 기준:
  - `scripts/generate_longform_dataset.py`로 long-form JSONL을 재생성할 수 있다.
  - `scripts/summarize_results.py`로 outputs 아래 result.json을 요약할 수 있다.

### 2026-04-23 stage runner scripts

- 배경:
  - milestone별로 어떤 명령을 실행해야 하는지 README와 docs에 흩어져 있어, stage 단위 실행 동선이 불명확했다.
- 목표:
  - M0~M4에 대응하는 shell script를 두고, 현재 구현된 stage는 바로 실행되게 하며 미구현 stage는 명확히 안내한다.
- 단계별 체크리스트:
  - [x] M0 runner script 추가
  - [x] M1 runner script 추가
  - [x] M2/M3 guard script 추가
  - [x] M4 summary runner script 추가
  - [x] stage dispatcher script 추가
- 완료 기준:
  - `bash scripts/run_stage.sh <stage>` 형식으로 stage별 진입점이 제공된다.

### 2026-04-23 M2 similarity inspection

- 배경:
  - 사용자 요구사항상, 같은 family prompt와 다른 family prompt가 실제로 얼마나 비슷한 반응을 만드는지 먼저 확인할 수 있어야 한다.
- 목표:
  - prompt-induced delta를 수집하고 within-family / across-family cosine similarity를 저장하는 최소 M2 도구를 추가한다.
- 단계별 체크리스트:
  - [x] prompt-induced delta 수집 코드 추가
  - [x] within-family similarity 집계 추가
  - [x] across-family similarity 집계 추가
  - [x] preview stability score 저장 추가
  - [ ] causal score 추가
- 완료 기준:
  - 같은 family prompt와 다른 family prompt의 유사도 차이를 JSON/CSV로 저장할 수 있다.

---

## 연구 구현의 큰 흐름

이 프로젝트는 아래 순서로 진행한다.

1. **M0. 실행 가능한 최소 baseline 구축**
   - Prompt Baking style KL-only LoRA baseline이 돌아가야 한다.
2. **M1. baseline 비교 하네스 구축**
   - main baseline들을 같은 코드베이스/같은 evaluator로 돌릴 수 있어야 한다.
3. **M2. localization 파이프라인 구축**
   - prompt family 기준으로 module delta를 수집하고 stability/causal score를 계산할 수 있어야 한다.
4. **M3. 제안 방법 구현**
   - 선택된 module에만 LoRA를 삽입하고 delta/preserve loss를 적용한다.
5. **M4. 통합 평가 및 리포트 자동화**
   - main table과 extension table을 재현 가능하게 생성한다.

**중요:** M0가 안정적으로 완료되기 전에는 M2 이후 작업을 본격적으로 진행하지 않는다.

---

## 현재 최우선 목표

### M0. Prompt Baking style KL-only LoRA baseline 구축

### 완료 기준

- config 기반으로 학습 실행 가능
- teacher prompted / student unprompted 구조 구현 완료
- single-GPU 환경에서 10~20 step smoke train 성공
- tiny eval run 성공
- 결과 JSON 1개 이상 저장 성공

### 산출물

- `src/` 기본 구조
- baseline 1종 실행 config
- smoke test 로그
- result JSON schema 초안 반영
- README와 실행 문서 연결

---

## 마일스톤별 실행 계획

### M0. 최소 baseline 구축

#### 목적

가장 단순한 Prompt Baking style KL-only LoRA baseline을 재현 가능한 형태로 만든다.

#### 작업 항목

- [x] `src/`, `configs/`, `scripts/`, `tests/` 생성
- [x] `src/baselines/`, `src/models/`, `src/data/`, `src/eval/` 생성
- [x] `configs/baselines/`, `configs/models/`, `configs/dataset/` 생성
- [x] 환경 파일(`requirements.txt` 또는 `pyproject.toml`) 작성
- [x] `README.md` 작성
- [x] `docs/CONFIG_SPEC.md` 기준의 YAML loader 작성
- [x] HF model loader 작성
- [x] prompt formatting 유틸 작성
- [x] teacher/student forward wrapper 작성
- [x] token-level KL loss 구현
- [x] small synthetic dataset loader 구현
- [x] 10~20 step smoke train 성공
- [x] tiny eval run 성공
- [x] `docs/RESULT_SCHEMA.md` 기준의 result JSON 저장 구현

#### 종료 조건

- `promptbake_kl` baseline이 config 1개로 실행된다.
- 학습 로그, eval 로그, result JSON이 모두 남는다.
- `tests/test_baseline_registry.py` 또는 동등한 smoke test가 통과한다.

---

### M1. baseline 하네스 구축

#### 목적

직접 비교 가능한 baseline들을 하나의 registry와 공통 evaluator 아래에서 실행할 수 있게 만든다.

#### main baseline 범위

- Prompt Baking style KL-only LoRA
- Full target-module LoRA + KL only
- All-layer LoRA + KL only
- Random subset selection + KL only
- Magnitude-based selection
- Gradient-based selection
- Ours

#### extension baseline 범위

- GenPI-lite
- OPCD-style refinement

#### 작업 항목

- [x] baseline registry 설계
- [x] baseline 공통 config schema 정의
- [x] baseline별 train/eval entrypoint 통일
- [x] 공통 evaluator 연결
- [x] baseline 결과 저장 포맷 통일
- [ ] summary CSV/JSON 병합 유틸 작성
- [ ] main baseline 3종 이상 smoke test 성공

#### 종료 조건

- baseline 이름만 바꿔도 같은 runner에서 최소 3개 baseline이 돌아간다.
- 결과가 동일한 JSON schema로 저장된다.

---

### M2. localization 파이프라인 구축

#### 목적

prompt family 기준으로 module delta를 수집하고, family-specific한 안정 모듈을 찾기 위한 점수를 계산한다.

#### 작업 항목

- [ ] forward hooks로 module output 추출
- [ ] module registry 설계
- [ ] base/teacher/student hidden dump 유틸 작성
- [ ] delta cache 포맷 결정
- [ ] human-defined prompt family loader 구현
- [ ] seen/unseen paraphrase split 로더 구현
- [ ] within-family pair sampler 구현
- [ ] cross-family pair sampler 구현
- [ ] `within_family_consistency` 구현
- [ ] `across_family_similarity` 구현
- [ ] `stability_score = within - alpha * across` 구현
- [ ] causal score의 최소 버전 구현
- [ ] family별 top-k module logging 구현

#### 종료 조건

- family별 top-k module 리스트를 파일로 저장할 수 있다.
- seen/unseen paraphrase split 기준으로 stability score가 계산된다.
- selection 결과가 config input으로 다시 주입될 수 있다.

---

### M3. 제안 방법 구현

#### 목적

선택된 module에만 LoRA를 삽입하고, KL + delta + preserve loss를 함께 최적화한다.

#### 작업 항목

- [ ] selected module list를 config에서 읽게 함
- [ ] selected module에만 LoRA 삽입
- [ ] delta loss 구현
- [ ] preserve loss 구현
- [ ] warmup / loss weight config 추가
- [ ] ablation configs 작성
  - [ ] KL only
  - [ ] KL + delta
  - [ ] KL + preserve
  - [ ] KL + delta + preserve

#### 종료 조건

- selective LoRA baseline이 end-to-end로 실행된다.
- ablation 2종 이상이 동일 evaluator로 비교 가능하다.

---

### M4. 통합 평가 및 리포트 자동화

#### 목적

main baseline과 extension baseline을 분리해 공통 포맷으로 평가하고 표를 자동 생성한다.

#### 작업 항목

- [ ] fidelity evaluator 완성
- [ ] preservation evaluator 완성
- [ ] memory/time logging 추가
- [ ] seen vs unseen paraphrase transfer 분리 평가
- [ ] main baseline comparison table 자동 생성
- [ ] extension baseline comparison table 자동 생성
- [ ] result summary markdown/CSV 생성 스크립트 작성

#### 종료 조건

- main table과 extension table이 자동으로 생성된다.
- 실험 결과를 paper 초안 표로 바로 옮길 수 있다.

---

## 이번 주 권장 작업

### 우선순위 A

1. M0 baseline repo 세팅
2. KL-only LoRA smoke run
3. result JSON schema 확정
4. config schema / evaluator entrypoint 고정

### 우선순위 B

5. baseline registry 초안 작성
6. prompt family schema 초안 작성
7. experiment log / decision log 파일 생성

### 이번 주에 하지 않을 것

- on-policy distillation
- encryption/coded prompt hybrid
- compositional baking
- full safety study
- automatic paraphrase clustering
- full GenPI reproduction

---

## 실험 기록 규칙

각 실행은 아래 항목을 반드시 남긴다.

- config path
- git commit hash
- model name
- baseline name
- seed
- prompt family
- seen / unseen split 여부
- trainable params
- runtime / peak memory
- 주요 metric
- result JSON path

기록 형식은 `docs/EXPERIMENT_LOG_TEMPLATE.md`를 따른다.

---

## 리스크와 대응

### 리스크 1. teacher forward 비용이 큼

- 대응: first pass에서는 online teacher forward로 시작하고, 필요 시 cache 추가

### 리스크 2. 모듈 naming이 모델별로 다름

- 대응: model별 module registry adapter를 분리

### 리스크 3. delta loss가 불안정함

- 대응: loss weight warmup, normalized residual 사용, block output부터 시작

### 리스크 4. family label이 잘못 정의되면 stability score가 무의미해짐

- 대응: 초기에는 manual family만 사용하고, quality control 규칙을 `docs/PROMPT_FAMILIES.md`에 별도 유지

### 리스크 5. generic module이 높은 stability를 보일 수 있음

- 대응: across-family similarity penalty와 causal score를 함께 사용

### 리스크 6. 외부 baseline과의 불공정 비교 가능성

- 대응: `docs/BASELINES.md` 기준으로 main baseline과 extension baseline을 분리하고, main table에는 직접 비교 가능한 방법만 포함

---

## 보류 항목

아래 항목은 현재 버전의 범위 밖이다.

- on-policy distillation
- GenPI full reproduction
- encryption/coded prompt hybrid
- compositional baking
- full safety attack/defense study
- automatic paraphrase clustering
