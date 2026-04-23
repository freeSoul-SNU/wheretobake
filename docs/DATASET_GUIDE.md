# DATASET_GUIDE.md

이 문서는 prompt baking 실험용 dataset을 어떻게 구성할지에 대한 실무 가이드를 정리한다.

핵심 원칙은 간단하다.

> **Prompt family dataset은 "짧은 지시문 몇 개"가 아니라, 같은 입력에 대해 family별로 다른 행동을 일관되게 유도하는 데이터여야 한다.**

---

## 1. 지금 toy dataset이 부족한 이유

`data/datasets/train.jsonl`의 기존 smoke dataset은 아래 목적만 충족한다.

- config/loader/trainer가 연결되는지 확인
- tiny model에서 10~20 step smoke train이 되는지 확인
- result JSON이 저장되는지 확인

하지만 이 데이터는 아래 실험에는 부족하다.

- long-context summarization
- seen / unseen paraphrase generalization
- family-specific behavior transfer
- baseline 간 비교

즉, smoke dataset은 **파이프라인 테스트용**이지 **연구 실험용**이 아니다.

---

## 2. 더 좋은 dataset 구조

실험용 dataset은 아래 3단 구조로 만드는 것이 좋다.

1. `source corpus`
   - 중립적인 긴 입력 문서
   - 예: 보고서, 회의록, 기사형 단락, 정책 메모
2. `prompt family`
   - 같은 행동 변화를 유도하는 prompt paraphrase 묶음
   - 예: `concise`, `formal`, `step_by_step`
3. `family-specific target`
   - 같은 source에 대해 family별 teacher output 스타일을 반영한 target

즉, 한 source document에 대해 다음이 함께 있어야 한다.

- concise summary
- formal summary
- step-by-step summary

이 구조가 있어야 teacher prompted / student unprompted distillation이 family 단위로 의미를 가진다.

---

## 3. 추천 생성 절차

1. 먼저 긴 입력 문서를 모은다.
   - 문단 길이 기준으로 최소 80~200단어 정도 권장
   - 너무 짧은 factoid QA는 지양
2. source마다 family별 target을 만든다.
   - `concise`: 1~2문장 핵심 요약
   - `formal`: 같은 내용이지만 공적/보고서 톤
   - `step_by_step`: 번호가 있는 단계/순서/구조화 요약
3. prompt family paraphrase를 seen / unseen으로 나눈다.
4. source split과 prompt split을 분리 관리한다.
   - source split: train / valid / test
   - prompt split: seen / unseen
5. preservation set은 별도로 둔다.
   - target family와 직접 관련 없는 중립 입력

---

## 4. 이 저장소에서 제공하는 생성 파이프라인

long-form synthetic 실험용 예시는 아래 파일들로 관리한다.

- seed corpus: `data/source_corpus/longform_seed_v1.yaml`
- prompt families: `data/prompt_families/prompt_family_longform_v1.yaml`
- generator: `scripts/generate_longform_dataset.py`
- generated output: `data/datasets/longform_v1/`

생성 명령:

```bash
PYTHONPATH=src python3 scripts/generate_longform_dataset.py
```

출력 파일:

- `data/datasets/longform_v1/train.jsonl`
- `data/datasets/longform_v1/valid.jsonl`
- `data/datasets/longform_v1/test.jsonl`
- `data/datasets/longform_v1/preserve.jsonl`

---

## 5. 실험용 dataset을 더 키울 때의 권장 기준

- family당 prompt paraphrase:
  - seen 10개 이상
  - unseen 5개 이상
- source 문서 수:
  - train 100개 이상부터 시작 권장
  - valid/test는 family별 비교가 가능하도록 충분히 확보
- 입력 길이:
  - 실제 목표가 summarization이면 짧은 한두 문장보다 문단/문서 수준 권장
- target 품질:
  - family별 출력 스타일 차이가 분명해야 함
  - lexical overlap이 아니라 behavior 차이가 있어야 함

---

## 6. 금지해야 할 생성 방식

- 문장 하나를 조금만 바꿔서 데이터 양만 늘리는 방식
- prompt family와 무관하게 target을 모두 같은 형태로 두는 방식
- unseen paraphrase를 train에 섞는 방식
- lexical similarity만으로 family를 자동 묶는 방식

