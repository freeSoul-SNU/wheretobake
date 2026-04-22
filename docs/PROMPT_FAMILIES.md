# PROMPT_FAMILIES.md

이 문서는 prompt family 정의 방식, 데이터 스키마, quality control 규칙을 명시한다.
핵심 원칙은 아래 한 문장이다.

> **Prompt family는 surface-form similarity가 아니라 behavioral equivalence로 정의한다.**

---

## 1. 초기 범위

초기 실험은 아래 family로 제한한다.

- `concise`
- `formal`
- `step_by_step`
- `refusal_safe` (선택)

각 family는 10~30개의 paraphrase를 가진다.

---

## 2. Family membership 기준

같은 family로 묶이려면 아래를 만족해야 한다.

1. 목표 행동 변화가 같다.
2. base model에 유도하려는 출력 차원이 같다.
3. 반대 의미 또는 다른 출력 규칙을 포함하지 않는다.

### 예시: 같은 family

- "Answer briefly."
- "Be concise."
- "Keep the response short and only include the key point."

### 예시: 같은 family가 아님

- "Do not be concise."
- "Give a detailed explanation."

---

## 3. 저장 스키마

권장 저장 형식: `jsonl` 또는 `yaml`

### 최소 필드

- `prompt_id`
- `prompt_family`
- `prompt_text`
- `split` (`train`, `valid`, `test`)
- `paraphrase_split` (`seen`, `unseen`)
- `qc_status` (`draft`, `reviewed`, `approved`, `rejected`)
- `notes`

### 예시

```yaml
- prompt_id: concise_001
  prompt_family: concise
  prompt_text: Answer briefly and only include the key point.
  split: train
  paraphrase_split: seen
  qc_status: approved
  notes: high lexical overlap exemplar

- prompt_id: concise_014
  prompt_family: concise
  prompt_text: Keep your answer minimal unless extra detail is necessary.
  split: test
  paraphrase_split: unseen
  qc_status: approved
  notes: low lexical overlap exemplar
```

---

## 4. QC 체크리스트

각 prompt는 아래를 통과해야 한다.

- [ ] family의 의도된 행동 변화와 일치하는가
- [ ] 반대 지시문이 섞여 있지 않은가
- [ ] family 내부 다른 prompt와 의미적으로 충돌하지 않는가
- [ ] seen/unseen 구분이 명확한가
- [ ] lexical overlap이 높은 예시와 낮은 예시를 모두 확보했는가

---

## 5. 운영 규칙

### 5.1 수동 검증

초기 버전에서는 family assignment를 사람이 직접 검토한다.

### 5.2 seen / unseen 분리

- seen paraphrase는 학습에 사용 가능
- unseen paraphrase는 학습에 절대 사용하지 않음
- unseen은 같은 family 내부 generalization 확인용

### 5.3 family 확장 규칙

새 family를 추가하려면 아래를 함께 추가한다.

1. family 정의 문장
2. positive example 10개 이상
3. negative / confusable example 5개 이상
4. style agreement rule 초안
5. preservation 관점에서 충돌 가능 family 목록

---

## 6. 금지 사항

- lexical similarity가 높다는 이유만으로 자동으로 같은 family에 넣지 말 것
- opposite directive를 같은 family에 넣지 말 것
- family 범위를 너무 넓게 잡아 서로 다른 행동 변화를 한 묶음으로 만들지 말 것
- unseen prompt를 학습 과정에 누출하지 말 것

