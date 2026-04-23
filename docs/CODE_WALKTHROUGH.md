# CODE_WALKTHROUGH.md

이 문서는 현재 코드가 **어떤 흐름으로 동작하는지**를 연구 목적과 연결해서 쉽게 설명한다.

---

## 1. 실행은 어디서 시작되는가

엔트리포인트는 [src/where_to_bake/run.py](/data/nksol0405/LLM/prompt_baking/where_to_bake/src/where_to_bake/run.py) 다.

역할:

1. `--config`로 YAML을 읽는다.
2. baseline 이름이 registry에 있는지 확인한다.
3. `run_experiment(...)`를 호출한다.

즉, 이 파일은 "어떤 실험을 돌릴지 선택하는 문"이다.

---

## 2. config는 어떻게 읽히는가

[src/where_to_bake/config.py](/data/nksol0405/LLM/prompt_baking/where_to_bake/src/where_to_bake/config.py) 는 YAML을 합치고 검증한다.

핵심:

- `defaults:`에 적힌 base config들을 먼저 읽는다.
- 그 뒤 baseline config가 위에 덮어쓴다.
- 최종 merged config를 `resolved_config.yaml`로 저장한다.

즉, 실험값의 source of truth는 코드가 아니라 config다.

---

## 3. 데이터는 어떻게 만들어지는가

toy/smoke 데이터는 `data/datasets/*.jsonl` 에 직접 있다.

long-form 데이터는 아래 흐름으로 생성된다.

1. seed source corpus: `data/source_corpus/longform_seed_v1.yaml`
2. prompt family spec: `data/prompt_families/prompt_family_longform_v1.yaml`
3. generator: [src/where_to_bake/data/longform_generator.py](/data/nksol0405/LLM/prompt_baking/where_to_bake/src/where_to_bake/data/longform_generator.py)

generator가 하는 일:

- 긴 입력 문서를 읽는다.
- family별 prompt paraphrase를 읽는다.
- 같은 입력에 대해 `concise`, `formal`, `step_by_step` target을 붙인다.
- train/valid/test/preserve JSONL을 생성한다.

---

## 4. teacher/student 입력은 어떻게 다르게 들어가는가

[src/where_to_bake/data/prompt_dataset.py](/data/nksol0405/LLM/prompt_baking/where_to_bake/src/where_to_bake/data/prompt_dataset.py) 가 핵심이다.

teacher 입력:

- `System: <prompt_text>`
- `User: <input_text>`
- `Assistant:`

student 입력:

- `User: <input_text>`
- `Assistant:`

즉 teacher는 prompt를 보고, student는 prompt 없이 입력만 본다.

이 구조가 바로 "prompt baking"의 기본 설정이다.

---

## 5. 학습은 무슨 loss로 되는가

[src/where_to_bake/train/losses.py](/data/nksol0405/LLM/prompt_baking/where_to_bake/src/where_to_bake/train/losses.py) 의 `compute_token_kl(...)` 가 핵심이다.

의미:

- teacher가 prompt를 넣었을 때 내는 next-token 분포
- student가 prompt 없이 내는 next-token 분포

이 둘의 KL divergence를 줄인다.

즉 student가 "prompt를 본 것처럼" 행동하게 만들려는 것이다.

---

## 6. baseline마다 무엇이 다른가

[src/where_to_bake/baselines/selection.py](/data/nksol0405/LLM/prompt_baking/where_to_bake/src/where_to_bake/baselines/selection.py) 가 baseline별 차이를 만든다.

지금 구현된 차이:

- `promptbake_kl`: config에 적힌 target module 사용
- `full_target_lora_kl`: 지정된 suffix 전체 사용
- `all_layer_lora_kl`: 후보 모듈 전부 사용
- `random_subset_kl`: 후보 중 랜덤 선택
- `magnitude_topk`: teacher/base activation 차이 크기 기준 선택
- `gradient_topk`: gradient norm 기준 선택

중요:

- 이건 아직 **mechanism-guided localization**이 아니다.
- 지금은 비교 baseline과 heuristic selection 수준이다.

---

## 7. trainer는 실제로 무엇을 하는가

[src/where_to_bake/train/trainer.py](/data/nksol0405/LLM/prompt_baking/where_to_bake/src/where_to_bake/train/trainer.py) 흐름:

1. config 저장
2. dataset 로드
3. baseline selection 수행
4. `selection_debug.json` 저장
5. teacher/student 모델 생성
6. train mode면 KL distillation 학습
7. eval 수행
8. `result.json` 저장

즉 이 파일은 실험 전체를 오케스트레이션한다.

---

## 8. 지금 왜 핵심 목표에 아직 못 갔는가

원래 연구 목표는:

- prompt family 내부 안정성
- family 간 구분성
- causal importance

를 계산해서 LoRA 위치를 고르는 것이다.

하지만 지금은 아직 아래가 없다.

- `within_family_consistency`
- `across_family_similarity`
- `stability_score`
- `causal_score`

그래서 현재 저장소는:

- "prompt baking baseline 코드베이스"
- "비교 실험용 harness"

까지는 왔지만,

- "mechanism-guided selective prompt baking"

의 핵심은 아직 구현 중이다.

---

## 9. 새로 추가된 similarity inspection은 무엇을 하는가

`scripts/run_prompt_similarity.py` 와
[src/where_to_bake/localization/similarity.py](/data/nksol0405/LLM/prompt_baking/where_to_bake/src/where_to_bake/localization/similarity.py)
는 M2의 최소 버전이다.

하는 일:

1. 같은 입력(source)에 대해 여러 prompt를 읽는다.
2. base model의 무프롬프트 내부 출력과 teacher prompted 내부 출력을 비교한다.
3. 각 module에서 `delta = teacher - base` 를 만든다.
4. 같은 family prompt끼리 cosine similarity를 계산한다.
5. 다른 family prompt끼리 cosine similarity를 계산한다.
6. `within - alpha * across` 형태의 preview stability score를 저장한다.

즉 이 도구는

- "유사한 prompt일 때 내부 변화가 실제로 비슷한가?"
- "다른 prompt family일 때는 덜 비슷한가?"

를 확인하기 위한 것이다.

중요:

- 이건 아직 causal score가 없다.
- 따라서 최종 localization 결론이 아니라 **관찰용/디버깅용 M2 최소 도구**다.
