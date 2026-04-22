# EXPERIMENT_LOG_TEMPLATE.md

아래 템플릿은 각 실험 실행마다 복사해서 사용한다.

---

## 실험 제목

- 날짜:
- 작성자:
- run name:

## 목적

- 이 실험으로 무엇을 확인하려는가?

## 설정

- baseline name:
- model:
- config path:
- resolved config path:
- git commit:
- seed:
- prompt family:
- paraphrase split:
- trainable params:

## 실행 명령

```bash
# command here
```

## 데이터

- train split:
- valid split:
- test split:
- preservation split:

## 결과 요약

- teacher fidelity:
- preservation:
- efficiency:
- result json path:

## 주요 metric

| Metric | Value | Notes |
|---|---:|---|
| token_kl |  |  |
| next_token_agreement |  |  |
| style_agreement |  |  |
| base_drift_kl |  |  |
| inference_latency_ms |  |  |

## 관찰

- 잘된 점:
- 이상한 점:
- 실패/에러:

## 해석

- 결과가 가설과 어떻게 연결되는가?

## 다음 액션

- [ ]
- [ ]
- [ ]

