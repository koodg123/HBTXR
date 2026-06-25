# Mix-Quant 코드베이스 정밀 분석

> 대상 repo: `REF/ViT-Quantization/Mix-Quant`
> 분석 범위: Mix-Quant 자체 추가/패치 코드(루트 `scripts/`, `evaluation/`, `requirements.txt`, `README.md`)
> 분석 제외: `vllm/` 서브모듈(외부 프레임워크), `.git`, `__pycache__`
> 작성일 기준: 2026-06-20

---

## 1. 개요

### 1.1 정체 (README 기반 확인)
- **공식 명칭**: "Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs"
  (`README.md:1-2`)
- **저자/소속**: Haiquan Lu, Zigeng Chen, Gongfan Fang, Xinyin Ma, Xinchao Wang —
  National University of Singapore, **xML Lab** (`README.md:20-21`)
- **논문**: arXiv 2605.20315, 2026 (`README.md:5`, citation `README.md:184-189`)
- **프로젝트 페이지**: `https://haiquanlu.github.io/Mix-Quant/` (`README.md:8`)

### 1.2 목적 / 문제의식
README Introduction(`README.md:27-28`)에 명시된 문제 정의:
- Agentic LLM 워크플로우는 tool / memory / retrieval / reasoning trace에서 나오는
  **긴 컨텍스트를 반복 처리**하므로, **prefilling 단계가 추론 병목**이 된다.
- 그러나 추론 전 과정에 저비트(low-bit) 양자화를 적용하면 **오차 누적(error accumulation)**으로
  생성 품질이 저하된다.

### 1.3 핵심 아이디어 — phase-aware (단계 인식) 혼합 정밀도
README(`README.md:28`)에 핵심 전략이 명시되어 있다.
- **Prefilling 단계**: 연산 집약적(compute-intensive)이므로 **고처리량 NVFP4 양자화** 적용 → 가속.
- **Autoregressive Decoding 단계**: 안정적이고 신뢰성 있는 생성을 위해 **BF16** 유지.
- 결과: long-context agentic 추론을 가속하면서도 downstream task 성능을 대부분 보존.

이 단계 분리를 **물리적으로** 구현하기 위해, Mix-Quant는 modified vLLM fork 위에
**prefill-decode disaggregated serving(분리 서빙)** 파이프라인을 구성한다 (`README.md:68`).
- 양자화(NVFP4) prefill 서버 + BF16 decode 서버 + 경량 proxy 서버.
- proxy 준비 후 사용자는 `http://localhost:8595/v1`로 표준 OpenAI 호환 요청을 전송 (`README.md:68`).

### 1.4 모델 구성 (확인됨)
- **Prefill 모델**: `RedHatAI/Qwen3-8B-NVFP4` (NVFP4 사전 양자화 체크포인트)
  (`scripts/run_server_qwen3.sh:7`, `README.md:75`)
- **Decode 모델**: `Qwen/Qwen3-8B` (BF16 원본) (`scripts/run_server_qwen3.sh:8`, `README.md:76`)
- **Tokenizer**: `Qwen/Qwen3-8B` 공통 (`scripts/run_server_qwen3.sh:9`)
- served-model-name으로 두 서버 모두 `Qwen/Qwen3-8B`와 `Qwen3-8B` alias를 등록하여
  클라이언트에게는 단일 모델로 노출 (`scripts/run_server_qwen3.sh:6`).

> 주의: 본 repo는 폴더 경로상 "ViT-Quantization" 아래에 분류되어 있으나,
> **실제 코드/논문 내용은 ViT(Vision Transformer)와 무관한 LLM(Qwen3-8B) 서빙·양자화 프로젝트**다.
> ViT 관련 코드/모델/데이터셋은 자체 코드에서 발견되지 않았다 (확인).

---

## 2. 디렉토리 구조

```
Mix-Quant/
├── README.md                         # 프로젝트 설명·설치·실행·평가 가이드 (자체)
├── requirements.txt                  # 자체 추가 의존성 (자체)
├── .gitmodules                       # vllm 서브모듈 정의 (자체)
├── assets/                           # intro.png, framework.png, speedup.svg 등 그림 (자체)
├── index.html, assets/project.css    # 프로젝트 페이지 (자체)
│
├── scripts/                          # ★ 자체 — 서빙·평가 진입점(shell)
│   ├── run_server_qwen3.sh           #   prefill/decode/proxy 분리 서빙 기동
│   ├── eval_qwen3_reasoning.sh       #   reasoning(math500/aime24/aime25/gsm8k) 평가 러너
│   ├── eval_qwen3_longbench-v2.sh    #   LongBench-v2 평가 러너
│   └── eval_qwen3_longmemeval.sh     #   LongMemEval 평가 러너 (+ optional judge)
│
├── evaluation/                       # ★ 자체 — 평가 파이프라인(Python)
│   ├── reasoning/
│   │   ├── evaluate.py               #   baseline: 로컬 vllm.LLM 오프라인 추론
│   │   ├── evaluate_mix.py           #   ★ mix: proxy 서버에 HTTP 요청(분리 서빙)
│   │   └── utils/
│   │       ├── eval_utils.py         #   정답 추출·채점(grade_answer) 로직
│   │       └── math_eval/            #   grader.py, math_normalize.py (수식 정규화/채점)
│   ├── longbench-v2/
│   │   ├── pred.py                   #   baseline: 로컬 vllm.LLM 오프라인 추론
│   │   ├── pred_mix.py               #   ★ mix: proxy 서버에 HTTP 요청(분리 서빙)
│   │   ├── result.py                 #   결과 집계(난이도/길이별 정확도)
│   │   ├── config/model2path.json    #   모델 키→HF 경로 매핑(NVFP4 변종 포함)
│   │   ├── config/model2maxlen.json  #   모델별 최대 컨텍스트 길이
│   │   └── prompts/*.txt             #   0shot / cot / rag / no_context 프롬프트
│   └── LongMemEval/
│       ├── pred.py                   #   baseline: 로컬 vllm.LLM 오프라인 추론
│       ├── pred_mix.py               #   ★ mix: proxy 서버에 HTTP 요청(분리 서빙)
│       ├── requirements.txt          #   judge용 의존성(openai 등)
│       └── src/                      #   LongMemEval 원본 evaluation/retrieval/generation
│           └── evaluation/evaluate_qa.py  # LLM-judge QA 채점
│
└── vllm/   ← 【외부 프레임워크, 분석 제외】
    └── (modified vLLM fork: github.com/haiquanlu/vllm @ branch mix-quant)
        proxy 서버 본체(toy_proxy_server.py)와 NVFP4 커널·NixlConnector 구현이 여기 위치.
```

- **자체 코드 핵심**: `scripts/*.sh`(오케스트레이션) + `evaluation/**/*_mix.py`(분리 서빙 클라이언트).
- **외부 프레임워크 `vllm/`**: `.gitmodules:1-4`에 의해
  `https://github.com/haiquanlu/vllm.git`, branch `mix-quant`로 등록된 modified vLLM fork.
  분리 서빙·NVFP4 양자화·KV transfer(NixlConnector)·proxy 서버 본체가 모두 여기 있으며,
  **본 분석에서는 이름만 언급하고 내부 코드 분석은 제외**한다.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `scripts/run_server_qwen3.sh` — prefill-decode 분리 서빙 오케스트레이터 (가장 중요)

이 스크립트가 Mix-Quant "phase-aware" 전략의 **실제 구현 핵심**이다.
세 프로세스(prefill 서버 → decode 서버 → proxy)를 순차 기동하고, 각 서버가
healthcheck를 통과할 때까지 대기한다.

**(1) 환경/인자 기본값** (`run_server_qwen3.sh:6-30`)
- `PREFILL_MODEL_NAME=RedHatAI/Qwen3-8B-NVFP4`, `DECODE_MODEL_NAME=Qwen/Qwen3-8B`
  (`:7-8`) — prefill만 NVFP4, decode는 BF16.
- 포트: prefill `8500`, decode `8600`, proxy `8595` (`:20-22`).
- NIXL side-channel 포트: prefill `5610`, decode `5611` (`:23-24`) — KV 전송용.
- GPU 분리: `PREFILL_GPU=0`, `DECODE_GPU=1` (`:11-12`) → 두 단계를 별도 GPU에 배치.
- `MAX_MODEL_LENGTH=131072`(기본), `MAX_NUM_SEQS=1`(`:14-15`).
- `HF_OVERRIDES` 기본값으로 **YaRN RoPE 확장**(`rope_type=yarn`, `factor=4.0`,
  `original_max_position_embeddings=32768`)을 주입 → 32768 → 131072 컨텍스트 확장 (`:17`).
  README는 reasoning 평가 시 `--hf-overrides ''`로 이를 비활성화하라고 안내(`README.md:114-124`).

**(2) `PROXY_SERVER` 위치** (`run_server_qwen3.sh:35`)
```
PROXY_SERVER="${REPO_DIR}/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
```
→ proxy 본체는 **vllm 서브모듈 내부**(외부, 분석 제외)에 위치. 자체 코드는 이를 호출만 한다.
파일 존재는 Glob으로 확인됨(`vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`).

**(3) `wait_for_server()`** (`run_server_qwen3.sh:97-117`)
- `/v1/completions` 엔드포인트를 `curl`로 폴링하여 ready 확인(`:102`).
- 동시에 모든 자식 PID가 살아있는지 검사, 죽으면 즉시 실패 반환(`:106-111`).

**(4) prefill 서버 기동 (NVFP4)** (`run_server_qwen3.sh:131-152`)
```
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=... VLLM_NIXL_SIDE_CHANNEL_PORT=5610 \
CUDA_VISIBLE_DEVICES=0 vllm serve RedHatAI/Qwen3-8B-NVFP4 \
  --port 8500 --served-model-name "Qwen/Qwen3-8B" "Qwen3-8B" \
  --tokenizer Qwen/Qwen3-8B --max-model-len 131072 \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both",
                         "kv_load_failure_policy":"fail"}'
```
- `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE`(`:132`): FP4 MoE 커널 관련 환경변수
  (vllm 내부에서 소비, NVFP4 경로 활성화 정황).
- `--kv-transfer-config`(`:150`): **NixlConnector** 사용, `kv_role=kv_both`로 KV cache를
  네트워크로 송수신. prefill이 만든 KV cache를 decode 서버로 넘기기 위한 핵심 설정.
- 양자화 자체는 `vllm serve`에 NVFP4 체크포인트를 넘기는 것으로 처리(가중치가 이미 양자화됨).

**(5) decode 서버 기동 (BF16)** (`run_server_qwen3.sh:154-173`)
- prefill과 거의 동일하나 `DECODE_MODEL_NAME=Qwen/Qwen3-8B`(BF16), `CUDA_VISIBLE_DEVICES=1`,
  side-channel 포트 `5611`, port `8600`만 다르다.
- 동일하게 `NixlConnector`, `kv_role=kv_both`로 prefill의 KV를 수신할 수 있게 구성(`:171`).
- prefill과 달리 `--max-num-seqs` 미지정(decode는 continuous batching).

**(6) proxy 서버 기동** (`run_server_qwen3.sh:175-183`)
```
python "$PROXY_SERVER" --port 8595 \
  --prefiller-hosts localhost --prefiller-ports 8500 \
  --decoder-hosts localhost  --decoder-ports 8600
```
- proxy는 prefiller(8500)와 decoder(8600)를 알고, 클라이언트 요청을 받아
  **prefill → KV 전송 → decode** 흐름으로 라우팅한다.
- 외부 클라이언트에게는 `8595` 단일 OpenAI 호환 엔드포인트로 노출.

**(7) cleanup / 종료** (`run_server_qwen3.sh:86-95`, `186-192`)
- `trap cleanup EXIT`로 종료 시 모든 PID kill.
- `READY_FILE` 지정 시 ready 마커 파일 생성(외부 자동화 연동용).

> **요약 (proxy 서버 역할)**: proxy는 양자화 prefill 서버와 BF16 decode 서버를
> KV-cache 전송(NIXL)으로 묶어, "긴 입력은 NVFP4로 빠르게 prefill → 그 KV를 BF16 decode로
> 넘겨 정밀 생성"하는 **phase-aware 추론 경로를 단일 OpenAI API 뒤에 캡슐화**한다.
> 단, proxy 본체와 NIXL/NVFP4 구현은 vllm 서브모듈(외부)에 있다.

---

### 3.2 평가 클라이언트: `pred_mix.py` / `evaluate_mix.py` vs baseline (`pred.py`/`evaluate.py`)

세 벤치마크 모두 동일한 패턴으로 **baseline(로컬 오프라인 vllm.LLM)**과
**mix(분리 서빙 HTTP 클라이언트)** 두 버전을 제공한다. **핵심 차이는 추론 백엔드뿐**이고,
프롬프트 구성·정답 추출 로직은 거의 동일하다.

#### 공통 차이 (baseline vs mix)
| 구분 | baseline (`pred.py` / `evaluate.py`) | mix (`pred_mix.py` / `evaluate_mix.py`) |
|---|---|---|
| 추론 백엔드 | `from vllm import LLM, SamplingParams` 로컬 로드 | `requests.post(...)` HTTP 호출 |
| 모델 로드 | `LLM(model=..., gpu_memory_utilization=0.95, max_model_len=...)` | 없음 (서버가 모델 보유) |
| 호출 대상 | 프로세스 내 GPU | `server_url + /v1/completions` (기본 proxy 8595) |
| 동시성 | vllm 내부 배칭 | `ThreadPoolExecutor(max_workers=max_concurrent_requests)` |
| 양자화/분리 | 단일 모델·단일 정밀도 | prefill(NVFP4)/decode(BF16) 분리는 **서버 측에서** 일어남 |

→ **즉, 자체 mix 코드는 "양자화/분리"를 직접 수행하지 않는다.** mix 스크립트는 단지
`/v1/completions`로 proxy(8595)에 요청을 보낼 뿐이며, NVFP4 prefill / BF16 decode 분리는
`run_server_qwen3.sh`가 기동한 vllm 서버 + proxy가 담당한다. mix 클라이언트의 본질은
**"오프라인 vllm.LLM 대신 분리 서빙 엔드포인트를 호출하도록 바꾼 것"**이다.

#### 3.2.1 LongBench-v2 — `evaluation/longbench-v2/pred.py` vs `pred_mix.py`
- baseline `pred.py:119-124`: 프로세스 내 `LLM(...)` 로드, `:155` `llm.generate(prompts, ...)`.
- mix `pred_mix.py`:
  - `build_request_payload()`(`:130-141`): `model/prompt/temperature/top_p/top_k/max_tokens/seed/stream`을
    OpenAI `/v1/completions` 페이로드로 구성. `skip_special_tokens=False`(`:138`).
  - `call_completion()`(`:144-152`): `requests.post(url, json=payload, timeout=6000)`,
    응답 `choices[0].text` 반환.
  - `run_disaggregated_generation()`(`:155-171`): URL = `server_url + "/v1/completions"`(`:156`),
    `ThreadPoolExecutor`로 동시 요청 후 인덱스 보존하여 결과 수집.
  - `get_pred()`(`:174-201`): 프롬프트 생성은 baseline과 동일,
    마지막 줄(`:201`)만 `run_disaggregated_generation()` 호출로 대체.
- 프롬프트/정답 추출(`extract_answer_cot`, `extract_answer`)은 두 파일이 사실상 동일
  (baseline `pred.py:58-93`, mix `pred_mix.py:59-92`).
- 프롬프트 분기: `rag>0 / no_context / cot / 기본` 4종(`pred_mix.py:185-195`).
- 긴 입력은 `prepare_prompt()`(`:104-110`)에서 `max_len`(model2maxlen) 초과 시
  **앞 절반 + 뒤 절반**으로 절단(중간 버림).
- 함수 명칭 `run_disaggregated_generation`이 분리 서빙(disaggregated) 의도를 직접 드러냄.

#### 3.2.2 Reasoning — `evaluation/reasoning/evaluate.py` vs `evaluate_mix.py`
- baseline `evaluate.py:83` `LLM(...)`, `:92-95` `model.generate(...)` (오프라인).
- mix `evaluate_mix.py`:
  - `build_request_payload()`(`:25-35`), `call_completion()`(`:77-90`) — 단,
    응답에서 `usage.completion_tokens`를 읽어 **생성 길이(length)**까지 추출(`:86-89`).
  - `run_disaggregated_generation()`(`:93-109`): `server_url + /v1/completions` 호출(`:94`),
    스레드풀 동시 요청.
  - `build_chat()`(`:13-22`): 시스템 지시 `"Put your final answer within \boxed{}."`를 붙여
    chat template 적용 — baseline과 동일(`evaluate.py:21-31`).
  - `save_decode_results()`(`:38-74`): `utils.eval_utils.evaluate_predictions`로 acc/추출률/
    평균길이 계산 후 `results/<model>/(thinking|no_thinking)/<dataset>_seed<n>_metrics.txt`에 저장.
- baseline은 `results/baselines/<model>/...`에 저장(`evaluate.py:98`), mix는 `results/<model>/...`에
  저장(`evaluate_mix.py:41`) — 결과 디렉토리만 다름.

#### 3.2.3 LongMemEval — `evaluation/LongMemEval/pred.py` vs `pred_mix.py`
- baseline `pred.py:132-137` `LLM(...)`, `:157` `llm.generate(...)`.
- mix `pred_mix.py`: `build_request_payload()`(`:126-137`), `call_completion()`(`:140-148`),
  `run_disaggregated_generation()`(`:151-167`) — 동일 패턴.
- 공통: `build_history_text()`(`:49-77`/baseline `:50-78`)로 세션을 타임스탬프 정렬해
  `### Session i` 형식 히스토리 문자열 구성, `truncate_prompt()`(`:80-85`)로 앞/뒤 절반 절단.
- 출력 후처리 `extract_hypothesis()`(`:35-47`): `<|channel>thought...<channel|>`(Gemma4) 또는
  `<think>...</think>`(Qwen/DeepSeek) 사고 블록을 제거하고 최종 답만 남김.
- 캐시: 이미 처리한 `question_id`는 건너뜀(`pred_mix.py:202-214`).

#### 3.2.4 LLM-judge 채점 — `src/evaluation/evaluate_qa.py`
- LongMemEval QA 정확도는 LLM judge로 채점(`evaluate_qa.py`).
- `model_zoo`(`:11-15`): `gpt-4o`, `gpt-4o-mini`(openai), `llama-3.1-70b-instruct`(local).
- task 유형별 채점 프롬프트(`get_anscheck_prompt`, `:24-43`): temporal-reasoning은 off-by-one
  관용, knowledge-update는 갱신값 우선 등 task-specific 규칙.
- `eval_qwen3_longmemeval.sh:104-110`에서 `--judge-model` 지정 시 자동 호출,
  결과는 `<hyp>.eval-results-<judge>` 파일로 저장(`evaluate_qa.py:56`).

---

### 3.3 결과 집계 — `evaluation/longbench-v2/result.py`
- `results/mix` 디렉토리의 모든 예측 파일을 읽어(`result.py:3-12`),
  난이도(easy/hard)·길이(short/medium/long)별 정확도를 표로 집계(`:13-37`).
- `Overall / Easy / Hard / Short / Medium / Long` 컬럼의 정확도(%)를 `result.txt`로 출력(`:39`).
- `compensated` 플래그(`:6`): 추출 실패 시 0.25점(4지선다 무작위 기댓값) 보정 옵션(`:17-18`, 기본 off).

---

### 3.4 평가 러너 셸 (`scripts/eval_*.sh`)
- 세 러너 모두 인자 파싱 후 해당 `*_mix.py`를 호출하는 래퍼.
- `eval_qwen3_reasoning.sh`: 기본 데이터셋 `math500 aime24 aime25`(`:12`),
  `MAX_TOKENS=32768`, `T=0.6/top_p=0.95/top_k=20`, `enable_thinking` 기본 on(`:14-18`),
  데이터셋×시드 이중 루프로 `evaluate_mix.py` 실행(`:74-97`).
  서버 URL 기본값은 proxy인 `http://localhost:8595`(`:10`).
- `eval_qwen3_longbench-v2.sh`: 모델 키 `Qwen3-8B`(`:8`), `MAX_NEW_TOKENS=30720`(`:13`),
  `cot/no_context/rag` 토글(`:60-63`), `pred_mix.py` 호출(`:105`).
- `eval_qwen3_longmemeval.sh`: `MAX_CONTEXT_LEN=112000`, `MAX_NEW_TOKENS=19072`(`:14-15`),
  데이터 파일 존재 검사(`:77-81`), `pred_mix.py` 호출(`:102`), optional judge(`:104-110`).

---

## 4. 알고리즘 — 혼합 정밀도 phase 분리 전략과 NVFP4 적용 지점

### 4.1 phase 분리는 "서빙 구조"로 구현됨 (자체 코드)
Mix-Quant의 phase-aware 전략은 **알고리즘 코드가 아니라 분리 서빙 토폴로지**로 실현된다.
1. NVFP4 가중치를 가진 prefill 서버(GPU 0)가 긴 입력을 고속 prefill → KV cache 생성.
2. NixlConnector(`kv_role=kv_both`)가 그 KV cache를 BF16 decode 서버(GPU 1)로 전송.
3. BF16 decode 서버가 autoregressive 토큰 생성을 정밀하게 수행.
4. proxy(8595)가 위 흐름을 단일 API로 캡슐화.
→ 근거: `scripts/run_server_qwen3.sh:131-183`, README `:28,68`.

### 4.2 NVFP4 양자화 — 자체 코드에 양자화 알고리즘 구현 **없음** (확인)
- 자체 코드(`scripts/`, `evaluation/`) 전체를 grep한 결과, `NVFP4`/`fp4`/`quant`/`bitwidth`/
  `allocat`(비트할당) 관련 **양자화 구현 로직이 전무**하다.
  - `scripts/`의 NVFP4 등장은 전부 **모델 이름 문자열**(`RedHatAI/Qwen3-8B-NVFP4`)과
    환경변수 이름(`VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE`)뿐(`run_server_qwen3.sh:7,28,132,155`).
  - `evaluation/`의 NVFP4 등장은 전부 **config의 HF 경로 매핑/최대길이**뿐
    (`model2path.json:14,16,18,20`, `model2maxlen.json:19,21,23,25`).
- **결론(확인)**: 자체 코드에는 양자화 알고리즘·비트할당(bit allocation) 알고리즘이 없다.
  NVFP4 모델은 **외부 RedHatAI 사전 양자화 체크포인트**(`RedHatAI/Qwen3-8B-NVFP4` 등)를 그대로
  로드하여 사용한다 (`run_server_qwen3.sh:7`, `model2path.json:14`).
- 실제 NVFP4 연산 커널 / 양자화 가중치 처리 / KV transfer(NIXL)·proxy 라우팅은 모두
  **modified vLLM fork(`vllm/` 서브모듈, 외부, 분석 제외)** 측에 존재한다 (`.gitmodules:1-4`,
  `run_server_qwen3.sh:35`). README Acknowledgements(`:192-199`)도 vllm / llm-compressor /
  evalscope / FP-Quant를 기반으로 명시한다 — 즉 양자화 자산은 외부 도구·체크포인트에 의존.
- **추정**: NVFP4 체크포인트 자체는 llm-compressor/FP-Quant 계열로 사전 생성된 것으로 보이나,
  그 생성 스크립트는 본 repo 자체 코드에 포함되어 있지 않다(자체 코드 내 미발견 — 확인).

### 4.3 비트할당(bit allocation) 알고리즘 유무
- 입력/단계별로 비트폭을 동적으로 고르는 비트할당 알고리즘은 자체 코드에 없다(확인).
- Mix-Quant의 "혼합 정밀도"는 **단계 단위의 고정 매핑**(prefill=NVFP4, decode=BF16)이며,
  레이어/토큰 단위 가변 비트할당이 아니다 — 근거: 서버 2종이 각각 단일 정밀도 모델을 로드
  (`run_server_qwen3.sh:137`(NVFP4), `:160`(BF16)).

---

## 5. 학습/평가 파이프라인

> 학습(training/fine-tuning) 코드는 자체 코드에 없다(확인). 본 repo는 **추론·평가 전용**이다.

### 5.1 서빙 기동 (전제)
```bash
bash scripts/run_server_qwen3.sh \
  --prefill-model-name RedHatAI/Qwen3-8B-NVFP4 \
  --decode-model-name Qwen/Qwen3-8B \
  --prefill-gpu 0 --decode-gpu 1 \
  --tensor-parallel-size 1 --max-model-length 131072 --proxy-port 8595
```
(`README.md:73-82`)

### 5.2 Reasoning 벤치마크 (math500 / aime24 / aime25 / gsm8k)
- 데이터셋 로딩(`evaluate_mix.py:131-142`):
  `openai/gsm8k`, `HuggingFaceH4/aime_2024`, `math-ai/aime25`, `HuggingFaceH4/MATH-500`.
- 채점: `\boxed{}` 내용 추출(`eval_utils.py:118-142`) 후 `grade_answer`(math_eval)로 정오 판정.
- 실행(`README.md:130-134`): `bash scripts/eval_qwen3_reasoning.sh --seed 42 --max-concurrent-requests 32`.
- reasoning 서버는 `--hf-overrides ''`로 native 컨텍스트(40960) 사용 권장(`README.md:114-124`).
- 결과: `evaluation/reasoning/results/<model>/thinking/` (`README.md:136`).

### 5.3 LongBench-v2
- 데이터셋: `THUDM/LongBench-v2`(`pred_mix.py:216`).
- 모델 키 `Qwen3-8B`는 `config/model2path.json`에서 해석(`eval_qwen3_longbench-v2.sh:8`,
  `model2path.json:13`).
- 실행(`README.md:143-146`): `bash scripts/eval_qwen3_longbench-v2.sh --seed 42 --save-dir results/qwen3-8b`.
- 출력: 예측·정오가 JSONL로 `evaluation/longbench-v2/results/`에 기록(`README.md:148`),
  집계는 `result.py`.

### 5.4 LongMemEval
- 데이터 준비: `longmemeval_s_cleaned.json` 다운로드(`README.md:152-158`).
- 실행(`README.md:163-167`):
  `bash scripts/eval_qwen3_longmemeval.sh --data-file data/longmemeval_s_cleaned.json --seed 42 --save-dir results/qwen3-8b`.
- 출력: `evaluation/LongMemEval/results/`(`README.md:168`).
- QA 채점은 LLM judge: `OPENAI_API_KEY` 설정 후 `--judge-model gpt-4o` 전달
  (`README.md:169-179`, `eval_qwen3_longmemeval.sh:104-110`).

### 5.5 baseline 비교 경로
- 각 벤치마크의 `pred.py`/`evaluate.py`(로컬 오프라인 단일 모델)는 분리 서빙 없이 직접
  추론하는 baseline. mix(`*_mix.py`)와 결과를 비교하여 "NVFP4 prefill + BF16 decode"의
  정확도/속도 trade-off를 측정하는 구조(추정 — 두 버전의 결과 디렉토리 분리로 뒷받침:
  `evaluate.py:98` baselines vs `evaluate_mix.py:41`).

---

## 6. 의존성

### 6.1 루트 `requirements.txt` (자체 추가, `requirements.txt:1-6`)
```
transformers==5.6.2
datasets
tqdm
tiktoken
pylatexenc
nixl
```
- `transformers==5.6.2`: 토크나이저·chat template (Qwen3, Gemma4 등 신모델 가정).
- `nixl`: KV cache 전송용 라이브러리(NixlConnector 백엔드) — 분리 서빙의 핵심 의존성.
- `pylatexenc`: 수식 정규화(reasoning 채점)용.

### 6.2 vLLM 설치 (서브모듈, `README.md:46-64`)
- modified vLLM fork(`vllm/`)를 `--editable .`로 설치, 사전컴파일 wheel commit
  `28ee78af543c563a2fbf78829a7688120e4e4eb5` 지정(`README.md:55`).

### 6.3 LongBench-v2 `evaluation/longbench-v2/requirements.txt`
```
python=3.10 / transformers==4.56.0 / vllm==0.10.1 / datasets==2.21.0 / openai / tqdm / tiktoken
```
(`longbench-v2/requirements.txt:1-7`)
> 주의: 루트(`transformers==5.6.2`)와 이 파일(`transformers==4.56.0`, `vllm==0.10.1`)의
> 버전이 상충한다. 후자는 LongBench-v2 **원본 baseline(`pred.py`, 로컬 vllm.LLM)** 기준일 가능성.
> mix 경로(HTTP 클라이언트)는 vllm 로컬 설치가 불필요하므로 루트 요건만 충족하면 됨(추정).

### 6.4 LongMemEval judge `evaluation/LongMemEval/requirements.txt`
```
packaging / openai==1.35.1 / tqdm==4.66.4 / backoff==2.2.1 / numpy==1.26.3 / nltk==3.9.1
```
(`LongMemEval/requirements.txt:1-6`) — LLM judge 채점 전용(openai 클라이언트, backoff 재시도).

### 6.5 외부 기반 프로젝트 (README Acknowledgements `:192-199`)
vllm, llm-compressor, evalscope, FP-Quant.

---

## 7. 강점 / 한계 / 리스크

### 7.1 강점
- **개념의 단순·명료함**: "긴 입력 prefill만 저비트, 생성은 고정밀" 이라는 phase-aware
  전략을 **표준 분리 서빙(disaggregated PD) + KV transfer**라는 기성 인프라로 깔끔하게 구현.
- **재현성**: NVFP4 모델을 RedHatAI 공개 체크포인트로 고정하고, vllm fork를 특정 commit으로
  pin(`README.md:55`)하여 재현 가능성 확보.
- **클라이언트 비침습성**: 사용자는 표준 OpenAI API(`/v1`)만 쓰면 되고, 양자화/분리는 서버에
  은닉(`README.md:84-105`) → 평가 코드도 baseline 대비 추론 호출부만 교체(낮은 결합도).
- **평가 폭**: reasoning(math/aime/gsm8k) + long-context(LongBench-v2) + agentic memory
  (LongMemEval, LLM-judge)까지 agentic·long-context 시나리오를 폭넓게 커버.

### 7.2 한계
- **자체 코드의 알고리즘 기여 부재**: 자체 repo에는 양자화/비트할당/커널 알고리즘이 없고,
  핵심 가속 로직(NVFP4 커널, NIXL KV transfer, proxy 라우팅)은 전부 vllm fork에 의존(확인).
  즉 본 repo만으로는 "어떻게 NVFP4가 작동하는가"를 알 수 없다.
- **고정 매핑**: 단계 단위 고정 정밀도(prefill=4bit, decode=16bit)로, 입력 난이도·레이어별
  적응형 비트할당은 없다(§4.3).
- **2-GPU·서버 의존**: 기본 구성이 prefill/decode 별도 GPU(`:11-12`) + 3프로세스 상시 가동으로,
  단일 디바이스/엣지 환경엔 직접 적용 곤란.

### 7.3 리스크
- **버전 충돌**: transformers 5.6.2(루트) vs 4.56.0(longbench)·vllm 0.10.1 상충(§6.3) →
  환경 분리 필요.
- **미래 날짜/버전 표기**: arXiv 2605.20315(2026), `transformers==5.6.2`, `Qwen3.5`,
  `gemma-4` 등은 분석 시점(2026-06) 기준으로 비표준/미래형 표기 → repo 자체가 신규/프리뷰
  성격일 가능성(추정).
- **KV 전송 의존**: NIXL 기반 KV transfer 실패 시 `kv_load_failure_policy=fail`(`:150,171`)로
  요청 자체가 실패 → 운영 안정성은 NIXL/네트워크에 종속.

---

## 8. 우리 프로젝트(HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

> 본 절은 PRJXR-HBTXR(HG-PIPE 계열 FPGA 가속기 + XR eye-tracking)와의 연관을 **추정**으로
> 기술한다. Mix-Quant는 GPU/vLLM 기반 LLM 서빙 프로젝트로, FPGA·ViT·시선추적과 **직접적인
> 코드 공유는 없다**(확인). 아래는 설계 원리 차원의 시사점.

### 8.1 단계별 정밀도 분리 → FPGA 연산 단계별 정밀도 분리 (추정)
- Mix-Quant의 "prefill=저비트 / decode=고비트" 단계 분리는, FPGA 가속기에서
  **연산 단계(스테이지)별로 다른 비트폭을 할당**하는 설계와 1:1 대응 가능(추정).
- 예: HG-PIPE류 파이프라인에서 초반 대량 연산(예: patch embedding / attention QK)을 저비트
  PE로, 정밀이 중요한 후단(예: 분류 head / 좌표 회귀)을 고비트 PE로 분리하는 전략.

### 8.2 NVFP4 = HW 친화적 저비트 포맷의 레퍼런스 (추정)
- NVFP4(4비트 부동소수, block scaling)는 FPGA DSP/LUT로 구현 가능한 **블록 스케일 저비트 포맷**의
  최신 예시. ViT 양자화 가속기에서 INT4 대안으로 FP4 계열 포맷 채택 시 정확도/면적 trade-off
  참고점이 될 수 있음(추정). 단, 본 repo에는 커널 구현이 없어 포맷 사양은 vllm/RedHatAI 측 참조 필요.

### 8.3 prefill/decode 분리 = 가속기 파이프라인 스테이지 분리 (추정)
- prefill(병렬·compute-bound) vs decode(순차·memory-bound) 분리는, FPGA에서
  **throughput 지향 스테이지**와 **latency 지향 스테이지**를 서로 다른 하드웨어 자원/클럭으로
  나누는 dataflow 설계와 유사. XR 시선추적의 실시간 파이프라인(프레임 단위 대량 전처리 +
  순차 추적 갱신)에 동일한 분리 원리 적용 가능(추정).

### 8.4 KV transfer = 스테이지 간 버퍼/중간표현 전달 (추정)
- NIXL KV transfer는 "한 스테이지가 만든 중간 상태를 다음 스테이지로 넘기는" 메커니즘.
  FPGA에서는 BRAM/URAM/스트림 FIFO를 통한 스테이지 간 중간 텐서 전달에 대응(추정).

### 8.5 직접 재사용 가능 자산
- **평가 방법론**: long-context·reasoning·memory를 분리 측정하고, baseline vs 양자화 경로를
  동일 프롬프트로 비교하는 구조(§5.5)는, FPGA 양자화 ViT의 정확도 회귀 검증 프레임으로 차용 가능.
- **단계별 고정 정밀도 매핑의 단순성**: 적응형 비트할당보다 구현·검증이 쉬워, FPGA 초기
  설계에서 "스테이지별 고정 비트폭" 출발점으로 적합(추정).

---

## 9. 근거 표기 / 불확실성 명시

- **확인됨 (코드/README 직접 근거)**
  - phase-aware 전략(prefill NVFP4 / decode BF16): `README.md:28`, `run_server_qwen3.sh:7-8,137,160`.
  - 분리 서빙 + proxy + NIXL KV transfer: `run_server_qwen3.sh:131-183`, `README.md:68`.
  - proxy 본체가 vllm 서브모듈에 위치: `run_server_qwen3.sh:35` + Glob 확인.
  - mix vs baseline 차이는 추론 백엔드(로컬 vllm.LLM ↔ HTTP /v1/completions): §3.2 각 파일 라인.
  - 자체 코드에 양자화/비트할당 알고리즘 없음, NVFP4는 외부 RedHatAI 체크포인트: §4.2 (grep 전수).
  - 학습 코드 없음, 추론·평가 전용: `evaluation/` 전수 + 학습 스크립트 미발견.
  - 의존성: 각 `requirements.txt` 라인, `.gitmodules:1-4`.

- **추정 (명시)**
  - baseline vs mix를 정확도/속도 비교 목적으로 둔 구조(§5.5).
  - longbench requirements와 루트 requirements 버전 차이의 적용 범위(§6.3).
  - NVFP4 체크포인트가 llm-compressor/FP-Quant로 생성됐을 가능성(§4.2) — 생성 스크립트 미포함.
  - §8 FPGA/XR 시선추적 연관 시사점 전반 — 설계 원리 차원의 유추.

- **확인 불가 / 분석 제외**
  - NVFP4 커널·NixlConnector·proxy 내부 알고리즘: `vllm/` 서브모듈(외부 프레임워크)에 위치하여
    본 분석 범위에서 제외.
  - 실제 speedup/정확도 수치: 논문 본문(arXiv 2605.20315) 영역이며 자체 코드로는 미확인.

- **정체 관련 솔직한 명시**
  - 폴더 분류상 "ViT-Quantization" 아래 있으나, **실제 내용은 ViT가 아닌 LLM(Qwen3-8B) 분리 서빙
    + NVFP4 prefill 프로젝트**다. ViT 관련 코드/모델은 자체 코드에서 발견되지 않음(확인).
