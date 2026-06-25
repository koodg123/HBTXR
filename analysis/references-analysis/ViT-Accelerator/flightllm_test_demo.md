# flightllm_test_demo 정밀 분석

## 1. 개요
- **목적**: FlightLLM(FPGA'24, "FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs")의 공개 테스트/재현 데모. LLaMA2-7B를 FPGA에서 추론할 때의 (a) 성능 프로파일링(시뮬레이션 기반)과 (b) 실제 U280 온보드 검증을 제공.
- **한줄요약**: LLM 추론용 **명령어(ISA) 기반 FPGA 오버레이 가속기**의 사이클-정확 성능 모델(Python)과 사전 컴파일된 비트스트림/명령어를 이용한 온보드 정확성 검증 데모.
- **원논문**: FlightLLM (Zeng et al., FPGA 2024). 본 repo는 논문의 Figure 1(55 token/s), Figure 12(LLaMA2-7B throughput speedup)를 재현하기 위한 artifact evaluation 패키지.
- **타깃 디바이스**:
  - 프로파일러: **Xilinx Versal VHK158** FPGA 가정(성능 모델 파라미터). GPU 베이스라인은 A100/V100.
  - 온보드: **Xilinx Alveo U280** (`xilinx_u280_gen3x16_xdma_1_202211_1`, XRT 2.15.225 / 2023.1).
- **근거**: `README.md` L1-9, `profile/README.md` L1-48, `fpga_implementation/README.md` L1-20.

## 2. 디렉토리 구조 (자체 소스 중심)
```
flightllm_test_demo/
├── README.md                       # 데모 전체 개요(2-part 구조)
├── profile/                        # [핵심] 성능 프로파일러 (자체 Python 소스)
│   ├── run.py                      # GPU(torch/vllm) vs FPGA(시뮬) 비교 엔트리포인트
│   ├── config.yaml                 # HW 파라미터(주파수/병렬도/버퍼/HBM) 전체 정의
│   ├── requirements.txt            # vllm==0.1.7, transformers==4.34.0, torch==2.0.1
│   ├── inst_gen/
│   │   ├── isa.py                  # [핵심] 6종 명령어 ISA 비트필드 정의 + 인코더
│   │   ├── inst_generator.py       # IR(YAML) → 명령어 시퀀스 생성, 레이어 퓨전
│   │   ├── inst_profiler.py        # [핵심] 의존성 기반 단일스레드 사이클 시뮬레이터
│   │   └── layer_support/          # 레이어 타입별 명령어 생성기
│   │       ├── IR_linear_mm.py / IR_linear_mv.py     # GEMM / GEMV 선형층
│   │       ├── IR_attention_mm.py / IR_attention_mv.py # attention QK^T/PV
│   │       ├── IR_misc.py          # softmax/layernorm/eltwise/silu
│   │       └── IR_concat.py        # concat
│   └── .compiler_output/
│       ├── ir_output/*.yaml        # 사전 생성된 LLaMA2-7B IR(토큰 길이별, 생성물)
│       └── ir_output/attention_mask/*.npy  # 희소 attention 마스크(생성물)
└── fpga_implementation/            # 온보드 검증(바이너리 + 호스트)
    ├── README.md
    ├── host/fpgaHost               # 사전 컴파일된 호스트 실행파일(소스 비공개)
    ├── bitstream/stc-v1.xclbin     # U280 비트스트림(바이너리, 생성물)
    └── case/.../{input,inst,param,output}/*.bin  # 채널별 입력/명령/가중치/골든(생성물)
```
**제외(third-party/생성물)**: `.compiler_output/ir_output/*.yaml`·`*.npy`(컴파일러 산출물), `*.xclbin`/`*.bin`(비트스트림·바이너리), `host/fpgaHost`(소스 비공개 바이너리). 본 데모에는 RTL/HLS 소스가 포함되어 있지 않음 — **하드웨어는 비트스트림 형태로만 제공**되고, 자체 분석 가능한 코드는 Python 프로파일러뿐임(확인됨).

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `inst_gen/isa.py` — 가속기 ISA 정의 (가장 중요)
FlightLLM의 핵심 설계 사상은 "**고정 오버레이 + 명령어 스트림**" 구조다. CGRA/시스토릭이 아니라 6종 명령어로 모든 LLM 연산을 표현한다.
- **명령어 6종**(`ALL_INST_TYPE`, L5): `LD, ST, MM, MV, MISC, SYS`. 순서가 곧 의존성 순서이며 바꾸면 안 됨(주석 명시).
- **opcode**(L6-13): LD=0001, ST=0010, MM=0011, MV=0100, MISC=0101, SYS=1111 (4비트).
- **명령어 길이**(L15-22, 32비트 워드 수): LD=7, ST=3, MM=4, MV=4, MISC=3, SYS=1. → 가변길이 VLIW형 명령어.
- **MISC 연산 6종**(L24-31): eltwise_add, eltwise_mul, softmax, layernorm, RMSlayernorm, silu. (LLaMA 계열에 맞춘 후처리 유닛.)
- **비트필드 정의**(`inst_set`, L33-123): 각 명령어를 `(line_num, start_bit, end_bit, 허용값)` 튜플로 정밀 정의.
  - **LD**(L36-53): `mode`(목적 버퍼 선택, 3bit), `hbm_channel_id`, `bank_addr`, 1D/2D/3D `stride`·`loop`(다차원 DMA), `zero_fill`(0패딩 만충 대역폭 쓰기), `1d_length`. 즉 LD는 **3중 중첩 루프 스트라이드 DMA** 엔진.
  - **ST**(L56-67): bank_addr→hbm_addr, `rs_col`(열 재배열 플래그).
  - **MM**(L70-85): `K`(K블록 수), `output_flag/bias_flag/relu_flag/sparse_flag`, 그리고 `bias/meta/fsb/A/B/out_start_addr`. **`meta`+`fsb`(Fine-grained Structured Block) 주소가 희소 GEMM을 위한 메타데이터**임을 알 수 있음.
  - **MV**(L88-103): MM과 동일 필드 구조(GEMV용, decode 단계).
  - **MISC**(L106-116): `operation_flag`(연산 종류), `mask_flag`(희소 attention 마스크), in_a/in_b/out 주소, K.
  - **SYS**(L119-123): wait/release만 — 레이어 경계 동기화 배리어.
- **검증 로직**(L126-158): 모든 비트가 정확히 1회씩 커버되는지(중복·누락 없음) 자동 assert. 견고한 ISA 정의 방식.
- **인코더**(`encode_inst` L162-186): 필드값을 비트 위치에 OR-shift로 패킹. 리틀엔디언 hex 출력(L241).
- **wait/release 메커니즘**(`wait_release_list_to_value` L188-203): 각 명령어가 "자신을 제외한 5개 모듈" 중 어떤 것을 기다리고(wait) 어떤 것에 신호를 보내는지(release)를 5비트 비트마스크로 인코딩. → **하드웨어 모듈 간 데이터 의존성을 명령어 레벨 세마포어로 처리**하는 구조(in-order issue, out-of-order 모듈 실행). 이것이 inst_profiler의 시뮬레이션 모델의 근거.
- **명령어 생성 헬퍼**(`generate_*_inst` L256-468): 각 명령어 타입별 고수준 래퍼. 특히 `generate_LD_inst`의 `all_bank_name`(L277-284): **6개 온칩 버퍼** = B buffer, meta buffer, bias buffer, FSB buffer, A buffer, global buffer. 시뮬레이션용 NOTE에 `cross_hbm_channel`, `parallel_channel_num`, `bandwidth_ratio`(INT8=1, INT4=2 — **비트폭에 따른 대역폭 증가** 모델링) 부착.

### 3.2 `inst_gen/inst_profiler.py` — 의존성 기반 사이클 시뮬레이터 (가장 중요)
`InstProfiler` 클래스가 명령어 리스트를 받아 **단일스레드로 6개 하드웨어 모듈의 병렬 실행을 시뮬레이션**한다.
- **연산별 시간 모델**(L25-99):
  - `time_LD`(L25-46): `total_data_B / bandwidth_B_s`. 동일채널/교차채널 효율(`HBM_BW_SAME_CHANNEL_EFFICIENCY=0.7`, `CROSS=0.35`), `bandwidth_ratio`(비트폭), `zero_fill` 시 버퍼 만대역폭 가정. → **메모리 바운드 모델링**.
  - `time_MM`(L59-68): `computation = MM_START_M_NUM(128) * K * FSB_BLOCK_SIZE(16) * MM_START_N_NUM(16)`, sparse면 `sparse_ratio` 곱, `/MM_PARALLEL_TOTAL` 후 `*CLOCK_TIME_MS`. → **컴퓨트 바운드 모델링, 희소성이 직접 사이클 감소**.
  - `time_MV`(L70-79): GEMV용, MV_START_N_NUM(512) 사용.
  - `time_MISC`(L81-94): softmax/layernorm은 "데이터를 두 번 통과"하므로 `*2`(L86-88). silu/eltwise는 1패스.
  - `time_SYS`(L96-99): 1 사이클.
- **correction_factor=3.2**(L21): 실측 대비 보정 계수 — 즉 이 모델은 이상적 사이클을 3.2로 나눠 실측에 맞춤(**모델의 한계/캘리브레이션 흔적**, L180).
- **핵심 스케줄링 루프**(`run` L145-303):
  1. 모든 명령어를 6개 모듈별 큐로 분리, 각 명령어의 (wait_id_list, release_id_list, time) 추출(L161-187).
  2. wait/release 카운트 일관성 검증(L189-195) — pub/rev 행렬.
  3. **이벤트 기반 시뮬레이션**(L224-282): 매 스텝마다 "의존성이 만족되면서 시작시간이 가장 빠른" 명령어를 선택해 실행, hardware_time(모듈별 이전 명령 종료시각)과 wr_array(release 시각) 갱신. → **모듈 간 파이프라인 오버랩을 정확히 모델링**(LD/ST/MM/MV/MISC가 동시 진행).
  4. 총 시간 = `max(inst_hardware_time)`(L284), 모듈별 점유시간/비율을 CSV로 출력(L286-298).
- **의의**: RTL 없이도 명령어 스트림만으로 throughput을 예측하는 **analytical+event-driven 하이브리드 성능 모델**. 우리 프로젝트의 HW 설계 전 성능 추정에 직접 차용 가능한 패턴.

### 3.3 `inst_gen/inst_generator.py` — IR→명령어 컴파일 + 레이어 퓨전
- **지원 레이어**(`ALL_HW_LAYER_TYPE` L12): linear_mm/mv, attention_mm/mv, eltwise, layernorm, softmax, concat, silu, input, output.
- **레이어 퓨전**(L61-118, 핵심 최적화):
  - attention_mm/mv + 후속 **softmax 퓨전**(L62-69): QK^T 결과를 HBM에 안 쓰고 바로 softmax.
  - linear_mm/mv + **silu 퓨전**(L80-83) 또는 **eltwise 퓨전**(L84-97). 특히 연속 eltwise가 RoPE면 3개까지 연속 융합(`fuse_layer_num in (1,3)`, L96) — **Q,K RoPE 융합**.
- 레이어 경계마다 SYS 명령으로 wait/release 동기화(L55-58).
- 각 레이어는 `layer_support/IR_*.py`로 위임(MM/MV 타일링, 버퍼 주소 할당, 명령어 시퀀스 생성).

### 3.4 `config.yaml` — 하드웨어 구성 (아키텍처 파라미터의 단일 소스)
- **주파수**: `CLOCK_FREQ_MHZ=225`. **SLR_NUM=4**(Versal 다이 4분할).
- **MM 병렬도**(L18-21): `MM_PARALLEL_N=2, K=16, M=128`. → MM 한 사이클 throughput 관련. MV: N=64, K=16, M=1(L23-25, 벡터 한 행).
- **희소 블록**: `FSB_BLOCK_SIZE=16`, `MASK_LAYOUT_BLOCK_SIZE=64` — FlightLLM의 "configurable sparse DSP chain"의 블록 단위.
- **온칩 버퍼 계층**(L37-81): A buffer(다뱅크, 활성값), B buffer(가중치, 1뱅크), global buffer(A와 동형), meta/FSB/bias buffer(희소 메타데이터·바이어스). HBM 32채널, 819.2GB/s.
- **이 파일은 RTL이 없는 본 데모에서 "하드웨어 사양의 명세서" 역할** — 병렬도·버퍼 크기·대역폭이 모두 명시되어 우리 설계의 파라미터 비교 기준이 됨.

### 3.5 `profile/run.py` — 엔트리포인트 / GPU 베이스라인
- `test_case_list`(L16-25): (batch, prefill, decode) 6케이스. 모두 batch=1.
- `align_token_length`(L28-35): prefill은 128, decode는 16 단위로 정렬 → **하드웨어 타일 정렬 요구**.
- `profile_case_list_on_fpga`(L66-112): prefill 시간 + decode 시간(토큰마다 align된 길이별 캐싱하여 누적). FPGA 측은 모두 사전 IR→명령어→시뮬레이션.
- `profile_case_list_on_gpu`(L115-195): torch(naive)·vllm(opt) 두 백엔드로 prefill/decode latency 측정. CUDA Event 타이밍, warmup=5, freq=10.
- 결과 테이블 출력(L205-213): GPU-naive / GPU-opt / FPGA throughput·speedup.

## 4. 데이터플로우 / 실행 흐름
1. **컴파일(사전 수행, 산출물 제공)**: LLaMA2-7B → 그래프 IR(YAML) + 희소 attention 마스크(npy).
2. **명령어 생성**(`InstGenerator.run`): IR 레이어 순회 → 퓨전 판단 → `layer_support`가 타일링·버퍼 할당·LD/MM/MV/MISC/ST/SYS 시퀀스 생성. wait/release로 모듈 의존성 인코딩.
3. **성능 시뮬레이션**(`InstProfiler.run`): 명령어별 시간 = analytical 모델, 전체 시간 = 의존성 만족 이벤트 스케줄링으로 모듈 병렬 오버랩 반영, /correction_factor 보정.
4. **온보드(별도)**: `fpgaHost`가 xclbin 로드 → 채널별 inst/param/input 바이너리를 HBM에 적재 → 실행 → output을 골든과 비교.
- **메모리 계층**: HBM(32ch) ↔ 온칩 6버퍼 ↔ MM/MV/MISC 연산 유닛. LD 명령의 다차원 stride로 HBM→버퍼 전송, A/B 분리(활성/가중치).
- **병렬화**: SLR 4분할 + MM(128×16×2)/MV(64×16) 공간 병렬. 모듈 간 파이프라인(LD/ST/MM/MV/MISC 동시).
- **양자화/데이터타입**: INT8 기본, INT4 지원(`bandwidth_ratio` INT4=2). 가중치 희소(FSB 블록, sparse_ratio). decode는 MV(GEMV), prefill은 MM(GEMM).

## 5. HW/SW 매핑
- **SW(Python)**: 컴파일러(IR→명령어, `inst_generator`/`layer_support`), 성능 모델(`inst_profiler`), ISA 인코더(`isa.py`).
- **명령어 스트림(중간층)**: 6종 ISA, wait/release 동기화 — SW가 HW를 제어하는 인터페이스. 바이너리(`*.bin`)로 HBM에 적재.
- **HW(RTL, 본 데모엔 비공개)**: xclbin으로만 제공. 명령어를 해석하는 6개 모듈(LD/ST DMA, MM/MV 시스토릭/DSP 체인, MISC 후처리, SYS 동기화)과 6개 온칩 버퍼로 구성된 것이 config·ISA로부터 역추론됨.
- **호스트(C++)**: `fpgaHost`(소스 비공개) — XRT API로 xclbin·데이터 적재 및 검증.

## 6. 빌드·실행 방법
- **프로파일러**: `conda create -n fpga python==3.11`; `pip install -r profile/requirements.txt`(vllm 0.1.7, transformers 4.34.0, torch 2.0.1); `cd profile && python run.py`. GPU(A100/V100, CUDA≥11.7) 필요. 사전 생성된 profiler_output 있으면 재사용, 지우면 재시뮬레이션.
- **온보드**: U280 + XRT 2023.1 환경에서 `chmod +x host/fpgaHost; host/fpgaHost bitstream/stc-v1.xclbin case/decode_token_512_ae` → 골든 비교(TEST OK).

## 7. 의존성
- Python 3.11, PyTorch 2.0.1, transformers 4.34.0, **vLLM 0.1.7**(GPU-opt 베이스라인), numpy, pyyaml.
- 온보드: Xilinx XRT 2.15.225(2023.1), U280 플랫폼.

## 8. 강점 / 한계 / 리스크
- **강점**:
  - 명령어 기반 오버레이라 **재컴파일 없이** 다양한 토큰 길이·모델을 같은 비트스트림으로 실행 — 우리 ViT 가속기의 유연성 설계에 시사점.
  - **희소성(FSB)·저비트(INT4)를 ISA·성능모델에 1급 시민으로** 반영.
  - 의존성 기반 이벤트 시뮬레이터가 RTL 없이 throughput을 예측 — 빠른 DSE 가능.
  - 레이어 퓨전(softmax/silu/RoPE)으로 HBM 트래픽 감소.
- **한계/리스크**:
  - **RTL/HLS 소스 미포함**: 하드웨어 마이크로아키텍처는 비트스트림·ISA·config로 역추론만 가능(확인됨, 직접 분석 불가).
  - 성능 모델에 `correction_factor=3.2`라는 경험적 보정이 들어가 절대치 신뢰도는 제한적(추정).
  - LLaMA(디코더 전용 LLM) 특화 — ViT/인코더 구조엔 직접 적용 불가, attention/MISC 연산 매핑 재설계 필요(추정).
  - vLLM 0.1.7 등 구버전 의존으로 재현 환경 구축 난이도 높음.

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점
- **재사용 가능한 설계 패턴**:
  1. **명령어 기반 오버레이 + wait/release 세마포어 동기화**: HG-PIPE류 고정 파이프라인과 달리, 같은 비트스트림으로 여러 ViT 변형/시퀀스 길이를 돌리려면 본 ISA 구조가 유용. 특히 `isa.py`의 비트필드 정의·검증 패턴은 우리 ISA 정의에 그대로 차용 가능.
  2. **의존성 기반 사이클 시뮬레이터(`inst_profiler.py`)**: 우리 가속기의 RTL 작성 전 throughput/병목(어느 모듈이 점유율 높은지) 분석에 직접 이식 가능. 모듈별 analytical time + 이벤트 스케줄링 구조가 핵심.
  3. **레이어 퓨전 컴파일러 로직(`inst_generator.py`)**: ViT의 LayerNorm+GELU+MHSA 융합 판단에 동형 적용. softmax 퓨전(QK^T 결과 비저장)은 ViT attention에 그대로 적용 가능.
  4. **`config.yaml` 단일 파라미터 소스**: 병렬도·버퍼·HBM을 한 곳에서 정의하고 SW/시뮬레이터가 공유하는 방식은 우리 DSE에 유용.
- **차용 주의**: LLM(GEMV decode 지배) vs ViT(GEMM 지배·고정 시퀀스). XR 시선추적은 저지연·소형 ViT라 prefill-only·GEMM 중심이므로, MV 경로보다 MM 경로·퓨전·희소 마스크 처리가 더 관련 깊음(추정).

## 10. 근거 / 한계 표기
- ISA/시뮬레이터/컴파일러/config 분석은 모두 실제 소스(`isa.py`, `inst_profiler.py`, `inst_generator.py`, `config.yaml`, `run.py`)의 라인 근거에 기반(확인됨).
- 하드웨어 마이크로아키텍처(MM/MV 유닛 내부 구조, DSP 체인 구성)는 **RTL 미공개로 직접 확인 불가**, ISA·config·논문 제목으로부터의 **역추론(추정)**임을 명시.
- `fpgaHost`·`*.xclbin`·`*.bin`은 바이너리이므로 내부 동작은 README 기재 사항 외 확인 불가.
- "우리 프로젝트" 성격(HG-PIPE 계열 ViT 가속기 + XR 시선추적)은 과제 지시에 따른 **추정 전제**.
