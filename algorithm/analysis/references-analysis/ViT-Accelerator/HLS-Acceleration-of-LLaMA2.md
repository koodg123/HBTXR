# HLS-Acceleration-of-LLaMA2 정밀 분석

## 1. 개요
- **한 줄 요약**: Karpathy의 `llama2.c` TinyLlama(stories15M) 추론 **전체 forward pass를 단일 Vitis HLS 커널**(`kernel_forward`)로 구현하고, ZCU106 보드에서 PS 대비 약 5배 가속을 보인 LLaMA2 FPGA 가속 PoC.
- **목적**: float32 LLaMA2 디코더 1토큰 forward(임베딩→N층 디코더→최종 logits)를 통째로 가속기로 오프로드.
- **출처**: README. 원 알고리즘은 Andrej Karpathy의 llama2.c(stories15M.bin). 데모 영상(youtu.be/Mrl_r8l6aDE) 명시.
- **타깃 디바이스**: AMD Zynq UltraScale+ MPSoC **ZCU106**, Vitis/Vivado 2024.2, PetaLinux 2024.2.

## 2. 디렉토리 구조 (자체 소스)
```
HLS-Acceleration-of-LLaMA2/
├── README.md          # 빌드/배포/실행/벤치 결과
├── kernel_forward.cpp # ★ HLS 커널: LLaMA2 전체 forward (핵심)
└── main.cpp           # XRT 호스트 (가중치 로드·토크나이저·생성 루프)
```
- **제외 third-party**: 없음. (모델 가중치 stories15M.bin, tokenizer.bin은 데이터로 분리)

## 3. 핵심 모듈·파일별 정밀 분석 (kernel_forward.cpp)

### 3.0 모델 상수 (kernel_forward.cpp:3~13)
stories15M 고정 파라미터: `P_DIM=288`, `P_HIDDEN_DIM=768`, `P_N_LAYERS=6`, `P_N_HEADS=6`, `P_VOCAB_SIZE=32000`, `P_SEQ_LEN=256`, `P_HEAD_SIZE=48`, `P_KV_DIM=288`. 역수 상수 `INV_P_DIM`, `INV_SQRT_P_HEAD_SIZE`를 미리 계산해 나눗셈 제거.

### 3.1 rmsnorm() (kernel_forward.cpp:15~44)
- LLaMA RMSNorm: 가중치 `w`를 로컬 `W[P_DIM]`로 캐시(cyclic factor=32 파티션, UNROLL 8) → `ss=Σx²·INV_P_DIM+1e-5` → `hls::rsqrt` → `o[i]=W[i]·(ss·x[i])`.
- **핵심 HW 기법**: 누산을 `partial[4]` 4-way 부분합으로 분할(kernel_forward.cpp:25~34)하여 누산 종속성 체인을 끊어 II=1 파이프라인 가능하게 함. (hls-fpga-accelerators의 단일 누산기 대비 개선된 관용구)

### 3.2 softmax() (kernel_forward.cpp:46~78)
- **수치안정 3-pass**: ① max 탐색 → ② `exp(x-max)` 및 partial[4] 합 → ③ `1/sum` 곱. hls-fpga-accelerators softmax와 달리 **max-subtraction을 포함**(저장소 간 비교 포인트). `hls::expf` 표준 사용.

### 3.3 matmul_* 계열 (kernel_forward.cpp:80~168)
- 형상별로 5개 전용 함수: `matmul_dim_dim`(288×288), `matmul_dim_kvdim`, `matmul_dim_hiddendim`(768×288), `matmul_hiddendim_dim`, `matmul_dim_vocabsize`(32000×288).
- 공통 패턴: 외부 출력행 루프(loop_flatten off) × 내부 reduction 루프(PIPELINE II=1) + `partial[4]` 부분합. 가중치는 행우선(`w[i*K+j]`) 매 호출 m_axi 스트리밍(상주 캐시 없음).

### 3.4 RoPE() (kernel_forward.cpp:170~210)
- 회전 위치 임베딩. 미리 계산된 `TABLE`(cos/sin쌍 fcr/fci)을 받아 Q,K의 인접 2원소를 복소 회전. `i<P_KV_DIM`이면 Q·K 모두, 아니면 Q만 회전(GQA/부분KV 대응). UNROLL factor=16로 쌍 단위 병렬.

### 3.5 attention() (kernel_forward.cpp:212~274)
- 헤드별 루프: ① Q·K 내적(score)을 partial[4]로, `INV_SQRT_P_HEAD_SIZE` 스케일 → ② `softmax(att, pos+1)` → ③ att·V 누산(`acc[P_HEAD_SIZE]`, PIPELINE) → xb write-back.
- **KV-cache 명시 사용**: `S_key_cache`/`S_value_cache`를 `loff+h*HEAD_SIZE+t*KV_DIM`으로 인덱싱 — autoregressive 디코딩의 표준 캐시 패턴.

### 3.6 residual() / SwiGLU() (kernel_forward.cpp:277~292)
- residual: `S_x += S_xb` (UNROLL 32). SwiGLU: `hb·sigmoid(hb)·hb2` (SiLU·gate, hls::expf로 sigmoid).

### 3.7 kernel_forward() top (kernel_forward.cpp:294~386)
- **인터페이스**: 12개 가중치 포인터 + KV캐시 + logits + table을 m_axi로, gmem0~3 4뱅크에 분산 배치(대역폭 분산), 모든 스칼라 s_axilite(control 번들), `max_read_burst_length=64`.
- 모든 중간 상태(S_x, S_xb, S_q, S_hb…)를 on-chip 배열로 두고 `ARRAY_PARTITION cyclic factor=32`로 32-way 동시접근.
- **실행 흐름**(kernel_forward.cpp:344~385): token embedding → table 로드 → `layer` 루프 6회{rmsnorm→Q/K/V matmul→RoPE→attention→Wo matmul→residual→rmsnorm→W1/W3 matmul→SwiGLU→W2 matmul→residual} → 최종 rmsnorm → classifier matmul(logits).

## 4. 데이터플로우 / 실행 흐름
- 호스트(main.cpp)가 토큰마다 `kernel_forward(pos, token, …)` 호출 → 커널이 1토큰 logits 반환 → 호스트가 샘플링/argmax로 다음 토큰 생성(autoregressive). KV캐시는 글로벌 메모리 상주(커널이 in-place 갱신).
- **병렬화**: 레이어 간은 순차(데이터 종속). 레이어 내부는 (a) reduction partial[4] 분할, (b) ARRAY_PARTITION factor=32 + UNROLL/PIPELINE으로 ILP 확보. dataflow(태스크 병렬)는 사용 안 함 — 단일 큰 순차 커널.
- **데이터타입**: 전부 float32(양자화 없음). gmem 4뱅크 분산 + burst=64로 메모리 대역폭 최적화.

## 5. HW/SW 매핑
- **SW(main.cpp, 호스트/PS)**: 가중치 bin 로드, 토크나이저, 샘플링, XRT 버퍼 관리, 커널 enqueue.
- **HW(kernel_forward.cpp, PL)**: forward 연산 전부. 경계는 XRT m_axi(가중치/KV/logits) + s_axilite(pos/token/포인터). RTL 수작업 없음(HLS 단일 계층).
- 벤치(README:73~78): ZCU106 PS-only 1.75 tok/s → 가속 8.71 tok/s(~5x).

## 6. 빌드·실행
- Vitis 2024.2로 커널을 `binary_container_1.bin`(xclbin)으로 빌드, 호스트 `Llama2_host` 컴파일. SD카드에 호스트/xclbin/가중치/토크나이저 복사 후 PetaLinux 부팅, `./Llama2_host stories15M.bin` 실행 (README:19~69).

## 7. 의존성
- Vitis/Vivado/PetaLinux 2024.2, XRT, `hls_math.h`. 원본 llama2.c 모델 포맷. 외부 ML 프레임워크 불필요(추론만).

## 8. 강점 / 한계 / 리스크
- **강점**: 실제 보드(ZCU106)에서 end-to-end 동작·측정된 드문 사례. RMSNorm/Softmax/MatMul/RoPE/Attention/SwiGLU 전 연산자의 HLS 구현이 한 파일에 모여 있어 LLaMA 디코더 가속 학습용 레퍼런스로 우수. partial[4] 누산 분할, ARRAY_PARTITION cyclic, 4뱅크 분산 등 실전 HLS 관용구가 풍부.
- **한계/리스크**:
  - float32 고정(양자화 미적용) → 면적/대역폭 비효율, tok/s 절대값 낮음(8.71).
  - 가중치 비상주(매 matmul 글로벌 메모리 재독) → 메모리 바운드. 레이어 융합/타일링/dataflow 없음.
  - stories15M 형상 하드코딩(P_* 상수) → 다른 모델로 일반화하려면 재작성.

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA, HG-PIPE 계열) 관점 시사점
- **재사용 아이디어**:
  - **수치안정 softmax(max-subtract + partial 합)** 패턴을 ViT MHSA softmax에 그대로 채용.
  - **partial[N] 부분합 누산 분할** 관용구는 모든 reduction(LayerNorm, dot-product)에 적용해 II=1 달성 — 우리 PE/누산 설계의 기본기.
  - **gmem 다뱅크 분산 + burst** 설정은 HBM 대역폭 활용 템플릿.
  - RoPE/KV-cache 구현은 (디코더 LLM을 다룰 경우) 직접 참조 가능.
- **반면교사**: 단일 거대 순차 커널 + 가중치 비상주는 고처리량과 반대 방향. 우리는 레이어 파이프라인(dataflow)·상주 가중치·양자화로 가야 함. 즉 "정확성 레퍼런스"로는 훌륭하나 "처리량 아키텍처"로는 부적합.

## 10. 근거/한계 표기
- 근거: kernel_forward.cpp 전문, README 직접 확인.
- main.cpp(호스트) 내부는 미열람 → 샘플링/토크나이저 세부는 **확인 불가**(README 기반 추정).
- 합성 리소스(LUT/DSP/BRAM) 리포트 부재 → **확인 불가**.
