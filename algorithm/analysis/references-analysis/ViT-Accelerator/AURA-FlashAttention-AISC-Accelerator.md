# AURA — FlashAttention ASIC Accelerator 정밀 분석

> 분석 대상 repo: `REF/ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator`
> 분석 방식: SystemVerilog RTL / C++ 참조모델 / Python QKV 생성 스크립트를 라인 단위로 직접 읽고 근거 표기.
> 작성일: 2026-06-20

---

## 1. 개요

- **무엇인가**: AURA(A.U.R.A.)는 Transformer의 **FlashAttention 커널을 전용 RTL로 구현한 ASIC 가속기**다. SystemVerilog로 작성되었고, Synopsys Design Compiler(`dc_shell`) 기반 250nm 표준셀 합성 흐름을 갖춘다. (`README.md:1-2`, `synth/AURAsynth.tcl:96`, `Makefile:15`)
- **한 줄 요약**: 8비트 고정소수점 Q/K/V 입력에 대해 **online-softmax(running max/sum) 기반 streaming attention**을 수행하고, exp 계산을 곱셈 없이 시프트로 근사(ExpMul)하며, 4개 PE 병렬 + 트리 리덕션 + 핸드셰이크 파이프라인으로 동작하는 edge 지향 attention ASIC.
- **출처/논문**: 별도 논문 PDF는 repo에 없음(확인 불가). README가 선행연구로 **SwiftTron**(tinyML attention ASIC)을 명시적으로 언급하고, 본 설계의 차별점으로 **FlashAttention + ExpMul + FLASH-D** 같은 HW-알고리즘 co-optimization 채택을 든다. (`README.md:9-11`) ExpMul의 핵심 근사 공식(`expmul_stage.sv:14` 주석의 `Log2Exp(X) = −⌊X + (X≫1) − (X̂≫4)⌉`)은 ExpMul 류 연구에서 온 것으로 **추정**된다(원논문 인용 없음).
- **타깃 디바이스 / 합성 타깃**: **ASIC**. EECS 470/570 수업 인프라에 맞춰진 합성 스크립트로, target library `lec25dscc25_TT.db`(250nm 표준셀), wire load model `tsmcwire`를 사용한다. (`synth/AURAsynth.tcl:96`, `:106`, `:171`) 클럭 주기 기본값은 `CLOCK_PERIOD = 10`(ns). (`Makefile:2`) → **FPGA가 아니라 표준셀 ASIC 합성**을 전제로 한다. (이 점이 우리 FPGA 프로젝트와의 가장 큰 차이.)
- **설계 규모 파라미터**(`include/sys_defs.svh`):
  - `INTEGER_WIDTH = 8` (Q/K/V/O는 INT8 = Q0.7 고정소수점) (`:22`, `:87-94`)
  - `MAX_EMBEDDING_DIM = 64` (head dim dk=64) (`:24`)
  - `MAX_SEQ_LENGTH = 512` (`:27`)
  - `NUM_PES = 4` (병렬 PE 수), `NUM_TILES = 512/4 = 128` (`:36`, `:38`)
  - 메모리 블록 64bit(8 elem/cycle) (`:28`, `:248`)

---

## 2. 디렉토리 구조

### 2.1 자체 소스 트리 (분석 대상)

```
AURA-FlashAttention-AISC-Accelerator/
├── README.md                  # 문제정의/동기/선행연구(SwiftTron 등)
├── Makefile                   # VCS 시뮬, dc_shell 합성, C++/Python 빌드, 테스트벡터 생성
├── include/
│   └── sys_defs.svh           # ★ 전역 파라미터 + 고정소수점 Q-format typedef (설계의 두뇌)
├── verilog/                   # ★ RTL 본체
│   ├── AURA.sv                # top: mem_ctrl + 4 SRAM + NUM_PES개 PE 생성
│   ├── PE.sv                  # 단일 처리요소: dot→max→expmul→division 파이프라인
│   ├── dot_product.sv         # Q·K 내적 + root(dk) 스케일 + V 지연 정합
│   ├── tree_reduce.sv         # 파라미터화 트리 합산 (stage 체인)
│   ├── reduction_step.sv      # 트리 한 스테이지(쌍 합산, sign-extend)
│   ├── max.sv                 # running max (online-softmax의 m_i)
│   ├── expmul.sv              # ExpMul 상위: O*와 V* 두 스테이지 + running 누적
│   ├── expmul_stage.sv        # ★ exp 근사(시프트만) + 2^-L 배럴시프트 곱셈
│   ├── vector_division.sv     # exp_o / exp_sum 원소별 나눗셈 (최종 정규화)
│   ├── int_division.sv        # 부호처리 + 비복원형 unsigned divu 반복 나눗셈
│   ├── memory_controller.sv   # ★ K→V→Q load / O drain FSM, mem tag FIFO
│   ├── KSRAM.sv / VSRAM.sv    # K/V FIFO 버퍼 (전체 시퀀스 저장, 반복 read)
│   ├── QSRAM.sv               # Q ping-pong dual-bank (타일 단위)
│   ├── OSRAM.sv               # O ping-pong dual-bank (drain용)
│   ├── q_convert.sv           # ★ Qm.n → Qp.q 변환 래퍼 (frac정렬→int정렬)
│   ├── q_align_frac.sv        # 소수부 정렬 (좌/우 시프트 + 라운딩)
│   ├── q_align_int.sv         # 정수부 정렬 (sign-extend / saturate 분기)
│   ├── q_saturate.sv          # 포화 클리핑 (MAX/MIN signed)
│   └── q_sign_extend.sv       # 부호확장
├── cpp/                       # SW 참조모델 / 양자화 / 정밀도 측정
│   ├── attention_fp32.cpp     # FP32 골든 attention (online-softmax)
│   ├── attention_f8.cpp       # INT8 입력 attention (FP 누적, 결과 재양자화)
│   ├── fp32_to_f8.cpp         # FP32 .mem → INT8(Q0.7) .mem 양자화
│   ├── fp32_to_f16.cpp        # FP32 → f16 변환 변형
│   ├── input_to_f8.cpp / output_to_f8.cpp     # 입출력 f8 변환 보조
│   ├── generate_mem.cpp / generate_output_f8.cpp / generate_output_fp64.cpp
│   ├── precision_measure.cpp  # ASIC 출력 vs 골든 MAE/RMSE/Max/rel/Top-1 비교
│   └── precision_measuref16.cpp
├── python/
│   ├── Generate_QKV.py        # ★ HuggingFace BERT에서 1개 head의 Q/K/V 추출→FP32 .mem
│   ├── QKV_obtain.py          # 위와 거의 동일(구버전 변형으로 추정)
│   ├── strip_out_file.py      # 시뮬 출력 정리(O_cleaned.mem)
│   ├── convert_to_decimal.py / convert_to_dec.py  # hex→decimal 보조
├── test/                      # 테스트벤치
│   ├── aura_test.sv           # 통합 테스트벤치
│   ├── mem.sv                 # 지연 모델 가진 단일 통합 메모리 모델(BE/100ns 지연)
│   ├── dot_product_test.sv / expmul_test.sv / int_division_test.sv / q_convert_test.sv
└── synth/
    └── AURAsynth.tcl          # Synopsys DC 합성 스크립트(EECS 470 템플릿 기반)
```

### 2.2 분석 제외 (이름만 언급)

- **`_deprecated/`** — 구버전/실험 RTL 폴더. `memory_controller_old.sv`, `memctrlV2.sv`, `dot_product.sv`, `int2_division.sv`, `int_div_comb.sv`, `expmul_comb.sv`, `handshake_reg.sv`, `multiply.sv`, `vec_add.sv`, `math_utils_pkg.sv`, `SRAM_DB_example.sv`, `SRAM_ARRAY.sv`, `SRAM_FIFO.sv` 등. **현 빌드(`Makefile` `AURA_SOURCES`)에서 컴파일되지 않음** → 분석 제외.
- **`models/*.mem`, `models/*.dec`, `models/*.dump`** — Makefile이 자동 생성하는 테스트 벡터/출력 데이터 파일. 현재 repo에는 생성 산출물 없음(Glob `models/**` 결과 없음). 데이터 파일이므로 분석 제외.
- **`test/__pycache__`, `python/__pycache__`** — Python 캐시. 제외.
- **표준셀 라이브러리** `lec25dscc25_TT.db`, `lec25dscc25.v` — 외부 vendor/cell 라이브러리(`Makefile:15`, `synth/AURAsynth.tcl:96`). 분석 제외.

---

## 3. 핵심 모듈·파일별 정밀 분석 (★ 가장 중요)

### 3.0 설계의 두뇌: `include/sys_defs.svh` — 고정소수점 타입 체계

이 헤더가 데이터패스 전체의 비트폭을 **수식으로 연쇄 정의**한다. 핵심:

- **입력/출력 양자화**: `INPUT_VEC_I=0`, `INPUT_VEC_F = INTEGER_WIDTH-1 = 7` → **Q0.7 (INT8, 부호 포함 8비트)**. (`:87-89`, `:92-94`) 즉 Q/K/V/O 원소는 [-1, 1) 범위 8비트 고정소수점.
- **Q-format 헬퍼 매크로**: `Q_WIDTH(M,N) = M+N+1`(+1 부호), `Q_TYPE(M,N)` = `logic signed`. (`:63-66`)
- **중간 비트폭이 전방 의존으로 자동 확장**된다. 곱(`PRODUCT`), 내적합(`DOT`), 스코어(`SCORE`), exp 차이(`EXPMUL_DIFF`), exp 누적벡터(`EXPMUL_VEC = Q9.17`), 분모/분자(`DIV_INPUT`) 등이 모두 `$clog2(MAX_SEQ_LENGTH)`, `$clog2(MAX_EMBEDDING_DIM)`에 의존해 계산된다. (`:101-131`) 예: `EXPMUL_VEC_F = clog2(512)+7+1 = 17`, `EXPMUL_VEC_I = EXPMUL_SHIFT_STAGE_I = clog2(512)+0 = 9` → **Q9.17 누적 벡터**(`:103`, `:126-129`, `:146-147`).
- **I/O 타입**(`:197-207`): `Q_VECTOR_T`/`K`/`V`는 `INPUT_VEC_QT[0:63]`(64 elem), `STAR_VECTOR_T`는 `EXPMUL_VEC_QT[0:64]`(**65 elem** — index 0이 softmax 분모(exp sum), 1..64가 출력 누적). 이 "분모를 벡터 0번에 동봉"하는 트릭이 FLASH-D식 단일 패스 정규화의 핵심 자료구조다.

### 3.1 `AURA.sv` — Top 모듈

- 포트: 470 템플릿식 메모리 인터페이스(`mem2proc_*`, `proc2mem_*`)와 `done`. (`AURA.sv:6-21`)
- 구성: `memory_controller` 1개 + `QSRAM/KSRAM/VSRAM/OSRAM` 각 1개 + **`generate`로 `NUM_PES`개 `PE` 인스턴스**. (`AURA.sv:57-161`)
- **데이터 분배 구조**: `QSRAM`은 `q_vectors[0:NUM_PES-1]` 배열을 한 번에 출력 → PE i가 `q_vectors[i]`를 받는다(각 PE = 서로 다른 Q row). `KSRAM/VSRAM`은 단일 `k_vector`/`v_vector`를 **모든 PE에 브로드캐스트**(`AURA.sv:155-157`). 즉 K/V는 공유, Q는 분배 → 한 K/V 벡터에 대해 4개 Q row를 동시 처리하는 **output-stationary 스타일 타일링**.
- 핸드셰이크: SRAM-PE 간 `*_rdy[i]`/`*_vld`, PE-OSRAM 간 `O_vld[i]`/`O_sram_rdy`. PE들이 lockstep이라 가정하여 `Q_rdy[0]`처럼 0번만 검사. (`AURA.sv:88-91` 주석)

### 3.2 `PE.sv` — 단일 처리요소 (FlashAttention 1행 파이프라인)

PE는 한 Q row에 대한 전체 attention을 **valid/ready 핸드셰이크로 연결된 4단 스트리밍 파이프라인**으로 수행한다. (`PE.sv:60-119`)

```
dot_product → max → expmul → vector_division
   (s, v지연)  (running max)  (exp근사·누적)  (정규화)
```

- 단계별 신호: `dot_product_valid → max_ready → expmul_ready → vector_division_ready → O_sram_rdy`로 backpressure가 거꾸로 전파(`PE.sv:67,82,98,114`). 전형적 elastic pipeline.
- **`v_star` 생성**(`PE.sv:44-58`): `v_star[0] = 1<<EXPMUL_VEC_F`(상수 1.0, Q9.17) → 분모 누적 채널. `v_star[1..64]`는 V 원소를 `q_convert`로 Q0.7→Q9.17 확장. 이 65-원소 벡터가 expmul에 들어가 "출력 누적 + 분모 누적"을 **동일 하드웨어로 동시 계산**한다.
- 주의: `expmul`의 `o_star_prev_in('0)`이 0으로 묶임(`PE.sv:104`). 즉 PE 내부에서 running 누적은 expmul 모듈 내부 레지스터(`kv_counter`)로 처리하고, 외부 prev 입력은 미사용 — 시퀀스 누적을 한 PE가 streaming으로 처리하는 구조.

### 3.3 `dot_product.sv` — QK^T 내적 + 스케일

- **3개 독립 입력 래치**(Q/K/V 각각 valid 추적). `row_counter`로 한 번 로드된 Q를 `MAX_SEQ_LENGTH`개 K/V에 대해 재사용. (`dot_product.sv:44-46`, `:55-70`) → Q는 한 번 받고 K/V는 매 사이클 새로 받는 streaming.
- 곱: `intermediate_products[i] = q[i]*k[i]` (Q1.14), 이후 `q_convert`로 Q1.12로 축소(`PRODUCT`). (`:124-143`)
- **트리 리덕션**: `tree_reduce #(.STAGES(NUM_REDUCE_STAGES))`로 64개 곱 합산(`:146-159`). `NUM_REDUCE_STAGES = clog2(64)/2 = 3`(`sys_defs.svh:54`), `REDUCTIONS_PER_STAGE=2` → 한 스테이지당 4:1 축소(64→16→4→1).
- **root(dk) 스케일**: `assign shifted_sum = sum >>> 3;` (`:162`) dk=64 → 1/sqrt(64)=1/8 = `>>3`. 주석에 명시(`:1-3`). 그 후 `q_convert`로 `EXPMUL_DIFF_IN`(Q4.4)로 변환해 `s_out` 생성(`:163-171`).
- **V 지연 정합**: 트리 리덕션의 `NUM_REDUCE_STAGES` 사이클 latency만큼 V를 `v_pipe[]` 시프트 레지스터로 지연시켜 score와 정렬(`:174-200`). 데이터패스 타이밍 정합의 전형.

### 3.4 `tree_reduce.sv` / `reduction_step.sv` — 파라미터화 가산 트리

- `tree_reduce`: `STAGES`개 `reduction_step`을 `generate`로 체인 연결, 스테이지마다 입력 길이 `LEN>>(s*RPS)`, 비트폭 `W_IN+s*RPS`로 자동 확장(`tree_reduce.sv:42-82`). 각 스테이지가 valid/ready로 분리되어 **파이프라인 레지스터 역할**도 겸함(스테이지당 1사이클).
- `reduction_step`: `STEPS`번 쌍 합산을 **조합 루프**로 수행하되 입력은 레지스터로 래치. 핵심은 `stageN[i] = {{(W_OUT-W_IN){list[i][W_IN-1]}}, list[i]}` 부호확장 후 `temp[i]=stageN[2i]+stageN[2i+1]` 누적(`reduction_step.sv:67-93`). 한 스테이지에서 `2^STEPS:1` 축소.
- 설계 의의: **임베딩 차원·시퀀스 길이가 바뀌어도 합산 트리가 자동 재구성**된다. 우리 FPGA 프로젝트의 LayerNorm/softmax 리덕션 트리에 그대로 차용 가능한 패턴(섹션 9 참조).

### 3.5 `max.sv` — Online-softmax의 running max (m_i)

- 한 사이클 래치 + 조합 비교. `m_out = (s > m_prev) ? s : m_prev`(`max.sv:56-61`).
- **online-softmax 키 포인트**: `m_prev <= (row_counter == 0) ? '0 : m_prev_in;`(`max.sv:45`). 한 Q row의 시퀀스 첫 원소(`row_counter==0`)에서는 prev max를 0으로 리셋, 이후엔 직전 running max를 이어받음. 즉 **각 K 원소를 볼 때마다 running max를 갱신**하는 FlashAttention의 m_i = max(m_{i-1}, s_i) 패턴. (`row_counter`는 매 핸드셰이크에서 감소, `:48`)
- 출력으로 `m_out`(현 max), `s_out`(현 score 지연), `m_prev_out`(직전 max)을 모두 expmul로 넘김 → 다음 단계가 **rescaling factor 2^(m_prev - m_cur)**을 계산할 수 있게 한다.

### 3.6 `expmul.sv` + `expmul_stage.sv` — ★ ExpMul: 곱셈 없는 exp 근사 + running 누적

이 두 모듈이 본 가속기의 가장 독창적 부분이다.

**(a) `expmul.sv` (상위)** — 두 개의 `expmul_stage`를 병렬 구동(`expmul.sv:50-78`):
- `expmul_o_inst` (`o_star_mode=1`): 이전 누적 출력 O*를 **새 running max로 rescale**. 입력 `a=m_prev, b=m`, 벡터 `v=exp_o_input`.
- `expmul_v_inst` (`o_star_mode=0`): 현재 score를 exp화하여 **현재 V*에 곱함**. 입력 `a=s, b=m`, 벡터 `v=v_star`.
- **running 누적 합산**(`expmul.sv:38-46`):
  - `exp_o_out[i] = exp_o_out_partial[i] + exp_v_out[i]` — 즉 `O_new = 2^(m_prev−m)·O_old + 2^(s−m)·V`.
  - `exp_o_input = (kv_counter_1==0) ? o_star_prev_in : exp_o_out` — 시퀀스 첫 원소면 외부(0)에서 시작, 이후엔 자기 출력을 되먹임 → **FlashAttention의 online rescaling 점화식을 단일 모듈 루프로 구현**.
  - 65-원소 벡터이므로 index 0(분모 exp-sum)과 1..64(출력)가 **동일 회로로 동시에** rescale·누적된다(FLASH-D식 분모 동봉 누적).
  - `vld_out = ... && (kv_counter_1==0)`: 시퀀스 전체(512개)를 다 누적한 시점에만 출력 valid(`expmul.sv:35`).

**(b) `expmul_stage.sv` (핵심 연산)** — `vec_out[i] = exp(a−b)·vec_in[i]`을 **곱셈기 없이** 계산(`expmul_stage.sv:3-19` 주석):

1. **Stage 1 — 차이 + log2(e) 근사**:
   - `x_diff = a − b` (`:74`, Q5.4)
   - `log_e_x = x_diff + (x_diff>>>1) − (x_diff>>>4)` (`:167`)
     → 이는 `x·log2(e) ≈ x + x/2 − x/16 = 1.4375·x` (실제 log2(e)=1.4427). **시프트 3개와 가산만으로 log2(e) 곱셈을 근사**. (주석 `:14`의 `Log2Exp(X)=−⌊X+(X≫1)−(X̂≫4)⌉`와 동일 구조)
   - `q_convert`로 `l_hat`(Q4.0, 정수 지수)로 양자화(`:114-117`). 즉 exp(x)=2^(x·log2 e)에서 지수 L을 4비트 정수로 클리핑([-16,0] 범위, `sys_defs.svh:152`).

2. **Stage 2 — 2^(−L)·V 배럴 시프트**(`:170-178`):
   - `l_hat`의 각 비트가 시프트 양을 제어하는 **5단 배럴 시프터**:
     - `l_hat[4]`→`>>>16`, `l_hat[3]`→`<<<8`, `l_hat[2]`→`<<<4`, `l_hat[1]`→`<<<2`, `l_hat[0]`→`<<<1`.
   - 즉 2의 거듭제곱 곱셈을 시프트로 치환 → **exp(score)·V를 곱셈기 0개로** 수행. (전형적 hardware-friendly softmax 근사.)
   - `kv_counter`로 시퀀스 진행 추적(`:155`), `v_stage_2`는 `o_star_mode`에 따라 `v_in`(O*) 또는 `v`(V*) 선택(`:154`).
- 비트폭: 입력 Q4.4(`EXPMUL_DIFF_IN`), 시프트 단계 내부는 Q9.23(`EXPMUL_SHIFT_STAGE`)로 확장 후 다시 Q9.17(`EXPMUL_VEC`)로 축소(`:124-140`).

> **요약**: AURA의 softmax는 (1) running max(`max.sv`), (2) exp를 log2-도메인 + 정수지수 + 배럴시프트로 근사(`expmul_stage.sv`), (3) 출력과 분모를 65-원소 벡터로 동시 online 누적(`expmul.sv`), (4) 마지막에 한 번만 나눗셈(`vector_division`)으로 정규화 — 즉 **곱셈기·지수함수 LUT·재방문(2-pass) 없이** FlashAttention을 streaming으로 완성한다.

### 3.7 `vector_division.sv` + `int_division.sv` — 최종 정규화 (O = 누적/분모)

- `vector_division`: 65-원소 누적벡터에서 `vec_in[0]`(분모 exp-sum)을 분모로, `vec_in[1..64]`를 분자로 하여 **64개 `int_division`을 병렬 인스턴스화**(`vector_division.sv:37-60`). 입력을 `q_convert`로 `DIV_INPUT`(Q9.8)로 변환 후 나눗셈.
- `int_division`: 부호 분리(`sign_q = sign_n ^ sign_d`) + 절댓값 추출 후 unsigned `divu`에 위임, 결과를 부호 복원하고 `q_convert`로 출력 Q0.7로 축소(`int_division.sv:60-123`).
- `divu`: 고전적 **비복원(non-restoring/long division) 반복 분주기**. `ITER = WIDTH + FBITS` 사이클 동안 시프트-비교-감산 반복(`int_division.sv:155-192`). 멀티사이클(분주가 가장 느린 단계)이며 `busy/done/valid`로 PE 파이프라인에 backpressure.

### 3.8 고정소수점 변환 체인: `q_convert` / `q_align_frac` / `q_align_int` / `q_saturate` / `q_sign_extend`

데이터패스 거의 모든 비트폭 변경 지점에 삽입되는 **재사용 가능한 양자화 빌딩블록**. 동작 순서(`q_convert.sv:19-28`):

1. **`q_align_frac`** (소수부 정렬, `q_align_frac.sv`):
   - `OUT_F > IN_F`: 좌측 시프트 + 0 패딩(`{in, {(OUT_F-IN_F){1'b0}}}`).
   - `OUT_F < IN_F`: 우측 시프트. **`ROUNDING=1`이면 round-half-up**: `abs_bias = 1<<((IN_F-OUT_F)-1)` 더한 뒤 `>>>` 후 `q_saturate`(`:28-49`). `ROUNDING=0`이면 단순 절단(`:51`). `ROUNDING`은 헤더 기본 1(`sys_defs.svh:84`).
2. **`q_align_int`** (정수부 정렬, `q_align_int.sv`):
   - `IN_I > OUT_I`: `q_saturate`(클리핑).
   - `IN_I < OUT_I`: `q_sign_extend`(부호확장).
   - 같으면 통과(`:17-36`).
- **`q_saturate`**: signed MAX(`{1'b0,{1...}}`)/MIN(`{1'b1,{0...}}`)으로 포화(`q_saturate.sv:10-20`).
- **`q_sign_extend`**: 상위 비트를 부호비트로 복제(`q_sign_extend.sv:11`).

> 이 5개 모듈은 **임의의 Qm.n → Qp.q 변환을 합성가능 RTL로 정확히** 처리(라운딩/포화/부호확장 모두). 우리 FPGA 양자화 데이터패스에 거의 그대로 이식 가능 (섹션 9).

### 3.9 `memory_controller.sv` — K→V→Q load / O drain FSM

- **상위 FSM 5상태**(`memory_controller.sv:47-55`): `PH_RESET → PH_LOAD_K → PH_LOAD_V → PH_COMPUTE → PH_DONE`. K, V 전체 시퀀스를 먼저 SRAM에 적재한 뒤 compute 단계에서 Q 타일 로드와 O 타일 drain을 인터리빙.
- **Compute 모드 3상태**(`:58-64`): `CMP_IDLE / CMP_LOAD_Q / CMP_DRAIN_O`. 우선순위: O drain > Q load(`:239-263`). `o_tiles_drained_cnt == NUM_TILES`면 종료(`:260-261`).
- **주소 계산**: base(`K_BASE/V_BASE/Q_BASE/O_BASE`, `sys_defs.svh:223-229`) + tile/vec/blk 오프셋. 한 벡터 = `MEM_BLOCKS_PER_VECTOR`개 64bit 블록(`:244`).
- **out-of-order 메모리 응답 처리**: `expected_tag_fifo`로 발급한 mem tag를 순서대로 추적, `mem2proc_data_tag`가 head tag와 일치할 때만 버퍼에 채움(`:106-111`, `:175-187`). 64bit 블록당 8개 INT8 원소를 `byte_level` union으로 언패킹(`:180-183`).
- 한계: "full uninterrupted access to the memline" 가정(`:1` 주석), `ELEMENTS_PER_MEMBLOCK` 하드코딩(주석 `:179`에 generalize 필요 명시).

### 3.10 SRAM 4종 — 메모리 계층

- **`KSRAM` / `VSRAM`**: 순환 FIFO(`fifo[0:511]`). `tail==NUM_ENTRIES`면 full, `tail==0`이면 empty. **K/V 전체 시퀀스를 저장하고 head 포인터로 반복 read** — 한 번 적재 후 모든 Q에 대해 재사용(`KSRAM.sv:32-43`). attention의 K/V 재사용성을 반영.
- **`QSRAM`**: **dual-bank ping-pong**. 한 뱅크가 채워지는 동안 다른 뱅크가 PE에 안정적 Q 타일(NUM_PES개 row) 제공(`QSRAM.sv:39-63`). `fill_bank`/`read_bank` 토글, `bank0_full`/`bank1_full` 플래그. read 시 뱅크를 0으로 클리어(`:104-118`).
- **`OSRAM`**: **dual-bank ping-pong drain**. PE들이 lockstep으로 한 뱅크 전체(NUM_PES개 O벡터)를 동시에 write(`OSRAM.sv:78-96`), 다른 뱅크는 mem_ctrl가 한 벡터씩 drain(`:100-117`).
- 의의: K/V는 streaming FIFO(전체 보관), Q/O는 더블버퍼 — **load/compute/store 중첩**으로 throughput 확보.

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 전체 흐름 (Q/K/V SRAM → PE → softmax → O)

```
DRAM(mem.sv)
   │  K_BASE/V_BASE/Q_BASE/O_BASE
   ▼
memory_controller (FSM)
   │ PH_LOAD_K → KSRAM(FIFO, 512 vec)
   │ PH_LOAD_V → VSRAM(FIFO, 512 vec)
   │ PH_COMPUTE: CMP_LOAD_Q → QSRAM(dual-bank, 4 row/tile)
   ▼
QSRAM ──q_vectors[i]──┐         KSRAM ──k_vector(브로드캐스트)──┐
                       ▼                                          ▼
                    PE[0..3]  (각 PE = 1 Q row, K/V 공유)
                       │
   ┌───────────────────┼───────────────────────────────────────┐
   │ dot_product:  s = (Q·K)>>3  + V 지연                        │
   │ max:          m_i = max(m_{i-1}, s)   ← online running max  │
   │ expmul:       O* = 2^(m_prev−m)·O*_old + 2^(s−m)·V*         │  ← 곱셈無
   │               (65-elem: [0]=분모 exp-sum, [1..64]=출력)     │
   │ vector_division: O = O*[1..64] / O*[0]   ← 마지막 1회 정규화 │
   └───────────────────┼───────────────────────────────────────┘
                       ▼
                 output_vectors_scaled[i]
                       ▼
   OSRAM(dual-bank) ─drain→ memory_controller (CMP_DRAIN_O) ─→ DRAM(O_BASE)
```

### 4.2 메모리 계층

- **L0 (외부)**: `mem.sv` — 단일 통합 메모리, 100ns(=`MEM_LATENCY_IN_CYCLES`) 지연, 최대 `NUM_MEM_TAGS=15` outstanding 트랜잭션(`sys_defs.svh:231-238`, `test/mem.sv:65-87`).
- **L1 (온칩 SRAM)**: KSRAM/VSRAM(전체 시퀀스 FIFO), QSRAM/OSRAM(타일 더블버퍼).
- **L2 (PE 내부 레지스터)**: dot/max/expmul/division 각 단계의 파이프라인 레지스터 + running 누적 레지스터.

### 4.3 병렬화 / 파이프라이닝

- **공간 병렬**: `NUM_PES=4` PE가 4개 Q row 동시 처리(K/V 공유 브로드캐스트). 트리 리덕션 내부도 64-way 곱셈 병렬.
- **시간 병렬(파이프라인)**: PE 내부 dot→max→expmul→div 4단이 valid/ready elastic pipeline. 트리 리덕션 각 스테이지도 1사이클 파이프 레지스터.
- **메모리-연산 중첩**: QSRAM/OSRAM ping-pong으로 다음 Q 타일 로드와 현재 출력 drain을 compute와 중첩.
- 헤더에 `MAX_NUM_PES` 산식(`sys_defs.svh:34`)으로 메모리 대역폭(8B/2/cycle) 대비 최적 PE 수 도출 로직 존재.

### 4.4 양자화 / 데이터타입 (f8 / fixed-point)

- **입출력**: INT8 = **Q0.7** ([-1,1) 고정소수점). (`fp32_to_f8.cpp:17` `Q_FACTOR=128`, `sys_defs.svh:87-94`)
- **내부 중간값**: 연산 단계마다 정밀도가 다른 Q-format으로 자동 확장/축소 (Q1.14 곱 → Q1.12 → Q7.12 내적합 → Q4.4 score → Q9.17 exp누적 → Q9.8 나눗셈 입력 → Q0.7 출력). 모든 변환은 `q_convert`(라운딩+포화) 경유.
- 파일명 `f8`은 **8비트 고정소수점(INT8/Q0.7)**을 의미하며, IEEE FP8(E4M3/E5M2)이 아님에 주의(코드상 `lround(x*128)` 정수 양자화). (`fp32_to_f8.cpp:56-66`)

---

## 5. HW/SW 매핑 (참조모델 ↔ QKV생성 ↔ RTL)

| 단계 | Python | C++ 참조모델 | SystemVerilog RTL |
|---|---|---|---|
| 입력 생성 | `Generate_QKV.py`: BERT(`bert-base-uncased`) 1번 head의 Q/K/V를 FP32 .mem으로 추출 (`:75-103`) | — | — |
| 양자화 | — | `fp32_to_f8.cpp`: FP32→INT8(Q0.7), `lround(x*128)` 포화 (`:56-66`) | `q_convert` 계열(런타임 내부 변환) |
| 골든 attention(FP) | — | `attention_fp32.cpp`: online-softmax(running max + sumexp + 정규화) (`:120-147`) | 전체 PE 파이프라인의 reference |
| 골든 attention(INT8) | — | `attention_f8.cpp`: INT8 입력, FP로 누적 후 INT8 재양자화 (`:92-123`) | AURA 실제 동작에 가장 근접한 비트정확 reference(단, exp는 실수 `exp()` 사용) |
| 내적 스케일 | — | `SCALE = 1/sqrt(64)` (`attention_fp32.cpp:7`) | `sum >>> 3` (`dot_product.sv:162`) |
| running max | — | `max_score = max(...)` (`attention_fp32.cpp:121-126`) | `max.sv` `m_out=(s>m_prev)?s:m_prev` |
| exp | — | `exp(scores[j]-max_score)` (정확) | `expmul_stage.sv` log2 근사 + 배럴시프트 (근사) |
| 정규화 | — | `weights[j] /= sumexp` | `vector_division.sv` / `int_division` |
| 출력 정리 | `strip_out_file.py` → `O_cleaned.mem` (`Makefile:302-303`) | — | OSRAM→mem.sv |
| 정밀도 검증 | — | `precision_measure.cpp`: MAE/RMSE/Max/rel/Top-1 비교 (임계 MAE≤3, Top-1≥95%) (`:8-13`) | `O_fixed_correct.mem` vs `O_cleaned.mem` |

- **핵심 매핑 관찰**: C++ 참조모델은 attention을 **표준 2-pass softmax**(max 먼저 전부, 그다음 exp/sum)로 구현하지만, RTL은 **1-pass online(FlashAttention)**로 구현 → 둘은 수학적으로 동치이나 RTL이 메모리/재방문을 줄인다. exp만 RTL이 근사이므로 `precision_measure`의 오차 임계가 곧 ExpMul 근사 허용오차다.
- BERT를 쓰지만 dk=64, seq=512는 ViT(예: ViT-B의 head dim 64)와도 호환되는 일반 attention 형상.

---

## 6. 빌드 · 실행

`Makefile` 기반(VCS/Synopsys 가정).

- **테스트 벡터 생성**:
  - `python python/Generate_QKV.py --model <name>` → `models/<name>/{Q,K,V}32.mem` (FP32) (`Makefile:276-282`)
  - `cpp/fp32_to_f8` → `{Q,K,V}.mem` (INT8) (`Makefile:286-292`)
  - 골든: `cpp/attention_fp32` → `O_float_correct.mem`, `cpp/fp32_to_f8` → `O_fixed_correct.mem` (`Makefile:294-300`)
- **시뮬레이션**:
  - `make build/AURA.simv` — VCS로 전체 컴파일(`Makefile:198-201`, `:149`).
  - `make output/<test>.out` — `+Q_MEMORY/+K_MEMORY/+V_MEMORY/+OUTPUT` plusarg로 실행(`Makefile:334-343`).
  - `make <module>.pass` — `@@@ Passed/Failed` grep(`Makefile:180-184`). 모듈별 테스트벤치(`MODULES` 변수, `:88`).
  - `make <module>.verdi` — Verdi GUI 디버깅(`:187-189`).
- **합성**:
  - `make synth/<module>.vg` — `dc_shell-t -f synth/AURAsynth.tcl` 호출, `MODULE`/`SOURCES`/`CLOCK_PERIOD` 환경변수 전달(`Makefile:204-210`).
  - 산출물: `.vg`(netlist), `.ddc`, `.rep`(area/power/timing), `.chk`. `make slack`로 slack 확인(`Makefile:213-214`, `AURAsynth.tcl:279-301`).
  - 합성 옵션: `compile -map_effort medium`(기본), `compile_ultra` 주석 처리됨(`AURAsynth.tcl:260-261`). 계층적 합성(`CHILD_MODULES`/`DDC_FILES`) 및 파라미터 합성 지원(`AURAsynth.tcl:40-54`, `:135-152`).
- **정밀도 측정**: `make output/<test>.prec` → `cpp/precision_measure`(`Makefile:376-377`).
- **커버리지**: `make <module>.cov`(line+tgl+cond+branch) (`Makefile:231-257`).

---

## 7. 의존성

- **시뮬레이터**: **Synopsys VCS**(`vcs -sverilog ...`, `Makefile:5-6`). iverilog/verilator는 사용하지 않음. SystemVerilog typedef/union/generate 다용 → verilator 포팅 시 일부 수정 필요할 것으로 추정.
- **파형 디버거**: **Verdi**(`-gui=verdi`, `Makefile:9`).
- **합성 툴**: **Synopsys Design Compiler**(`dc_shell-t`, `AURAsynth.tcl`). 표준셀 라이브러리 `lec25dscc25_TT.db`(250nm), wire load `tsmcwire`. EECS 470/570(미시간대) 클래스 인프라 경로 하드코딩(`AURAsynth.tcl:106`, `Makefile:15`).
- **C++**: g++ `-std=c++17`(`Makefile:28-29`), `<bits/stdc++.h>` 사용 → GCC/리눅스 전제.
- **Python 라이브러리**(`Generate_QKV.py:1-8`): `torch`, `transformers`(HuggingFace), `numpy`. BERT 가중치 다운로드 필요.
- 외부 데이터셋/모델: HuggingFace `bert-base-uncased` (Generate_QKV.py 기본값).

---

## 8. 강점 / 한계 / 리스크

### 강점
- **곱셈기 없는 exp**: ExpMul(log2 근사 + 정수지수 + 5단 배럴시프트)로 지수함수 LUT·곱셈기 제거 — area/power에 매우 유리(`expmul_stage.sv`).
- **진짜 1-pass online softmax**: running max + 분모를 출력벡터 0번에 동봉해 동일 회로로 누적, 마지막에 1회만 나눗셈 — 재방문/2-pass 제거(`expmul.sv`, `max.sv`).
- **완전 파라미터화 + 합성가능 양자화 IP**: `q_convert` 계열이 라운딩·포화·부호확장을 정확히 처리, 트리 리덕션이 차원에 따라 자동 재구성.
- **elastic pipeline(valid/ready)**: 멀티사이클 분주기와도 backpressure로 안전하게 연결.
- **SW/HW co-verify 파이프라인**: FP32 골든 → INT8 골든 → RTL 출력 → 정량 비교(`precision_measure`)가 Makefile로 자동화.

### 한계 / 리스크
- **하드코딩 다수**: `dk=64`일 때만 `>>3` 스케일 정확(`dot_product.sv:162`), `ELEMENTS_PER_MEMBLOCK`/64bit 블록 가정(`memory_controller.sv:179` 주석에 "make generalizable later"), ROWS/COLS=512/64 C++ 상수 고정.
- **ExpMul 정밀도 손실**: log2(e)를 1.4375로 근사(오차 ~0.36%)하고 지수를 4비트 정수로 클리핑([-16,0]) → 큰 음수 score에서 0으로 포화. 정확도는 `precision_measure` 임계(MAE≤3, Top-1≥95%)로만 검증되며 데이터 의존적. (`expmul_stage.sv:167`, `sys_defs.svh:152`)
- **나눗셈 비용**: `divu`가 `WIDTH+FBITS` 사이클 멀티사이클 → 파이프라인 throughput 병목 가능(`int_division.sv:152`).
- **단일 head / 단일 메모리**: multi-head, KV-cache, causal mask 미구현. `mem.sv`는 단순 단일 통합 메모리 모델.
- **ASIC(250nm) 전용 합성 흐름**: 우리 FPGA 타깃과 직접 합성 호환 안 됨(Vivado/Vitis 아님). RTL 자체는 합성가능하나 SRAM 매크로·메모리 모델은 FPGA용으로 교체 필요.
- **검증 성숙도 불확실**: 테스트벤치는 있으나 통과 로그/커버리지 결과는 repo에 없음 → 실제 검증 완성도는 **확인 불가**.
- **deprecated 잔존**: `_deprecated/`에 다수 구버전 — 혼동 위험(빌드엔 미포함).

---

## 9. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적)

우리 연구는 **FPGA 기반 고처리량 ViT/Transformer 가속기(HG-PIPE 계열) + XR 시선추적**이다. AURA는 ASIC 타깃이지만 **알고리즘-아키텍처 co-design 패턴과 합성가능 RTL 블록**이 직접 재사용 가능하다.

### 즉시 재사용 가능(High value)
1. **ExpMul 곱셈없는 exp 근사** (`expmul_stage.sv:167,170-178`): `x + (x>>1) − (x>>4)`로 log2(e) 곱 근사 후 배럴시프트로 `2^(-L)·V` — FPGA에서 DSP를 거의 안 쓰고 softmax 지수를 구현. HG-PIPE류 파이프라인의 attention softmax 단계에 DSP 절약용으로 이식 가치 큼. (단, FPGA에서는 LUT 기반 시프트가 매우 저렴 → 더 유리.)
2. **1-pass online softmax + 분모 동봉 누적** (`max.sv`, `expmul.sv:38-46`): 출력벡터 0번에 exp-sum을 함께 누적해 동일 회로로 rescale → **streaming/파이프라인 친화적**. HG-PIPE의 "끊김 없는 파이프라인" 철학과 정확히 부합. KV-cache 없이 streaming attention을 구성할 때 참고.
3. **`q_convert` 양자화 IP 5종**(`q_convert/q_align_frac/q_align_int/q_saturate/q_sign_extend`): 임의 Qm.n↔Qp.q를 라운딩/포화/부호확장 포함해 정확히 변환하는 **합성가능 빌딩블록**. FPGA INT8/혼합정밀 데이터패스에 거의 그대로 복붙 가능(라이선스/출처만 확인). 우리 ViT 양자화 가속기의 비트폭 경계마다 삽입할 표준 변환기로 활용.
4. **파라미터화 트리 리덕션**(`tree_reduce.sv`/`reduction_step.sv`): 내적합·LayerNorm·softmax-sum 등 모든 reduction에 재사용. 스테이지별 자동 비트확장 + 파이프 레지스터 내장 → FPGA 고주파 타이밍 클로징에 유리.
5. **valid/ready elastic pipeline 패턴**: 멀티사이클 유닛(나눗셈 등)을 throughput 깨지 않고 끼워넣는 표준 핸드셰이크. HG-PIPE의 dataflow에 안전하게 변동지연 유닛을 통합할 때 참고.

### 구조적 참고 (재설계 필요)
6. **메모리 더블버퍼링 패턴**(QSRAM/OSRAM ping-pong, KSRAM/VSRAM FIFO 재사용): FPGA에서는 BRAM/URAM으로 매핑. K/V는 한 번 적재 후 반복 read, Q/O는 ping-pong — 우리 XR 실시간(저지연) 시선추적에서 프레임 단위 더블버퍼링에 그대로 차용 가능.
7. **HW/SW co-verification 흐름**(FP32 골든 → INT8 골든 → RTL → precision_measure): 우리 양자화 정확도-하드웨어 검증 자동화에 그대로 이식. `precision_measure.cpp`의 MAE/RMSE/Top-1 메트릭은 XR 시선추적의 좌표 회귀 오차 평가로 변형 가능.

### XR 시선추적 특화 관점
8. 시선추적은 **저지연·저전력·소형 모델**이 핵심 — AURA의 edge/저전력 지향(README) 및 곱셈없는 softmax는 XR 헤드셋의 전력예산에 부합. 다만 시선추적 백본이 ViT일 경우 attention 비중이 작을 수 있어, ExpMul/online-softmax 재사용 효과는 **모델 attention 비중에 비례**(우선 프로파일링 권장).
9. AURA의 `MAX_NUM_PES` 산식(메모리 대역폭÷벡터바이트, `sys_defs.svh:34`)은 **대역폭-제약 하 최적 PE 수 도출** 방법론 — 우리 FPGA의 HBM/DDR 대역폭 대비 PE 수 결정에 동일 논리 적용 가능.

### 차용 시 주의
- ASIC(250nm) 합성 스크립트·표준셀·SRAM 매크로는 FPGA에 무의미 → **RTL 연산 블록만** 추출하고 메모리는 FPGA primitive로 교체.
- VCS/SystemVerilog 고급문법(union packed, generate) → Vivado/verilator 호환성 점검 필요.
- `dk=64`, INT8 하드코딩 가정 다수 → 우리 형상에 맞게 파라미터 일반화 필요.

---

## 10. 근거 / 한계 표기

### 확실(코드 직접 확인)
- 모든 RTL 모듈 동작·포트·핵심 라인은 `verilog/*.sv`를 직접 읽고 파일명:라인으로 표기.
- 고정소수점 비트폭 산식: `include/sys_defs.svh` 직접 확인.
- 빌드/합성/실행 흐름: `Makefile`, `synth/AURAsynth.tcl` 직접 확인.
- C++ 참조모델/양자화/정밀도: `cpp/attention_fp32.cpp`, `attention_f8.cpp`, `fp32_to_f8.cpp`, `precision_measure.cpp` 직접 확인.
- QKV 생성: `python/Generate_QKV.py`, `QKV_obtain.py` 직접 확인.

### 추정 (코드 정황 근거, 단정 불가)
- ExpMul 근사식의 원논문/출처: README가 "ExpMul"을 언급(`README.md:11`)하나 인용 없음 → 외부 ExpMul 연구 기반으로 **추정**.
- "FLASH-D식 분모 동봉" 표현: 65-원소 벡터(0=분모)와 expmul 누적 구조로부터 **추정한 해석**(README가 FLASH-D 언급, `:11`). 정확한 FLASH-D 알고리즘 일치 여부는 미검증.
- verilator 포팅 난이도: SystemVerilog 문법 다용으로부터 **추정**.

### 확인 불가
- 별도 설계 논문/PDF: repo 내 없음.
- 실제 합성 결과(area/power/timing 수치): `.rep` 산출물 없음.
- 검증 통과/커버리지 실측치: 로그 없음.
- `models/*.mem` 실제 테스트 데이터: 현재 repo에 생성물 없음(Glob 결과 없음).

### 완전 분석하지 않은 부분 (의도적 제외 또는 시간상)
- `cpp/fp32_to_f16.cpp`, `precision_measuref16.cpp`, `input_to_f8.cpp`, `output_to_f8.cpp`, `generate_mem.cpp`, `generate_output_{f8,fp64}.cpp`: 보조/변형 도구로 판단해 핵심만 다룸(미정독).
- `python/strip_out_file.py`, `convert_to_dec*.py`: 출력 후처리 유틸(Makefile 용도만 확인, 본문 미정독).
- `test/aura_test.sv` 및 모듈별 테스트벤치: `mem.sv` 일부만 정독, 나머지 테스트벤치는 미정독.
- `_deprecated/` 전체: 제약상 제외.
