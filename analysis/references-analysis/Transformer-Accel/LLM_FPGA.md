# Transformer-Accel / LLM_FPGA 정밀 분석

> 대상: `REF/Transformer-Accel/LLM_FPGA`
> 분석일: 2026-06-20
> 분석 방식: Glob/Read 기반. **핵심 한계 — 이 repo 스냅샷에는 HLS 자체 소스(.cpp/.h)가 1개도 없다.**
> 빌드 인프라(Makefile), 설정(kernel.cfg), 빌드 로그(logs/), 합성 리포트(reports/*.rpt)만 존재한다.
> 따라서 커널 내부 구조는 **HLS 합성 리포트의 모듈 계층**에서 역으로 재구성했다(라인 근거 명시).

---

## 1. 개요

- **정체성**: W8A8(weight 8-bit / activation 8-bit) 양자화 Transformer/LLM을 **Xilinx Alveo U280(HBM, 3-SLR)** FPGA에 매핑하는 HLS 가속기 모음. Vitis HLS 2022.1.2로 합성됨(`*_csynth.rpt` line 8, target `xcu280-fsvh2892-2L-e` line 12).
- **모델 5종**: `bert_12L`, `GPT_prefilling_24L`, `GPT_decoding_24L`, `LLaMA_prefilling_32L`, `LLaMA_decoding_32L` (W8A8_version 하위 디렉토리).
- **공통 아키텍처**: 모든 모델이 **하나의 Transformer layer**를 3개 커널(region_1/2/3)로 분할하고, 각 region을 **별도 SLR**에 배치한다(`kernel.cfg` line 13–15: `slr=...region_1_1:SLR0`, `region_2:SLR1`, `region_3:SLR2`). region 간은 AXIS 스트림으로 직접 연결되어 SLR-to-SLR 파이프라인을 형성한다(`kernel.cfg` line 16–20 `stream_connect=...`).
- **흥미로운 명명 규칙**: 모델이 BERT/GPT/LLaMA로 다양하지만 커널 이름은 전부 `Bert_layer_dataflow_region_*` 로 동일하다(예: LLaMA_prefilling `Bert_layer_dataflow_region_2_csynth.rpt` line 4). 즉 **하나의 HLS 템플릿을 모든 모델이 재사용**한다(추정: HeteroCL/Allo 코드 생성 기반 — 근거는 8절 의존성 참조).
- **출처 추정**: Makefile의 경로 `/home/jz2292/project/transformer/heterocl_file/` (`Makefile` line 19, `utils.mk` line 20)와 `$(XF_PROJ_ROOT)/allo/harness/readme_gen/...`(`utils.mk` line 113)는 이 디자인이 **HeteroCL → Allo HLS 프레임워크**로 생성되었음을 시사한다(추정).

---

## 2. 디렉토리 구조

### 2-1. 자체(분석 대상) 파일 — 모델 디렉토리당 동일 구성

각 모델(`W8A8_version/<model>/`)은 다음을 가진다(`bert_12L` 기준, Glob 확인):

```
W8A8_version/
├── bert_12L/
│   ├── Makefile                 # 플랫폼 분기 진입점 (line 38–52)
│   ├── makefile_us_alveo.mk     # 실제 빌드 규칙 (v++ 컴파일/링크) ★핵심
│   ├── makefile_zynqmp.mk       # Zynq MPSoC 변형
│   ├── makefile_versal_alveo.mk # Versal Alveo 변형
│   ├── makefile_versal_ps.mk    # Versal PS 변형
│   ├── utils.mk                 # 환경체크/XSA 유틸
│   ├── report_copy.mk           # 합성 리포트 복사 규칙
│   └── kernel.cfg               # v++ 링크 connectivity (SLR/HBM/stream) ★핵심
├── GPT_prefilling_24L/   (동일 7 mk + kernel.cfg)
├── GPT_decoding_24L/     (동일 7 mk; kernel.cfg는 Glob 캐시 이슈로 일부 미검출)
├── LLaMA_prefilling_32L/ (동일)
└── LLaMA_decoding_32L/   (동일 7 mk + kernel.cfg)
```

> **누락 확인**: Makefile이 참조하는 커널 소스 `bert_region_1.cpp`/`bert_region_2.cpp`/`bert_region_3.cpp`, 호스트 소스 `host.cpp`, `xcl2.cpp`(`makefile_us_alveo.mk` line 64, 91–101)는 **저장소에 존재하지 않음**(확인: 전체 `**/*.cpp` Glob 결과 0건). → 본 분석의 커널 내부는 합성 리포트로 재구성.

### 2-2. 생성물 / 제외 (이름만 언급)

- `logs/link/`, `logs/bert_region_{1,2,3}/`: v++/vivado 빌드 로그(`v++.log`, `vivado.log`, `*.steps.log`, `*_vitis_hls.log`, `my_rm_synth_1_runme.log`, `impl_1_runme.log`) — 제외.
- `reports/link/imp/`: 배치·라우팅 후 합성 리포트(`impl_1_*_util_*.rpt/.pb/.xutil`, `impl_1_hw_bb_locked_timing_summary_*.rpt/.rpx/.rpv`) — 타이밍/리소스 최종 결과, 본문은 HLS csynth 위주라 제외.
- `reports/<model>/bert_region_{1,2,3}/hls_reports/Bert_layer_dataflow_region_{1,2,3}_csynth.rpt`: **HLS C-synthesis 리포트 — 본 분석의 1차 근거**(제외 아님).
- third-party/vendor: 없음(소스 자체가 없음). `.bit/.xclbin` 산출물도 스냅샷에 없음.

### 2-3. 합성 리포트 보유 현황 (Read 직접 확인)

| 모델 | region_1 | region_2 | region_3 | kernel.cfg |
|---|---|---|---|---|
| bert_12L | O | O | O | O (단일 weight 포트) |
| GPT_prefilling_24L | O | (추정 O) | (추정 O) | O (bert와 동일) |
| GPT_decoding_24L | O | O | O | (Glob 캐시로 미검출, 존재 추정) |
| LLaMA_prefilling_32L | O | O | O | O (bert와 동일) |
| LLaMA_decoding_32L | O | (추정 O) | (추정 O) | O (16-bank 분산) ★ |

> "추정 O" = 동일 디렉토리 구조이나 본 세션에서 직접 Read까지 확인하진 않음. region_1만 헤더 Read로 존재 확정.

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

> 근거: 각 region의 `*_csynth.rpt` "Performance Estimates > Instance" 표(모듈 인스턴스 목록)와 "Utilization > Instance/Memory" 표.
> 대표로 **bert_12L(prefilling 계열)** 을 깊게 보고, **GPT_decoding_24L(decoding 계열)** 과의 차이를 비교한다.

### 3-A. Region 1 — 입력 로딩 + QKV 선형사상

bert_12L region_1 인스턴스(`bert_region_1/.../region_1_csynth.rpt` line 41–47):

| 모듈 | 역할(추정) | latency(cycles) | DSP/FF/LUT (line 86–99) |
|---|---|---|---|
| `input_loader_1` | activation(inp) 입력 로드 | 49227 | 0 / 938 / 890 |
| `input_loader_kv_1` | K/V 경로용 입력 로드 | 49233 | 24 / 3549 / 4019 |
| `input_loader_q_1` | Q 경로용 입력 로드 | 49233 | 24 / 3549 / 3999 |
| `weight_loader_r1_1` | QKV weight HBM→on-chip 프리페치(dataflow) | 2359369 | 0 / 4851 / 2627 |
| `Linear_layer_qkv_1` | Q(또는 K/V) projection GEMM | 2379718 | 112 / 33842 / 38180 |
| `Linear_layer_qkv_40_1` | 추가 QKV projection 인스턴스 | 2379718 | 112 / 33842 / 38180 |
| `Linear_layer_qkv_41_1` | 추가 QKV projection 인스턴스 | 2379718 | 112 / 33842 / 38180 |

근거·해석:
- **W8A8 증거**: region_1의 출력 AXIS 스트림 `outp_k/outp_v`는 **TDATA 128bit**(`region_1_csynth.rpt` line 471, 477), `outp_q` 128bit(line 483), `outp_inp` 256bit(line 489). 128bit = int8 × 16 lane → 활성값이 8-bit로 패킹됨(확인). HBM 포트 `m_axi_gmem*`는 **512bit WDATA/RDATA**(line 216, 236) → int8 weight 64개/transfer(확인).
- **QKV 3-way 병렬**: `Linear_layer_qkv`가 3개 인스턴스(_1, _40, _41)로 복제 → Q/K/V 세 projection을 공간 병렬 처리(추정). 각 112 DSP 사용(line 86–88).
- **weight_loader가 전체 latency 지배**: region_1 총 latency 2,379,797 cycle(line 32) 중 weight_loader 2,359,369(line 44)가 99% — region_1은 **weight 메모리 대역폭 바운드**(확인).
- **인터페이스**: 입력 weight 3종(wk/wv/wq)은 각각 별도 HBM 포트 gmem1/2/3(`bert_12L/kernel.cfg` line 5–7), inp은 gmem0(line 2–4)으로 분리되어 동시 로드(확인).

### 3-B. Region 2 — Attention + Softmax + Context + Out-proj + Residual + LayerNorm0

bert_12L region_2 인스턴스(`bert_region_2/.../region_2_csynth.rpt` line 41–50, util line 89–100):

| 모듈 | 역할 | latency(cycles) | BRAM/DSP/FF/LUT/URAM |
|---|---|---|---|
| `weight_ds0_loader_1` | attn out-proj weight 로드(dataflow) | 2359369 | 0/0/1618/886/0 |
| `K_writer` | K를 on-chip KV-RAM에 기록 | 74369 | 4/0/301/812/0 |
| `V_writer` | V를 on-chip KV-RAM에 기록 | 418369 | 4/0/1350/4284/0 |
| `Q_buffer_1` | Q 버퍼링 | 74369 | 4/0/107/327/0 |
| `Attention_layer_1` | QK^T 스코어 GEMM | 1765441 | 2/**320**/46348/36026/0 |
| `Softmax_layer_1` | row-wise softmax | 2200321 | 8/38/5486/8260/0 |
| `Context_layer_1` | score·V GEMM (context) | 1640513 | 4/112/34121/37538/0 |
| `Linear_layer_ds0_1` | attention output projection(GEMM) | 2379333 | 10/112/33439/33981/0 |
| `Res_layer0_1` | residual add (x + attn_out) | 74817 | 16/16/3636/1949/0 |
| `Layer_norm0_1` | LayerNorm #0 | 255821 | 20/**80**/18751/12736/0 |

근거·해석:
- **Attention 분해형(prefilling)**: prefilling은 `Attention_layer`(QK^T) → `Softmax_layer` → `Context_layer`(·V) 의 **3-stage 분리** 구조(line 45–47). 전체 시퀀스 행렬을 GEMM으로 처리하므로 가장 무겁다(Attention 320 DSP, line 89).
- **on-chip KV 캐시(URAM)**: K/V가 `K_V_RAM_2P_URAM_1R1W` 모듈에 저장됨 — `K_V_U`/`V_V_U` 각 24576 words × 128bit, **24 URAM씩(총 48)**(`region_2_csynth.rpt` line 112–116). 즉 전체 시퀀스의 K/V를 외부 메모리 왕복 없이 on-chip 보관(확인). 128bit = int8 head_dim 묶음.
- **Softmax 비용**: Softmax latency 2.2M cycle(line 46)로 Attention/Context보다 큼 — exp/정규화의 직렬성이 병목 후보. 내부 FIFO `sfm_outp`는 depth 512 × 64bit(line 128)로 한 행 버퍼.
- **LayerNorm DSP 80**: 평균/분산/정규화에 부동·고정소수 연산 다수(line 92) → 양자화 LLM에서도 LayerNorm은 고정밀(추정 int16/fp) 유지 가능성.
- **출력**: region_2 → region_3 연결은 `outp_ln0` 스트림(`kernel.cfg` line 20) — LayerNorm0 결과를 FFN region으로 전달.

### 3-C. Region 3 — FFN(up) + 활성화 + FFN(down) + Residual + LayerNorm1 + 출력

bert_12L region_3 인스턴스(`bert_region_3/.../region_3_csynth.rpt` line 41–49, util line 88–100):

| 모듈 | 역할 | latency(cycles) | BRAM/DSP/FF/LUT/URAM |
|---|---|---|---|
| `input_loader_ds1_res1_1` | FFN 입력 + residual 경로 로드 | 49161 | 0/24/3091/3251/0 |
| `weight_loader_r3_1` | FFN weight 프리페치(dataflow) | 4718665 | 0/0/2985/1558/0 |
| `Linear_layer_ds1_1` | FFN up-projection GEMM (d→4d) | 4757957 | 4/**176**/61399/67348/1 |
| `Gelu_layer_1` | GELU 활성화 | 200708 | 0/**256**/26731/15293/**8 URAM** |
| `Linear_layer_ds2_1` | FFN down-projection GEMM (4d→d) | 4732677 | 34/176/61695/67727/0 |
| `Res_layer1_1` | residual add | 74817 | 16/16/3380/1947/0 |
| `Layer_norm1_1` | LayerNorm #1 | 255884 | 20/80/19008/12739/0 |
| `output_writer_1` | 결과 HBM(gmem0) 기록 | 49226 | 0/0/933/1122/0 |

근거·해석:
- **FFN 2-GEMM**: `Linear_layer_ds1`(up, 176 DSP) + `Linear_layer_ds2`(down, 176 DSP)가 region_3의 latency를 지배(각 ~4.7M cycle, line 44, 46). region_3 총 4,853,717 cycle(line 32)으로 **layer 내 가장 무거운 region**.
- **GELU = LUT 방식**: `Gelu_layer`가 **8 URAM** + 256 DSP 사용(line 88) → GELU를 URAM 룩업테이블 + 보간으로 근사 구현(추정). exp 직접 계산 대신 LUT.
- **LLaMA prefilling도 동일 `Gelu_layer` 사용**: LLaMA_prefilling region_3에도 `Gelu_layer_1`이 존재(`LLaMA_prefilling_32L/.../region_3_csynth.rpt` line 45). **LLaMA의 정식 활성화는 SiLU/SwiGLU**임에도 모듈명이 GELU → 템플릿 공유의 흔적이거나 SiLU를 동일 LUT 인프라로 구현(추정/확인 불가). region_3에 별도 gate-projection 모듈은 리포트상 보이지 않음 → SwiGLU의 gate 분기는 미구현이거나 ds1에 융합(추정).

### 3-D. Decoding 변형 (GPT_decoding_24L) — Prefilling과의 핵심 차이

GPT_decoding region_2 인스턴스(`GPT_decoding_24L/.../region_2_csynth.rpt` line 41–50, util line 89–101):

| 모듈 | prefilling 대비 차이 |
|---|---|
| `head_spliter_1` | **신규** — 멀티헤드 분할(line 43) |
| `Self_attention_1_wrapper` | **융합형** — prefilling의 Attention+Softmax+Context를 하나로 통합(line 46), 380 DSP/161709 FF(util line 93) |
| `head_merger_1` | **신규** — 헤드 병합(line 47) |
| `weight_sfa_loader_1` | self-attention용 weight 로더 추가(line 45) |
| `K_writer`/`V_writer` | 유지하되 LUT 급증(K 6287, V 6028 LUT, util line 89,94) → KV-cache append 로직 |

decoding region_2 KV-cache(util line 109–123):
- `K_V_U`~`K_V_3_U`, `V_V_U`~`V_V_3_U` = **8 뱅크**, 각 2048 words × 512bit, **16 URAM씩(총 128 URAM = SLR의 40%)**(line 113–122). prefilling(48 URAM, 단일뱅크)과 달리 **헤드별 다중 뱅크로 분할**하여 디코딩 시 KV append/read 병렬도 확보(확인).

latency 스케일 차이(가장 중요):
- **Prefilling**: region 단위 latency가 ms~수백 ms. 예 LLaMA_prefilling region_3 = **90,315,592 cycle ≈ 0.301 sec**(`LLaMA_prefilling_32L/.../region_3_csynth.rpt` line 32) — 전체 시퀀스 일괄 처리.
- **Decoding**: region 단위 latency가 us. 예 GPT_decoding region_2 = **10,404 cycle ≈ 34.7 us**(`GPT_decoding_24L/.../region_2_csynth.rpt` line 32), region_3 = 20,635 cycle ≈ 68.8 us(line 32) — **토큰 1개**만 처리하므로 ~200배 이상 빠름(확인). decoding은 GEMM이 GEMV로 축소.

### 3-E. 양자화(W8A8) 데이터패스 — 정밀 분석

소스가 없어 양자화 커널(quantize/dequantize/scale)을 직접 볼 수 없으나, 리포트에서 다음을 확인:
- **활성 8-bit 패킹**: region 간 AXIS 스트림 폭 = 128bit(K/V/Q), 256bit(inp)로 int8 × {16,32}(`region_1_csynth.rpt` line 471–489) — 확인.
- **weight 8-bit**: HBM m_axi 512bit/beat(line 216) → int8 × 64. weight_loader가 이를 on-chip으로 프리페치(확인).
- **누적/스케일 재양자화 모듈은 리포트에 독립 인스턴스로 보이지 않음** → matmul 모듈(`Linear_layer_*`, `Attention_layer`, `Context_layer`) 내부에 int8×int8→int32 누적 후 스케일·클립으로 int8 재양자화하는 융합 로직이 들어간 것으로 추정(확인 불가, 소스 부재).
- LayerNorm/Softmax/GELU는 DSP·URAM을 별도로 크게 쓰므로 **부분적 고정밀(int16/고정소수) 경로**일 가능성(추정).

---

## 4. 데이터 플로우

한 Transformer layer의 흐름(kernel.cfg + 리포트 종합):

```
[HBM]
 inp(gmem0), Wk/Wv/Wq(gmem1/2/3)
        │
        ▼  region_1 (SLR0)  ── Bert_layer_dataflow_region_1
 input_loader(+kv,+q) → weight_loader_r1 → Linear_layer_qkv ×3
        │ AXIS stream: outp_k(128b), outp_v(128b), outp_q(128b), outp_inp(256b)
        ▼  region_2 (SLR1)  ── Bert_layer_dataflow_region_2
 [prefilling] K_writer/V_writer/Q_buffer → KV-URAM
            → Attention(QKᵀ) → Softmax → Context(·V)
            → Linear_layer_ds0(out-proj) → Res_layer0(+inp) → Layer_norm0
 [decoding]  head_spliter → Self_attention_wrapper(KV-cache append) → head_merger
            → Linear_layer_ds0 → Res_layer0 → Layer_norm0
        │ AXIS stream: outp_ln0(16-deep FIFO)
        ▼  region_3 (SLR2)  ── Bert_layer_dataflow_region_3
 input_loader_ds1_res1 → weight_loader_r3
            → Linear_layer_ds1(FFN up d→4d) → Gelu_layer → Linear_layer_ds2(FFN down 4d→d)
            → Res_layer1(+residual) → Layer_norm1 → output_writer
        │
        ▼
[HBM] outp(gmem0)
```

- region 내부는 Vitis **DATAFLOW**(모든 region csynth "Pipeline Type = dataflow", line 32) → 모듈들이 task-level 파이프라인으로 동시 실행.
- region 간은 **SLR 경계를 넘는 AXIS 스트림 직결**(kernel.cfg `stream_connect`, depth 16) → 한 layer가 3-SLR을 통과하는 매크로 파이프라인.
- **32/24/12 layer 반복**: 단일 layer 커널을 호스트가 layer 수만큼 재호출하거나 가중치 주소만 바꿔 반복하는 것으로 추정(host.cpp 부재로 확인 불가).

---

## 5. HW/SW 매핑

| 계층 | 구현 | 근거 |
|---|---|---|
| SW(호스트) | `host.cpp` + `xcl2.cpp`(OpenCL/XRT), x86 또는 aarch64 | `makefile_us_alveo.mk` line 64; `Makefile` line 30–36 HOST_ARCH 분기 |
| SW→HW IF | XRT/OpenCL `-lOpenCL`, AXI-Lite `s_axi_control` | mk line 57; `region_1_csynth.rpt` line 181–197 |
| HW 커널 | 3× v++ `-k Bert_layer_dataflow_region_{1,2,3}` | `makefile_us_alveo.mk` line 93,97,101 |
| 메모리 | HBM 0–6(bert) / 0–16(LLaMA-decode) m_axi 512bit | `bert_12L/kernel.cfg` line 2–11; `LLaMA_decoding/kernel.cfg` line 2–21 |
| 배치 | region_1→SLR0, 2→SLR1, 3→SLR2 | `kernel.cfg` line 13–15 |
| 클럭 | 245 MHz 링크 목표(`--kernel_frequency 245`), HLS 3.33ns(300MHz) 타깃 | `makefile_us_alveo.mk` line 45; csynth line 23 |
| 디바이스 | Alveo U280 `xcu280-fsvh2892-2L-e`, virtexuplus | csynth line 11–12 |

플랫폼 분기(`Makefile` line 38–52): zynqmp(aarch64)→`makefile_zynqmp.mk`, US Alveo(x86)→`makefile_us_alveo.mk`, versal→alveo/ps 분기. 즉 동일 커널을 **여러 보드 패밀리**로 빌드 가능하게 설계(확인).

---

## 6. 빌드·실행

- 진입: `make all TARGET=hw PLATFORM=xilinx_u280_...` → 플랫폼 자동 분기(`Makefile` line 26 default U250, 실제 합성은 U280).
- 커널 컴파일: `v++ -c -k Bert_layer_dataflow_region_N --save-temps --optimize 3` → `*.xo`(`makefile_us_alveo.mk` line 91–101, 71).
- 링크: `v++ -l --config ./kernel.cfg --kernel_frequency 245` → `Bert_layer.link.xclbin` → `-p` 패키징 → `Bert_layer.xclbin`(line 45, 103–106).
- 호스트: `g++ host.cpp xcl2.cpp -lOpenCL -lxrt...` → `./Bert_layer`(line 109–110).
- 실행: `make run` → emu면 `XCL_EMULATION_MODE`, hw면 직접 `./Bert_layer Bert_layer.xclbin`(line 117–123).
- 리포트 추출: `report_copy.mk` — `out_region3.prj/solution1/syn/report/*.rpt` 복사 후 PE*/solution1 정리(line 1–7).
- **빌드 불가 상태**: 커널 .cpp/호스트 .cpp 부재로 현 스냅샷에서는 `make`가 즉시 실패(확인 — 소스 0건).

---

## 7. 의존성

- **Xilinx Vitis 2022.1.2** (HLS + v++ + XRT) — csynth line 8, mk line 66 `/opt/xilinx/2022.1/...`.
- **XRT/OpenCL** 런타임 — mk line 57.
- **HBM2 Alveo U280** 보드.
- **HeteroCL / Allo** (추정) — `heterocl_file` 경로(`Makefile` line 19), `$(XF_PROJ_ROOT)/allo/harness/readme_gen/readme_gen.py`(`utils.mk` line 113). 커널 .cpp가 이 프레임워크로 자동 생성되었을 가능성. → 이 repo는 "생성 결과의 빌드 셸"만 보관.
- third-party 라이브러리 vendoring 없음.

---

## 8. 강점 · 한계

**강점**
- **SLR-aware 3-region 분할**: layer를 3개로 쪼개 3-SLR에 분산 → SLR 간 라우팅 혼잡 완화 + 매크로 파이프라인(kernel.cfg line 13–15).
- **on-chip KV 캐시(URAM)**: prefilling 48 URAM, decoding 128 URAM 헤더 분할 → 외부 메모리 왕복 제거(region_2 memory 표).
- **prefilling/decoding 전용 커널 분리**: GEMM(분해형 attention) vs GEMV(융합 Self_attention_wrapper)로 단계별 최적 구조(3-D).
- **HBM 대역폭 활용**: decoding은 16 HBM 뱅크에 weight 분산(LLaMA_decoding/kernel.cfg line 2–21)로 memory-bound 디코딩 완화.
- **W8A8 스트림 패킹**: 128/256/512bit 스트림으로 int8 다중 lane 동시 처리(region_1 line 216, 471–489).
- 멀티 보드 빌드 지원(zynqmp/us-alveo/versal).

**한계**
- **소스 부재(치명적)**: 본 repo에는 HLS .cpp/.h, host.cpp가 없어 양자화 세부(스케일 산출, 재양자화 위치), 루프 타일링/언롤 factor, dataflow depth를 **직접 검증 불가**. 본 분석은 합성 리포트 역추정.
- **단일 layer 단위**: 32/24/12 layer 전체 오케스트레이션(가중치 스트리밍, layer 루프)은 host에 있으나 부재로 확인 불가.
- **latency 매우 큼(prefilling)**: LLaMA prefilling region당 ~0.3sec(region_3 line 32) × 32 layer → 단일 forward가 수~수십 초 규모(추정). weight_loader가 region마다 latency 지배(region_1 line 44) = **weight 메모리 바운드**.
- **LLaMA 활성화 정합성 불명**: SwiGLU 대신 `Gelu_layer` 명칭(LLaMA_prefilling region_3 line 45) → RoPE/SwiGLU 미구현 또는 융합 여부 확인 불가.
- **양자화 정확도 데이터 없음**(W8A8 PTQ/QAT 방식, 캘리브레이션 결과 부재).

---

## 9. 우리 프로젝트(PRJXR-HBTXR: 고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 시사점

> 우리 목표 추정: HG-PIPE류 고처리량 ViT 가속기 + XR eye-tracking 저지연 추론.

1. **Region/SLR 분할 전략 차용**: 한 Transformer block을 attention/FFN 경계로 쪼개 SLR/die에 매핑하고 AXIS로 직결하는 패턴은 우리 ViT 블록 파이프라인에 그대로 적용 가능(kernel.cfg line 13–20). 단 우리는 layer-pipeline(HG-PIPE의 fully-pipelined)을 노린다면 region 분할보다 **layer 전개(unfold)** 가 더 적합 — 본 repo는 layer를 재사용(반복 호출)하는 area-efficient 방식이라 처리량 면에선 우리 목표와 상반(주의).
2. **prefilling vs decoding 분리 → XR에선 "단발 추론" 최적화**: XR eye-tracking은 프레임당 1회 추론(배치=1, decoding과 유사한 GEMV-heavy)이므로, decoding 커널의 **융합 Self_attention_wrapper + head_spliter/merger + 다중 URAM 뱅크**(3-D) 구조가 저지연 단발 추론 레퍼런스로 유용.
3. **on-chip KV/feature 캐시 URAM 패턴**: `K_V_RAM_2P_URAM_1R1W`(region_2 line 113)처럼 중간 텐서를 URAM에 상주시키는 기법은 XR 저지연에 직접 이식 가능. ViT는 KV-cache가 없지만 patch token/attention map 버퍼에 동일 적용.
4. **W8A8 스트림 패킹(int8×lane)**: 128/256/512bit 스트림으로 int8 패킹(region_1 line 471–489)은 우리 양자화 ViT 가속기의 대역폭 설계 그대로 활용. 우리는 ViT-Quantization repo(기 분석)와 정합되는 8-bit 데이터패스 설계.
5. **메모리 바운드 경고**: 본 repo의 병목은 weight_loader(region당 latency 지배). HG-PIPE 목표(고처리량)를 위해선 **weight를 on-chip 상주**시키는 fully-pipelined 설계가 필수 — 본 repo의 weight streaming 방식은 처리량 한계를 보여주는 반면교사.
6. **활성화 LUT 구현(GELU 8 URAM)**: `Gelu_layer`의 URAM-LUT 근사(region_3 line 88)는 우리 GELU/SiLU 하드웨어 구현 시 DSP 절감 레퍼런스.
7. **HeteroCL/Allo 생성 흐름(추정)**: 알고리즘→HLS 자동생성 툴체인을 쓴다면(우리 algo2fpga 스킬과 연계) 본 repo 같은 region 분할 코드 생성 패턴 참고 가능.

---

## 10. 근거 표기 요약

- **확인(라인 근거)**: 디렉토리/파일 구성, 빌드 흐름(Makefile/mk/kernel.cfg 라인), 커널 모듈 계층·latency·리소스·URAM KV캐시·스트림 비트폭·prefilling↔decoding 차이(각 csynth.rpt 라인).
- **추정**: 모듈의 정확한 알고리즘 내부(양자화 스케일 위치, 타일링/언롤, SwiGLU/RoPE 구현 여부), 멀티 layer 오케스트레이션, HeteroCL/Allo 생성 여부 — 모두 소스 부재로 인한 역추정.
- **확인 불가**: 양자화 정확도, host.cpp/커널.cpp 실제 코드, 빌드 재현 — 해당 파일이 스냅샷에 없음.
