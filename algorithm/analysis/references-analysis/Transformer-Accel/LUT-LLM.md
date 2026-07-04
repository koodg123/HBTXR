# LUT-LLM 코드베이스 정밀 분석

> 대상: `REF/Transformer-Accel/LUT-LLM`
> 분석 도구: Glob/Grep/Read (라인 근거 기반)
> 작성일: 2026-06-20
> 근거 표기 규칙: **확인=`파일명:라인`** / 불확실=`추정` / 자료 없음=`확인 불가`

---

## 0. 핵심 요약 (한 문단)

LUT-LLM은 **LUT(Look-Up Table) 기반 가중치 양자화(벡터 양자화, VQ)** 를 사용해 LLM(Qwen 계열)의 선형 연산을 DSP 곱셈 대신 **LUT/메모리 매칭**으로 치환하는 **TAPA 기반 HLS Transformer 가속기**이다. AMD/Xilinx **Versal V80 (HBM, `xcv80-lsva4737-2MHP-e-S`)** 를 1차 타깃으로 하며, Qwen 디코더 블록 전체(`qwen_block`)를 단일 HLS 커널로 구현해 **55개(prefill) / 28개(decode) HBM 채널** 에 직결한다. 핵심 연산 엔진은 `memory_matcher_w_vq_half_dsp_final_int4`(INT4 + half-DSP VQ matmul)이고, 빌드 플로우는 TAPA → RapidStream(autobridge floorplan/pipeline) → Vivado(custom_design BD)로 구성된다. **중요 한계: 이 repo에는 HLS 소스(`.cpp`/`.h`)가 단 한 개도 없다** — Makefile, 빌드 산출물(`.xo`), 구현 리포트, TCL/Python 인프라 스크립트만 존재하므로, 연산 알고리즘은 빌드 산출물 메타데이터와 인프라 스크립트의 포트/이름 규약으로 **역추정**한다.

---

## 1. 개요

| 항목 | 값 | 근거 |
|---|---|---|
| 프로젝트 성격 | LUT/VQ 양자화 기반 LLM(Qwen) FPGA 가속기 | `lut-dla/Makefile:2` (`TOP=lut_dla_core`), `qwen_block/Makefile:2` (`TOP=qwen_block`) |
| HLS 프레임워크 | **TAPA** (`tapa compile`/`tapa g++`) | `qwen_block/Makefile:1,5,14` (`TOOL=tapa`) |
| 백엔드 최적화 | **RapidStream** (`rapidstream-tapaopt`, autobridge floorplan) | `rapidstream_script/run_rs.sh:3`, `floorplan.py:1` |
| 1차 타깃 디바이스 | Versal V80 HBM `xcv80-lsva4737-2MHP-e-S` | 모든 `Makefile`의 `--part-num`, `device.py:3` |
| 2차 타깃 (lut-dla 단독) | Versal Premium `xcvp1802-lsvc4072-2MP-e-S` | `lut-dla/impl_1_full_util_routed.rpt:8` |
| 클럭 | HLS 3.33 ns 목표 → 구현 250 MHz (4.0 ns) | `Makefile`들의 `--clock-period 3.33`; `arm_bd.tcl:98` (`clk_out1=250.000`) |
| 양자화 | INT4 가중치 + VQ centroid + scale/zero | `floorplan.py:7` (`...final_int4_0`), `arm_bd.tcl:195,279,289` |
| 대상 모델 | Qwen 디코더 블록 (RMSNorm+RoPE+GQA+SiLU FFN) | 모듈 디렉토리명 + `qwen_lut_model/` |

LUT-LLM은 LUT-DLA 계열(가중치를 소수의 centroid로 VQ 후, 입력×centroid의 부분곱을 LUT로 사전계산하여 런타임에 인덱스 조회로 matmul을 대체)을 LLM 디코더 블록으로 확장한 것으로 `추정`된다. (figs/`lutlm.png`, `lutlinear.png` 가 이를 시사하나 이미지라 내용 `확인 불가`.)

---

## 2. 디렉토리 구조

### 2-1. 자체 산출물 (분석 대상)

```
LUT-LLM/
├─ create_bd_design_final.tcl        # Vivado IPI BD 생성 (Serpens 백본 통합)
├─ .gitignore                        # build/, *.log, proj_* 제외
│
├─ lut-dla/                          # ★ LUT-DLA 단독 코어 (VQ matmul 엔진)
│   ├─ Makefile                      #   TOP=lut_dla_core, tapa
│   ├─ lut_dla_core.xo               #   빌드 산출물(바이너리)
│   └─ impl_1_full_util_routed.rpt   #   VP1802 구현 리소스 리포트
│
├─ qwen_block/                       # ★ Qwen 디코더 블록 전체 (top-level 가속기)
│   ├─ Makefile                      #   TOP=qwen_block, csim/csim_decode/e2e_latency/hls
│   ├─ qwen_block.xo                 #   빌드 산출물
│   ├─ qwen_v80.pdi                  #   V80 비트스트림/PDI
│   ├─ timing.rpt                    #   V80 post-route 타이밍 (WNS 0.000)
│   └─ example.pwr                   #   전력 리포트 (62.751 W)
│
├─ attention_block/Makefile          # TOP=attention_block
├─ gqa/Makefile                      # TOP=gqa (Grouped-Query Attention)
├─ ffn/Makefile                      # TOP=ffn_core
├─ rms_norm/Makefile                 # TOP=rms_norm_top
├─ rope/Makefile                     # TOP=rope (Rotary Pos. Embedding)
├─ silu/Makefile                     # TOP=silu_top
├─ ccu/Makefile                      # TOP=ccu_fp32_top (FP32 연산 유닛 추정)
├─ imm/Makefile                      # TOP=imm (중간 단계 모듈 추정)
│
├─ rapidstream_script/               # RapidStream(autobridge) 백엔드
│   ├─ run_rs.sh                     #   rapidstream-tapaopt 실행
│   ├─ device.py                     #   V80 가상 디바이스(3×1 슬롯) 생성
│   ├─ floorplan.py / floorplan_config.json
│   ├─ pipeline.py  / pipeline_config.json
│   └─ v80_device.json               #   슬롯별 LUT/FF/BRAM/DSP/URAM 용량
│
├─ custom_design/                    # Vivado 통합 (PS + NoC + HBM + DUT)
│   ├─ arm_bd.tcl                    #   prefill BD (55 HBM 포트)
│   ├─ arm_bd_wide.tcl               #   decode BD (28 HBM 포트)
│   ├─ run.tcl                       #   Vivado 프로젝트/합성/구현 플로우
│   └─ constraint.tcl                #   클럭/리셋 더미 핀
│
├─ qwen_lut_model/                   # 성능 모델 산출물(PNG만, 소스 없음)
│   └─ roofline_model.png, throughput_vs_seqlen.png, param_comparison_*.png
└─ figs/                             # 논문/문서용 그림 (PNG)
```

### 2-2. 제외 (third-party / vendor / 생성물 — 이름만)

- `.git/` 전체 — Git 내부 (pack, hooks, refs)
- `*.xo`, `*.pdi`, `*.rpt`, `*.pwr` — TAPA/Vivado **빌드 산출물** (바이너리/리포트, 소스 아님)
- `figs/*.png`, `qwen_lut_model/*.png` — 이미지 산출물 (텍스트 분석 `확인 불가`)
- `serpens32_none/` (참조만, repo에 미포함) — **Serpens** TAPA SpMV 가속기 템플릿을 RTL 백엔드 프로젝트로 사용 (`run.tcl:19-22`, `create_bd_design_final.tcl:42` "module references: Serpens")

> ⚠️ **결정적 사실**: Makefile들이 참조하는 자체 HLS 소스 — `lut_dla.h`, `qwen_block.h`, `qwen_block_tb.cpp`, `attention_block.h`, `gqa.h`, `ffn.h`, `rms_norm.h`, `rope.h`, `silu.h`, `ccu_fp32.h`, `imm.h`, `e2e_latency.cpp` 등 — 은 **디스크에 존재하지 않는다** (Glob 전수 확인: `lut-dla/*`, `qwen_block/*`, 각 모듈 dir에 `.xo`/`.rpt`/`Makefile`만). 따라서 함수/라인 단위 알고리즘 분석은 불가하며, 본 문서는 **인터페이스/포트/이름 규약 + 빌드 메타데이터** 기반 역공학이다.

---

## 3. 핵심 모듈 정밀 분석

> 소스 부재로 "라인 근거"는 **Makefile(타깃·TB 구성)**, **arm_bd.tcl(HBM 포트 맵 = 커널 입출력 시그니처)**, **device/floorplan 스크립트(셀 이름·자원)**, **리포트(자원·타이밍·전력)** 에서 취한다.

### 3-1. `qwen_block` — Qwen 디코더 블록 통합 커널 (top-level 가속기) ★★★

`qwen_block`이 **실제 칩에 올라가는 top-level DUT**이다. `custom_design/arm_bd.tcl:179`에서 `create_bd_cell -type module -reference qwen_block dut_0`로 BD에 인스턴스되고, V80 NoC를 통해 HBM에 직결된다.

#### (a) 커널 외부 인터페이스 = HBM 포트 맵 (Qwen 블록 데이터 구조 직접 노출)

`arm_bd.tcl`의 `connect_bd_intf_net $dut/m_axi_*` 라인들이 `qwen_block` 커널의 AXI 마스터 포트 전체를 드러낸다 (prefill 변형, 55 채널):

| AXI 마스터 포트 (커널 인자) | 역할 (추정) | 근거 라인 |
|---|---|---|
| `m_axi_lut_weight_idx_buffer_0..50` (총 51개) | **VQ 가중치 인덱스 버퍼** — LUT 매칭용 centroid 인덱스. 압도적 다수 채널 = 가중치 대역폭이 병목 | `arm_bd.tcl:195-451` |
| `m_axi_centroid_buffer_0..3` | **VQ 코드북(centroid)** — 인덱스가 가리키는 실제 벡터 값 | `arm_bd.tcl:279,284,466,471` |
| `m_axi_scale_zero_buffer` | **양자화 scale/zero-point** (dequant 파라미터) | `arm_bd.tcl:289` |
| `m_axi_input_buffer_0..3` | 입력 액티베이션(토큰 임베딩/이전 레이어 출력) | `arm_bd.tcl:259-274` |
| `m_axi_out_buffer_0..3` | 블록 출력 | `arm_bd.tcl:324-340` |
| `m_axi_k_cache_buffer_0..1`, `m_axi_v_cache_buffer_0..1` | **KV 캐시** (자기회귀 디코딩) | `arm_bd.tcl:248,253,456,461` |
| `m_axi_sin_buffer_0..1`, `m_axi_cos_buffer_0..1` | **RoPE** sin/cos 테이블 | `arm_bd.tcl:294-309` |
| `m_axi_rms_norm_weight_buffer_0..1` | **RMSNorm** 가중치(γ) | `arm_bd.tcl:314,319` |

> 이 포트 맵은 `qwen_block`이 **RMSNorm → (RoPE 적용) GQA → RMSNorm → SiLU-gated FFN** 의 디코더 블록 전 단계를 단일 모놀리식 커널 안에 dataflow로 융합했음을 시사한다 (RMSNorm γ, RoPE sin/cos, KV cache, VQ 가중치/centroid가 모두 한 커널의 인자로 모임). `확인`(포트 존재) + `추정`(내부 단계 융합 순서).

#### (b) 두 가지 변형: prefill vs decode

- **prefill** (`arm_bd.tcl`): 55 HBM 채널, `lut_weight_idx_buffer`를 51개 분산, 채널당 `read_bw 2000 / write_bw 2000` (`arm_bd.tcl:192-471`). 토큰 병렬 처리로 가중치 재사용↓ → 가중치 채널 대량 필요.
- **decode** (`arm_bd_wide.tcl`): 28 HBM 채널, `lut_weight_idx_buffer`를 16개로 축소, 채널당 `read_bw 13900 / write_bw -100`(읽기 전용 극대화) (`arm_bd_wide.tcl:189-349`). 배치=1 자기회귀라 가중치 read 대역폭이 절대 병목 → 채널당 대역폭을 6.95배(2000→13900) 높임. KV cache는 `read 6500/write 6500` 양방향 (`arm_bd_wide.tcl:221,341`).
- Makefile에서도 두 TB 분리: `csim`(prefill)과 `csim_decode`(decode) (`qwen_block/Makefile:4-8`).

> **이는 LLM 가속기 설계의 정석적 분리**: prefill=compute-bound(토큰 병렬), decode=memory-bound(가중치 스트리밍). LUT-LLM은 이를 **별도 HBM 토폴로지/BW 할당**으로 대응. `확인`(BW 수치 라인).

#### (c) 구현 결과 (V80, post-route)

- **타이밍**: `clk_pl_0` / `clkout1` 기준 **WNS 0.000 ns, TNS 0.000, 위반 endpoint 0 / 총 5,073,980 endpoint** → 250 MHz 타이밍 완전 충족 (`timing.rpt:148-153, 176-179`). 디바이스 `xcv80-lsva4737 -2MHP` (`timing.rpt:8-9`).
- **전력**: **Total On-Chip 62.751 W** (예산 70 W), ACAP 47.420 W (static 22.036 / dynamic 25.384), Junction 100 °C (`example.pwr:60-64,81`). PL static 18.66 W가 정적 전력 지배 (`example.pwr:101`).
- 빌드: Vivado 2024.2, 8-job impl, `AggressiveExplore` 라우팅, phys_opt 활성 (`custom_design/run.tcl:51-59`).

### 3-2. `lut-dla` — LUT-DLA VQ matmul 코어 ★★★

LUT-LLM의 **연산 심장부**. `lut-dla/Makefile:2`에서 `TOP=lut_dla_core`. `floorplan.py:6`이 핵심 셀 이름을 노출:

```python
cell_pre_assignments={
    "memory_matcher_w_vq_half_dsp_final_int4_0": "SLOT_X1Y0:SLOT_X1Y0"
}   # floorplan.py:5-7
```

이름 분해 (역공학):
- `memory_matcher` — matmul을 곱셈이 아닌 **메모리(LUT) 매칭**으로 수행. 입력 벡터를 VQ centroid 인덱스로 매칭 후, 사전계산 부분곱 테이블을 조회·누적.
- `w_vq` — **weight Vector Quantization**: 가중치를 코드북(centroid) 인덱스로 표현.
- `half_dsp` — DSP를 절반만 사용(LUT가 곱셈 대체) → DSP-light, LUT-heavy 데이터패스. `추정`(이름 근거).
- `final_int4` — **INT4** 인덱스/양자화 비트폭.

핵심 셀을 `SLOT_X1Y0` 단일 슬롯에 고정 배치 (`floorplan.py:7`) → 대형 단일 매크로 엔진임을 시사.

#### lut-dla 단독 구현 리소스 (VP1802, `impl_1_full_util_routed.rpt`)

> ⚠️ 이 리포트는 V80이 아닌 **Versal Premium `xcvp1802`** 타깃, design명 `ext_platform_wrapper` (`rpt:7-8`). lut-dla 코어를 별도 평가 플랫폼에서 측정한 것으로 `추정`.

| 자원 | 사용 | 가용 | Util% | 근거 |
|---|---|---|---|---|
| CLB LUTs | 505,450 | 3,360,896 | 15.04% | `rpt:43` |
| └ LUT as Logic | 388,908 | — | 11.57% | `rpt:44` |
| └ LUT as Memory | 116,542 | — | 6.94% | `rpt:45` |
| └ Distributed RAM | 28,672 | — | — | `rpt:46` |
| └ Shift Register(SRL) | 87,870 | — | — | `rpt:47` |
| Registers(FF) | 465,842 | 6,721,792 | 6.93% | `rpt:40` |
| URAM | 128 | 2,549 | 5.02% | `rpt:100` |
| BRAM Tile | 0.5 | 4,941 | 0.01% | `rpt:95` |
| **DSP Slices** | **1,105** | 14,352 | 7.70% | `rpt:113` |
| └ DSPFP32 | 1,040 | — | — | `rpt:116` |
| └ DSP58 | 65 | — | — | `rpt:114` |

> **핵심 통찰**: BRAM ≈ 0(0.5 tile), URAM·DSP도 매우 낮은 사용률인데 **LUT-as-Memory(116K) + Distributed RAM(28K) + SRL(87K)** 가 두드러진다. 이는 "곱셈/대용량 온칩 메모리 대신 **분산 LUT 메모리로 부분곱 테이블을 들고 조회**"하는 LUT-DLA 철학과 정확히 일치 (`rpt:45-49`). DSPFP32 1,040개는 LUT 매칭 후 **누적/스케일(dequant)·정규화·activation** 등 잔여 FP 연산용으로 `추정`.

### 3-3. `gqa` — Grouped-Query Attention ★★

`gqa/Makefile:2` `TOP=gqa`, TB `gqa_tb.cpp`. Qwen 계열은 GQA(K/V head를 query head보다 적게 공유)를 사용 → `qwen_block`의 KV cache 포트(`k/v_cache_buffer`)와 결합되어 어텐션을 수행. 독립 csim/hls 타깃 존재(`gqa/Makefile:4-8`)로 **단위 검증 가능한 모듈식 설계**. 내부 알고리즘(score=QKᵀ/√d, softmax, ×V) 라인 근거는 소스 부재로 `확인 불가`.

### 3-4. `attention_block` — 어텐션 통합 블록 ★★

`attention_block/Makefile:2` `TOP=attention_block`, TB `attention_block_tb.cpp`. `gqa`보다 상위 — RoPE 적용 + GQA + KV cache write-back을 묶은 어텐션 서브시스템으로 `추정`. `qwen_block`이 이 블록을 흡수/재사용했을 가능성(`추정`, 소스로 `확인 불가`).

### 3-5. 나머지 모듈 (요약, 각 Makefile 근거)

| 모듈 | TOP | TB | 역할 | 근거 |
|---|---|---|---|---|
| `ffn` | `ffn_core` | `ffn_tb.cpp` | SwiGLU/SiLU-gated FFN (up/gate/down proj) | `ffn/Makefile:2,4` |
| `silu` | `silu_top` | `silu_tb.cpp` | SiLU activation (x·σ(x)), FFN gate | `silu/Makefile:2,4` |
| `rms_norm` | `rms_norm_top` | `rms_norm_tb.cpp` | RMSNorm (Qwen 정규화) | `rms_norm/Makefile:2,4` |
| `rope` | `rope` | `rope_tb.cpp` | Rotary Positional Embedding (sin/cos) | `rope/Makefile:2,4` |
| `ccu` | `ccu_fp32_top` | `ccu_fp32_tb.cpp` | FP32 연산 유닛(Compute/Central Unit 추정); `--enable-synth-util`로 자원 측정 활성 | `ccu/Makefile:2,4,8` |
| `imm` | `imm` | `imm_tb.cpp` | 중간(intermediate) 모듈; `-g` 디버그 csim | `imm/Makefile:2,5` |

> 모든 모듈이 **동일 패턴**: `tapa g++`로 csim(소프트 검증) → `tapa compile`로 `.xo` 생성, 동일 `--part-num xcv80 --clock-period 3.33`. 모듈식으로 각각 단독 검증 후 `qwen_block`에 통합하는 워크플로 (`확인`, 8개 Makefile 동형).

---

## 4. 데이터 플로우

### 4-1. 추론 데이터 플로우 (qwen_block 내부, 추정)

```
[HBM] input_buffer ─┐
                    ▼
            RMSNorm (rms_norm_weight_buffer:γ)         ← rms_norm
                    ▼
   ┌──── Q/K/V projection (LUT-VQ matmul) ────┐         ← lut_dla_core (memory_matcher_w_vq)
   │  lut_weight_idx_buffer(인덱스) +          │
   │  centroid_buffer(코드북) + scale_zero(dequant)
   ▼
  RoPE (sin/cos_buffer) ─► GQA (k/v_cache_buffer R/W) ─► softmax·×V   ← rope + gqa/attention_block
                    ▼
            RMSNorm
                    ▼
   SiLU-gated FFN: gate=SiLU(W_g·x), up=W_u·x, down=W_d·(gate⊙up)      ← ffn + silu (+ LUT-VQ matmul)
                    ▼
              [HBM] out_buffer
```

근거: 포트 맵(`arm_bd.tcl:248-340`) + 모듈 Makefile 집합. 단계 **순서·융합**은 `추정`(소스 `확인 불가`), 단계 **구성요소 존재**는 `확인`(포트/모듈).

### 4-2. 빌드 데이터 플로우 (확인)

```
*.h (HLS C++, 부재) ──tapa compile──► *.xo
                                       │  qwen_block/Makefile:13-14
                                       ▼
                rapidstream-tapaopt (run_rs.sh:3)
                  ├─ device.py → v80_device.json (3×1 슬롯, 슬롯별 자원)
                  ├─ floorplan.py → floorplan_config.json (DSE 0.63~0.90, 셀 고정)
                  └─ pipeline.py → pipeline_config.json (pp_scheme="double")
                                       │  → SLR-aware floorplan + 자동 파이프라인
                                       ▼
            Vivado (custom_design/run.tcl)
              ├─ serpens32_none/rtl 임포트 + arm_bd.tcl (PS+NoC+HBM+DUT)
              └─ synth_1 → impl_1 (Explore/AggressiveExplore) → qwen_v80.pdi
```

---

## 5. HW/SW 매핑

| 계층 | 구현 | 자원/디바이스 | 근거 |
|---|---|---|---|
| **SW (host/PS)** | Versal CIPS (Cortex-A72), AXI-Lite control, INTC | `versal_cips:3.4`, `axi_intc:4.1` | `arm_bd.tcl:36,82,485-486` |
| **인터커넥트** | AXI-NoC (CIPS NoC 8 SI + DUT NoC), HBM 채널 라우팅, smartconnect 제어 경로 | `axi_noc:1.1`, `smartconnect:1.0` | `arm_bd.tcl:48,112,164` |
| **메모리** | HBM2e — prefill 55 / decode 28 채널, 채널당 2000~13900 BW | `HBM_NUM_CHNL 16`, `NUM_HBM_BLI 55/28` | `arm_bd.tcl:189`, `arm_bd_wide.tcl:189` |
| **HW (PL 커널)** | `qwen_block` 단일 HLS 커널, 250 MHz | V80 PL: WNS 0.0, 62.75 W | `timing.rpt:150`, `example.pwr:60` |
| **연산 엔진** | `memory_matcher_w_vq_half_dsp_final_int4` (LUT-VQ matmul) | LUT-Memory heavy, DSP-light | `floorplan.py:6`, `lut-dla rpt:45,113` |
| **플로어플랜** | RapidStream 3×1 슬롯, 슬롯간 wire capacity 20000, DSE 0.65~0.90 | v80 6슬롯 자원표 | `device.py:8-21`, `v80_device.json` |

- **클럭 도메인**: `clk_wizard_0` clk_out1 = **250 MHz**가 DUT/NoC/control 단일 도메인 (`arm_bd.tcl:98,483`). PS ref clk = 99.999 MHz (`arm_bd.tcl:40` `PMC_CRP_PL0_REF_CTRL_FREQMHZ`).
- **제어 경로**: `icn_ctrl/M01_AXI → dut_0/s_axi_control`, `dut_0/interrupt → axi_intc` (`arm_bd.tcl:485-486`).

---

## 6. 빌드·실행

### 6-1. 단위 모듈 (예: qwen_block)

```bash
# C 시뮬레이션 (소프트 검증) — prefill / decode
make csim          # tapa g++ -- qwen_block.h qwen_block_tb.cpp -o qwen_block
make csim_decode   # ... qwen_block_decode_tb.cpp
# 분석용 e2e latency 모델 (순수 C++)
make e2e_latency   # g++ -std=c++17 -O2 e2e_latency.cpp
# HLS 합성 → .xo
make hls           # tapa compile --top qwen_block --part-num xcv80-... --clock-period 3.33
```
근거: `qwen_block/Makefile:4-17`. (⚠️ 참조 소스 부재로 실제 실행 불가 — `확인 불가`.)

### 6-2. RapidStream 백엔드

```bash
python device.py      # v80_device.json 생성
python floorplan.py   # floorplan_config.json
python pipeline.py    # pipeline_config.json
bash run_rs.sh        # rapidstream-tapaopt -j6 --tapa-xo-path qwen_block.xo ...
```
근거: `rapidstream_script/{device,floorplan,pipeline}.py`, `run_rs.sh:3-7`.

### 6-3. Vivado 통합/구현

```bash
vivado -mode batch -source custom_design/run.tcl
#  → vivado_proj 생성, serpens32_none/rtl 임포트, arm_bd.tcl source,
#     synth_1 → impl_1(Explore) → 비트스트림
```
근거: `custom_design/run.tcl:19-59`. 환경변수 `VIVADO_SYNTH_JOBS` (`run.tcl:2,45`).

---

## 7. 의존성

| 의존성 | 용도 | 근거 |
|---|---|---|
| **TAPA** (`tapa`) | HLS 컴파일·csim (task-parallel HLS) | 전 Makefile `TOOL=tapa` |
| **RapidStream** (`rapidstream`, `rapidstream-tapaopt`) | SLR floorplan/pipeline 자동화 | `floorplan.py:1`, `run_rs.sh:3` |
| **Vivado 2024.2** | 합성/구현/비트스트림 | `timing.rpt:3`, `run.tcl` |
| **Power Design Manager 2024.1.2** | 전력 추정 | `example.pwr:3` |
| **Serpens** (TAPA SpMV 가속기) | RTL 백엔드 프로젝트 템플릿(`serpens32_none/`) | `run.tcl:19-22`, `create_bd_design_final.tcl:42` |
| Xilinx IP | versal_cips, axi_noc, clk_wizard, smartconnect, proc_sys_reset, axi_intc, axis_ila, hw_discovery | `arm_bd.tcl`, `create_bd_design_final.tcl:53-60` |

> Serpens 의존은 주목할 점: LUT-VQ 가중치 매칭이 **희소 행렬(SpMV) 데이터패스**와 구조적으로 유사(인덱스 기반 비정형 접근)하여 Serpens 인프라(HBM 스트리밍·NoC 결선)를 재활용한 것으로 `추정`.

---

## 8. 강점·한계

### 강점
1. **DSP 우회 LUT-VQ matmul** — `half_dsp_final_int4` 엔진으로 곱셈을 LUT 메모리 조회로 치환. lut-dla 리포트에서 LUT-as-Memory(116K)/Dist.RAM/SRL이 지배, BRAM≈0(`rpt:45,95`) → DSP 한정 디바이스에서 유리.
2. **prefill/decode 분리 최적화** — 동일 커널을 두 HBM 토폴로지(55ch 2000BW vs 28ch 13900BW)로 빌드, LLM 두 단계의 상반된 병목에 각각 대응 (`arm_bd.tcl` vs `arm_bd_wide.tcl`).
3. **HBM 직결 + NoC** — 55채널 동시 가중치 스트리밍으로 decode memory-bound 완화 (`arm_bd.tcl:473`).
4. **모듈식 검증** — 8개 서브모듈 각각 독립 csim/hls (rms_norm/rope/gqa/silu/ffn/...), 통합 전 단위 검증 가능.
5. **타이밍·전력 마감** — 250 MHz WNS 0.0, 62.75 W < 70 W 예산 (`timing.rpt:150`, `example.pwr:60,81`).
6. **자동화 백엔드** — RapidStream floorplan/pipeline로 대규모 multi-SLR 설계의 타이밍 클로저 자동화.

### 한계
1. **소스 코드 전무** — repo에 HLS `.cpp`/`.h` 0개. 알고리즘 정밀도(VQ 그룹 크기, centroid 개수, 부분곱 테이블 차원, INT4 정확도)는 모두 `확인 불가`. 본 분석은 산출물/포트 역공학.
2. **재현 불가** — `make csim/hls`가 부재 소스 참조 → 빌드 불가능. 산출물(`.xo`,`.pdi`)만 제공.
3. **양자화 정확도 데이터 부재** — `qwen_lut_model/`은 PNG뿐(roofline, throughput) → perplexity/정확도 손실 수치 `확인 불가`.
4. **타깃 분산** — lut-dla는 VP1802, qwen_block은 V80. 두 평가를 직접 합산 비교 어려움 (`rpt:8` vs `timing.rpt:8`).
5. **Junction 100 °C / Thermal Margin 0 °C** — 전력 마진이 빠듯 (`example.pwr:64-65`).
6. **단일 클럭 250 MHz** — HLS 목표 300 MHz(3.33 ns)였으나 250 MHz로 마감 (`Makefile` 3.33 vs `arm_bd.tcl:98`).

---

## 9. 우리 프로젝트(PRJXR-HBTXR: 고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적) 시사점

> 전제: 우리 프로젝트는 HG-PIPE 계열 고처리량 ViT 가속기 + XR eye-tracking으로 `추정`.

1. **LUT-VQ matmul의 ViT 적용 검토** — LUT-LLM의 `memory_matcher_w_vq` 철학(곱셈→LUT 조회)은 **DSP가 부족한 임베디드 XR FPGA에서 ViT 선형층 가속**에 직접 유효. HG-PIPE의 DSP-heavy systolic array를 보완해, ViT의 QKV/FFN projection을 INT4 VQ로 LUT 오프로드하면 DSP 압력 완화 가능. (단, ViT는 LLM과 달리 시퀀스가 짧고 배치가 있으므로 prefill 토폴로지에 가까움.)
2. **prefill/decode 이원화 → ViT는 prefill-like** — XR 시선추적 ViT 추론은 **단일 프레임 전체 토큰 병렬**(=prefill)에 해당. `arm_bd.tcl`의 prefill 다채널 가중치 스트리밍 토폴로지를 참고하되, eye-tracking은 저지연 단일 프레임이라 **on-chip weight resident**가 더 적합할 수 있음(HBM 스트리밍 불필요).
3. **TAPA+RapidStream 백엔드 채택 가치** — 모듈식 HLS(rms_norm/rope/gqa/ffn 분리) + RapidStream 자동 floorplan은 우리 가속기의 타이밍 클로저·multi-SLR 확장에 재사용 가능한 검증된 플로우. 특히 `device.py`의 슬롯 자원 모델링 방식 참고.
4. **모놀리식 블록 융합 vs HG-PIPE 파이프라인** — LUT-LLM은 디코더 블록 전체를 단일 커널에 융합(dataflow). HG-PIPE는 레이어 파이프라인 스트리밍. XR 저지연 요구상 **HG-PIPE식 layer-pipeline + LUT-VQ PE**의 하이브리드가 유망 `추정`.
5. **전력 예산 교훈** — 62.75 W는 데이터센터급. XR 디바이스(수 W급)에는 **VQ centroid 수·INT4 비트폭을 더 공격적으로 축소**하고 HBM 대신 LPDDR/온칩 메모리 사용 필요.
6. **가져올 자산** — RapidStream 스크립트 3종(`device/floorplan/pipeline.py`), `arm_bd.tcl`의 NoC/HBM 결선 패턴, prefill/decode BW 튜닝 표는 우리 V80/Versal 타깃 시 직접 템플릿화 가능.

---

## 10. 근거 표기 요약

- **확인(라인 근거)**: 디렉토리 구조, 8개 모듈 TOP/TB명·빌드 커맨드(각 Makefile), HBM 포트 맵·BW(arm_bd*.tcl), 핵심 셀명(floorplan.py:6), V80 타이밍 WNS 0.0(timing.rpt:150), 전력 62.75W(example.pwr:60), lut-dla 자원(rpt:40-119), RapidStream 설정(device/floorplan/pipeline.py), Serpens 의존(run.tcl:19, final.tcl:42).
- **추정**: 내부 연산 알고리즘·단계 융합 순서, `half_dsp`/`imm`/`ccu` 의미, VP1802가 lut-dla 평가 플랫폼인 점, Serpens 재활용 동기, ViT 적용 시사점.
- **확인 불가**: 모든 HLS 소스 내용(소스 부재), VQ 하이퍼파라미터(centroid/그룹/테이블 차원), 양자화 정확도 손실, qwen_lut_model PNG·figs PNG 내용, e2e_latency.cpp 로직, 빌드 재현.
