# XJTU-Tripler (HiPU) 정밀 분석

> 분석 대상 repo: `REF/CNN-Accel/XJTU-Tripler-master`
> 분석 방식: 자체 RTL(.v) / C·asm / Python 소스를 직접 Read하여 라인 근거 기반 작성. 추측은 "추정", 미확인은 "확인 불가"로 명시.
> 작성일: 2026-06-20

---

## 1. 개요

| 항목 | 내용 |
|---|---|
| **목적** | DAC'19 System Design Contest(FPGA Track) 단일 물체 검출용 CNN 가속기. ShuffleNet V2 + YOLO 기반 ShuffleDet을 8bit 양자화로 FPGA에 가속. (`README.md:1-3, 53-59, 74-76`) |
| **한줄 요약** | RISC-V(picorv32) 스케줄러가 레지스터맵으로 제어하는 **명령어 기반(fine-grained ISA) DNN 가속기 HiPU**. MAC/Vector 연산코어는 암호화 netlist(.edf), 컨트롤·메모리계층·DDR·SW는 완전 RTL/C 소스. |
| **원논문/대회** | DAC'19 SDC 2위 (XJTU-Tripler). 후속 저널: Zhao W. et al., "HIPU: A Hybrid Intelligent Processing Unit With Fine-Grained ISA...", IEEE TVLSI 31(12):1980-1993, 2023 (`README.md:318-333`) |
| **타깃 디바이스** | Xilinx Ultra96 (Zynq UltraScale+ **ZU3**: LUT 70K / FF 141K / BRAM 216 / DSP 360). HiPU는 PL 전용, PS는 이미지 IO만. 동작 233MHz, 피크 268Gops, conv 효율 >80% (`README.md:30-36, 71-72, 152-154`) |
| **성과** | IoU 0.615, 9248mW, 50.91 FPS. 자원: LUT 65% / FF 41% / BRAM 91% / DSP 98%, Verilog 구현 (`README.md:284, 289`) |

**연산 지원**: CONV, FC, Depth-wise CONV, Pooling, Element-wise Add/Mul, Channel shuffle/divide/concat(무비용 융합) (`README.md:63-67, 160-166`).

---

## 2. 디렉토리 구조

### 2.1 자체 핵심 소스 (분석 대상)

```
XJTU-Tripler-master/
├── README.md                         # 설계 전체 설명(알고리즘+HW)
├── hpu_core/                         # HiPU RTL (자체 핵심)
│   ├── constrs/hpu_reg.xdc           # FPGA 제약(클럭/핀)
│   └── src/
│       ├── ps_pl_top.v               # PL 최상위(ZYNQ PS 연결, AXI DDR)
│       ├── common/                   # 공용 RTL 헬퍼
│       │   ├── dec_bin_to_onehot.v   # binary→one-hot 디코더
│       │   ├── reg_dly.v             # 지연 레지스터
│       │   └── sync_fifo.v           # 동기 FIFO
│       └── pl_top/
│           ├── pl_top.v / pl_regs.v / clk_gen.v / rst_gen.v / riscv_top.v
│           ├── dl_dtcm_ctrl.v / dl_itcm_ctrl.v   # TCM 다운로드 컨트롤
│           ├── picorv_top/           # RISC-V 서브시스템(picorv32=3rd-party)
│           └── hpu_top/              # ★ HiPU 본체
│               ├── hpu_top.v         # HiPU 통합(2-core 구조, 현재 1코어 활성)
│               ├── hpu_ctrl.v        # 컨트롤러 집합(conv/dwc/dtrans/ftrans/mpu/vpu/vputy_ctrl)
│               ├── regmap_mgr.v      # RISC-V↔레지스터맵 디스패처
│               ├── load_mtxreg_ctrl_v2.v / save_mtxreg_ctrl.v / save_mtxreg_data_wrap.v
│               ├── pal_reoder_pic.v  # ★ 입력영상 3ch→채널확장 reorder
│               ├── ddr_intf.v + ddr_intf/{ddr_arbiter_r,ddr_arbiter_w}.v
│               ├── hpu_ctrl/         # 각 연산 스케줄러 FSM
│               │   ├── conv_ctrl.v   # ★ convolution 스케줄러(주소생성 FSM)
│               │   ├── dwc_ctrl.v    # depth-wise conv 스케줄러
│               │   ├── datatrans_ctrl.v / fmttrans_ctrl.v  # 데이터/포맷 전송
│               │   ├── mpu_ctrl.v / vpu_ctrl.v / vputy_ctrl.v  # 연산코어 명령 디코더
│               └── hpu_core/         # ★ 데이터패스 본체
│                   ├── hpu_core.v    # MPU/VPU/VPUTY + 레지스터파일 통합
│                   ├── mtxrega/b/c.v # 행렬 레지스터(SRAM 기반)
│                   ├── vecreg.v / biasregb.v / biasregc.v
│                   ├── mtxreg_hub.v  # DDR↔레지스터 데이터 라우팅
│                   ├── mpu_stub.v / vpu_stub.v / vputy_stub.v   # IP 블랙박스 포트선언
│                   └── *.edf / mpu/*.edn / vputy/*.edn          # ← 암호화 netlist(생성물 아님이나 비공개 IP)
└── hpu_software/                     # RISC-V 펌웨어(HiPU 스케줄링)
    ├── makefile / config.lds / memory_map.inc
    ├── inc/{global.h, hpu_api.h, intr.h, shufflenet_v2.h}
    ├── src/{_init.s, hpu_api.s, intr.s, main.c, shufflenet_v2.c, shufflenet_v2_data.c}
    └── gen_verilog_data.py / gen_verilog_data_process.py   # ELF→verilog mem 변환
```

### 2.2 제외 항목 (3rd-party / 생성물 / 비분석)

| 경로/유형 | 제외 사유 |
|---|---|
| `hpu_core/fpga/hpu_reg_version/**` | Vivado 프로젝트 산출물: `.runs`(수백 vrs_config_*.xml), `.Xil`, `.hw`, `.bit`, ILA wcfg/wdb/vcd 등 전부 생성물 |
| `hpu_core/.../hpu_core/*.edf`, `mpu/*.edn`, `vputy/*.edn` | MPU/VPU/VPUTY 연산코어 **암호화 EDIF netlist**(자체 IP이나 RTL 비공개). 본문은 `_stub.v` 포트선언만 분석 가능 |
| `hpu_core/.../picorv_top/picorv32.v` | **3rd-party**: PicoRV32, Clifford Wolf, ISC License (`picorv32.v:1-18`). 통합용으로만 분석 |
| `hpu_software/.vscode/ipch/*.ipch` | IDE 인덱스 캐시(생성물) |
| `src/common/.shift_left_algorithm.v.swp` | vim swap(임시파일) |
| `slides/`, `images/` | 발표자료/이미지 (코드 아님) |

> **확인 불가**: MPU/VPU/VPUTY의 내부 MAC array·PE 구조는 EDIF netlist로만 존재. 포트 폭(아래 §3.4)으로 병렬도는 역산 가능하나 마이크로아키텍처(systolic vs. 곱셈기 트리 등)는 소스 미존재로 **확인 불가**.

---

## 3. 핵심 모듈·파일별 정밀 분석

전체 제어 위계: `ps_pl_top → pl_top → riscv_top(picorv32) + hpu_top`. `hpu_top` 내부는 **(1) regmap_mgr(레지스터맵 디스패처) → (2) hpu_ctrl(스케줄러 FSM 묶음) → (3) hpu_core(데이터패스)** 로 흐른다.

### 3.1 `hpu_top.v` — HiPU 통합 최상위

- 역할: regmap_mgr, hpu_ctrl, hpu_core, load/save_mtxreg_ctrl, ddr_intf를 한 데 묶는 통합 모듈. (`hpu_top.v:425-1089`)
- **핵심 파라미터 (데이터패스 폭의 근원)** (`hpu_top.v:53-67`):
  - `MR_PROC_WTH=8` (행렬 원소 8bit = 양자화 폭), `MR_PROC_H_PARAL=8`, `MR_PROC_V_PARAL=8`
  - `MTX_DATA_WTH = 8×8 = 64`, `MR_DATA_WTH = 64×8 = 512` (행렬 레지스터 1워드 = 512bit)
  - `VR_PROC_WTH=32`(누산 32bit), `VR_PROC_PARAL=64`, `VR_DATA_WTH=64×32=2048`(벡터 레지스터 워드)
  - `DDRIF_DATA_WTH=512` (DDR 내부 버스), 단 외부 AXI는 `axi_ddr_*data[127:0]`로 128bit (`hpu_top.v:95,100`)
- **2-core 구조**: `hpu_core0_inst`는 활성, `hpu_core1_inst`는 주석처리(`hpu_top.v:736-893`). ZU3 자원 제약으로 **본 reg_version은 1코어**. ldmr/svmr에 `_hpu_core_sel`로 코어 선택 로직 존재(`hpu_top.v:894-906`)→원래 2코어 확장 설계.
- 외부 인터페이스: PS↔regmap(`riscv_regmap__*`, `hpu_top.v:75-83`), PL→DDR AXI HP0(`axi_ddr_*`, `hpu_top.v:89-103`).

### 3.2 `regmap_mgr.v` — RISC-V ↔ 레지스터맵 디스패처

- 역할: picorv32가 쓰는 메모리맵 주소(예 `0x20000100`)를 받아 각 연산 컨트롤러(conv/dwc/dtrans/ftrans/ldmr/svmr/dldata/upldata)의 레지스터 쓰기 채널로 분배하고, 각 컨트롤러의 intr를 picorv32 인터럽트(`riscv_regmap__intr_o[7:0]`)로 모은다. (`hpu_top.v:358-423`)
- `fshflg_ps_o`: "모든 레이어 완료" 플래그를 PS로 통지(`hpu_top.v:421-422`).
- 메모리맵 정의는 SW측 `memory_map.inc:26-35`:
  `DPU_REGMGR=0x20000000`, `DPU_CONV=0x20000100`, `DPU_DWCALC=0x20000200`, `DPU_DTRANS=0x20000300`, `DPU_FTRANS=0x20000400`, `DPU_LDMR=0x20000500`, `DPU_SVMR=0x20000600`, `DPU_DLCTRL=0x20000700`, `DPU_UPLCTRL=0x20000800`.

### 3.3 `conv_ctrl.v` — Convolution 스케줄러 (가장 중요, 자체 RTL의 핵심)

순수 RTL로 작성된 **convolution 주소생성·명령발행 FSM**. MAC array는 블랙박스지만, "어떤 IFM/weight 주소를 어떤 순서로 읽어 MPU에 흘려보내는가"의 알고리즘이 전부 여기 있다.

- **레지스터맵 파라미터 디코드**(`conv_ctrl.v:279-312`): host가 쓴 32bit 워드를 ifm_width/ifm_channel/wt_width/wt_height/ofm_channel/stride/dilation/clip/relu_en/bias_en/channel_shuffle_type/channel64_priority/pad_left/pad_offset/각종 mr_index·addr로 분해. 비트 레이아웃은 SW `hpu_api.h:22-31`의 `conv_param` 구조체와 1:1 대응.
- **start 시 파라미터 스냅샷**(`conv_ctrl.v:356-396`): `SET_START` 비트 수신 시 모든 파라미터를 `*_reg` 사본에 래치 → 계산 중 host 재기록과 격리.
- **메인 FSM**(`conv_ctrl.v:91-95, 409-449`): `ST_IDLE → ST_CALC_OCH_FIRST_PH → ST_CALC_OCH_LEFT_PH → ST_WAIT_VPROC → ST_DONE`. FIRST_PH는 첫 출력채널(=MUL 명령), LEFT_PH는 누산(=MAC 명령).
- **5중 루프 카운터**(`conv_ctrl.v:451-489`): 우선순위 `ich_cnt(입력채널) → wtw_cnt(커널폭) → wth_cnt(커널높이) → och_cnt(출력채널) → ifmw_cnt(출력폭)`. 입력채널·커널폭 주소가 연속이라 누산 효율 극대화(`conv_ctrl.v:513-521` 주석).
- **MPU 명령 인코딩**(`conv_ctrl.v:97-101, 703-717`): `MPU_CODE_MMUL=2'h1`(첫 누산), `MPU_CODE_MMAC=2'h3`(누산). type은 `channel64_priority`에 따라 `MPU_TYPE_VM(=1)` / `MPU_TYPE_MM(=0)`.
- **IFM 주소생성 + padding/shift**(`conv_ctrl.v:527-675`): 2개 MPU(`mpu0/mpu1`, dual line core)에 대해 line0/1/2(3-line 캐스케이드) 주소를 각각 base/pre/post로 관리. `pad_left`/`pad_offset`로 좌우 padding 처리. shift-left/shift-right 신호(`mrs0_sl/sr`)로 경계 처리: 주소가 last 초과면 `sl`, init 미만이면 `sr` → 그에 맞춰 post/pre 주소·index 선택(`conv_ctrl.v:657-675`).
- **weight 주소**(`conv_ctrl.v:680-699`): `mrs1`은 weight. 매 사이클 +1, 출력폭 변화 시 base로 리셋.
- **VPU 지연 정합(VPU_DLY=17)**(`conv_ctrl.v:89, 227-237, 742-799`): MPU 누산 결과가 VPU에 도달하는 17사이클 파이프 지연을 맞추기 위해 vpu_code/clip/shfl/bias_addr/sv_addr를 **delay chain**으로 시프트. 이것이 conv→bias→relu→clip→shuffle→store 파이프라인 정렬의 핵심.
- **VPU 연산 인코딩**(`conv_ctrl.v:82-87, 749-757`): `EN_ACT/EN_SUM/EN_BIAS/EN_RELU/EN_SHFL/EN_CHPRI` 각 비트를 update 신호·레지스터값으로 조립. → bias add, ReLU, clip(양자화 재스케일 shift), channel shuffle을 conv 출력에 융합.
- **출력 strobe**(`conv_ctrl.v:798-799`): channel64 우선 시 `dec_bin_to_onehot`로 8-way strobe 생성, 아니면 `8'hff`(전부 기록).

### 3.4 `hpu_core.v` + 연산코어 stub — 데이터패스 본체

- `hpu_core.v`는 다음을 인스턴스화: **mpu, vpu, vputy**(블랙박스) + **vecreg, mtxrega/b/c, biasregb/c, mtxreg_hub**(완전 RTL). (`hpu_core.v:292-733`)
- 연산코어 포트 폭(stub에서 역산한 병렬도·자료형):
  - **MPU**(`mpu_stub.v:5-26`): 입력 `mra__rdata[511:0]`(=IFM 512bit=8ch×8×8), `mrb__rdata[511:0]`(weight), vmode `mrb__vmode_rdata[4095:0]`, 출력 `vr__wdata[2047:0]`(=64×32bit 누산). 즉 **8bit×8bit 곱 → 32bit 누산, 64-lane**. clk_2x 입력 → MAC array는 2× 클럭으로 더블펌핑 추정.
  - **VPU**(`vpu_stub.v:5-31`): `vr__rs0/rs1_rdata[2047:0]` 입력, `mra__wdata[511:0]` 출력 + strobe_h/v[7:0]. 누산결과(2048b)에 sum/clip/bias/relu/shfl 적용 후 8bit 행렬(512b)로 환원. → **양자화 재스케일·activation·shuffle 융합 유닛**.
  - **VPUTY**(vpu tiny, `vputy_stub.v:5-41`): `mra/mrc/brc__rdata[511:0]` 입력, `mra__wdata[511:0]` 출력. op: mul/ldsl/ldsr/acc/max → **depth-wise conv·pooling(max)·데이터 이동** 전용 경량 유닛.
- **MPU_TYPE 의미**: MM(matrix-matrix)=일반 conv, VM(vector-matrix)=channel64 우선 모드(=FC/1×1 conv에 가까운 경로 추정).

### 3.5 `mtxrega.v` — 행렬 레지스터(온칩 SRAM 계층)

- 역할: IFM/OFM/weight를 담는 **멀티포트 SRAM 뱅크**. MPU read, VPU write, VPUTY r/w, save/load 포트를 1:다 중재. (`mtxrega.v:30-108`)
- 구조: `MR_PROC_N_PARAL=8` 뱅크를 `generate`로 복제(`mtxrega.v:179-225`). 각 뱅크는 `sdp_w512x64_r512x64_wrap` (단순 듀얼포트 512deep×64wide BRAM 래퍼) 8개(`MR_PROC_H_PARAL`)로 구성 → 뱅크당 512×512bit.
- **write 중재**(`mtxrega.v:181-195`): vpu / vputy / ldmr(DDR load) 중 one-hot index로 선택. strobe_h(가로 8) & strobe_v(세로 8)로 부분기록 → channel shuffle·divide를 주소·strobe만으로 구현.
- **read + shift**(`mtxrega.v:239-248`): `sl`(좌시프트=상위 MTX 추출+0패딩), `sr`(우시프트), `frcz`(force-zero=padding) 적용 후 index로 뱅크 선택. → conv_ctrl이 만든 sl/sr/frcz가 여기서 실제 시프트로 구현. **이것이 "추가 시간 없는 padding/shuffle"의 메커니즘**.
- 파이프 지연: read 경로 `TOT_DLY=3`, shift 정합 `MRA_DLY=2` (`mtxrega.v:115-116, 254-267`).
- mtxregb(weight·vmode 전용), mtxregc(VPUTY 전용), biasregb/c, vecreg(`vecreg.v`)도 동형의 멀티포트 SRAM 래퍼.

### 3.6 `pal_reoder_pic.v` — 입력영상 채널 reorder (자체 최적화 핵심)

- 역할: 첫 conv1의 입력이 RGB 3채널뿐이라 MPU(8ch 병렬) 효율이 3/8로 떨어지는 문제 해결. **행 방향으로 픽셀을 재배열해 3ch→16ch로 확장** → 효율 0.38→0.56 (`README.md:204-216`, `pal_reoder_pic.v:30-55`).
- 구현: 8개 `pal_reoder_buffer`(BRAM) 병렬(`pal_reoder_pic.v:95-182`), 320픽셀/라인(`PIC_LINE_LEN=320`) 단위 3-cycle 시프트 누적(`ddr_intf_3data_shift`, `pal_reoder_pic.v:282-294`).
- **부호 변환**(`pal_reoder_pic.v:183-247`): 각 바이트 MSB 반전(`~rdata_i[...]`) = unsigned[0,255]→signed 중심화(−128 오프셋) 양자화 정합. `reoder_en_i`(`ldmr_param.reoder_en`, `hpu_api.h:64`)로 활성.

### 3.7 RISC-V 펌웨어 (`hpu_software/`)

- **`main.c`**(`main.c:25-62`): init_intr → 입력 phase별 DDR base addr 설정(`PHASE_CONV1/BLK1_1.../CONVF`) → `dlctrl_set` → 무한루프에서 `intr_stcalc_act` 대기 → `shufflenet_v2()` 실행 → `fshflg_ps()`로 PS 통지.
- **`hpu_api.h`**(`hpu_api.h:22-115`): 명령 구조체(conv/dwcalc/dtrans/ftrans/ldmr/svmr/dlctrl/uplctrl) + set/start/check API. 각 구조체 비트필드 주석이 RTL 레지스터맵과 1:1 대응(설계 단일출처).
- **`shufflenet_v2.c`**(`shufflenet_v2.c:1-160`): ShuffleNet V2 네트워크를 HiPU 명령 시퀀스로 수작업 스케줄. matrix register 주소표를 정적 배열로 정의(`:20-32`), conv1_pool1 등 함수가 ldmr(weight/bias/ifm load)→conv_set/conv_start→check를 순차 발행. **inter-layer 캐스케이드**(bottleneck 단위로 DDR 왕복 최소화, `README.md:179-202`)가 이 스케줄에 구현됨.
- **빌드 산출물 변환**: `gen_verilog_data_process.py`(`:1-91`)가 `hpu.hex`(ELF→verilog)를 파싱해 ITCM(코드, `0x80000000~`)·DTCM(데이터, `0x90000000~`)을 분리, 640워드 패킷 정렬로 `1_itcm.verilog`/`1_dtcm.verilog` 생성 → 시뮬레이터/메모리 초기화에 사용.

### 3.8 공용 RTL 헬퍼 (`src/common/`)

- `dec_bin_to_onehot.v`(`:30-52`): 파라미터화 binary→one-hot. conv_ctrl·mtxrega의 index/strobe 디코드에 광범위 사용.
- `reg_dly.v`, `sync_fifo.v`: 지연 레지스터·동기 FIFO(`save_mtxreg_data_wrap`의 데이터 FIFO 등에 사용 추정).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 메모리 계층
```
외부 LPDDR(ZYNQ PS, AXI HP0 128bit)
   │  ddr_intf(512bit 내부버스, r/w arbiter)
   ▼
온칩 행렬 레지스터(mtxrega/b/c, BRAM 멀티뱅크, 512×512b/뱅크 ×8)  ← IFM/OFM/weight
   │  (load_mtxreg_ctrl: reorder 옵션, save_mtxreg_ctrl: FIFO 경유 writeback)
   ▼
연산코어(MPU 64-lane MAC / VPU activation / VPUTY dwconv·pool)
   │
   ▼
vecreg(2048b 누산버퍼) → VPU 재양자화 → 다시 mtxrega
```

### 4.2 실행 파이프라인 (1 layer)
1. picorv32가 `ldmr_*`로 weight/bias/IFM을 DDR→mtxreg 로드(첫 conv는 `reoder_en`로 채널확장).
2. `conv_set`→`conv_start`: conv_ctrl FSM이 5중 루프로 mtxrega(IFM)·mtxregb(weight) 주소를 매 사이클 생성, MPU에 MUL/MAC 명령 스트림 발행.
3. MPU가 8b×8b→32b 누산을 64-lane 수행, 결과 vecreg 적재.
4. 17사이클 지연 정렬 후 VPU가 sum/bias/ReLU/clip(재스케일)/channel-shuffle을 융합 적용 → 8b로 환산, mtxrega에 store.
5. depth-wise/pooling은 VPUTY 경로(`dwc_ctrl`/`vputy_ctrl`).
6. layer 종료 시 `svmr_*`로 OFM을 DDR writeback(또는 inter-layer 캐스케이드면 온칩 잔류).

### 4.3 병렬화·자료형
- **공간 병렬**: 8(H)×8(V) 행렬 처리 × 64 누산 lane (`hpu_top.v:53-66`).
- **클럭**: core clk + clk_2x(MPU 더블펌핑 입력, `mpu_stub.v:11`). 보드 233MHz(`README.md:153`).
- **양자화**: weight·activation 모두 **8bit 대칭 양자화**, BN 융합, 재학습 fine-tune(`README.md:122-137`). 누산 32bit, VPU에서 clip(shift) 재스케일. `pal_reoder_pic`의 MSB 반전으로 unsigned↔signed 정합.
- **무비용 융합**: channel shuffle/divide/concat을 SRAM 주소·strobe로 구현 → 추가 사이클 0(`README.md:164-166`, `mtxrega.v:190-195`).

---

## 5. HW/SW 매핑

| 기능 | SW (RISC-V) | 인터페이스 | HW (RTL) |
|---|---|---|---|
| 네트워크 스케줄 | `shufflenet_v2.c`, `main.c` | — | (펌웨어가 명령 생성) |
| 명령 발행 | `conv_set/start` (`hpu_api.h:86-87`) | 레지스터맵 `0x20000100` (`memory_map.inc:28`) | `regmap_mgr`→`conv_ctrl` |
| conv 파라미터 | `c_conv_param` 비트필드 (`hpu_api.h:22-31`) | 32b 워드 | `conv_ctrl.v:279-312` 디코드 |
| weight/IFM load | `ldmr_set/start`, `reoder_en` | `0x20000500` | `load_mtxreg_ctrl_v2`+`pal_reoder_pic` |
| OFM store | `svmr_set/start` | `0x20000600` | `save_mtxreg_ctrl`+`ddr_intf` |
| 완료 통지 | `intr.s`, `fshflg_ps()` | PLIC/intr 라인 | `regmap_mgr.intr_o`, `fshflg_ps_o` |
| 메모리 초기화 | `gen_verilog_data_process.py` | ITCM/DTCM verilog | `dl_itcm_ctrl`/`dl_dtcm_ctrl` |

- 분업 원칙: **PS(ARM)=이미지 read/decode/결과 출력만**, **picorv32(PL)=DNN 스케줄링**, **HiPU 데이터패스=연산**. PS 의존 없음(`README.md:71-72, 218-224`).

---

## 6. 빌드·실행

- **SW 빌드**(`hpu_software/README.md:6-15`, `makefile:1-37`): `riscv32-unknown-elf-gcc` 툴체인(`-march=rv32i -mabi=ilp32`). `make all` → `output/hpu.mo`(ELF), `hpu.dump`, `hpu.hex`, 이어 `python gen_verilog_data.py`로 `1_itcm.verilog`/`1_dtcm.verilog` 생성.
- **HW 빌드**: Vivado 프로젝트 `hpu_core/fpga/hpu_reg_version/`(산출물). 제약 `constrs/hpu_reg.xdc`. 최상위 `ps_pl_top.v`→`pl_top`→`hpu_top`. (정확한 합성 스크립트/tcl는 **확인 불가** — 산출물만 존재.)
- 실행: Ultra96 PYNQ에서 PS가 비트스트림 로드 후 ITCM/DTCM에 펌웨어 다운로드(`ps_rvram_*`, `ps_pl_top.v:45-50, 84-90`), DDR base 설정, start 트리거.

---

## 7. 의존성

| 의존성 | 종류 | 근거 |
|---|---|---|
| Xilinx Vivado + ZYNQ UltraScale+ IP | HW 툴/플랫폼 | `fpga/hpu_reg_version`, AXI HP0(`hpu_top.v:89-103`) |
| Xilinx BRAM IP (`sdp_w512x64...`, `pal_reoder_buffer`) | HW IP | `mtxrega.v:213`, `pal_reoder_pic.v:95` |
| MPU/VPU/VPUTY EDIF netlist | 비공개 IP(자체) | `*.edf`, `_stub.v` |
| PicoRV32 (Clifford Wolf, ISC) | 3rd-party RTL | `picorv32.v:1-18` |
| riscv32 GNU toolchain | SW 빌드 | `makefile:1-8`, riscv-gnu-toolchain |
| NumPy | 빌드 스크립트 | `gen_verilog_data_process.py:1` |

---

## 8. 강점 / 한계 / 리스크

**강점**
- 명령어 기반(fine-grained ISA) 범용 DNN 가속기 → conv/fc/dwconv/pool/eltwise/shuffle을 한 데이터패스로 처리. RISC-V로 임의 네트워크 스케줄 가능.
- channel shuffle/divide/concat을 SRAM 주소·strobe로 **무비용 융합**(`mtxrega.v:190-195`).
- inter-layer 캐스케이드로 DDR 대역폭 절감(`README.md:179-202`), 입력 reorder로 첫 layer 효율 개선(`pal_reoder_pic.v`).
- conv 효율 >80%, DSP 98% 활용 — Verilog 직접설계로 HLS팀 대비 자원효율 우위(`README.md:289`).
- HW/SW 레지스터 비트필드가 단일출처(`hpu_api.h`↔`conv_ctrl.v`)로 일치 — 유지보수성 양호.

**한계**
- **연산코어(MPU/VPU/VPUTY) RTL 비공개**(EDIF) → MAC array 마이크로아키텍처 재현·수정 불가. 확인 불가 영역.
- 네트워크 스케줄이 **수작업 C 코드**(`shufflenet_v2.c`의 정적 주소표) → 새 네트워크마다 펌웨어 수동 작성. 컴파일러 자동화 없음.
- conv_ctrl `ST_WAIT_VPROC`가 고정 30사이클 대기(`conv_ctrl.v:438` "TODO: replace by accurate value") — 보수적 마진, 정밀화 여지.
- 8bit 양자화로 IoU 0.056 손실(`README.md:293-295`).
- 2-core 확장이 주석처리(`hpu_top.v:736-893`) — ZU3 자원으로 비활성, 미검증.

**리스크**
- 외부 AXI 128bit ↔ 내부 512bit 변환 정합·DDR 중재(`ddr_intf`) 병목 가능성(README도 DDR 대역 강조).
- 고정 320픽셀/라인·하드코딩 주소표 → 입력 해상도/네트워크 변경 시 RTL+SW 동시 수정 필요.

---

## 9. 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

> 본 repo는 CNN(ShuffleDet) 전용이나, **재사용 가능한 패턴**이 다수.

1. **명령어 기반 제어 + 경량 코어 스케줄러 패턴** (재사용도 높음)
   - regmap_mgr → 연산별 ctrl FSM → 데이터패스 구조는 ViT의 다양한 op(MatMul/Softmax/LayerNorm/GELU)에도 그대로 적용 가능. 우리 HG-PIPE 계열에 "RISC-V/마이크로컨트롤러가 레이어 시퀀스 스케줄" 패턴 도입 검토.
   - `conv_ctrl`의 **delay-chain 기반 파이프라인 정렬(VPU_DLY=17)** 기법은 Transformer의 attention→softmax→projection 다단 파이프 정렬에 직접 응용 가능(`conv_ctrl.v:742-799`).

2. **MatMul = systolic/MAC array 매핑** (핵심 재사용)
   - MPU(8b×8b→32b, 64-lane)는 ViT의 Q·Kᵀ, attention·V, FFN GEMM에 동형. ViT 가속기의 GEMM 엔진 설계 시 "행렬 레지스터(SRAM 멀티뱅크) + strobe 부분기록"으로 타일링하는 패턴 차용 가능(`mtxrega.v`).
   - 단, ViT는 token×dim GEMM이 dominant → channel-우선 루프(`conv_ctrl` ich→wtw...) 대신 token/dim 루프로 재구성 필요(추정).

3. **무비용 데이터 재배열(shuffle/reorder)** → ViT의 patch embedding / head split·concat / transpose에 직접 대응. SRAM 주소·strobe만으로 transpose·head 분할을 구현하면 attention의 reshape 오버헤드 제거 가능(`mtxrega.v:190-195`, `pal_reoder_pic.v`).

4. **8bit 대칭 양자화 + 재스케일 융합** → ViT-Quantization 라인과 직접 연결. VPU의 clip(shift) 재스케일·bias·activation 융합 구조는 Transformer의 INT8 GEMM 후처리(dequant→LayerNorm)에 응용. 단 Softmax/LayerNorm은 비선형 → 별도 LUT/reduction 유닛 필요(본 repo엔 없음).

5. **inter-layer 캐스케이드(DDR 왕복 최소화)** → ViT의 encoder block 내부(attention→FFN)를 온칩 잔류로 fusing하면 동일 효과. HG-PIPE의 파이프라인 fusing 전략과 정합.

6. **SpMV/TVM 흐름 관점**: 본 repo는 dense 8bit MAC만, **sparse 미지원**(README "future work"로 sparse·DeCONV 언급, `README.md:172-175`). TVM 등 컴파일러 흐름도 없음(수작업 C). → 우리가 ViT 가속에 TVM/자동 스케줄러를 붙인다면 본 repo의 "수작업 스케줄"이 반례(피해야 할 한계)로 유용.

7. **XR 시선추적 직접 연관 없음**: 본 repo는 UAV 물체검출용. 시선추적과의 접점은 "경량 백본(ShuffleNet) + 저전력 edge FPGA(Ultra96)" 라는 시스템 제약 공유 정도(추정).

---

## 10. 근거 표기 규칙

- 모든 사실 주장에 `파일명:라인` 근거 병기. README는 `README.md:라인`.
- 소스 미존재로 단정 불가한 항목은 **"확인 불가"** 명시:
  - MPU/VPU/VPUTY 내부 MAC array 마이크로아키텍처(EDIF netlist만 존재).
  - Vivado 합성/구현 tcl 스크립트(산출물만 존재).
  - XR 시선추적과의 직접 관계.
- 포트 폭에서 역산하거나 정황상 도출한 항목은 **"추정"** 명시(예: clk_2x 더블펌핑, VM 모드=FC 경로, ViT 루프 재구성 필요성).
- 수치(268Gops, 50.91FPS, IoU 0.615, 자원% 등)는 모두 `README.md` 보고값 인용이며 본 분석에서 재측정하지 않음.
