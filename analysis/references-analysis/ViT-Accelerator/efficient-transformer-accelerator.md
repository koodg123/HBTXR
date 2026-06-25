# efficient-transformer-accelerator (Glide Accelerator) 정밀 분석

> 분석 대상 경로: `REF/ViT-Accelerator/efficient-transformer-accelerator`
> 분석 일자: 2026-06-20 / 분석 도구: Glob·Grep·Read (정적 코드 분석, 시뮬레이션·합성 미실행)
> 근거 표기 원칙: 코드/문서에 명시된 사실은 단정, 추론은 "추정", 자료 부재는 "확인 불가"로 구분.

---

## 1. 개요

- **프로젝트명**: "Glide Accelerator" (루트 `readme.md:1`), 내부 명칭 "Efficient transformer accelerator" (`hw/src/pe.sv:9`).
- **목적(한줄요약)**: 선형 어텐션 근사(Taylor-series softmax 근사) + INT8 양자화를 적용한 **Vision Transformer 추론용 저전력 systolic-array 가속기 RTL**. 32×16(=512 PE) systolic MAC 배열 뒤에 누산기 뱅크와 시분할 공유 양자화기를 직렬 연결한 단일 데이터패스다.
- **핵심 스펙(문서 명시값, `readme.md:7-21`, `PRODUCTION_RELEASE_NOTES.md:24-37`)**:
  - Systolic array: 32 rows × 16 cols = 512 MAC
  - 정밀도: INT8 입력(8-bit signed) + 32-bit 누산 + INT8 재양자화 출력
  - 어텐션: Taylor 급수 근사 softmax (degree-1, degree-2 두 종류 학습됨)
  - 목표 주파수 200 MHz, 피크 처리량 102.4 GOPS(=512 MAC × 200 MHz × 2 추정), end-to-end ~61 cycle
  - 자원: LUT ~36%, DSP ~15% (ZU9EG 기준, `PRODUCTION_RELEASE_NOTES.md:107-121`)
- **원논문/계보(추정)**: 자체 논문은 저장소 내 없음(확인 불가). 단, `readme.md:44-47`이 참조로 (a) **ViTALiTy** (Linear Taylor Attention, HPCA 2023, GaTech-EIC; `ViTALiTy/README.md:2,10`)와 (b) **AMD QTViT**(quantization)을 명시. degree-1/degree-2 Taylor 어텐션 개념은 ViTALiTy에서 유래한 것으로 **추정**되며, 본 repo는 그 알고리즘을 INT8 systolic HW로 구현한 co-design 시도로 보인다.
- **타깃 디바이스**:
  - **FPGA**: 주 타깃은 **ZCU104 보드 / Zynq UltraScale+ `xczu7ev-ffvc1156-2-e`** (`hw/scripts/vivado_synth.tcl:18`, `vivado_flow.tcl:13`, `IMPLEMENTATION_GUIDE.md:404`). 문서에는 ZU9EG 자원수치도 병기(`PRODUCTION_RELEASE_NOTES.md:107`). 타깃 변경 예시로 Zynq-7000/Artix-7/Kintex-7도 주석에 존재(`vivado_synth.tcl:13-17`).
  - **ASIC**: `hw/constraints/synthesis_asic.sdc` 존재 → Design Compiler/Genus 등 ASIC 합성도 의도(`synthesis_asic.sdc:5,145`). 즉 **FPGA·ASIC 양쪽 타깃**.
- **상태**: 문서상 "Production Ready v0.02"(`PRODUCTION_RELEASE_NOTES.md:3-5`), 테스트 3종 통과·200MHz timing met 주장(시뮬·합성 로그는 repo 내 없어 **재현 확인 불가**).

---

## 2. 디렉토리 구조

자체 소스 트리(third-party·체크포인트 제외):

```
efficient-transformer-accelerator/
├── readme.md                       # 프로젝트 개요(Glide Accelerator)
├── PRODUCTION_RELEASE_NOTES.md     # v0.02 릴리스 노트, 버그픽스/검증 기록
├── hw/
│   ├── README_VIVADO.md            # Vivado sim/synth/impl 사용 가이드
│   ├── IMPLEMENTATION_GUIDE.md     # ZCU104 timing closure 가이드
│   ├── run_vivado.sh               # 빌드 자동화 진입점 (sim/synth/impl/full/clean)
│   ├── src/                        # ── 핵심 RTL ──
│   │   ├── pe.sv                       # Processing Element (3-stage MAC 파이프)
│   │   ├── systolic_mac.sv            # (구) 4×4 정방 systolic 코어 (module Systolic)
│   │   ├── systolic_top.sv            # (구) 4×4 벡터 래퍼
│   │   ├── systolic_mac_rect.sv       # (신) M×N 직사각 systolic 코어 32×16
│   │   ├── accumulator_bank.sv        # 멀티패스 누산기 뱅크(+overflow saturation)
│   │   ├── quant.sv                   # requant: 단일 재양자화 유닛(4-stage)
│   │   ├── quant_top.sv               # requant 단독 래퍼(합성 타깃)
│   │   ├── quant_shared.sv            # 64유닛 시분할 공유 양자화기 + FSM
│   │   ├── systolic_quant_integrated.sv # (구) 4×4 통합 top
│   │   └── systolic_quant_32x16.sv    # (신·주력) 32×16 통합 top
│   ├── tb/
│   │   ├── systolic_tb.sv             # (구) 4×4 systolic TB
│   │   ├── systolic_quant_tb.sv       # (구) 통합 TB
│   │   ├── systolic_32x16_tb.sv       # (주력) 32×16 통합 TB (3 테스트)
│   │   └── debug_signals.tcl          # 파형 디버그용 TCL
│   ├── scripts/
│   │   ├── vivado_sim.tcl             # 시뮬 플로우
│   │   ├── vivado_synth.tcl           # 합성 플로우(top=systolic_quant_32x16)
│   │   ├── vivado_impl.tcl            # 구현(P&R) 플로우
│   │   └── vivado_flow.tcl            # requant 단독 합성+구현 풀플로우(top=quant_top)
│   └── constraints/
│       ├── timing.xdc                 # FPGA(ZCU104) 200MHz 제약
│       └── synthesis_asic.sdc         # ASIC 합성 제약(500MHz target 예시)
└── models/
    ├── degree_2_quant/test.py         # (자체 SW) npz 활성값 인스펙터 (7줄)
    ├── degree_2_train/                # 학습 산출물(.pth, log.txt) — 제외, 이름만
    └── degree_1_train/                # 학습 산출물(.pth, log.txt) — 제외, 이름만
```

- **ViTALiTy cross-ref(내부 분석 생략)**: `ViTALiTy/` 하위는 GaTech-EIC의 HPCA'23 "Linear Taylor Attention" 원본 코드(DeiT/CaiT/ResMLP 등 ViT 학습·양자화)가 통째로 vendoring 되어 있으며, **본 분석에서는 다루지 않고 ViT-Quantization 카테고리에서 별도 분석**한다. (degree Taylor 어텐션 개념의 출처로만 참조)
- **제외 대상(이름만 언급, 미분석)**: `models/degree_{1,2}_train/checkpoint.pth`·`best_checkpoint.pth`(사전학습 체크포인트), 동 디렉토리 `log.txt`(학습 로그), `ViTALiTy/logs/*.log`, `ViTALiTy/.github/*.png`·`figures/*.png`(이미지), `.git`·`.gitattributes`·`.gitignore`.

---

## 3. 핵심 RTL 모듈별 정밀 분석 (라인 근거)

> 본 repo는 **두 세대 데이터패스가 공존**한다. (1) *구세대 4×4 정방형*: `systolic_top.sv`→`Systolic`(in `systolic_mac.sv`)→`systolic_quant_integrated.sv`. (2) *신세대 32×16 직사각형*(주력·릴리스 대상): `systolic_mac_rect.sv`→`quant_shared.sv`→`systolic_quant_32x16.sv`. 공통 하위 블록은 `pe.sv`, `accumulator_bank.sv`, `quant.sv`(requant). 아래는 신세대 위주로, 구세대는 차이점/버그 중심으로 분석한다.

### 3.1 `pe.sv` — Processing Element (공통 연산 단위)

- **역할**: systolic 배열의 1개 셀. **곱셈만 수행**하고 누산은 외부 `accumulator_bank`에 위임(`pe.sv:43` 주석 명시). 즉 PE 자체는 누적 레지스터가 없는 "multiply + 데이터 통과" 셀이다.
- **파라미터**: `DATA_WIDTH=8`, `ACC_WIDTH=32` (`pe.sv:29-30`).
- **포트**: `clk/reset/enable`, `in_a/in_b`(signed 8b 입력), `out_a/out_b`(signed 8b 통과 출력), `out_c`(signed 32b 곱 결과) (`pe.sv:32-40`). `enable`은 clock-gating 용도 주석(`pe.sv:34`).
- **데이터플로우/파이프라인(3-stage, `pe.sv:49-73`)**:
  - Stage1: `a_reg<=in_a; b_reg<=in_b` (입력 등록, `:60-61`)
  - 통과: `out_a<=a_reg; out_b<=b_reg` — **등록값을 통과**시켜 systolic 타이밍 유지(`:63-65`). 릴리스 노트가 기록한 버그픽스 "현재 입력 대신 등록값 통과"가 여기 반영됨(`PRODUCTION_RELEASE_NOTES.md:81-86`).
  - Stage2: `mult_result<=a_reg*b_reg` (16b=2×DATA_WIDTH 곱, `:68`)
  - Stage3: 부호확장 `mult_extended<={{(ACC_WIDTH-2*DATA_WIDTH){mult_result[15]}}, mult_result}` 후 `out_c<=mult_extended` (32b로 sign-extend, `:71-72`)
- **데이터타입/비트폭**: 전구간 signed. 곱 16b → 32b sign-extend. **누산은 PE 밖**이므로 PE 단독으로는 곱 결과 한 개만 전달.
- **특이점**: 주석은 "saturation/overflow protection"을 표방하나(`:21`), 실제 PE 본문에는 곱셈만 있고 포화 로직은 없음 → 포화는 `accumulator_bank`에 구현됨(분석 3.4). 주석과 구현 간 경미한 불일치(설계 의도가 모듈 이전된 흔적, **추정**).

### 3.2 `systolic_mac_rect.sv` — 32×16 직사각 systolic 코어 (주력)

- **역할**: M×N(=ROWS×COLS) 직사각형 PE 배열. transformer 워크로드(예: 32 출력 feature × 16 reduction)를 겨냥(`systolic_mac_rect.sv:3-9`).
- **파라미터**: `ROWS=32`(M, 출력 feature), `COLS=16`(N, 입력/reduction dim), `DATA_WIDTH=8`, `ACC_WIDTH=32` (`:8-11`).
- **포트**: `a_in[ROWS-1:0]`(행 입력 벡터, 32×8b), `b_in[COLS-1:0]`(열 입력 벡터, 16×8b), 출력 `c_out[ROWS-1:0][COLS-1:0]`(32×16×32b 행렬) (`:18-22`).
- **systolic 데이터플로우(핵심)**:
  - **입력 skew(`:45-72`)**: A는 수평(좌→우) 전파용 skew 레지스터 `a_skew[i][j]`로 행마다 j 방향 지연 삽입(`:57-62`), B는 수직(상→하) 전파용 `b_skew[i][j]`로 열마다 i 방향 지연(`:65-70`). 이 삼각형 skew가 systolic 동기화를 만든다.
  - **PE 입력 선택(`:99-117`)**: 첫 열(j==0)은 `a_skew[i][0]`, 이후 열은 좌측 PE의 `a_bus[i][j-1]`(`:101-107`). 첫 행(i==0)은 `b_skew[0][j]`, 이후 행은 상단 PE의 `b_bus[i-1][j]`(`:110-116`). → **A는 행을 따라 오른쪽으로, B는 열을 따라 아래로 흐르는 전형적 weight/data-flowing systolic**.
  - **출력 등록**: `c_out<=c_bus`(`:78-84`)로 PE 곱 결과를 1단 등록.
  - **reset 동기화**: 2-FF 동기화기 `reset_sync1/2`(`:38-43`).
- **버그픽스 반영**: 릴리스 노트가 지목한 "PE가 대각 원소를 읽던 버그"의 *수정본*. 구버전 `a_skew[i][i]` → 신버전 `a_skew[i][0]`로 교정(`:103`, `:112`; cf. `PRODUCTION_RELEASE_NOTES.md:75-79`).
- **dataflow 유형(분석)**: 누산을 PE 밖에서 하므로 **순수 출력-stationary가 아니라, 각 사이클 outer-product를 만들어 외부 누산기에 더하는 "outer-product/partial-sum 방출형" systolic**으로 보는 것이 정확(테스트가 `a_in`/`b_in` 단일 벡터의 외적을 검증, `systolic_32x16_tb.sv:119-136`). 즉 1 패스 = 1 외적, K개 외적을 누산기에 누적해 32×16×K MatMul을 수행(`systolic_quant_32x16.sv:14` "K+~40 cycles"). — 데이터타입 전구간 signed INT8 입력 / 32b 출력.

### 3.3 `systolic_mac.sv`(module `Systolic`) + `systolic_top.sv` — 구세대 4×4 (레거시)

- **역할**: 4×4 정방형 systolic 코어와 그 벡터 래퍼. `systolic_top`은 4-원소 벡터 `a_in/b_in`을 `Systolic`의 스칼라 포트 a1..a4/b1..b4와 16개 스칼라 출력 c1..c16에 수동 매핑(`systolic_top.sv:36-59`).
- **파라미터**: `ARRAY_SIZE=4, DATA_WIDTH=8, ACC_WIDTH=32` (`systolic_mac.sv:5-7`).
- **구조**: rect 버전과 동일한 skew+genvar PE 배열이나 **포트가 스칼라 16개로 평탄화**되어 확장성이 없음(`systolic_mac.sv:12-18`).
- **⚠ 잔존 버그(중요)**: PE 입력 선택이 **여전히 대각 인덱스** `pe_in_a=a_skew[i][i]`(`:100`), `pe_in_b=b_skew[j][j]`(`:107`)을 사용. 릴리스 노트는 이 버그를 rect 버전에서만 고쳤다고 기록(`PRODUCTION_RELEASE_NOTES.md:75-79`). 따라서 **구세대 4×4 경로는 미수정 상태로 남아 기능 오류 가능성**이 높다(정적 분석 기준 **추정**; 시뮬 미실행으로 확정 불가). 또한 skew 인덱싱도 rect와 달리 `a_skew[i][k]<=a_skew[i][k-1]`을 행/열 방향으로 다르게 적용(`:57-69`).
- **현황**: `vivado_synth.tcl:48-58`은 9개 src를 모두 컴파일 목록에 넣되 **top은 신세대 `systolic_quant_32x16`로 고정**(`:10`). 즉 구세대 파일은 빌드에 포함되나 실제 최상위로는 미사용. 레거시/프로토타입 잔재로 분류.

### 3.4 `accumulator_bank.sv` — 멀티패스 누산기 뱅크

- **역할**: PE가 매 패스 방출하는 partial sum을 셀별로 누적(타일드 MatMul의 K-차원 누적 담당).
- **파라미터**: `ROWS, COLS, ACC_WIDTH=32` (`accumulator_bank.sv:5-7`). *단, 기본값이 `ROWS=4,COLS=4`로 선언*되어 있어 32×16 사용 시 인스턴스에서 override 필요(`systolic_quant_32x16.sv:95-99`에서 32/16 전달).
- **포트**: `enable/clear`, `partial_sums[ROWS][COLS]`(32b 입력 행렬), `accumulated_sums`(32b 누적 출력), `overflow_flag` (`:9-18`).
- **알고리즘/비트폭(`:25-51`)**:
  - 누적 시 1bit 확장(`ACC_WIDTH:0`=33b) `accum_temp = {sign,partial}+{sign,accum}` 으로 **signed 오버플로 검출**(`:35-36`).
  - 부호비트 불일치(`accum_temp[32]!=accum_temp[31]`) → overflow → **포화 saturation**: 음수 오버플로는 `0x8000_0000`(-2^31), 양수 오버플로는 `0x7FFF_FFFF`(+2^31-1)로 클램프(`:38-44`).
  - `clear`로 누산 레지스터 0 초기화(멀티패스 시작, `:29-31`).
  - 임의 셀 overflow 발생 시 `overflow_flag` 집계(`:53-60`).
- **분석**: 32-bit 누산 + saturation은 INT8×INT8 누적의 정수 안전성을 확보하는 표준 기법. PE에서 누산을 분리해 이 뱅크에 집중시킨 설계는 PE 면적 절감과 멀티패스 타일링 유연성을 동시에 얻는 합리적 선택.

### 3.5 `quant.sv`(module `requant`) — 단일 재양자화 유닛

- **역할**: 32b 누적값을 INT8로 재양자화(scale 곱 → 산술 시프트 → 포화). transformer 양자화의 requantization(=`round(acc * scale >> shift)`)을 4-stage 파이프로 구현.
- **포트(`quant.sv:1-11`)**: `in[31:0]`(누적값), `b[31:0]`(scale_factor), `shift_factor[7:0]`, 출력 `out[7:0]`. + `clk/rst/en`.
- **파이프라인(4-stage, `:30-106`)**:
  - S1 입력 등록(`:31-43`)
  - S2 곱셈 `mult_result(64b)<=in_s1*b_s1`, DSP 추론 의도 주석(`:45-59`)
  - S3 배럴 시프터 `shift_temp=mult_result>>shift_s2; shifted<=shift_temp[31:0]`(`:61-77`)
  - S4 **signed 8b 포화**: 음수면 상위비트(`shifted[31:7]`)가 전부 1이 아니면 `-128(0x80)`, 양수면 상위비트 중 하나라도 1이면 `+127(0x7F)`, 아니면 하위 8b(`:79-106`)
- **비트폭 주의**: 포트 선언이 `logic [31:0] in`/`[7:0] out`로 **unsigned 선언**이나 S4 포화는 `shifted[31]`을 부호비트로 다루는 **signed 의미**로 동작(`:87,97`). 곱셈도 unsigned `*`라 scale·in이 음수일 때의 부호 처리는 선언과 어긋날 소지가 있음(정적 분석상 잠재 이슈, **추정** — 실측 미확인). 시프트는 산술이 아닌 **논리 시프트(>>)**라 음수 누적값의 시프트 정확도는 검증 필요.
- **`quant_top.sv`**: `requant`를 그대로 감싼 단독 합성 래퍼(`quant_top.sv:14-22`). `vivado_flow.tcl`의 top(=`quant_top`)으로, requant만 독립 합성/타이밍 측정하는 용도.

### 3.6 `quant_shared.sv` — 64유닛 시분할 공유 양자화기 (자원효율 핵심)

- **역할**: 512개(32×16) 누적값을 **64개 requant 유닛으로 8배 시분할**해 양자화. 512개 전용 유닛 대비 87.5% 유닛 절감(`PRODUCTION_RELEASE_NOTES.md:121`).
- **파라미터**: `ROWS=32,COLS=16,ACC_WIDTH=32,OUT_WIDTH=8,QUANT_UNITS=64` (`quant_shared.sv:8-13`).
- **상수 도출(`:31-33`)**: `TOTAL_ELEMENTS=512`, `CYCLES_PER_BATCH=ceil(512/64)=8`, `PIPELINE_DEPTH=4`(requant 파이프 깊이).
- **FSM(IDLE→PROCESSING→FLUSHING→DONE, `:39-114`)**:
  - `start_pulse`(enable 상승엣지)로 PROCESSING 진입, 입력 512개를 `input_buffer`에 1-shot 캡처(`:122-130`).
  - PROCESSING 8 사이클: 매 사이클 `cycle_count*64+u` 인덱스로 64개씩 requant에 공급(`:141-150`), 64개 requant 인스턴스 병렬 생성(`:153-166`).
  - FLUSHING 4 사이클: 파이프 잔여 비움.
  - 출력 수집: `output_cycle>=PIPELINE_DEPTH` 시점부터 requant 출력을 `output_buffer`에 기록(`:175-194`).
  - DONE에서 512개를 `data_out[i][j]`로 매핑, `valid<=1`(`:200-214`).
- **버그픽스 반영(중요)**: 릴리스 노트가 "출력버퍼가 0만 나오던 버그"를 `write_cycle`→연속 증가 `output_cycle`로 교정했다고 기록(`PRODUCTION_RELEASE_NOTES.md:69-74`), 본 파일 `:173-194`이 그 수정본. 양자화 레이턴시 = 8(batch)+4(pipe)+1(done) = **~13 cycle**(`PRODUCTION_RELEASE_NOTES.md:54`).
- **분석**: systolic 출력 512개를 통째로 펼치는 대신 시분할하는 것은 ViT/transformer처럼 **batch한 행렬 출력**을 다룰 때 양자화기 면적을 크게 줄이는 정석 기법. 64라는 폭은 8 사이클이라는 양자화 지연과 면적의 trade-off 선택.

### 3.7 `systolic_quant_32x16.sv` — 신세대 통합 Top (주력)

- **역할**: 3-stage 파이프라인 통합. **systolic(곱) → accumulator(누적) → quant_shared(재양자화)**를 직렬 연결(`systolic_quant_32x16.sv:48-135`).
- **파라미터/포트(`:17-46`)**: 입력 `a_in[32]`/`b_in[16]`(INT8), 누산 제어 `accum_clear/accum_enable`, 양자화 파라미터 `scale_factor[31:0]/shift_amount[7:0]/quant_enable`, 출력 `quant_out[32][16]`(INT8) + `systolic_valid/accum_overflow/quant_valid`.
- **데이터플로우**:
  - Stage1 systolic: `u_systolic`(systolic_mac_rect). valid는 `SYSTOLIC_LATENCY=ROWS+COLS+2=50` 길이 shift-register로 지연 생성(`:75-89`).
  - Stage2 accumulator: `accum_enable & systolic_valid_reg`로만 누적(`:102`).
  - Stage3 quant_shared: 누적 결과를 INT8로(`:120-135`).
- **레이턴시(문서)**: systolic 48 cyc + quant 13 cyc → end-to-end ~61 cyc(`PRODUCTION_RELEASE_NOTES.md:53-56`).

### 3.8 `systolic_quant_integrated.sv` — 구세대 통합 Top (레거시)

- 4×4(`ARRAY_SIZE=4`) 통합. systolic_top→accumulator_bank→**requant 16개 병렬**(시분할 없이 셀당 1개, `:91-108`).
- **⚠ 포트 불일치(잠재 컴파일 이슈)**: `accumulator_bank`를 `.ARRAY_SIZE(ARRAY_SIZE)`로 인스턴스화(`:67-69`)하지만 `accumulator_bank.sv`는 `ROWS/COLS`만 선언(`accumulator_bank.sv:5-7`) → **존재하지 않는 파라미터 override**. 엄격한 합성기에선 경고/오류 가능(정적 분석 기준 **추정**; 신세대 top이 주력이라 실무 영향은 제한적). 구세대 잔재의 미정합 사례.

---

## 4. 데이터플로우 / 실행 흐름

- **systolic array 유형**: A 행 방향(좌→우) + B 열 방향(상→하) 전파의 **2-D systolic outer-product 엔진**. PE에 누산기가 없으므로 weight/output-stationary가 아니라 **매 패스 외적(partial sum)을 방출 → 외부 누산기 누적** 방식(3.2 참조). "stationary" 분류로는 입력이 흐르고 부분합이 외부에 모이는 형태라 엄밀히는 *no-local-accumulate, partial-sum-streaming* 구조(**분석/추정**).
- **정방 vs 직사각**: 구세대 4×4 정방(`systolic_mac.sv`, 레거시·버그 잔존) vs 신세대 32×16 직사각(`systolic_mac_rect.sv`, 주력). 직사각은 transformer의 비대칭 차원(출력 feature ≠ reduction dim)에 맞춰 면적/매핑을 최적화한 선택(`systolic_mac_rect.sv:3-9`).
- **양자화-systolic 통합 파이프라인**:
  ```
  a_in[32], b_in[16] (INT8)
     │  systolic_mac_rect (512 PE, 곱)            ← Stage1, latency ROWS+COLS(+pipe)
     ▼  systolic_out[32][16] (INT32 partial sum)
     │  accumulator_bank (512 acc, 멀티패스 누적+포화)  ← Stage2
     ▼  accum_out[32][16] (INT32)
     │  quant_shared (64 requant, 8× 시분할)        ← Stage3, latency ~13
     ▼  quant_out[32][16] (INT8) + quant_valid
  ```
- **데이터타입/비트폭 요약**:
  | 단계 | 타입/폭 | 근거 |
  |---|---|---|
  | 입력 a/b | signed INT8 | `systolic_mac_rect.sv:18-19` |
  | PE 곱 | signed 16b→32b sign-ext | `pe.sv:46,71` |
  | 누산 | signed 32b (검출 33b) | `accumulator_bank.sv:20-21` |
  | scale_factor | 32b | `systolic_quant_32x16.sv:37` |
  | shift_amount | 8b | `:38` |
  | requant 중간곱 | 64b | `quant.sv:19` |
  | 출력 | signed INT8 (±127/-128 포화) | `quant.sv:98-99` |
- **제어**: `enable`(clock-gating 겸용) / `accum_clear`(패스 경계) / `accum_enable`(누적 게이트) / `quant_enable`(양자화 트리거, 상승엣지로 FSM 기동). valid 신호는 shift-register 지연으로 생성.

---

## 5. HW/SW 매핑 (Python 모델 ↔ RTL)

- **SW 측 실체**: 자체 Python 코드는 `models/degree_2_quant/test.py` **단 1개(7줄)**로, `attn_float_activations.npz`(또는 quant 버전)의 배열 shape/dtype를 출력하는 **활성값 인스펙터**일 뿐 모델 정의·추론 코드가 아님(`test.py:1-8`). 즉 **degree-2 Taylor 어텐션 모델 자체의 소스는 본 repo의 models/ 안에 없고**, ViTALiTy(별도 분석) 또는 외부에 있음(**추정**).
- **학습 산출물로 본 모델 정황**(로그만 근거, 가중치 .pth는 제외):
  - `degree_2_train/log.txt`: ImageNet 추정(1000-class, top-1/top-5 보고), **n_parameters=5,717,416**(~5.7M) → **DeiT-Tiny급 소형 ViT**로 추정(`degree_2_train/log.txt:1`). 100여 epoch에 걸쳐 top-1이 0.2%→증가하는 from-scratch 학습 곡선.
  - degree-1/degree-2 두 변종 = Taylor 급수 **차수(order m)** 차이. ViTALiTy 문서가 m=1(linear) Taylor 어텐션을, m>1을 더하면 vanilla softmax에 근접한다고 설명(`ViTALiTy/README.md:25-26`). 본 repo의 degree-2는 그 2차항까지 포함한 근사를 학습한 것으로 **추정**.
- **알고리즘 ↔ RTL 매핑(추정)**:
  - Taylor 선형 어텐션의 핵심 연산은 결국 **행렬곱(Q·Kᵀ 또는 K·V context, FFN, projection)** → systolic_mac_rect의 INT8 MatMul로 대응.
  - 모델의 per-tensor scale/shift 양자화 파라미터 → RTL `scale_factor`/`shift_amount` 포트로 주입(`systolic_quant_32x16.sv:37-38`).
  - `.npz` 활성값(test.py가 검사하는 대상)은 SW에서 추출한 어텐션 활성을 **HW 시뮬 입력/검증 골든값**으로 쓰려는 의도로 추정(확인 불가 — TB는 합성 입력만 사용, `.npz`를 읽지 않음).
- **결론**: HW↔SW가 **느슨하게 연결**되어 있다. RTL은 일반 INT8 MatMul+requant 엔진이고, "Taylor degree-2"는 SW(별도) 모델 쪽 개념이며 둘을 잇는 명시적 컴파일러/매퍼/포맷 변환 코드는 repo 내 부재(**확인 불가**).

---

## 6. 빌드·실행 방법

- **진입점**: `hw/run_vivado.sh <sim|sim-gui|synth|synth-gui|impl|impl-gui|both|full|clean>` (`run_vivado.sh:182-205`). Vivado가 PATH에 없으면 즉시 종료(`:21-26`).
- **시뮬**: `./run_vivado.sh sim` → `scripts/vivado_sim.tcl` batch 실행, 로그 `vivado_sim.log`(`run_vivado.sh:42`). 주력 TB는 `systolic_32x16_tb.sv`(3 테스트: 외적[1×1]=1, [2×3]=6, 멀티패스 누적 6+2=8; `systolic_32x16_tb.sv:119-268`).
- **합성**: `./run_vivado.sh synth` → `scripts/vivado_synth.tcl`, **top=`systolic_quant_32x16`**, part=`xczu7ev-ffvc1156-2-e`(`vivado_synth.tcl:10,18`). 9개 src 전체 추가(`:48-58`), out-of-context 모드(`:86`), 리포트 생성(util/timing/power/drc).
- **구현(P&R)**: `./run_vivado.sh impl` → `vivado_impl.tcl`(존재; Performance_Explore 전략 언급 `IMPLEMENTATION_GUIDE.md:194`). 200MHz(5ns) 제약 `constraints/timing.xdc:12`.
- **requant 단독 플로우**: `vivado_flow.tcl`은 별개로 **top=`quant_top`**, requant만 합성+구현+주파수 산출(`vivado_flow.tcl:7-13,131-137`), target_period 2.0ns(=500MHz) 기준 achieved freq 계산(`:133-136`).
- **ASIC 타깃 여부**: **예.** `constraints/synthesis_asic.sdc`로 ASIC 합성 의도 명시. 클럭 2.0ns(500MHz) 예시, uncertainty/transition/latency·input/output delay·max_fanout/transition·`set_dynamic_optimization`·retiming/`set_balance_registers` 등 ASIC 표준 제약 포함(`synthesis_asic.sdc:11-167`). 단 `set_driving_cell`/라이브러리 셀이 `<BUFFER_CELL>`,`<LIB_NAME>` 등 **플레이스홀더**라 실제 PDK 연결 전 상태(`:71`). `IMPLEMENTATION_GUIDE.md:342-346`도 "ASIC flow 시 RTL+synthesis_asic.sdc를 DC/Genus에 투입"이라 안내. → **FPGA·ASIC 듀얼 타깃이나 ASIC은 골격만 제공**.

---

## 7. 의존성

- **HW 툴**: **Xilinx Vivado 2019.2+ 권장, 2022.1 테스트**(`README_VIVADO.md:6,459`). ASIC은 Synopsys Design Compiler / Cadence Genus 등(SDC 주석, `synthesis_asic.sdc:145`, `IMPLEMENTATION_GUIDE.md:346`).
- **타깃 part**: ZCU104 `xczu7ev-ffvc1156-2-e`(주), ZU9EG/Zynq-7000/Artix-7/Kintex-7(대안 주석).
- **SW(자체)**: `models/degree_2_quant/test.py`는 **numpy만** 사용(`test.py:1`). 그 외 ML 의존성은 ViTALiTy 측(`ViTALiTy/requirement.txt`, 별도 분석)이며 본 repo 자체에는 requirements 파일 없음(**확인 불가**).
- **언어**: SystemVerilog(RTL, `.sv`), Tcl(빌드), Bash(자동화), Python(검사 스크립트).

---

## 8. 강점 / 한계 / 리스크

**강점**
- INT8 systolic + 32b 누산 + 시분할 공유 양자화의 **end-to-end 통합 데이터패스**가 단일 top(`systolic_quant_32x16`)에 깔끔히 묶여 있어 재사용·통합이 쉽다.
- `quant_shared`의 64유닛 8× 시분할은 양자화기 면적을 크게 절감(문서상 448 DSP 절감)하는 실효적 마이크로아키텍처.
- 누산기 포화·오버플로 검출, 2-FF reset 동기화, clock-gating enable, FPGA(XDC)+ASIC(SDC) 제약 동시 제공 등 **production을 의식한 디테일**.
- 직사각(32×16) 배열로 transformer의 비대칭 차원에 맞춤.

**한계**
- **고정 차원**: 32×16 하드코딩, 다른 크기는 RTL 수정 필요(`PRODUCTION_RELEASE_NOTES.md:182`).
- **단일 정밀도 INT8**, per-tensor 단일 scale/shift만(채널별 양자화·BF16/FP16 없음, `:184-185`).
- **AXI/DMA 인터페이스 없음** — 포트 직결만 제공, 시스템 통합 시 별도 래퍼 필요(`:174,186`).
- **HW↔SW 단절**: Taylor degree-2 모델 소스·HW 매퍼·골든값 연결 코드가 repo 내 부재(5절).
- **검증 근거 빈약**: 시뮬/합성 로그·리포트가 repo에 없어 "200MHz·테스트 통과" 주장은 **재현 확인 불가**. TB도 외적/스칼라 상수 3종뿐이라 실제 어텐션 행렬·랜덤 패턴 커버리지 부족.

**리스크**
- **구세대 4×4 경로의 잔존 버그**(`systolic_mac.sv:100,107`의 `a_skew[i][i]`/`b_skew[j][j]`)와 **레거시 top의 파라미터 불일치**(`systolic_quant_integrated.sv:67`의 `.ARRAY_SIZE` ↔ accumulator_bank `ROWS/COLS`) → 해당 파일을 top으로 쓰면 오동작/합성오류 가능(**정적 분석 기준 추정**).
- `requant`의 unsigned 포트 선언 + 논리 시프트(`>>`)와 signed 포화 혼용 → 음수 누적값 양자화 정확도 검증 필요(**추정**).
- ASIC SDC의 라이브러리/드라이빙셀 플레이스홀더 → 실 PDK 미연결.
- `models/` 가중치(.pth)·ViTALiTy vendoring으로 저장소가 비대(분석 대상 외).

---

## 9. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

1. **systolic+quant 통합 RTL의 즉시 참고 가치**: `systolic_quant_32x16`의 *MAC→누산→시분할 requant* 3-stage 직렬 구조는, HG-PIPE류 파이프라인 ViT 가속기에서 **layer 출력 요구양자화 단을 어떻게 면적효율적으로 붙이는가**에 대한 실전 레퍼런스. 특히 `quant_shared`의 N:1 시분할(여기선 512:64=8×)은, HG-PIPE가 추구하는 fully-pipelined 흐름에서 **양자화기를 라인레이트로 깔지 않고도 throughput을 맞추는 절충안**으로 차용 가능(우리 설계가 더 공격적인 II=1을 노린다면 시분할 폭/뱅킹을 재설계해야 함).
2. **직사각 systolic의 차원 매핑**: 32×16처럼 (출력 feature × reduction)을 비대칭으로 잡는 방식은 ViT의 head_dim/embed_dim 비대칭과 patch token 수에 맞춰 PE 배열을 튜닝하는 출발점. XR 시선추적용 경량 ViT(작은 해상도·적은 토큰)에선 reduction dim이 작아 **더 납작한 직사각(예: ROWS≫COLS)**이 유리할 수 있음 — 본 repo의 파라미터화(rect)가 그 실험을 쉽게 함(`systolic_mac_rect.sv:8-11`).
3. **Taylor degree-2 linear attention의 ViT/XR 적용성**: ViTALiTy 계열 linear Taylor 어텐션은 softmax의 O(N²)를 O(N)으로 낮춰 **저지연·저메모리**가 핵심 → XR 시선추적처럼 **프레임당 수 ms 지연·온디바이스 전력예산**이 빡빡한 워크로드에 직접적으로 부합. degree-1(가장 가벼움) vs degree-2(정확도↑) 트레이드오프는, 시선추적 정확도 요구에 맞춰 **차수를 가변**하는 설계공간을 제공. 단 본 repo RTL은 일반 MatMul 엔진이라 **Taylor 특유의 K·V context(global matrix G) 누적·정규화 단을 별도 RTL로 추가**해야 진짜 linear-attention 가속이 됨(현재는 그 부분이 RTL에 없음 — **확인 결과 부재**).
4. **양자화 파이프라인 재사용**: `quant.sv`의 *곱→시프트→signed 포화* 4-stage requant는 우리 가속기의 per-layer requant 블록으로 거의 그대로 이식 가능(단 unsigned 포트/논리시프트 이슈는 산술시프트·signed 명시로 보강 권장).
5. **반면교사**: HW↔SW 연결 부재, 단일 패턴 TB, 미수정 레거시 경로는 우리 프로젝트에선 **(a) SW 골든 추출→HW 입력 포맷→cocotb/UVM 회귀, (b) 레거시 정리, (c) per-channel/혼합정밀 확장**을 처음부터 설계에 포함시켜야 한다는 교훈.
6. **듀얼 타깃(FPGA+ASIC) 제약 분리**: timing.xdc(FPGA)와 synthesis_asic.sdc(ASIC)를 분리 관리하는 패턴은, XR용으로 FPGA 프로토 후 ASIC 이행을 노릴 때의 제약 관리 템플릿으로 유용.

---

## 10. 근거 / 한계 표기

- **단정(코드/문서 직접 근거)**: 모듈 포트·파라미터·파이프라인 단수·비트폭·FSM·빌드 타깃·제약 수치는 모두 위에 인용한 정확한 파일:라인에 근거. 릴리스 노트의 버그픽스 3건은 해당 RTL 라인에서 *수정본*을 직접 확인.
- **추정(명시)**: ① 원논문·계보가 ViTALiTy라는 점(참조 링크 기반 추론), ② degree-2 모델이 ~5.7M DeiT-Tiny급 ViT라는 점(파라미터 수·로그 기반), ③ dataflow를 "partial-sum-streaming(non-stationary)"로 분류, ④ 구세대 4×4 경로 버그·레거시 top 파라미터 불일치의 실제 오동작 여부, ⑤ requant 음수처리 정확도 이슈, ⑥ `.npz`가 HW 골든값 용도라는 점 — 모두 정적 분석 기반 추정이며 시뮬레이션 미실행.
- **확인 불가**: ① 200MHz timing met·테스트 통과(시뮬/합성 로그·리포트가 repo에 없음), ② degree-2 Taylor 어텐션 모델의 실제 소스 위치(models/ 내 부재), ③ 본 repo 전용 Python 의존성 목록(requirements 부재), ④ HW를 구동하는 SW 드라이버/컴파일러/매퍼 존재 여부(부재로 보이나 단정 불가), ⑤ ASIC SDC의 실 PDK 연결 결과.
- **방법론 한계**: 정적 코드 리딩만 수행. 시뮬레이션·합성·파형 미확인. `.pth`/`.git`/이미지/`ViTALiTy/` 내부는 의도적으로 미분석(범위 외).
