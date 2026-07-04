# llama-fpga (EdgeLLM) 정밀 분석

> 분석 대상: `REF/ViT-Accelerator/llama-fpga`
> 자체 소스 기준: `scala/src/main/scala/**/*.scala` (SpinalHDL), SDK C 호스트(`*_sdk.c`, `au250/host_*.c`), `python/model2bin.py`
> Vivado 프로젝트(`kv260/`, `zcu104_pl/`, `zcu104_ps_pl/`, `alveo_u250/` 내 `*.gen/`, `*.srcs/`, `bd_*_wrapper.v`, `DataPath_xN.v` 등)는 **보드별 Vivado 배포 프로젝트(생성물)** 로만 다루며 본문 분석에서 제외.

---

## 1. 개요

- **무엇인가:** LLaMA2-7B(decoder-only Transformer)를 **AWQ 4-bit 양자화** 형태로 임베디드/데이터센터 FPGA에서 추론하는 오픈소스 LLM 가속기. 내부 SpinalHDL 코드/생성 RTL/AXI-Lite 레지스터 맵에서 통칭 **EdgeLLM**으로 불림 (`au250/host_sdk.c:10` `#define EDGELLM_BASE_ADDR`, `top/EdgeLLMInst.scala` = SpinalVerilog 엔트리포인트).
- **한 줄 요약:** SpinalHDL(Scala) 로 **파라미터화된 FP16 데이터패스 + AWQ W4A16 / KV-cache INT8 양자화 + RMSNorm/RoPE/Safe-Softmax** 를 생성하고, 코어를 `numOfCore`만큼 복제·링(ring) 연결하여 여러 Xilinx 보드에 Vivado로 배포하는 메모리대역폭 지향 LLM 디코딩 가속기.
- **논문(README.md:194-212):**
  - Li et al., *"Pushing up to the limit of memory bandwidth and capacity utilization for efficient LLM decoding on embedded FPGA"*, **DATE 2025**.
  - Li et al., *"Hummingbird: A Smaller and Faster Large Language Model Accelerator on Embedded FPGA"*, **ICCAD 2025**.
  - 즉 핵심 컨셉은 "메모리 대역폭/용량 활용 극대화" — 디코딩은 weight-bound 이므로 양자화 + DMA 분할 + 멀티채널 메모리로 대역폭을 짜내는 설계.
- **타깃 보드(README.md:13-35):** Xilinx **KV260**(PS측 4GB RAM), **ZCU104**(PL-only / PS+PL 하이브리드 두 변형), **Alveo U250**(DDR4 4채널, XDMA). 성능: KV260 ~5, ZCU104 PL ~4, ZCU104 PS+PL ~8-9, U250 ~18-19 tokens/s (README.md:137-142).
- **한계(README 자기명시, README.md:157-186):** 디코딩 단계만 HW 가속(프리필은 디코더 재사용으로 토큰 단위 처리), 최대 컨텍스트 1024 토큰, 단일턴, LLaMA2-7B 구조에 강결합, SpinalHDL 자동생성 Verilog 가독성 낮음.

> **중요 정정:** 작업 지시문은 "Chisel/Scala"로 표기했으나 실제 코드는 **SpinalHDL**(`import spinal.core._`, `import spinal.lib._`, `extends Component`, `SpinalVerilog(...)`, `Flow`/`Stream`/`Bundle`/`RegNext`/`m2sPipe`)이다. Chisel과 같은 Scala 기반 RTL 생성 프레임워크지만 API/세대기는 다르다. 본 문서는 SpinalHDL 문법 기준으로 분석한다.

---

## 2. 디렉토리 구조 (자체 소스)

```
llama-fpga/
├─ README.md                         # EdgeLLM 개요·보드·성능·논문·BaiduPan/GDrive 바이너리 링크
├─ python/
│  └─ model2bin.py                   # AWQ 4-bit LLaMA2-7B → 보드별 weight 바이너리 패킹
├─ scala/src/main/scala/             # SpinalHDL 소스 (가속기 데이터패스 정의)
│  ├─ top/                           # 데이터패스 최상위 + 서브모듈
│  │   ├─ EdgeLLMInst.scala          # main: DataPath_xN 인스턴스화 + Verilog 생성 진입점
│  │   ├─ DataPath_xN.scala          # 코어(DataPath) numOfCore 복제 + c2c 링 연결 + 클럭/리셋
│  │   ├─ DataPath.scala             # 단일 코어 데이터패스 (전 서브모듈 결선)
│  │   ├─ MulAddSGNew.scala          # split-bank MulAdd + FP32 누산 트리(Fp32AccEngine)
│  │   ├─ MulAddEngine(New).scala    # MulEngine(dot)+AddEngine(axpy) 묶음
│  │   ├─ AttnSubMod.scala           # RoPE + KV 양자화 + QK·Softmax 묶음
│  │   ├─ NormSubMod(New).scala      # RMSNorm(FP32) 래퍼
│  │   ├─ BusInSubMod(New).scala     # AXI 입력→INT4/INT8 역양자화→엔진 공급
│  │   ├─ AllGatherSubMod(New).scala # 멀티코어 AllGather/AllReduce 노드 묶음
│  │   ├─ ScalarOutSubMod / VecOutSubMod / VecOutBuf
│  │   ├─ StateGen / GlobalStateGen / AxiLiteCtrl / AddressRemap
│  ├─ core/                          # 연산 엔진 (MulEngine, AddEngineNew/Bk, Fp32AccEngine, VecNto1)
│  ├─ quant/                         # 동적 양자화 (LinearQuant, FindRange, GetScaleZero, QuantWrapper)
│  ├─ convert/                       # 정수→FP16 변환 (Int2FP16, Int4Int8FP16Conv)
│  ├─ c2c/                           # 칩-투-칩/코어-투-코어 (AllReduce, AllGatherNode, Node, LowLatencyNode, Reorder, ReduceFilter)
│  ├─ busdemux/                      # AXI 버스 분배 (AxiBusDistributor, Dense/Sparse/KvCacheCase)
│  ├─ schedule/                      # 모델 메모리맵·스케줄 (LLMCfg, MemCmd)
│  ├─ residual/                      # ResidualBuffer, SerialResAdd
│  └─ util/                          # FP IP 래퍼·LUT·FIFO·DMA·고정소수↔FP16 변환 (ExpFunc, Fix48ToFp16, Fp16ToFix24, LUT2/3/4, AXIDataMoverWrapper, SplitAxiDatamover, HBM*, Stream*Fifo 등 다수)
├─ kv260/        kv260_sdk.c       (+ Vivado 배포 프로젝트·xsa·demo gif)
├─ zcu104_pl/    zcu104_pl_sdk.c   (+ Vivado 배포 프로젝트)
├─ zcu104_ps_pl/ zcu104_sdk.c      (+ Vivado 배포 프로젝트)
└─ au250/        host_sdk.c, host_program_x2.c, load_param.sh (+ Vivado 배포 프로젝트)
```

`scala/` 모듈은 패키지 단위로 잘 분리되어 있다: 연산(`core`/`top`), 양자화(`quant`/`convert`), 통신(`c2c`/`busdemux`), 스케줄(`schedule`), 정규화/잔차(`norm`*/`residual`), 인프라(`util`). (`*` norm 패키지는 `RMSNormFp32` import로 존재 확인, NormSubModNew.scala:4)

---

## 3. 핵심 모듈 정밀 분석 ★

### 3.0 공통 데이터타입 / 인터페이스 컨벤션
- **연산 데이터타입:** 활성/액티베이션은 전부 **FP16(`width = 16`)**. DataPath.scala:151 `val width = 16`, MulAddEngine.scala:23 `serialBit = width`. weight는 외부에서 **INT4(AWQ) / INT8(KV)** 로 들어와 온칩에서 FP16으로 역양자화 후 곱셈 (3.5, 3.6 참조).
- **누산 정밀도:** 도트프로덕트 부분합은 **FP32로 승격하여 누산**(`Fp32AccEngine`, MulAddSGNew.scala:115-128) 후 다시 FP16으로 강하. RMSNorm/Softmax도 FP32 경로 존재(3.4, 3.7).
- **스트림 표준:** SpinalHDL `Flow`(valid-only, backpressure 없음)와 `Stream`(valid/ready)을 혼용. AXI-Stream 신호명은 `util.AxiStreamSpecRenamer(...)`로 일괄 표준화. 데이터 페이로드는 대부분 `util.AxiFrame(Bits, userBit=6[, destBit])` — 6비트 `tuser` 가 **연산 종류 태그(tag)** 로 쓰여 모듈 내 라우팅을 결정(태그 기반 데이터플로우).
- **버스폭:** `busWidth = 512`(EdgeLLMInst.scala:52), `bankLen = busWidth/4 = 128`, `parallelWidth = width*bankLen = 16*128 = 2048비트` 병렬 벡터. 즉 한 사이클에 FP16 128개를 병렬 처리.

### 3.1 EdgeLLMInst.scala — 생성 진입점 / 모델 파라미터 바인딩
`object EdgeLLMInst extends App`(EdgeLLMInst.scala:7) 이 `cfg.generateVerilog(new DataPath_xN(...))`로 RTL을 토해낸다. 여기서 보드 변형이 결정된다:
- `numOfCore`, `DMA_SPLIT`, `cmdAddrWidth`, `splitBaseAddr`, `sync` 조합을 주석으로 토글(EdgeLLMInst.scala:11-50). 예: U250 4채널=`numOfCore=1, DMA_SPLIT=List(4), cmdAddrWidth=40`; 듀얼 코어=`numOfCore=2`; KV260/4코어=`numOfCore=4`. 활성 설정은 `numOfCore=2, DMA_SPLIT=List(1,1)`.
- 모델 차원은 `cfgGen.LLaMA2_7B.modelCfg` 에서 주입(EdgeLLMInst.scala:73-82): `dim/head/headDim/mlpDim/layer/vocabSize`. `schedule/LLMCfg.scala`에 동일 값이 명시: **dim=4096, head=32, layer=32, mlpDim=11008, predDim=1024, maxToken=1024, group=128**(LLMCfg.scala:8-14), 양자화 폭 **w=4, kv=8, scale=16**(LLMCfg.scala:16-20).
- 모든 산술기는 **함수 인자**로 주입된다 — Xilinx Floating-Point IP 래퍼들(`util.fp16mul6`, `fp16add6`, `fp16acc16`, `fp16div12`, `fp32mul8`, `fp32acc22`, `fp32exp20`, `fp32rsqrt32`, `fp16ex12` 등, EdgeLLMInst.scala:141-188)이 mul/add/acc/exp/rsqrt/div/lt 및 FP16↔FP32/Int 변환을 담당. 각 IP의 `.latency` 가 함께 전달되어 파이프라인 정렬(`Delay(...)`)에 사용. → **연산기 교체/정밀도 변경이 인자 교체만으로 가능**한 매우 파라미터화된 설계.

### 3.2 DataPath_xN.scala — 코어 복제 + 멀티코어 링 통신
- `coreArea = for (i <- 0 until numOfCore) yield new ClockingArea(...) { val core = new DataPath(...) }`(DataPath_xN.scala:168-182): **단일 코어 `DataPath`를 numOfCore개 복제**. 각 코어는 자체 클럭도메인(`internalClock(i)`, 비동기 `sync=false` 시 코어별 clk/rstn) 위에서 동작 → 멀티 SLR/멀티채널 친화적.
- 코어별 `m_axi`(DMA_SPLIT==1) 또는 `m_axi_hp` Vec(분할 DMA) 를 IO로 노출(DataPath_xN.scala:184-198) → **보드의 DDR/HBM 채널 수에 맞춰 마스터 AXI 포트 개수가 가변**.
- **c2c 링 토폴로지:** `numOfCore != 1` 일 때 `(coreArea, coreArea.drop(1) ++ List(coreArea.head)).zipped.foreach { (a,b) => b.core.c2c.to >> a.core.c2c.from }`(DataPath_xN.scala:236-247) — 코어들을 **원형(ring)** 으로 연결. 비동기일 경우 사이에 CDC 큐(`queue(size=32, pushClock=..., popClock=...)`) 삽입. 이것이 모델을 코어 수만큼 **텐서 병렬(tensor-parallel)** 분할하는 통신 백본이다.

### 3.3 MulAddSGNew / MulAddEngine(New) / core.MulEngine — 행렬곱 엔진 (★ systolic 아님)
- **MulEngine(core/MulEngine.scala):** systolic array가 **아니다**. 구조는 *distributed-RAM 기반 스트리밍 dot/axpy 엔진*:
  - 온칩 `Mem(Bits(parallelBit), maxFirstDim)` 에 `ram_style="distributed"` 부여(MulEngine.scala:64-68) — LUTRAM 기반 작은 작업버퍼.
  - `Config()` 번들(MulEngine.scala:22-32)이 `firstDim`(8b)/`secondDim`(16b)/`isAxpy`(1b) 를 디코드 → `StreamDemux`로 dot 경로/axpy 경로 분기(MulEngine.scala:53-57). 즉 **하나의 엔진이 cfg 워드에 따라 dot-product(GEMV)와 axpy(스칼라·벡터 누적) 두 모드를 시분할**.
  - `LoopsCntGen.wireOvf(List(firstDim, secondDim), enInc)`(MulEngine.scala:80) 로 2중 루프 카운터를 생성 → 가변 차원 행렬의 타일 순회를 cfg로 제어.
- **MulAddEngineNew(top/MulAddEngineNew.scala):** `MulEngine`(곱) + `AddEngineNew`(누산/axpy)를 묶고, cfg를 `StreamFork2`로 둘에 분배(MulAddEngineNew.scala:53-55). `mul`→`add.io.mulRes`로 직결, 출력은 `vecOut`(병렬 2048b)과 `scalarOut`(직렬 16b, dot 결과)로 분리.
- **MulAddSGNew(top/MulAddSGNew.scala) = "Split-bank" 래퍼:** `require(isPow2(split))`(:37). `banks = Array.fill(split)(new MulAddEngineNew(width, bankLen/split, ...))`(:69-73). 즉 **2048비트 입력을 `split`(=`sgSplit=4`)개로 `subdivideIn`(:75-77) 하여 4개 엔진이 병렬로 부분합 계산** → 공간 병렬화.
  - 4개 뱅크의 `scalarOut`을 `Fp32AccEngine`(:115)로 모아 **FP16→FP32 승격 후 합산 트리(`fp32Add * log2(split)`)→FP16 강하**(:131-133). 이것이 도트프로덕트의 **FP32 누산** 핵심.
  - `postCfgTag := Delay(banks.head.io.postCfgTag, fp32Add_latency*log2Up(split)+toFp32_latency)`(:133) — 태그를 누산 트리 지연만큼 정렬.
- 정리: **(공간) split 병렬 뱅크 + (정밀) FP16 곱·FP32 누산 + (모드) dot/axpy 시분할** 의 GEMV 엔진. 모든 곱/덧셈/누산기는 외부 주입 함수(Xilinx FP IP).

### 3.4 NormSubModNew + norm.RMSNormFp32 — RMSNorm (FP32)
- LLaMA는 LayerNorm 대신 **RMSNorm** 사용. `RMSNormFp32`(NormSubModNew.scala:70-74)는 `fp16toFp32`, `fp32mul`, `fp32acc`, **`fp32rsqrt`**(역제곱근) 함수를 받아 **FP32 정밀도로 RMSNorm 계산** (제곱합 누산→rsqrt→스케일 곱). EdgeLLMInst.scala:172-173에서 `fp32acc22`, `fp32rsqrt32` 주입.
- 입력은 멀티코어 출력(`allGatherOut`/`allReduceOut`)을 `FlowGate.keepTag`로 태그 필터 후 `FlowMux`로 머지(NormSubModNew.scala:38-45). `attnLnTag`/`logitsLnTag` 구분으로 **attention 직전 norm 과 lm_head 직전 norm 을 같은 하드웨어로 시분할**(:51-54, 76-79). `dim` 카운터로 한 벡터(4096) 경계 검출.

### 3.5 BusInSubModNew + convert.Int4Int8FP16Conv — 온칩 역양자화 (W4 / KV INT8)
weight는 DRAM/HBM에 **압축(INT4 + scale + zero)** 으로 저장되고, 엔진 직전에 온칩에서 FP16으로 복원된다.
- **Int4Int8FP16Conv(convert/Int4Int8FP16Conv.scala):** `selInt8` 로 **INT4(weight) / INT8(KV-cache)** 두 경로 선택(:31-49).
  - INT8: 두 비트 입력을 누적(`io.inputData ## dataDly`)해 bankLen 슬라이스(:46); INT4: `subdivideIn(bankLen).map(_.resize(8))`(:47).
  - **zero-point 빼기:** `dSub = subDiv.map(ds => (ds.asUInt.expand.asSInt - zero).asBits)`(:62), zero는 `zeroInt4/zeroInt8` 스트림에서 mux(:35-38). 즉 **uint - zero** 의 비대칭 양자화 복원.
  - 변환기 `convertor`(=`util.fp16int9d4.from`/`fp16int5d4.from`, 정수→FP16 IP)를 bankLen개 병렬 적용(:68), 출력은 `Fp16ScaleDown.vec(wPy, 2)`(:73) 로 스케일 보정(model2bin 의 `scale_up` 과 대응, 3.10).
- **Int2FP16(convert/Int2FP16.scala):** 위의 단순화 버전 — busWidth를 dataWidth 단위로 쪼개 `(uint - zero)` 후 변환기 적용, `group>1` 이면 `History`로 묶어 bankLen 벡터로 재패킹(:51-75). 동일한 W4A16 복원 패턴.
- **BusInSubModNew(top/BusInSubModNew.scala):** AXI 입력 버스(`io.bus`, Fragment+tuser)를 받아 (a) prefill/token-square 경로는 `Serial2Parallel`(:106-109), (b) weight/KV 경로는 위 `Int4Int8FP16Conv`(:111-116) 로 보내고, 결과를 `LargeBankFifo`(:118)에 적재 후 `io.wkv`(엔진 입력)로 공급. `kvHit = kvInTag.map(_===io.bus.tuser).reduce(_||_)`(:57) 로 KV 여부 판정 → `conv.io.selInt8 := kvHit`(:112). FIFO availability 기반 backpressure(`io.bus.ready := fifo.availability >= convert_latency+3`, :147-150).

### 3.6 quant 패키지 — 동적(activation/KV) 양자화 W4A16 보조 + KV INT8 생성
weight는 오프라인(model2bin) AWQ 양자화지만, **KV-cache(K,V)는 런타임에 동적으로 INT8 양자화**된다.
- **FindRange(quant/FindRange.scala, QuantWrapper.scala:39):** headDim 길이 벡터의 min/max 탐색(`lt_func` 사용).
- **GetScaleZero(quant/GetScaleZero.scala):** `scale = (max-min)/maxIntFP16`(:39-41, maxIntFP16Init=0x5bf8=FP16 상수), `zero = round(-min/scale)` 를 FP16→Int 변환(:43-56)으로 산출하고 `[0, 2^q-1]` 클램프(:54-59). 비대칭 양자화의 scale/zero 산출 회로.
- **LinearQuant(quant/LinearQuant.scala):** `q = clamp(round(x/scale) + zero, 0, 2^q-1)`(:30-46). `x/scale`(div_func)→정수변환(convert_func=`fp16toint9d4`)→`+zero`→포화. 표준 비대칭 양자화 식의 하드웨어 구현.
- **QuantWrapper(quant/QuantWrapper.scala):** Find→GetScaleZero→LinearQuant 파이프라인을 묶고, scale/zero 를 `repeat(headDim)`(:54,59)로 한 그룹 전체에 브로드캐스트. K/V 두 포트(`numOfPort=2`)를 태그로 게이팅(:33-35). 출력 `afterQuant`(headDim 단위 last 표시, :74-87), `quantScale`, `quantZero` → AttnSubMod를 거쳐 KV-cache로 저장. **양자화 폭 `quantWidth = kvQuantWidth = 8`**(DataPath.scala:154).

### 3.7 AttnSubMod + attn/rope 서브모듈 — RoPE, QK, Safe-Softmax
- **RoPE:** `SerialRoPE`(AttnSubMod.scala:75-85) — Rotary Position Embedding. `lowPcs/highPcs mul_func`, `toInt/fromInt` 변환을 받아 위치 회전 적용. `rope.io.pos := status.token`(:175) 로 현재 토큰 위치 주입. `ropePoint = 1<<14`(EdgeLLMInst.scala:80) = sin/cos LUT 해상도.
- **QKMul(attn.QKMul, AttnSubMod.scala:112-119):** Q·K^T 도트프로덕트, `sqrtHeadDim`(=√headDim)로 스케일링(softmax 전 1/√d). `acc_func`로 누산.
- **Safe-Softmax:** `SerialSafeSoftmax`(AttnSubMod.scala:121-132) — **수치안정 softmax**(max 빼고 exp). `lt_func`(max용), `sub_func`, `acc_func`(denom 합), `div_func`, **`exp_func`** 사용. 대안으로 FP32판 `SerialSoftmaxFp32`(주석처리, :134-146)도 존재.
- **KV 양자화 결선:** RoPE 출력과 지연된 dotOut을 `QuantWrapper`에 공급(:154-156) → K/V를 INT8로 양자화, scale/zero/quantized 출력을 상위(DataPath)에서 `KvScaleZeroPacker`로 패킹해 KV-cache로 기록(DataPath.scala:551-555, 683-690).

### 3.8 util.ExpFunc / Fix48ToFp16 / Fp16ToFix24 — 지수·고정소수 변환
- **ExpFunc(util/ExpFunc.scala):** softmax 의 exp. 입력을 `FlowMux`로 다중포트 머지(:38) 후 **언더플로 클립** `if x < fp16(-16)(0xcc00) then x := -16`(:43-46) → `exp_func`(주입된 Xilinx exp IP, 예 `fp16ex12`) 적용. 즉 exp 자체는 IP가 처리하되 클리핑·포트조정 래퍼. (별도 LUT2/3/4.scala 가 `util`에 있어 일부 함수는 LUT 근사로 구현됨 — 본 모듈은 IP exp 사용.)
- **Fix48ToFp16(util/Fix48ToFp16.scala):** 48비트 고정소수 → FP32(`fp32fix48d6.from`) → FP16(`fp32toFp16.to`)(:12-16). 누산 결과(긴 고정소수)를 FP16으로 되돌릴 때 사용. `Fix32ToFp16`도 동일 패턴(:25-39).
- **Fp16ToFix24(util/Fp16ToFix24.scala):** 역방향 FP16→FP32→24비트 고정소수(:12-16). 고정소수 누산 경로 진입용.
- 이들은 모두 **함수 객체(object) + Xilinx FP IP 래퍼 조합** 으로, 정밀도 변환을 모듈식으로 캡슐화.

### 3.9 c2c.AllReduce / busdemux — 멀티코어 통신 & 버스 분배
- **AllReduce(c2c/AllReduce.scala):** `require(isPow2(numOfCore))`(:16). 링을 한 바퀴 돌며 부분합을 누적: `idCnt` 카운터가 `numOfCore-1` 도달(`idOvf`) 시점에 `acc_func`(=`fp16acc16`)로 누산 완료(:44-62). **텐서병렬 분할된 출력(예: o_proj, down_proj, logits)의 코어 간 합산**. `listOfTag`로 AllReduce 대상 텐서만 게이팅(:28-32). `numOfCore==1` 이면 패스스루(:33-37).
- **AllGatherSubModNew(top/AllGatherSubModNew.scala):** `numOfCore==4` → `LowLatencyNode`, 그 외 → `AllGatherNode` 선택(:58-60). `dotOut/resOut/p2sOut` 를 태그 게이팅 후 `FlowMux`로 노드 입력에 머지(:52-56). AllGather(분산된 활성을 모음)와 AllReduce(합산) 두 컬렉티브를 제공 → 코어 간 텐서병렬 동기화 백본.
- **AxiBusDistributor(busdemux, DataPath.scala:329-346):** DRAM에서 들어온 단일 AXI read 스트림을 태그(`lnScaleBusTag`/`denseBusTag`/`kvCacheBusTag`/`sparseDot`/`sparseAxpy`...)별로 **Dense/Sparse/KvCache 케이스로 분배**(busdemux/DenseCase·SparseCase·KvCacheCase.scala). MLP의 U/G/D 행렬(sparse)과 attention(dense), KV-cache를 서로 다른 디패킹 경로로 라우팅.

> **요지(섹션3):** 연산 코어는 systolic이 아니라 **distributed-RAM 스트리밍 GEMV 엔진을 split만큼 공간복제**한 구조이며, 데이터타입은 **FP16 활성 + INT4(AWQ weight)/INT8(KV) 압축 + FP32 누산/Norm/Softmax**. 모든 산술기는 **Xilinx FP IP를 함수로 주입**하는 고도 파라미터화 설계이고, 멀티코어는 **링 기반 AllGather/AllReduce 텐서병렬**.

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 오프라인: 가중치 양자화·패킹 (python/model2bin.py)
1. **AWQ 4-bit 로드:** `AutoAWQForCausalLM`/`WQLinear_GEMV` 로 LLaMA2-7B 의 AWQ 양자화 weight(`qweight`/`scales`/`qzeros`)를 사용(model2bin.py:1-18, 235-298). LM head/down_proj 등 일부는 런타임 `pseudo_quantize_tensor_dot/axpy`(group=128 비대칭, max_int=2^4-1)로 W4 양자화(:164-196, 465).
2. **언패킹/패킹:** `pack_int32_to_uint8`(int32에 8개의 4비트 패킹 분해, :24-31), `pack_uint4_array`(2개 uint4→1 byte, :44-49), `pack_weight_scale_zero_fast`(zero/scale/weight를 bankLen 단위로 인터리브, :102-127) → **HW 디패킹 순서(zero, scale*4, weight)** 와 정확히 일치하도록 바이트 레이아웃 생성.
3. **scale_up:** scale에 `scale_up` 곱(:247,279,...) → HW의 `Fp16ScaleDown`(3.5)과 짝. FP16 동적범위 보정.
4. **멀티코어/멀티채널 분할:** `allgather_roll`(코어별 텐서 회전, :34-42), `dma_split_bytes`(busWidth/split 인터리브, :156-161), `split_by_page_size`(PAGE_SIZE=8192 정렬, :204-217). `num_core==4` 분기 처리(:281-284, 444-450).
5. **메모리맵 직렬화:** `gen_head_bin`(QKV+빈 KV-cache 영역, :536-550) → `gen_layer_bin`(attn_norm, head들, o_proj, mlp_norm, G/U/D, :552-564) → `gen_model_bin`(embed → 32 layer → final norm → lm_head, :566-573). PS+PL 하이브리드용 `first_half/second_half`(:575-588). 이 직렬 순서가 `schedule/LLMCfg.scala` 의 `tfOsPerCore` 오프셋 테이블(:83-109)과 정확히 대응 — **SW 패킹과 HW 주소생성이 같은 메모리맵을 공유**.
6. 결과 `.bin` 을 SD카드/DDR/HBM에 적재(README 의 BaiduPan/GDrive 링크 또는 `load_param.sh`+XDMA).

### 4.2 온라인: 한 토큰 디코딩 (HW)
호스트가 토큰 인덱스를 AXI-Lite로 주입(`Xil_Out32(BASE, 0x00030000+tok)`, host_sdk.c:115) → `tokenIndex` 스트림(DataPath.scala:186) → 각 레이어마다:

```
DRAM/HBM weight(.bin)
  └─ AXIDataMover/SplitAxiDatamover (util) ── GenMemCmdLenAlign(cmdGen) 가 주소/길이 명령 생성
       └─ AxiBusDistributor (태그별 Dense/Sparse/KvCache 분배)
            ├─ BusInSubModNew → Int4Int8FP16Conv (INT4 weight → FP16 / INT8 KV → FP16)
            │       └─ MulAddSGNew (split 4뱅크 GEMV, FP16 곱·FP32 누산)
            │            ├─ scalarOut(dot 결과) → AttnSubMod(RoPE→QK→SafeSoftmax) / ScalarOutSubMod(SiLU gate)
            │            └─ vecOut(병렬) → VecOutSubMod / ResidualBuffer
            ├─ NormSubModNew (RMSNorm FP32) ── attn 직전 / lm_head 직전
            └─ KV: QuantWrapper(동적 INT8) → KvScaleZeroPacker → KV-cache write
  ── residual add (SerialResAdd) ── (멀티코어) AllGather/AllReduce(c2c) ──
  └─ 마지막 layer 후 lm_head GEMV → GreedySampler(argmax, DataPath.scala:519) → argMaxIndex
       └─ AXI-Lite status 로 호스트가 polling (host_sdk.c:127-134, bit31=valid, bits[30:16]=token)
```

- **메모리 계층:** off-chip(DDR4/HBM/PS-RAM, weight+KV-cache) → AXI DataMover(util) → on-chip `LargeBankFifo`/`StreamFifo`(util, BRAM/URAM) → MulEngine 내 `distributed`(LUTRAM) 작업버퍼. **weight는 스트리밍(저장 안 함), KV-cache만 누적**.
- **병렬화 3계층:** (1) **코어 복제** `numOfCore`(텐서병렬, §3.2), (2) **split 뱅크** `sgSplit=4`(GEMV 공간병렬, §3.3), (3) **bankLen=128 FP16 SIMD 레인**(데이터병렬). 추가로 (4) **DMA_SPLIT**(메모리채널 병렬, §3.2/4.1).
- **파이프라이닝:** 모든 산술기 latency를 `Delay(sig, latency)`로 정적 정렬(예 AttnSubMod.scala:148-152, MulAddSGNew.scala:133). `m2sPipe`/FIFO로 스테이지 분리. cfg/tag가 데이터와 함께 흐르며 다운스트림 라우팅 결정(태그-드리븐 파이프라인).
- **멀티FPGA/멀티칩:** `c2c` 패키지(`Node`/`LowLatencyNode`/`AllReduce`/`AllGatherNode`) + `util.AuroraInterfaceTest` 존재. 단, 본 repo 의 `DataPath_xN`은 **단일 FPGA 내 멀티코어 링**을 구성(DataPath_xN.scala:236-247); 진짜 멀티FPGA(Aurora 칩간)는 c2c 의 `destBit=idWidth` AxiFrame 으로 확장 가능한 형태이나 본문 결선상으로는 온칩 링이 주용도(README의 멀티노드는 별개 프로젝트 TeraFly, README.md:216).

---

## 5. HW/SW 매핑

| 계층 | 산출물 | 역할 | 근거 |
|---|---|---|---|
| **SW(오프라인)** | `python/model2bin.py` | LLaMA2-7B(AWQ W4) → 보드/코어/채널별 `.bin` 패킹 (zero/scale/weight 인터리브, page 정렬) | model2bin.py 전반, §4.1 |
| **HW 정의(소스)** | `scala/src/main/scala/**` (SpinalHDL) | 데이터패스/엔진/양자화/통신 RTL 생성기 | EdgeLLMInst.scala:62 `generateVerilog` |
| **HW(생성물)** | `DataPath_xN.v` 등 Verilog | SpinalHDL 출력 RTL (보드별 Vivado 배포 프로젝트에 포함) | README.md:170-172 (auto-generated) |
| **HW 통합(생성물)** | 보드별 Vivado 배포 프로젝트 + `.xsa` | DMA/AXI/MIG/HBM/XDMA + 가속기 BD 통합, 비트스트림 | README.md:43-49 |
| **SW(온라인)** | `*_sdk.c`(KV260/ZCU104 bare-metal), `au250/host_sdk.c`(Linux+XDMA) | 토크나이저(llama2.c 차용) + AXI-Lite 제어 + 토큰 polling 루프 | host_sdk.c:63-160, README.md:224 |

- **AXI-Lite 레지스터 맵(host_sdk.c 근거):** `BASE+0x00`=토큰 입력 명령(0x00030000=prefill, 0x00050000=마지막 prefill, 0x00090000=decode, host_sdk.c:115/120/134), `BASE+0x04`=출력 토큰 status(bit31 valid, [30:16] index, :127-131), `BASE+0x10`=total_token/timer(:107,147), `BASE+0x80`=next-token trigger(:135). DataPath.scala 의 `toAxiLite`(tokenCnt/argMaxVld/argMaxIndex/prefill/layerCnt, :178-184, 717-721)와 대응.
- **메모리맵 공유:** `model2bin.gen_*_bin` 직렬 순서 ↔ `schedule/LLMCfg.tfOsPerCore` 오프셋 ↔ HW `GenMemCmdLenAlign(cmdGen)` 주소생성(DataPath.scala:163-176). **세 곳이 동일 레이아웃을 약속**해야 동작 — 모델 강결합의 근원.
- **보드 차이:** KV260=PS 4GB, ZCU104 PL/PS+PL=DDR4 분할, U250=DDR4 4채널+XDMA(load_param.sh, README.md:110-130). 차이는 `EdgeLLMInst`의 `numOfCore/DMA_SPLIT/cmdAddrWidth` 와 Vivado BD(메모리 IP)로 흡수.

---

## 6. 빌드 / 실행

- **RTL 생성(SW→HW):** SpinalHDL/Scala 빌드(sbt/mill 추정 — 빌드파일은 본 분석 범위 밖)로 `EdgeLLMInst` 의 `main` 실행 → `DataPath_xN.v` 등 Verilog 출력. 보드 변형은 `EdgeLLMInst.scala`의 설정 블록 주석 토글로 선택(§3.1).
- **하드웨어 통합:** 생성 Verilog를 보드별 Vivado 프로젝트(생성물)에 넣어 합성/임플/비트스트림 → `.xsa` 익스포트(README.md:43-49).
- **모델 바이너리:** LLaMA2-7B 다운로드 → AWQ 4-bit 양자화 → `model2bin.py`(Jupyter) 실행 → 보드별 `.bin` 생성(README.md:67-78). 또는 제공 바이너리 다운로드(README.md:84-108).
- **임베디드(KV260/ZCU104):** Vitis bare-metal 플랫폼 + Hello World 앱 생성 → `helloworld.c`를 제공 `*_sdk.c`로 교체 → 빌드/배포(README.md:51-59). 토크나이저 `tkz.bin` 필요(host_sdk.c:45).
- **Alveo U250:** XDMA DMA 모드. `dma_ip_drivers` 설치 → `load_param.sh`+`host_sdk.c`를 `XDMA/linux-kernel/tools/` 복사 → `gcc host_sdk.c -o host_sdk` → `load_param.sh`로 파라미터 로드 → 실행(README.md:110-130). 실행 시 stdin 으로 프롬프트 입력, `[INST] ... [/INST]` 렌더링 후 디코딩(host_sdk.c:96-101).

---

## 7. 의존성

- **HW 생성:** SpinalHDL(`spinal.core`, `spinal.lib`, `spinal.lib.bus.amba4.axi/axis/axilite`) + Scala. 모든 부동소수 연산은 **Xilinx Floating-Point IP** 래퍼(`util.fp16*`, `util.fp32*`, `XilinxFloatIP*.scala`) 와 **AXI DataMover/MIG/HBM IP**(`AXIDataMoverWrapper`, `HBMWrapper`, `SplitAxiDatamover`) 에 강하게 의존 → **Xilinx 툴/디바이스 전용**(이식성 제약).
- **SW 오프라인:** `awq`(AutoAWQ), `transformers`(LLaMA modeling), `torch`, `numpy`(model2bin.py:1-18). LLaMA2-7B 가중치(외부 다운로드).
- **SW 온라인:** 표준 C(`stdio/stdlib/string/sys/time`), Xilinx `Xil_In32/Xil_Out32`(bare-metal) 또는 XDMA `dma_to_device`/mmap(Linux). 토크나이저는 **Karpathy llama2.c** 차용(README.md:224).
- **외부 데이터:** `tkz.bin`(토크나이저), 보드별 weight `.bin`(BaiduPan/GoogleDrive, README.md:84-108).

---

## 8. 강점 / 한계 / 리스크

**강점**
- **극도의 파라미터화:** 연산기·정밀도·latency 를 전부 함수/Int 인자로 주입 → FP16/FP32/Int 변형, latency 변경, 보드 변형(`numOfCore`/`DMA_SPLIT`/`cmdAddrWidth`)을 소스 수정 없이 재구성(EdgeLLMInst.scala).
- **다층 병렬화 + 메모리대역폭 지향:** 코어복제(텐서병렬)·split뱅크·SIMD레인·DMA분할로 weight-bound 디코딩을 대역폭에 맞춰 스케일(논문 주제와 일치).
- **압축-온칩복원 파이프라인:** weight는 INT4로 스트리밍(저장 안 함), KV만 INT8 누적 → 용량/대역폭 절감. SW 패킹과 HW 디패킹이 바이트 단위로 정합.
- **수치 안정성 배려:** safe softmax(max 차감), exp 언더플로 클립, FP32 누산/Norm/rsqrt.
- **재현성:** 4종 보드 동작 검증 + 사전생성 바이너리 + 데모 제공.

**한계**
- **LLaMA2-7B 강결합:** 차원/메모리맵/스케줄이 모델에 하드코딩(LLMCfg.scala, README.md:165-168). 타 모델은 RTL 수정 필요.
- **디코딩 전용:** 프리필 미가속 → 긴 컨텍스트 초기화 느림. 컨텍스트 1024 토큰 상한, 단일턴(README.md:174-186).
- **systolic 아님:** distributed-RAM 스트리밍 GEMV — 처리량은 대역폭/뱅크수에 의해 제한(디코딩엔 적합하나 prefill/대배치엔 비효율).
- **생성 Verilog 가독성 낮음**, 코드 주석 부족(README 자기명시).

**리스크**
- **Xilinx 종속:** FP IP/DataMover/HBM/XDMA 전제 → 타 벤더 이식 시 util 계층 대거 재작성.
- **3중 메모리맵 동기화 위험:** model2bin ↔ LLMCfg ↔ cmdGen 주소가 어긋나면 침묵하는 오동작(KV260 바이너리 "currently buggy" 명시, README.md:90).
- **`scale_up`/FP16 동적범위:** 양자화 스케일 보정이 수동 상수 — 모델/레이어별 오버플로 가능성.
- **빌드 환경 의존:** SpinalHDL/Scala 빌드 정의(build.sbt 등)와 Vivado 버전 의존이 문서화 약함.

---

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 관점 시사점

우리는 **고처리량 ViT/Transformer 가속기(HG-PIPE 계열)** 와 **XR 시선추적** 이 목표다. llama-fpga 는 LLM 디코딩(저배치, weight-bound)이라 워크로드 성격은 다르지만, **재사용 가능한 모듈/패턴**이 상당하다.

1. **파라미터화 데이터패스 패턴(직접 차용):** 연산기/정밀도/latency 를 함수·정수 인자로 주입하는 설계(EdgeLLMInst.scala, MulAddSGNew.scala)는 HG-PIPE 의 다단 파이프라인을 **정밀도/타일크기/병렬도 스윕** 가능하게 만든다. ViT 의 patch-embed/attention/MLP 블록을 동일하게 인자화하면 DSE(latency↔area)가 쉬워진다.
2. **W4A16 + FP32 누산 양자화 모듈(`quant`/`convert`):** `LinearQuant`(비대칭 q), `GetScaleZero`(scale/zero 산출), `Int4Int8FP16Conv`(온칩 INT4/INT8→FP16 복원)는 **ViT weight를 INT4/INT8로 압축 저장→온칩 복원** 하는 데 거의 그대로 이식 가능. ViT 의 LayerNorm/attention 활성에는 INT8 동적양자화(KV 경로의 `QuantWrapper`)가 좋은 출발점. group-wise(128) 비대칭 양자화 + scale_up/ScaleDown 짝은 우리 양자화 파이프라인 설계의 참조 구현.
3. **FP16↔FP32↔Int 변환 캡슐화(`util/Fix48ToFp16`, `Fp16ToFix24`, `convert/Int2FP16`):** 누산 정밀도 경로(FP16 곱→고정소수/ FP32 누산→FP16 강하)를 object 단위로 캡슐화한 방식은 HG-PIPE 의 MAC 어레이 누산기 설계에 재사용. ViT 의 긴 reduction(D=768/1024)에서 FP32 누산은 정확도에 중요.
4. **exp LUT/Softmax 인프라(`ExpFunc`, `SerialSafeSoftmax`):** ViT attention 의 softmax 에 **수치안정(max 차감)+exp 언더플로 클립** 패턴을 그대로 적용. 우리는 throughput 우선이므로 exp 를 `LUT2/3/4.scala`(piecewise LUT)로 대체해 DSP/latency 절감하는 변형을 검토할 가치가 큼 — 이 repo 가 IP-exp 와 LUT 두 옵션을 모두 보유.
5. **RMSNorm/LayerNorm FP32 정규화 블록(`NormSubModNew`+`RMSNormFp32`):** reduction→rsqrt→scale 구조는 ViT LayerNorm(평균/분산 2-pass)로 약간만 변형하면 재사용. rsqrt 를 IP 로 주입하는 방식 유지.
6. **멀티보드 이식 흐름(직접 참조):** **단일 소스(SpinalHDL) → `numOfCore`/`DMA_SPLIT`/`cmdAddrWidth` 토글 → 보드별 Vivado 배포 프로젝트** 라는 워크플로는 우리의 ZCU104/KV260/Alveo 멀티타깃 전략의 청사진. 특히 `DataPath_xN` 의 **코어복제+클럭도메인분리+ring c2c** 는 멀티-SLR(U250) 배치에 직접 응용 가능.
7. **SW↔HW 메모리맵 공유 규율:** model2bin ↔ LLMCfg ↔ cmdGen 의 3자 메모리맵 합의는 우리의 weight 패킹 툴 ↔ RTL 주소생성기 설계에 그대로 적용해야 할 **계약(contract) 패턴**. 단, 이 repo 의 "buggy 바이너리" 사례처럼 **동기화 검증(테스트)** 을 반드시 자동화할 것.
8. **XR 시선추적 관점:** 시선추적은 소형 CNN/ViT 의 **저지연 단일프레임 추론**이 핵심 → llama-fpga 의 weight-streaming(저장 안 함)·동적양자화·태그드리븐 파이프라인은 **온칩 메모리 부족한 임베디드(KV260)** 에서 모델을 흘려보내며 추론하는 데 유용. 다만 시선추적은 batch=1·실시간 latency 가 관건이라, 이 repo 의 대역폭지향(throughput)보다 **HG-PIPE 식 완전 파이프라인(layer-by-layer 상주)** 이 더 적합 — llama-fpga 에서는 *양자화/변환/Norm/Softmax 서브모듈*만 부품으로 떼어 쓰는 것이 현실적.

**요약 차용 우선순위:** (높음) `quant`+`convert` 양자화·역양자화, FP16↔Int 변환 util, 파라미터화 주입 패턴, 멀티보드 토글 워크플로 / (중간) Softmax·Norm 블록, ring c2c 텐서병렬 / (낮음, 워크로드 상이) MulEngine 시분할 GEMV(우리는 완전파이프라인 선호).

---

## 10. 근거 / 한계 표기

**직접 Read 하여 라인 근거로 분석한 파일(자체 소스):**
- `README.md` (전체), `python/model2bin.py` (1-588 전체), `au250/host_sdk.c` (1-160)
- `top/`: DataPath.scala(전체), DataPath_xN.scala(1-90, 160-248), EdgeLLMInst.scala(전체), MulAddEngine.scala, MulAddEngineNew.scala, MulAddSGNew.scala, AttnSubMod.scala, NormSubModNew.scala, BusInSubModNew.scala, AllGatherSubModNew.scala(1-60)
- `core/`: MulEngine.scala(1-80)
- `quant/`: LinearQuant.scala, GetScaleZero.scala, QuantWrapper.scala
- `convert/`: Int2FP16.scala, Int4Int8FP16Conv.scala
- `util/`: ExpFunc.scala, Fix48ToFp16.scala, Fp16ToFix24.scala
- `c2c/`: AllReduce.scala
- `schedule/`: LLMCfg.scala

**파일목록(Glob)만 확인, 본문 미정독(역할은 import/네이밍 근거 추정):**
- `core/`: AddEngineNew.scala, AddEngineBk.scala, Fp32AccEngine.scala, VecNto1.scala, Vec2to1.scala (← MulAddSGNew/MulAddEngineNew 결선과 클래스명으로 역할 추정)
- `quant/FindRange.scala` (QuantWrapper.scala:39 사용처로 역할 확인, 내부 미정독)
- `c2c/`: AllGatherNode.scala, Node.scala, LowLatencyNode.scala, Reorder.scala, ReduceFilter.scala, AllGatherNode 등 (AllGatherSubModNew import 로 역할 추정)
- `busdemux/`: AxiBusDistributor.scala, DenseCase/SparseCase/KvCacheCase.scala (DataPath.scala:329-346 결선으로 역할 추정)
- `top/`: StateGen/GlobalStateGen/AxiLiteCtrl/ScalarOutSubMod/VecOutSubMod/VecOutBuf/AddressRemap, 구버전(MulAddEngine/MulAddSG/NormSubMod/BusInSubMod/AllGatherSubMod) — DataPath 가 *New 버전을 사용하므로 구버전은 미사용/레거시 추정
- `util/` 다수 FIFO/DMA/LUT/HBM 래퍼 — 인프라 유틸로 분류만, 개별 미정독
- `norm/RMSNormFp32` — NormSubModNew.scala:4 import 로 존재·역할 확인, 파일 내부 미정독
- `cfgGen.LLaMA2_7B`, `cfgGen.GenMemCmdLenAlign`, `GenCfg`, `InsertCfg`, `GreedySampler`, `mlp.*`, `attn.QKMul/SerialSafeSoftmax/KvScaleZeroPacker`, `rope.SerialRoPE` — DataPath/AttnSubMod 에서 사용처·인자로 역할 확인, 정의 파일 미정독

**미완/불확실 영역:**
- **build.sbt / mill 등 빌드 정의** 미확인 → 정확한 RTL 생성 커맨드는 README 추론.
- **systolic 여부:** MulEngine.scala 80라인까지만 정독 → "distributed-RAM 스트리밍 GEMV(systolic 아님)" 결론은 RAM 구조·dot/axpy 시분할 근거이나, AddEngineNew 의 누산 어레이 세부는 미정독.
- **멀티FPGA(Aurora 칩간):** `util/AuroraInterfaceTest.scala`, c2c destBit 구조로 *가능성* 확인했으나, 본 repo 의 `DataPath_xN`은 온칩 멀티코어 링 결선만 정독 — 진짜 멀티FPGA 배포 흐름은 미검증(README 는 멀티노드를 별도 프로젝트 TeraFly 로 안내).
- **KV260 바이너리 "buggy"** 원인은 코드상 미확인(README 명시만).
- 작업 지시의 "Chisel" 은 실제 **SpinalHDL** 임을 §1에서 정정.
