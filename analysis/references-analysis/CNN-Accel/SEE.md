# SEE 코드베이스 정밀 분석

> 분석 대상: `REF/CNN-Accel/SEE/ESDA`
> 작성 원칙: 실제 소스 Read/Grep 후 `파일:라인` 근거 표기. 라인 근거 없는 추론은 "추정", 코드로 확인 불가는 "확인 불가".
> 분석 제외(이름만): `.bit`/`.hwh`, `weight.h`, `data/*.txt`, `.npy`, build 산출물.

---

## 0. 핵심 전제: SEE = ESDA의 사례연구 래퍼

`SEE/ESDA`는 **ESDA(`REF/CNN-Accel/ESDA`)와 동일한 프로젝트의 사본**이며, eventNet(이벤트 기반 비전 CNN)을 **데이터셋별로 적용한 사례 모음**이다. 다음은 ESDA와 SEE/ESDA가 동일함을 보이는 직접 근거다.

- `optimization/solver/scip_solver.py`, `optimization/pipeline.py`, `hardware/gen_prj.py`, `hardware/template_e2e/*`, `hardware/board/{evaluate.py,power_monitor.py,hw_e2e.py}`가 ESDA와 **동일 파일 구조**로 존재(Glob 대조). `hardware/README.md`도 동일 절차(`SEE/ESDA/hardware/README.md:1-206`).
- 따라서 **ESDA 본체(optimization + HLS 템플릿)의 정밀 분석은 `ESDA.md`를 참조**하고, 본 문서는 **SEE 고유 부분**에 집중한다:
  1. `hardware/cfgs/*.json` — 데이터셋별 DSE 결과(5개 데이터셋).
  2. `eventNet/archived_hw/{SEE-A,SEE-B,SEE-C,SEE-D,MobileNetV2}` — 합성/배포된 아키텍처 변형(특히 **SEE-B vs SEE-D** 비교).
  3. `hardware/board/{evaluate.py,power_monitor.py,hw_e2e.py}` — ZCU102 보드 실측 평가(ESDA와 동일 파일이나 본 문서에서 정밀 분석).

- 명칭 추정: **SEE = Sparse Event-based ... (Evaluation/Engine)** 계열로 보이나, 코드/README에 풀네임·논문 표기 없음 → **확인 불가**. archived_hw의 `SEE-A~D`는 동일 데이터플로 템플릿의 **네트워크/병렬도 변형**(설계공간 탐색 산물)으로 판단(아래 §3.2 근거).

---

## 1. 개요

- **목적**: ESDA 가속기 템플릿을 ASL / DVS / N-Caltech101(NCal) / N-MNIST / Roshambo 등 이벤트 비전 데이터셋에 적용하여 ZCU102에서 정확도·지연·전력을 실측 평가.
- **원논문 추정**: ESDA와 동일 계열(event-based sparse-dense dataflow). 정확한 논문 **확인 불가**.
- **타깃 보드**: ZCU102 + PYNQ. 근거 `hardware/README.md:5`, `hardware/board/power_monitor.py:7-28`(ZCU102의 PMBus INA226 18채널 센서 매핑).

---

## 2. 디렉토리 구조 (SEE 고유 부분 중심)

```
SEE/ESDA/
├── optimization/                  # = ESDA와 동일 (scip_solver.py, pipeline.py) → ESDA.md 참조
├── hardware/
│   ├── README.md, gen_prj.py, common.py, template_e2e/  # = ESDA와 동일
│   ├── cfgs/*.json                # [SEE 고유] 데이터셋별 DSE 결과 8종   [정밀: §3.1]
│   └── board/                     # [정밀 분석 대상]
│       ├── evaluate.py            #   지연+전력 실측 (PYNQ)
│       ├── power_monitor.py       #   INA226 PMBus 전력 샘플러
│       └── hw_e2e.py              #   end-to-end 분류 추론
├── eventNet/
│   ├── archived_hw/               # [SEE 고유] 합성된 아키텍처 변형     [정밀: §3.2]
│   │   ├── MobileNetV2/           #   베이스 5-block
│   │   ├── SEE-A/  SEE-B/         #   변형 A,B (14-block 계열)
│   │   ├── SEE-C/  SEE-D/         #   변형 C,D (9-block 계열)
│   │   └── 각 변형: top.cpp,conv.h,conv_pack.h,linebuffer.h,mem.h,para.h,type.h,weight.h
│   │                              #   (weight.h, *.bit/*.hwh = 제외, 이름만)
│   └── hw/<dataset>/full/         # 생성 프로젝트 사본 (= gen 산출물, 제외)
```

---

## 3. 핵심 모듈 정밀 분석

### 3.1 데이터셋별 DSE 설정 — `hardware/cfgs/*.json`

8개 cfg(=`en-result.json` 사본). 각 파일은 `dataset`, `input_shape`, `input_sparsity`, MobileNetV2 `settings`(t,c,n,s,...), `total_dsp`, `total_bram`, `obj`(latency 목적치), `layers[].parallelism` 등을 담는다. 데이터셋별 의미:

| cfg | dataset | input_shape | input_sparsity | total_dsp/bram | 근거 |
|---|---|---|---|---|---|
| `DVS_1890_shift16-zcu102_80res` | DVS (DVS128 Gesture류) | 128×128 | 0.0497 | 1804 / 1049 | `cfgs/DVS_1890...json:3-8,49-50` |
| `NMNIST_shift16-zcu102_60res` | N-MNIST | 34×34 | 0.2284 | 1472 / 953 | `cfgs/NMNIST...json:3-8,49-50` |
| `Roshambo_shift16-zcu102_80res` | Roshambo (가위바위보) | 64×64 | 0.104 | (json) | `cfgs/Roshambo...json:3-8` |
| `ASL_0p5/2929_shift16-zcu102_80res` | ASL-DVS (수어) | (json) | — | — | 파일명/`evaluate.py:16` |
| `NCal_2751/w0p5_shift32-zcu102_{80,50}res` | N-Caltech101 | (json) | — | — | 파일명 |

- **데이터셋별 의미 핵심**:
  - `input_sparsity`가 데이터셋마다 다름(DVS 5%, N-MNIST 23%) → 희소도가 낮을수록(NMNIST) 동일 정확도에 더 많은 DSP/지연 필요(추정; obj=4795 vs DVS obj=2948).
  - `NCal`은 양자화 비트가 더 큼(`shift32`, 파일명) — `gen_code.py:312-315`의 "NCAL이면 SW/BW/EXP=32" 분기와 일치(101-class 난이도 대응).
  - `Roshambo`는 `shift16`이나 마스크폭 `CFG_MW=64`(ESDA `gen_code.py:316-317`; `evaluate.py:176` `mask_bits = 64 if Roshambo else 128`) — 입력 64폭에 맞춘 마스크 패킹.
  - 보드 예산 접미사 `zcu102_{50,60,80}res`가 cfg에 박혀 있어, **데이터셋×자원예산 조합**이 곧 하나의 가속기 인스턴스.

### 3.2 아키텍처 변형 비교 — `archived_hw/` (특히 SEE-B vs SEE-D)

모든 변형은 **동일한 HLS 커널 라이브러리**(`conv.h`/`conv_pack.h`/`linebuffer.h`/`mem.h`)를 공유하고, `top.cpp`의 **블록 시퀀스**와 `para.h`의 **레이어 형상·병렬도**만 다르다. 즉 SEE-A~D는 같은 dataflow 템플릿의 **서로 다른 네트워크 토폴로지/병렬도 인스턴스**.

#### SEE-B (14-block, residual 다수)
- `para.h`(`SEE-B/para.h`): 입력 `CONV1_H=60, W=80`(`:19-20`), `CONV1_OC=32`(`:18`). 블록 **BLOCK_0~13**(14개) + `CONV8`(`:299-308`) + `FC`(`:310-315`). 최종 `CONV8_OC=160`, `FC_IC=160`(`:303,313`).
- `top.cpp` 연산 시퀀스(`SEE-B/top.cpp`): `conv_3x3_first_layer`(`:91`) → BLOCK_0(stride2) → **BLOCK_1~3 residual**(`:93-95`) → ... → **BLOCK_8 residual**(`:100`) → ... → BLOCK_13 → `conv8`(`:106`) → `global_avgpool`(`:107`). residual 블록(`conv_1x1_3x3_dw_1x1_stride1_residual`)이 다수(`:93-100` 등) — **깊고 잔차 연결이 풍부한 구성**.
- 채널 진행: 32→16→16→...→128→144→160 (`para.h:18,31,247,266,303`) — 점진적 확장.

#### SEE-D (9-block)
- `para.h`(`SEE-D/para.h`): 입력 `CONV1_H=60, W=80`(`:19-20`)로 SEE-B와 동일 해상도이나 `CONV1_OC=24`(`:18`, SEE-B는 32). 블록 **BLOCK_0~8**(9개)만(`:25-198`) + `CONV8`(`:200-209`) + `FC`(`:211-216`). 최종 `CONV8_IC=256, OC=64`, `FC_IC=64`(`:203-204,214`).
- `top.cpp` 시퀀스(`SEE-D/top.cpp`): `conv_3x3_first_layer`(`:71`) → BLOCK_0(stride2,`:72`) → BLOCK_1(stride2,`:73`) → **BLOCK_2 residual**(`:74`) → BLOCK_3(stride1,`:75`) → BLOCK_4 residual(`:76`) → BLOCK_5(stride2,`:77`) → BLOCK_6,7 residual(`:78-79`) → BLOCK_8(stride1,`:80`) → `conv8`(`:81`) → `global_avgpool`(`:82`).
- 채널 진행: 24→32→48→48→64→72→...→256→64 (`para.h:18,31,50,69,89,128,187,204`) — **더 얕고(9블록) 후반부 채널이 급팽창(256)**하는 구성.

#### SEE-B vs SEE-D 차이 요약
| 항목 | SEE-B | SEE-D | 근거 |
|---|---|---|---|
| 블록 수 | 14 (BLOCK_0~13) | 9 (BLOCK_0~8) | `SEE-B/para.h:25-296` vs `SEE-D/para.h:25-198` |
| CONV1_OC | 32 | 24 | `SEE-B/para.h:18` vs `SEE-D/para.h:18` |
| 입력 해상도 | 60×80 | 60×80 (동일) | 양쪽 `para.h:19-20` |
| 최종 채널(CONV8) | 160(OC) | 256(IC)→64(OC) | `SEE-B/para.h:303` vs `SEE-D/para.h:203-204` |
| residual 분포 | BLOCK_1,2,3,5,7,8...(다수) | BLOCK_2,4,6,7(소수) | `SEE-B/top.cpp:93-100` vs `SEE-D/top.cpp:74-79` |
| 설계 성향 | 깊고 잔차 풍부, 점진 확장 | 얕고 후반 급팽창 | (상기 채널 진행) |
| 공유 커널 | 동일 conv.h/conv_pack.h/linebuffer.h | 동일 | DSP_AM 시그니처 동일(아래) |

- **공유 커널 동일성 근거**: 양쪽 `conv.h:1-8`의 `DSP_AM`이 `template<int _W_1,int _W_2> ap_int<_W_1+_W_2> DSP_AM(ap_int<_W_1> in1, ap_int<_W_1> in2, ap_int<_W_2> in3)`로 동일(`SEE-D/conv.h:1-8`, `SEE-B/conv.h:2`). 2-MAC 패킹/`rom_2p`/`LUTRAM` 바인딩도 동일(`SEE-B/conv.h:299-305,401-402`).
- **template_e2e와의 미세 차이**: archived 변형들은 마지막에 `global_avgpool`(`SEE-D/conv.h:493`)을 쓰는 반면, 현행 `template_e2e/conv.h`는 GAP+FC 융합형 `global_avgpool_linear`(template `conv.h:490`)를 쓴다 → **archived_hw가 더 이른 세대(FC 분리)**, template_e2e가 개선판(FC 융합)으로 추정.
- SEE-A(`SEE-A/para.h`)는 `CONV1_OC=64`(`:18`), `FC_IC=128`(`:175`)로 SEE-B/D보다 채널이 큰 또 다른 변형 — A/B/C/D는 **동일 데이터셋(60×80 입력)에 대한 병렬도/채널 설계공간 스윕**(추정).

### 3.3 보드 실측 평가 — `hardware/board/` (정밀)

#### 3.3.1 `evaluate.py` — 지연·전력 실측 드라이버
파일: `board/evaluate.py` (219줄).

- **`inverted_residual_block`(`:27-156`)**: PYNQ Overlay로 비트스트림 로드.
  - `__init__`(`:28-56`): `Overlay(bitstream)`(`:30`), `self.accel = overlay.top_0`(`:31`), `overlay.download()`(`:32`). 입력/출력/마스크 버퍼를 `pynq.allocate`로 물리연속 할당(int8 입력 `H*W*IC`, int8 출력 `H*W*OC`, uint8 마스크; `:38-52`). 가속기 레지스터맵에 물리주소 바인딩(`:54-56`) — `act_in_1/act_out_1/mask_1`.
  - `pack_mask`(`:58-74`): 마스크를 8비트씩 비트패킹(`:66-73`). `pad_mask`(`:76-89`): 폭을 `mask_bits`(128 또는 64) 배수로 zero-pad(`:79-88`).
  - `load_mask`(`:91-98`): `.npy` 마스크 로드→pad→pack→DMA 버퍼 기록+`flush()`(`:96-98`).
  - **`run`(`:100-143`)**: 데이터셋 npy 디렉토리(`/home/xilinx/jupyter_notebooks/event_dataset/<dataset>`, `:102-103`)에서 무작위 샘플 인덱스 선택(`:108-117`; Roshambo는 존재 파일 확인 루프 `:109-114`). 각 샘플마다 마스크 로드, `num_nz = count_nonzero(mask)`를 레지스터에 기록(`:126-127`), `CTRL.AP_START=1`로 가속기 기동(`:130`), `CTRL.AP_IDLE` 폴링으로 완료 대기(`:131-132`), `(end-begin)*1000` ms 누적(`:133-135`). 종료 후 평균 런타임(ms) 출력(`:143`). **희소도(num_nz)가 지연에 직접 반영**되는 구조(가속기가 nz만 처리하므로).
  - 데이터셋 크기 테이블(`:18-24`): ASL 20160, DVS 6959, NCAL 11831, NMNIST 19942, Roshambo 205695 샘플.
- **`main`(`:159-218`)**: `cfg.json`에서 top 형상(IH/IW/IC/OC) 파싱(`:171-176`), `mask_bits = 64 if Roshambo else 128`(`:176`), 디렉토리명에서 dataset 파싱(`:177-185`). `--enable_pm`이면 `../power_monitor.py`를 **별도 subprocess로 띄워**(`:189-194`) 동시 전력 샘플링, 실행 후 SIGINT으로 종료·로그 회수(`:200-214`). → **지연 측정과 전력 측정을 병렬 프로세스로 동기 수행**.

#### 3.3.2 `power_monitor.py` — INA226 PMBus 전력 샘플러
파일: `board/power_monitor.py` (78줄).

- **`pmbus_mapping`(`:7-28`)**: ZCU102의 18개 전력 레일을 `lm-sensors` 칩 접두사(uXX)→레일명으로 매핑. **PS 10채널**(VCCPSINTFP/LP, VCCPSAUX, ...; `:9-18`) + **PL 8채널**(VCCINT, VCCBRAM, VADJ_FMC, MGTAVCC, ...; `:20-28`). 단위는 W/mW 혼재(주석 표기).
- **`PowerMonitor.__init__`(`:32-42`)**: `sensors.init()`(`:34`), 검출된 `ina226_*` 칩을 접두사로 식별해 레일에 바인딩(`:35-39`), 누락·중복 검증(`:39-42`).
- **`record(interval, num_runs)`(`:47-71`)**: `overhead=0.027`s 보정한 실측 간격으로(`:48-50`) 매 샘플 `time.time()` + 각 센서의 모든 측정값(`f.get_value()`)을 행으로 적층(`:53-59`). KeyboardInterrupt(SIGINT)로 중단(`:60-62`). 결과를 `power_record.npy`로 저장(`:71`).
- **메인 설정**(`:74-78`): `interval=0.1`(ms 주석이나 실제 초 단위 0.1s), `num_runs=6000`(=약 10분). → 가속기 구동 중 0.1s 주기로 18레일 전력 시계열 수집.

#### 3.3.3 `hw_e2e.py` — end-to-end 분류 추론
파일: `board/hw_e2e.py` (228줄). `evaluate.py`와 골격 동일하나 **정확도 검증용 단일 추론**.

- 데이터셋별 `classes_size`(ASL 25, DVS 10, NCAL 101, NMNIST 10, Roshambo 4; `:26-32`)와 `input_shape`(`:34-40`) 추가.
- `__init__`(`:43-75`): 출력 버퍼를 `int32 ×128`로 할당(`:59-62`, logit 출력) — `evaluate.py`의 int8 활성 버퍼와 다른 점.
- `load_feat_npy`(`:77-84`) / `load_mask_npy`(`:86-96`): 실제 feature·mask npy를 DMA로 로드.
- **`run`(`:131-148`)**: feature·mask 로드→`num_nz` 기록→`AP_START`→`AP_IDLE` 폴링(`:135-143`)→출력에서 `valid_classes`개만 잘라 `argmax`로 예측(`:144-148`). 즉 가속기 출력(GAP+linear logit)을 그대로 분류에 사용.
- `main`(`:166-227`): `tb_input_feature.npy`/`tb_spatial_mask.npy`를 입력(`:176-177`)으로 단일 샘플 분류. 전력 모니터 동시 구동 옵션 동일(`:198-214`).

---

## 4. 데이터플로우

(연산 dataflow는 ESDA와 동일 → `ESDA.md §4` 참조.) **보드 측 평가 데이터플로**는:
```
npy(mask, feature) → pynq.allocate DMA 버퍼 → register_map(물리주소)
  → AP_START → [가속기: read_sparse_input → conv 파이프 → GAP] → AP_IDLE
  → (evaluate: 지연 ms 누적) / (hw_e2e: logit argmax)
  ∥ power_monitor: 0.1s 주기 18레일 전력 → power_record.npy
```

---

## 5. HW/SW 매핑 (SEE 관점)

| SW | HW/보드 | 근거 |
|---|---|---|
| `cfgs/<dataset>.json` parallelism | 합성된 `archived_hw/SEE-*` para.h 형상 | §3.1, §3.2 |
| `evaluate.py` register_map 바인딩 | 가속기 AXI-lite 슬레이브(top.cpp.tpl) | `evaluate.py:54-56` ↔ `template_e2e/top.cpp.tpl:24-28` |
| `num_nz`(희소도) | 가속기 처리 토큰 수 = 지연 결정 | `evaluate.py:126-127` ↔ `mem.h:178-194` |
| `power_monitor` 레일 매핑 | ZCU102 PS/PL 전력 도메인 | `power_monitor.py:7-28` |

---

## 6. 빌드·실행

- 가속기 생성·합성은 ESDA와 동일(`README.md:9-185`).
- **보드 직접 실행**(`README.md:172-205`): board 코드를 보드에 복사 + 비트스트림 전송 후
  - 지연·전력: `python3 evaluate.py -1 -d <bitstream-path>`(`README.md:183`) — `-1`은 전체 데이터셋 실행(`evaluate.py:105` `dataset_size if num_run<=0`).
  - e2e 분류: `python3 hw_e2e.py 1 -d <bitstream-path>`(`README.md:185`).
  - 예시: `archived_hw/MobileNetV2/hw/{top.bit, top.hwh}`를 대상으로 실행(`README.md:189-205`).

---

## 7. 의존성
- 보드: **pynq**(`Overlay`,`allocate`; `evaluate.py:2-4`), **PySensors**(`import sensors`; `power_monitor.py:2`, lm-sensors 바인딩), `numpy`, `tqdm`. 보드 OS는 PYNQ 이미지(xilinx 사용자, `/home/xilinx/...` 경로 `evaluate.py:103`).
- 합성: ESDA와 동일(Vitis HLS, Boost, Vivado).
- 데이터셋 경로 하드코딩: `/home/xilinx/jupyter_notebooks/event_dataset`(`evaluate.py:103`).

---

## 8. 강점·한계

**강점**
- 동일 dataflow 템플릿으로 **5개 데이터셋 × 여러 자원예산**을 일괄 인스턴스화(`cfgs/`) → 강력한 재사용성.
- **지연·전력 동시 실측** 인프라(`evaluate.py` + `power_monitor.py` 병렬 프로세스, 18레일 PMBus) — 정밀한 에너지/지연 트레이드오프 측정 가능.
- 희소도(`num_nz`)가 지연에 직접 반영되어, 데이터 의존적 성능(이벤트 희소성 활용)을 실측으로 입증하는 구조.
- archived_hw에 **여러 아키텍처 변형(A~D)**을 보존 → 설계공간 비교 자료로 유용.

**한계**
- 보드/경로/센서 매핑이 **ZCU102 PYNQ에 완전 종속**(`power_monitor.py:7-28`, `evaluate.py:103`) → 타 보드 이식 시 전면 재작성.
- archived 변형은 **생성물(weight.h/top.cpp)** 형태로만 보존되어, 각 변형의 정확도/지연 수치 자체는 코드에 없음 → 성능 비교는 **확인 불가**(로그/논문 필요).
- `power_monitor.py`의 `interval` 주석("ms")과 실제 단위(0.1s) 불일치 — 측정 해석 시 주의.
- SEE/ESDA의 DSE 본체(`eventnet.py` 등)는 ESDA와 마찬가지로 **부재** → DSE 재현 불가.

---

## 9. 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 시사점

1. **실측 평가 인프라 재사용(최우선)**: `evaluate.py`(register_map 바인딩 + AP_START/AP_IDLE 폴링 + num_nz 기반 지연) + `power_monitor.py`(INA226 18레일 동시 샘플링)는 우리 ViT 가속기의 **ZCU102 보드 실측 하네스로 거의 그대로 차용 가능**. 특히 PS/PL 전력 도메인 분리 측정(`power_monitor.py:9-28`)은 에너지 효율 보고에 직접 활용.
2. **데이터 의존적 지연 측정 패턴**: `num_nz`를 레지스터로 넘겨 희소도별 지연을 측정하는 방식(`evaluate.py:126-135`)은, ViT의 token pruning/early-exit처럼 입력 의존 연산량을 갖는 가속기 평가에 그대로 적용.
3. **설계공간 변형 보존 전략**: archived_hw처럼 동일 커널·다른 토폴로지(SEE-A~D)를 보존해 비교하는 방식은, 우리 ViT의 헤드 수/임베딩 차원/병렬도 변형을 체계적으로 스윕·아카이브하는 데 참고.
4. **데이터셋×자원예산 매트릭스(`cfgs/`)**: "한 템플릿 → cfg 교체로 다중 타깃" 구조는 우리 XR 시선추적의 다양한 입력 해상도/정확도 요구를 단일 가속기 패밀리로 커버하는 설계에 시사점.
5. **희소성↔정확도↔자원 상관**(NMNIST sparsity 0.23/obj 4795 vs DVS 0.05/obj 2948, §3.1): 입력 희소도가 낮을수록 비용이 급증 — 이벤트 기반 XR 시선추적에서 **전처리 단계의 희소도 확보**가 가속 효율의 핵심 레버임을 정량 시사.
6. 보드 종속·생성물-only 한계는 우리가 일반화 시 재작성·재측정 필요.

---
*근거 파일(절대경로)*: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\SEE\ESDA\hardware\board\{evaluate.py,power_monitor.py,hw_e2e.py}`, `.../hardware/cfgs/{DVS_1890_shift16-zcu102_80res.json, NMNIST_shift16-zcu102_60res.json, Roshambo_shift16-zcu102_80res.json}`, `.../eventNet/archived_hw/{SEE-B,SEE-D,SEE-A}/{para.h,top.cpp,conv.h}`, `.../hardware/README.md`. (공유 커널·optimization은 `ESDA.md` 참조.)
