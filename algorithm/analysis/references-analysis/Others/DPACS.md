# DPACS 코드베이스 정밀 분석

> 분석 대상 경로: `REF/Others/DPACS`
> 분석 방식: 실제 소스 Read 기반(라인 근거 제시). bash 미사용(UNC), Glob/Grep/Read만 사용.
> 대표 변형 정밀 분석: `hardware/source/resnet_bottleneck/sparse/resnet_bottleneck_sparse_parallel` (나머지 변형은 차이점만 요약).

---

## 1. 개요

- **목적/한줄요약**: DPACS는 "동적 신경망 프루닝(spatial + channel)"을 **알고리즘-아키텍처 공동설계(co-design)**로 구현한 FPGA 가속기. 입력별로 공간(spatial) 마스크와 채널(channel) 마스크를 동적으로 예측하고, 하드웨어의 **elastic sparse dataflow 엔진**이 프루닝된 연산만 스트리밍 처리해 실제 지연시간을 단축한다.
- **원논문**: Yizhao Gao, Baoheng Zhang, Xiaojuan Qi, Hayden So, *"DPACS: Hardware Accelerated Dynamic Neural Network Pruning through Algorithm-Architecture Co-design"*, ASPLOS 2023 (DOI 10.1145/3575693.3575728). 근거: 루트 `README.md` 5-7행, 34-46행.
- **타깃 디바이스**: ZCU102 개발보드, Zynq UltraScale+ MPSoC **XCZU3EG**. 개발도구 Vitis HLS + Vivado Design Suite **2020.2**, PYNQ 2.6. 근거: `hardware/README.md` 3행, 8행, 15행.
- **대상 네트워크**: ResNet basicblock(ResNet-18/34), ResNet bottleneck(ResNet-50/101), MobileNet, CIFAR ResNet-32. 근거: `hardware/README.md` 25-26행, `software/README.md` 124-269행.
- **핵심 성과(논문 자체보고)**: ResNet-50 parallel baseline 32.84ms → s75-c75에서 19.90ms 등. 근거: `hardware/README.md` 30-49행.

DPACS는 **HW/SW 완전 분리** 구조다. `software/`는 PyTorch 기반 동적 프루닝 알고리즘(마스크 예측기 학습, budget loss), `hardware/`는 학습된 마스크를 받아 sparse 연산을 수행하는 Vitis HLS 스트림 데이터플로우 커널이다.

---

## 2. 디렉토리 구조 (자체 소스 + 제외 목록)

### 자체 핵심 소스 트리
```
DPACS/
├── README.md                       # 논문/구조 개요
├── software/                       # PyTorch 동적 프루닝 알고리즘
│   ├── main_imagenet.py            # ImageNet 학습/평가 엔트리
│   ├── main_cifar.py               # CIFAR 학습/평가 엔트리
│   ├── main.py / train_cmd.py
│   ├── config/                     # args_imagenet.py, args_cifar.py
│   ├── dataloader/                 # imagenet.py, imagenet_cache.py
│   ├── utils/
│   │   ├── loss.py                 # ★ budget loss(spatial/channel/network)
│   │   ├── flopscounter.py         # 동적 FLOPs 측정
│   │   ├── scheduler.py / optimizer.py / logger.py / utils.py / viz.py
│   └── models/                     # 모델/마스크 유닛 정의(아래 "확인 불가" 참조)
└── hardware/                       # Vitis HLS 가속기 (DPUnit)
    ├── README.md
    ├── drive/                      # 비트스트림/노트북(온보드 테스트) — 제외(산출물)
    └── source/
        ├── resnet_bottleneck/{dense,sparse}/{serial,parallel}/
        └── resnet_basicblock/{dense,sparse}/{serial,parallel}/
            # 각 변형 공통: top.cpp/top.h, conv.h, conv_pack.h, mem.h, para.h, tb.cpp, tb_func.h
            # sparse 변형 추가: mask.h, linebuffer.h, c_prune.h
```

각 HW 변형 디렉토리는 동일 8개 기본 파일 + sparse 한정 3개 파일을 가진다. 총 8개 변형(bottleneck/basicblock × dense/sparse × serial/parallel).

### 제외 목록(이름만 언급, 분석 제외)
- `hardware/drive/` : 비트스트림(.bit), 하드웨어 description(.hwh), Jupyter 노트북 등 **생성물/바이너리**.
- `software/DPACS_checkpoint/` : 학습된 모델 가중치(외부 OneDrive 배포). 근거: `software/README.md` 41-62행.
- `figures/` : DPACS.jpg, sparse_dataflow.gif 등 그림.
- 로그/빌드 산출물(`./log` 등).

### 확인 불가
- `software/models/` 패키지: `main_imagenet.py` 6행 `import models`, 35행 `models.__dict__[args.model]`로 **임포트는 확정**되나, Glob `software/models/*`가 매칭 0건(WSL UNC 경로 필터/심볼릭 이슈 추정)이어서 마스크 유닛(Gumbel-softmax 예측기)·ResNet 변형 모델 정의의 라인별 분석은 **확인 불가**. 다만 호출부(`main_imagenet.py`)와 `utils/loss.py`로 동작은 역추적 가능(아래 3.5).

---

## 3. 핵심 모듈·파일별 정밀 분석

> **대표**: `resnet_bottleneck/sparse/resnet_bottleneck_sparse_parallel`. 본 변형 기준으로 라인 근거를 제시하고, 다른 변형은 5절·3.8에서 차이만 요약.

### 3.1 데이터타입/파라미터 (`para.h`)

- 비트폭 정의: `FW 8`(입력 feature), `WW 8`(weight), `PW 24`(partial sum), `SW 16`(quantization shift), `BW 16`(bias), `CW 12`(channel), `HW 10`(image height). 근거: `para.h` 2-9행. → **INT8 입력·INT8 weight, 24-bit psum 누산** 구조.
- 타입 별칭: `T_F=ap_int<8>`, `T_W=ap_int<8>`, `T_P=ap_int<24>`, `T_C=ap_uint<12>`, `T_H=ap_uint<10>`. 근거: 13-19행.
- **키(key) 구조체** `T_K{ap_uint<1> end; ap_uint<10> x; ap_uint<10> y;}`: sparse dataflow에서 **비-제로 픽셀의 좌표(x,y)**를 흐르게 하는 토큰. `end=1`이면 스트림 종료. 근거: 28-32행. → 이것이 DPACS sparse 엔진의 핵심: feature 대신 "유효 좌표 + 해당 좌표 채널 벡터"만 스트림.
- **번들 타입** `BundleT<N,T>{ T data[N]; }`: 병렬 채널 N개를 한 워드로 묶음(SIMD lane). 근거: 34-37행.
- 병렬화 인자: `PI_0=64, PO_0=16, PI_3=16, PO_3=16, PI_1=16, PO_1=64`. 근거: 42-47행. → 1x1(첫째) 입력 64-lane, 3x3은 16-lane, 마지막 1x1 출력 64-lane. (serial 변형은 이 값들이 작음 — 3.8 참조)
- 채널 프루닝 그래뉼: `CPRUNE_FACTOR=64`, `W_FACTOR=64`. 근거: 50-51행. → 채널은 64개 그룹 단위로 keep/skip.
- 마스크 워드폭 `MW = PI_0*WW = 512`. 근거: 53행. → spatial mask는 512비트 워드 단위로 패킹.
- 최대치: `MAX_IC=2048, MAX_OC=2048, MAX_C=256, MAX_H=256`. 근거: 56-60행.

### 3.2 Top 커널 — DATAFLOW 파이프라인 (`top.cpp`)

`top()`은 AXI 인터페이스 래퍼이고 실제 계산은 `wrapper()`. `wrapper()`는 `#pragma HLS DATAFLOW`(21행)로 전 스테이지가 **stream으로 연결된 태스크 파이프라인**.

- **AXI 인터페이스**(top, 333-358행): `m_axi` 6포트 — fin/fout(gmem0), fres/smask_out(gmem1), weight/cmask_out(gmem2). `s_axilite control` 번들로 스칼라 제어. → 입력/출력/잔차/가중치/공간마스크/채널마스크가 각기 다른 HBM-like 뱅크로 분산.
- **제어 flags 16비트 디코딩**(79-91행): `skip_0/skip_3/skip_1`(각 conv bypass), `enable_pool`(다음 블록 채널마스크 예측), `stride2`, `residual`, `use_mask`(사전계산 spatial mask 사용 vs on-the-fly 생성), `use_cprune`(채널 프루닝 on/off), `first_layer`, `return_mask`, `relu_0/3/1`. → **한 비트스트림으로 전체 bottleneck/basicblock의 모든 레이어 변형을 런타임 제어**(매우 유연한 설계).
- `first_layer` 특수처리(93-102행): skip_0=1, skip_3=0, skip_1=1, OC_3=OC_1=64, stride2=1 강제 → 첫 7x7 conv를 3x3 경로로 우회 매핑.
- **채널 프루닝 시 OC 재계산**(127-140행): `cmask_ic`/`cmask_oc`(각 8비트)의 set-bit 수 × CPRUNE_FACTOR(64)로 실제 OC_0, OC_3 산출. → 동적으로 conv 폭이 줄어듦.

**데이터플로우 스테이지 순서**(wrapper 본문):
1. `Load_Weight_Wrap`(105행): 외부 weight를 stream으로 적재(프루닝 시 keep 채널만).
2. `route_weight`(144행): 통합 weight stream을 mask용/conv0/conv3/conv1/fc(채널예측)/spatial-mask로 **분배**.
3. `input_unit`(167행): spatial mask 생성 또는 사전 마스크로 sparse 입력 적재 + key 스트림 생성.
4. `conv1x1_dsp_wrap`(180행): 첫 1x1 conv (IC_0→OC_0).
5. `adjust_stream_same`(195행): 스트림 lane 폭 조정(PO_0→PI_3).
6. `conv3x3_dsp_wrap`(204행): 3x3 conv (+ elastic linebuffer + 잔차 읽기).
7. `conv1x1_dsp_residual`(242행): 마지막 1x1 conv + identity add.
8. `S2M_key`(259행): 출력 feature를 메모리로 쓰고 spatial mask 출력.
9. `max_pool`(275행): 다음 블록용 채널마스크 예측(global pooling → fc).

→ **9-스테이지 dataflow 파이프라인**이 한 residual block을 끝까지 처리. 모든 스트림 깊이는 2~4(34-77행 `#pragma HLS STREAM depth`)로 작게 잡아 II=1 파이프라인 유지.

### 3.3 Spatial Mask & sparse 입력 (`mask.h`)

DPACS sparse 엔진의 입출력 표현 변환을 전담한다.

- **`conv_1x1_mask_wrap`**(178-239행): **on-the-fly spatial mask 생성기**. 입력 feature에 1x1 conv(mask weight)를 적용해 픽셀별 스칼라 `sum`을 구하고, `sum>0`이면 그 좌표(`key`)를 keep으로 판정(223행 `out_flag = (sum>0)?1:0`). keep 픽셀만 `s_key.write(key)`(224행) + 채널 벡터 출력(225-233행). → **공간 프루닝을 "1x1 conv + sign" 으로 HW에서 직접 계산**. 마지막에 `key.end=1` 토큰(236-237행).
- **`M2S_mask`**(60-108행): 외부 비트마스크(MW=512비트 워드)를 풀어 keep 좌표 key 스트림으로 변환. nz_flag(89행)가 1인 좌표만 출력.
- **`M2S_from_key`**(3-35행): key 스트림이 가리키는 좌표의 채널 벡터만 메모리에서 gather(`index=(x+y*Width)*ICPI`, 23행). → **랜덤 액세스 기반 sparse gather**.
- **`M2S_mask_merge`**(113-174행): 마스크 디코딩 + gather를 한 함수에 융합(INLINE).
- **`input_unit`**(341-364행): dispatcher — first_layer면 `first_layer_unit`, use_mask면 `load_mask_wrap`(사전 마스크 gather), 아니면 `conv_1x1_mask_wrap`(생성). → top.cpp 167행에서 호출.
- **`first_layer_unit`/`first_layer_key`**(294-336행): 첫 레이어는 마스크 없이 전 픽셀 key 생성(밀집).
- **`M2S_reduce`**(269-291행): 첫 레이어 3채널 입력을 P4=4 lane으로 정렬.

### 3.4 Conv 계산 커널 — DSP 패킹 MAC (`conv.h`) ★핵심

DPACS의 연산 효율 비결은 **1개 DSP48에 2개 INT8 MAC을 패킹**하는 기법이다.

- **`DSP_AM`**(1-8행): `(in1+in2)*in3` 형태의 pre-adder+multiplier 매핑. `#pragma HLS INLINE OFF`로 DSP 1개에 고정.
- **`conv_3x3_double`**(12-132행) / **`conv1x1_dsp_double`**(139-252행): 2-MAC 패킹의 본체.
  - weight 2개(`w_0`,`w_1`)를 27비트 워드에 시프트 배치: `w_1_shift.range(26,18)=w_1; w_0_expend=w_0`(108-110행 / 233-235행). 활성 `in`은 18비트 확장(`in_expend`, 103행).
  - 단일 곱 `mul_temp = DSP_AM(w_1_shift, w_0_expend, in_expend)`(111행 / 237행): `(w_1<<18 + w_0) * in` = `(w_1*in)<<18 + (w_0*in)`. 상위 18비트가 `w_1*in`, 하위가 `w_0*in`.
  - 분리: `low = mul_temp[15:0]`, `high = mul_temp[33:18] + 부호보정`(112-113행 / 238-239행). → **한 DSP로 두 출력채널(po*2, po*2+1)의 부분합 동시 누산**.
  - psum 버퍼는 `ARRAY_PARTITION cyclic factor=PO`(37행)로 병렬 누산. weight 버퍼 `w_buffer[9][M_OC][M_IC/PI]`(39행): 3x3 9탭.
  - 채널 프루닝 시 `pi_factor/po_factor/ic_ceil/oc_ceil`을 CPRUNE_FACTOR 기준으로 재계산(46-58행)해 keep 채널만 적재.
- **`conv1x1_dsp_single`**(255-333행): 패킹 없는 기본 1x1(비교/대체용, `sum += in*w_0` 322행).
- **`quantize_shift`**(340-382행): 24비트 psum을 **`psum >> 10`** 후 [-128,127] 클램프로 INT8 재양자화(373-377행). 주석(336-339행): "현재는 shift만 사용, ad-hoc하므로 자기 양자화 알고리즘에 맞게 수정 가능".
- **`quantize_shift_res`**(387-436행): 위 + identity(잔차) add(`psum += res_read.data[po]`, 426행) → residual block의 skip-connection을 양자화 단계에서 융합.

→ **요약**: INT8×INT8 곱 2개를 DSP48의 27-bit×18-bit 곱으로 패킹(Xilinx의 "INT8 DSP packing" 기법). 이것이 parallel 변형 throughput의 근간.

### 3.5 Elastic Linebuffer — sparse 3x3 conv (`linebuffer.h`) ★핵심

밀집 linebuffer는 모든 픽셀을 순차 처리하지만, sparse에서는 **존재하지 않는(프루닝된) 픽셀**을 다뤄야 한다. DPACS는 "key FIFO + valid bitmap" 기반 **탄력적 linebuffer**를 도입.

- **`conv_3x3_line_buffer_residual_stride`**(94-282행):
  - 3-row 라인버퍼(`line_buff[3][BUFFER_WIDTH]`, 112행) + `valid[3][MAX_H]` 비트맵(115행)으로 각 좌표의 유효성 추적.
  - 입력 key FIFO(`key_fifo[FIFO_DEPTH]`, 125행)에 비-제로 좌표를 적재. `read_enable`/`out_enable` 핸드셰이크(196-197행)로 입력 픽셀이 충분히 모이면(window가 채워질 조건) 9-탭 window를 출력.
  - **stride2 처리**: anchor 좌표(`x_anchor,y_anchor`)를 2씩 증가(275-279행), 출력 key는 `>>1`(236-237행).
  - **window valid 판정**(214-228행): 패딩 영역(`padding`) 또는 valid 비트=0(`invalid`)이면 해당 탭을 0으로(`win_valid[ki][kj]=0`). → **프루닝되어 비어있는 이웃 픽셀은 0으로 채워 conv 정확성 보장**.
  - 출력 enable은 window에 유효 탭이 하나라도 있을 때(`out_empty_check`, 226행/230행).
- **`line_buffer_first_layer`**(5-89행): 첫 레이어 전용 소형 linebuffer(stride2, P4=4 lane), 밀집 처리.

→ **이것이 "elastic sparse dataflow"의 핵심**: 입력/출력 non-zero 인덱스를 동일하게 유지하면서, 누락 픽셀을 valid bitmap으로 추적해 3x3 conv를 sparse하게 수행.

### 3.6 Channel Pruning — global pool + fc 예측 (`c_prune.h`, `mem.h`)

- **`max_pool`**(`conv_pack.h` 내 정의는 c_prune.h, 2-79행): `enable_pool`이면 ① fc weight 적재(29-35행) ② 전 좌표에 대한 채널별 max-pooling(45-63행, `pool_buff`에 max 유지) ③ pooled 벡터 × fc weight로 채널 점수 산출(65-77행) → `cmask_out[oc]=psum`(76행). → **다음 블록의 채널 마스크를 현재 블록에서 미리 예측**(channel mask prediction).
- **`Load_Weight_C_PRUNE`**(`mem.h` 472-566행): `cmask_ic`/`cmask_oc` 비트에 따라 **keep 그룹의 weight만 메모리에서 읽음**(523행 conv0, 538행 conv3 — ic_flag&&oc_flag 동시 만족, 551행 conv1). `read_weight_cprune`(456-469행)가 CPRUNE_FACTOR=64개씩 burst. → **프루닝된 채널의 weight는 아예 로드하지 않아 메모리 대역폭 절감**.
- **`Load_Weight_Merge`**(`mem.h` 395-454행): 프루닝 없을 때 전체 weight stream.
- **`route_weight`**(`mem.h` 634-743행): 통합 weight stream을 w_mask/fc/w0/w3/w1/mask로 폭 변환하며 분배.

### 3.7 메모리 유틸 & 스트림 어댑터 (`mem.h`)

- `ceil_div`(1-10행), `S2M_F`/`S2M_key`(출력 feature를 메모리로 쓰며 출력 spatial mask 비트 누적, 113-204행), `Residual_read`(잔차 gather, 207-239행), `adjust_stream_{larger,smaller,same,first_layer}`(244-392행): 인접 스테이지 간 lane 폭(PI↔PO) 불일치를 흡수하는 어댑터. → dataflow 스테이지를 자유롭게 조립하기 위한 glue.
- `S2M_key`(113-204행): `enable_pool`이면 출력을 pool 스테이지로도 복제(166행, 183행), `return_mask`면 출력 비트마스크 누적(172-201행) → 다음 레이어가 use_mask로 재사용.

### 3.8 소프트웨어 — 동적 프루닝 알고리즘 (`software/`)

- **`main_imagenet.py`**: 모델은 `models.__dict__[args.model]`로 생성(35행), forward에 `meta` dict 전달(`masks`, `gumbel_temp`, `gumbel_noise`, `channel_prediction` 등, 44-45행). → 마스크가 **Gumbel-softmax**로 미분가능하게 학습됨을 시사(122-123행 `set_gumbel`). 평가 시 `flopscounter`로 동적 FLOPs 측정(162-163행).
- **`utils/loss.py`** ★ budget loss 설계:
  - `SpatialLoss`(126-196행): 블록별 FLOPs 비율 `layer_perc`이 budget을 넘으면 상한 페널티 `relu(layer_perc-weight)^2`(191행), 미달이면 하한 페널티(193행). → **공간 sparsity를 목표 budget으로 유도**.
  - `ChannelLoss`(226-252행): 채널 keep 비율을 budget으로 유도(246행), budget이 0/1 밖이면 lasso(251행).
  - `FLOPsReductionLoss`(199-223행): 전체 네트워크 FLOPs 감소율을 budget으로 유도(220-222행).
  - `Loss`(292-336행): task(CE) + n_weight·network + s_weight·spatial + c_weight·channel(332행). "balance" 모드는 spatial/channel budget을 `sqrt(network_budget)`로 설정(310-311행).
  - `SampleAdaptor`(8-76행)/`AdaptiveLoss`(255-289행): 샘플별 난이도에 따라 budget 가중치를 적응(over/under-computing). → **easy 샘플은 더 프루닝, hard 샘플은 덜 프루닝**(입력 적응형).
- `s25-c25` 등 표기는 spatial budget 25% - channel budget 25%를 의미(README 표).

---

## 4. 데이터플로우 / 실행 흐름

```
[DRAM] fin/weight/fres
   │  (m_axi gmem0/1/2)
   ▼
Load_Weight_Wrap ─► route_weight ─► (w_mask, w0, w3, w1, fc, spatial-mask streams)
   │
input_unit ─(use_mask?)─► gather sparse 입력 + key 생성  /  conv_1x1_mask_wrap로 마스크 생성
   │  (key 스트림 = 비-제로 좌표 토큰)
   ▼
conv1x1(IC0→OC0) ─► adjust ─► conv3x3(elastic linebuffer, stride/residual) ─► conv1x1(OC3→OC1)+identity add
   │  (DSP 2-MAC 패킹, 24b psum, >>10 재양자화 INT8)
   ▼
S2M_key (출력 feature → DRAM, spatial mask 출력) ─► max_pool (global pool + fc) ─► cmask_out (다음 블록 채널마스크)
```

- **메모리 계층**: 외부 DRAM(fin/fout/fres/weight) ↔ on-chip stream FIFO(depth 2-4) ↔ 라인버퍼(BRAM). 채널마스크/공간마스크는 별도 작은 버퍼.
- **병렬화**: 채널 SIMD(`BundleT<PI/PO>`), DSP 2-MAC 패킹, 9-스테이지 dataflow 태스크 병렬.
- **데이터타입/양자화**: INT8 feature/weight, 24b psum, `>>10` shift 양자화(고정 시프트, 알고리즘 의존적, 사용자 수정 권장).
- **sparse 표현**: dense feature map 대신 **(좌표 key + 채널 벡터)** 스트림 — Coordinate-list 형태의 SpMV/sparse conv.

---

## 5. HW/SW 매핑

| 소프트웨어(학습) | 하드웨어(추론) | 근거 |
|---|---|---|
| spatial mask 예측기(Gumbel) | `conv_1x1_mask_wrap`(1x1+sign) 또는 사전 마스크 gather | mask.h 178-239 / 341-364 |
| channel mask 예측기 + budget | `max_pool`(global pool+fc) → cmask_out, `Load_Weight_C_PRUNE` | c_prune.h 2-79 / mem.h 472-566 |
| budget(s/c) 목표치 | top flags의 use_mask/use_cprune + cmask_ic/oc | top.cpp 79-140 |
| residual block 구조 | 9-스테이지 dataflow(`wrapper`) | top.cpp 3-286 |
| INT8 양자화 | FW/WW=8, `quantize_shift` >>10 | para.h 2 / conv.h 340-382 |

- 학습은 마스크가 미분가능(soft), 추론(HW)은 hard 마스크(비트). HW는 학습된 마스크 가중치를 그대로 받아 on-the-fly 생성하거나 사전계산 비트마스크를 재사용.

## 6. 빌드·실행

- **HW**(각 변형 디렉토리 Makefile, sparse_parallel README 13-31행):
  - `make tb_gen`(PyTorch로 테스트벤치 데이터 생성) → `make csim` → `make hls`(HLS 합성 + C-RTL co-sim) → `make bitsteam` → `make unpack` → `make all`.
  - 사전: `source /path/to/xilinx/2020.2/settings64.sh`. 출력 비트스트림은 drive 폴더로 복사, log는 ./log.
- **SW**(software/README): `conda create -n DPACS python=3.6` + `torch==1.6.0 torchvision==0.7.0`(13-21행). 평가 `python main_imagenet.py --model resnet50 --resolution_mask ... --budget 0.75 --channel_budget 0.75 -e`(73행). 일괄 `sh scripts/evaluate_imagenet.sh`(78행), 아티팩트 `sh evaluate_asplos.sh`(7행).
- 온보드 테스트: `hardware/drive`의 Jupyter 노트북(PYNQ 2.6).

## 7. 의존성

- HW: Ubuntu 18.04, Vivado/Vitis HLS 2020.2, PYNQ 2.6, Python3+numpy+PyTorch(tb 생성용). HLS 헤더 `hls_stream.h`, `ap_int.h`(top.h 4-5행).
- SW: Python 3.6, PyTorch 1.6/torchvision 0.7, tensorboardX(loss.py 2행), apex(선택, mix precision — 실제로는 비활성, main_imagenet 17-21행), tqdm.

## 8. 강점 / 한계 / 리스크

**강점**
- **실측 지연 단축형 동적 프루닝**: 대부분의 동적 프루닝은 FLOPs만 줄고 HW에서 가속 안되지만, DPACS는 elastic linebuffer + coordinate-stream으로 **실제 latency 단축**(README 성능표).
- **런타임 재구성성**: 16-bit flags 한 세트로 한 비트스트림이 모든 레이어/skip/stride/residual/relu/프루닝 변형을 처리.
- **DSP 2-MAC 패킹**으로 INT8 throughput 2배.
- HW/SW 분리 + 양자화 함수 교체 용이(conv.h 주석 명시).

**한계/리스크**
- 양자화가 **고정 `>>10` shift**(per-tensor 아님, ad-hoc) — 정확도-민감 모델엔 부적합, 사용자 수정 필요.
- `MAX_H=256`, 마스크 버퍼 64×64(`S2M_key` 134행) 등 **고정 크기 가정** — 큰 해상도/긴 시퀀스엔 재파라미터화 필요.
- elastic linebuffer 제어가 복잡(읽기/출력 핸드셰이크, 184-280행) — 검증·디버깅 난이도 높음.
- `software/models/` 미확인으로 마스크 예측기 정확한 구조는 호출부 역추적에 의존(아래 10절).
- ResNet(CNN) 전용 — attention/transformer 직접 지원 없음.

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA + XR 시선추적) 관점 시사점

- **동적 토큰 프루닝 ↔ spatial mask**: DPACS의 "1x1+sign으로 keep 좌표 생성 → coordinate stream"은 **ViT token pruning/early-exit**에 직접 대응. 시선추적은 ROI(눈 주변)가 명확하므로, "유효 토큰만 스트림" 구조가 latency를 크게 줄일 수 있음(재사용 1순위).
- **elastic linebuffer ↔ sparse attention/conv**: 누락 토큰을 valid bitmap으로 추적하는 패턴은 sparse attention의 가변 길이 처리에 응용 가능. HG-PIPE류 파이프라인에 sparse 게이트로 삽입 검토.
- **DSP 2-MAC 패킹**(conv.h 95-117행): GEMM/systolic의 INT8 MAC 효율 2배 기법 — 우리 GEMM PE에 그대로 이식 가능(가장 즉시 재사용 가능).
- **channel mask 예측(global pool+fc) ↔ 동적 채널/head 프루닝**: MHA의 head pruning, FFN의 동적 채널 선택에 응용. "현재 레이어에서 다음 레이어 마스크 예측"으로 파이프라인 stall 방지.
- **budget loss(spatial/channel/network) ↔ 지연 예산 학습**: 시선추적의 실시간 예산(예: <5ms)을 학습 목표로 두는 데 직접 차용 가능.
- **입력 적응형(SampleAdaptor)**: easy/hard 프레임에 따른 동적 연산량 조절 → XR의 프레임별 가변 정확도/지연 트레이드오프.
- **dataflow 조립 + lane 어댑터 패턴**: 우리 가속기의 모듈 조립(QKV→attn→FFN) 시 stream lane 폭 어댑터 설계에 참고.

## 10. 근거 표기

- **확인**: 본문에 인용한 라인 번호는 실제 Read한 파일 기준(top.cpp/top.h, para.h, mask.h, conv.h, conv_pack.h, c_prune.h, linebuffer.h, mem.h, software/main_imagenet.py, software/utils/loss.py, 각 README).
- **추정**: (a) 마스크가 Gumbel-softmax로 학습됨 — main_imagenet.py의 `gumbel_temp/gumbel_noise/set_gumbel`(44행, 122행)에서 강하게 추정되나 정의부 미확인. (b) parallel vs serial PE 수 차이의 정확한 값은 serial `para.h` 미정독으로 일반적 추정.
- **확인 불가**: `software/models/` 패키지 내부(마스크 유닛 클래스, ResNet/MobileNet 변형 정의) — Glob 매칭 0건으로 라인 분석 불가. 동작은 호출부+loss로 역추적.
- **대표 변형 외 미정독**: dense 변형(top.cpp 등), basicblock 전 파일, serial 변형 전 파일 — 구조는 sparse_parallel과 동형이며 차이는 (1) sparse 한정 파일(mask/linebuffer/c_prune) 유무, (2) para.h의 PE 병렬도, (3) dense는 key 스트림 대신 dense linebuffer로 추정.
