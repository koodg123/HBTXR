# SJTU_microe-main 정밀 분석

> 경로: `REF/CNN-Accel/SJTU_microe-main/`
> 분석 방식: hls/ 좁은 Glob 후 핵심 소스 Read. 라인 근거 표기.

---

## 1. 개요 (목적/대회/타깃보드)

- **정체**: 상하이교통대(SJTU) SEIEE의 **DAC-SDC 2021 3위** 설계 (standalone 배포판). README.md:3-9. (작업지시의 "2021 SJTU_microe 독립 배포판" 가설 **확정**.)
- **성능**: 70.3% IoU, **249.38 fps**, 4.95 W on Ultra96v2. README.md:3.
- **타깃보드**: **Xilinx Ultra96 V2**. README.md:40.
- **모델**: **UltraNet + Bypass** 모듈 (작은 객체 검출 개선, 65.6%→70.3%). README.md:13-17. top 함수 `UltraNet_Bypass`(top.cpp:41). conv0~8 (3x3 ×8 + 1x1 ×1) + **reorg/concat bypass**(top.cpp:407-595).
- **양자화**: **W4A4** (DoReFa 계열). README.md:21 ("quantize weights and activations to 4 bits, map four 4-bit multiplications on a DSP"). pt 파일명 `UltraNet_Bypass_4w4a.pt`(quantization/).
- **핵심 기여**(README.md:15-27): ① Bypass 모듈로 소형객체 정확도↑ ② DSP에 4×4-bit MAC 매핑 ③ PS/PL 부하 균형(이미지 로딩 8스레드, 배치 64, PL 125MHz).

---

## 2. 디렉토리 구조 (자체 + 제외 이유)

```
hls/                ← HLS 가속기 소스(자체 핵심)
  PE_array.h        ← 2D PE 어레이 + DSP 패킹(핵심★)
  conv3x3.h, conv1x1.h ← 컨볼루션 래퍼(핵심)
  shift_reg.h       ← 슬라이딩 윈도우 시프트레지스터(핵심)
  function.h        ← BN_QUReLU + HW resize + padding(핵심)
  maxpool.h, reorg.h ← 풀링/reorg bypass
  stream_tools.h    ← 폭변환/스트림 유틸
  top.cpp           ← top dataflow(핵심)
  config.h, param.h ← 하이퍼파라미터/가중치(param.h 거대→제외)
  testbench.cpp, test_img/
quantization/       ← 양자화·param 생성 스크립트(자체 핵심)
  quant_ultra.py, quantization.py, qnn_param_reader.py,
  ultranet_param_gen.py, qnn_mem_process.py, torch_export.py, mymodel.py
training/           ← YOLOv3-tiny 포크 학습코드 — 대부분 제외
deploy/, script/    ← .bit/.hwh/.ipynb 배포(.bit/.npy 제외)
```
- **제외(이름)**: hls/param.h, quantization/param/hls/param.h, *.pt, *.npz, *.npy, deploy/*.bit/*.hwh/*.so, training/utils/{datasets,general,...}(외부 yolov3 학습 포크), training/models/*.yaml, image/rank.png.
- **소스 미동봉**: 해당 없음. hls 핵심 전부 실재.

---

## 3. 핵심 모듈 정밀 분석

### 3.1 ★ 2D PE 어레이 + DSP 패킹 — PE_array.h

UltraScale+ DSP48E2를 "**1 DSP에 4 MUL**"로 활용(2 conv window × 2 out-channel). README.md:21.

**1D PE (DSP 패킹 곱)** `_1D_PE_array` (PE_array.h:19-42):
- 가중치 2개를 1 피연산자에: `weight_shrink = (w1<<11) + w0` (18-bit, PE_array.h:34).
- 활성 2개(2 윈도우)를 1 피연산자에: `in_act_shrink = (in_act1<<22) + in_act0` (27-bit, PE_array.h:35).
- 단일 곱 `result_shrink = weight_shrink * in_act_shrink` (45-bit, PE_array.h:37) → **w0·a0, w0·a1, w1·a0, w1·a1 4부분곱이 45-bit 안에 비트영역 분리**되어 동시 산출. IN_CH_PARA개 입력채널 누산(PE_array.h:27 UNROLL).

**2D PE 어레이** `_2D_PE_array_act` (PE_array.h:67-179):
- `IN_CH_PARA × (OUT_CH_PARA/2)` DSP로 IN_CH_PARA 입력채널 × OUT_CH_PARA 출력채널 병렬 (PE_array.h:44-48 주석).
- `#pragma HLS PIPELINE II=1`(PE_array.h:100). 입력 재사용 버퍼 `in_buffer0/1`(BRAM, PE_array.h:84-87) — out_ch 첫 iteration에 읽고 이후 **데이터 재사용**(PE_array.h:103-112).
- **45-bit 결과 4분할 추출**: `acc_shrink(43,33)+[32]` → acc1[2i+1], `(32,22)+[21]` → acc1[2i], `(21,11)+[10]` → acc0[2i+1], `(10,0)` → acc0[2i] (PE_array.h:136-143) — 각 비트영역 + 반올림 캐리. acc0=윈도우0, acc1=윈도우1.
- 출력 시 `BN_QUReLU`로 4-bit 양자화 후 out_act0/out_act1(2윈도우) 동시 write (PE_array.h:158-165).

**첫 레이어 전용** `_1D_PE_array_L1`/`_2D_PE_array_act_L1` (PE_array.h:193-346):
- conv0는 입력 8-bit라 가중치 패킹 없이 **2 MUL/DSP**(2 윈도우만): `in_act_shrink=(in_act1<<16)+in_act0` (PE_array.h:206), 16-bit씩 2분할(PE_array.h:306-309).

### 3.2 슬라이딩 윈도우 — conv3x3.h + shift_reg.h
- `conv3x3_bn_act`(conv3x3.h:40-75) `#pragma HLS DATAFLOW`(conv3x3.h:49): padding → 폭변환(IN_CH→IN_CH_PARA) → **`Shift_Register_2O`**(conv3x3.h:70, shift_reg.h) → `_2D_PE_array_act`.
- `Shift_Register_2O<3,1,...>`: K=3,S=1 윈도우를 **2출력(2픽셀) 동시** 생성(conv3x3.h:68-70) → PE 어레이의 2-윈도우 패킹과 정합. MAT_ROW=9×IN_CH (conv3x3.h:73, 즉 3×3×IN_CH를 한 차원으로 펼침 = im2col형 행 전개).
- L1 버전 `conv3x3_bn_act_L1`(conv3x3.h:102-136)은 `_2D_PE_array_act_L1` 호출.

### 3.3 Top dataflow + Bypass — top.cpp
- `UltraNet_Bypass`(top.cpp:41) `#pragma HLS DATAFLOW`(top.cpp:84). 입력 640×360 → **HW resize 320×160**(`resize_batch`, top.cpp:103, function.h:80).
- 레이어: conv0(`conv3x3_bn_act_L1` top.cpp:124) → pool0 → conv1~3 → **bypass 분기**.
- **Bypass(작은 객체용)**: conv3 출력을 `Stream_Broadcast`로 2분기(top.cpp:416) → 한쪽 `ReOrg_2D`(reorg, K=2,S=2, top.cpp:427) → 다른쪽 pool3→conv4~6 → **`StreamConcat`로 reorg+conv6 결합**(top.cpp:594) → conv7 → conv8(1x1, top.cpp:658). README.md:17의 "shallow+deep feature merge".
- **PARA→SERIAL 변환** `Stream_PISO`(top.cpp:171 등): PE 어레이의 2-병렬 출력(out0/out1)을 직렬화. 매 conv 후 반복.
- 최종 `AddLast`(top.cpp:699)로 AXIS 출력.

### 3.4 BN + 양자화 ReLU — function.h:29-54
- `BN_QUReLU`: `bn_res = in*inc + bias`(function.h:38) → 양수면 `(bn_res+D/2) >> (W_BIT-1+DATA_BIT+L_SHIFT)`(function.h:43) → [0,15] 클램프 4-bit. `D=1<<(W_BIT-1+DATA_BIT+L_SHIFT)`(function.h:33). 2022 champion `bn_qurelu`와 동일 수식 계보(DoReFa).

### 3.5 양자화 파이프라인 — quantization/quant_ultra.py
- **DoReFa 균일 양자화** `uniform_quantize(k)`(quant_ultra.py:7-26): `round(input*n)/n`, n=2^k-1.
- **가중치 양자화** `weight_quantize_fn`(quant_ultra.py:29-51): `tanh(x)` 정규화 후 max로 나눠 [-1,1] → k-1 bit(부호1) 양자화(quant_ultra.py:45-50).
- **활성 양자화** `activation_quantize_fn`(quant_ultra.py:54-67): `clamp(x,0,1)` 후 k-bit(quant_ultra.py:65) → unsigned [0,1] 균일.
- **BN 양자화 fold** `batchNorm2d_Q_fn`(quant_ultra.py:87-116): BN을 `w=gamma/sqrt(var), b=bias-mean·w`로 접고(quant_ultra.py:103-104) w,b를 [-1,1] 양자화(quant_ultra.py:106-111) → HLS의 inc/bias 정수상수로 export. `torch_export.py`/`ultranet_param_gen.py`가 param.h 생성(README.md:64-68).

---

## 4. 데이터플로우
```
AXIS →resize(640×360→320×160) →conv0(L1,2MUL/DSP) →pool0
  →conv1~3 →[broadcast]→ reorg(K2S2) ─┐
                       └→pool3→conv4~6 ┤→concat→conv7→conv8(1x1)→AddLast
```
- 매 conv: padding→widthadjust→shift_reg(2픽셀윈도우)→2D PE array→PISO 직렬화.

---

## 5. HW/SW 매핑
- HW(PL): UltraNet_Bypass 단일 IP, 125MHz(README.md:27). resize/reorg/concat까지 PL 내재화.
- SW(PS): 8-스레드 이미지 로딩, 배치 64(README.md:27). deploy/SJTU_microe.ipynb로 PYNQ 구동. 학습은 YOLOv3-tiny 포크(training/).

---

## 6. 빌드 · 실행
- 학습: `python3 train.py --img-size 320 ...`(README.md:59) → `torch_export.py`+`ultranet_param_gen.py`(README.md:64-68).
- HLS→Vivado→PYNQ 배포(README.md:70-96).

---

## 7. 의존성
- Vivado HLS(`ap_int.h`,`hls_stream.h`,`hls_video.h` for resize). PyTorch(학습/양자화). 학습은 ultralytics yolov3 포크 utils.

---

## 8. 강점 · 한계
**강점**
- **명시적 2D 시스토릭형 PE 어레이**(PE_array.h:67) — IN_CH_PARA×OUT_CH_PARA 그리드 + 입력 재사용 버퍼(PE_array.h:84). 본 5개 repo 중 가장 "어레이"에 가까운 구조.
- 4MUL/DSP 패킹(PE_array.h:34-37) + 2-윈도우 동시.
- Bypass+reorg+concat까지 완전 PL dataflow(top.cpp:407-595) → 249fps.
- im2col형 행 전개(MAT_ROW=9·IN_CH, conv3x3.h:73)로 GEMM화.

**한계**
- 패킹 비트영역 하드코딩(11/22 시프트, PE_array.h:34-35) → W4A4 전용.
- config/레이어 하드코딩, 모델 변경 시 재합성.
- PL 125MHz로 PS 보조에 맞춤(고클럭 미사용).

---

## 9. 우리 프로젝트 시사점 (ViT/Transformer + XR)
- **2D PE 어레이 + 입력 재사용 버퍼**(PE_array.h:67-112)는 ViT의 GEMM(Q·K^T, attn·V, FFN)에 **가장 직접적으로 매핑** 가능한 구조 — IN_CH_PARA/OUT_CH_PARA를 토큰차원/특징차원 타일로 재해석. HG-PIPE의 systolic/output-stationary 설계와 정합.
- **im2col형 행 전개(MAT_ROW)**(conv3x3.h:73)는 ViT에서 이미 GEMM이라 윈도우 없이 곧장 PE 어레이 행에 매핑 가능 → CNN보다 ViT가 이 PE 어레이에 더 잘 맞음.
- **4MUL/DSP 패킹**(PE_array.h:34-37): 2023의 UINT-Packing과 함께 ViT 저비트 GEMM DSP 밀도 향상 후보.
- **HW resize 전처리**(function.h:64): XR 카메라 원본→ViT 패치 해상도 다운스케일을 PL에서 처리.
- **DoReFa 양자화 + BN fold export**(quant_ultra.py:87-116): ViT 양자화 학습→HLS 상수 export 파이프라인 템플릿으로 재사용(LayerNorm fold는 별도 처리 필요, 추정).
