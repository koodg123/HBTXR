# SkrSkr-master 정밀 분석

> 경로: `REF/CNN-Accel/SkrSkr-master/`
> 분석 방식: Develop/C/src 좁은 Glob 후 핵심 소스 Read. 라인 근거 표기.
> 주의: 작업지시의 "im2col 가속기" 가설은 부정확 — 실제는 **SkyNet(Depthwise+Pointwise) tiled 가속기**(아래 3절 근거). README.md:1-11.

---

## 1. 개요 (목적/대회/타깃보드)

- **정체**: 상하이과기대(ShanghaiTech) RICL의 **DAC-SDC 2020 준우승** 설계 SkrSkr (standalone). README.md:1-5.
- **베이스**: **SkyNet**(DAC-SDC 2019 우승작) 재구현/최적화. README.md:11.
- **성능**: 73.13% IoU, 52.5fps(Ultra96v2)/57fps(Ultra96v1), 333MHz. README.md:11, README.md:21.
- **타깃보드**: **Xilinx Ultra96** (v1/v2). README.md:31.
- **모델**: SkyNet = **MobileNet형 Depthwise-Separable Conv** 백본 (DWConv3x3 + PWConv1x1 반복) + reorg bypass + bbox 검출. config(SkyNet.cpp:3-23) 19-레이어.
- **양자화**: **W6A8** (weight int6, activation uint8). README.md:16. SkyNet.h:25-29(`ADT=ap_uint<8>`, `WDT=ap_int<6>`). One-shot fully-integer 양자화(파인튜닝 불필요, 캘리브레이션 8장). README.md:12-16.
- **핵심 기여**(README.md:12-29): ① One-shot fully-integer 양자화(추론 전구간 정수, bbox만 CPU) ② 병렬도 2배(PWCONV 512, DWCONV 57.6) ③ 모든 II=1 ④ RGB→RGBA 32-bit 로딩 ⑤ 333MHz.

---

## 2. 디렉토리 구조 (자체 + 제외 이유)

```
Develop/C/src/
  SkyNet.cpp     ← HLS 커널 전체(DW/PW/REORG/POOL/ACT/bbox)(핵심★)
  SkyNet.h       ← 타입(W6A8)·레이어config·메모리오프셋(핵심)
  main.cpp       ← C-sim 드라이버
  transform.cpp  ← stitch/distitch/색공간 변환(SW 전처리)
  utils.cpp      ← load/check/show 유틸
Develop/C/blob/  ← 골든 중간결과(conv0.bb 등) — 제외(바이너리)
Develop/C/weight/ ← SkyNet.bin/.wt/.bm — 제외(가중치)
Develop/{hls.tcl, rtl.tcl}
Deploy/          ← run.py/run_multiprocess.py + SkyNet.bit/.bin/.hwh(.bit/.bin 제외)
sample1000/      ← 테스트 이미지 1000장 — 제외(.jpg)
README.md, RICL.png(제외), LICENSE
```
- **제외(이름)**: *.bit/*.bin/*.bb/*.wt/*.bm, sample1000/*.jpg, RICL.png.
- **소스 미동봉**: 해당 없음. HLS 커널(SkyNet.cpp) 완전 실재.

---

## 3. 핵심 모듈 정밀 분석 — SkyNet.cpp

설계 철학: **채널 32 고정 타일** + 모든 함수 `#pragma HLS ARRAY_PARTITION dim=1 complete`로 32채널 완전 병렬. 외부 DRAM(`fm`)에 타일 단위로 중간 feature를 주고받음(im2col 아님 — 직접 타일 컨볼루션).

### 3.1 Depthwise Conv 3x3 — DWCONV3X3 (SkyNet.cpp:105-155)
- **라인버퍼 + 윈도우버퍼 슬라이딩 윈도우**: `line_buffer[32][3][43]`(SkyNet.cpp:113), `window_buffer[32][3][3]`(SkyNet.cpp:111), 둘 다 완전분할.
- 입력 한 픽셀 읽어 line_buffer 갱신→window 우측열 채움(SkyNet.cpp:127-131), 윈도우 좌측 시프트(SkyNet.cpp:149-150).
- **32채널 UNROLL**(SkyNet.cpp:125) — 각 채널 독립 3x3(depthwise). `#pragma HLS PIPELINE II=5`(SkyNet.cpp:122).
- 곱셈은 `MAC9`(SkyNet.cpp:72-103): 9개 곱 후 **수동 가산트리**(sum_0~3→res_0~2→res, SkyNet.cpp:92-101)로 비트폭 정밀 제어(ap_int<14>곱→ap_int<18>합).
- 출력 [rmin,rmax] 클램프(SkyNet.cpp:143).

### 3.2 Pointwise Conv 1x1 — PWCONV1X1 (SkyNet.cpp:233-277)
- `MAC16`(SkyNet.cpp:157-219): 16-입력채널 1x1 누산을 **완전 펼친 가산트리**(mul0~15→add0~7→...→res, SkyNet.cpp:182-217) — II=1 달성용.
- `LOAD_W1x1`(SkyNet.cpp:221-231)로 16채널 가중치 타일 적재. 본체는 ci를 16씩(SkyNet.cpp:243), co 32 UNROLL(SkyNet.cpp:251), `#pragma HLS PIPELINE II=1`(SkyNet.cpp:250). OFM 누산 후 [rmax,rmin] 클램프(SkyNet.cpp:272).
- README의 "PWCONV 병렬도 512" = 16(ci)×32(co) MAC 동시(SkyNet.cpp:255-271).

### 3.3 활성/양자화 — ACTIVATION (SkyNet.cpp:287-315)
- **fully-integer 양자화**: `qy = IFM + BBUF`(bias 가산, SkyNet.cpp:300) → ReLU(`qy<0→0`, SkyNet.cpp:302-305) → `(qy*MBUF)>>nm`(스케일 곱+시프트, SkyNet.cpp:307, nm=17 SkyNet.h:52) → [amin,amax]=[0,255] 클램프 8-bit(SkyNet.cpp:309). 부동소수 전무 → "One Shot Fully Integer"(README.md:12-16)의 코드 증거.

### 3.4 reorg / pooling / 타일 입출력
- **REORG**(SkyNet.cpp:35-60): 2×2 stride-2 공간→채널 재배치(40×80→20×40, ch×4), `#pragma HLS PIPELINE II=1`(SkyNet.cpp:44). README의 "원본 SkyNet reorg는 미최적화였음"(README.md:19)을 II=1로 개선.
- **POOL**(SkyNet.cpp:392-412): 2×2 max(`MAX`, SkyNet.cpp:279-285), II=4(SkyNet.cpp:402).
- **타일 DRAM 입출력**: `Load_FM`/`Export_FM`/`Load_FM1`/`Export_FM1`(SkyNet.cpp:361-496)이 외부 `fm` 버퍼에서 42×82 타일을 읽고 쓰며 h_o/w_o 오프셋으로 타일 위치 계산(SkyNet.cpp:363-374). **온칩 BRAM이 부족해 중간 feature를 DRAM 왕복**하는 타일 전략 — UltraNet류 full-dataflow와 대조되는 핵심 차이.

### 3.5 검출 후처리 — Compute_BBOX (SkyNet.cpp:507-578)
- 4개 그리드 영역(SkyNet.cpp:514-522)별 2앵커 confidence 최대(`OFM[4]·MBUF[4]` vs `OFM[9]·MBUF[9]`, SkyNet.cpp:547-549) → 박스좌표 추출+clamp(SkyNet.cpp:569-576). bbox는 HW에서 산출(예외적으로 PL 내 검출).

### 3.6 Top — SkyNet (SkyNet.cpp:601-938)
- `#pragma HLS INTERFACE m_axi`로 img/fm/weight/biasm 4개 DRAM 포트(SkyNet.cpp:603-606), img·fm 버스 bundle 공유(BRAM 절감, README.md:179-185).
- `#pragma HLS ALLOCATION ... limit=1 function`(SkyNet.cpp:609-617)로 DWCONV/PWCONV/POOL/ACT 등 **단일 인스턴스 자원공유**(레이어를 순차 재사용) — full-dataflow(레이어별 인스턴스)와 정반대 전략.
- DWCONV1~6 + PWCONV1~6 블록을 명시 루프로 호출(SkyNet.cpp:619-924), 4-batch 처리(SkyNet.cpp:633), reorg+concat bypass(SkyNet.cpp:860-898).

---

## 4. 데이터플로우 (타일 + 자원공유, UltraNet류와 대조)
```
img(RGBA32) →[DRAM타일 로드]→ DWCONV3x3 →ACT →PWCONV1x1 →ACT →POOL →[DRAM 저장]
   (DW+PW 5블록 반복, 단일 인스턴스 재사용) → reorg+concat → DWCONV6 →PWCONV6 →CONV13 →bbox
```
- 레이어 간 데이터는 **외부 DRAM `fm`** 경유(SkyNet.cpp:73-84 오프셋 상수). 온칩은 32×43×83 타일 4개(FM1~4, SkyNet.cpp:25-28)만.

---

## 5. HW/SW 매핑
- HW(PL): SkyNet IP(DW/PW/pool/reorg/act/bbox 전부), 333MHz(README.md:21).
- SW(PS): run.py/run_multiprocess.py(Deploy/)가 PYNQ로 .bit 로드, 가중치(.bin) 적재, 이미지 로딩+색공간변환(transform.cpp 계열). 멀티프로세스로 로딩 은닉(README.md:28).

---

## 6. 빌드 · 실행
- C-sim(g++): `cd C; mkdir build; cmake ..; make; ./SkyNet`(README.md:114-121). HLS Arbitrary Precision Types로 ap_int 시뮬.
- HLS: `vivado_hls hls.tcl`(README.md:156). Vivado: `vivado -mode tcl -source rtl.tcl`(README.md:163). 또는 SDSoC(README.md:178-193).
- 데모: `cd Deploy; sudo python3 run.py`(README.md:46).

---

## 7. 의존성
- Vivado HLS / SDSoC, `ap_int.h`(SkyNet.h:14), PYNQ. `__SDSCC__`면 sds_alloc, 아니면 malloc(SkyNet.h:16-21). C-sim은 HLS_arbitrary_Precision_Types.

---

## 8. 강점 · 한계
**강점**
- **MobileNet형 DW+PW**로 연산량 대폭 절감 → 73% IoU를 저자원에서 달성.
- **Fully-integer 추론**(ACTIVATION SkyNet.cpp:300-309) — 부동소수 0, 파인튜닝 불필요(One-shot, README.md:12-16). 정확도 손실 0(README.md:15).
- **모든 II=1 최적화**(PWCONV/reorg/load, README.md:19) + 32채널 완전병렬.
- 수동 가산트리(MAC9/MAC16)로 비트폭/타이밍 정밀 제어 → 333MHz.

**한계**
- **DRAM 타일 왕복**(Load_FM/Export_FM, SkyNet.cpp:361-496)으로 DRAM 대역폭이 병목 — full-dataflow(UltraNet) 대비 지연 큼(52fps vs UltraNet류 200+fps).
- 단일 인스턴스 자원공유(ALLOCATION limit=1, SkyNet.cpp:609)로 레이어 순차 → 처리량 제약.
- 32채널·43×83 타일·레이어 오프셋 전부 하드코딩(SkyNet.h:73-127) → 모델 변경 어려움.
- im2col 아님(직접 타일 컨볼루션) — 작업지시 가설 정정.

**SkyNet(DW+PW tiled) vs UltraNet(full-dataflow) 대비표**
| 항목 | SkrSkr/SkyNet | UltraNet 계열(2022/2023) |
|---|---|---|
| 컨볼루션 | DW3x3 + PW1x1 | 표준 3x3 + 1x1 |
| 중간 feature | **외부 DRAM 타일** | **온칩 스트림(full dataflow)** |
| 자원 | 단일 인스턴스 공유 | 레이어별 인스턴스 |
| 양자화 | W6A8 fully-int | W4A4 |
| DSP | MAC9/MAC16 가산트리 | INT/UINT-Packing |
| fps | 52~57 | 200+ |

---

## 9. 우리 프로젝트 시사점 (ViT/Transformer + XR)
- **DRAM 타일 전략의 교훈(반면교사)**: SkyNet의 DRAM 왕복이 처리량 병목(52fps)임을 보여줌 → ViT는 시퀀스가 짧고(XR 시선추적은 작은 입력) **온칩 full-dataflow(UltraNet/HG-PIPE)** 가 유리. 단 ViT 가중치가 큰 FFN은 타일+DRAM이 불가피할 수 있어, SkyNet의 `Load_FM`/오프셋 타일링(SkyNet.cpp:361-374)이 weight-stationary 타일 설계 참고가 됨.
- **Fully-integer 추론**(SkyNet.cpp:300-309)은 ViT 양자화의 이상적 목표 — bias가산→ReLU→스케일곱+시프트 패턴을 ViT의 (LayerNorm 근사/GELU 양자화 후) 정수 활성에 적용 시도 가능(추정).
- **DW-Separable 구조**: ViT 변형 중 MobileViT/효율 ViT의 DWConv 블록은 SkyNet DWCONV3X3(SkyNet.cpp:105) 라인버퍼를 그대로 재사용 가능 — 하이브리드 CNN-ViT 백본 시 직접 차용.
- **수동 가산트리(MAC16, SkyNet.cpp:157)** 비트폭 제어 기법은 ViT 누산기 오버플로 방지·고클럭 달성에 참고.
- **RGBA32 로딩/PL 색공간변환**(README.md:20,27): XR 카메라 입력 파이프라인 전처리 최적화 차용.
