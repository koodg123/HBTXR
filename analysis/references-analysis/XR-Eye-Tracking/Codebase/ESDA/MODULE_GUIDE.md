# ESDA 모듈 가이드 (XR 시선추적 관점 cross-ref)

> 1차 요약: [`../ESDA.md`](../ESDA.md)
> HW 심층 분석(통합 가이드): [`../../../CNN-Accel/ESDA/MODULE_GUIDE.md`](../../../CNN-Accel/ESDA/MODULE_GUIDE.md)
> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\ESDA`
> 작성 원칙: 실제 소스 Read 후 `파일:라인` 근거 표기. 라인 근거 없는 추론은 "추정", 코드로 확인 불가는 "확인 불가". (UNC 경로라 bash 미사용 — Glob/Grep/Read만 사용.)

---

## 0. 머리말

- **본 문서는 XR(시선추적) 관점의 cross-ref 문서다.** ESDA의 HW(HLS dataflow 가속기)·DSE·codegen·board 측 **심층 모듈 분석은 중복이므로 작성하지 않는다.** 그 내용은 [`../../../CNN-Accel/ESDA/MODULE_GUIDE.md`](../../../CNN-Accel/ESDA/MODULE_GUIDE.md)(토큰 자료형/zero-skip line buffer/DSP packing MAC/inverted-residual 조립/codegen/DSE/board harness 14절)를 참조한다.
- 본 문서가 다루는 것: ① 이 XR 사본이 HW 사본(`CNN-Accel/ESDA`)과 **동일한지/무엇이 다른지**(파일 비교), ② 분류용 ESDA를 **이벤트 기반 시선·동공 좌표 회귀**로 쓰려면 무엇을 바꿔야 하는가, ③ 우리 프로젝트(XR + FPGA on-device)에의 시사점.
- ESDA의 정체: 이벤트 카메라 데이터를 sparse하게 표현한 뒤 **양자화 Sparse MobileNetV2를 FPGA(ZCU102)에서 layer-pipeline dataflow로 저지연 추론**하는 SW/HW 통합 reproducibility 패키지(FPGA'24). 상세는 `../ESDA.md` 1절.

---

## 1. HW 사본(CNN-Accel/ESDA)과의 동일/차이

### 1.1 결론 한 줄
- **XR 사본 = 상위(superset, "full") 사본. HW 사본 = HW 측만 남긴 부분 사본.** XR 사본은 SW 양자화 학습부(HAWQ)와 HW 측을 **모두** 포함하지만, HW 사본에는 **SW HAWQ 학습부가 부재**하다.

### 1.2 근거(파일 존재 비교)
| 항목 | XR 사본 | HW 사본(CNN-Accel) | 근거 |
|---|---|---|---|
| SW 양자화 모델 `software/models/HAWQ_mobilenetv2.py` | **존재** | **부재** | XR: Glob `software\models\HAWQ_mobilenetv2.py` 매치 / HW: Glob `**/HAWQ_mobilenetv2.py` "No files found" |
| SW QAT 모듈 `HAWQ_quant_module/` | **존재** | **부재** | XR Glob 매치(`software\models\HAWQ_quant_module\quant_modules.py`, `quant_utils.py`) / HW Glob 0건 |
| HW HLS 템플릿 `hardware/template_e2e/` | **존재** | **존재** | 양쪽 모두 `hardware\template_e2e\type.h` 매치 |
| 토큰 자료형 `template_e2e/type.h` | **동일(바이트 일치)** | (기준) | XR `type.h:1-16` ≡ HW `type.h:1-16` 직접 Read 대조: `T_K{end,x,y}`, `BundleT<N,T>`, `T_OFFSET=ap_uint<4>`, `end_3x3 15` 전부 동일 |

- 즉 HW 사본의 통합 가이드(`CNN-Accel/ESDA/MODULE_GUIDE.md` 1.2 "부재" 목록, 같은 문서 50행)가 명시한 대로 — "HAWQ 모델 구현은 동형 사본 XR-Eye-Tracking/Codebase/ESDA에 존재"라는 진술이 **본 사본에서 그대로 확인**된다.
- **추정**: HW 측(`template_e2e/`의 conv.h/linebuffer.h/conv_pack.h/mem.h/gen_code.py 등)·`optimization/`·`board/`는 두 사본이 동형이다(type.h 바이트 일치 + 디렉토리 구조 동일로 추정). 전 파일 바이트 diff는 미수행 → **확인 불가**(UNC·bash 금지로 해시 비교 불가). 단 HW 심층 분석은 어차피 HW 사본 가이드로 대체되므로 본 문서 목적상 영향 없음.

### 1.3 그래서 XR 관점에서 "추가로 볼 가치"가 있는 부분
- HW 사본에 없는 **SW 학습/익스포트 파이프라인**(`software/main.py`, `int_inference.py`, `models/HAWQ_*`, `dataset/`)이 XR 사본의 고유 가치. 시선추적 태스크로 개조하려면 결국 이 SW부를 손대야 하므로, XR 관점 분석은 **HW가 아니라 이 SW부와 데이터 표현**에 집중하는 것이 맞다. (SW 모듈 라인 근거 상세는 `../ESDA.md` 3-A절.)

---

## 2. XR 시선추적 적용 관점

### 2.1 현 상태: ESDA는 "분류 전용", 시선추적 헤드 없음(근거)
- 모델 최종단은 **분류기**다: `software/main.py:96,99`에서 모델을 `num_classes=nr_classes`로 생성(MobileNetV2 / MobileNetV2ME). 동공/시선 좌표 회귀용 인자·헤드는 없음.
- HW 최종단도 분류: `global_avgpool_linear`로 클래스 수만큼 `ap_int<32>` logit 출력(HW 가이드 4.5/12절, `conv.h:490`). GAP+FC = 분류 헤드.
- 시선추적 관련 식별 결과: **eye/gaze/pupil/eye_tracking/regress 키워드는 시선추적 헤드로서 0건.** `software/` 전체 Grep(`eye|gaze|pupil|regress|coordinate|center` i) 68파일 매치는 전부 **MinkowskiEngine 내부의 좌표맵(coordinate map)·좌표 센터링** 또는 데이터 전처리 잔재이며 시선추적 회귀 헤드가 아님(예: `MinkowskiCoordinateManager.py`, `utils/coords.py`). `HAWQ_mobilenetv2.py`에 eye/gaze/pupil/head/regress **0 매치**.
- 시선추적 전용 경로(예 `pipeline.py`의 eye_tracking 분기)·전용 데이터셋·좌표 회귀 head: **없음**. `**/pipeline*.py` Glob "No files found", 데이터셋은 분류용(ASL-DVS/DVSGesture/N-MNIST/RoShamBo/N-Caltech101 — `../ESDA.md` 1·5절).
- **정리**: XR 사본이 `XR-Eye-Tracking` 폴더 아래 있는 이유는 **시선추적 구현이 들어 있어서가 아니라**, "이벤트 → FPGA 저지연 sparse 추론"의 **레퍼런스 아키텍처**로서 포함된 것으로 추정(`../ESDA.md` 1.17·8절과 동일 결론). 시선추적 헤드 자체는 미구현 → 그 부분은 **확인 불가**.

### 2.2 동공/시선 좌표 회귀로 쓰려면 필요한 변경
ESDA(이벤트 sparse 분류)를 시선추적(동공 중심 (x,y) 또는 시선 벡터 회귀)으로 전환할 때 필요한 변경점:

1. **헤드 교체(분류 → 회귀)**: GAP+FC(`global_avgpool_linear`) 출력을 N_CLASS logit이 아니라 **연속 좌표 K차원**(동공 (x,y)=2, 양안+신뢰도면 더 큼)으로 바꿔야 한다. SW에선 `num_classes`(main.py:96,99)를 회귀 차원으로 재해석·재학습, HW에선 GAP 뒤 FC를 그대로 K-출력 GEMM으로 쓰되 **분류용 argmax 대신 정수 logit을 좌표로 디양자화(dequant)**해야 함(추정 — HW 최종 dequant 스케일 처리 필요).
2. **손실/타깃 변경**: CrossEntropy → 좌표 회귀 손실(L1/L2 또는 wing/heatmap 류). 현재 `main.py`에 회귀 손실 없음(2.1 근거) → SW 학습부에 추가 필요.
3. **라벨/데이터셋 교체**: 분류 라벨 대신 (x,y) 좌표 GT를 가진 이벤트 시선추적 데이터셋(예: 이벤트 동공 데이터)으로 `dataset/` 로더 교체. 현 로더는 분류 전용(`../ESDA.md` 5절).
4. **양자화 재보정**: 회귀는 출력 동적범위가 분류 logit과 다르므로 마지막 단 scale/bias/EXP(`para.h`, 데이터셋별 16/32b)와 QAT 재캘리브레이션 필요(추정). HAWQ QAT 골격(`quant_modules.py`)은 재사용 가능.
5. **HW 코어는 대부분 재사용 가능**: 토큰 기반 zero-skip line buffer, INT8 DSP-pack MAC, inverted-residual dataflow 조립, codegen(`gen_code.py`)은 backbone 공통이라 **헤드만 교체**하면 됨(추정). 좌표 정밀도 요구가 높으면 마지막 1~2 레이어 비트폭 상향 검토.
6. **해상도 주의**: 토큰 좌표가 8비트(`type.h:4-5`)라 최대 255×255 입력. 동공 추적이 더 고해상도라면 좌표폭 확장(=재합성) 필요(HW 가이드 2.5·14.1절 동일 지적).

### 2.3 저지연·on-device 이점(시선추적 맥락)
- **이벤트 희소성 직접 활용**: 시선추적 입력(이벤트 카메라)은 본질적으로 극희소(ESDA 분류 입력 sparsity 0.01~0.5, `../ESDA.md` 4절)이며, ESDA는 토큰(좌표)+특징 쌍 스트림으로 **비영점 픽셀만 MAC**(zero-skip, HW 가이드 3절). XR 시선추적의 고프레임률·저전력 요구에 정합 — dense CNN 대비 연산·대역폭 절감.
- **layer-pipeline = 초저지연**: 모든 레이어가 동시에 살아 스트림 연결(단일 `#pragma HLS DATAFLOW`)이라 frame-level latency가 짧음. AR/VR foveated rendering·gaze 인터랙션의 ms급 지연 예산에 유리(추정).
- **온칩 가중치 ROM**: 가중치를 BRAM/LUTRAM에 임베드(외부 트래픽 0, HW 가이드 4.6절) → on-device(보드 단독) 추론에 적합.

---

## 3. 우리 프로젝트(XR + FPGA) 시사점

- **직접 참조 가치 = 매우 높음.** 본 repo는 "이벤트 카메라 → FPGA on-device 저지연" 파이프라인의 거의 모든 빌딩블록을 제공(상세 5개 항목 `../ESDA.md` 8절). 시선추적용으로는 **backbone·HW 코어를 그대로 두고 헤드/손실/데이터만 교체**하는 전략이 비용 대비 효과 큼(2.2).
- **재사용 우선순위(시선추적 관점)**:
  1. `linebuffer.h` 토큰+valid zero-skip(이벤트 극희소 직접 활용) — HW 가이드 3절.
  2. `conv.h` DSP_AM(2 MAC/DSP)로 INT8 처리량 ×2 — HW 가이드 4절.
  3. `gen_code.py` + cfg.json 자동 HLS 생성 → 시선추적 모델로 cfg만 교체 — HW 가이드 7절.
  4. `optimization/`(SCIP ILP)로 우리 보드 자원예산에 맞춰 레이어별 병렬도 재매핑 — HW 가이드 10절.
- **우리가 추가/검증할 것**: (a) **분류 헤드 → 좌표 회귀 헤드 + 회귀 손실**(2.2-1,2), (b) 이벤트 **시선추적 데이터셋·라벨** 로더(2.2-3), (c) 회귀 출력 dequant·양자화 재캘리브레이션(2.2-4), (d) 좌표 정밀도 vs INT8 trade-off 검토(필요 시 마지막 단 비트폭 상향), (e) 8비트 좌표 해상도 상한 확인(2.2-6).

### 확인 불가 / 추정 정리
- **확인 불가**: ① XR↔HW 사본의 HW 측 전 파일 바이트 동일 여부(type.h만 일치 확인, 전체 diff 미수행 — bash 금지). ② 시선추적 헤드의 실제 구현(부재 → 코드로 검증 불가). ③ 회귀 전환 시 정확도/지연(미실행). ④ HLS 합성 LUT/FF/주파수 실측(.rpt 미커밋, `../ESDA.md` 5·9절).
- **추정**: ① XR 사본이 XR 폴더에 든 사유(시선추적 레퍼런스 아키텍처로 포함). ② HW 코어가 헤드 교체만으로 회귀 재사용 가능하다는 점. ③ layer-pipeline의 ms급 지연 이점.

---

*근거 파일(절대경로)*:
`\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\ESDA\hardware\template_e2e\type.h`(1-16, HW 사본과 바이트 일치),
`...\REF\XR-Eye-Tracking\Codebase\ESDA\software\main.py`(96,99 num_classes 분류 헤드),
`...\REF\XR-Eye-Tracking\Codebase\ESDA\software\models\HAWQ_mobilenetv2.py`(eye/gaze/pupil/head/regress 0 매치),
`...\REF\XR-Eye-Tracking\Codebase\ESDA\software\models\HAWQ_quant_module\{quant_modules.py,quant_utils.py}`(XR 사본 고유),
대조: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\CNN-Accel\ESDA\hardware\template_e2e\type.h`(1-16) 및 `...\CNN-Accel\ESDA`(HAWQ_* 부재).
