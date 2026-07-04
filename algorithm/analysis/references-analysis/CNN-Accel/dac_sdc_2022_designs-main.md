# dac_sdc_2022_designs-main 정밀 분석 (서브팀 비교)

> 경로: `REF/CNN-Accel/dac_sdc_2022_designs-main/`
> 분석 방식: 각 서브팀 폴더 좁은 Glob 후 핵심 소스 Read. 라인 근거 표기.

---

## 1. 개요 (목적/대회/타깃보드)

- **정체**: DAC-SDC **2022** 대회의 **여러 참가팀 설계 모음집**. 서브팀: `SEUer/`, `InvolutionNet/`, `ultrateam/`. 대회/보드는 2022 트랙과 동일(Ultra96 v2, 저전력 객체검출).
- 본 문서는 서브팀별 HLS 아키텍처를 **비교**한다.

---

## 2. 디렉토리 구조 (자체 + 제외 이유) — 서브팀 실재 여부 확인

Glob 결과 기준 서브팀별 실재 여부:

| 서브팀 | 소스 실재 | 비고 (Glob/Read 근거) |
|---|---|---|
| **SEUer/** | ✅ 풀 HLS 소스 | src/conv2d_DSPopt3.hpp, conv1x1DSP2.hpp, conv2d_l0_opt.hpp, ultranet.cpp 등 — champion repo와 **동일 파일군**. SEUer/README.md:1 = "refer to dac_sdc_2022_champion". |
| **InvolutionNet/** | ✅ 풀 HLS 소스 + deploy(.bit/.ipynb) | src/ 14+개 .hpp/.h + Resize_opr_linear_opt.h(자체 특화). README(최상위)는 **빈 파일**(InvolutionNet/README:1 공백). |
| **ultrateam/** | ❌ **소스 미동봉** | Glob 결과 `ultrateam/README` 단 1개 파일, 그나마 **빈 파일**(ultrateam/README:1 공백). 좁은 Glob 재시도해도 .cpp/.h 없음 → **소스 미존재 확정**. |
| (최상위) | — | LICENSE 1개. |

- **제외(이름)**: InvolutionNet/deploy/*.bit, *.hwh, *.so(yolo.so), src/data/*.bin·*.jpg, weights*.hpp/weight3.hpp(가중치 생성물) — 알고리즘 정보 없음.
- **소스 미동봉 repo**: **ultrateam** — README만 존재, 그마저 빈 파일.

---

## 3. 핵심 모듈 정밀 분석 (서브팀별)

### 3.1 SEUer/ (= 2022 champion)
- 파일/커널이 champion-master와 동일. INT-Packing DSP MAC(`simd_MAC`, SEUer/src/conv2d_DSPopt3.hpp), conv0 LUT-MAC(conv2d_l0_opt.hpp), conv1x1 2출력/DSP(conv1x1DSP2.hpp). 상세는 `dac_sdc_2022_champion-master.md` 참조(동일 코드).
- 입력 160×320, W4A4, 9-레이어. config_opt3.h 동일.

### 3.2 InvolutionNet/ (자체 특화 = UltraNet + 전처리 리사이즈)
공통 컨볼루션 커널은 SEUer 계열과 **거의 동일**(conv2d_DSPopt.hpp, conv1x1DSP2.hpp, conv2d_l0.hpp, function.h의 `bn_qurelu_fixed` 동일 — InvolutionNet/src/function.h:182-201). 차별점:

- **하드웨어 양선형 리사이즈(bilinear resize) 전처리** `Resize_opr_linear_opt.h`:
  - `Resize_opr_linear_try`(Resize_opr_linear_opt.h:11-181): 360×640 → 160×320 다운스케일. `LineBuffer<2,...>`+`Window<2,2,...>` 2×2 윈도우 양선형 보간 (Resize_opr_linear_opt.h:19-20, 171-174). `ap_fixed<32,16>` 비율 계산(Resize_opr_linear_opt.h:29-30).
  - **SIMD2 버전** `Resize_opr_linear_simd2`(Resize_opr_linear_opt.h:186-306): 입력 버스 `ap_uint<24*2>`로 2픽셀 동시 리사이즈, `#pragma HLS PIPELINE II=1`(Resize_opr_linear_opt.h:230). DDR 대역폭 절감용.
  - 즉 champion이 호스트/외부에서 하던 리사이즈를 **PL로 내재화**한 것이 InvolutionNet의 차별 기여(추정: 팀명 Involution과 무관하게 실제 src는 UltraNet+resize 구조).
- config.h: conv0 PE=16(InvolutionNet/src/config.h:12, CONV_0_PE_DSP6 16 config.h:23) — champion(PE_DSP6=8)보다 conv0 병렬도 2배. 입력 8-bit×3ch 동일(config.h:13-15).
- `padding_var`(가변 패딩, function.h:15-58)로 런타임 크기 패딩 지원.

### 3.3 ultrateam/ — 소스 미동봉 (분석 불가)
- ultrateam/README 빈 파일만 존재. 아키텍처 확인 불가.

### 서브팀 아키텍처 비교 요약

| 항목 | SEUer | InvolutionNet | ultrateam |
|---|---|---|---|
| 모델 | UltraNet 9-layer | UltraNet 9-layer + HW resize | 확인 불가 |
| DSP MAC | INT-Packing 4MAC/DSP | 동일(conv2d_DSPopt.hpp) | 확인 불가 |
| conv0 | LUT-MAC | LUT-MAC | 확인 불가 |
| 전처리 | 외부 | **PL 내 bilinear resize** | 확인 불가 |
| conv0 PE | 8 | 16 | 확인 불가 |
| 소스 | 풀 | 풀 + .bit | README(빈) |

---

## 4. 데이터플로우
- SEUer: champion과 동일(AXIS→폭변환→conv0~8→AddLast).
- InvolutionNet: **AXIS→ExtractPixels→resize(360×640→160×320)→conv0~8** — resize가 추가 전단. function.h의 padding/`padding_var`가 각 conv 전 패딩.

---

## 5. HW/SW 매핑
- InvolutionNet: deploy/dac_sdc.bit + dac_sdc.ipynb로 PYNQ 배포, 박스 후처리 `yolo.cpp`→`yolo.so`(g++ 컴파일, InvolutionNet/deploy/readme.md:3)로 PS측 수행. deploy_mFreq/는 다른 클럭 변형(추정).

---

## 6. 빌드 · 실행
- InvolutionNet: scripts/InvolutionNet_hls.tcl, InvolutionNet_vivado.tcl. 박스 lib: `g++ -shared -O3 yolo.cpp -o yolo.so -fPIC -lpthread` (InvolutionNet/deploy/readme.md:3).
- SEUer: champion과 동일 스크립트.

---

## 7. 의존성
- Vivado HLS + `hls_video.h`(InvolutionNet resize의 Mat/LineBuffer/Window, Resize_opr_linear_opt.h:5). 박스 후처리는 pthread(libpthread).

---

## 8. 강점 · 한계
**강점**: 동일 베이스(UltraNet) 위에서 팀별 차별화 관찰 가능 — InvolutionNet의 HW resize는 대역폭/지연 개선의 좋은 사례(Resize_opr_linear_opt.h:186, simd2 II=1).
**한계**: ultrateam 소스 부재로 3팀 완전 비교 불가. InvolutionNet README가 빈 파일이라 설계 의도는 코드 역추적에 의존(추정 다수).

---

## 9. 우리 프로젝트 시사점
- **HW 전처리 내재화**(InvolutionNet resize): XR 시선추적은 카메라 원본→ROI 리사이즈가 필수인데, `Resize_opr_linear_simd2`(Resize_opr_linear_opt.h:186)처럼 **PL에서 2픽셀 II=1 양선형 리사이즈**를 ViT 전단에 두면 PS-PL 대역폭 절감 + 저지연 확보(직접 차용 가능).
- **동일 백본 위 팀별 변형 패턴**은 우리도 HG-PIPE 기준선 위에 변형(resize/양자화/병렬도) 실험 시 좋은 ablation 구조 참고.
- DSP 패킹/dataflow 시사점은 champion 문서와 동일.
