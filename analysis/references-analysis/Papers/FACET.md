# FACET: Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality

## 1. 서지정보
- **제목**: FACET: Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality
- **저자**: Junyuan Ding (Beihang Univ.), Ziteng Wang·Min Liu (DVSense), Chang Gao (TU Delft), Qinyu Chen* (Leiden Univ., 교신)
- **연도**: 2025
- **학회/저널**: IEEE ICRA 2025 (Atlanta), DOI 10.1109/ICRA55743.2025.11127327
- **코드**: github.com/DeanJY/FACET
- **원본파일명**: FACET_Fast_and_Accurate_Event-Based_Eye_Tracking_Using_Ellipse_Modeling_for_Extended_Reality.pdf

## 2. 문제정의·배경
- XR 시선 기반 상호작용에 아이트래킹은 필수이나 프레임 기반 시스템은 XR의 고정확도·저지연·저전력 요구를 동시 충족 못함(HMD 추적 지연 45~81ms, kHz 프레임은 watt급 전력).
- 이벤트 카메라는 sparse·고시간해상도·저전력으로 대안. 그러나 기존 이벤트 기반 방법 대부분은 매 step 검출 NN을 돌려 연산 비용이 큼. detection-tracking schema(타원 추적+분실 시만 NN)는 효율적이나, **segmentation 모델(U-Net)로 마스크를 얻고 타원 피팅** → 연산 큼, 이벤트가 동공 *경계*를 강조한다는 특성 미활용.

## 3. 핵심 기여
- **end-to-end 이벤트 기반 동공 검출기**: 이벤트 입력 → 동공 **타원 파라미터(x,y,a,b,θ) 직접 출력**(분할-피팅 불필요). 기존 detection-tracking schema에 바로 삽입 가능.
- **데이터셋 강화**: 반지도학습으로 EV-Eye를 9,000개 라벨 → **150만 샘플** 라벨링, 마스크 라벨을 타원 파라미터 라벨로 변환.
- **Trigonometric Loss**: 타원 각도 예측의 불연속 문제(0°와 180°가 동일 타원) 해결.
- **Fast Causal Event Volume**: 이벤트 누적 표현법(누적값 제한으로 처리시간 단축, min-max 정규화로 안정 학습).

## 4. 방법론
### 이벤트 표현
- **Fixed-Count Binning**: 고정 시간 대신 **고정 이벤트 수(5000개)** 단위로 binning → 무동작 시 불필요 추론 회피, 동작 시 과부하 방지.
- **Fast Causal Event Volume** (causal event volume 개선): 시점 t 이전 이벤트만 사용(실시간 인과성). 누적값에 limit $l$ 적용 → 같은 좌표 이벤트가 limit 도달 시 누적 중단(시간 단축).
$$V_{pos}(x,y) = \sum_{\{E_i|p_i=1\}}\delta_{x_i,x}\cdot\delta_{y_i,y}\cdot k\left(\frac{t-t_i}{\Delta t}\right), \quad V_{neg}(x,y) = \sum_{\{E_i|p_i=0\}}\delta_{x_i,x}\cdot\delta_{y_i,y}\cdot k\left(\frac{t-t_i}{\Delta t}\right)$$
- 커널: $k(\tau) = H(\tau)\max(1-|\tau|, 0)$ ($H$=Heaviside step). 입력 표현은 256×256×2(pos/neg).

### 모델 구조 (경량 검출기)
- **Backbone**: MobileNetV3 (DSC=depthwise separable conv + SE 블록, 엣지 친화).
- **Neck**: FPN의 일반 conv를 모두 **DSC로 교체**. $P_i = DSC(C_i) + \text{Upsample}(P_{i+1})$, $i\in\{5,4,3,2\}$. DSC 복잡도 $O(HWC^2) \to O(HWC+C^2)$. 최종 P2=(64,64,64).
- **4개 Head** (각 3×3 conv + ReLU + 1×1 conv):
  - Heatmap Head(64×64, 동공 중심), Offset Head(양자화 오차 보정), Size Head(a,b), Rotation Head(회전).
  - 회전은 raw로 $\hat{\vec{r}} = (\hat{\sin}(2\theta), \hat{\cos}(2\theta))$ 출력 후 $\vec{r}=\hat{\vec{r}}/\|\hat{\vec{r}}\|_2$로 정규화하여 θ 복원.

### 손실함수
$$L = \lambda_H L_H + \lambda_O L_O + \lambda_S L_S + \lambda_G L_G + \lambda_T L_T$$
- $L_H$(heatmap focal loss), $L_O$/$L_S$(smooth L1, CenterNet), $L_G$(Gaussian IoU, ElDet — 타원을 2D Gaussian으로 보고 Wasserstein 거리).
- **Trigonometric Loss** $L_T = L_2(\hat{\vec{r}}_p, \vec{r}_g)$: θ를 $(\sin 2\theta, \cos 2\theta)$로 매핑하여 불연속 도메인 [0,180)을 연속 2D로 변환. (예: 179° vs 1° → $L_T\approx0$, vs 90° → 큰 loss. 일반 angle loss $L_A$는 반대로 어긋남.)

### 학습
- PyTorch, RTX 3090, batch 32, 70 epoch, Adam, 초기 lr 1e-3, weight decay 1e-5, 5 epoch warm-up(1e-5), 10 epoch마다 0.7배 감쇠. fast causal event volume limit $l$=25. 증강(회전/스케일/이동/수평flip).

## 5. 실험
- **데이터셋**: 강화 EV-Eye(48명, 150만 grayscale + 27억 이벤트, DAVIS346 ×2). train 20k/val 5k/test 5k. 평가 해상도 64×64.
- **메트릭**: P10/P5/P1(중심이 n픽셀 내 확률), PE(평균 픽셀 오차), Params, GFLOPs, Inference Time.
- **정확도/효율 주요 결과 (Table II)**:
  - **FACET: P10 100, P5 99.98, P1 99.59, PE 0.2030px, 3.92M params, 3.44 GFLOPs, 0.5302ms** (대부분 항목 최고).
  - EV-Eye: PE 0.3231, 17.27M, 40.11 GFLOPs, 0.9438ms → FACET 대비 4.4× params, 11.7× ops, 1.8× 지연.
  - E-Track: PE 1.6680, 17.27M, 40.19 GFLOPs.
  - TennSt: 0.81M(최소), 0.3384ms(최속)이나 PE 1.1291·P1 73.67%(저정확), 타원 추적 모듈과 비호환.
- **Ablation (Table III)**:
  - Fast causal event volume이 causal/event volume 대비 EPT 0.13/0.20ms 단축 + 정확도 동급.
  - Fixed-count 5000 evts가 최적 밸런스(500→P1 93.52%, 10000→P1 99.89%이나 EPT 2.90ms).
  - Trigonometric Loss vs Angle Loss: P1 99.59% vs 98.90%, PE 0.2030 vs 0.2878 (loss 설계 효과 확인).

## 6. 강점/한계
- **강점**: end-to-end 타원 직접 출력(분할-피팅 제거); MobileNetV3+DSC FPN으로 매우 경량(3.92M/3.44G)·저지연(0.53ms, TensorRT); fixed-count + fast causal volume로 무동작 시 추론 절약; Trigonometric Loss로 각도 불연속 해결; XR 타깃 명확.
- **한계**: 단일 프레임 검출기(시간 모듈 없음 → 자체적으로 시퀀스 시간 일관성 미보장, detection-tracking schema의 검출 단계 역할); 평가 64×64 저해상도; 양자화/FPGA 구현은 미수행(future work에서 NPU+이벤트 센서 통합 언급 → 현재 논문엔 "확인 불가").

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속 — 추정)
- **MobileNetV3 + DSC FPN**은 FPGA/NPU 가속에 매우 친화적(DSC는 MAC 수가 적고 채널 분리로 PE 매핑 용이). **INT8 양자화** 시 MobileNet 계열은 검증된 사례 많아 우리 가속기 후보 백본으로 유력(추정).
- **Fast Causal Event Volume의 limit 누적**은 FPGA에서 포화 가산기(saturating accumulator)로 직접 구현 가능하며, limit 도달 시 조기 종료로 처리 사이클·전력 절감 → 이벤트 전처리 IP로 매력적(추정).
- **Fixed-count binning**은 입력 데이터 크기를 고정 → FPGA 버퍼/파이프라인 설계가 결정적(deterministic)이 되어 저지연 보장에 유리(추정).
- **타원 직접 출력(5 파라미터)** 은 후단 추적기(저비용 타원 추정)로 high-frequency 추적을 구성하는 detection-tracking 구조 → 우리 시스템도 "무거운 검출 IP(저빈도) + 경량 타원 추적(고빈도)" hybrid로 설계해 지연·전력 최적화 가능(추정).
- **Trigonometric Loss / Gaussian IoU(Wasserstein)** 는 학습 단계 기법이라 HW 비용 무관 → 채택 가능.
- 직접 비교군: EV-Eye, E-Track, TennSt(=Pei 경량 STN)와 동일 데이터셋 벤치마크 → 우리 정확도-효율 곡선 작성에 활용.

## 8. 근거표기
- 4-6섹션 수식·구조·수치(Table II/III, 0.20px/0.53ms/3.92M/3.44G 등)는 PDF 본문(pp.10347-10354) 직접 근거.
- 7섹션 FPGA/양자화 매핑은 분석자 해석 "추정"(논문엔 NPU 통합을 future work로만 언급, 양자화/FPGA 구현 결과 "확인 불가").
