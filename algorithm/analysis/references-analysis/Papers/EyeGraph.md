# EyeGraph: Modularity-aware Spatio-Temporal Graph Clustering for Continuous Event-based Eye Tracking

## 1. 서지정보
- **제목**: EyeGraph: Modularity-aware Spatio Temporal Graph Clustering for Continuous Event-based Eye Tracking
- **저자**: Nuwan Bandara, Thivya Kandappu, Argha Sen(IIT Kharagpur), Ila Gokarn(SMART), Archan Misra (Singapore Management University)
- **연도**: 2024
- **학회/저널**: NeurIPS 2024 Track on Datasets and Benchmarks
- **원본 파일명**: `EyeGraph Modularity-aware Spatio Temporal Graph Clustering for Continuous Event-based Eye Tracking.pdf`
- **성격**: 데이터셋(EyeGraph) + **비지도(unsupervised)** 그래프 클러스터링 벤치마크

## 2. 문제정의·배경
- 고주파·미세 안구운동(saccade, microsaccade, 가속도 24,000°/s²)은 RGB로 포착 곤란 → 이벤트 카메라가 대안.
- 기존 이벤트 기반 시선추적의 4대 한계: (a) **실세계 대표성 부족 데이터셋**(통제된 실험실, 머리 고정), (b) **라벨 희소성**(EV-Eye는 PoG 100Hz만), (c) **dense 2D framed 표현**이 기하·공간·시간 관계를 부적절 포착, (d) **RGB-guided 추론**(고정간격 RGB가 비동기 이벤트와 불일치).
- → 이벤트 스트림을 **동적·시간진화 그래프**로 처리 + **비지도 modularity-aware 그래프 클러스터링**으로 동공 추적.

## 3. 핵심 기여
- **EyeGraph 데이터셋**: 40명, 웨어러블 이벤트 카메라(DAVIS346) HMD, **in-the-wild 모사**(조명 변화 348→24 Lux, 머리/몸 자유이동) 단안(monocular) near-eye. 기존 데이터셋(EBV-Eye/EV-Eye/3ET/3ET+) 대비 유일하게 head-movement·조명변화·사용자 이동 반영.
- **동적 임계 기반 엣지 구성** + **Hawkes 프로세스 기반 엣지 특징**으로 시간진화 spatio-temporal 그래프 구성.
- **VGAE + modularity 최대화**의 비지도 topology-aware 클러스터링 + **tube-model RANSAC**으로 동공 중심선 추정.
- 문헌 최초의 **비지도 이벤트 기반 시선추적** 벤치마크. supervised에 근접하면서 라벨링 노력 불필요.

## 4. 방법론
*비지도 그래프 학습 접근으로, 앞선 supervised CNN/SSM 계열과 패러다임이 근본적으로 다름.*

### 이벤트 → 동적 그래프 표현
- 이벤트 $e_i=(x_i,y_i,p_i,t_i)$를 그래프 노드로. 노드 위치 $p_{v_i}=[\lambda_1 t_i,\lambda_2 x_i,\lambda_3 y_i]\in\mathbb{R}^3$, 노드 특징 $[\lambda_1 t_i,\lambda_2 x_i,\lambda_3 y_i,p_i]\in\mathbb{R}^4$ (λ는 정규화 계수). → 희소·비동기 spatio-temporal point cloud.
- Hebbian 원리("함께 발화하는 픽셀은 연결") 기반 엣지 구성.

### 동적 임계 기반 엣지 구성 (핵심)
- 고정반경/kNN은 이벤트 비전 특성에 부적합(global/local 구조 손실, blur). → **동적 임계 반경**.
- 공간-시간 거리 분포를 **GMM(가우시안 혼합)** 으로 적합: $F(c,P)=f^2_{k}(k, f^1[\|p_{v_i}-p_{v_j}\|])$. $f^1$은 상삼각행렬 추출, $f^2$는 GMM 적합 $\mathcal{N}(\mu_a,\sigma_a)=\sum_a\pi_a\mathcal{N}(x|\mu_a,\sigma_a)$, BIC 최소화. 최대 클러스터 $c=5$(동공/홍채/상하 눈꺼풀·속눈썹/눈썹).
- 동적 임계: $\xi_1=\lambda\times\min(\mu_i-3\sigma_i)$. 공간 엣지: $\|p_{v_i}-p_{v_j}\|\le\xi_1$ (동일 시각 ±δ, 노드 degree ≤ N 제약).
- 시간 엣지(directed, 진화 반영): $\{v_i,v_j\}\in E\ \forall t_i<t_j$ iff $\lambda_1(t_j-t_i)\le\xi_2$ 이고 공간 이웃 조건.
- **엣지 특징**: Hawkes 프로세스 기반 attribution(과거 이벤트의 영향을 decay factor로 누적).

### 비지도 Topological 클러스터링
- 그래프 분할 함수 $\Upsilon: V\to\{1,...,c\}$, 그래프 인코더 $f_\theta(G)=Z\in\mathbb{R}^{n\times d_h}$. 목표: 눈 해부학 topology + 시간진화 보존하며 modularity 최대화.
- **VGAE(Variational Graph Autoencoder)**: GCN 인코더 message passing $X^{(l+1)}=\eta(\tilde A X^{(l)}W^{(l)})$. 인코더 $q(Z|X,A)=\prod_i\mathcal{N}(z_i|\mu_i,\text{diag}(\sigma_i^2))$, 디코더 $p(A|Z)=\prod_{ij}p(A_{ij}|z_i,z_j)$, $p(A_{ij}=1|z_i,z_j)=\eta(z_i^\top z_j)$.
- **결합 손실**: $L=\gamma_1\mathbb{E}_{q}[\log p(A|Z)]-\gamma_2\frac{Tr(BXX^\top)}{2m}-\gamma_3 KL[q(Z|X,A)\|p(Z)]$. 1항=엣지 재구성(topology 유도), 2항=modularity 행렬 B 기반 클러스터 멤버십(모듈성 최대화), 3항=KL 정규화.
- **동공 좌표 추정**: 홍채+동공이 spatio-temporal에서 tube(원기둥+토러스)형 궤적 → **tube-model RANSAC**으로 중심선 추정.

## 5. 실험
- **데이터셋**: EyeGraph(40명) + EBV-Eye(평가) + 3ET+(정확도 비교, 100Hz 동공 GT).
- **메트릭**: 클러스터 품질 - Silhouette(SC↑), Davies-Bouldin(DB↓), Modularity(Mo↑), Conductance(Cd↓). 동공좌표 - p-accuracy(p1/p5/p10↑), Mean Euclidean(l2↓), Manhattan(l1↓).
- **그래프 구성(Table 2)**: Ours가 EyeGraph SC 0.66/Mo 69.34/Cd 11.30, EBV-Eye Mo 75.70/Cd 5.07로 대부분 우수.
- **클러스터링(Table 3)**: Ours Mo 최고(EyeGraph 69.34, EBV-Eye 75.70). DMoN/DGI 등 GNN 기반과 경쟁.
- **동공 좌표 추정(Table 4, 3ET+)**:
  | 분류 | 방법 | p10↑ | p5↑ | p1↑ | l2↓ | l1↓ |
  |---|---|---|---|---|---|---|
  | Supervised | MambaPupil | 99.42 | 97.05 | 33.75 | 1.67 | 2.11 |
  | Supervised | bigBrains | 99.00 | 97.79 | 45.50 | 1.44 | 1.82 |
  | Unsupervised | DMoN | 77.45 | 75.07 | 10.20 | 8.36 | 9.66 |
  | **Unsupervised** | **Ours** | **91.45** | **89.22** | 28.34 | 3.88 | 4.24 |
  - Ours는 supervised 최고 대비 p10 ~8%↓이나 **비지도 중 최고**(라벨 불필요). DMoN에 클러스터링 결합 시 p10 +14%.
- **자체 EyeGraph(조명/이동 변화)**: Ours p10 93.67, MambaPupil(supervised) 95.88 → 라벨 없이 2% 차로 근접·강건.

## 6. 강점/한계
- **강점**: **라벨 불필요(비지도)** + in-the-wild(조명/이동) 강건. 이벤트의 희소·비동기성을 그래프로 자연 보존(dense framing 회피). 새 대규모 실세계 데이터셋 기여.
- **한계**: 단안·단일 눈 → saccade/fixation 등 양안 시선특징 직접 활용 불가(conjugate 운동 가정). GMM 적합·VGAE·RANSAC 파이프라인은 **연산 무겁고 비결정적**(latency 보고 없음). 통제된 시각자극(자연 cue 아님). FPGA/임베디드 실측 전무.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)
- **HW 가속 직접 후보로는 부적합(추정)**: GMM 적합(BIC), VGAE message passing, RANSAC 반복은 동적·비결정적 제어흐름이 많아 FPGA 저지연 데이터패스에 매핑이 어렵고, latency·전력 보고도 없음. 우리 "저지연 가속" 목표와는 거리가 있음.
- **데이터셋으로서의 가치(확인)**: EyeGraph는 **조명변화·머리이동을 반영한 유일한 이벤트 시선 데이터셋** → 우리 모델의 **robustness 평가/검증셋**으로 활용 가치. 3ET+/EV-Eye 학습 모델의 in-the-wild 일반화 테스트에 적합.
- **비지도 학습 철학(추정)**: 라벨 없이 동공 추적이 supervised 2% 내로 근접 → 우리 시스템도 비라벨 실데이터 보조(self/pseudo-label)로 일반화 보강 가능(FoundationDistill/SynUnlabeled와 동일 방향).
- **이벤트→tube형 궤적 통찰(확인)**: 동공/홍채가 spatio-temporal에서 tube(원기둥) 궤적을 형성한다는 관찰은 우리 입력 표현·트래킹 prior 설계에 참고. 단, RANSAC 자체는 HW보다 후처리(CPU) 적합.
- **baseline 비교 데이터(확인)**: Table 4에 MambaPupil/bigBrains/FreeEVs/GoSparse/Efficient의 3ET+ p-accuracy가 정리되어 있어 **우리 모델 정확도 비교표 작성 시 직접 인용** 가능.

## 8. 근거표기
- 구조/수식(식1~3)/Table 1~4/Figure 1~4 수치는 본문(NeurIPS 2024 D&B, p.1~15)에서 직접 확인.
- "FPGA 부적합·robustness 검증셋 활용" 해석은 **추정**(논문은 비지도 알고리즘 품질 중심, HW 미언급).
- GMM/VGAE/Hawkes/RANSAC의 정확한 하이퍼파라미터(λ_1~3, γ_1~3, ξ_2, α, N, decay factor)는 본문에 일부만 기재, 상세는 supplementary 의존 → **부분 확인**.
