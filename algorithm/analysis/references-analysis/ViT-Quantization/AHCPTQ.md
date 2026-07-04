# AHCPTQ 정밀 분석

> 분석 대상: `REF/ViT-Quantization/AHCPTQ`
> 분석 방식: 자체 양자화 핵심 소스(`ahcptq/`, `exp/`, `README.md`, `environment.sh`) 함수/클래스 단위 정독. 외부 프레임워크(mmdetection, projects/instance_segment_anything/ops CUDA 커널)는 이름만 언급.
> 근거 표기 원칙: 모든 핵심 주장은 `파일:라인` 으로 표기. 코드로 확인한 사실과 "추정"/"확인 불가"를 구분.

---

## 1. 개요

### 1.1 정식 명칭 / 원논문 (README로 확인)

- **정식 명칭**: **AHCPTQ — Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model** (`README.md:1`).
- **학회/연도**: **ICCV 2025** (IEEE/CVF International Conference on Computer Vision), pp. 22383–22392 (`README.md:95,123-129`).
- **저자**: Wenlun Zhang, Yunshan Zhong, Shimpei Ando, Kentaro Yoshioka (`README.md:125`).
- **링크**: ICCV 2025 open-access 페이지 (`README.md:95`).
- **기반(포크)**: **PTQ4SAM** (chengtao-lv/PTQ4SAM). README가 명시적으로 PTQ4SAM 환경을 따르고 PTQ4SAM 위에 구축했다고 밝힘 (`README.md:7,133`). PTQ4SAM 자체는 QDrop/AdaRound 계열 PTQ 프레임워크의 SAM 확장이다(코드 내 `qdrop.*` 잔존 import, `quant_coco.py:22-27`로 확인).

### 1.2 핵심 아이디어 (README + 코드로 확인)

AHCPTQ가 지목한 SAM 양자화의 두 가지 난점(`README.md:97`):
1. **post-GELU 활성의 heavy-tailed / skewed 분포** — 작은 값이 조밀하고 큰 값이 희소.
2. **linear projection 활성의 채널 간 분포 편차(inter-channel variation)**.

이를 해결하는 두 고유 기법:
- **HLUQ (Hybrid Log-Uniform Quantization)**: post-GELU 활성에 대해 **조밀한 작은 값 영역은 log2 양자화**, **희소한 큰 값 영역은 uniform 양자화**로 처리하는 하이브리드 양자화. 즉 "Adaptive Hybrid"의 hybrid는 **하나의 텐서를 값 크기에 따라 log2 + uniform 두 양자화 격자로 분할**하는 것이다 (`util_quant.py:80-101`, 관찰자 `observer.py:654-704`, 양자화기 `quantized_module.py:674-724`). FPGA에서 log2 부분은 shift로 구현 가능해 hardware-compatible.
- **CAG (Channel-Aware Grouping)**: inter-channel variation 완화를 위해 분포가 유사한 활성 채널들을 **점진적(progressive) k-means 클러스터링**으로 묶어 같은 양자화 파라미터(scale/zero-point)를 공유 (`quantized_module.py:598-671`, 점진적 그룹 감축 `recon.py:336-372`).

`HLUQ + CAG` 조합으로 W4A4에서 SAM-L+DINO instance segmentation 36.6% mAP, FPGA 구현에서 FP 대비 7.89배 속도/8.64배 에너지효율 주장 (`README.md:97`).

### 1.3 "Adaptive Hybrid"의 정체 규명 (코드 검증)

이름의 "Adaptive Hybrid"가 코드상 무엇인지 정확히 분해하면:

- **Hybrid (HLUQ)**: 단일 텐서를 **threshold `scale_log` 기준으로 log2 영역 / uniform 영역으로 분기**(`util_quant.py:86` `mask_log = (xq <= scale_log)`). 두 양자화 방식의 격자 비율은 `grid_rate`로 배분(`util_quant.py:82-83`).
- **Adaptive(1) — HLUQ의 적응적 파라미터 탐색**: `range_rate`(log영역이 차지하는 동적범위 비율)와 `grid_rate`(log영역이 차지하는 양자화 레벨 비율)를 후보 공간에서 grid search 하여 **재구성 손실이 최소인 조합을 적응적으로 선택**(`observer.py:658-659,691-703`, 선택 `quantized_module.py:685-704`).
- **Adaptive(2) — softmax용 적응적 granularity(AGQ, PTQ4SAM 유래지만 AHCPTQ가 log양자화로 강화)**: post-softmax 값에 대해 log2 base `tau`를 `{1,2,4}` 후보 중 손실 최소로 선택(`observer.py:483-555`, `fake_quant.py:541-595`). 즉 "adaptive granularity"는 **log2 양자화의 밑(τ)을 채널/레이어별로 적응 선택**하는 것.
- **Adaptive(3) — CAG의 적응적 그룹 수 감축**: 채널 그룹 수를 `group*8 → *4 → *2 → group`으로 학습 진행에 따라 점진적으로 줄이며 군집(`recon.py:336-347`).

> 정리: AHCPTQ의 "Adaptive Hybrid"는 (a) **log2+uniform 두 격자의 텐서 내 영역 분할(hybrid)**과 (b) **그 분할 파라미터·log 밑·채널 그룹 수를 손실 기반으로 적응 선택(adaptive)** 하는 것을 묶은 표현이다. quantizer를 통째로 전환하는 방식이 아니라, 한 텐서 안에서 영역별로 다른 양자화 격자를 쓰는 방식임이 핵심.

---

## 2. 디렉토리 구조

```
AHCPTQ/
├── README.md                         # 논문 명칭/기법/명령어/PTQ4SAM 버그 지적 (분석함)
├── environment.sh                    # 설치 스크립트 (분석함)
├── ahcptq/                           # ★ 자체 양자화 핵심 (전부 정독)
│   ├── quantization/
│   │   ├── observer.py               # 모든 Observer + HybridParamObserver + Log/Sign Observer
│   │   ├── fake_quant.py             # 모든 FakeQuantize + HybridQuantize + AdaptiveGranularity + GroupLSQ
│   │   ├── util_quant.py             # fake_quant 커널: uniform/log/hybrid + STE
│   │   ├── quantized_module.py       # QLinear/QConv/Quantizer factory + PreQuantizedLayer + 설정 라우팅
│   │   ├── quantized_module_matmul.py# (레거시; tools.modifier.MatMul 의존 — SAM 경로 미사용으로 추정)
│   │   ├── state.py                  # observer/fake_quant enable/disable 토글
│   │   └── __init__.py               # (빈 파일)
│   ├── model/
│   │   └── quant_model.py            # ★ SAM image encoder/mask decoder 양자화 래핑 (HLUQ/CAG/BIG/AGQ 배선)
│   └── solver/
│       ├── test_quant.py             # ★ 평가 엔트리 (calibrate→recon→eval, dropout 버그 픽스)
│       ├── recon.py                  # 블록 단위 재구성(QDrop+AdaRound), CAG 점진 군집
│       ├── utils.py                  # config 파싱, calibration 데이터 로더, hook
│       └── quant_coco.py             # (레거시 detector backbone 양자화; qdrop.* import — SAM 미사용)
├── exp/
│   ├── config44.yaml / config55.yaml / config66.yaml   # W4A4 / W5A5 / W6A6 양자화 설정
├── projects/
│   ├── instance_segment_anything/    # SAM + detector 통합 (외부; image_encoder.py 등은 원본 SAM)
│   │   └── ops/                      # CUDA 커널 (외부 — 이름만)
│   └── configs/                      # detector×SAM 조합 mmdet config (yolox/hdetr/faster_rcnn/focalnet_dino)
├── mmdetection/                      # 외부 프레임워크 (이름만; 분석 제외)
└── ckpt/                             # 체크포인트 폴더 (이름만)
```

- **PTQ4SAM 계열 포크임**: `quant_model.py`가 `ptq4sam_config.BIG`/`ptq4sam_config.AGQ`를 그대로 받아 사용(`quant_model.py:271-287`), config에 `ptq4sam:` 섹션 존재(`config44.yaml:28-33`), `quant_coco.py`의 `qdrop.*` import 잔존(`quant_coco.py:22-27`)으로 확인.
- **외부 분석 제외**: `mmdetection/`(detection 프레임워크), `projects/instance_segment_anything/ops`(CUDA), `projects/.../segment_anything/modeling/`(원본 SAM 모델 정의 — import 대상으로만 참조), `.git`, `ckpt`.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 양자화 수치 커널 — `util_quant.py`

기본 STE 라운딩과 세 종류의 fake-quant 커널이 정의된다.

- `round_ste` (`util_quant.py:4-8`): `(x.round()-x).detach()+x` — straight-through estimator.
- **uniform affine** `fake_quantize_per_tensor_affine` (`util_quant.py:11-15`): 표준 `clamp(round(x/s)+zp)` 후 dequant. per-channel 버전(`:28-36`), learnable(LSQ) 버전들(`:39-77`)도 존재.
- **log2 양자화** `fake_logquantize_per_tensor_affine` (`util_quant.py:17-26`): 
  - `x_int = round(-log2(x/scale) * tau)` (`:22`) — 양수 값을 log2 스케일로 양자화, `tau`가 밑 세분도(2^(1/tau) 격자).
  - 레벨 초과분(`x_int >= levels`)은 0으로 매핑(`:23-24`) — post-softmax의 매우 작은 확률을 0으로 처리(softmax_mask).
- **하이브리드 log+uniform** `fake_hybrid_quantize_per_tensor_affine` (`util_quant.py:80-101`) — **HLUQ의 핵심 커널**:
  1. `levels_log = levels*grid_rate`, `levels_uni = levels - levels_log` — 전체 레벨을 두 격자로 분배(`:82-83`).
  2. `xq = x - fp_min` 으로 최소값 시프트(`:85`).
  3. `mask_log = (xq <= scale_log)` — **작은 값 영역은 log2 양자화**, 나머지는 uniform(`:86-94`).
  4. log 영역: `round(-log2(xq/scale_log))` → `scale_log * 2^(-x)` 복원, 초과분 0(`:90-94`).
  5. uniform 영역: `round((xq-scale_log)/scale_uni)` → 균등 dequant 후 `+scale_log`(`:96-98`).
  6. `xq + fp_min` 으로 원위치 복원(`:100`).
  - **이것이 "Hybrid Log-Uniform"의 직접 구현**이다. 한 텐서를 threshold(`scale_log`)로 분할해 영역별로 다른 격자를 적용.

### 3.2 Observer — `observer.py`

기본 클래스 `ObserverBase`(`observer.py:22-70`)는 bit/symmetric/ch_axis를 받고 `calculate_qparams`로 scale/zp 산출:
- **symmetric**: `scale = max(|min|,max)/((qmax-qmin)/2)`, zp=0 (`:61-64`).
- **asymmetric(affine)**: `scale=(max-min)/(qmax-qmin)`, `zp=qmin-round(min/scale)` clamp (`:65-69`).
- bit별 `quant_min/max`: symmetric은 `[-2^(b-1), 2^(b-1)-1]`, asym은 `[0, 2^b-1]` (`:30-35`).

주요 Observer 종류 (모두 `quantized_module.py:6-18` `ObserverDict`에 등록):
- `MinMaxObserver`/`MinMaxObserver2` (`:73-115`): 전역/현재 min-max.
- `AvgMinMaxObserver` (`:118-143`): 캘리브레이션 배치 평균 min-max. **활성 기본 observer**(config의 `a_qconfig.observer: AvgMinMaxObserver`, `config44.yaml:3`).
- `AvgMinMaxGroupObserver` (`:146-181`): `ch_axis='det'`로 마지막 축을 동적 채널축으로 잡음 — **CAG용**.
- `MSEObserver`/`AvgMSEObserver` (`:211-330`): grid search로 MSE 최소 clip 범위 탐색(1D/2D search, lp_loss p=2.4). **가중치 기본 observer**(`w_qconfig.observer: MSEObserver`, `config44.yaml:9`).
- `MSEFastObserver`/`AvgMSEFastObserver` (`:333-481`): scipy `minimize_scalar` golden-section search로 MSE clip 범위 고속 탐색.
- **`LogAvgMSEFastObserver`** (`:483-561`) — **AGQ(softmax) 전용 log observer**:
  - `taus=[1,2,4]`(`:490`, `2**i for i in range(3)`).
  - 각 τ에 대해 golden-section으로 최적 log scale을 찾고, **손실은 `lp_loss(x_q @ value, x @ value)`** 로 평가(`:494-505`) — 즉 단순 양자화 오차가 아니라 **post-softmax 양자화가 `attn @ v` 출력에 미치는 영향**으로 측정(value=v 주입). τ별 scale·error를 저장(`best_tau_scales`, `tau_errors`).
- **`SignAvgMSEFastObserver`** (`:563-584`) — BIG(bimodal) 전용. channel sign을 곱해 정렬 후 MSE.
- `PCTObserver` (`:586-651`): percentile 기반 clip(0.99~0.99999).
- **`HybridParamObserver`** (`:654-704`) — **HLUQ 파라미터 탐색 observer**:
  - 탐색 공간: `range_rate_space=[0.1,0.3,0.5]`(log영역 동적범위 비율), `grid_rate_space=[1/8,1/4,1/2]`(log영역 레벨 비율) (`:658-659`).
  - `forward(input, weight, bias)` — 활성뿐 아니라 **후속 linear의 weight/bias까지 받아** 출력 기준 손실 측정(`:663-689`).
  - `loss_fx`: hybrid 양자화 후 `lp_loss(F.linear(xq,w,b), F.linear(x,w,b))` — **레이어 출력 재구성 손실**로 평가(`:685-689`).
  - `solve_range_loss`: `scale_log = range*range_rate`, `scale_uni = range*(1-range_rate)/((qmax-qmin)*(1-grid_rate))` 로 후보 scale 산출(`:699-703`). 9개(3×3) 조합 손실을 `loss_list`에 누적(`:691-697`).

### 3.3 FakeQuantize 모듈 — `fake_quant.py`

기본 `QuantizeBase`(`fake_quant.py:22-111`): observer 보유, `observer_enabled`/`fake_quant_enabled` 토글, `scale`/`zero_point` 버퍼 직렬화, `drop_prob`(QDrop) 보유.

주요 양자화기 (모두 `quantized_module.py:20-30` `FakeQuantizeDict` 등록):
- `FixedFakeQuantize` (`:114-149`): observer로 scale/zp 고정, QDrop drop_prob 지원.
- `LSQFakeQuantize` (`:152-198`): scale을 학습가능 파라미터로(LSQ). **활성 기본 양자화기**(`a_qconfig.quantizer: LSQFakeQuantize`, `config44.yaml:2`).
- **`LSQSignFakeQuantize`** (`:201-272`) — **BIG(bimodal) 처리**: `judge_bimodal`(`:215-227`)가 KDE+`find_peaks`로 peak가 2개면 bimodal 판정, 채널별 평균 부호 `sign`을 저장. 부호 정렬로 bimodal을 단봉화. (PTQ4SAM의 BIG에 해당.)
- `LSQPlusFakeQuantize` (`:276-322`): scale+zero_point 모두 학습(LSQ+).
- `LSQPlusSignFakeQuantize` (`:324-443`): two-peak 판정(`_judge_two_peak`, 비대칭율 `gamma=0.8` 기준 `:352-378`) 후 부호 정렬 양자화.
- `AdaRoundFakeQuantize` (`:447-537`) — **가중치 기본 양자화기**(`w_qconfig.quantizer: AdaRoundFakeQuantize`, `config44.yaml:8`): learned hard-sigmoid 라운딩(AdaRound), `rectified_sigmoid`로 up/down 라운딩 학습(`:481-503`).
- **`AdaptiveGranularityQuantize`** (`:541-595`) — **AGQ(post-softmax log 양자화)**:
  - observer(`LogAvgMSEFastObserver`)의 `tau_errors`에서 최소 오차 인덱스로 `tau`와 `scale`을 선택(`init_quantization_scale`, `:576-583`) — **적응적 granularity 선택**.
  - `quantize`: `round(-log2(x/scale)*tau)` → `scale*2^(-x/tau)`, 레벨 초과는 0(`:585-595`) — post-softmax 전용 log 양자화.
- **`GroupLSQFakeQuantize`** (`:598-671`) — **CAG(채널 그룹화)**:
  - `group_channel(num_groups)`: 채널별 (scale, zero_point) 2D 벡터를 **k-means(`sklearn.cluster.KMeans`)로 군집**, 군집 중심을 공유 파라미터로(`:616-634`). `labels`로 각 채널→그룹 매핑.
  - `map_vector`: 그룹 파라미터를 채널 수만큼 broadcast(`:610-614`).
  - forward는 per-channel LSQ를 그룹 공유 scale로 수행(`:636-671`).
- **`HybridQuantize`** (`:674-724`) — **HLUQ 양자화기**:
  - `init_quantization`(`:685-704`): `HybridParamObserver.loss_list`의 9조합 평균 손실을 집계해 **최소 손실 (range_rate, grid_rate) 선택**, `scale_log`/`scale_uni`/`grid_rate`/`fp_min` 확정. scale은 학습 파라미터로 등록(재구성 시 미세조정).
  - `quantize_activation`: `fake_hybrid_quantize_per_tensor_affine` 호출(`:706-709`).

### 3.4 양자화 모듈 팩토리 / 설정 라우팅 — `quantized_module.py`

- `ObserverDict`/`FakeQuantizeDict` (`:6-30`): 이름→클래스 매핑.
- **`update_specialized_quantizer_config`** (`:32-43`): `'group'`→`GroupLSQFakeQuantize`+`AvgMinMaxGroupObserver`, `'hybrid'`→`HybridQuantize`+`HybridParamObserver`로 config 치환. **CAG/HLUQ 라우팅의 분기점**.
- `QLinear`/`QConv2d`/`QEmbedding` (`:70-136`): weight fake-quant 내장. `QLinear.forward`는 optional `gamma`로 BIG의 채널 부호 융합(`:102-108`).
- `Quantizer` 팩토리 (`:180-196`): module이 None이면 활성 양자화기, weight 모듈이면 가중치 양자화 래핑.
- **`PreQuantizedLayer`** (`:222-250`) — 입력측 활성 양자화 레이어. `type` 인자로 분기:
  - `'group'` → CAG config 적용 + `detect_ch_axis=True`(`:229-231`).
  - `'hybrid'` → HLUQ config 적용, 후속 module의 weight/bias를 observer에 주입(`:240-242`) — HybridParamObserver가 출력 손실을 계산하기 위함.
- `QuantizedMatMul` (`:252-266`): a@b 두 입력을 각각 양자화 (디코더 attention의 Q@K, attn@V 경로용으로 설계, 단 SAM 경로는 `quant_model.py`에서 inline 처리).

### 3.5 SAM 모델 양자화 래핑 — `model/quant_model.py` (가장 중요)

SAM의 image encoder(ViT)와 mask decoder를 양자화 블록으로 교체한다. 핵심은 **ViT attention의 어떤 텐서에 어떤 기법을 거느냐**이다.

- **`update_specialized_quantizer_config`** (`:18-29`): `'softmax'`→`AdaptiveGranularityQuantize`+`LogAvgMSEFastObserver`(AGQ), `'bimodal'`→`LSQSignFakeQuantize`+`SignAvgMSEFastObserver`(BIG).
- **`QuantImageEncoderOurViT`** (`:340-365`): patch_embed/pos_embed는 비양자화, 각 Block을 `QunatEncoderOurBlock`으로 교체, neck도 양자화.
- **`QuantEncoderOurAttentionBlock`** (`:396-448`) — **encoder ViT self-attention 양자화**:
  - `qkv`/`proj` linear projection을 `PreQuantizedLayer`로 래핑하되, `ahcptq_config.cag`면 **qkv에 `proj_type='group'`(CAG) 적용**(`:404-408`).
  - q/k/v 각각 post-act 양자화기(`:423-425`).
  - attention: `(q*scale) @ k.T` → (rel_pos 가산) → **softmax 후 `softmax_post_act_fake_quantize`** (`:438-443`).
  - `AGQ`면 softmax 양자화기가 AGQ(log) config로 교체되고, **`value=v`를 주입**해 출력 기반 손실로 τ/scale 결정(`:416-421,443`).
- **`QuantEncoderMLPBlock`** (`:102-118`) — **encoder MLP(post-GELU) 양자화**:
  - `lin1`은 `cag`면 `'group'`(CAG), `lin2`는 `hluq`면 **`'hybrid'`(HLUQ)** (`:105-114`). **즉 post-GELU 활성(=lin2 입력)에 HLUQ가 적용**됨 — 논문 주장(heavy-tailed post-GELU)과 정확히 일치.
- **`QuantDecoderOurAttentionBlock`** (`:253-337`) — **decoder cross/self-attention**:
  - q/k/v proj에 CAG(`proj_type='group'`) 적용(`:261-267`).
  - **`k_post_act`에 BIG(bimodal) config**(`:281`, `sign_a_config`), **`softmax_post_act`에 AGQ**(`:279`).
  - attention: `q@k.T / sqrt(d)` → softmax → `softmax_post_act(attn, value=v)` → `attn@v` (`:316-323`).
  - `bimodal_adjust`(`:329-337`): bimodal로 판정된 k의 부호를 q_proj/k_proj weight·bias에 융합(reparametrization) — PTQ4SAM BIG의 부호 정렬 흡수.
- **`QuantDecoderOurTwoWayAttentionBlock`** (`:196-250`): self_attn / cross_attn(token↔image) / MLP를 각각 양자화 블록으로. MLP의 lin2에도 HLUQ 적용(`QuantDecoderMLPBlock`, `:121-137`).
- `specials` 매핑(`:450-454`): `TwoWayAttentionBlock`/`Attention`/`ImageEncoderViT` → 양자화판.
- `bimodal_adjust(model, logger)` (`:456-464`): `token_to_image` cross-attn 블록의 bimodal 정렬을 일괄 적용.

> **요약 (어떤 텐서에 무엇이 걸리는가)**:
> | 위치 | 기법 | 근거 |
> |---|---|---|
> | encoder/decoder linear proj 활성 (qkv, q/k/v proj, MLP lin1) | **CAG (그룹 LSQ)** | `quant_model.py:404-408,261-267,105` |
> | MLP lin2 입력 (post-GELU 활성) | **HLUQ (log+uniform 하이브리드)** | `quant_model.py:109-114,128-133` |
> | post-softmax 확률 | **AGQ (적응 log2, value-aware)** | `quant_model.py:416-421,279,443,320` |
> | decoder k 활성 (bimodal) | **BIG (부호 정렬 LSQ-sign)** | `quant_model.py:281,329-337` |
> | 일반 weight | AdaRound (W) / LSQ (A) | `config44.yaml:2-12` |

### 3.6 양자화 상태 토글 — `state.py`

- `enable_calibration_woquantization`(`:6-17`): 지정 타입 quantizer만 observer ON, fake-quant OFF (캘리브레이션).
- `enable_quantization`(`:20-31`): observer OFF, fake-quant ON (추론).
- `disable_all`(`:34-40`): 전부 OFF.
- 이름 매칭(`'act_fake_quant'`/`'weight_fake_quant'`)으로 활성/가중치를 분리 캘리브레이션(`test_quant.py:474,488`).

### 3.7 평가 엔트리 — `solver/test_quant.py`

- `quantize_model`(`:430-464`): `--quant-encoder`면 image_encoder를 `QuantImageEncoderOurViT`로 교체(`:451-452`), mask_decoder는 `replace_module`로 재귀 교체(`:454`). `patch_embed`/`output_upscaling`/`iou_prediction_head`/`output_hypernetworks_mlps`는 양자화 제외(`:434`).
- `calibrate`(`:467-493`): BIG면 먼저 1배치로 bimodal 탐지·정렬(`:471-473`), 활성 observer→가중치 observer 순서로 캘리브레이션.
- `recon_model`(`:496-513`): 블록 단위로 `reconstruction` 호출.
- **PTQ4SAM dropout 버그 픽스**(`:360-362`): 평가 직전 모든 `drop_prob`를 1.0으로 강제 — README가 지적한 핵심 baseline 버그(아래 7.3).

### 3.8 블록 재구성 — `solver/recon.py`

- `save_inp_oup_data`(`:11-47`): forward hook으로 블록 입출력 캐싱(FP/quant 모델 각각).
- `LossFunction`(`:67-126`): QDrop+AdaRound 재구성 손실 = MSE 재구성 + 라운딩 정규화(`temp_decay`로 b annealing).
- `reconstruction`(`:136-359`): 블록별 weight alpha(AdaRound)와 activation scale(LSQ/AGQ/Group/Hybrid)을 Adam으로 최적화. 학습 대상 파라미터 수집 시 **`HybridQuantize`는 `scale_log`,`scale_uni`** 를(`:167-169`), **`GroupLSQFakeQuantize`는 `grouped_scales`** 를(`:165-166`), `AdaptiveGranularityQuantize`는 `scale`을(`:163-164`) 등록.
- **CAG 점진적 군집**(`:336-347`): `cag`일 때 학습 진행 20/40/60/80% 시점에 그룹 수를 `group*8→*4→*2→group`으로 단계적 k-means 재군집(`group_channel`, `:361-372`) — README의 "progressively clustering" 직접 구현.
- `drop_prob`: 재구성 중에는 config값(0.5), 종료 시 1.0 복원(`:358-359`).

### 3.9 레거시/미사용 파일 (확인)

- `quantized_module_matmul.py`: `from tools.modifier import MatMul`(`:7`)에 의존하나 repo 내 `class MatMul` 정의 없음(grep 결과 0건). SAM attention의 matmul은 `quant_model.py`에서 inline 처리되므로 **이 파일은 SAM 경로에서 미사용으로 추정**.
- `quant_coco.py`: `qdrop.*` import(`:22-27`) — PTQ4SAM/QDrop 원본의 detector backbone 양자화 스크립트. SAM 경로(`test_quant.py`)와 무관한 **레거시로 추정**.

---

## 4. 알고리즘 / 수식 (코드에서 유도)

### 4.1 HLUQ (Hybrid Log-Uniform Quantization)

총 레벨 `L = qmax-qmin+1`을 두 격자로 분배 (`util_quant.py:81-83`):
- log 레벨: `L_log = L · grid_rate`
- uniform 레벨: `L_uni = L − L_log`

시프트 `x' = x − fp_min` 후 임계값 `scale_log`로 분기 (`util_quant.py:85-86`):

- **log 영역** (`x' ≤ scale_log`):
  - 양자화: `i = round( −log₂(x'/scale_log) )`, clamp `[0, L_log−1]`
  - 복원: `x̂ = scale_log · 2^(−i)` (`util_quant.py:90-93`)
  - `i ≥ L_log` (너무 작은 값) → `x̂ = 0` (`util_quant.py:91,94`)
- **uniform 영역** (`x' > scale_log`):
  - 양자화: `j = round( (x' − scale_log)/scale_uni )`, clamp `[qmin, L_uni−1]`
  - 복원: `x̂ = j · scale_uni + scale_log` (`util_quant.py:96-98`)
- 최종: `x̂ + fp_min` (`util_quant.py:100`)

scale 산출 (`observer.py:699-703`, `quantized_module.py:701-702`), `range = max − min`:
- `scale_log = range · range_rate`
- `scale_uni = range · (1 − range_rate) / ( (qmax−qmin) · (1 − grid_rate) )`

**적응적 선택 기준**: `(range_rate, grid_rate) ∈ {0.1,0.3,0.5}×{1/8,1/4,1/2}` 9조합 중, 후속 linear 출력 재구성 손실 `‖f(x̂)·W − f(x)·W‖²` 평균 최소 조합 선택 (`observer.py:685-697`, `quantized_module.py:685-695`).

> **하드웨어 의의**: log 영역의 `2^(−i)` 복원은 **시프트 연산**, uniform 영역은 표준 정수 MAC. dense small / sparse large 분리로 저비트에서 해상도 확보.

### 4.2 AGQ (Adaptive Granularity, post-softmax log)

post-softmax 확률 `p`에 대해 (`fake_quant.py:585-595`, `util_quant.py:17-26`):
- `i = round( −log₂(p/scale) · τ )`, clamp `[0, L−1]`
- `p̂ = scale · 2^(−i/τ)`, `i ≥ L` → `p̂ = 0`

**적응적 granularity**: `τ ∈ {1,2,4}` 중, **value-aware 손실** `‖(p̂@v) − (p@v)‖^2.4` 최소 τ를 선택 (`observer.py:494-505,541-555`, 선택 `fake_quant.py:576-583`). τ가 클수록 격자가 조밀(`2^(1/τ)` 간격).

### 4.3 CAG (Channel-Aware Grouping)

채널별 (scale, zp) 벡터에 대해 k-means 군집 (`fake_quant.py:616-634`):
- `argmin Σ_c ‖(s_c, z_c) − μ_{label(c)}‖²`, 군집 수 `G`
- 군집 중심 `μ_g`를 그룹 공유 scale/zp로, 채널은 `label(c)`로 매핑

**점진적 감축**: `G: group×8 → ×4 → ×2 → group`을 재구성 학습 20/40/60/80% 시점에 단계적 재군집 (`recon.py:336-347`). 최종 그룹 수는 config `ahcptq.group=4` (`config44.yaml:15`).

> **하드웨어 의의**: per-channel(채널마다 scale) 대비 G개 그룹만 scale을 가지므로 dequant 비용·메모리 절감. per-tensor의 정확도 손실과 per-channel의 비용 사이 절충.

---

## 5. 학습/평가 파이프라인

### 5.1 데이터셋 / 태스크

- **COCO** instance segmentation (val2017), SAM이 detector(YOLOX/Faster-RCNN/HDETR/DINO)의 box prompt로 마스크 생성 (`README.md:42-65`, `projects/configs/_base_/datasets/coco_instance.py`).
- 평가 지표: `segm` mAP (`test_quant.py:78`).
- 캘리브레이션: COCO train2017에서 `calibrate=32` 샘플(`config44.yaml:17`, `utils.load_calibration` `:99-139`).

### 5.2 실제 명령어 (README 기반)

설치 (`environment.sh`, `README.md:8-40`):
```
conda create -n ahcptq python=3.7 -y
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113
pip install -U openmim && mim install "mmcv-full<2.0.0"
pip install -r requirements.txt
cd projects/instance_segment_anything/ops && python setup.py build install && cd ../../..
cd mmdetection/ && python3 setup.py build develop && cd ..
```

양자화 평가 (`README.md:69-87`):
```
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder
```
- `config44.yaml`=W4A4, `config55.yaml`=W5A5, `config66.yaml`=W6A6 (`exp/`).
- 메모리 부족 시 `keep_gpu: False` + recon.py 일부 주석 처리 안내(`README.md:89-91`).

### 5.3 내부 흐름 (`test_quant.py:main`)

`mmcv Config 로드 → build_detector → quantize_model(encoder=ViT, decoder 교체) → calibrate(BIG bimodal 탐지+observer) → recon_model(QDrop+AdaRound, CAG 점진 군집) → enable_quantization → drop_prob=1 강제 → single/multi_gpu_test → COCO evaluate(segm mAP)`.

### 5.4 양자화 설정 (config44.yaml = W4A4)

- 활성: `LSQFakeQuantize` + `AvgMinMaxObserver`, **bit=4, asymmetric, per-tensor(ch_axis=-1)** (`:1-6`).
- 가중치: `AdaRoundFakeQuantize` + `MSEObserver`, **bit=4, asymmetric, per-channel(ch_axis=0)** (`:7-12`).
- `ahcptq: {cag:True, group:4, hluq:True}` (`:13-16`).
- `ptq4sam: {BIG:True, AGQ:True, global_num:128, peak_distance:32, peak_height:0.01}` (`:28-33`).
- recon: iters=20000, drop_prob=0.5, scale_lr=4e-5, AdaRound `learned_hard_sigmoid` (`:18-27`).

---

## 6. 의존성

- **PyTorch 1.10.2 + cu113, torchvision 0.11.3** (`environment.sh:1`).
- **mmcv-full <2.0.0, mmdet**(소스 빌드) (`environment.sh:2-9`, `README.md:14-39`) — detection 프레임워크(외부).
- **scipy**(`minimize_scalar`, `find_peaks`) — golden-section/peak 탐지 (`observer.py:5`, `fake_quant.py:16`).
- **scikit-learn**(`KMeans`) — CAG 군집 (`fake_quant.py:18`).
- **easydict, pyyaml** — config 파싱 (`utils.py:5,15`).
- **projects/instance_segment_anything/ops** CUDA 커널(외부, 빌드 필요).
- Python 3.7 (`README.md:11`).

---

## 7. 강점 / 한계 / 리스크

### 7.1 강점
- **하드웨어 친화 설계 명시**: log2(=shift) + uniform 하이브리드, 채널 그룹 공유로 dequant 비용 절감 — FPGA 7.89×/8.64× 주장(`README.md:97`). 양자화 기법이 datapath 비용을 직접 고려.
- **출력/value-aware 손실**: HLUQ는 linear 출력 손실(`observer.py:685-689`), AGQ는 `attn@v` 출력 손실(`observer.py:504`)로 파라미터를 정해 단순 텐서 MSE보다 task-aligned.
- **post-GELU·post-softmax·bimodal**을 분리 처리해 SAM 특유 분포에 정밀 대응.
- baseline 버그(dropout)를 솔직히 지적·수정(`README.md:99-116`).

### 7.2 한계
- **HLUQ는 per-tensor 전용**: `HybridParamObserver`가 `assert ch_axis == -1`(`observer.py:657`) — per-channel hybrid 미지원.
- **CAG의 k-means는 학습 중 CPU 군집**(`fake_quant.py:623-624`) — 재구성 시간 비용. 그룹 수는 고정 스케줄(적응적 그룹 수 자동 결정 아님).
- **재구성 비용**: 블록당 20000 iters(`config44.yaml:23`), A6000 48G에서도 HDETR/DINO는 메모리 부족으로 CPU offload 필요(`README.md:89-91`).
- **레거시 코드 잔존**: `quant_coco.py`(qdrop), `quantized_module_matmul.py`(미해결 import) 등 정리 미흡 — 가독성/혼동 리스크.
- **SAM 자체가 무거움**: image encoder ViT-H 등 대형. 직접 임베디드 배포보다는 기법 전이가 현실적.

### 7.3 리스크 / 주의
- README의 **PTQ4SAM dropout 버그**: 평가 시 drop_prob가 1.0으로 복원되지 않아 활성 절반이 FP로 남아 mAP 과대평가됨(`README.md:99-116`). AHCPTQ는 `test_quant.py:360-362`에서 수정. **PTQ4SAM/QDrop 보고 수치 직접 비교 시 주의** 필요(SAQ-SAM README도 동일 버그 인용).
- 일부 observer 클래스에 명백한 코드 결함 존재(예: `AvgSignMinMaxObserver.__init__`가 부모 `AvgMinMaxObserver.__init__`를 잘못 호출, `observer.py:190`) — 사용되지 않는 경로로 추정되나 코드 신뢰성 표지.

---

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

> 전제: 우리 프로젝트는 "ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적"으로 추정. SAM image encoder는 ViT이므로 어텐션 양자화 기법은 전이 가능. (SAM 자체 무게/태스크는 우리와 직접 연관성 낮음.)

### 8.1 직접 전이 가능 (높은 가치)
- **post-softmax log2 양자화(AGQ)**: `attn` 확률은 [0,1]의 heavy-tailed 분포. log2 양자화는 **복원이 시프트(`2^(-i/τ)`)** 라 FPGA에서 곱셈기 없이 attention 재정규화 datapath 구성 가능(`util_quant.py:17-26`). HG-PIPE류 파이프라인 어텐션 모듈에 저비트 적용 시 곱셈 자원 절감. **value-aware τ 선택**(`observer.py:504`)은 오프라인 캘리브레이션이므로 HW 비용 없음.
- **HLUQ의 log 영역 = shift**: post-GELU/FFN 중간 활성에 적용 시 작은 값 영역을 shift로 처리. 단, **텐서 내 영역 분기(`mask_log`)는 런타임 비교+분기 datapath**를 요구 → FPGA에서 두 경로(log/uniform) 병렬 + threshold 비교기 + mux가 추가됨. **제어 복잡도 증가는 trade-off**로 명시 검토 필요.

### 8.2 부분 전이 / 검토 필요
- **CAG(채널 그룹 공유 scale)**: per-channel scale을 G개로 줄여 **dequant LUT/곱셈 자원과 scale 메모리 절감** — FPGA의 가중치/활성 dequant 단계에 유리. 우리 가속기에서 채널 그룹 수 G를 타일 폭/PE 배열과 맞추면 효율적. 단 그룹 매핑(label) 테이블 저장·인덱싱 오버헤드 고려.
- **outlier(heavy-tail) 처리 자체의 중요성**: 저비트(W4A4)에서 정확도 유지의 핵심이 outlier 분리(HLUQ)와 채널 편차(CAG)임을 코드가 입증 — 우리 가속기가 저비트를 노린다면 **유사한 분포 인지 양자화가 필수**. 단순 per-tensor uniform은 ViT FFN/attention에서 큰 정확도 손실 위험.

### 8.3 낮은 직접성 / 비전이
- SAM 전체 파이프라인(detector prompt, mask decoder TwoWayAttention, COCO 평가)은 우리 XR 시선추적 태스크와 무관 — **기법(HLUQ/AGQ/CAG)만 추출**하고 SAM 모델/데이터 파이프라인은 비전이.
- mmdet/CUDA ops 의존은 우리 HW 흐름과 무관.

### 8.4 실험 아이디어 (추정)
- 우리 ViT 백본 어텐션에 **AGQ(post-softmax log2)** 를 단독 적용해 곱셈 자원 vs mAP/정확도 trade-off 측정.
- FFN post-GELU에 **HLUQ vs uniform** 의 FPGA 자원(LUT/DSP)·정확도 비교로 hybrid 분기의 HW 비용 정량화.
- CAG 그룹 수 G를 PE 배열 폭에 정렬했을 때의 자원/정확도 sweep.

---

## 9. 근거 표기 (확인 사실 vs 추정)

### 9.1 코드/README로 확인된 사실
- 정식 명칭/학회/저자/기법(HLUQ·CAG)·기반(PTQ4SAM): `README.md:1,95,97,123-129,133`.
- HLUQ 커널·수식: `util_quant.py:80-101`; observer `observer.py:654-704`; quantizer `quantized_module.py:674-724`.
- AGQ(log2, value-aware τ): `util_quant.py:17-26`, `observer.py:483-561`, `fake_quant.py:541-595`, `quant_model.py:416-421,443,320`.
- CAG(k-means 그룹, 점진 감축): `fake_quant.py:598-671`, `recon.py:336-372`, `observer.py:146-181`.
- BIG(bimodal 부호 정렬): `fake_quant.py:201-272`, `quant_model.py:281,329-337`.
- post-GELU=lin2에 HLUQ, proj에 CAG 배선: `quant_model.py:105-114,128-133,261-267,404-408`.
- W/A 비트·observer·대칭성: `config44.yaml`(W4A4), `config55.yaml`(W5A5), `config66.yaml`(W6A6).
- dropout 버그 픽스: `README.md:99-116`, `test_quant.py:360-362`, `recon.py:358-359`.

### 9.2 추정 / 확인 불가
- `quantized_module_matmul.py`와 `quant_coco.py`가 **SAM 경로 미사용 레거시**라는 판단: import 경로(`tools.modifier.MatMul` 부재, `qdrop.*`)와 엔트리(`test_quant.py`)가 이들을 참조하지 않음에 근거한 **강한 추정** (직접 실행 검증은 안 함).
- FPGA 7.89×/8.64× 수치: README 주장이며 본 repo에 **HDL/HLS 구현 코드는 미포함**(확인 불가, 논문 별도 구현으로 추정).
- "우리 프로젝트" 성격(HG-PIPE 계열 + XR 시선추적): 작업 지시 기반 **추정**. AHCPTQ 코드에는 FPGA/시선추적 관련 산출물 없음.
- 8장 전이 시사점은 **분석자 해석/제안**(추정)이며 코드가 직접 보장하지 않음.
