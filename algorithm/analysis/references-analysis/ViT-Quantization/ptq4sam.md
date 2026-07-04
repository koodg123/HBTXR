# PTQ4SAM: Post-Training Quantization for Segment Anything (CVPR 2024) 정밀 분석

> 분석 대상 repo: `REF/ViT-Quantization/ptq4sam`
> 분석 범위: 자체 양자화 핵심 소스(`ptq4sam/quantization`, `ptq4sam/model`, `ptq4sam/solver`)만 함수/클래스 단위로 정밀 분석.
> 외부 프레임워크(`mmdetection/` 전체, `projects/instance_segment_anything/ops` CUDA 커널, `.pth` 체크포인트)는 이름만 언급.
> 모든 근거는 **파일:라인** 형식으로 표기. 코드로 확인되지 않은 부분은 "추정" / "확인 불가"로 명시.

---

## 1. 개요

- **목적**: SAM(Segment Anything Model)은 대규모 모델이라 메모리/연산 비용이 커서 실제 배포가 어렵다. PTQ4SAM은 SAM 전용 **PTQ(Post-Training Quantization, 사후 양자화)** 프레임워크를 제안한다(README.md:9). 재학습 없이 소수의 calibration 데이터만으로 W6A6, W4A4 같은 저비트로 양자화한다.
- **원논문**: Lv, Chen, Guo, Ding, Liu, "PTQ4SAM: Post-Training Quantization for Segment Anything", CVPR 2024, pp.15941-15951 (README.md:89-95). arXiv 2405.03144.
- **핵심 아이디어 두 가지** (README.md:9):
  1. **BIG (Bimodal Integration)**: SAM의 **post-Key-Linear activation**(Key projection 출력)에서 나타나는 **bimodal(쌍봉) 분포**가 양자화 병목임을 발견. 수학적으로 등가인 **sign 연산**으로 bimodal 분포를 양자화하기 쉬운 unimodal(정규에 가까운) 분포로 **오프라인 변환**한다.
  2. **AGQ (Adaptive Granularity Quantization)**: SAM은 self-attention, two-way cross-attention 등 다양한 어텐션을 가져 **post-Softmax 분포 편차**가 크다. 따라서 Softmax 출력에 대해 **최적 power-of-two base(τ)를 탐색**하는 적응적 granularity log 양자화를 적용한다(하드웨어 친화적).
- 코드 기반: BIG/AGQ는 config 플래그(`ptq4sam.BIG`, `ptq4sam.AGQ`)로 on/off되며(exp/config66.yaml:24-26), 두 기법 모두 기본 활성화(`True`).
- 기반 코드: **QDrop**(reconstruction/drop) + **Prompt-Segment-Anything**(SAM 추론 래퍼) 위에 구축(README.md:100). `qdrop` 네임스페이스 잔재가 일부 파일에 남아 있다(quant_coco.py:22-27).

---

## 2. 디렉토리 구조

### 2.1 자체 양자화 핵심 소스 (정밀 분석 대상)

```
ptq4sam/
├── README.md                         # 개요/설치/실행/인용
├── environment.sh                    # 설치 스크립트 (torch 1.10.2+cu113 등)
├── requirements.txt                  # mmdetection requirements 참조
├── exp/
│   ├── config66.yaml                 # W6A6 양자화 설정 (BIG/AGQ on)
│   └── config44.yaml                 # W4A4 양자화 설정 (BIG/AGQ on)
└── ptq4sam/
    ├── quantization/
    │   ├── __init__.py               # (빈 파일)
    │   ├── observer.py               # Observer/scale·zp 계산/bimodal·log observer 핵심
    │   ├── fake_quant.py             # FakeQuantize 군 + BIG sign + AGQ log quant
    │   ├── util_quant.py             # affine/log fake-quant 커널, STE
    │   ├── quantized_module.py       # QLinear/QConv/QMatMul, Quantizer 팩토리
    │   ├── quantized_module_matmul.py# (구버전/보조) MatMul 양자화 모듈
    │   └── state.py                  # observer/fake-quant enable·disable 상태 제어
    ├── model/
    │   └── quant_model.py            # SAM(ViT encoder + mask decoder) 양자화 래핑, BIG/AGQ 적용 지점
    └── solver/
        ├── test_quant.py             # ★ PTQ4SAM 실제 평가 엔트리
        ├── quant_coco.py            # (구) QDrop CNN detector 엔트리 — qdrop 네임스페이스, SAM 미사용
        ├── recon.py                  # block-wise reconstruction (AdaRound/LSQ + QDrop)
        └── utils.py                  # config 파싱, calibration 데이터 로딩, hook
```

### 2.2 외부 프레임워크 (이름만, 내부 분석 제외)

- `mmdetection/` : OpenMMLab 객체검출 프레임워크 전체. detector(YOLOX, Faster R-CNN, DETR, DINO) 빌드/평가/COCO 메트릭 담당. 본 분석 제외.
- `projects/instance_segment_anything/` : Prompt-Segment-Anything 기반 SAM 래퍼. 그중 `ops/` CUDA 커널(deformable attention 등)은 이름만 언급(README.md:36 컴파일 단계). 단, `models/segment_anything/modeling/`의 `image_encoder.py`(ViT Attention/Block), `transformer.py`(TwoWayAttentionBlock/Attention), `common.py`(MLPBlock)는 양자화 래핑이 직접 참조하므로 "어텐션 구조 연결"에서 인터페이스만 언급(quant_model.py:8-15).
- `.pth` 체크포인트(SAM-B/L/H, detector들): 이름만(README.md:62-68).
- `mmdetection/requirements.txt`, `mmdetection/.pre-commit-config.yaml`: 외부.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 양자화 방식의 큰 그림 (PTQ 파이프라인)

전체 흐름은 `test_quant.py:main()`에 있다:
1. mmdet으로 detector + SAM predictor 모델 빌드(test_quant.py:316).
2. `quantize_model()`으로 SAM의 image_encoder(옵션) + mask_decoder를 양자화 모듈로 교체(test_quant.py:342, 422-456).
3. `fp_model = copy.deepcopy(model)`로 FP 기준 모델 보관(test_quant.py:345).
4. `calibrate()`로 observer 통계 수집 + BIG 적용(test_quant.py:347, 460-485).
5. (옵션) `recon_model()`로 block-wise reconstruction(AdaRound weight + LSQ activation scale 튜닝)(test_quant.py:349-352).
6. `enable_quantization()`로 fake-quant 켜고 평가(test_quant.py:354).

→ **재학습 없는 PTQ**이며, weight는 AdaRound, activation은 LSQ scale 미세조정(reconstruction)으로 보강하는 QDrop 계열 구조.

### 3.2 observer.py — Observer / scale·zp / bimodal·log 통계

**ObserverBase** (observer.py:21-69)
- 비트 폭, symmetric 여부, ch_axis(채널 축, -1=per-tensor)로 quant 범위 설정.
- symmetric일 때 `quant_min=-2^(bit-1)`, `quant_max=2^(bit-1)-1`; asymmetric일 때 `0 ~ 2^bit-1`(observer.py:29-34).
- `calculate_qparams(min_val, max_val)` (observer.py:50-69): scale/zero-point 계산 핵심.
  - symmetric: `scale = max(|min|,max) / ((qmax-qmin)/2)`, zp=0 (observer.py:60-63).
  - asymmetric: `scale = (max-min)/(qmax-qmin)`, `zp = qmin - round(min/scale)` 후 clamp (observer.py:64-68).
- `_transform_to_ch_axis` (observer.py:8-18): per-channel 통계를 위해 채널축을 0번으로 permute 후 flatten.

**Observer 종류** (observer.py 전체, dict는 quantized_module.py:6-16):
- `MinMaxObserver` (observer.py:72-91): 전체 calibration의 running min/max.
- `MinMaxObserver2` (observer.py:94-114): 현재 배치 min/max 반환(누적 안 함).
- `AvgMinMaxObserver` (observer.py:117-142): min/max의 **이동평균**. config66/44의 activation observer로 사용(exp/config66.yaml:3).
- `AvgSignMinMaxObserver` (observer.py:144-169): (버그성) `super().__init__`에서 `AvgMinMaxObserver`를 호출 — **확인된 버그/미사용 추정**.
- `MSEObserver` (observer.py:171-263): grid search(100 후보) 기반 MSE 최소화 min/max. 1D/2D search, `p=2.4`. **weight observer로 사용**(exp/config66.yaml:9, w_qconfig.observer: MSEObserver).
- `AvgMSEObserver` (observer.py:265-291): MSE + 이동평균.
- `MSEFastObserver` (observer.py:293-413): golden section search(scipy `minimize_scalar`)로 MSE 최소화 — grid보다 빠름.
- `AvgMSEFastObserver` (observer.py:416-441): 위 + 이동평균.
- `LogAvgMSEFastObserver` (observer.py:443-521): ★ **AGQ 전용 log observer**. 후술(3.6).
- `SignAvgMSEFastObserver` (observer.py:523-544): ★ **BIG 전용 sign-aware observer**. 후술(3.5).
- `PCTObserver` (observer.py:546-612): percentile(0.99~0.99999) 기반 outlier clipping observer.

→ **per-tensor vs per-channel**: activation은 `ch_axis: -1`(per-tensor), weight는 `ch_axis: 0`(per-out-channel)(exp/config66.yaml:6,12). **symmetric/asymmetric**: 둘 다 `symmetric: False`(asymmetric)(exp/config66.yaml:5,11).

### 3.3 util_quant.py — fake-quant 커널과 STE

- `round_ste` (util_quant.py:4-8): Straight-Through Estimator. `(x.round()-x).detach()+x`로 forward는 반올림, backward는 항등.
- `fake_quantize_per_tensor_affine` (util_quant.py:11-15): 표준 affine `x_q = clamp(round(x/scale)+zp) ; deq=(x_q-zp)*scale`.
- `fake_logquantize_per_tensor_affine` (util_quant.py:17-26): ★ **AGQ 핵심 log 양자화**. `x_int = round(-log2(x/scale) * tau)`, level 초과분(`softmax_mask`)은 0으로 처리. 후술(3.6).
- `fake_quantize_per_channel_affine` (util_quant.py:28-36): 채널축 reshape 후 affine.
- learnable 버전 3종(util_quant.py:39-77): LSQ(scale grad), LSQ+(scale+zp grad). `grad_scale`(util_quant.py:80-81)로 scale gradient 스케일링.

### 3.4 fake_quant.py — FakeQuantize 군 (BIG/AGQ 구현체 포함)

**QuantizeBase** (fake_quant.py:20-109): observer 보유, observer_enabled/fake_quant_enabled 토글, scale/zp 직렬화.

**FixedFakeQuantize** (fake_quant.py:112-147): observer로 scale/zp 계산 후 고정. QDrop drop_prob(랜덤 FP/quant 혼합) 지원(fake_quant.py:143-145).

**LSQFakeQuantize** (fake_quant.py:150-196): scale을 `nn.Parameter`로 학습. **config66/44의 activation quantizer**(exp/config66.yaml:2).

**LSQSignFakeQuantize** (fake_quant.py:199-270): ★ **BIG의 핵심 quantizer**. 후술(3.5).

**LSQPlusFakeQuantize / LSQPlusSignFakeQuantize** (fake_quant.py:274-441): scale+zp 모두 학습(LSQ+). Sign 버전은 채널 비대칭성(two-peak) 판정 후 sign으로 보정하는 또 다른 BIG 변형(fake_quant.py:322-441) — `_judge_two_peak`은 양/음 비율 비대칭률(`asy_rate >= gamma=0.8`)로 판정(fake_quant.py:350-376). **단, quant_model.py의 실제 BIG 경로는 `LSQSignFakeQuantize`를 사용**(quant_model.py:25)하므로 `LSQPlusSignFakeQuantize`는 대안/실험 경로로 추정.

**AdaRoundFakeQuantize** (fake_quant.py:445-535): ★ **weight quantizer**(exp/config66.yaml:8). AdaRound(learned rounding). `rectified_sigmoid()`로 [0,1] 라운딩 마스크 학습(fake_quant.py:479-482), `adaround_forward`로 soft/hard rounding(fake_quant.py:484-501).

**AdaptiveGranularityQuantize** (fake_quant.py:539-593): ★ **AGQ의 forward quantizer**. 후술(3.6).

### 3.5 BIG (Bimodal Integration) — 정밀 분석

**적용 텐서**: SAM **mask decoder의 cross-attention(token_to_image)의 Key projection 출력**(post-K-Linear). 코드상 `k_post_act_fake_quantize`(quant_model.py:238)에 sign 설정이 들어가며, `bimodal_adjust`는 이름에 `token_to_image`가 들어간 블록에만 적용(quant_model.py:412).

**(1) 양자화기 선택** (quant_model.py:18-29, 232-244)
- `ptq4sam_config.BIG`가 True면 `update_specialized_quantizer_config(a_qconfig,'bimodal')`로 K 경로의 quantizer를 `LSQSignFakeQuantize` + observer를 `SignAvgMSEFastObserver`로 교체(quant_model.py:24-26, 232-238).
- BIG 하이퍼파라미터를 K quantizer에 주입: `global_num`, `peak_distance`, `peak_height`(quant_model.py:241-244; 값은 exp/config66.yaml:27-29 → 128/32/0.01).

**(2) bimodal 판정** (fake_quant.py:213-225, `LSQSignFakeQuantize.judge_bimodal`)
- 입력 텐서를 flatten → numpy → **`scipy.stats.gaussian_kde`로 밀도추정**(fake_quant.py:216-217).
- `np.linspace`로 `global_num`(=128)개 점에서 밀도 y 계산(fake_quant.py:218-219).
- **`scipy.signal.find_peaks`**로 peak 탐색: 높이 임계 `peak_height*sum(y)`(=0.01*Σy), 최소 간격 `peak_distance`(=32) (fake_quant.py:219).
- **peak 개수가 정확히 2개면 bimodal로 판정**(`self.is_bimodal = len(peaks)==2`)(fake_quant.py:220).

**(3) sign factor 결정** (fake_quant.py:222-225)
- bimodal일 때 텐서를 채널축으로 전치/flatten 후, **채널별 평균 부호**를 sign factor로 사용: `sign[c] = torch.sign(mean(channel_c))`(fake_quant.py:223-224).
- 즉 sign factor 탐색 방식은 **KL/gaussian-fit가 아니라**: ① KDE+find_peaks로 bimodal 여부만 판정, ② 채널별 평균의 부호(±1)를 sign으로 채택. (코드로 확인됨)

**(4) reparameterize (오프라인 가중치 흡수)** (quant_model.py:286-294, `bimodal_adjust`)
- `addjust_linear(linear, sign)`: `linear.weight.mul_(sign.unsqueeze(1))`, `linear.bias.mul_(sign)` (quant_model.py:289-291).
- **Q projection과 K projection 가중치/바이어스에 동일 sign을 곱해 흡수**(quant_model.py:292-293). Q·K^T = (sign·Q)(sign·K)^T로 부호가 상쇄되어 어텐션 logit이 수학적으로 불변하면서, K 분포는 unimodal로 정렬됨(이것이 "mathematically equivalent sign operation"의 구현).
- 적용 후 `is_bimodal=False`로 한 번만 수행(quant_model.py:294).
- 실행 트리거: `calibrate()`에서 `if BIG:` → 1회 forward(`model.extract_feat(cali_data[0])`)로 분포 관찰 후 `bimodal_adjust(model, logger)` 호출(test_quant.py:463-465).

**(5) sign-aware observer** (observer.py:523-544, `SignAvgMSEFastObserver`)
- `loss_fx`에서 fake-quant 결과와 원본 모두에 sign을 곱해(`x_q*sign`, `x*sign`) MSE 계산(observer.py:538-543). sign 정렬된 공간에서 scale을 탐색하기 위함.

> **요약(BIG)**: 텐서 = mask decoder cross-attn(token_to_image)의 K projection 출력. 판정 = KDE+find_peaks로 peak 2개. sign = 채널별 평균 부호. reparam = Q/K Linear의 weight·bias에 sign 곱해 오프라인 흡수(quant_model.py:286-294). 어텐션 등가성 보존.

### 3.6 AGQ (Adaptive Granularity Quantization) — 정밀 분석

**적용 텐서**: **post-Softmax attention 확률**. mask decoder의 모든 어텐션 블록(`softmax_post_act_fake_quantize`, quant_model.py:236, 277)과, encoder 양자화 시 self-attention의 softmax 출력(quant_model.py:374, 396)에 적용.

**(1) 양자화기/observer 선택** (quant_model.py:18-29, 228-231, 369-374)
- `ptq4sam_config.AGQ`가 True면 softmax 경로 config를 `quantizer: AdaptiveGranularityQuantize`, `observer: LogAvgMSEFastObserver`로 교체(quant_model.py:22-24, 229).

**(2) log 양자화 수식** (fake_quant.py:583-593 `AdaptiveGranularityQuantize.quantize`, util_quant.py:17-26과 동일 형태)
- `levels = quant_max - quant_min + 1` (fake_quant.py:584).
- `x = clamp(x, 1e-20, None)` (fake_quant.py:585).
- `x_int = round_ste(-1 * log2(x/scale) * tau)` (fake_quant.py:586). ★ **log2 기반** 양자화이며, `tau`(τ)가 **granularity 파라미터(=power-of-two base의 분해능)**.
- level 초과(`x_int >= levels`)는 마스크 후 0으로 설정(`softmax_mask`)(fake_quant.py:588-591).
- 역양자화: `X = scale * 2^(-x_q / tau)` (fake_quant.py:590). 즉 재구성 값은 `scale * 2^(-x_q/τ)` 형태의 power-of-two 양자화.

**(3) 최적 τ(power-of-two base) 탐색** (observer.py:443-521 `LogAvgMSEFastObserver`)
- 후보 τ: `self.taus = [2**i for i in range(3)]` = **[1, 2, 4]** (observer.py:450). → AGQ가 탐색하는 granularity 후보.
- 각 τ에 대해 golden section search로 scale을 찾고, **출력에 미치는 영향을 value(V)까지 고려**해 오차 계산:
  - `loss_fx`: log-quant된 attn과 원본 attn을 **각각 V와 곱한 결과의 MSE** `lp_loss(x_q @ value, x @ value)`(observer.py:454-465). 즉 단순 attn MSE가 아니라 **attn@V 출력 기준 오차**로 τ/scale을 고른다. value는 forward에서 주입(quant_model.py:277 `self.softmax_post_act_fake_quantize(attn, value=v)`; fake_quant.py:558-562 → observer.py:467-469).
  - 각 τ별 `best_tau_scales`, `tau_errors`를 저장(observer.py:498-515).
- **최종 τ 선택**: `AdaptiveGranularityQuantize.init_quantization_scale`에서 `tau_errors`가 최소인 인덱스의 τ와 scale을 채택(fake_quant.py:574-581). → 어텐션마다 다른 분포에 맞춰 [1,2,4] 중 최적 base를 "적응적"으로 선택.

> **요약(AGQ)**: post-softmax 확률을 `scale * 2^(-x_q/τ)` 형태 log2 양자화. granularity 파라미터 τ ∈ {1,2,4}를 attn@V 출력 MSE 기준 golden-section search로 선택(observer.py:450,464; fake_quant.py:574-590). long-tail/power-law 분포에 적합한 하드웨어 친화적(시프트 기반) 양자화.

### 3.7 quantized_module.py — 양자화 연산자/팩토리

- **ObserverDict / FakeQuantizeDict** (quantized_module.py:6-26): config 문자열 → 클래스 매핑. AGQ/BIG용 클래스 등록됨(`AdaptiveGranularityQuantize`, `LSQSignFakeQuantize`, `LogAvgMSEFastObserver`, `SignAvgMSEFastObserver`).
- **ActivationQuantizer / WeightQuantizer** (quantized_module.py:29-41): config로 quantizer 인스턴스 생성.
- **QConv2d / QLinear / QEmbedding** (quantized_module.py:48-114): weight에 fake-quant 적용. QLinear는 `gamma` 융합 옵션(quant_model.py BIG reparam과 별개, quantized_module.py:80-86).
- **Quantizer 팩토리** (quantized_module.py:158-172): module=None이면 activation quantizer, Conv/Linear/Embedding이면 weight 양자화 래핑.
- **QuantizedLayer / PreQuantizedLayer** (quantized_module.py:180-213): 각각 후처리/전처리 fake-quant 붙인 래퍼. SAM 어텐션 래핑이 `PreQuantizedLayer`를 사용(quant_model.py:40-41,222-225).
- **QuantizedMatMul** (quantized_module.py:215-229): 두 입력 a,b를 각각 fake-quant 후 `a@b`. (mask decoder 등에서 matmul 양자화)

### 3.8 quantized_module_matmul.py — 보조/구버전 MatMul 모듈

- 구조는 quantized_module.py와 거의 동일하나 **AGQ/BIG 클래스가 등록되지 않음**(quantized_module_matmul.py:9-23). `from tools.modifier import MatMul`에 의존(quantized_module_matmul.py:7).
- `QuantizedLayer.mm_forward`에서 `A@B` 후 후처리 양자화(quantized_module_matmul.py:188-194). 'MatMul' 타입이면 forward를 mm_forward로 교체(quantized_module_matmul.py:175-178).
- **현 PTQ4SAM 경로(quant_model.py)는 이 파일을 import하지 않음** → 구버전/보조 모듈로 추정. SAM 어텐션의 matmul은 quant_model.py 내부에서 직접 `q@k`, `attn@v`로 처리(quant_model.py:273-280).

### 3.9 quant_model.py — SAM 양자화 래핑 (BIG/AGQ 적용 지점의 중심)

- **specials 매핑** (quant_model.py:403-407): 원본 SAM 모듈 → 양자화 모듈.
  - `TwoWayAttentionBlock → QuantDecoderOurTwoWayAttentionBlock`
  - `Attention(decoder) → QuantDecoderOurAttentionBlock`
  - `ImageEncoderViT → QuantImageEncoderOurViT`
- **QuantImageEncoderOurViT** (quant_model.py:297-322): patch_embed/pos_embed는 양자화 제외, 각 ViT Block을 `QunatEncoderOurBlock`으로 교체, neck 양자화(quant_model.py:303-310).
- **QuantEncoderOurAttentionBlock** (quant_model.py:353-401): SAM image encoder의 ViT 어텐션. qkv/proj는 PreQuantizedLayer, q/k/v 출력 각각 fake-quant, `attn = (q*scale)@k^T`, rel_pos 추가, softmax 후 AGQ 양자화(`softmax_post_act_fake_quantize(attn.softmax(-1), value=v)`)(quant_model.py:387-398). **encoder에는 AGQ만 적용, BIG 미적용**(K 경로가 일반 quantizer)(quant_model.py:369-377).
- **QuantDecoderOurAttentionBlock** (quant_model.py:214-294): mask decoder 어텐션. ★ **BIG + AGQ가 동시에 적용되는 핵심 지점**.
  - q/k/v proj = PreQuantizedLayer(quant_model.py:222-225).
  - AGQ: softmax 경로(quant_model.py:228-229,236,277).
  - BIG: K 경로 quantizer를 sign 버전으로 + 하이퍼파라미터 주입(quant_model.py:232-244, 238).
  - forward: `attn = q@k^T / sqrt(c)`, softmax, AGQ(value=v), `out = attn@v`, out_proj(quant_model.py:257-284).
  - `bimodal_adjust()`로 K/Q Linear에 sign 흡수(quant_model.py:286-294).
- **QuantDecoderOurTwoWayAttentionBlock** (quant_model.py:157-211): self-attn + cross-attn(token→image, image→token) + MLP의 two-way 구조 래핑(SAM mask decoder의 핵심 블록).
- **quantize_model** (test_quant.py:422-456): `--quant-encoder`일 때만 image_encoder 양자화(test_quant.py:443-444), mask_decoder는 항상 양자화(test_quant.py:446). `patch_embed`/`output_upscaling`/`iou_prediction_head`/`output_hypernetworks_mlps`는 양자화 제외(test_quant.py:426).
- **bimodal_adjust(model, logger)** (quant_model.py:409-417): mask decoder 내 이름에 `token_to_image`가 들어간 `QuantDecoderOurAttentionBlock`만 BIG 적용 → **README의 "bimodal은 SAM-B/L mask decoder에서 주로 발생"과 일치**(README.md:84).

### 3.10 state.py — 양자화 상태 제어

- `enable_calibration_woquantization(model, quantizer_type)` (state.py:6-17): 이름에 quantizer_type 포함된 모듈만 observer ON / fake-quant OFF. calibration 단계에서 `act_fake_quant` → `weight_fake_quant` 순으로 호출(test_quant.py:466,480).
- `enable_quantization` (state.py:20-31): observer OFF / fake-quant ON (실제 추론).
- `disable_all` (state.py:34-40): 둘 다 OFF.

### 3.11 recon.py — block-wise reconstruction (QDrop 계열)

- `save_inp_oup_data` (recon.py:11-47): forward hook(DataSaverHook, utils.py:38-55)으로 각 블록 입/출력 캐시.
- `LinearTempDecay` / `LossFunction` (recon.py:50-126): AdaRound용 round_loss(temp decay) + rec_loss(L_p). `round_loss = weight*(1-|2(round-0.5)|^b).sum()`(recon.py:118-119).
- `reconstruction` (recon.py:136-342):
  - weight quantizer `init`으로 AdaRound alpha 등록(recon.py:153-154).
  - activation은 LSQ/LSQ+/AGQ의 scale(및 zp)을 학습 파라미터로 수집(recon.py:155-164). **AdaptiveGranularityQuantize.scale도 튜닝 대상**(recon.py:163-164).
  - `only4` 옵션 시 k_proj/q_proj만 reconstruction(recon.py:150) — BIG 블록 한정 튜닝.
  - QDrop drop_prob로 quant/FP 입력 랜덤 혼합(recon.py:280-308).
  - Adam + CosineAnnealing으로 `iters`(=20000) 동안 최적화(recon.py:167-169,278-329).

### 3.12 SAM image encoder ViT 어텐션 구조와의 연결

- import로 외부 SAM 모듈 인터페이스 참조: `image_encoder.Attention`(EncoderAttention), `transformer.Attention`(DecoderAttention), `TwoWayAttentionBlock`, `MLPBlock`, `window_partition/unpartition`, `add_decomposed_rel_pos`(quant_model.py:8-15).
- **Encoder(ViT)**: window attention + relative position. q/k/v 분리 → `(q*scale)@k^T` → (+rel_pos) → softmax(AGQ) → `@v` → proj(quant_model.py:380-401). 여기서 BIG는 적용 안 함(encoder K 분포는 bimodal이 아니라는 논문 관찰과 일치).
- **Decoder(TwoWay)**: self-attn / cross-attn(token↔image). cross-attn token→image의 K projection 출력이 bimodal → BIG 적용(quant_model.py:412).

---

## 4. 알고리즘 / 수식 (코드에서 유도)

### 4.1 표준 affine fake-quant (util_quant.py:11-15)
```
x_q   = clamp( round(x / s) + z,  q_min, q_max )
x_deq = (x_q - z) * s
```
- symmetric:  s = max(|min|,max) / ((q_max-q_min)/2),  z = 0           (observer.py:60-63)
- asymmetric: s = (max-min)/(q_max-q_min),  z = q_min - round(min/s)    (observer.py:64-68)

### 4.2 BIG: sign factor 변환 (quant_model.py:223-224, 286-294)
채널 c에 대해
```
sign[c] = sign( mean_over_tokens( K_proj_out[:, :, c] ) )     # (fake_quant.py:224)
```
weight/bias 흡수(오프라인):
```
W_k' = W_k ⊙ sign(행 broadcast),  b_k' = b_k ⊙ sign            # quant_model.py:290-291
W_q' = W_q ⊙ sign,                b_q' = b_q ⊙ sign            # quant_model.py:292
```
등가성: attention logit ∝ Q·Kᵀ. Q,K 각 채널에 같은 ±부호를 곱하면 내적에서 sign²=1로 상쇄 → logit 불변, 단 K(및 Q)의 채널 분포는 bimodal→unimodal로 정렬되어 양자화 오차 감소.

bimodal 판정(fake_quant.py:216-220):
```
y = gaussian_kde(K_flat)(linspace(min,max, global_num=128))
peaks = find_peaks(y, height=0.01*Σy, distance=32)
is_bimodal = (len(peaks) == 2)
```

### 4.3 AGQ: adaptive granularity log 양자화 (fake_quant.py:583-590)
```
levels = q_max - q_min + 1
x      = clamp(x, 1e-20, ∞)
x_int  = round( -log2(x / s) · τ )            # log2 기반, τ=granularity
x_q    = clamp(x_int, 0, levels-1)
X      = s · 2^( -x_q / τ )                    # 역양자화 (power-of-two)
X[x_int ≥ levels] = 0                          # tail/소값 절단
```
τ ∈ {1, 2, 4} (observer.py:450). 선택 기준(observer.py:464, fake_quant.py:574-581):
```
for τ in {1,2,4}:
    s*(τ) = argmin_s  || (logquant_{s,τ}(A) · V) − (A · V) ||_p   (golden section)
    err(τ) = 위 최소값
τ* = argmin_τ err(τ),   s* = s*(τ*)
```
즉 attn@V **출력 기준 MSE**로 τ와 scale을 동시 최적화 → 어텐션별 분포 차이에 적응.

---

## 5. 학습/평가 파이프라인

### 5.1 데이터셋
- **COCO**(detection/instance segmentation)가 주 데이터셋. README.md:49-58에 `data/coco/{annotations,train2017,val2017,test2017}` 구조. (LVIS는 README/코드에서 직접 확인 안 됨 → **확인 불가**, 논문에는 언급될 수 있으나 repo 코드 기준 COCO만 확인.)
- Calibration: train set에서 `q_config.calibrate`(=32)장 샘플(utils.py:99-139 `load_calibration`, exp/config66.yaml:13).
- SAM은 detector(YOLOX/Faster R-CNN/DETR/DINO)가 만든 box prompt로 instance segmentation 수행(prompt-SAM 구조). 평가 metric은 `segm`(test_quant.py:78).

### 5.2 실행 명령어 (README.md:71-79 기준)
W6A6, SAM-B + YOLO detector:
```
python ptq4sam/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-l.py \
  --q_config exp/config66.yaml --quant-encoder
```
- `--quant-encoder`: SAM image encoder까지 양자화(없으면 mask decoder만).
- `--q_config exp/config44.yaml`: W4A4.
- `--fp`: 양자화 없이 FP 평가.
- `--show-dir`: 예측 시각화.
- 권장: 40GB 이상 GPU(README.md:82).
- 비고: `quant_coco.py`는 SAM이 아닌 CNN detector용 (구)QDrop 엔트리(quant_coco.py:22-27, `qdrop.*` import) → PTQ4SAM 실험에는 `test_quant.py` 사용.

### 5.3 단계별 내부 흐름 (test_quant.py:main → calibrate → recon)
1. 모델 빌드 + 양자화 모듈 치환(test_quant.py:316,342).
2. `calibrate(model, cali_data, BIG)`(test_quant.py:347, 460-485):
   - BIG=True면 1회 forward 후 `bimodal_adjust`(test_quant.py:463-465).
   - activation observer ON → calibrate 데이터 32장 forward로 min/max 통계(test_quant.py:466-469).
   - 분산 환경이면 min/max all_reduce(test_quant.py:471-478).
   - weight observer ON → 1회 forward(test_quant.py:480-481).
3. recon(있으면): block-wise AdaRound+LSQ scale 튜닝(test_quant.py:349-352, recon.py:136-342).
4. `enable_quantization` 후 single/multi GPU 평가, COCO segm metric 계산(test_quant.py:354,372-417).

---

## 6. 의존성

- **PyTorch**: `torch==1.10.2+cu113`, `torchvision==0.11.3+cu113`(environment.sh:1). README는 python 3.7 + 일반 torch 설치도 안내(README.md:16-18).
- **MMCV**: `mmcv-full<2.0.0`(environment.sh:3, README.md:24).
- **MMDetection**: 소스 빌드(`python3 setup.py build develop`)(environment.sh:8-9, README.md:43-44). detector/데이터로더/COCO 평가 담당.
- **CUDA ops**: `projects/instance_segment_anything/ops` 빌드(environment.sh:5-6) — deformable attention 등 외부 커널(이름만).
- **scipy**: `minimize_scalar`(observer.py:4, golden section), `find_peaks`/`gaussian_kde`(fake_quant.py:15-16, BIG 판정).
- **easydict / pyyaml**: config 파싱(utils.py:4-5,13-28).
- 기타: numpy, pandas. requirements.txt는 mmdetection requirements를 참조(requirements.txt:1-4).

---

## 7. 강점 / 한계 / 리스크

### 강점
- **재학습 없는 PTQ**로 SAM을 W6A6/W4A4까지 양자화(메모리/연산 대폭 절감).
- BIG: 가중치에 sign을 흡수하는 **오프라인·등가 변환**이라 런타임 추가 비용 0이며, bimodal outlier 문제를 근본적으로 완화.
- AGQ: post-softmax를 `2^(-x_q/τ)` 형태 **power-of-two log 양자화** → 시프트 기반으로 하드웨어 친화적, τ를 attn@V 출력 기준으로 적응 선택.
- 모듈식 설계: observer/quantizer dict 기반으로 기법 조합이 config 한 줄(quantizer/observer 이름)로 교체 가능(quant_model.py:18-29).

### 한계 / 리스크 (코드 기준)
- **SAM 의존**: mmdet + prompt-SAM + CUDA ops + 40GB GPU 등 무거운 스택. 경량 ViT 단독에 바로 이식하기엔 의존성 많음.
- **bimodal 판정 휴리스틱**: KDE+find_peaks(peak 정확히 2개) + 채널 평균 부호. 임계값(height 0.01·Σy, distance 32, global_num 128) 민감성 존재(fake_quant.py:219). 분포가 약하게 bimodal이거나 3-modal이면 미적용/오판 가능.
- **BIG 적용 범위 제한**: `token_to_image` cross-attn에만 적용(quant_model.py:412). 다른 K 경로의 bimodal은 미처리.
- **코드 위생 이슈**: `AvgSignMinMaxObserver`의 `super().__init__`이 잘못된 부모 호출(observer.py:150) → 사용 시 에러 추정(미사용 추정). test_quant.py:27에 하드코딩 절대경로 `sys.path.append("/nvme/...")`. `LSQPlusSignFakeQuantize`(또 다른 BIG 변형)는 실제 경로 미사용 추정. `quantized_module_matmul.py`는 현 경로에서 미import(구버전 추정).
- **AGQ τ 후보가 {1,2,4}로 고정**(observer.py:450) — 더 넓은 base는 코드 수정 필요.
- **평가 데이터**: repo 코드상 COCO만 확인. LVIS 등은 확인 불가.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 + XR 시선추적)

> 우리 프로젝트는 "ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적"으로 추정됨. 아래는 그 관점의 전이 가능성 평가(추정 포함).

- **어텐션 양자화 기법 전이 가능 (높음)**: SAM image encoder는 표준 ViT이고, PTQ4SAM의 q/k/v fake-quant, post-softmax 처리, per-tensor activation / per-channel weight 구조(exp/config66.yaml)는 우리의 ViT 가속기에 **그대로 차용 가능**. 특히 W4A4까지 PTQ로 내려가는 레시피(AdaRound weight + LSQ activation + reconstruction)는 저비트 FPGA 가속의 정확도 확보에 직접 도움.
- **AGQ ↔ FPGA softmax (매우 높음)**: `X = s·2^(-x_q/τ)`는 **시프트 연산으로 dequant 가능**한 log 양자화(fake_quant.py:590). FPGA에서 곱셈기 없이 barrel shifter로 attn·V 전처리를 구현할 여지가 큼. τ∈{1,2,4} 후보도 하드웨어 LUT/시프트량과 매핑이 쉬움. HG-PIPE류 파이프라인 softmax 블록에 적용 시 DSP 절감 기대(추정).
- **BIG ↔ outlier/저비트 정확도 (높음)**: bimodal/outlier는 저비트 양자화에서 정확도 붕괴의 주원인. BIG은 **추가 런타임 비용 없이 가중치에 흡수되는 등가 변환**(quant_model.py:286-294)이라 FPGA에서 별도 하드웨어 변경 없이 정확도를 끌어올릴 수 있음 — XR 환경의 작은 INT4/INT6 가속기에 특히 유효(추정). 단 sign 흡수는 calibration 단계 SW 작업이므로 하드웨어 영향 없음.
- **XR 시선추적 직접 적용성 (낮음)**: SAM 자체는 무겁고(ViT-B/L/H + mask decoder + prompt) 실시간 시선추적용 경량 모델과 거리가 멂. **기법(BIG/AGQ)은 전이 가치가 높으나, SAM 모델 자체를 시선추적 백본으로 쓰는 것은 부적합**(추정). 우리 쪽 경량 ViT/Transformer eye-tracking 백본에 BIG/AGQ만 이식하는 방향이 현실적.
- **연계 참고 문서**: 동일 디렉토리의 `AdaLog.md`(log 양자화), `RepQ-ViT.md`/`outlier-free-transformers.md`/`NoisyQuant.md`(outlier 처리), `HG-PIPE.md`(우리 가속기 계열)와 교차 비교 권장.

---

## 9. 근거 표기 (확인 / 추정 / 확인 불가)

### 코드로 확인된 사실
- BIG quantizer/observer = `LSQSignFakeQuantize`/`SignAvgMSEFastObserver` (quant_model.py:24-26, 232-238).
- BIG 적용 텐서 = mask decoder cross-attn(token_to_image) K projection 출력 (quant_model.py:238, 412).
- bimodal 판정 = gaussian_kde + find_peaks, peak==2 (fake_quant.py:216-220).
- sign factor = 채널별 평균 부호 (fake_quant.py:223-224).
- BIG reparam = Q/K Linear weight·bias에 sign 곱(오프라인) (quant_model.py:286-294).
- AGQ quantizer/observer = `AdaptiveGranularityQuantize`/`LogAvgMSEFastObserver` (quant_model.py:22-24, 229).
- AGQ = log2 기반, `X=s·2^(-x_q/τ)`, τ∈{1,2,4}, attn@V MSE로 선택 (fake_quant.py:583-590, observer.py:450,464,574-581).
- weight=AdaRound/MSEObserver/per-channel, activation=LSQ/AvgMinMax/per-tensor, asymmetric, W6A6 또는 W4A4 (exp/config66.yaml, config44.yaml).
- 평가 데이터=COCO, entry=test_quant.py, calibrate 32장 (README.md:49-79, exp/config66.yaml:13).

### 추정 (코드 정황 기반)
- `LSQPlusSignFakeQuantize`(two-peak/asy_rate 기반)는 대안 BIG 경로이며 실제 quant_model.py 경로 미사용으로 추정.
- `quantized_module_matmul.py`는 구버전/보조(현 경로 미import)로 추정.
- `AvgSignMinMaxObserver`는 부모 호출 버그로 미사용 추정(observer.py:150).
- SAM 자체의 XR 시선추적 직접 적용성은 낮음(모델 무게 기반 추정).

### 확인 불가
- LVIS 등 COCO 외 데이터셋 사용 여부 (repo 코드에서 미확인; 논문 본문은 별도).
- 논문 정확도 수치(mAP 등)는 코드가 아닌 논문 영역으로 본 분석에서 미검증.
- BIG의 "정확히 2-peak" 임계가 논문 수식과 1:1 대응되는지(논문 미참조, 코드 기준만).

---

### 핵심 한 줄 정리
PTQ4SAM은 SAM(ViT encoder + two-way mask decoder)을 재학습 없이 W6A6/W4A4로 양자화하는 PTQ 프레임워크로, (1) mask-decoder Key 출력의 bimodal 분포를 sign factor를 가중치에 오프라인 흡수해 unimodal화하는 **BIG**(quant_model.py:286-294)와, (2) post-softmax를 `s·2^(-x_q/τ)` log 양자화하며 τ∈{1,2,4}를 attn@V MSE로 적응 선택하는 **AGQ**(fake_quant.py:583-590)가 두 축이다.
