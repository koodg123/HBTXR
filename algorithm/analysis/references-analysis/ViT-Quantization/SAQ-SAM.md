# SAQ-SAM 정밀 분석

> 분석 대상 루트: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\SAQ-SAM`
> 분석 도구: Glob/Grep/Read 만 사용 (bash 미사용). 모든 핵심 근거는 `파일:라인` 표기.
> 표기 규칙: **[코드확인]** = 소스에서 직접 확인한 사실, **[추정]** = 코드/문맥 기반 추론, **[확인불가]** = 본 repo 소스만으로 확정 불가.

---

## 1. 개요

- **목적**: Segment Anything Model(SAM)에 대한 PTQ(Post-Training Quantization)를 "의미적 정렬(semantic alignment)" 관점에서 개선. 원논문은 **SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model**, AAAI 2026, arXiv:2503.06515 (README.md:1-4, 130-135).
- **핵심 아이디어 3종** (README.md:14-18):
  1. **PCC (Perceptual-Consistency Clipping)**: attention focus overlap(주목 영역 중첩도)을 이용해, 의미 능력을 보존하면서 공격적(aggressive)으로 activation을 클리핑. CLI 플래그 `--FFC` (README.md:15, 105).
  2. **PAR (Prompt-Aware Reconstruction)**: mask decoder의 cross-attention(image-prompt 상호작용)을 reconstruction에 반영하여, 분포(distribution)와 의미(semantic) 양면 정렬. CLI 플래그 `--CAM_loss=PAR` (README.md:16, 99, 108).
  3. **Layer-skipping**: 상호작용 효율을 위해 encoder의 image token에 대한 스킵 전략 (README.md:17). 코드상으로는 PAR 상호작용 계산 시 image/prompt 토큰을 16개로 서브샘플링하는 형태로 구현됨 (3절 참조).
- **베이스라인**: PTQ4SAM에서 포크됨 (README.md:25, 140). PTQ4SAM의 BIG(Bimodal Integration)·AGQ(Adaptive Granularity Quantization)를 그대로 계승하고, 그 위에 PCC/PAR/layer-skip을 추가. 환경/데이터 준비도 PTQ4SAM 기준 (README.md:24-26).
- **성능 주장**: SAM-B를 4-bit 양자화 시 baseline 대비 instance segmentation mAP +11.7% (README.md:19).
- **베이스 프레임워크 dropout 버그**: AHCPTQ(AGCPTQ)가 지적한 버그를 그대로 안고 있음 — 평가 단계에서 dropout 확률(`drop_prob`)이 1.0으로 복원되지 않아 activation의 절반이 양자화되지 않음. PCC-only(SAQ-SAM*) 결과는 영향 없고, reconstruction(PAR) 사용 시에만 영향 (README.md:116-124, test_quant_SAQ_m.py:674-686).

### 패키지명 혼동 주의 (중요)
SAQ-SAM은 PTQ4SAM에서 포크되었기 때문에 **패키지 디렉토리 이름이 동일하게 `ptq4sam/`** 이다. 따라서 `import ptq4sam...` 구문이 보여도 이는 SAQ-SAM 자체 소스다. SAQ-SAM 고유 기여는 아래 위치에 있다:
- PCC: `ptq4sam/model/quant_model.py` (AttentionOverlapLoss, search_range_AOL) + `ptq4sam/solver/test_quant_SAQ_m.py` (calibrate의 `--FFC` 분기)
- PAR: `ptq4sam/solver/recon.py` (CAMS/CLM 재구성, PAR loss) + 수정된 `mask_decoder.py`(predict_calib_recon)
- 메인 엔트리: `ptq4sam/solver/test_quant_SAQ_m.py`

---

## 2. 디렉토리 구조

### 2.1 자체 양자화 핵심 소스 (PTQ4SAM 계승 + SAQ-SAM 확장)
```
SAQ-SAM/
├── README.md                         # 사용법, PCC/PAR/layer-skip 설명, dropout 버그 고지
├── environment.sh                    # torch 1.10.2+cu113, mmcv-full<2.0, CUDA ops 빌드
├── exp/                              # 양자화 비트/재구성 설정 YAML
│   ├── config_SA_66.yaml             # SAQ-SAM용 W6A6 (encoder/decoder 분리 qconfig) ★
│   ├── config_SA_44.yaml             # SAQ-SAM용 W4A4 ★
│   ├── config66.yaml                 # PTQ4SAM 스타일 W6A6 (단일 a/w qconfig)
│   └── config44.yaml                 # PTQ4SAM 스타일 W4A4
├── ptq4sam/                         # ← 패키지명은 ptq4sam지만 SAQ-SAM 소스
│   ├── solver/
│   │   ├── test_quant_SAQ_m.py       # ★ SAQ 메인 엔트리 (PCC/PAR/recon/layer-skip 오케스트레이션)
│   │   ├── recon.py                  # ★ CAMS/CLM 재구성, PAR loss, image-token 변환
│   │   ├── utils.py                  # config 파서, DataSaverHook, calibration 로더
│   │   ├── test_quant.py             # (구) PTQ4SAM 스타일 엔트리
│   │   ├── quant_coco.py             # COCO 양자화 엔트리(보조)
│   │   └── test_quant_rep.py         # RepQ-ViT 스타일 별도 실험 엔트리(quant_rep/repq_utils 사용)
│   ├── quantization/
│   │   ├── observer.py               # ★ Observer 군 + PCT/MSE/Log observer
│   │   ├── fake_quant.py             # ★ LSQ/AdaRound/AGQ/Sign fake-quant + KDE bimodal 판정
│   │   ├── quantized_module.py       # QuantizedLayer/PreQuantizedLayer/QuantMatMul, 디스패치 dict
│   │   ├── quantized_module_matmul.py# MatMul 양자화 변종
│   │   ├── util_quant.py             # round_ste, per-tensor/channel/log fake-quant 커널
│   │   └── state.py                  # observer/fake-quant enable/disable 상태 전환
│   ├── model/
│   │   └── quant_model.py            # ★ SAM 양자화 래핑(specials), PCC 구현, layer/stage 구획
│   ├── quant_rep/                    # RepQ-ViT 스타일 reparam 양자화(별도 실험용, 메인 미사용)
│   │   ├── quantizer.py, quant_modules_rep.py, quant_model_rep.py, __init__.py
│   └── repq_utils/                   # KDE 분포모델 + build_model (별도 실험용, 메인 미사용)
│       ├── kde.py, build_model.py, __init__.py
└── projects/instance_segment_anything/
    └── models/segment_anything/modeling/
        ├── image_encoder.py          # ViT image encoder (window/global attn)
        ├── transformer.py            # ★ TwoWayTransformer (mask decoder cross-attn)
        └── mask_decoder.py           # ★ predict_calib_recon 추가 (PAR 상호작용용)
```

### 2.2 외부 프레임워크 (이름만 — 내부 분석 제외)
- **mmdetection/** : OpenMMLab 검출 프레임워크 전체. detector(Faster R-CNN, YOLOX, DETR, DINO) 빌드/평가에만 사용 (test_quant_SAQ_m.py:20-26). **분석 제외.**
- **projects/instance_segment_anything/ops/** : CUDA 커널(Deformable 등). `setup.py build install`로 컴파일 (README.md:46-51, environment.sh:5-7). **분석 제외.**
- **projects/instance_segment_anything/** 의 SAM 외 부분(predictor, detector adapter 등)은 Prompt-Segment-Anything 기반 (README.md:140). 양자화에 직접 관련된 modeling 3파일만 분석.

### 2.3 제외/미분석 항목
- `mmdetection/configs/*` 수백 개 detector config (Glob 결과 다수). 이름만.
- 대용량 체크포인트(.pth: sam_b/l/h, detector weights) — 다운로드 대상, README.md:72-80. 이름만.
- `quant_rep/`, `repq_utils/` : Grep 검증 결과 **메인 SAQ 파이프라인(`test_quant_SAQ_m.py`)에서 import되지 않음.** 오직 `test_quant_rep.py`만 사용 (Grep: test_quant_rep.py:42-43). 즉 RepQ-ViT 계열 별도 실험 코드이며 SAQ-SAM 고유 기여(PCC/PAR)와 무관. **[코드확인]** repq_utils/kde.py는 pytorch-generative KDE 구현(GaussianKernel 등, kde.py:87-128)으로 PCC와 직접 연관 없음. (PCC는 KDE를 쓰지 않음 — 4절 참조.)

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.0 전체 PTQ 파이프라인 (test_quant_SAQ_m.py:main)
실행 순서 (main, test_quant_SAQ_m.py:346-737):
1. config/q_config 로드 → detector(mmdet) + SAM predictor 빌드 (test_quant_SAQ_m.py:364-490).
2. calibration 데이터 로드: `utils.load_calibration` (test_quant_SAQ_m.py:484, utils.py:115-157). COCO train에서 `calibrate`(=32)장 추출, 첫/9번째 샘플 swap (utils.py:153).
3. `quantize_model`: SAM encoder/decoder를 양자화 모듈로 교체 (test_quant_SAQ_m.py:522, 739-775).
4. `calibrate`: BIG(bimodal merge) → PCC(QK calib, `--FFC`) → activation observer → weight observer (test_quant_SAQ_m.py:606, 778-829).
5. `enable_quantization` (test_quant_SAQ_m.py:608).
6. `recon_model`: PAR 재구성 (CAMS=encoder, CLM=decoder) (test_quant_SAQ_m.py:631-633, 832-923).
7. encoder/decoder 선택적 양자화 활성화 후 COCO 평가 (test_quant_SAQ_m.py:638-737).

### 3.1 양자화 모듈 교체 — `quantize_model` (test_quant_SAQ_m.py:739-775)
- `replace_module`(중첩함수, :741-759): `nn.Conv2d`/`nn.Linear`→`QuantizedLayer`, `nn.ReLU/ReLU6/GELU`는 직전 quant layer의 activation으로 흡수 후 `Identity`로 치환.
- **양자화 제외 레이어** (test_quant_SAQ_m.py:743): `patch_embed`, `output_upscaling`, `iou_prediction_head`, `output_hypernetworks_mlps` 는 skip → FP 유지. **[코드확인]** (= mask 생성 최종 head와 patch embedding은 양자화 안 함.)
- image_encoder는 `specials[ImageEncoderViT]` = `QuantImageEncoderOurViT` 로 통째 교체 (test_quant_SAQ_m.py:762, quant_model.py:794).
- mask_decoder는 `replace_module`로 재귀 교체, 그 안의 `TwoWayAttentionBlock`/`Attention`이 `specials`로 교체 (test_quant_SAQ_m.py:764, quant_model.py:791-795).
- `specials` 디스패치 (quant_model.py:791-795):
  - `TwoWayAttentionBlock → QuantDecoderOurTwoWayAttentionBlock`
  - `Attention(decoder) → QuantDecoderOurAttentionBlock`
  - `ImageEncoderViT → QuantImageEncoderOurViT`

### 3.2 Observer 종류 / per-tensor·channel / sym·asym / W·A 비트 (observer.py, config*.yaml)

**[코드확인]** config_SA_66.yaml / config_SA_44.yaml 기준 (SAQ-SAM 전용):
| 대상 | quantizer | observer | bit | symmetric | ch_axis | 입도 |
|---|---|---|---|---|---|---|
| encoder activation | `LSQFakeQuantize` | `AvgMSEFastObserver` | 6/4 | False(asym) | -1 | per-tensor |
| encoder weight | `AdaRoundFakeQuantize` | `MSEObserver` | 6/4 | False | 0 | per-channel |
| decoder activation | `LSQFakeQuantize` | `AvgMSEFastObserver` | 6/4 | False | -1 | per-tensor |
| decoder weight | `AdaRoundFakeQuantize` | `MSEObserver` | 6/4 | False | 0 | per-channel |

(config_SA_66.yaml:1-24, config_SA_44.yaml:1-24)

- **bit 의미**: `W6A6` = weight 6bit / activation 6bit, `W4A4` = 4/4. encoder/decoder 동일 비트 (config 상). 비대칭(asym, symmetric=False) → zero_point 사용 (observer.py:64-68).
- **special quantizer 자동 치환** (quant_model.py:24-35): activation observer/quantizer가 두 종류로 분기:
  - `softmax` 출력: AGQ on이면 `AdaptiveGranularityQuantize` + `LogAvgMSEFastObserver` (log 양자화, 4절).
  - bimodal(K activation): BIG on이면 `LSQSignFakeQuantize` + `SignAvgMSEFastObserver`.
- **Observer 군** (observer.py): `MinMaxObserver`(72), `AvgMinMaxObserver`(117, running 평균), `MSEObserver`(172, grid search 2.4-norm), `AvgMSEObserver`(270), `MSEFastObserver`(298, golden-section), `AvgMSEFastObserver`(421, 메인 activation observer), `LogAvgMSEFastObserver`(448, softmax log), `SignAvgMSEFastObserver`(528, bimodal), `PCTObserver`(550, percentile clipping).
  - `MSEObserver`: candidate 100개 grid, 1D(symmetric/one-side)/2D(asym) 탐색 (observer.py:202-267). p=2.4 (observer.py:176).
  - `PCTObserver`: percentile list [0.99,0.999,0.9999,0.99999]로 클리핑 후보 탐색, MSE 최소 선택 (observer.py:555, 586-606). **단, config_SA에는 미사용** (activation은 AvgMSEFastObserver). PCC는 PCTObserver와 별개 메커니즘(아래 3.4).
- **qparam 계산** (observer.py:50-69): asym scale=`(max-min)/(qmax-qmin)`, zp=`qmin - round(min/scale)`을 [qmin,qmax]로 클램프. sym은 `max(|min|,max)/((qmax-qmin)/2)`.

### 3.3 fake-quant 군 (fake_quant.py)
- `QuantizeBase` (fake_quant.py:20-119): observer 보유, `observer_enabled`/`fake_quant_enabled`/`drop_prob`/`clip_input` 플래그. forward 시 observer로 min/max 갱신 후 scale/zp 계산.
- `LSQFakeQuantize` (fake_quant.py:213-290): scale을 **학습 가능 Parameter**로 둠 (재구성 시 LSQ로 scale 최적화). grad scaling factor `1/sqrt(numel*qmax)` (fake_quant.py:271-272). `clip_input==1`이면 forward 입력을 `[clip_input_l, clip_input_r]`로 사전 클램프 (fake_quant.py:242-243). `drop_prob<1`이면 QDrop 방식으로 양자화/원본을 확률 혼합 (fake_quant.py:283-285).
- `LSQSignFakeQuantize` (fake_quant.py:293-364): BIG용. `judge_bimodal`(:307-319)이 **scipy `gaussian_kde`로 KDE → `find_peaks`로 peak 2개면 bimodal 판정**, 채널별 부호(sign) 산출. PTQ4SAM의 BIG 계승. (이 KDE가 repo 내 유일하게 "동작하는" KDE이며, PCC가 아니라 BIG에 쓰임.)
- `LSQPlusFakeQuantize`/`LSQPlusSignFakeQuantize` (fake_quant.py:368-535): scale+zero_point 모두 학습. Sign 변종은 `_judge_two_peak`로 비대칭율(gamma=0.8) 기반 two-peak 판정 (fake_quant.py:444-470).
- `AdaRoundFakeQuantize` (fake_quant.py:539-632): weight용 adaptive rounding. `init`(:555)에서 alpha 초기화, `rectified_sigmoid`(:575)로 [0,1] round mask 생성, `get_hard_value`(:599)로 최종 hard 반올림. 재구성 시 alpha 학습.
- `AdaptiveGranularityQuantize` (fake_quant.py:636-691): AGQ. softmax 출력 log 양자화. `init_quantization_scale`(:672)이 observer의 tau별 error 중 최소를 선택, `quantize`(:681)는 `2^(-x_q/tau)` 로그 그리드. value(=V matrix)를 곱한 출력 기준으로 scale 탐색(observer.LogAvgMSEFastObserver.loss_fx, observer.py:459-470) — softmax×V 후 오차 최소화.

### 3.4 PCC (Perceptual-Consistency Clipping) — ★ SAQ-SAM 고유 기여 1

PCC는 두 부분으로 구성: (A) FP 기준 attention focus mask 저장, (B) 양자화 후보 클리핑 범위를 attention-IoU(overlap)로 탐색.

**(A) calibrate에서 `--FFC` 분기** (test_quant_SAQ_m.py:778-829):
```
if args.FFC:                                            # :789
    enable_get_ori_ATM(model, logger)                  # :792  FP ATM 저장 모드 on
    model.extract_feat(cali_data[j])                   # :794  FP attention focus mask 계산·저장
    enable_calibration_woquantization(...act_fake_quant)  # :795
    for i in range(2):                                 # :796  2 iter QK calib
        enable_rank_observe(model, logger)             # :799  AttnRank 모드 on
        model.extract_feat(cali_data[j])               # :800  PCC 범위 탐색
    # 캐시된 ori_ATM 삭제 (:802-804)
```
- `enable_get_ori_ATM` (quant_model.py:815-820): 모든 `QuantDecoderOurAttentionBlock`에 `get_ori_ATM=True` 설정 → forward에서 `set_ori_ATM`(quant_model.py:421-422)이 FP softmax attention map을 `self.ori_ATM`에 저장.
- `enable_rank_observe` (quant_model.py:807-813): `AttnRank=True, AttnRank_calib=True` → forward가 PCC 탐색 분기로 진입.

**(B) attention-IoU 기반 클리핑 범위 탐색** — `search_range_AOL` (quant_model.py:349-418):
- 대상 activation: q_proj/k_proj의 입력(`layer_pre_act_fake_quantize`)과 출력(`q/k_post_act_fake_quantize`) — QK 경로의 4개 quantizer (quant_model.py:431-461).
- 핵심 루프: percentile 후보 `pct_list`(quant_model.py:397, [0.85 … 0.99999] 29개)에 대해
  - `calib_sz_across_pct`(:359-385): 양측 분위수 `clip_r=quantile(act, pct)`, `clip_l=quantile(act, 1-pct)`를 구해 quantizer의 `update_sz`로 min/max 강제 설정 (fake_quant.py:228-239).
  - 그 클리핑으로 양자화된 attention map `q_ATM = self.get_ATM(qkv)` 생성 (quant_model.py:403, get_ATM=327-346).
  - **AOL_loss = AttentionOverlapLoss(q_ATM, self.ori_ATM)** 로 FP attention focus와의 1-IoU 계산 (quant_model.py:405).
  - 최소 loss의 pct를 best로 선택, 그 범위로 최종 확정 (quant_model.py:408-414).

**AttentionOverlapLoss** (quant_model.py:37-86) — PCC의 수학적 핵심:
- `threshold`(기본 0.5, `--FFC_threshold`로 조정, test_quant_SAQ_m.py:311-315, 515)로 각 attention map의 "high-attention 영역"(focus mask) 정의:
  - `mask = (attn > threshold * rowmax).float()` (quant_model.py:67-71). row-wise max(dim=-1) 기준.
  - `intersection=min(mask1,mask2)`, `union=max(...)`, `IoU = ∑intersection / (∑union+eps)` (quant_model.py:73-83).
  - `loss = mean(1 - IoU)` (quant_model.py:83-86).
- **의미**: 양자화로 attention의 분포가 약간 변하더라도, "어디를 주목하는가(focus 영역)"가 FP와 일치하면 의미가 보존된다고 보고, 그 일치도를 직접 최적화 대상으로 삼아 **더 공격적인(높은 pct가 아닌 낮은 pct까지 포함된) 클리핑**을 허용. 단순 MSE observer보다 의미 보존 + 더 좁은 범위(작은 scale) 가능.
- **KDE 사용 여부**: **[코드확인]** PCC(search_range_AOL/AttentionOverlapLoss)는 KDE를 전혀 쓰지 않음. 분위수(`torch.quantile`/`np.percentile`)와 attention-IoU만 사용. (repo의 KDE는 BIG의 bimodal 판정용 `gaussian_kde`이며 PCC와 무관.) "PCC가 KDE 기반 분포 분석과 연관"이라는 추정은 **코드상 부정됨**.

### 3.5 PAR (Prompt-Aware Reconstruction) — ★ SAQ-SAM 고유 기여 2

PAR는 재구성(reconstruction) 단계에서, 양자화된 image token이 mask decoder의 cross-attention을 통과했을 때의 "image-prompt 상호작용 응답"을 FP의 그것과 정렬한다. `recon_encoder='CAMS'`, `recon_decoder='CLM'`, `CAM_loss='PAR'` 조합 (README.md:99, test_quant_SAQ_m.py:242-269).

**(A) recon_model 오케스트레이션** (test_quant_SAQ_m.py:832-923):
- `recon_unit` 기본 `per_stage` (test_quant_SAQ_m.py:294-297). recon 단위 모듈 묶음 선택 (:834-839): per_stage면 `QuantizedTransformerStage`까지 포함.
- 보조 모듈(`Aux_module`)에 FP의 `image_encoder.neck`, `mask_decoder`를 복사해 PAR 상호작용 supervision으로 사용 (test_quant_SAQ_m.py:626-628).
- `save_sam_inputs`(recon.py:387-440): calibration set에서 image 입력, image_pe, sparse/dense prompt embedding을 hook으로 추출. prompt가 `max_pt=16` 초과 시 16개로 랜덤 서브샘플 (recon.py:422-428) ← **layer-skip/효율과 연결**.
- encoder 재구성: `recon_encoder=='CAMS'` → `_recon_model_IE` → `reconstruction_IE` (test_quant_SAQ_m.py:904-905, 850-861).
- decoder 재구성: `recon_decoder=='CLM'` → `_recon_model_MD` → `reconstruction_MD` (test_quant_SAQ_m.py:915-918, 864-874).

**(B) Encoder 재구성 = CAMS** — `reconstruction_IE` (recon.py:520-727):
- 각 stage/layer 단위로 양자화 입출력을 hook으로 캐시 (recon.py:537-538, save_inp_oup_data_en).
- PAR일 때 (recon.py:540-543):
  1. FP encoder 출력 image token → `transform_image_token`으로 neck 통과·차원 변환(필요 시 `window_unpartition`) (recon.py:443-464).
  2. 변환된 FP image token을 mask_decoder에 통과시켜 **cross image token 응답** `fp_dc_out` 저장 — `save_CIT_list`(recon.py:469-495). 여기서 `CA_block = mask_decoder.transformer`의 출력 hook (recon.py:471-475), `calib_num=16`으로 image token 16개 제한 (recon.py:484).
- 최적화 루프(recon.py:619-702): LSQ scale + AdaRound alpha를 Adam으로 학습. QDrop 방식 입력 혼합(`drop_prob=0.5`). 매 iter:
  - 양자화 image token → neck 변환 → `get_CIT`로 양자화 cross image token `cur_dc_out` 획득 (recon.py:664-679, get_CIT=497-518).
  - **loss = LossFunction(pred=quant_out, tgt=fp_out, pred_attention=cur_dc_out, target_attention=fp_dc_out)** (recon.py:681).
  - `recon_loss='encoder'` 분기 (LossFunction.__call__, recon.py:335-339): **total_loss = CAM_sim_loss = lp_loss(pred_attention, target_attention, p=2)** — encoder 재구성은 상호작용 응답 정렬만으로 학습.

**(C) Decoder 재구성 = CLM** — `reconstruction_MD` (recon.py:1010-1229):
- 양자화 image embedding을 먼저 추출(`get_q_image_embedding`, recon.py:960-978)해 supervision에 사용 (recon.py:1025-1026).
- PAR일 때 FP의 cross-attention 응답 `fp_QKcos_list = save_dc_Qout_list(...)` 저장 (recon.py:1033, save_dc_Qout_list=934-955). hook 대상 = `mask_decoder.transformer`의 출력[0] (recon.py:935-949).
- `final_attn_token_to_image` 레이어는 loss가 크므로 iters=10000으로 별도, 일반 blk 재구성으로 처리 (recon.py:1012-1016).
- 최적화 루프(recon.py:1111-1213): 매 iter `get_dc_Qout`로 양자화 응답 `cur_QK_cos` 획득 (recon.py:1180-1187, get_dc_Qout=982-1007).
  - **loss = LossFunction(cur_quant_oup, cur_fp_oup, cur_QK_cos, fp_QKcos_list[idx])** (recon.py:1189).
  - `recon_loss='decoder'` 분기 (recon.py:340-346): **total_loss = beta·rec_loss + alpha·CAM_sim_loss** (alpha=beta=1 기본, test_quant_SAQ_m.py:271-281). rec_loss는 출력 MSE(tuple이면 `0.01·lp(out0)+lp(out1)`), CAM_sim_loss는 상호작용 응답 정렬.

**(D) PAR loss 정의** — `LossFunction`/`lp_loss` (recon.py:266-382):
- `lp_loss(pred,tgt,p,sum_dim=-1) = mean(∑_sum_dim |pred-tgt|^p)` (recon.py:378-382).
- `CAM_loss=='ori'`이면 PAR 비활성 → 순수 출력 MSE (recon.py:353-356). `'PAR'`이면 위 (B)/(C) 분기.
- **prompt 상호작용을 어떻게 loss에 반영?** 양자화 image token을 **실제 FP mask decoder의 cross-attention(`transformer`)에 통과**시킨 응답을, 동일 prompt(sparse/dense embedding, image_pe)로 만든 FP 응답과 비교 (recon.py:497-518, 982-1007). 즉 "prompt가 주어졌을 때 image token이 만드는 상호작용 결과"를 정렬 대상으로 삼아, 단순 feature MSE를 넘어 prompt-aware 의미 정렬 달성.

### 3.6 Layer-skipping (encoder image token) — ★ SAQ-SAM 고유 기여 3
README는 "상호작용 효율을 위해 encoder의 image token에 대한 layer-skipping 전략 설계"라고 함 (README.md:17). 코드상 명시적 "토큰을 건너뛰는 layer skip"은 다음 형태로 나타남:
- **image token 서브샘플링(16개)**: PAR 상호작용 계산에서 image token/prompt를 16개로 제한.
  - `save_sam_inputs`의 `max_pt=16` (recon.py:387, 422-428): prompt embedding 16개로 랜덤 선택.
  - `save_CIT_list`/`get_CIT`/`predict_calib_recon`의 `calib_num=16` (recon.py:484, 508; mask_decoder.py:160, 171-173 주석). mask_decoder의 `predict_calib_recon`은 (현재 주석 처리된) `sparse/dense` 토큰을 calib_num로 자르는 로직을 보유 (mask_decoder.py:170-174). **[코드확인]** 현재는 그 슬라이싱이 주석이고, prompt 수 제한은 save 단계의 max_pt에서 수행.
- **stage 구획(layer 묶음)**: `QuantImageEncoderOurViT`(quant_model.py:647-680)는 encoder block들을 window_size>0(local)·global attn 경계로 **Stage 단위로 묶음** (quant_model.py:660-666). 재구성을 stage 단위로 수행(per_stage)함으로써 layer별 개별 처리를 건너뛰고 효율화. (실제 디폴트 forward는 `QuantImageEncoderOurViT_ori`(:684-709)가 block 순차 실행 — "용한"이라는 중국어 주석, :683.)
- **[추정]** README가 말하는 "layer-skipping for image tokens"의 본질은, mask decoder 상호작용 supervision을 만들 때 전체 image token이 아니라 일부(16)만 통과시켜 PAR 비용을 줄이는 것 + encoder를 stage로 묶어 재구성 단위를 줄이는 것의 결합으로 보임. 본 repo 소스에는 "encoder의 특정 transformer layer 자체를 추론 시 스킵"하는 코드는 발견되지 않음 (encoder forward는 모든 block 실행, quant_model.py:704-707, image_encoder.py:112-113). **[확인불가]** 논문에서 말하는 layer-skip이 추론 연산량 절감용 layer drop인지, 위 supervision 효율화인지는 소스만으로 단정 어려움.

### 3.7 SAM image encoder ViT 구조 연결 (modeling)
- **ImageEncoderViT** (image_encoder.py:17-116): patch_embed → +pos_embed → depth개 Block → neck(Conv 1x1 → LN → Conv 3x3 → LN, :89-105). window attention(local)과 global_attn_indexes 블록 혼합 (:84). 입력 1024×1024, patch 16 → 64×64 token, embed 768(B), out 256.
- **양자화 래핑** (quant_model.py):
  - `QuantImageEncoderOurViT(_ori)`: blocks → `QunatEncoderOurBlock`, neck → `QuantNeck` (quant_model.py:693-697).
  - `QunatEncoderOurBlock` (quant_model.py:711-739): norm1 → `QuantEncoderOurAttentionBlock`(attn) → window unpartition → residual → mlp. window_size 처리 (:726-734).
  - `QuantEncoderOurAttentionBlock` (quant_model.py:741-789): qkv를 `PreQuantizedLayer`로, q/k/v/softmax 출력에 post-act quantizer 부착. q·k에 scale·softmax 후 V matmul. rel_pos 지원 (:781-782). AGQ면 softmax에 Log 양자화 (:757-762).
- **mask decoder cross-attention** (transformer.py):
  - `TwoWayTransformer`(:16-119): depth개 `TwoWayAttentionBlock` + `final_attn_token_to_image`. forward는 (queries=prompt token, keys=image token) 양방향 (:62-119).
  - `TwoWayAttentionBlock`(:122-202): ① self-attn(token) ② cross_attn_token_to_image ③ MLP ④ cross_attn_image_to_token. 4단계 (:164-202).
  - 양자화판 `QuantDecoderOurTwoWayAttentionBlock`(quant_model.py:214-271): 동일 구조이나 각 attn이 `QuantDecoderOurAttentionBlock`이고 입력을 `(q,k,v)` 튜플로 받음. self_attn에서 ATM_feature 반환(:242-245), 각 attn이 `(QK, q,k,v)`를 함께 반환해 PCC/재구성에서 사용 (quant_model.py:501,537).
  - `QuantDecoderOurAttentionBlock`(quant_model.py:275-547): q/k/v/out proj를 `PreQuantizedLayer`로, K activation은 BIG면 `LSQSignFakeQuantize`(sign config) (:306). PCC의 AttnRank 분기 forward 보유 (:426-537).
- **predict_calib_recon 신설** (mask_decoder.py:154-203): 일반 `predict_masks`(:112-152)에서 mask 생성 후단(upscaling/hypernet)을 제거하고 `transformer`까지만 실행 → PAR이 cross-attention 응답만 필요하기 때문. PAR supervision 추출의 진입점 (recon.py 전반에서 호출).

---

## 4. 알고리즘/수식 (코드에서 유도)

### 4.1 기본 affine 양자화 (util_quant.py:11-15, observer.py:50-69)
- 양자화/역양자화: `x_q = clamp(round(x/s)+z, qmin, qmax)`, `x̂ = (x_q - z)·s` (util_quant.py:12-14).
- asym: `s = (max⁺ - min⁻)/(qmax-qmin)`, `z = clamp(qmin - round(min⁻/s), qmin, qmax)` (observer.py:65-68).
- sym: `s = max(|min⁻|, max⁺) / ((qmax-qmin)/2)`, `z=0` (observer.py:60-63).
- round는 STE: `round_ste(x)=(round(x)-x).detach()+x` (util_quant.py:4-8).

### 4.2 AGQ softmax 로그 양자화 (fake_quant.py:681-691, observer.py:459-470)
- `x_int = round_ste(-log2(x/s)·τ)`, `x̂ = s·2^(-x_q/τ)`, `x_q = clamp(x_int, 0, levels-1)`; level 초과(softmax_mask)는 0 (fake_quant.py:684-689).
- τ는 후보 `{1,2,4}`(observer.taus, observer.py:455) 중, **softmax×V 결과의 오차**를 최소화하는 값 선택: `error = lp_loss(x_q@V, x@V, p=2.4)` (observer.py:469).

### 4.3 PCC 클리핑 기준식 (quant_model.py:67-86, 397-414)
- focus mask: `M(A) = [A > threshold · max_row(A)]` (threshold 기본 0.5), row max는 dim=-1 (quant_model.py:67-71).
- attention overlap (IoU): `IoU = ∑ min(M(A_q), M(A_fp)) / (∑ max(M(A_q), M(A_fp)) + ε)` (quant_model.py:73-83).
- PCC 목적함수: 클리핑 비율 pct에 대해
  `pct* = argmin_{pct ∈ P} (1 - IoU( M(Â_pct), M(A_fp) ))`
  여기서 Â_pct = 분위수 [1-pct, pct]로 클리핑·양자화한 QK로 만든 attention, P = {0.85,…,0.99999} (quant_model.py:397-414).
- 선택된 클리핑 범위: `clip_r=quantile(act, pct*)`, `clip_l=quantile(act, 1-pct*)` → observer min/max로 강제 (quant_model.py:363-385, fake_quant.py:228-239).
- (calib 누적식, recon에서 호출 시: `clip ← (clip·cnt + new)/(cnt+1)` 이동평균, quant_model.py:377-381 — 단 search_range_AOL 본 호출은 calib=False로 단발.)

### 4.4 PAR loss 수식 (recon.py:300-382)
- 공통: `lp_loss(p,t) = mean_batch( ∑_{last dim} |p−t|^p )`, p=2 (recon.py:378-382).
- Encoder(CAMS, recon_loss='encoder', recon.py:335-339):
  `L_enc = lp_loss( CIT_q, CIT_fp )` — CIT = mask decoder cross-attention을 통과한 cross image token. (출력 MSE는 계산만 하고 total에서 제외.)
- Decoder(CLM, recon_loss='decoder', recon.py:340-346):
  `L_dec = β · L_rec + α · lp_loss( R_q, R_fp )`, α=β=1.
  - `L_rec = lp_loss(out, fp_out)` (tuple이면 `0.01·lp(out₀)+lp(out₁)`), R = transformer cross-attention 응답.
- `CAM_loss='ori'`: `L = lp_loss(out, fp_out)` (순수 출력 MSE, QDrop와 동일).

### 4.5 AdaRound (weight) (fake_quant.py:575-597)
- round mask: `h(α) = clamp(σ(α)·(ζ−γ)+γ, 0, 1)`, γ=−0.1, ζ=1.1 (fake_quant.py:553, 575-578).
- forward: `x̂ = (clamp(⌊x/s⌋ + h(α) + z, qmin, qmax) − z)·s` (fake_quant.py:589-596). 재구성에서 α 학습 후 hard(`α≥0`)로 확정.

---

## 5. 학습/평가 파이프라인

- **데이터셋**: COCO (annotations/train2017/val2017/test2017) (README.md:60-70). calibration은 COCO train에서 32장 (config_SA_*.yaml:25 `calibrate:32`, utils.py:115-157). 평가는 COCO segm(instance segmentation) 기본 (`--eval segm`, test_quant_SAQ_m.py:111-117).
- **태스크**: instance segmentation(주), oriented object detection, semantic segmentation (README.md:18). detector: Faster R-CNN/YOLOX/DETR/DINO가 box prompt 생성, SAM이 mask 생성.
- **체크포인트**: SAM ViT-B/L/H + detector weights를 `ckpt/`에 배치 (README.md:72-80).
- **실제 명령어** (README.md:84-100):
  - SAQ-SAM* (PCC만, 재구성 없음):
    ```
    python ptq4sam/solver/test_quant_SAQ_m.py \
      --quant-encoder --quant-decoder \
      --config='./projects/configs/yolox/yolo_l-sam-vit-b.py' \
      --q_config='./exp/config_SA_66.yaml' \
      --FFC \
      --recon_encoder='no' --recon_decoder='no'
    ```
  - SAQ-SAM (PCC + PAR 재구성):
    ```
    python ptq4sam/solver/test_quant_SAQ_m.py \
      --quant-encoder --quant-decoder \
      --config='./projects/configs/yolox/yolo_l-sam-vit-b.py' \
      --q_config='./exp/config_SA_66.yaml' \
      --FFC \
      --recon_encoder='CAMS' --recon_decoder='CLM' --CAM_loss='PAR'
    ```
- **비트 설정**: W6A6 = `config_SA_66.yaml`, W4A4 = `config_SA_44.yaml` (README.md:101-102, config 파일). 재구성 하이퍼파라미터: scale_lr=0.002, weight_lr=0.001, warm_up=0.2, iters=iters_CAMS=2000, drop_prob=0.5, b_range=[20,2] (config_SA_66.yaml:26-39).
- **자원**: 40GB+ GPU 권장. huge/large는 `--gpu2`로 2-GPU 재구성 (README.md:111, test_quant_SAQ_m.py:601-602).
- **시각화**: `--show --show-dir` (README.md:112).
- **로그/평가**: mmdet `single_gpu_test`/`multi_gpu_test`로 추론 후 `dataset.evaluate(metric=segm)` (test_quant_SAQ_m.py:688-737).

---

## 6. 의존성
- **PyTorch 1.10.2 + cu113**, torchvision 0.11.3 (environment.sh:1). README는 py3.7 권장 (README.md:28).
- **mmcv-full < 2.0.0**, **mmdet**(빌드 설치) (environment.sh:2-3,8-9, README.md:32-37,53-58).
- **CUDA ops**: `projects/instance_segment_anything/ops` 컴파일 (environment.sh:5-7).
- 라이브러리: scipy(`gaussian_kde`, `find_peaks`, `minimize_scalar`), numpy, pandas, easydict(util config), wandb(선택), mmcv runner. (fake_quant.py:15-16, observer.py:4,10-11, utils.py:5, test_quant_SAQ_m.py:46).
- 코드 출처: PTQ4SAM(베이스), Prompt-Segment-Anything(SAM 어댑터), QDrop(재구성 drop) (README.md:140).

---

## 7. 강점 / 한계 / 리스크

**강점**
- PCC가 단순 MSE/percentile observer를 넘어 **attention focus 일치도(IoU)**를 직접 목적함수로 삼음 → 저비트(특히 4-bit)에서 의미 보존하며 공격적 클리핑 (quant_model.py:349-418). SAM-B 4-bit에서 baseline +11.7% mAP 주장 (README.md:19).
- PAR가 feature MSE가 아니라 **prompt-conditioned cross-attention 응답**을 정렬 → image-prompt 상호작용까지 고려한 task-aware 재구성 (recon.py:497-518, 982-1007).
- PTQ4SAM의 BIG(bimodal sign)·AGQ(log softmax)를 그대로 흡수해 SAM 특유의 K-activation bimodal·softmax long-tail 문제도 동시 대응.
- W/A를 encoder/decoder 분리 config로 관리 (config_SA_*.yaml).

**한계 / 리스크**
- **dropout(`drop_prob`) 버그를 의도적으로 미수정 상태로 둠** (test_quant_SAQ_m.py:674-686 주석, README.md:116-124). 평가 시 `drop_prob`이 1.0으로 복원되지 않아 activation 절반이 미양자화 → **재구성(PAR) 사용 결과가 실제보다 좋게 나올 수 있음**. PCC-only(SAQ-SAM*)는 영향 없음. 공정 비교를 위해선 `m.drop_prob=1` 복원 코드 주석 해제 필요 (README.md:119-122). 논문 수치 재현과 "공정" 수치가 다를 수 있는 중대한 리스크.
- 코드 정돈도 낮음: 중국어 주석, 절대경로 하드코딩(`sys.path.append("/home/zhangjing/...")` test_quant_SAQ_m.py:28, recon.py:16; `--show-dir` 기본값이 개인 경로 test_quant_SAQ_m.py:120,157), 미사용/중복 함수(`reconstruction_IE_PB` recon.py:730-931, `QuantDecoderOurAttentionBlock_ori` quant_model.py:550-635) 다수.
- **layer-skipping의 코드 근거가 약함**: 추론 시 실제 layer를 스킵하는 로직은 미발견. supervision용 token 16-서브샘플 + stage 묶음으로 해석됨(3.6). 논문 주장과 코드 매핑이 불명확. **[확인불가]**
- 매우 무거운 SAM 전체(encoder ViT-H까지) + 40GB GPU 요구. 재구성은 FP/quant 모델 동시 보유 + 2GPU 필요 (test_quant_SAQ_m.py:601-635).
- `quant_rep`/`repq_utils`는 메인과 분리된 죽은(혹은 실험) 경로 → 혼동 유발.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적)

- **SAM image encoder는 표준 ViT**(patch_embed + window/global attention block + neck, image_encoder.py:17-116). 따라서 **어텐션 양자화/클리핑 기법은 우리 ViT 가속기로 전이 가능**. 특히 q/k/v/softmax post-activation을 개별 quantizer로 분리하는 구조(quant_model.py:764-766)는 FPGA에서 어텐션 경로별 비트/스케일을 다르게 두는 설계와 잘 맞음.
- **PCC 같은 outlier-aware 클리핑은 FPGA 저비트(W4A4) 정확도에 핵심**. attention-IoU 기반 클리핑은 "의미 보존하며 좁은 동적 범위" → 작은 scale → 고정소수점/INT4 표현에 유리. 다만 PCC는 calibration-time 탐색(29개 pct × attention forward)이라 **추론 HW 비용은 0**(런타임엔 고정 clip 범위만 적용, fake_quant.py:242-243). FPGA 관점에서 이상적: 오프라인 비용으로 온라인 정확도 확보.
- **AGQ의 softmax log 양자화**(`2^(-x_q/τ)`, fake_quant.py:684-689)는 FPGA에서 시프트 연산으로 구현 가능 → softmax 후단을 곱셈 없이 처리하는 가속기 설계 아이디어로 직접 활용 가치 있음.
- **layer-skipping(연산량 절감 관점)**은 흥미로우나, 본 repo에선 명확한 추론 layer-skip 구현이 없어 직접 차용할 코드 자산은 제한적. token 서브샘플링 아이디어 정도가 참고 가능. **[추정]** XR 시선추적의 실시간성 요구를 고려하면 "토큰/레이어 동적 스킵"은 방향성은 맞지만 본 repo는 청사진 수준.
- **직접성은 낮음**: SAM 자체가 매우 무거운 모델(ViT-B/L/H + two-way decoder)이라 XR 엣지/FPGA에 그대로 올리긴 부적합. 우리에게 유효한 것은 **(a) PCC 클리핑 철학, (b) 어텐션 경로별 양자화 분리, (c) softmax log 양자화** 등 기법 단위 전이이며, 모델/파이프라인 통째 차용은 비권장.

---

## 9. 근거표기 요약 (코드확인 vs 추정/확인불가)

- **[코드확인]**: PCC가 AttentionOverlapLoss(1-IoU)로 pct 클리핑 범위 탐색(quant_model.py:349-418), `--FFC`가 calibrate의 PCC 분기 트리거(test_quant_SAQ_m.py:789-804), PAR이 mask decoder cross-attention 응답 정렬(recon.py:335-346,497-518,982-1007), CAMS=encoder/CLM=decoder 재구성, W/A 비트·observer·quantizer 매핑(config_SA_*.yaml), softmax AGQ log 양자화, BIG의 KDE bimodal 판정(fake_quant.py:307-319), dropout 버그 미수정(test_quant_SAQ_m.py:674-686), quant_rep/repq_utils가 메인 미사용(Grep).
- **[추정]**: README의 "layer-skipping for image tokens"이 실제로는 PAR supervision의 token 16-서브샘플(recon.py:387,484) + encoder stage 묶음(quant_model.py:660-666)의 결합일 것; PCC가 단순 observer보다 작은 scale 유도.
- **[확인불가]**: 추론 시 transformer layer 자체를 건너뛰는 동적 layer-drop이 코드에 있는지(미발견 — encoder forward는 전 block 실행). 논문 수치가 dropout 버그 미수정 상태의 것인지 수정본인지(README는 "수정 후 업데이트 예정"이라고만 명시, README.md:124).
- **불명확 시 솔직 명시**: layer-skipping의 정확한 정의/구현 위치는 본 repo 소스만으로 단정 불가. PCC는 KDE를 사용하지 않음(과제 가설과 다름 — 코드상 부정).
