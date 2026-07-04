# QuantVLA 정밀 분석 (Vision-Language-Action 모델 PTQ: DuQuant W4A8 + ATM + OHB)

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\QuantVLA`
> 작성일: 2026-06-20 / 실제 소스 코드 기반. 라인 근거(파일:라인) 표기.
> **제외**: Isaac/gr00t 외부 대용량 사전학습 가중치, demo_data(parquet/mp4), `.git/`, `__pycache__/`.
> **양자화 코드 유무: 있음** — `gr00t/quantization/`(DuQuant W4A8), `gr00t/atm/`(ATM + OHB) 자체 구현 존재.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **목적**: NVIDIA **GR00T N1.5 (3B) VLA 모델**을 **학습 불필요(training-free) PTQ**로 저비트 양자화하여 메모리·지연을 줄이면서 작업 성공률 유지. (README.md:5, 34-36)
- **원논문**: *QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models*, Jingxuan Zhang 외, **CVPR 2026** (arXiv 2602.20309). VLA 최초 PTQ, **diffusion transformer(DiT) action head를 양자화한 최초** 사례 주장. (README.md:5-13, 34-36)
- **핵심 3요소(scale-calibrated components)** (README.md:31, 36):
  1. **Selective quantization layout**: 언어 백본(Eagle VLM)과 DiT의 **모든 Linear를 정수화**하되, **attention projection은 FP 유지**(원래 operator schedule 보존). (코드상: include 정규식에서 LLM q/k/v/o/gate/up/down_proj + DiT ff만, attn1 제외; calibrate_atm_simpler_perhead_ohb.py:46-47)
  2. **Attention Temperature Matching (ATM)**: per-head 스케일(alpha)로 attention logit 분산을 안정화, 추론 시 dequant 스케일에 folding. (dit_atm.py:111-119, README.md:36)
  3. **Output Head Balancing (OHB)**: per-layer/per-head 잔차 인터페이스 보정(beta)으로 projection 후 에너지 드리프트 완화. (dit_atm.py:121-148)
- 결과: LIBERO에서 FP baseline 성공률 상회, 양자화 컴포넌트 약 70% 메모리 절감, 1.22× 속도. (README.md:22, 36)

---

## 2. 디렉토리 구조 (자체 소스, 대용량 외부 제외)

```
QuantVLA/
├── gr00t/
│   ├── quantization/                # ★ DuQuant W4A8 fake-quant (자체 핵심)
│   │   ├── duquant_layers.py        # ★ DuQuantLinear, DuQuantConfig, select_targets, wrap_duquant
│   │   └── duquant_preprocess.py    # ★ pack_weight(회전/순열), fake_quantize_sym, PercentileCalibrator, MSE scale
│   ├── atm/dit_atm.py               # ★ ATM(temperature) + OHB(per-head balancing) 프로세서/calib
│   ├── model/
│   │   ├── policy.py                # Gr00tPolicy: 양자화/ATM 적용 진입점(load 시 wiring)
│   │   ├── gr00t_n1.py              # GR00T-N1.5 모델 정의
│   │   ├── action_head/             # flow_matching_action_head, cross_attention_dit (DiT)
│   │   └── backbone/eagle_backbone.py, eagle2_hg_model/ (Eagle2 VLM: SigLIP ViT + Qwen3 LLM)
│   ├── data/, experiment/, eval/, utils/
├── deployment_scripts/              # ★ export_onnx.py, trt_torch.py, trt_model_forward.py
├── tools/                           # ★ calibrate_atm_*.py (ATM/OHB calibration), memory/speedup 분석
│   ├── calibrate_atm_simpler.py, calibrate_atm_simpler_perhead_ohb.py, calibrate_atm_dit.py
│   ├── calc_gr00t_duquant_memory.py, visualize_speedup.py, visualize_atm_*.py
├── scripts/                         # gr00t_finetune, eval_policy, scan_linear_layers, *_service
├── examples/ (Libero, SO-100, RoboCasa), getting_started/, environments/
```

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `gr00t/quantization/duquant_preprocess.py` — DuQuant 전처리/양자화 커널 (가장 중요)

#### (a) `fake_quantize_sym(x, scale, bits)` (duquant_preprocess.py:120-137)
- **symmetric fake quantization**: `max_q = qmax(bits)=(1<<(bits-1))-1`, `x_clamped = clamp(round(x/scale), -max_q-1, max_q)`, 반환 `x_clamped*scale`. (duquant_preprocess.py:130-134)
- INT4면 qmax=7(범위 -8..7), INT8이면 127(-128..127). dequant까지 한 번에 수행하는 **STE 없는 inference-only fake-quant**.
- `_DuQuantProfiler`(duquant_preprocess.py:31-117)로 양자화 호출 시간/GB·s 프로파일(환경변수 `OPENPI_DUQUANT_PROFILE`).

#### (b) `pack_weight(W, block_size, block_out_size, enable_permute, lambda_smooth)` (duquant_preprocess.py:206-309) — DuQuant 회전/순열 사전계산
- **채널 에너지 평활**: `channel_energy = mean(W^2, axis=0)`, `lambda_smooth`(기본 0.15)로 평균과 혼합. (duquant_preprocess.py:219-222)
- **zigzag permutation**(duquant_preprocess.py:188-203): 에너지 내림차순 정렬 후 양끝에서 번갈아 뽑아 큰 값 분산(outlier 분산). `enable_permute`로 on/off.
- **입력측 블록 회전 R_in**(duquant_preprocess.py:226-239): block_size(기본 16, calib 스크립트는 64) 단위로 `compute_block_rotation`(W_block^T의 SVD U; GPU torch.linalg.svd, fallback NumPy; duquant_preprocess.py:158-185) → 직교행렬 R_in.
- **출력측 블록 회전 R_out**(duquant_preprocess.py:255-289): 행 블록(block_out_size) 단위 SVD U → R_out.
- **per-output-channel symmetric scale**(duquant_preprocess.py:291-293): 변환된 W_t의 row max-abs / qmax(4) (기본 4비트 가정, 레이어가 override).
- 결과 `PackResult`(R_in_blocks, perm, R_out_blocks, weight_scale, meta; duquant_preprocess.py:140-149) → `save_pack`/`load_pack`로 `.npz` 캐시. (duquant_preprocess.py:494-551)

#### (c) `compute_mse_scales(W, bits)` (duquant_preprocess.py:474-491) — MSE 그리드 스케일
- max-abs 기준 후보 스케일 grid `alphas=[0.5,0.75,1.0,1.25,1.5]`(duquant_preprocess.py:466-471)로 양자화-복원 MSE 최소화하는 per-output scale 선택. → 가중치 스케일을 단순 max-abs가 아닌 **MSE 최적**으로.

#### (d) `PercentileCalibrator` (duquant_preprocess.py:429-463) — 활성 observer
- per-channel(마지막 dim) **percentile(기본 99.9%)** 캘리브레이터. `observe`에서 배치별 `torch.quantile(|x|, p)`의 **running max**를 누적(duquant_preprocess.py:442-454), `max_batches`(기본 32) 도달 시 finalize. → outlier-robust per-channel 활성 스케일.

#### (e) transform/restore 함수군
- `apply_input_transform[_optimized]`(312-337, 560-607): 입력 x에 perm → block R_in 적용(x @ P @ R_in). optimized는 einsum 배치 블록 matmul(duquant_preprocess.py:590-595).
- `transform_weight_for_forward[_optimized]`(340-386, 652-703): W에 perm·R_in(열) + R_out(행) 적용 후 MSE/ones scale 산출.
- `apply_output_restore[_optimized]`(389-408, 610-649): 출력 y에 R_out 우곱으로 원 basis 복원(row_rot_mode='restore'일 때).
- `apply_bias_row_rot[_optimized]`(411-426, 706-728): bias에 R_out 적용(row_rot_mode='propagate').

### 3.2 `gr00t/quantization/duquant_layers.py` — DuQuant Linear/배치 (가장 중요)

#### `DuQuantConfig` (duquant_layers.py:31-70)
- 환경변수 기반 설정(인스턴스 시점 read): `weight_bits`(GR00T_DUQUANT_WBITS_DEFAULT, 기본 **4**), `act_bits`(ABITS, 기본 **8**) → **W4A8**. `block_size`(16), `lambda_smooth`(0.15), `enable_permute`, `act_percentile`(99.9), `calib_batches`(32), `row_rot_mode`(restore), `block_out_size`. (duquant_layers.py:51-70)

#### `DuQuantLinear(nn.Module)` (duquant_layers.py:90-340)
- `__init__`(93-168): 원 Linear의 weight를 버퍼로 복제, `load_pack` 없으면 `pack_weight`로 회전/순열 사전계산 후 저장(106-115). R_in/R_out/perm을 buffer로 캐시(118-138). `PercentileCalibrator` 생성(145-147). `_W_t`/`_w_scales`/`_W_t_quantized`(precache) 버퍼(153-164).
- `_maybe_update_weight_cache`(196-247): 디바이스/dtype/bits 변경 시 `transform_weight_for_forward_optimized`로 W 변환+scale, precache 시 `fake_quantize_sym`으로 **가중치를 미리 INT4 양자화**해 캐시(219-224).
- `_get_act_scale`(248-283): calibrator로 활성 percentile 관측→full이면 `scale = clamp(p_vec/qmax(act_bits), 1e-6)`로 per-channel 활성 스케일 산출(259-267). 캘리브 미완 시 현재 배치 quantile fallback(269-281).
- **`forward(x)`** (285-340):
  1. `apply_input_transform_optimized`로 입력에 perm·R_in 적용(288-290).
  2. act_bits>0이면 `fake_quantize_sym(x_t, s_a, act_bits)`로 **활성 A8 fake-quant**(293-295).
  3. `_maybe_update_weight_cache()` → precache된 INT4 가중치로 `F.linear`(300-302) 또는 즉시 `fake_quantize_sym` 가중치(303-313).
  4. row_rot_mode='restore'면 `apply_output_restore_optimized`로 출력 복원 후 bias(318-324); 'propagate'면 회전된 bias 사용(325-332).

#### 레이어 선택/치환
- `select_targets`(351-378): include/exclude 정규식으로 양자화 대상 Linear 선택. 기본 include = `q_proj|k_proj|v_proj|out_proj|fc1|fc2|up_proj|down_proj|gate_proj`, exclude = `norm|ln|layernorm|emb`. (duquant_layers.py:354-355)
- `wrap_duquant`(381-429): 선택 레이어를 `DuQuantLinear`로 in-place 교체. **action_head/action_out_proj은 기본 제외**(GR00T_DUQUANT_INCLUDE_ACTION_HEAD=0; duquant_layers.py:393-401). per-layer wbits override 가능. dry_run 지원.
- `enable_duquant_if_configured(model)`(432-509): `GR00T_DUQUANT_*` 환경변수 있으면 활성. **기본 include = LLM(Eagle) language_model의 q/k/v/o/gate/up/down_proj + DiT action_head.model의 attn1.to_q/k/v/to_out + ff.net.0.proj/2**, exclude = vision_model/radio/norm/embed/lm_head/encoder류. (duquant_layers.py:454-472) → §1의 "selective layout" 구현.

### 3.3 `gr00t/atm/dit_atm.py` — Attention Temperature Matching + Output Head Balancing

#### `_ATMProcessor(AttnProcessor2_0)` (dit_atm.py:40-158) — diffusers DiT attention 대체
- forward에서 to_q/k/v 후 멀티헤드 reshape(dit_atm.py:77-92).
- **logit 통계 캡처**: `_compute_logits_std`(dit_atm.py:161-188)로 QK^T logit의 per-head std 계산(ATM alpha 캘리브용). capture 콜백으로 외부 수집(dit_atm.py:99-109).
- **ATM 적용**(dit_atm.py:111-119): `alpha = _atm_alpha_all` (per-head)을 `query *= alpha`로 곱 → logit temperature 조정 후 `scaled_dot_product_attention`.
- **per-head OHB**(dit_atm.py:121-133): attention 출력(reshape 전)의 per-head RMS 캡처(`_compute_rms_per_head`, dit_atm.py:195-208), `_ohb_beta_perhead`로 `hidden_states *= beta_perhead`.
- **per-layer OHB**(dit_atm.py:141-148): to_out 후 RMS 캡처, `_ohb_beta_scalar`로 스칼라 곱.
- 즉 ATM/OHB는 가중치 양자화 자체가 아니라, **양자화로 인한 attention logit 분산/출력 에너지 드리프트를 per-head 스케일로 보정**하고 추론 시 dequant 스케일에 folding하는 보정층.

#### 패칭/등록/활성
- `ensure_dit_attention_patch`(211-219): DiT(`action_head.model.transformer_blocks`) attention의 processor를 `_ATMProcessor`로 교체. (`_is_dit_attention`, dit_atm.py:30-37)
- `register_atm_capture`/`register_ohb_perhead_capture` 등(222-275): calibration용 콜백 등록.
- `enable_dit_atm_if_configured`(305-379): `GR00T_ATM_ENABLE`/`GR00T_OHB_ENABLE` + alpha JSON(`GR00T_ATM_ALPHA_PATH`) 로드 → 각 DiT attention에 alpha(per-head)·beta(per-head/per-layer) 주입. (dit_atm.py:336-365)

### 3.4 `gr00t/model/policy.py` — 양자화 적용 진입점
- `Gr00tPolicy` 로드 시(policy.py:278-297):
  1. `ensure_dit_attention_patch`로 DiT attention을 ATM-aware로 (282).
  2. `enable_duquant_if_configured(model)` — DuQuant W4A8 적용 (288-289).
  3. `enable_dit_atm_if_configured(model)` — ATM/OHB 스케일 적용 (295). action_head 재생성 후 호출되어 DiT까지 양자화 보장(276-286).

### 3.5 `tools/calibrate_atm_simpler_perhead_ohb.py` — ATM/OHB 캘리브레이션 (대표)
- LIBERO/SimplerEnv 환경에서 teacher(FP) vs quant 모델 forward를 32스텝 수집, **per-head alpha(logit std 매칭) + per-head beta(512=16블록×32헤드, OHB)** 계산해 JSON 출력. (calibrate_atm_simpler_perhead_ohb.py:1-19, 82-88)
- 기본 DuQuant 환경(W4/A8, block 64, permute off, row_rot restore, act_pct 99.9, calib 32)을 dict로 세팅(calibrate_atm_simpler_perhead_ohb.py:44-56). DuQuant include는 LLM q/k/v/o/gate/up/down + DiT `ff.net.0.proj/2`(attn1 제외). (line 46-47)
- OHB log-domain clamp(0.30)/neutral(0.03)로 beta 산출(line 84-87).

### 3.6 `deployment_scripts/export_onnx.py` — 배포 익스포트 (양자화 무관 FP16 경로)
- GR00T를 3개 ONNX로 분할 export: **Eagle2 ViT(SigLIP)**(export_eagle2_vit, export_onnx.py:49-141), **Eagle2 LLM(Qwen3)**(export_eagle2_llm, 144-238, attn=eager, flash 미사용), **action head**(vlln/vl_self_attention, state/action encoder, **DiT**, action_decoder; 253-385). 모두 **FP16**(torch.float16), opset 19. (export_onnx.py:115, 128-141)
- → ONNX/TRT 배포 경로는 현재 **FP16 분할 export 중심**, DuQuant INT4/INT8 정수 커널을 ONNX/TRT로 내보내는 코드는 **확인 불가**(fake-quant는 PyTorch 런타임 내 시뮬레이션). `trt_torch.py`/`trt_model_forward.py`는 TRT 추론 래퍼.

---

## 4. 알고리즘 / 수식 — 적용 양자화 알고리즘

### 4.1 DuQuant W4A8 (training-free PTQ)
- **가중치(W4)**: per-output-channel symmetric. 변환 `W' = R_out · (P·W·R_in)` (perm P, 블록 직교회전 R_in/R_out; SVD로 산출, outlier 평탄화). 스케일 `s_w = compute_mse_scales(W', 4)` (MSE 그리드). 양자화 `Ŵ = clamp(round(W'/s_w), -8, 7)·s_w`. (duquant_preprocess.py:206-309, 474-491; fake_quantize_sym duquant_preprocess.py:120-134)
- **활성(A8)**: per-channel percentile(99.9%) 스케일 `s_a = clamp(p99.9(|x'|)/127, 1e-6)`, `x' = P·R_in·x`(입력 동일 회전). 양자화 `x̂ = clamp(round(x'/s_a), -128, 127)·s_a`. (duquant_layers.py:248-283)
- **회전 불변성**: R_in/R_out은 직교행렬이라 `(W·R_in)(R_in^T·x) = W·x` 보존, 출력은 `apply_output_restore`로 R_out 역회전. → 양자화 grid에 friendly한 basis로 변환 후 정수화.

### 4.2 ATM (Attention Temperature Matching)
- per-head alpha로 logit 스케일: `logit' = (alpha·Q)·K^T/√d`. alpha는 teacher logit std와 quant logit std를 매칭하도록 calibration(calibrate_atm). 추론 시 alpha는 query 스케일에 folding(dit_atm.py:111-119, 161-188).

### 4.3 OHB (Output Head Balancing)
- per-head/per-layer beta로 attention 출력 RMS 보정: `out' = beta·out`. beta는 log-domain에서 teacher/quant RMS 비로 산출, clamp/neutral 적용. (dit_atm.py:121-148, calibrate ...:84-87)

---

## 5. 학습/평가 파이프라인

- **모델**: NVIDIA GR00T-N1.5-3B (Eagle2 VLM = SigLIP ViT + Qwen3 LLM 백본 + flow-matching DiT action head). (export_onnx.py:24-33, README.md:47)
- **데이터/벤치마크**: LIBERO(시뮬), SO-100/RoboCasa/SimplerEnv. demo_data(parquet/mp4)는 대용량(이름만). (examples/*)
- **환경**: dual conda — `groot_test`(torch 2.5.1+cu124, transformers, diffusers, flash-attn 2.7.1, gr00t) + `libero_test`(LIBERO/robosuite/mujoco). (README.md:51-104)
- **양자화 실행(환경변수 기반)**:
  - DuQuant: `GR00T_DUQUANT_WBITS_DEFAULT=4 GR00T_DUQUANT_ABITS=8 GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_ROW_ROT=restore ...` 설정 후 policy 로드 → 자동 적용. (duquant_layers.py:432-509)
  - ATM/OHB calibration: `python tools/calibrate_atm_simpler_perhead_ohb.py --teacher-checkpoint ... --quant-checkpoint ... --steps 32 --out atm_alphas.json --calibrate-ohb 1` → 적용 시 `GR00T_ATM_ENABLE=1 GR00T_ATM_ALPHA_PATH=atm_alphas.json GR00T_OHB_ENABLE=1`.
- **배포**: `deployment_scripts/export_onnx.py`로 ViT/LLM/action-head FP16 ONNX 분할 → TRT(trt_torch.py). (getting_started/5_policy_deployment.md)
- **평가**: `scripts/eval_policy.py`, `examples/Libero/eval/run_libero_eval.py` (성공률).

---

## 6. 의존성
- torch 2.5.1+cu124, transformers(Qwen3/SigLIP), **diffusers**(DiT Attention/AttnProcessor2_0; dit_atm.py:10-11), flash-attn 2.7.1, numpy, gr00t. (README.md:84-104) LIBERO/robosuite/mujoco(eval). DuQuant는 OpenPI duquant 구현 차용(duquant_layers.py:1-6).

---

## 7. 강점 / 한계 / 리스크

**강점**
- VLA 최초 PTQ + **DiT action head 양자화** 최초 — LLM/DiT Linear까지 W4A8.
- DuQuant 회전/순열(R_in/R_out/perm)로 outlier 평탄화 → 저비트(W4) 정확도 확보, MSE 스케일·percentile observer로 robust.
- ATM/OHB로 **양자화 유발 attention 드리프트를 per-head 스케일 보정** + dequant folding(추가 op 없음).
- training-free, 작은 unlabeled calibration buffer(32스텝)만 필요. 70% 메모리 절감, 1.22× 속도 주장.
- 환경변수 기반 selective layout(attention proj FP 유지)으로 operator schedule 보존.

**한계 / 리스크**
- **fake-quant(시뮬레이션)**: `fake_quantize_sym`은 float로 round-clamp-rescale(duquant_preprocess.py:130-134) — 실제 INT4/INT8 정수 커널 추론이 아니라 PyTorch 내 시뮬. 실 정수 GEMM 커널/ONNX·TRT INT export는 **확인 불가**(export_onnx는 FP16).
- DuQuant pack은 레이어마다 SVD(GPU) 사전계산 + `.npz` 캐시 의존 → 초기 비용·디스크.
- 블록 회전/복원·perm·per-block matmul 등 forward 오버헤드(optimized einsum 있어도 추가 연산).
- ATM/OHB alpha/beta JSON은 환경/태스크별 calibration 필요(일반화 리스크).
- vision encoder(SigLIP)·embedding·lm_head는 비양자화(exclude) → 전체 모델 압축률 제한.
- 거대 의존 스택(diffusers+flash-attn+LIBERO/mujoco), 외부 GR00T 가중치 필요 → 재현 난도 높음.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적 — 추정)

- **참고적(멀티모달 VLA)**: 본 repo는 로봇 제어 VLA로 XR 시선추적과 직접 도메인은 멀다. 다만 양자화 기법은 ViT 가속기에 전이 가능.
- **DuQuant 회전(R_in/R_out)·zigzag perm**(duquant_preprocess.py:158-203): outlier를 직교회전/순열로 평탄화하는 전처리는 **FPGA INT4/INT8 가중치 양자화 전 단계**로 유용 — 정수 grid 효율을 높여 동일 비트폭에서 정확도 향상. 단, 회전행렬 matmul이 HW 추가 비용이므로 가속기에선 R을 가중치에 사전 folding하는 방식 검토.
- **per-channel percentile observer + MSE scale**(duquant_preprocess.py:429-491): FPGA용 scale/zp 산출 시 max-abs보다 robust한 기준 레퍼런스.
- **ATM/OHB의 dequant-folding per-head 스케일**(dit_atm.py:111-148): attention softmax 전후 per-head 스케일을 dequant에 folding하는 패턴은 가속기의 **per-head requantize 스테이지** 설계에 시사(qflash의 fixed-point rescale과 상보적).
- **selective layout**(attention FP, Linear INT): HW에서도 softmax/attention은 정밀 유지, GEMM만 정수화하는 하이브리드 데이터패스 설계 근거.

---

## 9. 근거 표기 규칙
- 모든 기술 주장은 (파일:라인) 근거. **"추정"**: §8 FPGA/XR 적용 해석, ATM의 "dequant folding" 효과(README 서술 + 코드 query 스케일 곱으로 추론).
- **확인 불가**: 실제 INT 정수 커널/ONNX·TRT INT8 export(현 export_onnx는 FP16), 70% 메모리·1.22× 속도 수치(README 주장, 코드로 직접 재현 미확인), 외부 GR00T 가중치 내부.
- **양자화 코드 유무: 있음** — `gr00t/quantization/`(DuQuant W4A8 fake-quant) + `gr00t/atm/`(ATM/OHB) 자체 구현 확인.
