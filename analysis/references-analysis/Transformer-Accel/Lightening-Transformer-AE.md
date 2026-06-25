# Lightening-Transformer-AE (HPCA'24 DOTA) 코드베이스 정밀 분석

> 분석 대상: `REF/Transformer-Accel/Lightening-Transformer-AE`
> 분석 방식: 전 소스 Read 직독 + 파일:라인 근거. bash 미사용(UNC), Glob/Grep/Read만 사용.
> 근거 표기 규칙: **[확실]** = 코드 직접 확인, **[추정]** = 코드 정황상 추론, **[확인 불가]** = 파일 미존재/렌더 불가.

---

## 1. 개요

- **무엇인가**: HPCA 2024 논문 *"Lightening-Transformer: A Dynamically-operated Optically-interconnected Photonic Transformer Accelerator"* (Hanqing Zhu 외, UT Austin/MIT)의 **Artifact Evaluation(AE) 패키지**다. 가속기 이름은 **DOTA** (Dynamically-Operated, optically-interconnected photonic Tensor core Accelerator). 루트 `readme.md:1-2`, `hardware_simulator/readme.md:1-7`에서 확인. **[확실]**
- **한 줄 요약**: **광학 신경망(ONN/photonic) 기반 Transformer 추론 가속기**를, (1) 광학 비이상성(noise)을 주입한 PyTorch 양자화 모델로 **정확도**를, (2) 광소자 device parameter 기반 behavior-level 시뮬레이터로 **area/power/energy/latency**를, (3) GPU 프로파일러로 **베이스라인**을 각각 평가하는 순수 Python 패키지다. **[확실]**
- **DOTA의 핵심 광학 코어**: MZI 변조기 + micro-comb(다파장 WDM) + 크로스바형 결합기(coupler) + 위상천이기(PS) + 차동 광검출기(PD)로 구성된 **DDOT(Dynamically-operated Dot-product unit) 크로스바**. `[Nh, Nx] × [Nw, Nx]` 행렬곱을 광 도메인에서 수행 (`photonic_crossbar.py:16-21` docstring). **[확실]**
- **광학 ONN 특성**: 연산을 빛의 간섭/세기로 수행 → 전자 곱셈기 불필요, 초고속(work_freq 5GHz), 그러나 (a) DAC/ADC가 면적·전력 지배(아래 6장), (b) 파장분산(WDM dispersion)·위상오차 같은 **아날로그 비이상성**이 정확도를 떨어뜨림 → SW 모델에서 noise injection으로 모델링.
- **AE 패키지 성격**: 학습/평가 가능한 알고리즘 코드 + 분석적(analytic) 하드웨어 시뮬레이터 + 재현 스크립트. **실제 칩 테이프아웃·실측이 아니라 device parameter 기반 추정**임을 readme가 명시(`hardware_simulator/readme.md:4` "behavior-level simulation"). **[확실]**
- **타깃 워크로드**: DeiT-T/S/B (ImageNet, 197 토큰), BERT-B/L (SST 분류, 128~384 토큰). `utils/model.py:9-15`. **[확실]**

---

## 2. 디렉토리 구조

```
Lightening-Transformer-AE/
├── readme.md                         # 루트: 3개 서브패키지·의존성 설명
├── HPCA24_LT_poster_v1_02.pdf        # 포스터 (pdftoppm 부재로 렌더 불가 [확인 불가])
│
├── software_model/                   # (1) 양자화+광학노이즈 DeiT/BERT 학습·평가
│   ├── main.py                       # 진입점: arg parsing, create_model, train/eval
│   ├── engine.py                     # 표준 DeiT train_one_epoch/evaluate 루프
│   ├── ops/
│   │   ├── _quant_base.py            # LSQ 양자화 베이스 클래스 (_Conv2dQ/_LinearQ/_ActQ)
│   │   ├── quantize.py               # ★ QuantLinear/QuantConv2d/QuantAct + 광학 노이즈 주입
│   │   ├── simulator.py              # ★ WDM 파장분산(coupler kappa) 오차 모델
│   │   ├── plot.py, __init__.py
│   ├── losses.py, samplers.py, augment.py, utils.py, hubconf.py, process_logs.py
│   ├── scripts/                      # 학습/평가/노이즈스윕/로그처리 .sh 4종
│   ├── (models/ datasets.py)         # ★ 미존재 → 외부 DeiT/timm 의존 [확인 불가, 4장]
│   ├── *.csv, logs/*.log             # 노이즈 스윕 결과 (제외: 이름만)
│
├── hardware_simulator/               # (2) 광학 코어 area/power/energy/latency 시뮬
│   ├── entry_area_power_profile.py   # ★ 진입점: 가속기 area/power 프로파일
│   ├── entry_energy_latency_workload.py  # ★ 진입점: 워크로드 energy/latency
│   ├── simulator_attn.py             # ★ MHA(QK^T, SV) energy/latency
│   ├── simulator_FFN.py              # ★ linear/FFN energy/latency
│   ├── hardware/
│   │   ├── photonic_core_base.py     # PhotonicCore 추상 베이스 (device param 로더)
│   │   ├── photonic_crossbar.py      # ★ DOTA 크로스바 코어 (PhotonicCrossbar)
│   │   ├── photonic_MZI.py           # MZI mesh 베이스라인 (PhotonicMZI)
│   │   ├── photonic_mrr_bank.py      # MRR bank 베이스라인 (PhotonicMRRBank)
│   │   ├── ADC.py, DAC.py, SRAM.py   # 디지털 인터페이스/메모리 모델
│   ├── utils/
│   │   ├── config.py                 # torchpack 스타일 YAML Config
│   │   ├── model.py                  # 모델→GEMM op-list 변환 (modelParams)
│   │   ├── cal_flops_for_transformer.py  # non-GEMM(softmax/LN/GELU) FLOPs
│   │   ├── general.py                # 로깅/IO 유틸
│   ├── params/
│   │   ├── device_params/            # Dota_B/L_{4,8}bit, Bs_mzi/mrr_bank, default.yaml
│   │   ├── models/                   # deit_{t,s,b}.yaml, bert_{b,l}.yaml
│   ├── scripts/                      # area_power_all, energy_latency_* .sh 5종
│
└── profile/                          # (3) GPU(A100) 베이스라인 프로파일
    ├── vit_infer.py, bert_infer.py   # torch.utils.benchmark 지연 측정
    ├── model.py, customized_layer.py # 단순화 DeiT/BERT + (C)RMSNorm
    ├── power_monitor.sh              # nvidia-smi dmon 전력 모니터
    ├── benckmark_logs/, power_results/  # (제외: 이름만) *.csv 로그
```

- **제외 항목(이름만)**: `.git/`, `software_model/*.csv`(deit_t_sweep_*), `software_model/logs/*.log`, `profile/benckmark_logs/*.csv`, `profile/power_results/*.csv`, 모든 `__pycache__`.
- **자체 소스 규모**: 핵심 Python ~3,000줄. 가장 큰 파일은 `simulator_FFN.py`(1,035줄), `simulator_attn.py`(762줄), `entry_area_power_profile.py`(414줄), `quantize.py`(350줄).

---

## 3. 핵심 모듈 정밀 분석 ★

### 3.1 광학 텐서코어 베이스: `PhotonicCore` (`hardware/photonic_core_base.py`)

모든 광학 코어가 상속하는 추상 베이스. **연산 로직은 없고 device parameter 로더 + 추상 메서드 시그니처만** 정의. **[확실]**

- `__init__` (`:13-17`): `photonic_core_type`, `width`, `height`만 초기화.
- **device param 로더 그룹** (`:20-143`): 각 `_obtain_*_param(config)`은 YAML config 객체에서 광소자 파라미터를 꺼내 인스턴스 속성으로 저장. config=None이면 하드코딩 디폴트. 다루는 소자:
  - `_obtain_laser_param` (`:20-32`): `laser_power`, `laser_area=length×width`, `laser_wall_plug_eff`.
  - `_obtain_micro_comb_param` (`:34-41`): WDM 다파장 광원. `micro_comb_area=length×width` (디폴트 1184×1184 µm²).
  - `_obtain_modulator_param` (`:44-57`): `modulator_energy_per_bit`, `insertion_loss`.
  - `_obtain_y_branch_param`(`:60-68`), `_obtain_photodetector_param`(`:70-80`, `sensitivity=-25dBm`), `_obtain_direction_coupler_param`(`:82-90`), `_obtain_phase_shifter_param`(`:92-106`), `_obtain_mrr_router_param`(`:109-119`).
  - `_obtain_TIA_param`(`:121-128`), `_obtain_ADC_param`(`:130-135`, ADC 객체 생성+`core_ADC_sharing_factor`), `_obtain_DAC_param`(`:138-143`).
- **추상 메서드** (`:147-172`): `cal_insertion_loss`, `cal_TX_energy`, `cal_D2A_energy`, `cal_RX_energy`, `cal_A2D_energy`, `cal_comp_energy`, `cal_laser_energy`, `cal_core_area`, `cal_core_power` — 전부 `raise NotImplementedError`. 서브클래스(crossbar/MZI/MRR)가 구현. **이것이 코어별 area/power 수식의 다형성 분기점.** **[확실]**

> **리뷰 노트**: 라이터 디폴트(`_obtain_modulator_param`에서 `self.modulator_static_power` 오타 — `modulator_power_static`이어야 함, `:55`)가 있으나 config 경로에서는 무해(디폴트 미사용). 일관성 결함.

### 3.2 DOTA 크로스바 코어: `PhotonicCrossbar` (`hardware/photonic_crossbar.py`) — 가장 중요

논문의 핵심 기여인 **DDOT(dynamically-operated dot-product) 크로스바**. `[Nh, Nx] × [Nw, Nx]` 행렬곱을 한 코어에서 수행. docstring(`:16-21`): "laser → micro-comb(파장 D개) → 고속 변조기 → 각 교차점에서 1/N 광 분기 → len-D 벡터곱(PS+coupler) → 차동 PD 검출". 핵심은 **세 차원 병렬**: `core_width(Nw) × core_height(Nh) × num_wavelength(Nx=WDM)`.

핵심 멤버:
- `__init__` (`:22-72`): `core_type=="dota"` 강제(`:43`). `core_width/height/num_wavelength`, 정밀도 `in_bit/w_bit/act_bit`, `work_freq`(GHz). 생성자에서 insertion loss → laser power → modulator → ADC/DAC를 순차 계산(`:65-72`).
- `_initialize_params` (`:74-92`): laser, **mzi_modulator**(입력 변조에 MZI 사용), **mrr_router**(WDM Mux/Demux용 마이크로링), phase_shifter, direction_coupler, photodetector, y_branch, micro_comb, ADC, DAC, TIA 로드.

**핵심 수식 1 — 삽입손실** `cal_insertion_loss` (`:121-140`):
```
insertion_loss_modulation = modulator_IL + mrr_router_IL×2 + y_branch_IL×ceil(log2(max(Nh,Nw)))   # 1:N 스플리터 트리
insertion_loss_computation = y_branch_IL + phase_shifter_IL + direction_coupler_IL                  # 단일 DDOT 노드
insertion_loss = computation + modulation
```
→ 광 경로상 손실을 dB로 합산. 스플리터가 `log2(N)` 단(stage)이라 코어가 커질수록 손실↑. **[확실]**

**핵심 수식 2 — 레이저 파워** `cal_laser_power` (`:142-156`):
```
P_laser_dbm = PD_sensitivity + IL_modulation + IL_computation + 10·log10(Nw·Nh)
laser_power(mW) = 10^(P_laser_dbm/10) / wall_plug_eff × 2^act_bit
```
→ DAC2021 CrossLight 방식. `10·log10(Nw·Nh)`는 빛이 Nw·Nh개 도파관으로 분배되는 손실, `2^act_bit`은 **정밀도가 1비트 오를 때마다 신호대잡음비 요구가 2배 → 레이저 파워 2배**라는 광학 ADC/검출 한계 모델. **광학 가속기의 정밀도-에너지 trade-off의 핵심 수식.** **[확실]**

**핵심 수식 3 — 코어 면적** `cal_core_area` (`:200-228`):
- DDOT 노드 1개 크기: `node_width = y_branch_len + PS_len + DC_len + PD_width + 30(spacing)`, `node_height = y_branch_len + max(PS_w, DC_w, PD_len×2) + 20`.
- `photonic_core_node_area = Nh·Nw·node_h·node_w` (크로스바 전체 노드).
- `photonic_core_adc_area = Nh·Nw / ADC_sharing × ADC_area` (ADC를 Nh·Nw개에 공유인자 적용).
- `photonic_core_dac_area = (Nh+Nw)·Nx·DAC_area` (행/열 변조용 DAC).
- `mzi_modulator_area`, `mrr_area`(×2), `y_branch_area`(스플리터 트리, `log2` 단). 합산 반환. **[확실]**

**핵심 수식 4 — 코어 파워** `cal_core_power` (`:230-258`): 입력부(`laser + DAC×(Nh+Nw)·Nx + 변조 (modulator+mrr_router×2)`) + 연산부(`Nh·Nw·PS_power×2`) + 출력부(`ADC + PD + TIA`). **[확실]**

**에너지 분해 메서드**(per-MAC, 단위 pJ; `:261-288`): `cal_D2A_energy`(DAC), `cal_TX_energy`(변조: modulator+mrr_router×2), `cal_A2D_energy`(ADC), `cal_RX_energy`(PD×2 + TIA, 차동검출), `cal_comp_energy`(PS), `cal_laser_energy`. 전부 `power / work_freq`로 에너지 환산. 이 6개 값이 `simulator_attn/FFN`의 에너지 모델 빌딩블록. **[확실]**

`__main__` 자가테스트(`:291-320`)는 TOPS/W, TOPS/mm², fJ/MAC을 출력 — `2·W·H·λ·freq`가 peak MAC 처리량(2는 MAC=곱+덧).

### 3.3 베이스라인 코어 2종

**`PhotonicMZI` (`hardware/photonic_MZI.py`)** — Clement-style MZI mesh. `[Nw,Nh]×[Nh,1]` MVM(행렬-벡터)만 지원.
- `num_mzis = Nh(Nh-1)/2 + Nw(Nw-1)/2` (`:40`): 유니터리 분해에 필요한 MZI 수 → 면적/파워 지배.
- 삽입손실(`:111-120`): `mzi_IL × (Nh+Nw) + mzi_IL` (mesh를 통과하는 경로 길이에 비례, 크로스바보다 훨씬 큼).
- `mzi_response_time`(`:151`, 2µs)가 **재프로그래밍 지연** → `simulator_FFN.get_latency_mzi`에서 별도 가산. **MZI는 weight를 위상으로 인코딩해 동적 변경이 느림** → readme가 "MZI는 attention을 효율적으로 못함(on-the-fly activation 분해가 매우 비쌈)"이라 명시(`hardware_simulator/readme.md:130`). attention 시뮬에서 mzi는 제외. **[확실]**

**`PhotonicMRRBank` (`hardware/photonic_mrr_bank.py`)** — 마이크로링 공진기 bank. `[Nw,Nh]×[Nh,Nw]` MVM.
- 가중치/입력 모두 ring 변조기(`_obtain_modulator_param`/`_obtain_input_modulator_param`, `:79-114`, `type=='ring'` 강제).
- `insertion_loss_uc`(uncoupled ring loss)를 `(Nw-1)`개 링에 누적(`:147-150`) — bank 길이에 비례한 손실.
- `full_range_support_factor`(`device_params/Bs_mrr_bank_4bit.yaml:39`=2): **add-drop ring은 음수 가중치 불가** → 양수만 표현 → full-range 지원에 추가 패스 필요(시뮬에서 에너지 ×factor). **광학 코어의 부호 표현 한계를 정량화하는 핵심 파라미터.** **[확실]**

### 3.4 디지털 인터페이스: `ADC`/`DAC`/`SRAM`

**`ADC` (`hardware/ADC.py`)**: `ADC_list[1] = {area:2850µm², prec:8, power:14.8mW, sample_rate:10GS/s, type:'sar'}` (`:9-11`). `cal_ADC_param`(`:53-65`): SAR는 `P ∝ N`(`power × freq/sample_rate × prec/8`), flash는 `P ∝ 2^N-1`. 정밀도/주파수 초과 시 ValueError. **[확실]**

**`DAC` (`hardware/DAC.py`)**: `DAC_list[1] = {area:11000µm², prec:8, power:50mW, sample_rate:14GS/s, type:'cap'}` (`:10-12`). `cal_DAC_param`(`:54-65`): `P ∝ 2^N/N × f`. **DAC가 ADC보다 면적·파워 모두 큼** → 입력 변조 공유(input_mod_sharing)가 중요해지는 이유. **[확실]**

**`SRAM` (`hardware/SRAM.py`)**: 최대 2MB(`:11`), DRAM→SRAM 1TB/s, SRAM 대역폭은 CACTI(`0.604347` 계수, `:16`), 500MHz 클럭(`:18`). `preload_DRAM_SRAM`/`load_SRAM_RF`/`load_GB_SRAM`이 byte 수 → cycle 변환. latency 시뮬의 메모리 cycle 산출에 사용. **[확실]**

### 3.5 Attention 하드웨어 시뮬레이터: `attnPrediction` (`simulator_attn.py`) ★

MHA의 두 batched-GEMM (`Q·K^T → [h,N,N]`, `S·V → [h,N,D]`)를 시뮬. `__init__`(`:26-105`)에서 op_info(num_heads/embed_dim/num_tokens)와 config로 코어(dota/mrrbank) 인스턴스화, **arch-level 최적화 플래그** 결정:
- `full_range_support_factor`/`weight_reuse_factor`: dota는 1(완전 full-range, 시간누적), 베이스라인은 config 값(`:47-49`).
- `time_accum_factor`: dota만 활성(`:49`) — **WDM 시간영역 누적**으로 ADC 호출 횟수 감소.
- `input_mod_sharing_flag`/`adc_share_flag`/`disable_crossbar_topology` (`:52-55`): DOTA 아키텍처 최적화 3종.

**Latency 모델 `get_latency_crossbar` (`:107-188`)**:
```
iter_N1=ceil(N1/core_height); iter_N2=ceil(N2/core_width); iter_D=ceil(D/num_wavelength)
cycles_comp = ceil(iter_D·iter_N1·iter_N2·h / (num_tiles·num_pe_per_tile))   # 타일/PE 병렬
latency_comp = cycles_comp / work_freq × 1e-6  (ms)
latency = max(latency_comp, latency_memory)   # comp/mem 중 큰 쪽 (오버랩 가정)
```
메모리 부분(`:140-166`)은 DRAM→SRAM preload를 ViTCoD 방식으로 추정하나, 실제로 `:161`에서 `cycles_preload_data_dram_sram=0`으로 강제 0 처리 — **DOTA는 활성값 fully on-chip 가정이라 메모리가 병목이 아님**을 의도적으로 표현(주석 `:137-138`). **[확실]**

**Energy 모델 `get_energy_crossbar` (`:256-470`)** — 가장 중요. 구성요소별 에너지를 op 횟수×per-op 에너지로 누적:
- **laser**(`:295`): `laser_energy·h·iter_N1·iter_N2·iter_D`.
- **D2A(DAC)+TX(변조)** (`:300-321`): `disable_crossbar_topology` 분기. 크로스바 토폴로지가 켜지면 mat1을 `iter_N2`만큼만 재로드(공유), 꺼지면 전체 `N2`만큼 → **크로스바의 양방향 broadcast 이득을 에너지로 정량화**. `input_mod_sharing_flag`면 mat2를 `/num_tiles`로 분산.
- **comp(PS)**(`:324`): `comp_energy·num_computation`.
- **output(ADC+TIA+adder+detection)** (`:327-354`): `time_accum_factor`로 부분합 수를 줄임(`:327-328`, WDM 시간누적). `adc_share_flag`면 PE 간 ADC 공유로 `ps_size`를 `/num_pe_per_tile` (`:331-339`). → **ADC가 출력 에너지 지배인데 이를 공유/누적으로 줄이는 게 DOTA 핵심 최적화.**
- **datamovement**(`:358-424`): RF/GLB1(LB)/GLB2(GB)/DRAM 4계층. `num_byte=in_bit/16`로 비트폭 스케일. partial-sum NoC 비용(`:382`)까지. DRAM은 0(`:422`, 활성 on-chip 가정).
- 결과를 `energy_dict['comp'|'datamovement']`에 컴포넌트별 [에너지, %]로 저장(`:451-468`) → CSV로 덤프.

**`get_energy_mrrbank` (`:472-648`)**: MRR은 weight-stationary + `full_range_support_factor`(음수 미지원) + `weight_reuse_factor`로 분기. 부호 처리를 위해 입력 datamovement ×2(`:555` 주석). **[확실]**

`run` (`:694-737`): Q·K^T 먼저(`:701-713`), 그 다음 S·V(`:716-737`). S·V에서 `full_range_support_factor //= 2` (`:722`, **softmax 출력 S는 양수라 부호 패스 절반 절감**). mrrbank는 S·V에서 행렬 swap(`:729-730`, mat1을 양수 입력으로). **[확실]**

### 3.6 FFN/Linear 하드웨어 시뮬레이터: `FFNPrediction` (`simulator_FFN.py`) ★

`[N1,D]×[D,N2]` 단일 GEMM(weight×input). 구조는 attn과 평행하나 **head 차원이 없고**, **MZI까지 지원**(`:80-88`). attn과 동일하게 `get_latency_*`/`get_energy_*`를 코어별로 분기.

특기점:
- `get_latency_mzi` (`:192-266`): **MZI 재프로그래밍 지연 가산** (`:213-214`, `latency_comp_program_mzi = ceil(iter_N1·iter_D/(tiles·pe)) × mzi_response_time`). 메모리 preload는 `num_mzis + mzi_sigma_dim` 개의 위상값을 로드(`:228-229`). **MZI의 동적 비효율을 latency로 명시.**
- `get_energy_mzi` (`:580-760`): weight 인코딩 에너지가 `num_mzis × comp_energy_dynamic` + sigma_dim × TX (`:643-646`). weight-stationary라 `weight_reuse_factor=N2`(`:595`)로 재사용.
- `get_energy_crossbar` (`:354-578`): attn과 동일 철학이나 DRAM에서 weight만 로드(`:528`, 활성은 on-chip). **[확실]**

`run` (`:992-1011`): `matrix_dim1=(out_features,in_features)`=weight, `matrix_dim2=(in_features,bs)`=input.

### 3.7 진입점 & 워크로드 매핑

**`entry_area_power_profile.py` — `areaPrediction`**: 단일 코어 area/power를 **전체 가속기(tiles×PE) 규모로 스케일업**.
- `predict_power_crossbar` (`:77-157`): laser/MZM/DAC/ADC/TIA/PD/adder/memory를 `num_tiles×num_pe_per_tile`로 곱해 합산. CACTI 기반 메모리 파워(`GB_power=315.25mW`/4tile, `LB_power`, `buffer_power`, `:126-138`). `input_mod_sharing`/`adc_share` 플래그로 MZM·DAC·ADC·TIA 카운트 조정.
- `predict_area_crossbar` (`:159-237`): 동일하게 면적 스케일. 메모리 면적은 CACTI 상수(GB 14.35mm²/2MB, LB 0.068mm²/4KB, buffer 0.0003mm²/256B, `:204-209`). readme의 예시 표(`hardware_simulator/readme.md:49-60`)와 일치 — **DAC가 26%, mem 24%, photonic_core 19%로 디지털 인터페이스+메모리가 면적 지배.** **[확실]**
- `predict_area_mzi`/`predict_area_mrrbank`(`:239-359`): 베이스라인 면적(파워 리포트는 dota만, `:382-384`).

**`entry_energy_latency_workload.py` — `main`** (`:17-82`): 
1. `modelParams.obtain_ops_list`로 모델→GEMM op-list 생성.
2. 각 op을 `FFNPrediction`(fc) 또는 `attnPrediction`(attn)으로 시뮬, energy/latency 누적.
3. **레이어 수 보정**: head/embed 제외 op은 `×12`(depth) `×factor`(bert-l은 2, `:22-25`,`:48-62`).
4. `obtain_other_costs`로 non-GEMM(softmax/LN/GELU/residual) 비용 가산(`:69`).
5. `total.csv` + 모듈별 breakdown 저장.
- **arch 최적화 플래그 설정**(`:101-120`): `optimize_flag in {broadcast, crossbar, arch_opt}`. `arch_opt`면 `adc_share=1, time_accum=3, input_mod_sharing=1` 모두 켬(`:108-111`) — 이것이 DOTA 풀옵션. `crossbar`는 토폴로지만, `broadcast`는 `disable_crossbar_topology=1`. **ablation을 명령행으로 제어.** **[확실]**

**`utils/model.py` — `modelParams`**: `model_zoo`(`:9-15`)에 DeiT-T/S/B, BERT-B/L의 patch/depth/embed_dim/heads/mlp_ratio. `obtain_ops_list`(`:57-133`)가 embed→qkv→attn→proj→FFN1→FFN2→head를 op dict 리스트로 생성. `obtain_other_costs`(`:23-55`)는 softmax(`51.6/44.8 nJ/byte`), LN(5 FLOPS), GELU(8 FLOPS) 비용을 별도 추정. **[확실]**

**`utils/cal_flops_for_transformer.py` — `TransformerHparams`**: Google ELECTRA flops_computation 포팅. `get_block_flops`(`:76-107`)가 블록당 softmax/residual/layer_norm/activation FLOPs를 누적. `get_infer_ops`(`:178-185`)가 model.py에 non-GEMM op 수를 반환. **[확실]**

### 3.8 소프트웨어 양자화 + 광학 노이즈 모델 ★ (가장 독창적)

**`ops/_quant_base.py`**: LSQ(Learned Step Size Quantization) 베이스. `_LinearQ`/`_Conv2dQ`/`_ActQ`가 학습가능 스케일 `alpha`(layer-wise 또는 kernel-wise)와 `init_state` 버퍼 보유(`:158-207`). `round_pass`(`:37-40`, STE), `grad_scale`(`:25-28`)이 양자화 미분 트릭. **[확실]**

**`ops/quantize.py` — `QuantLinear`** (가장 중요, `:143-272`): LSQ 양자화 위에 **4종 광학 비이상성**을 주입:
1. **input encoding magnitude variation** — `add_input_noise`(`:169-176`): `x += randn·input_noise_std·|x|` (곱셈성 가우시안, w_q에도 적용 `:228-229`).
2. **output computation variation** — `add_output_noise`(`:178-184`): 출력에 `randn·output_noise_std·|out|`.
3. **input encoding phase variation** — `add_phase_noise`(`:186-194`): `x *= cos(randn·noise_std·π/180)` — **위상오차가 cos로 진폭에 들어옴**(광 간섭의 본질). DATE O2NN 참조.
4. **WDM 파장분산** — `kappa_noise_term`(`:155-164`): `simulator.cal_coupler_wdm_error_list`로 파장별 결합오차 벡터 생성, in_features에 타일링.

추론 시 phase noise + WDM이 켜지면(`:231-264`) **특수 forward 경로**: out_features를 k=2 청크로 나눠 phase-noisy 입력과 einsum(`:250-258`), 그 후 WDM 오차항 `noise_x_2 + noise_w_q_2`를 가산(`:239-240`,`:264`). 이는 **광 도트곱의 1차 테일러 근사 오차 모델**. 일반 학습 경로는 `F.linear(x, w_q)`(`:266`). **[확실]**

`QuantAct`(`:274-349`): 활성값 LSQ + signed 자동판정(`:299-301`) + offset(zero-point) + input noise. `QuantConv2d`(`:73-140`): patch-embed conv용, 동일 noise 인터페이스(phase/WDM 제외).

**`ops/simulator.py` — `cal_coupler_wdm_error_list`** (`:10-36`): **광학 WDM 비이상성의 물리 모델.** 방향성 결합기의 결합길이 `L_c(λ)`가 파장 의존(`coupling_length`, 피팅 상수 a=-5.44,b=3.53,c=0.185,d=0.15, `:14-20`). 각 파장에서 결합비 `kappa = sin²(π/4 · L_c(1.55)/L_c(λ))`, 오차항 `2·kappa-1` 반환(`:33-34`). **중심파장(1.55µm)에서 설계된 50:50 결합기가 다른 파장에서 결합비가 틀어져 도트곱 가중치 오차 발생** → WDM 채널이 많을수록 정확도↓. 이 리스트가 quantize.py의 `kappa_noise`. **[확실]**

**`main.py`**: arg parsing(`:200-212`)에 `wbits/abits/headwise/input_noise_std/output_noise_std/phase_noise_std/enable_wdm_noise/enable_linear_noise/num_wavelength/channel_spacing`. **noise std는 2σ값이라 `/2` 후 모델 생성**(`:302`,`:316-318`). `'quant' in model`일 때만 noise 인자 전달(`:303-323`). `from models import quant_vit`(`:33`)로 ViT 정의 import — **단, `models/`·`datasets.py`는 본 패키지에 없음**(4장). **[확실]**

`engine.py`: 표준 DeiT `train_one_epoch`/`evaluate` 루프. **노이즈/양자화 로직은 전혀 없음**(Grep 결과 0 매치) — 모든 비이상성이 QuantLinear 내부에 캡슐화. **[확실]**

### 3.9 GPU 베이스라인: `profile/`

`vit_infer.py`(`:1-58`): `torch.utils.benchmark`로 A100 단일추론 지연 측정(AMP, batch=1, min_run=100). `PreDefinedViT`(profile/model.py)로 DeiT-T/S/B 구성. `customized_layer.py`(`:1-50`): `torch.compile` 데코레이터 붙인 LayerNorm/RMSNorm/CRMSNorm + `LinearZeroMeanOutput` (pre-RMSNorm transformer 참조). power_monitor.sh가 `nvidia-smi dmon`로 전력 트레이스 → idle 빼서 work power 산출(`profile/README.md:67-91`). **에너지=power×latency**. **[확실]**

---

## 4. 데이터플로우 (전체 평가 파이프라인)

```
[software_model] 정확도/강건성 평가
  ImageNet/SST → QuantLinear(LSQ 양자화 + input/output/phase/WDM noise 주입)
    → noise-aware training (scripts/train_quant_transformer_with_noise.sh)
    → noise 스윕 평가 (evaluate_quant_transformer_scan_noise.sh)
    → accuracy vs (input_noise_std / phase_noise_std / num_wavelength) CSV
        ↑ 광학 비이상성이 정확도에 주는 영향 정량화

[hardware_simulator] 면적/전력/에너지/지연 평가
  device_params/*.yaml (광소자 파라미터) + models/*.yaml (워크로드)
    → PhotonicCrossbar/MZI/MRRBank (코어 area/power/per-MAC energy)
    → entry_area_power_profile.py  → area.csv / power.csv (tiles×PE 스케일업)
    → entry_energy_latency_workload.py
         → modelParams.obtain_ops_list → [embed,qkv,attn,proj,FFN1,FFN2,head]
         → FFNPrediction / attnPrediction (op별 energy/latency, ×depth)
         → + obtain_other_costs (softmax/LN/GELU)
         → total.csv + modules/ breakdown

[profile] GPU 베이스라인
  vit_infer/bert_infer (torch.benchmark latency) + nvidia-smi (power)
    → energy = power × latency

[교차] DOTA(시뮬) vs MZI/MRR(시뮬) vs GPU(실측) 비교 → 논문 Fig/Table 재현
```

- **SW↔HW 일관성의 핵심**: `num_wavelength`, `in_bit/w_bit/act_bit`가 양쪽에 공통. SW의 WDM noise는 `num_wavelength`로, HW의 코어 차원은 `core.num_wavelength`로 같은 물리량을 다른 측면(정확도 vs 에너지)에서 평가. **단, 두 패키지는 자동 연결이 아니라 사람이 결과를 합치는 구조.** **[추정]**

---

## 5. HW/SW 매핑

| 물리 개념 | software_model (정확도) | hardware_simulator (비용) | params YAML |
|---|---|---|---|
| WDM 파장 수 | `num_wavelength` → kappa noise (`quantize.py:155`) | `core.num_wavelength`=Nx 병렬도 (`photonic_crossbar.py:48`) | `core.num_wavelength: 12` |
| 비트 정밀도 | `wbits/abits` → LSQ Qn/Qp (`quantize.py:201-202`) | `in/w/act_bit` → DAC/ADC/laser 스케일 (`:151-153`) | `core.precision.*: 4/8` |
| 위상오차 | `phase_noise_std` → cos noise (`:186-194`) | phase_shifter device param (정적) | `device.phase_shifter` |
| 결합기 비이상 | `cal_coupler_wdm_error_list` (`simulator.py`) | `direction_coupler.insertion_loss` | `device.direction_coupler` |
| 입력 변조 | input_noise_std (`:169`) | mzi_modulator energy/IL + input_mod_sharing | `device.mzi_modulator` |
| 코어 크기 | (암묵, in_features 타일링) | `core.width/height` = Nw/Nh | `core.width/height: 12` |
| 아키텍처 옵션 | — | `time_accum/adc_share/input_mod_sharing` | `arch.*` |

- **DOTA-B**: 4 tile × 2 PE, 12×12×12 코어 (`Dota_B_4bit.yaml`). **DOTA-L**: 8 tile (`Dota_L_8bit.yaml:28`). MZI: 2 tile, MRR: 7 tile — **모두 동일 면적 예산으로 공정 비교**(`hardware_simulator/readme.md:19`). **[확실]**
- `default.yaml`이 전 device param의 단일 소스(레이저 23.5mW, MZI 변조 450fJ/bit, MRR 변조 42fJ/bit 등, `:6-61`). 코어별 yaml은 `recursive` 로드로 default를 덮어씀(`config.py:49-62`). **[확실]**

---

## 6. 빌드·실행

**환경**(루트 `readme.md:18-51`): PyTorch + torchvision 0.8.1+ + **timm==0.3.2** + torchpack/einops/gdown. torch 2.0+에서 timm `torch._six` 패치 필요(readme에 코드 제공). hardware_simulator는 추가로 `multimethod`, `pyyaml`만 필요(경량). **[확실]**

**software_model**:
```bash
# noise-aware 학습 (4-bit DeiT-T)
torchrun --nproc_per_node=4 main.py --model deit_tiny_patch16_224_quant \
  --wbits 4 --abits 4 --headwise --input_noise_std 0.03 --output_noise_std 0.05 \
  --finetune pretrained/deit_tiny... --data-path /imagenet --enable_linear_noise
# noise 스윕 평가
./scripts/evaluate_quant_transformer_scan_noise.sh &> results.log
./scripts/process_output_logs.sh   # → mean/std CSV
```
(`software_model/readme.md:79-275`)

**hardware_simulator** (cwd = hardware_simulator/):
```bash
# area/power
python entry_area_power_profile.py -e area_power_single --config ./params/device_params/Dota_B_4bit.yaml
# energy/latency (워크로드)
python entry_energy_latency_workload.py -e elat -m deit-t -t 197 \
  --config ./params/device_params/Dota_B_4bit.yaml --optimize_flag arch_opt
# 배치: ./scripts/area_power_all.sh, ./scripts/energy_latency_all.sh
```
(`hardware_simulator/readme.md:26-128`)

**profile**: 터미널 2개 — `./power_monitor.sh > power_results/power_usage.csv` + `python vit_infer.py -m deit-b` (`profile/README.md:80-91`).

---

## 7. 의존성

- **software_model**: torch, torchvision, **timm 0.3.2**(create_model/Mixup/scheduler/optimizer), einops, numpy. 내부: `ops.quantize`, `ops.simulator`. **외부(이름만)**: `models`/`models_v2`/`datasets`/`quant_vit`은 본 패키지 미포함 — 원본 facebookresearch/deit + 사용자가 `models/quant_vit.py`를 별도 배치해야 함(`software_model/readme.md:15`이 존재를 전제). → **재현하려면 DeiT repo 클론 필수.** **[확실, 미포함]**
- **hardware_simulator**: pyyaml, multimethod, numpy/torch(general.py 유틸만). **torch는 시뮬 본체에 불필요**(behavior-level, 순수 산술). 경량·독립 실행 가능. **[확실]**
- **profile**: torch(+CUDA/AMP), `torch.utils.benchmark`, nvidia-smi(시스템). A100 가정.
- **광학 시뮬 가정**: 모든 device param(레이저 wall-plug 0.2, PD sensitivity -25dBm, DAC 50mW@14GS/s 등)은 문헌 인용 상수. **실측 아님.** ADC/DAC choice는 1종만(`ADC.py:9`, `DAC.py:10`).

---

## 8. 강점 · 한계

**강점**:
1. **SW(정확도)/HW(비용) 이중 평가의 일관성** — `num_wavelength`·비트폭 같은 동일 물리량으로 정확도와 에너지를 함께 본다. 광학 가속기 평가의 모범.
2. **광학 비이상성의 물리 기반 모델** — WDM 결합기 분산(`simulator.py`)·위상 cos 오차·곱셈성 노이즈를 분리 모델링. 단순 가우시안이 아니라 소자 물리에서 유도.
3. **공정한 베이스라인** — MZI/MRR을 동일 면적 예산으로 비교(`readme.md:19`), 각 코어의 부호 표현 한계(`full_range_support_factor`)·재프로그래밍 지연(`mzi_response_time`)까지 정량화.
4. **아키텍처 ablation의 명령행 제어** — `optimize_flag`로 broadcast/crossbar/arch_opt를 토글, time_accum·adc_share·input_mod_sharing의 개별 기여 분리.
5. **컴포넌트 단위 에너지 breakdown** — laser/DAC/ADC/TIA/TX/comp/datamovement(RF/LB/GB/DRAM)를 % 단위로 CSV 덤프 → 병목(ADC·DAC·메모리) 가시화.

**한계**:
1. **시뮬레이션 기반, 실측 부재** — 칩 테이프아웃·실리콘 검증 없음. behavior-level(`readme.md:4`). 모든 device param은 문헌 상수.
2. **메모리/지연 모델의 단순화** — latency 시뮬에서 `cycles_preload_data_dram_sram=0` 강제(`simulator_attn.py:161`), 활성 fully on-chip 가정 → 실제 대용량 시퀀스에서 메모리 병목 과소평가 가능. **[추정]**
3. **패키지 불완전성** — `software_model/models/`·`datasets.py` 미포함 → SW 정확도 재현은 외부 DeiT 의존. ADC/DAC 라이브러리 1종만.
4. **코드 품질 결함** — 베이스 디폴트 오타(`modulator_static_power`), `cal_comp_energy`의 `comp_energy_dynamic + comp_energy_dynamic`(MZI, `photonic_MZI.py:285`, dynamic 중복·static 누락 의심), `clamp`에 디버그 print 잔존(`quantize.py:67`). 기능엔 무해하나 정밀도 우려.
5. **하드코딩된 상수 산재** — 메모리 파워/면적(CACTI), adder 스케일(`0.2/4.39`), 노이즈 절반 처리(`/2`) 등이 코드 곳곳에 매직넘버.

---

## 9. 우리 프로젝트(HG-PIPE 계열 FPGA ViT 가속기 + XR 시선추적)에의 시사점

> **대전제 — 패러다임이 다름 [확실]**: 본 repo는 **광학 ASIC(photonic ONN)** 시뮬레이터다. 우리 타깃인 **FPGA(HG-PIPE)** 와는 (a) 연산 매체(빛 간섭 vs LUT/DSP), (b) 비이상성(파장분산·위상오차 vs 양자화·오버플로), (c) 비용 모델(레이저·DAC/ADC·삽입손실 vs LUT/FF/BRAM/DSP) 자체가 다르다. **area/power 수식·photonic core 모델은 직접 재사용 불가.** 그러나 *방법론* 일부는 이식 가능하다.

**재사용 가능(높은 가치)**:
1. **LSQ 양자화 + noise-aware training 프레임워크** (`ops/_quant_base.py`, `quantize.py`의 QuantLinear/QuantAct) — 광학 noise 부분만 제거하면 **순수 W4A4/W8A8 LSQ 양자화기**로 FPGA ViT에 그대로 쓸 수 있다. STE(`round_pass`), 학습가능 스케일(`alpha`), kernel/layer-wise 모드, signed 자동판정(`QuantAct:299`)이 검증된 구현. **우리의 FPGA 양자화 정확도 baseline에 즉시 이식 가능.**
2. **하드웨어-비이상성을 학습에 주입하는 패턴** — 광학은 phase/WDM이지만, **FPGA에서는 "고정소수점 오버플로/포화, DSP 누산기 truncation, BRAM 양자화 오차"** 가 대응 비이상성이다. `add_*_noise` 메서드처럼 **forward에 FPGA 수치 효과를 주입해 noise-aware fine-tuning** 하는 구조를 차용하면, post-training accuracy drop을 학습 단계에서 미리 보상할 수 있다.
3. **워크로드→op-list→레이어별 비용 누적 + breakdown CSV** 방법론 (`utils/model.py`, `entry_energy_latency_workload.py`) — 가속기 종류와 무관한 **공통 평가 골격**. 우리 FPGA 시뮬레이터도 `[embed,qkv,attn,proj,FFN1,FFN2,head]` op-list × depth로 latency/resource를 누적하고 컴포넌트별 % breakdown을 내는 구조를 그대로 채택할 만하다.
4. **Attention 타일링 모델** (`simulator_attn.py`의 `iter_N1/iter_N2/iter_D` + `time_accum_factor`) — 광학의 WDM 시간누적은 FPGA에서 **systolic 배열의 K-차원 누산 타일링**에 대응. QK^T와 SV를 별도 GEMM으로 다루고 S가 양수임을 활용(`run:722`)하는 점은 FPGA에서도 softmax 후 unsigned 처리로 비트폭/리소스 절감에 응용 가능.
5. **arch ablation 토글화** (`optimize_flag`) — 우리 DSE(설계공간탐색)에서 "타일 수/PE 수/버퍼 공유" 같은 옵션을 명령행 플래그로 켜고 끄며 Pareto를 뽑는 패턴의 좋은 선례.

**참고만(직접 이식 부적합)**:
- `PhotonicCrossbar`/`PhotonicMZI`의 area/power 수식, laser/insertion-loss/DAC-ADC 에너지 모델 — 광학 전용. FPGA는 LUT/DSP/BRAM 카운트 기반이라 무관.
- `cal_coupler_wdm_error_list`(WDM 분산) — 광학 물리 모델, FPGA 무관.
- GPU profile은 베이스라인 측정 방법(power=측정-idle, energy=power×latency)만 참고. XR 시선추적 워크로드(저지연·소형 입력)에는 batch=1·min_run 설정 철학이 유용.

**XR 시선추적 맥락 [추정]**: 시선추적은 초저지연·소형 ViT가 핵심인데, 본 repo의 **토큰 수 파라미터화**(`tokens` 인자, BERT 128~384 가변)와 **레이어별 latency breakdown**은 "토큰 수 줄이기 vs 정확도" trade-off를 우리 시선추적 ViT에서 분석할 때 같은 골격으로 재활용 가능하다. 다만 가속기 비용은 FPGA 자체 모델로 교체해야 한다.

---

## 10. 근거 / 한계 표기

- **[확실]** 본문 모든 파일:라인 인용 — 해당 소스를 직접 Read로 확인. (photonic_crossbar/MZI/mrr_bank, ADC/DAC/SRAM, simulator_attn/FFN, entry_*, utils/*, ops/_quant_base/quantize/simulator, main.py, profile/*, 전 YAML, 4개 readme)
- **[추정]** 표기 항목: (a) SW↔HW가 자동 파이프라인이 아니라 수동 결합이라는 점(5장), (b) latency 메모리 모델 단순화로 인한 병목 과소평가(8장), (c) XR 시선추적 적용 시사점(9장) — 코드 정황 기반 추론.
- **[확인 불가]**:
  - `HPCA24_LT_poster_v1_02.pdf` — 작업환경에 `pdftoppm` 부재로 페이지 렌더 실패. **포스터 내용 미반영.** 논문 본문은 별도 `PAPER_PRJXR`/`REF` 경로에 있을 수 있으나 본 분석 범위 밖.
  - `software_model/models/quant_vit.py`, `models_v2.py`, `datasets.py` — 본 AE 패키지에 **미존재**(Glob 0건). 외부 DeiT(facebookresearch/deit) 의존으로, ViT 모델에 QuantLinear가 실제로 어떻게 배선되는지(어느 레이어에 phase/WDM noise가 들어가는지)는 코드로 확인 불가. readme(`software_model/readme.md:15`)가 존재를 전제할 뿐.
- **패러다임 경고(재확인)**: 본 시스템은 **광학 ASIC**이다. 우리의 **FPGA(HG-PIPE)** 타깃과는 비용 모델·비이상성·연산 매체가 근본적으로 다르므로, **하드웨어 area/power 수식은 차용 금지**, 양자화/noise-aware 학습·op-list 평가 골격·타일링 사고방식만 선별 이식할 것.
