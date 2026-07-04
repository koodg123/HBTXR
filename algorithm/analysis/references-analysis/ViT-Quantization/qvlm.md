# Q-VLM (qvlm) 코드베이스 정밀 분석

> 대상 경로: `REF/ViT-Quantization/qvlm`
> 분석 도구: Glob/Grep/Read (bash 미사용)
> 근거 표기: `[코드]` = 실제 코드 라인으로 확인 / `[추정]` = 코드 정황상 추정 / `[확인불가]` = 저장소 내 근거 없음

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **목적**: 대형 Vision-Language Model(LLaVA 계열)을 **W4A4**(가중치 4bit + 활성 4bit)로 PTQ하면서 메모리를 절감하고 정확도를 유지. `[코드]` README.md:1-7
- **원논문**: *Q-VLM: Post-training Quantization for Large Vision-Language Models* (NeurIPS'24), arXiv:2410.08119 (Changyuan Wang 외, Tsinghua). `[코드]` README.md:1-6
  - 저장소 내 논문 PDF 없음(qvlm 하위 PDF는 bitsandbytes 벤치마크 plot 1개뿐). `[확인불가]` (Glob `**/*.pdf` 결과)
- **핵심 아이디어** (코드로 확인되는 범위):
  1. 가중치는 bitsandbytes의 **4bit(FP4/NF4)** Params4bit로 처리(외부 라이브러리). 활성은 자체 `QuantAct`로 **4bit fake-quant**. `[코드]` custom_bitsandbytes/.../nn/modules.py:206-311
  2. 활성 양자화 범위(min/max)를 **calibration + 탐색(search)**으로 결정하되, **레이어 간 통계적 의존성(entropy / cross-layer dependency)**을 이용해 어느 레이어에서 재탐색할지 동적으로 결정(블록 단위 효율 탐색). `[코드]` quantization_utils/quant_modules.py:113-167, 208-241
  3. **LLM 레이어(llama_layer)와 CLIP/projector 레이어를 구분**해 서로 다른 양자화 통계(채널-wise vs row-wise, 다른 탐색 step)를 적용. `[코드]` quant_modules.py:69-79, 146-179; modules.py:217-233
- **방식 요약**: PTQ, weight-activation(W4A4), 활성은 비대칭(asymmetric) per-channel/per-row, calibration 기반 동적 범위 + entropy 기반 탐색. `[코드]` quant_utils.py:108-136; quant_modules.py:146-179

---

## 2. 디렉토리 구조 (자체 소스 + 제외 표기)

```
qvlm/
├── README.md                       # Q-VLM 소개/설치/SQA 실행                 [정밀분석]
├── scripts/
│   ├── generate_sqa_response.sh    # W4A4 추론(=calibrate+생성) 진입          [정밀분석]
│   ├── generate_sqa_response_multi.sh / evaluate_*.sh / sqa_eval_*.sh         [요약]
│   └── (finetune/pretrain 등)       # LLaVA 학습 스크립트                      [제외: 이름만]
├── llava/                          # LLaVA 본체(외부 이식)
│   ├── eval/model_vqa_science.py   # ★ Q-VLM calibrate+search 구동 핵심        [정밀분석]
│   ├── eval/model_vqa_loader.py    # QuantAct 사용 평가 로더                   [요약]
│   ├── model/multimodal_encoder/clip_encoder.py  # 비전 인코더 로드(bnb 4bit)  [정밀분석]
│   ├── model/language_model/llava_llama.py, llava_mpt.py  # 모델 정의          [제외: 외부]
│   └── train/*, eval/* (기타)       # 일반 학습/평가                           [제외: 이름만]
└── custom_bitsandbytes/            # 수정된 bitsandbytes 포크
    ├── bitsandbytes/
    │   ├── quantization_utils/
    │   │   ├── quant_utils.py       # ★ 선형 양자화 기본(ZeroQ 유래)           [정밀분석]
    │   │   ├── quant_modules.py     # ★ QuantAct (활성 양자화/탐색/entropy)    [정밀분석]
    │   │   └── __init__.py                                                     [요약]
    │   ├── nn/modules.py            # ★ Linear4bit (W4 + QuantAct 삽입)        [정밀분석]
    │   ├── nn/modules_same.py       # Linear4bit 단순 변형(블록구분 없음)      [부분]
    │   ├── functional.py / autograd / triton / optim / cuda_setup  # 커널 등   [제외]
    │   └── nn/triton_based_modules.py                                          [제외]
    └── tests/, benchmarking/, *.md  # 테스트/문서                              [제외]
```

- **제외**: `custom_bitsandbytes` 내부 C/CUDA/triton 커널(`functional.py`, `triton/`, `autograd/`, `cuda_setup/`), optim, tests, 벤치마크. `llava`의 모델 정의/일반 eval/train 스크립트는 외부 이식이므로 이름만 언급.
- **자체 양자화 핵심 판단 결과**: 과제에서 지목한 `quantization_utils/{quant_utils,quant_modules}.py`는 **이 repo가 추가/수정한 Q-VLM 핵심**임을 코드로 확인(`QuantAct`의 entropy/DED/search 로직 + `nn/modules.py`의 layer-aware 삽입). `[코드]` quant_modules.py:113-167; modules.py:216-233

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 선형 양자화 기본 — `quantization_utils/quant_utils.py` (ZeroQ 유래)
- 파일 헤더에 ZeroQ(Cai/Yao/Dong/Gholami) 라이선스 명시 → **ZeroQ에서 가져온 비대칭 정수 양자화 유틸**. `[코드]` quant_utils.py:1-19
- `linear_quantize` = `round(scale·x − zero_point)`, `linear_dequantize` = `(x + zero_point)/scale`. `[코드]` 62-105
- `asymmetric_linear_quantization_params(num_bits, sat_min, sat_max)`: `n=2^bits−1`, `scale=n/(max−min)`, `zero_point=scale·min`(정수 반올림, signed면 `+2^(bits−1)`). → **비대칭(asymmetric) 양자화 파라미터**. `[코드]` 108-136
- `AsymmetricQuantFunction`: forward에서 양자화→clamp(절댓값 트릭 `0.5·(|−q−n|−|q−(n−1)|−1)`)→dequant, backward는 STE(grad 그대로 통과). `[코드]` 139-171
- `lp_loss`(L_p 노름 손실, p 가변)는 활성 범위 탐색의 score 함수로 사용됨. `[코드]` 26-33 (→ quant_modules.py:216,234)

### 3.2 활성 양자화 + 탐색 — `quantization_utils/quant_modules.py` (Q-VLM 핵심)
`class QuantAct(Module)` — Linear4bit 내부에 삽입되는 활성 양자화 모듈. `[코드]` quant_modules.py:36-276

- **상태/구분**:
  - `llama_layer=True`면 LLM 경로(범위 dim=input_dim, 기본 4096), False면 CLIP 경로(row dim=257 for v1.3). `[코드]` 69-78 (CLIP_row_dim=257)
  - `count_block`/`count_layer`로 현재 블록·서브레이어 위치를 추적(삽입 시점에 외부에서 주입). `[코드]` 62-64
- **양자화 연산** `quantization(inputs, q_min, q_max)`: `asymmetric_linear_quantization_params`로 scale/zero 계산 후 round→clamp(절댓값 트릭)→dequant. 입력 마지막 차원이 scale 크기와 다르면 transpose해 적용(채널/행 정렬). `[코드]` 93-110
- **entropy / cross-layer dependency (Q-VLM 차별점)**:
  - `cal_entropy(attn)`: `−Σ p·log p` (정규화 후) — 레이어 출력 분포 엔트로피. `[코드]` 127-130
  - `compute_DED(p_k, p_{k+1})`: 인접 두 레이어 양자화 분포의 **조건부 엔트로피 D(k,k+1)** = `−Σ joint·log(conditional)` — 레이어 간 의존성 측정. `[코드]` 113-125
  - `search_strategy_judge()`: 직전 레이어 엔트로피가 평균 이상이거나 `count_block%3==1`이면 재탐색(True), 아니면 건너뜀 → **민감한 레이어만 선택적으로 범위 재탐색**(효율적 블록 양자화). `[코드]` 132-144
- **calibration 범위 갱신** `calibrate_quantization`:
  - llama 경로: `search_flag`이면 토큰축 min/max로 `llama_range_min/max`를 in-place 누적 갱신(running) → 양자화 → 레이어1/7은 엔트로피, 그 외는 DED를 누적. `[코드]` 146-169
  - CLIP 경로: **row-wise**(마지막 차원) min/max로 `CLIP_range_min/max` 갱신. `[코드]` 170-179
- **forward 단계 탐색**(`self.search` & `_calibrate`):
  - 첫 탐색: 분포/엔트로피 초기화. 이후 llama 경로는 범위를 `×(1−0.1·a), a∈0..6`로 줄이며 `lp_loss(p=0.5)` 최소 범위 선택; CLIP 경로는 `×(1−0.001·a)` + **엔트로피 정규화 항(weight 0.01)** 추가한 score로 선택 → 비전 경로 탐색에 LLM 엔트로피를 결합. `[코드]` 208-241
  - 추론 단계: dim≠4096 또는 layer==4면 토큰축 채널-wise 동적 범위, 그 외 calibration 고정 범위 사용. `[코드]` 256-276
- 모듈 단위 `calibrate(model, loader, device)`: 모든 QuantAct를 calibrate 모드로 두고 4×8−1 배치를 모아 한 번 forward. `[코드]` 279-298
- 전역 상태(`last_layer_entropy`, `llama_entropy`, `llama_distribution`)로 레이어 간 정보를 공유 — **레이어 간 의존성 전파를 전역 변수로 구현**. `[코드]` 31-34, 134-135, 159-167 (`[추정]`: 전역 사용은 멀티스레드/배치 병렬에 취약)

### 3.3 4bit Linear에 활성 양자화 삽입 — `nn/modules.py` (Q-VLM 수정 핵심)
- `class Linear4bit(nn.Linear)`: 가중치를 `Params4bit`(FP4 기본, NF4 파생)로 4bit 양자화(bitsandbytes 원본 기능). `[코드]` modules.py:206-211, 317-323
- **활성 4bit 삽입**: `self.activation_bit=4`, `self.quant_activation = QuantAct(...)`를 레이어 위치(count_block/layer, llama_layer)와 함께 생성. `[코드]` 214-237
  - `input_features==1024` 또는 `(4096,1024)`면 **CLIP+mm_projector → llama_layer=False**(레이어 주기 7), 그 외는 **llama_layer=True**(주기 8). `[코드]` 217-229
  - 즉 **모델 그래프 구성 시 레이어 인덱스를 전역 카운터로 자동 부여** → QuantAct가 자기 위치를 인지. `[추정]`: 레이어 수/순서에 강하게 의존(LLaVA-v1.3 7B 가정, CLIP 257 토큰).
- forward: `quant_x = self.quant_activation(x)`(활성 fake-quant) → `bnb.matmul_4bit(quant_x, weight.t(), quant_state)`로 **W4×A4 행렬곱**(가중치는 4bit dequant은 bnb 커널 내부). `[코드]` 257-315 (299, 311)
- `modules_same.py`: 동일 클래스의 단순 버전(블록/레이어 구분·llama_layer 없이 `QuantAct(activation_bit, dim)`만). 초기/대조 구현으로 보임. `[코드]` modules_same.py:206-218 `[추정]`

### 3.4 비전 인코더 양자화 연동 — `llava/model/multimodal_encoder/clip_encoder.py`
- `CLIPVisionTower.load_model`: bitsandbytes `BitsAndBytesConfig(load_in_4bit, nf4, double_quant)`를 준비하나, **현재 활성 코드는 16bit로 로드**(4bit/8bit 로드 라인은 주석 처리). `[코드]` clip_encoder.py:22-37
- 즉 비전 인코더는 기본적으로 FP16. **W4A4 양자화는 LLM(Linear4bit) 경로 중심**이며, CLIP 경로는 `QuantAct(llama_layer=False)`가 적용될 때 row-wise 활성 양자화로 다뤄짐(load_pretrained_model이 4bit 로드를 켤 때). `[코드]` clip_encoder.py:33-37 + quant_modules.py:170-179 `[추정]`: 4bit 활성화 여부는 `load_pretrained_model(... load_4bit)` 경로에 의존.

### 3.5 calibration/search 구동 — `llava/eval/model_vqa_science.py`
- `run_calibrate`: 모든 QuantAct를 `set_calibrate(True)` → ScienceQA train에서 16×num_chunks 질문 샘플, 8장으로 calibrate 후 `set_search(True)`로 탐색 단계 진입, 2회 탐색 후 종료, 마지막에 `set_calibrate(False)`. `[코드]` model_vqa_science.py:33-113
  - calibrate는 실제 `model.generate`로 forward를 흘려 통계 누적. `[코드]` 86-95
- `eval_model`: `load_pretrained_model(..., load_4bit)`로 모델 로드 → `run_calibrate` → ScienceQA test 생성→답안 jsonl 기록. `[코드]` 116-193
- `model_vqa_loader.py`도 동일하게 QuantAct/calibrate 사용(일반 VQA 평가용). `[요약]` (Grep로 QuantAct 사용 확인)

---

## 4. 알고리즘 / 수식

### 4.1 비대칭 정수 양자화(활성/가중치 공통 정의)
```
scale = (2^b − 1) / (x_max − x_min)
zp    = round(scale · x_min)  (+2^(b−1) if signed)
x_q   = clamp(round(scale·x − zp), −2^(b−1), 2^(b−1)−1)
x_hat = (x_q + zp) / scale
```
`[코드]` quant_utils.py:108-136, 62-105; quant_modules.py:93-110

### 4.2 레이어 엔트로피 & 레이어 간 의존성(DED)
```
H(layer)   = −Σ_i p_i log p_i             (정규화된 활성 분포)
D(k,k+1)   = −Σ_ij joint(x^k, x^{k+1}) · log p(x^{k+1} | x^k)
```
- 재탐색 판단: `H(직전레이어) ≥ mean(H) 또는 block%3==1 ⇒ 재탐색`. `[코드]` quant_modules.py:113-144

### 4.3 범위 탐색(LLM vs CLIP)
```
LLM:  range *= (1 − 0.1·a), a∈{0..6};  score = L_{0.5}(quant, x);  argmin
CLIP: range *= (1 − 0.001·a), a∈{0..2}; score = mean|Δ|^0.5 + 0.01·mean(H_llm); argmin
```
`[코드]` quant_modules.py:208-241

> 비고: 코드에서 직접 유도. 논문 원식(D(k,k+1) 정의의 정확한 정규화/계수)과의 1:1 대응은 PDF 부재로 일부 `[추정]`.

---

## 5. 학습 / 평가(캘리브레이션) 파이프라인

- **순수 PTQ** — 가중치 학습 없음. 파이프라인: `load_pretrained_model(load_4bit)` → `run_calibrate`(calibrate→search→고정) → `model.generate`로 ScienceQA 응답 생성 → 별도 스크립트로 정답 평가. `[코드]` model_vqa_science.py:116-193; README.md:32-42
- **calibrate 데이터**: ScienceQA train(JSON)에서 셔플 후 16×num_chunks 샘플, 8장으로 범위 누적, 이후 탐색 2회. `[코드]` model_vqa_science.py:39-108
- **실행 스크립트**: `generate_sqa_response.sh`가 `--load-4bit`로 W4A4 추론(question-file-calibrate/image-folder-calibrate가 calibration 입력). `[코드]` generate_sqa_response.sh:1-8
- 평가 하드웨어: RTX 3090 24GB(저메모리 타깃). `[코드]` README.md:33

---

## 6. 의존성

- **수정된 bitsandbytes 포크**(필수: `pip uninstall bitsandbytes` 후 `custom_bitsandbytes` 설치). 4bit 행렬곱(`bnb.matmul_4bit`, `Params4bit`, FP4/NF4)이 가중치 양자화 백엔드. `[코드]` README.md:25-30; modules.py:210-211, 311
- LLaVA(haotian-liu), CLIP(transformers `CLIPVisionModel`/`BitsAndBytesConfig`), PyTorch, numpy, tqdm, shortuuid, PIL. `[코드]` clip_encoder.py:4; model_vqa_science.py:1-20; README.md:54-57
- ZeroQ 유래 양자화 유틸(quant_utils.py 헤더). `[코드]` quant_utils.py:1-19

---

## 7. 강점 / 한계 / 리스크

**강점**
- **W4A4** 실현: 가중치 4bit(bnb 커널) + 활성 4bit(QuantAct) 조합으로 MBQ류(주로 W-only/W4A8 fake-quant)보다 공격적. `[코드]` modules.py:214, 299, 311
- **레이어 간 의존성(entropy/DED) 기반 선택적 탐색**으로 calibration 비용을 줄임(모든 레이어 재탐색 회피). `[코드]` quant_modules.py:132-144
- LLM/CLIP 경로를 분리해 서로 다른 통계(채널-wise vs row-wise) 적용 — 모달리티별 분포 차이 반영. `[코드]` quant_modules.py:146-179

**한계 / 리스크**
- **모델 구조 하드코딩 의존**: `input_features∈{1024,4096}`, CLIP 257 토큰, 레이어 주기 7/8, layer==4/7 특수 분기 등 — **LLaVA-v1.3 7B 가정**에 강하게 결합. 다른 백본/해상도에서 깨지기 쉬움. `[코드]` modules.py:217-229; quant_modules.py:75-78, 160, 260
- **전역 변수 상태 공유**(`last_layer_entropy`, `llama_entropy`, `count_block/layer`) — 배치/멀티스레드/모델 2개 동시 로드 시 상태 오염 위험. `[코드]` quant_modules.py:31-34; modules.py:205, 216
- 활성 양자화는 **fake-quant**(dequant float로 복귀 후 bnb matmul) — 실제 A4 정수 연산 가속은 아님. `[코드]` modules.py:299-311
- 코드에 디버그용 주석/죽은 코드가 다수(주석 처리된 블록 카운터, `return 0` 등). 유지보수성 낮음. `[코드]` modules.py:273-298
- 비전 인코더는 기본 FP16 로드(주석상 4bit 가능하나 비활성). 실측 W4A4 적용 범위는 LLM 중심. `[코드]` clip_encoder.py:33-37

---

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

- **저비트(A4) 정수 양자화 정의가 명확** → FPGA 친화: `asymmetric_linear_quantization_params`의 `scale=n/(max−min)`, `zp=round(scale·min)`은 정수 scale+offset 회로로 직접 매핑 가능. clamp의 절댓값 트릭(`0.5(|−q−n|−|q−(n−1)|−1)`)은 분기 없는 포화 연산으로 RTL/HLS에 유리. `[코드]` quant_utils.py:108-171; quant_modules.py:100-110
- **per-channel(LLM) vs row-wise(CLIP) 범위**: ViT 가속기에선 row-wise(토큰별) 범위는 토큰마다 scale 갱신 회로가 필요 → FPGA에선 비용↑. 우리는 **calibration 고정 per-channel scale**(LLM 경로의 `llama_range_min/max` 고정값)을 채택하면 가중치/활성 scale을 사전 상수로 굳혀 정수 MAC + 상수 시프트로 단순화 가능. `[코드]` quant_modules.py:146-179, 256-276
- **outlier/범위 축소 탐색**: range를 `(1−step)`로 줄여가며 lp_loss 최소화하는 방식은 FPGA에서 **clipping threshold를 오프라인 캘리브레이션으로 고정**하는 전략과 동일 → 좁은 INT4 폭에서 outlier로 인한 포화 손실을 줄이는 설계 근거. `[코드]` quant_modules.py:208-241
- **entropy/DED 기반 레이어 민감도**: XR 시선추적 ViT에서 "어느 레이어/모달리티에 더 많은 비트를 줄지"를 결정하는 **혼합정밀도(mixed-precision) 비트 배분** 도구로 차용 가치. 단, 전역 상태 의존 구현은 그대로 쓰기보다 개념만 차용 권장. `[코드]` quant_modules.py:113-144
- **W4A4 타깃의 의의**: HG-PIPE 계열 ViT 가속기에서 활성까지 4bit로 낮추면 BRAM/대역폭 절감 큼. Q-VLM은 그 정확도 유지법(레이어 의존성 인지 범위 탐색)을 보여주는 참고 사례. 다만 본 repo는 LLaVA LLM 중심이라 ViT 인코더 가속 코드 직접 재사용은 제한적. `[코드]` modules.py:206-315; clip_encoder.py:33-37 `[추정]`

---

## 9. 근거 표기 규칙 요약
- `[코드]`: 해당 파일·라인을 Read/Grep로 직접 확인.
- `[추정]`: 코드 구조·주석·하드코딩 값으로부터의 합리적 추론(논문 PDF 부재, 외부 로더 경로 의존 등).
- `[확인불가]`: 저장소 내 근거 없음(예: 동봉 논문 PDF, 비전 4bit 로드의 실제 활성 여부 전체 경로).
