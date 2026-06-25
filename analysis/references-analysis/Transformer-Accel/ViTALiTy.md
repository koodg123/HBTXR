# ViTALiTy 정밀 분석 (Transformer-Accel / 최상위 ViTALiTy)

> 분석 대상 repo: `REF\Transformer-Accel\ViTALiTy`
> (주의: `efficient-transformer-accelerator\ViTALiTy` 는 별도 repo로 본 분석 범위 아님. 또한 `REF\Analysis\ViT-Quantization\ViTALiTy.md` 는 다른 위치의 분석이며 본 문서와 별개.)
> 논문: ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention, HPCA 2023 (arXiv:2211.05109, GATECH-EIC).

---

## 1. 개요

ViTALiTy는 **알고리즘–가속기 공동 설계(co-design)** 프레임워크다. 핵심 아이디어는 vanilla softmax attention을 **Taylor 1차 근사 기반 linear attention**으로 대체하되, 정확도 손실을 sparse attention 항으로 보강하는 것이다 (`README.md:17-37`).

- **알고리즘**: softmax attention을 "weak"(low-rank linear Taylor) + "strong"(sparse) 두 맵으로 분해. low-rank 항은 `K^T·V`로 만든 전역 컨텍스트 행렬 `G`를 사용해 시퀀스 길이 N에 대해 **선형 복잡도**를 달성하고, sparse 항이 국소 특징(정확도)을 보강한다 (`README.md:19`, `README.md:34`).
- **학습/추론 분리**: 학습 단계는 low-rank + sparse를 함께 사용(고차 Taylor 항을 SANGER 류 sparse로 근사), **추론 단계는 low-rank linear Taylor attention만** 사용해 HW 효율을 노린다 (`README.md:25-26`).
- **하드웨어**: chunk 기반 설계, systolic array(SA-General + SA-Diag) + pre/post-processor, DRAM/SRAM/NoC/Regs 4계층 메모리, intra-layer 파이프라인, down-forward accumulation dataflow (`README.md:37`, `README.md:43`).

**중요한 사실 (확인=라인 기반):** 본 repo에 **실제로 포함된 것은 알고리즘(소프트웨어, PyTorch) 코드뿐**이다. RTL(.v/.sv), HLS(.cpp/.h/.cu), 하드웨어 시뮬레이터 소스는 **존재하지 않는다** (Glob `**/*.{v,sv,cpp,cc,h,hpp,cu,tcl,c}` → "No files found"). 하드웨어 가속기는 README와 `figures/`의 PNG로만 기술된다. 따라서 본 문서의 HW 분석은 README 텍스트에 근거하며, 코드 근거가 없는 부분은 "확인 불가"로 표기한다.

이 codebase는 facebookresearch/deit 를 포크해 만든 것이다 (`README.md:64`, `main.py:1-2` Facebook 저작권 헤더).

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (실제 Read하여 분석)

```
ViTALiTy/
├── README.md                # 프레임워크 설명, 실행법 (확인)
├── requirement.txt          # 의존성 (확인)
├── LICENSE                  # Apache 2.0
└── src/
    ├── vision_transformer.py  # ★핵심: Attention(Taylor linear attn), Block, VisionTransformer
    ├── quantize.py            # QuantMeasure, UniformQuantize, QConv2d/QLinear, RangeBN
    ├── quant_utils.py         # QuantizedMatMul, QuantizedLinear, STE fake-quant, build_quant_matmul
    ├── mlp.py                 # Mlp/DwMlp/GluMlp/GatedMlp/ConvMlp (FFN 변형들)
    ├── main.py                # 학습/평가 엔트리(DeiT 스크립트 포크) + --vitality 플래그
    ├── engine.py              # train_one_epoch / evaluate (+ sparse 통계 누적)
    ├── patch_embed.py         # PatchEmbed (Conv2d 패치 임베딩)
    ├── drop.py                # DropPath (미열람, timm 표준 — 이름만 언급)
    ├── losses.py              # DistillationLoss (DeiT 표준, 미열람 — 이름만)
    ├── samplers.py            # RASampler (DeiT 표준, 미열람 — 이름만)
    ├── utils.py               # 분산학습/로깅 유틸 + RunningMean (미열람 — 이름만)
    ├── hubconf.py             # torch.hub 진입점 (미열람 — 이름만)
    └── run_with_submitit.py   # SLURM submitit 실행 래퍼 (미열람 — 이름만)
```

### 2.2 제외(third-party/vendor/생성물/문서 이미지) — 이름만 언급

- `figures/` : `hardware_overall.png`, `ViTALiTY-workflow.png`, `TaylorAttentionFlow2.png` (HW/알고리즘 다이어그램 이미지, 코드 아님)
- `.github/` : `deit.png`, `cait.png`, `resmlp.png`, `patch_convnet.png`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md` (DeiT 포크 잔재/거버넌스 문서)
- `.git/` : 버전관리 메타데이터

### 2.3 결손/미존재 모듈 (주의)

- `main.py:20,24` 가 `from datasets import build_dataset`, `import models` 를 import하지만 **`datasets.py`, `models.py` 파일이 repo에 없다** (Glob `**/*.py` 결과에 미포함). 즉 **이 repo만으로는 학습 스크립트가 바로 실행되지 않는다** (DeiT 원본의 `datasets.py`/`models.py`가 누락된 부분 포크로 추정). `--vitality` 플래그를 `create_model(..., vitality=args.vitality)` 로 모델에 전달하므로(`main.py:248-256`), 누락된 `models.py`가 `vision_transformer.VisionTransformer(vitality=...)` 를 래핑해 `deit_tiny_patch16_224` 등을 등록할 것으로 추정(확인 불가).

---

## 3. 핵심 모듈 정밀 분석 (라인 근거)

### 3.1 `Attention` — ViTALiTy Linear Taylor Attention (가장 중요)

파일: `src/vision_transformer.py:92-137`

#### (a) 생성자 (`vision_transformer.py:92-106`)
- 표준 multi-head attention 구성: `self.qkv = nn.Linear(dim, dim*3)`, `head_dim = dim//num_heads`, `self.scale = head_dim ** -0.5` (`:96-101`).
- `vitality=True`일 때만 추가로 두 개의 양자화 측정기를 생성 (`:104-106`):
  - `self.quant = QuantMeasure(4, ...)` — **4-bit** 양자화 (sparse 마스크 계산용 Q,K에 적용).
  - `self.quant_16 = QuantMeasure(16, ...)` — **16-bit** 양자화 (생성만 되고 forward에서 사용처 미발견, 확인 불가/잠재적 dead code).

#### (b) forward — QKV 투영 (`vision_transformer.py:108-112`)
```
B, N, C = x.shape
qkv = self.qkv(x).reshape(B, N, 3, num_heads, C//num_heads).permute(2,0,3,1,4)
q, k, v = qkv[0], qkv[1], qkv[2]
```
표준 ViT QKV 분리. 이후 `q,k,v` shape = `(B, num_heads, N, head_dim)`.

#### (c) ViTALiTy 경로 (`vision_transformer.py:115-128`) — 핵심 알고리즘

논문 Algorithm 1에 대응한다고 주석 명시 (`:114`). 단계별 정밀 분석:

1. **Sparse("strong") 항 — 양자화된 softmax + 임계 마스크** (`:116-120`)
   ```
   quant_q, quant_k = self.quant(q, 4), self.quant(k, 4)          # 4-bit Q,K
   quant_attn = (quant_q @ quant_k.transpose(-2,-1)) * self.scale # QK^T (저정밀)
   quant_attn = quant_attn.softmax(dim=-1)                        # softmax
   mask = quant_attn > 0.002                                      # 임계값 0.002
   sparse = mask * quant_attn                                     # strong/sparse map
   ```
   - 즉 **저정밀(4-bit) softmax attention을 한 번 계산**한 뒤, **0.002 임계값**으로 작은 값을 버려 희소 행렬 `sparse`를 만든다. 이것이 논문의 "strong"(고차 Taylor ≈ sparse) 항.
   - 주의: 여기서 `quant_attn`은 여전히 N×N (quadratic) 행렬을 형성한다. 즉 **학습 시점의 sparse 항 계산 자체는 quadratic**이며, 추론에서만 이를 버리고 linear 경로만 쓰는 것이 논문 주장과 일치(`README.md:26`).

2. **Low-rank("weak") 항 — Taylor 1차 linear attention** (`:122-126`)
   ```
   k = k - k.mean(dim=-1, keepdim=True)   # K 중심화(centering)
   kv = k.transpose(-2,-1) @ v            # G = K^T·V  (전역 컨텍스트 행렬, head_dim×head_dim)
   attn = (sparse @ v + (q @ kv)) * self.scale
   ```
   - **`k = k - k.mean(...)`** (`:122`): 키를 마지막 차원(head_dim) 기준 중심화. Taylor 근사에서 1차 항을 안정화/정규화하는 전처리(논문의 pre-processing 단계와 대응 추정).
   - **`kv = k.T @ v`** (`:124`): shape `(B, H, head_dim, head_dim)`. 이것이 README의 **global context matrix G** (`README.md:34`). N에 선형(O(N·d²)).
   - **`q @ kv`** (`:126`): `(B,H,N,head_dim) @ (B,H,head_dim,head_dim)` → `(B,H,N,head_dim)`. **시퀀스 길이 N에 선형**인 low-rank 출력.
   - **`sparse @ v`** (`:126`): sparse(N×N) × V → 국소 보강 항.
   - 최종 `attn = (sparse@v + q@kv) * scale`: **low-rank linear 항 + sparse 항의 합**이 ViTALiTy attention 출력. (단, 학습 경로에서는 N×N sparse가 들어가므로 완전 선형은 아님; 추론 전용 경로는 `q@kv`만 남는 구조로 의도됨 — 본 forward에는 추론 전용 분기 코드가 별도로 없어, eval 시에도 sparse 경로를 타는 것으로 보임. 이는 "추론은 low-rank만" 주장과 코드 간 **불일치 가능성**, 확인 필요/추정.)

3. **출력 재배열** (`:127-128`)
   ```
   drop_attn = self.attn_drop(attn)
   x = drop_attn.transpose(1,2).reshape(B, N, C)
   ```
   주의: 표준 attention과 달리 `attn`을 그대로 출력으로 reshape한다(이미 V가 곱해진 값이므로). vanilla 경로(`:133`)의 `(drop_attn @ v)`와 대조적으로 ViTALiTy 경로는 `attn`이 이미 출력(value-weighted)임.

#### (d) Vanilla 경로 (`vision_transformer.py:129-133`)
```
attn = (q @ k.transpose(-2,-1)) * self.scale
attn = attn.softmax(dim=-1)
x = (self.attn_drop(attn) @ v).transpose(1,2).reshape(B,N,C)
```
표준 softmax attention. `--vitality` 미지정 시 baseline.

#### (e) 출력 projection + 반환 (`vision_transformer.py:135-137`)
- `x = self.proj_drop(self.proj(x))`.
- **`return x, attn`** — 두 번째 반환값 `attn`을 상위에서 sparse 통계 수집에 사용.

**라인 근거 요약**: ViTALiTy 알고리즘의 전부가 `vision_transformer.py:115-128` 13줄에 담겨 있다. 핵심 두 줄은 `:124`(G=K^T·V) 와 `:126`(sparse@v + q@kv).

### 3.2 `Block` — Transformer 인코더 블록 (`vision_transformer.py:140-160`)

- `norm1 → Attention → residual → norm2 → Mlp → residual` 의 pre-norm 구조 (`:154-159`).
- forward 입력이 `[x, attn_list]` 튜플 (`:155`): 블록을 통과할 때마다 `attn_list.append(attn.clone())` 로 **모든 레이어의 attention map을 수집** (`:157`). 이는 sparse 비율 통계/시각화용.
- `self.attn` 은 `Attention(dim, vitality=vitality, ...)` (`:147`). `FNetBlock`(`:84-90`, FFT 기반 토큰 믹싱)도 정의되어 있으나 `:146`에서 주석 처리되어 미사용(실험 잔재).

### 3.3 `VisionTransformer` (`vision_transformer.py:163-306`)

- DeiT 호환 ViT: patch_embed → cls(+dist) token → pos_embed → blocks → norm → head (`:202-235`).
- `vitality` 플래그를 모든 `Block`에 전파 (`:212-216`).
- `forward_features` (`:275-290`): `x = [x, []]` 로 attn_list 컨테이너를 만들어 `self.blocks(x)` 에 통과, 마지막에 `(cls_token_feature, attn_list)` 반환.
- `forward` (`:292-306`): `global sparse_list` 를 비우고(`:293-294`), 헤드 통과 후 **`return x, np.array(sparse_list)`** — sparse 통계 배열을 함께 반환. (단, `sparse_list`는 본 파일에서 append되는 곳이 보이지 않아 비어 있을 수 있음 — `:82`에서 모듈 전역 선언만 됨; 통계 채움 로직은 누락/주석화된 `utils` 시각화 함수(`:9` 주석 import)에 있었던 것으로 추정.)
- distillation 분기(`:296-302`): dist token 사용 시 head/head_dist 평균(추론) 또는 둘 다 반환(학습).

### 3.4 양자화 인프라 (`quantize.py`, `quant_utils.py`)

ViTALiTy의 HW 효율 주장을 뒷받침하는 정밀도 인프라. 두 파일이 **출처가 다른 두 양자화 구현**을 담고 있다.

#### (a) `quantize.py` — CPT/LDP 계열 동적 양자화
- `calculate_qparams` (`:32-51`): min/max → range, zero_point 산출(비대칭 양자화 파라미터). `reduce_type='extreme'`/`'mean'` 지원.
- `my_clamp_round(InplaceFunction)` (`:54-72`): round+clamp의 STE. backward는 전 구간 gradient 통과(`:71`, 저자 커스텀).
- `UniformQuantize.quantize` (`:128-177`): 핵심 fake-quant. `scale=range/(qmax-qmin)`, affine 양자화 후 dequant. `prec_sf`(learnable precision scale factor)로 **학습 가능한 비트폭**(min_bit~max_bit) 지원 (`:144-150`) — 즉 LDP(Learnable Dynamic Precision) 류.
- `UniformQuantizeGrad` (`:180-209`): gradient 양자화(CPT 설정, `:205` `reduce_type='extreme'`).
- `QuantMeasure(nn.Module)` (`:224-267`): **EMA 기반 range/zero_point 추적**. 학습 중 momentum으로 running stats 갱신, eval 시 고정값 사용 → **ViTALiTy `Attention`의 4-bit Q/K 양자화기가 이 클래스** (`vision_transformer.py:105` ↔ `quantize.py:224`).
- `QConv2d`/`QLinear` (`:270-494`): 양자화 conv/linear. learnable `prec_w` 파라미터로 비트폭 학습(`:291,431`). mobilenet depthwise 특수 처리(`:309-331`). **단, ViTALiTy attention 본문은 표준 `nn.Linear`를 쓰므로(`vision_transformer.py:99,101`) 이 Q레이어들은 본 모델 경로에 직접 연결되지 않음** — 별도 양자화 실험/CNN용 잔재로 추정.
- `RangeBN`/`RangeBN1d` (`:499-574`): range 기반 정규화 BN(양자화 친화).

#### (b) `quant_utils.py` — IntelLabs nlp-architect 포팅
- 파일 상단 출처 명시: IntelLabs/nlp-architect (`:1-3`).
- `get_dynamic_scale`/`get_scale`/`calc_max_quant_value`/`quantize`/`dequantize` (`:15-41`): **대칭(symmetric)** 양자화 기본 함수들.
- `FakeLinearQuantizationWithSTE` (`:45-59`): STE 기반 fake-quant.
- `QuantizedLayer`(ABC)/`QuantizedLinear`/`QuantizedEmbedding` (`:71-380`): QAT 가능한 레이어. EMA/DYNAMIC/NONE 모드(`:62-66`).
- `QuantizedMatMul` (`:437-477`): 두 입력(x,y)을 각각 EMA threshold로 대칭 양자화 후 matmul. `input_bits ∈ {2,4,6,8}` 검증(`:440`).
- **`build_quant_matmul(quant_bits)`** (`:479-482`): `QuantizedMatMul('sym', input_bits=quant_bits)` 생성 헬퍼. `vision_transformer.py:13`에서 import되지만 **`Attention` forward에서 실제 호출되는 곳은 발견되지 않음** (확인 불가/미연결). 즉 양자화 matmul로 attention을 돌리는 변형은 코드에 준비만 되고 비활성.

**요약**: ViTALiTy 모델 추론 경로에서 실제 활성화되는 양자화는 `Attention`의 4-bit `QuantMeasure`(sparse 마스크 계산용 Q/K)뿐이다(`vision_transformer.py:116`). 나머지 양자화 기계장치(QLinear/QuantizedMatMul 등)는 import/정의만 되어 있고 모델 forward에 미연결.

### 3.5 FFN/MLP 변형 (`mlp.py`)

- `Mlp` (`:5-23`): 표준 ViT FFN(fc1→GELU→drop→fc2→drop). `Block`이 사용하는 것이 이것 (`vision_transformer.py:152`).
- 추가 변형들은 정의만 되어 있고 본 모델 미사용(이름만): `DwMlp`(depthwise conv 삽입, CLS 토큰 분리 처리 `:131-150`), `GluMlp`(`:152`), `GatedMlp`(gMLP, `:182`), `ConvMlp`(1x1 conv, `:211`), `BaseConv`/`DW`(`:46-108`).

### 3.6 학습/평가 루프 (`main.py`, `engine.py`)

- `main.py`: DeiT 학습 스크립트 포크. `--vitality` 플래그 추가(`:173`), `create_model(..., vitality=args.vitality)` 로 전달(`:248-256`). 그 외 distillation, mixup, EMA, 분산학습, finetune(pos_embed bicubic 보간 `:272-290`)은 DeiT 표준.
- `engine.py`:
  - `train_one_epoch` (`:18-64`): `[outputs, sparse] = model(samples)` 로 sparse 통계 수신(`:37`), 배치별 `attn_sparse = (attn_sparse + sparse)/2` 누적(`:40-43`) → 에폭 평균 sparse 비율 추적. AMP autocast는 주석 처리(`:36`).
  - `evaluate` (`:67-97`): `[output, sparse] = model(images)`(`:83`), acc@1/acc@5 측정. **eval에서도 모델이 sparse를 반환** → 3.1(c)에서 지적한 "추론도 sparse 경로를 탄다"는 정황 보강.

### 3.7 보조 모듈

- `patch_embed.py`: `PatchEmbed`(`:29-53`) — `Conv2d(in_chans, embed_dim, kernel=patch, stride=patch)` 후 flatten→transpose(BCHW→BNC). `to_2tuple`/`make_divisible` 유틸 포함. timm 표준 구현.
- `drop.py`(DropPath), `losses.py`(DistillationLoss), `samplers.py`(RASampler), `utils.py`(분산/로깅 + `RunningMean`), `hubconf.py`, `run_with_submitit.py` 는 DeiT/timm 표준 또는 인프라로, ViTALiTy 고유 로직 아님(미열람, 이름만).

---

## 4. 데이터 플로우

### 4.1 소프트웨어 학습 데이터 플로우 (코드 근거)
```
이미지 (B,3,224,224)
 → PatchEmbed: Conv2d(stride=16) → (B, N=196, C)            [patch_embed.py:45-52]
 → cls(+dist) token concat + pos_embed                       [vision_transformer.py:277-282]
 → Block × depth:                                            [vision_transformer.py:212-216]
      norm1 → Attention(vitality) → +residual                [:156-158]
      norm2 → Mlp → +residual                                [:159]
      (각 블록 attn_list.append)                              [:157]
 → norm → head → logits                                      [:286, :304]
 → (logits, sparse_stats) 반환                                [:306]
```

### 4.2 ViTALiTy attention 내부 데이터 플로우 (핵심, `vision_transformer.py:115-128`)
```
            ┌─ quant(Q,4), quant(K,4) → QK^T·scale → softmax → (>0.002 mask) → sparse  (N×N, "strong")
 Q,K,V ─────┤
            └─ K ← K - mean(K)  →  G = K^T·V (d×d, "global context")
                                   q@G  (N×d, low-rank linear, O(N))
 출력: (sparse@V + q@G) · scale → proj → x
```
- low-rank 경로(`q@G`): **O(N·d²)** 선형.
- sparse 경로(`sparse@V`): N×N 형성 → **O(N²·d)** (학습 시 존재; 추론에서 제거가 논문 의도).

### 4.3 하드웨어 데이터 플로우 (README 근거, 코드 없음 → "확인 불가")
- 4계층 메모리 DRAM→SRAM→NoC→Regs (`README.md:43`).
- chunk별 sub-processor = pre-processor(누산기 배열=토큰별 합, divider 배열, adder 배열) + systolic array (`README.md:43`).
- SA를 두 부분으로 분할: **SA-Diag**(행렬×대각행렬, 곱셈 적음) + **SA-General**(일반 행렬곱) (`README.md:43`).
- intra-layer pipeline + down-forward accumulation dataflow (`README.md:37`).
- ※ 위 모든 HW 데이터플로우는 RTL/HLS/시뮬레이터 코드 부재로 **코드 검증 불가**.

---

## 5. HW/SW 매핑

| SW 연산 (코드 라인) | 대응 HW 블록 (README) | 비고 |
|---|---|---|
| `K - K.mean(-1)` (`vision_transformer.py:122`) | pre-processor: accumulator 배열(토큰/컬럼 합) + divider/adder 배열 (`README.md:43`) | centering = 합/나눗셈/뺄셈 → pre-processor 매핑 (추정) |
| `G = K^T·V` (`:124`) | SA-General (일반 행렬곱) (`README.md:43`) | head_dim×head_dim 컨텍스트 |
| `q @ G` (`:126`) | SA-General | low-rank linear, 선형 복잡도 |
| `sparse @ V` (`:126`) | SA-General + sparse 처리 / 추론 시 제거 | strong 항, 추론 비활성 의도 |
| 대각/스케일링 류 곱 | SA-Diag (행렬×대각행렬) (`README.md:43`) | 곱셈 수 적은 연산 분리 (추정) |
| 4-bit Q/K 양자화 (`:116`, `quantize.py:224`) | (HW 정밀도 경로) | 코드상 sparse 마스크 계산용; HW 매핑은 확인 불가 |

**매핑의 한계(중요)**: HW RTL/HLS가 repo에 없으므로 위 표의 우측 열은 전부 README 텍스트 기반 **추정**이며 코드로 검증 불가. SW↔HW를 잇는 컴파일러/스케줄러/시뮬레이터도 부재.

---

## 6. 빌드·실행

- 환경: `pip install -r requirement.txt` (`README.md:49`).
- 학습(DeiT-Tiny, ViTALiTy): `python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --data-path <IMAGENET> --output_dir '' --vitality` (`README.md:53-55`).
- 추론: 위에 `--eval` 추가 (`README.md:59-61`).
- `--vitality` 미지정 시 vanilla softmax baseline (`README.md:50-52`).
- **실행 가능성 경고**: 2.3에서 지적했듯 `main.py`가 import하는 `datasets.py`/`models.py`가 repo에 없어, **현 repo 상태로는 그대로 실행 불가**(누락 파일 필요). 또한 `from vision_transformer import partial/OrderedDict` 미import(`:199,222`에서 `partial`/`OrderedDict` 사용하나 import 없음) → 모델 인스턴스화 시 NameError 가능성(확인=라인: `:199` `partial` 사용, 파일 상단 import에 `partial` 없음). 즉 본 repo는 **부분 공개/리팩토링 미완 상태**로 추정.

---

## 7. 의존성 (`requirement.txt`)

- 핵심: `torch`, `torchvision`, **`timm`**(create_model/scheduler/optimizer/Mixup/loss 전반, `main.py:13-18`), `numpy`, `einops`.
- 양자화/실험 보조: `cox`, `dill`, `robustness`, `sympy`, `ml-collections`, `easydict`.
- 로깅/시각화: `tensorboard(X)`, `wandb`, `matplotlib`, `tqdm`.
- 영상/과학: `scipy`, `scikit-learn`/`sklearn`, `scikit-image`, `imageio`, `pillow`, `pandas`.
- `odach`(객체검출 TTA) — 본 분류 모델과 무관, 잔재 추정.
- **하드웨어 EDA 의존성 없음**(Vitis/Vivado/verilator 등 부재) → HW 코드 부재와 일관.

---

## 8. 강점·한계

### 강점
- **알고리즘 핵심이 명료**: linear Taylor attention의 본질(G=K^T·V, q@G)이 단 13줄(`vision_transformer.py:115-128`)에 응축, 재현/이식이 쉬움.
- **선형 복잡도 컨텍스트**: `K^T·V` 전역 컨텍스트로 N에 선형(O(N·d²)) — 긴 시퀀스/고해상도 ViT에 유리.
- **low-rank + sparse 하이브리드**: 정확도 보강용 sparse 항을 학습에 포함, 추론은 low-rank만 쓰는 분리 설계(논문 의도).
- **양자화 인프라 동봉**: 4-bit/learnable-precision/EMA/STE 등 폭넓은 양자화 도구가 준비됨.

### 한계
- **HW 코드 전무**: RTL/HLS/시뮬레이터 부재. 가속기 정량 평가(latency/area/power) 재현 불가. 본 repo는 사실상 **알고리즘 reference만** 제공.
- **실행 불완전**: `datasets.py`/`models.py` 누락, `partial`/`OrderedDict` 미import 등으로 현 상태 그대로는 미실행(2.3, 6 참조).
- **dead/미연결 코드 다수**: `quant_16`(`:106`), `build_quant_matmul`(import만), `FNetBlock`(주석화 `:146`), `QLinear`/`QuantizedMatMul`, MLP 변형들 — 모델 forward 미연결.
- **추론 분기 부재 가능성**: forward에 train/eval 분기가 없어 eval에서도 sparse(N×N) 경로를 타는 것으로 보임 → "추론은 low-rank only" 주장과 코드 간 불일치 가능(추정, `vision_transformer.py:115-128`, `engine.py:83`).
- **sparse_list 미충전**: `forward`가 `np.array(sparse_list)` 반환하나(`:306`) 채우는 로직 부재 → 통계 빈 배열 가능성(추정).

---

## 9. 우리 프로젝트(PRJXR-HBTXR) 시사점

가정: 우리 프로젝트는 **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적**.

1. **Linear attention 채택 가치**: HG-PIPE류 완전 파이프라인 가속기에서 softmax의 N×N 버퍼/지연이 병목이다. ViTALiTy의 `G=K^T·V`(`vision_transformer.py:124`) + `q@G`(`:126`)는 N×N 행렬을 head_dim×head_dim(예: 64×64)으로 축소 → **온칩 버퍼·BRAM 사용량을 N²→d² 로 격감**, 파이프라인 깊이/스루풋에 직접 유리. XR의 고프레임레이트(저지연) 요구에 부합.
2. **추론 전용 low-rank 경로만 HW화**: 학습의 sparse 항은 버리고 추론은 `q@G`만 매핑하면, 가속기는 **두 번의 GEMM(K^T·V, q·G)** 만 처리하면 됨 → systolic array(HG-PIPE의 PE 배열)에 자연스럽게 맵핑. SA-Diag/SA-General 분할 아이디어(`README.md:43`)도 우리 SA 분할 설계에 참고 가능(추정).
3. **K centering 전처리(`:122`)의 HW 비용 주의**: 토큰별 평균 차감은 reduction+broadcast → 파이프라인에 reduction 스테이지 추가 필요. HG-PIPE의 LayerNorm/softmax-free 흐름에 통합 가능하나 추가 누산기 필요(README의 pre-processor accumulator와 동일 맥락).
4. **양자화 결합**: 본 repo의 4-bit QuantMeasure/EMA(`quantize.py:224`)와 우리 양자화 파이프라인(ViT-Quantization 계열)을 결합하면 linear attention을 저정밀로 HW화 가능. 단 attention 본문은 미양자화이므로 `build_quant_matmul`을 실제 연결하는 작업이 선행돼야 함(현 repo는 미연결).
5. **재현 시 주의**: 본 repo는 알고리즘 reference로만 쓰고, 실제 학습/정확도 재현은 누락 파일(`datasets.py`/`models.py`) 보완 또는 원 DeiT와의 병합이 필요. 가속기 수치는 논문 본문/별도 자료를 봐야 함(코드 부재).
6. **XR 시선추적 관점**: linear attention의 저지연·저메모리 특성은 안구 영상의 토큰 수가 많은 고해상도 입력에서 특히 이득. 다만 정확도 보강용 sparse 항을 추론에서 빼면 정확도 저하 가능 → XR 정밀도 요구가 높으면 추론에도 경량 sparse를 일부 유지하는 절충 검토(추정).

---

## 10. 근거 표기 범례

- **확인=라인**: 본문에 `파일명:라인` 표기된 항목은 실제 소스 Read로 검증됨.
- **README 근거(코드 부재)**: HW 가속기 관련 모든 서술(섹션 4.3, 5 우측열)은 `README.md` 텍스트 기반이며 RTL/HLS/시뮬레이터 코드가 repo에 없어 **코드로는 확인 불가**.
- **추정**: "추정"으로 명시한 항목(누락 `models.py` 역할, 추론 분기 부재 영향, SA-Diag 매핑, XR 절충 등)은 정황 기반 해석.
- **확인 불가**: `quant_16`/`build_quant_matmul` 미사용, `sparse_list` 충전 로직 등 — 코드상 연결점을 찾지 못함.
- **미열람(이름만)**: `drop.py`, `losses.py`, `samplers.py`, `utils.py`, `hubconf.py`, `run_with_submitit.py` (DeiT/timm 표준 인프라로 판단해 정밀 분석 생략).
