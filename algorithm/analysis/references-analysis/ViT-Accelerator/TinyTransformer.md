# TinyTransformer 정밀 분석

## 1. 개요
- **한 줄 요약**: RISC-V SoC용 **Tiny Transformer AI 가속기의 PyTorch "Golden Model"(참조 모델)**. RTL 검증 기준값 생성과 DL 연산→HW 블록(Systolic Array/SRAM/Vector ALU) 매핑 가이드를 목적으로 함.
- **목적**: (1) RTL 검증용 정확한 황금값 제공, (2) Attention/MLP/LayerNorm/Embedding을 하드웨어 블록으로 매핑하는 설계 지침(코드 주석)을 제공. **하드웨어 구현체는 없고**, HW를 의식해 작성된 SW 모델만 존재.
- **출처**: README(베트남어). 자체 SoC 설계 프로젝트의 일부. 원 논문 명시 없음.
- **타깃**: RISC-V CPU + 전용 AI 가속기 SoC(제한된 칩 자원). 멀티모달(Vision 64×64×3 patch 8×8, Audio MFCC 40feat).

## 2. 디렉토리 구조 (자체 소스)
```
TinyTransformer/
├── README.md          # 아키텍처 사양 + HW 매핑 설계 노트
└── src/
    ├── config.py      # TinyConfig: 하이퍼파라미터/HW 제약
    ├── core.py        # Encoder(Pre-LN) + TinyTransformer(top)
    ├── layers.py      # MultiHeadAttention / Mlp / LayerNorm
    └── embeddings.py  # PatchEmbed(Vision) / AudioEmbed(Audio)
```
- **제외 third-party**: 없음(전부 자체 PyTorch). PyTorch는 외부 프레임워크.

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 config.py — TinyConfig (config.py:1~16)
HW 제약을 명시: `embed_dim=64`, `num_heads=2`(head_dim=32), `depth=3`, `ffn_mul=2`(hidden=128), `num_classes=10`, `max_seq_len=64`. Vision: img 64, patch 8, in_ch 3 → 패치 8×8=64토큰. Audio: 40 features. 모든 차원이 작은 2의 거듭제곱 근방 — systolic array/SRAM 크기에 맞춘 의도.

### 3.2 layers.py — 핵심 연산자 (HW 매핑 주석 포함, 가장 중요)
- **MultiHeadAttention** (layers.py:4~44):
  - `self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=False)` — **Wq/Wk/Wv를 하나의 텐서로 융합**(주석 layers.py:12: SRAM 로드 대역폭 최적화). HW에서 단일 가중치 fetch로 Q/K/V 동시 생성 의도.
  - `attn = (q @ k.T) * scale` → `softmax(dim=-1)` → `attn @ v` → `proj`. 주석(layers.py:21,34)이 Q·Kᵀ, Attn·V를 **Systolic Array**에, **Softmax exp(x)를 LUT/다항식 근사**로 매핑하라 명시.
  - scale = head_dim^-0.5 (layers.py:10).
- **Mlp** (layers.py:46~64): fc1(64→128)→ReLU→fc2(128→64). 주석(layers.py:48,55): Attention의 **MMU/Systolic Array 재사용**(면적 절감), ReLU는 부호 비교로 구현.
- **LayerNorm** (layers.py:66~87): 표준 mean/var → `(x-u)/sqrt(s+eps)·γ+β`. 주석(layers.py:67~72): mean/var를 **Vector ALU**에, inverse-sqrt를 **Fast InvSqrt 또는 bit-shift**로 구현해 제수기 회피 권장.

### 3.3 core.py — Encoder & Top (core.py:1~43)
- **Encoder**: **Pre-normalization** 구조 — `x = x + attn(norm1(x)); x = x + mlp(norm2(x))` (core.py:14~17). residual + pre-LN.
- **TinyTransformer**: learnable `pos_embed`(1×64×64) 더하기 → depth(3) Encoder → 최종 LayerNorm → **Global Average Pooling**(mean dim=1) → classifier head(64→10) (core.py:30~43). 가변 길이 입력은 pos_embed 길이로 잘라 맞춤(core.py:32~33).

### 3.4 embeddings.py — 멀티모달 전처리 (embeddings.py:1~26)
- **PatchEmbed(Vision)**: `Conv2d(3, 64, kernel=8, stride=8)` → [B,3,64,64]→[B,64,8,8]→flatten→[B,64,64](64 토큰). 주석: Transformer MAC 재사용 or 별도 Conv2d accel or smart DMA로 매핑.
- **AudioEmbed(Audio)**: `Linear(40, 64)` 선형 사영. 주석: Transformer MAC 재사용.

## 4. 데이터플로우 / 실행 흐름
- 입력(이미지 또는 오디오) → Embed → +pos_embed → [Pre-LN Encoder]×3 → LN → GAP → head → 클래스 logits(10).
- **HW 매핑 의도(설계 의도, 구현 아님)**: QKV 융합 가중치를 SRAM에서 단일 fetch → Systolic Array가 QKᵀ/AttnV/Linear MMU 담당 → Softmax는 LUT exp → LayerNorm은 Vector ALU + InvSqrt. MLP가 Attention MMU 재사용(시분할).
- **데이터타입**: 모델 자체는 fp32(PyTorch). 양자화는 미구현(추후 RTL에서 결정될 영역).

## 5. HW/SW 매핑
- 본 repo는 **SW(golden) 계층만** 존재. RTL/HLS 구현은 별도 저장소(미포함). 본 모델의 각 연산이 어떤 HW 블록에 대응하는지는 README/코드 주석으로 문서화(§3 참조). 즉 "SW↔HW 매핑 명세서" 역할.

## 6. 빌드·실행
- 순수 PyTorch 모듈. 명시적 실행 스크립트/학습 루프는 저장소에 없음(추정). config.py로 모델 인스턴스화 후 forward 호출하여 RTL 비교용 텐서 덤프하는 용도로 추정.

## 7. 의존성
- PyTorch(torch, torch.nn)만. 외부 가속/툴 의존 없음.

## 8. 강점 / 한계 / 리스크
- **강점**: 매우 작고 읽기 쉬운 멀티모달 ViT/Transformer 정의 + **HW 매핑 의도가 명시적으로 주석화** → 알고리즘↔아키텍처 매핑 사고의 좋은 표본. QKV 융합·MMU 재사용·LUT softmax·InvSqrt LayerNorm 등 우리가 쓰는 기법과 직접 겹침.
- **한계**: 실제 가속 코드·합성 결과·정확도 수치 전무. 학습/데이터 파이프라인 없음. 멀티모달이지만 매우 toy 규모(embed 64, depth 3).

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA, HG-PIPE 계열) 관점 시사점
- **직접 차용 가능한 설계 결정**:
  - **QKV 가중치 융합**(단일 Linear, 단일 fetch) → on-chip 가중치 버퍼/DMA 대역폭 절감. ViT 가속기 가중치 레이아웃에 적용.
  - **MMU(Systolic Array) 재사용**: Attention QKᵀ/AttnV/Linear/MLP를 단일 MAC 어레이가 시분할 → 면적 효율. HG-PIPE류에서 PE 어레이 재사용 정책 참고.
  - **Softmax LUT exp, LayerNorm InvSqrt(bit-shift/Fast InvSqrt)** — 비선형 유닛 저비용 구현 지침(hls-fpga-accelerators의 LUT exp와 일맥상통).
  - **Pre-LN + residual + GAP head** — ViT 분류 백본 표준 구조(우리 모델 구조 정의 시 참조).
- **활용 방식**: 우리 RTL/HLS 구현의 **검증용 golden model**로 동일 방식(연산별 텐서 덤프) 채택 권장. 멀티모달(Vision/Audio)을 XR 시선추적 입력(이미지+센서)으로 확장하는 발판으로도 활용 가능.

## 10. 근거/한계 표기
- 근거: config.py / core.py / layers.py / embeddings.py 전문, README 직접 확인.
- 학습 스크립트·실행 진입점 부재 → 실제 사용 방식은 **추정**.
- 실제 RTL 구현체·정확도·면적은 본 repo에 없음 → **확인 불가**.
