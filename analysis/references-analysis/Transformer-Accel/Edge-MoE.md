# Edge-MoE 코드베이스 정밀 분석

> 대상: `REF/Transformer-Accel/Edge-MoE`
> 논문: Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-level Sparsity via Mixture-of-Experts (ICCAD 2023, arXiv:2305.18691)
> 저자: Rishov Sarkar, Hanxue Liang, Zhiwen Fan, Zhangyang Wang, Cong Hao (Georgia Tech / UT Austin)
> 분석 방식: 자체 소스(`src/*.cpp`, `include/*.hpp`, `testbench/*.cpp`, `Makefile`) 전수 Read. 라인 근거는 `파일명:라인` 형식으로 표기. 바이너리/생성물은 이름만 언급.

---

## 1. 개요

Edge-MoE는 **멀티태스크 ViT(Vision Transformer)**를 위한 **end-to-end FPGA 가속기**다. 핵심 아이디어는 MoE(Mixture-of-Experts)를 통한 **태스크 수준 희소성(task-level sparsity)** 활용과, 모든 expert 가중치를 온칩에 상주시키지 않고 **선택된 expert만 동적으로 외부 메모리(HBM/DDR)에서 로드**하여 메모리를 절약하는 것이다.

이 저장소는 **README의 설명대로 README + 사전합성 비트스트림 + 가중치 바이너리 + HLS C++ 소스**로 구성된 공개 릴리스다. 구현 전체가 **Vitis HLS 2021.1** 기반 C++로 작성되었으며(`Makefile:1`), RTL(.v/.sv) 자체 소스는 없다(비트스트림만 사전 빌드물로 존재). 즉 RTL은 Vitis HLS가 이 C++에서 합성한 산출물로 추정된다.

모델 구성(확인: `include/model.hpp`):
- 입력: 3채널, 256×128 이미지 (`model.hpp:4-6`)
- 패치: 16×16, 패치 수 = (256/16)×(128/16) + 1(cls token) = **129** (`model.hpp:12-14`)
- 임베딩 차원 FEATURE_DIM = 192, ViT MLP hidden = 768, Expert hidden = 384 (`model.hpp:8-10`)
- 레이어 12개, 헤드 3개 (`model.hpp:16-17`)
- Expert 16개 중 top-2 선택 (`model.hpp:19-20`)
- **짝수 레이어 = 일반 MLP, 홀수 레이어 = MoE** (구조는 `ViT_compute.cpp:178-202`에서 `layer % 2`로 분기)

이것은 SegFormer/멀티태스크 dense-prediction형 ViT로 추정된다(256×128 비정방 입력, cls token 포함, 가중치 파일에 `w_gate_T_task0/task1`이 존재 → 태스크별 게이트). "확인 불가": 정확한 다운스트림 태스크(세그멘테이션/깊이 등)는 소스만으로는 단정 불가.

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (분석 대상)

```
Edge-MoE/
├─ Makefile                  # csim 빌드 스크립트 (Vitis HLS include 링크)
├─ README.md                 # 논문 메타데이터
├─ include/                  # 선언 헤더 (인터페이스 + 자료형 + 모델 상수)
│   ├─ dcl.hpp               # 공통 include 묶음
│   ├─ model.hpp             # 모델 하이퍼파라미터 (constexpr)
│   ├─ datatypes.hpp         # ap_fixed 양자화 자료형 정의
│   ├─ hardware.hpp          # AXI 폭/블록/벡터 타입 정의
│   ├─ util.hpp              # FOR_BLOCK/FOR_OFFSET 매크로, constexpr 유틸
│   ├─ kernel.hpp            # 최상위 ViT_compute 커널 선언
│   ├─ conv.hpp              # 패치 임베딩 선언
│   ├─ attention.hpp         # attention 커널 선언 + AttentionLinear enum
│   ├─ linear.hpp            # linear(matmul) 엔진 선언 + ping/pong 버퍼
│   ├─ moe.hpp               # MoE 게이팅/expert 큐 선언
│   ├─ layernorm.hpp         # LayerNorm 선언
│   ├─ gelu.hpp              # GELU 선언
│   ├─ add.hpp               # residual add 선언
│   └─ tbutil.hpp            # 테스트벤치용 float32 → ap_fixed 로더
├─ src/                      # HLS 커널 구현 (핵심)
│   ├─ ViT_compute.cpp       # 최상위 오케스트레이션 (레이어 루프)
│   ├─ attention.cpp         # Q·Kᵀ, softmax, attn·V 데이터플로우
│   ├─ moe.cpp               # 게이팅, top-k, expert 큐, expert MLP 파이프라인
│   ├─ linear.cpp            # 범용 matmul 엔진 (direct/indirect 게더)
│   ├─ layernorm.cpp         # LayerNorm (2-pass dataflow)
│   ├─ conv.cpp              # Conv 기반 패치 임베딩
│   ├─ gelu.cpp              # GELU LUT 보간
│   └─ add.cpp               # residual add
└─ testbench/
    └─ e2e.cpp               # 단일 이미지 end-to-end csim + MSE/MAE 검증
```

### 2.2 제외 항목 (생성물/바이너리, 이름만 언급)

- `.git/` — git 메타데이터
- `bitstream/bitstream.{xclbin,bit,hwh}` — 사전 합성된 FPGA 비트스트림(바이너리, 생성물)
- `weights/*.float32.bin` — 학습된 가중치/입력 이미지/참조 출력 바이너리 (예: `l*_qkv_weight`, `l*_htoh4_weight`, `l*_w_gate_T_task0/task1`, `image.float32.bin`, `pos_embed.float32.bin` 등 다수)
- `images/edge-moe-arch.svg` — 아키텍처 다이어그램(문서용 자산)
- `demo.mp4` — 데모 영상(자산)

> 비고: Makefile은 `src/*.cpp`를 한 파일(`src/_single_file.cpp`)로 cat 후 csim 빌드한다. 이 cat 산출물과 `result`, `*.o/*.d`, `vitis_hls_project/`, `vivado_project/`는 `.gitignore`에 등록된 빌드 생성물이다(`.gitignore:1-9`).

---

## 3. 핵심 모듈 정밀 분석

> 분석 우선순위: MoE > Attention > Linear 엔진 > 최상위 오케스트레이션 > LayerNorm/PatchEmbed/GELU. 모든 커널은 Vitis HLS 데이터플로우/스트림 패러다임을 따른다.

### 3.0 공통 자료형 및 메모리 추상화 (근거: `datatypes.hpp`, `hardware.hpp`)

- **양자화**: feature map은 `ap_fixed<32,10>` (`datatypes.hpp:8`), linear 가중치는 `ap_fixed<16,2>` (`datatypes.hpp:9`), attention bias `ap_fixed<16,7>`, 일반 bias/norm `ap_fixed<16,5>`, patch-embed 가중치 `ap_fixed<16,0>`, 이미지 픽셀 `ap_ufixed<8,0>` (`datatypes.hpp:8-15`). 즉 **혼합 정밀도(weight 16-bit, activation 32-bit fixed)**.
- **블록 단위 메모리**: AXI 전송 폭 256-bit (`hardware.hpp:9`). `FEATURE_BLOCK_SIZE = 256/32 = 8` (`hardware.hpp:10`). feature map은 `fm_block_t = vector<fm_t,8>` 단위로 패킹되어 `patch_blocks_t x[129][24][8]` 형태(`hardware.hpp:17-19`). 이는 AXI burst와 SIMD 병렬도를 동시에 잡는 설계.
- **Linear 타일**: `LINEAR_IN_SIZE = LINEAR_OUT_SIZE = 16` (= 2×8) → matmul 내부 16×16 곱셈 타일 (`hardware.hpp:13-14`).
- **Attention 병렬도**: `ATTN_MATMUL_PARALLEL = 4` → 한 번에 q-patch 4개를 병렬 처리 (`hardware.hpp:15`).
- **유틸 매크로**: `FOR_BLOCK/FOR_OFFSET`는 루프를 (블록, 오프셋)으로 자동 분해해 II=1 파이프라인을 깔끔히 짜기 위한 핵심 DSL이다(`util.hpp:12-49`). 코드 전체가 이 매크로에 의존.

### 3.1 MoE 커널 — `src/moe.cpp` (가장 핵심)

이 파일이 Edge-MoE의 정체성이다. 게이트 점수 계산 → top-k 선택 → expert별 patch 큐 작성 → 선택된 expert만 weight를 ping/pong 더블버퍼로 로드하며 MLP 실행.

**(a) 게이팅 dataflow `compute_gating` (3-스테이지)** — `moe.cpp:272-289`
`#pragma HLS dataflow`로 3개 스테이지를 스트림 연결:
1. `read_gate_inp` (`moe.cpp:56-72`): 정규화된 입력 `gate_inp`를 patch별 feature 블록으로 스트림화.
2. `compute_gating(stream버전)` (`moe.cpp:244-270`): patch 루프마다 ① `compute_gating_for_patch`로 16 expert에 대한 게이트 로짓 계산, ② `update_softmax_info`로 online-softmax 통계 누적, ③ `top_k`로 top-2 선택, ④ `finalize_topk_scores_softmax`로 선택된 2개 점수만 softmax 정규화.
3. `write_gate_results` (`moe.cpp:74-126`): top-2 인덱스/점수를 **expert별 큐**(`expert_queues`, `expert_scores`)에 push하고 각 큐 길이(`expert_queue_lens`) 갱신. 추가로 비어있지 않은 expert만 모은 **metaqueue**(`expert_metaqueue`, `moe.cpp:115-125`)를 만든다 → 다음 단계에서 "실제 사용된 expert만" 순회하는 데 사용.

**(b) 게이트 로짓 계산 `compute_gating_for_patch`** — `moe.cpp:169-195`
`w_gate[16][192]`에 대해 in_dim을 8씩 블록 순회, `#pragma HLS unroll`로 16개 expert 출력을 완전 병렬 MAC(`moe.cpp:182-189`). 입력 1회 로드로 16 expert 로짓 동시 산출.

**(c) Online softmax `update_softmax_info`** — `moe.cpp:197-225`
수치 안정 softmax를 스트리밍으로 구현. 새 최댓값(bias)이 나오면 기존 sum을 `exp(old_bias - new_bias)`로 rescale(`moe.cpp:214-219`). 이는 attention softmax(3.2)와 동일한 패턴 → 코드 재사용 디자인.

**(d) Top-k 삽입정렬 `top_k`** — `moe.cpp:128-167`
16 expert를 순회하며 NUM_SELECTED_EXPERTS(=2) 크기의 정렬된 리스트에 삽입. `shift` 플래그로 한 칸 밀기 구현(`moe.cpp:147-162`). `#pragma HLS pipeline rewind`로 expert 루프 파이프라인.

**(e) Expert MLP 더블버퍼 파이프라인 `compute_moe`** — `moe.cpp:304-436` ★ 최중요
이 함수가 Edge-MoE의 메모리 효율 핵심이다.
1. `compute_gating` 실행 + `zero_output` (`moe.cpp:316-317`).
2. metaqueue의 **첫 expert l1 weight를 ping 버퍼에 선로드**(`moe.cpp:319-333`).
3. metaqueue 순회 루프(`moe.cpp:336-394`): 각 반복에서
   - 현재 expert의 **l1 matmul(htoh4, GELU 적용)** 실행 → `tmp_hidden` (`compute_linear(..., use_gelu=true, use_expert=true)`, `moe.cpp:346-357`),
   - 동시에 **다음에 쓸 l2 weight를 pong에 로드**(`moe.cpp:358-368`),
   - **l2 matmul(h4toh, score 가중합)** 실행 → `out` (`use_score=true`, `moe.cpp:371-382`),
   - 그 사이 **다다음 expert l1 weight를 ping에 다시 로드**(`moe.cpp:383-393`).
   → 즉 weight 로드와 연산을 겹치는 **소프트웨어 파이프라인(prefetch)**. 선택된 expert 가중치만, 또 그 patch에 해당하는 행만 외부메모리에서 가져온다.
4. 루프 잔여(마지막 expert) 처리(`moe.cpp:396-435`).

핵심: expert 가중치 전체(16개 × 2 layer × ~수십만 파라미터)를 온칩에 둘 필요가 없다. 실제 호출된 expert(보통 ≤ NUM_PATCHES×2/평균분산)만 DRAM→온칩 로드 → "Memory-Efficient"의 실체.

**(f) w_gate 로더 `load_w_gate`** — `moe.cpp:19-54`
DRAM의 `w_gate[16][192]`를 256-bit 정렬 블록으로 읽어 온칩 `w_gate`에 transpose-friendly하게 재배치. `array_partition complete`로 16 expert 병렬화(`moe.cpp:25-26`).

### 3.2 Attention 커널 — `src/attention.cpp`

**(a) Q·Kᵀ dataflow `compute_q_matmul_k`(상위)** — `attention.cpp:262-284`
5-스테이지 `#pragma HLS dataflow`:
`read_x`(q 스트림) + `read_kv`(k 스트림) → `compute_q_matmul_k`(내부, QKᵀ MAC) → `finalize_attn`(online softmax 통계) → `write_attn` + `write_attn_softmax_info`.

**(b) QKᵀ MAC `compute_q_matmul_k`(내부)** — `attention.cpp:45-125`
`ATTN_MATMUL_PARALLEL=4` q-patch를 동시에 처리(`q_blocks[4]`, `array_partition complete`, `attention.cpp:53-58`). k-patch 루프 안에서 8-차원 블록 단위 내적 누적, head는 `dim_block`으로 자동 분기(`attention.cpp:94`). `attn_scale`(=0.125 = 1/√(64), `ViT_compute.cpp:64`) 곱(`attention.cpp:104`). 결과를 head별로 누적 후 `attn_stream`에 push.

**(c) Online softmax 통계 `finalize_attn`** — `attention.cpp:127-213`
각 (q_patch, head)별로 running max(bias)/running sum을 유지하며 스트리밍 softmax. `softmax_info_row`에 `[recip(sum), bias]`를 head별로 packing(`attention.cpp:198-199`). q-patch 4-way 병렬에 대한 `#pragma HLS dependence ... distance=4`로 의존성 명시(`attention.cpp:152-153`). 주목: 여기서 **확률을 다 만들지 않고 통계만** 흘려보낸다 → attn·V 단계에서 재계산(메모리 절약).

**(d) attn·V dataflow `compute_attn_matmul_v`(상위)** — `attention.cpp:472-494`
6-스테이지 dataflow. `read_kv`(v) + `read_attn` + `read_attn_softmax_info` → `prepare_attn`(packing) → `compute_attn_matmul_v`(내부) → `write_attn_matmul_v`.

**(e) attn·V MAC `compute_attn_matmul_v`(내부)** — `attention.cpp:387-470`
softmax 확률을 여기서 `exp(logit - bias) * recip(sum)`으로 **즉석 복원**(`attention.cpp:431-434`) 후 V와 곱·누적. 4-way q-patch 병렬 누적 버퍼 `acc_blocks[4]`(`attention.cpp:396-397`). 이 "softmax 통계만 저장 → V 곱 단계에서 재계산" 구조는 **FlashAttention류 메모리 절약 패턴**으로, attention score 행렬(129×129×3)을 풀로 저장하지 않게 한다(단, 본 코드는 `attn`/`attn_softmax_info`를 DRAM tmp에 저장하므로 완전한 fused가 아니라 **2-pass + 통계 압축** 형태로 보인다 — 확인: `ViT_compute.cpp:161,167`이 둘을 별도 호출).

### 3.3 범용 Linear(matmul) 엔진 — `src/linear.cpp` (재사용성 핵심)

Edge-MoE의 모든 FC/MLP/QKV/proj는 이 한 엔진을 공유한다(`#pragma HLS allocation function instances=compute_linear limit=1`, `ViT_compute.cpp:128`). 자원 재사용으로 면적 절감.

**(a) 상위 `compute_linear`** — `linear.cpp:504-530`
3-스테이지 dataflow: `read_in_stream` → `compute_linear_on_stream` → `write_out_stream`. 인자 플래그가 다형성을 만든다:
- `use_gelu` — 출력에 GELU 적용(`linear.cpp:499`),
- `use_expert` — 길이를 `expert_queue_lens[expert]`로(=MoE 모드, `linear.cpp:526`),
- `use_score` — expert 점수 가중 + 누적(MoE l2, `linear.cpp:378-383`).

**(b) MAC 코어 `compute_linear_on_stream`** — `linear.cpp:430-502`
16×16 weight 타일(`weights[block][16][16]`)에 대해 in/out 차원을 16씩 타일링. bias 초기화(`linear.cpp:478-485`) 후 in_dim_offset 16개에 대해 16 out에 동시 MAC(`linear.cpp:487-495`). II=1 파이프라인.

**(c) Direct vs Indirect 게더** — `linear.cpp:387-428`
- **Direct**(`read_in_stream_direct` `linear.cpp:154-197`): 일반 레이어. 모든 patch 순차 처리.
- **Indirect**(`read_in_stream_indirect` `linear.cpp:246-308`): MoE 모드. `expert_queues[expert][patch_idx]`로 **patch를 간접 인덱싱하여 게더**(`linear.cpp:286`). 즉 해당 expert에 라우팅된 patch만 골라 연산. 출력도 `write_out_stream_indirect`가 같은 인덱스로 scatter + score 곱 + 누적(`linear.cpp:310-385`). 이것이 MoE의 sparse 연산을 dense 엔진 위에서 구현하는 방식.

**(d) Weight/bias 로더 `load_linear_weights`/`load_linear_bias`** — `linear.cpp:10-152`
DRAM의 [out][in] 가중치를 256-bit 블록으로 읽어 온칩 16×16 타일 레이아웃으로 재배치(`linear.cpp:44-89`). `ping`/`pong` 두 벌(`linear.cpp:5-8`)로 더블버퍼링 가능. bias는 `wt_bias_t`/`wt_attn_bias_t` 두 타입 템플릿 인스턴스화(`linear.cpp:92-101`).

### 3.4 최상위 오케스트레이션 — `src/ViT_compute.cpp`

**(a) 인터페이스** — `ViT_compute.cpp:68-126`
`extern "C"` HLS 커널. 모든 포트가 `m_axi ... offset=slave`로 외부 메모리 매핑, 제어는 `s_axilite`(`ViT_compute.cpp:100`). AXI 번들을 `inout1~4`/`weights`로 분리(`ViT_compute.cpp:102-126`)해 **다중 HBM/DDR 채널 동시 접근**을 노린다. `tmp1~4`는 레이어 간 중간버퍼(ping-pong 용도로 재사용).

**(b) 레이어 루프 본문** — `ViT_compute.cpp:146-205`
각 레이어마다 표준 ViT 블록을 펼친다:
1. `load_norms` + `compute_norm1` (`:150-151`)
2. QKV: `load_linear_weights`(Q)→`compute_linear`(Q), 같은 식으로 K (`:153-159`)
3. `compute_q_matmul_k` (`:161`)
4. V linear (`:163-165`) → `compute_attn_matmul_v` (`:167`)
5. proj linear (`:169-171`) → residual `compute_add` (`:173`)
6. `compute_norm2` (`:175`)
7. **분기**: `layer%2==0`이면 일반 MLP(fc1 GELU → fc2, `:178-188`), 홀수면 `load_w_gate`+`compute_moe`(`:189-202`)
8. residual `compute_add` (`:203`)

→ 즉 12 레이어 중 6개가 MoE, 6개가 dense MLP. weight 인덱스는 `layer/2`로 MoE/MLP 그룹 분리(`:180,191`).

**(c) `debug_id` 게이트** — 각 서브스텝 후 `if (debug_id == ...) return;`(`:144,152,...`)로 **부분 실행/단계별 검증** 지원. csim에서 중간 텐서를 참조 출력과 비교하기 위한 설계.

**(d) one-time weight 로더 `load_one_time_weights`** — `ViT_compute.cpp:6-66`
patch-embed 가중치/bias를 온칩 상주 버퍼로 1회 로드, `attn_scale=0.125`/`norm_eps=1e-6` 상수 설정(`:64-65`).

### 3.5 보조 커널

- **LayerNorm `compute_norm`** — `layernorm.cpp:138-158`: patch별 2-스테이지 dataflow. `layernorm_accumulate`(mean, mean² 1-pass 누적, `:74-101`) → `layernorm_output`(variance = E[x²]−E[x]²+eps, rstddev = 1/√var, affine, `:103-136`). norm1/norm2 weight는 온칩 상주(`:4-7`).
- **Patch Embed `compute_patch_embed`** — `conv.cpp:143-166`: 16×16 stride-16 conv = 패치 임베딩. y-블록 단위 dataflow(read→compute→output, `:101-141`). pos_embed 더하기(`:138`), cls token(patch 0)은 pos_embed로 초기화(`:160-165`). 이미지는 256-bit(32 픽셀) 블록으로 게더(`conv.cpp:4-5`).
- **GELU `gelu`** — `gelu.cpp:192-211`: `gelu(x) = relu(x) - delta(|x|)`로 분해, delta는 **184-entry LUT 선형보간**(step 0.03125, `gelu.cpp:189-190`). 음/양 대칭성 활용해 |x|만 테이블 조회. ROM(`#pragma HLS bind_storage ... ROM_NP`, `:195`). 벡터 버전은 unroll(`:213-223`).
- **Add `compute_add`** — `add.cpp:3-20`: residual 덧셈. 블록 단위 II=1, `dependence inter false`로 의존성 제거.

---

## 4. 데이터 플로우

### 4.1 추론 전체 흐름 (1 이미지)
```
image[3][128][256] (ap_ufixed8)
   └─ compute_patch_embed (16x16 conv stride16 + pos_embed)
        → x[129][192]   (cls + 128 patches, fm_t)
   for layer in 0..11:
     ├─ norm1
     ├─ Q = linear(norm1);  K = linear(norm1)
     ├─ attn,softmax_info = q_matmul_k(Q,K)      # online-softmax 통계만
     ├─ V = linear(norm1)
     ├─ av = attn_matmul_v(V, attn, softmax_info) # softmax 복원 후 ·V
     ├─ x += linear_proj(av)                       # residual
     ├─ norm2
     ├─ if layer even:  x += fc2(gelu(fc1(norm2))) # dense MLP
     │  else (odd):     x += MoE(norm2)            # top-2 expert
     └─
   → x[129][192] (최종, l11_x_post_moe와 비교)
```

### 4.2 MoE 내부 흐름 (홀수 레이어)
```
gate_inp = norm2(x)
   ├─ compute_gating → expert_queues[16][*], expert_scores[16][*], metaqueue
   for expert in metaqueue:   # 실제 사용된 expert만
     ├─ prefetch l1 weight (ping)           # DRAM→온칩
     ├─ tmp_hidden = gelu(linear_l1(gather(gate_inp by queue)))
     ├─ prefetch l2 weight (pong)
     └─ out[gathered patches] += score * linear_l2(tmp_hidden)
```
- **희소성 활용 지점**: `read_in_stream_indirect`(`linear.cpp:286`)가 patch를 큐 인덱스로 게더 → 라우팅된 patch만 연산. expert 가중치는 metaqueue 순서로 stream-in.

### 4.3 가중치/입출력 파일 흐름 (testbench, `e2e.cpp`)
- 입력: `weights/image.float32.bin`, `patch_embed_weight/bias`, `pos_embed` (`e2e.cpp:59-98`)
- attention: `l{0..11}_qkv_weight/bias`, `l*_attn_proj_weight/bias` (`e2e.cpp:99-154`)
- MoE(홀수 레이어): `l{1,3,..}_w_gate_T_task0`, `l*_htoh4_weight/bias`, `l*_h4toh_weight/bias` (`e2e.cpp:155-219`)
- MLP(짝수 레이어): `l{0,2,..}_fc1/fc2_weight/bias` (`e2e.cpp:220-271`)
- norm: `l*_norm1/2_weight/bias` (`e2e.cpp:272-323`)
- 참조: `l11_x_post_moe.float32.bin`과 비교, MSE/MAE 산출, MSE≤0.1이면 PASS (`e2e.cpp:325-438`)

> 비고: `weights/`에 `task0`/`task1` 두 게이트가 존재. testbench는 task0만 로드(`e2e.cpp:158`). 멀티태스크는 게이트 가중치 교체로 구현하는 것으로 추정.

---

## 5. HW/SW 매핑

| 구성요소 | 구현 | 위치 | HW 자원 매핑(추정) |
|---|---|---|---|
| 최상위 커널 | HLS `extern "C"` ViT_compute | `ViT_compute.cpp:68` | s_axilite 제어 + m_axi 다중 번들 |
| Linear/MLP/QKV/proj/expert | 단일 공유 엔진 | `linear.cpp:504` | `allocation limit=1` → DSP 타일 1벌 시분할(`ViT_compute.cpp:128`) |
| Attention QKᵀ / ·V | dataflow 스트림 | `attention.cpp` | 4-way q-patch 병렬 MAC |
| MoE 게이팅/큐 | 온칩 큐 + metaqueue | `moe.cpp` | array_partition으로 16 expert 병렬 |
| Weight 로드 | ping/pong 더블버퍼 | `linear.cpp:5-8` | BRAM/URAM + AXI prefetch |
| Feature map | 256-bit 블록 패킹 | `hardware.hpp:17` | AXI burst 256-bit |
| 외부 메모리 | tmp1~4/weights 번들 | `ViT_compute.cpp:102-126` | HBM/DDR 다채널(추정) |
| GELU | 184-entry LUT | `gelu.cpp:6` | ROM_NP |
| 비트스트림 | 사전 합성물 | `bitstream/` | 타겟 FPGA 미상(확인 불가, .hwh로 추정 가능하나 미파싱) |

- **SW 부분**: testbench(`e2e.cpp`)가 호스트 측 weight 로딩/검증. 실제 보드 런타임 호스트 코드(PYNQ/XRT)는 이 저장소에 없음(README가 비트스트림+데모만 제공) → **PYNQ 환경 추정**(`bitstream.hwh`는 PYNQ overlay 메타데이터).
- **HW 부분**: 전체 12-layer ViT가 단일 HLS 커널로 합성. 자원 공유(`allocation limit=1`)로 single-engine, 시분할 스케줄.

---

## 6. 빌드·실행

- **빌드(csim)**: `Makefile`이 `src/*.cpp`를 `src/_single_file.cpp`로 concat(`Makefile:19-23`) 후 컴파일하여 `result` 생성(`Makefile:25-26`). Vitis HLS 2021.1 include/lib에 링크(`Makefile:1-10`), C++14(`Makefile:8`).
- **의존**: `AUTOPILOT_ROOT=/tools/.../Vitis_HLS/2021.1`(`Makefile:1`) 하드코딩. ap_fixed/hls_vector/hls_math/hls_stream/gmp 필요.
- **실행**: `./result`를 저장소 루트에서 실행(상대경로 `weights/*.bin`을 읽음, `e2e.cpp:60` 등). 단일 이미지 추론 후 MSE/MAE 출력, MSE≤0.1이면 exit 0(`e2e.cpp:438`).
- **합성/구현**: `.gitignore`에 `vitis_hls_project/`, `vivado_project/`가 있어 별도 Vitis HLS/Vivado 프로젝트로 합성하는 흐름(스크립트는 저장소에 미포함 → 합성 tcl "확인 불가").
- **보드 실행**: 사전 합성 `bitstream/*.xclbin`/`.bit`/`.hwh` 제공. 호스트 코드 미포함(PYNQ 추정).

---

## 7. 의존성

- **툴**: Xilinx Vitis HLS 2021.1 (`Makefile:1`).
- **라이브러리**: `ap_fixed.h`, `hls_vector.h`, `hls_math.h`, `hls_stream.h` (Vitis HLS), `gmp.h`(HLS 버그 우회용, `kernel.hpp:5-6`).
- **표준**: C++14, `<fstream>/<sstream>`(testbench만, `e2e.cpp:1-7`).
- **외부 ML 프레임워크 의존 없음**: 학습은 별도(가중치는 float32 bin으로 export). 양자화는 로드 시 float→ap_fixed 캐스팅(`tbutil.hpp:10-20`).
- **자체 모듈 의존 그래프**: `dcl.hpp`(util+datatypes+model+hardware) → 모든 커널이 include. `linear.hpp`가 attention/moe/conv의 공통 엔진. 순환 없음.

---

## 8. 강점·한계

### 강점
- **메모리 효율 MoE**: 선택된 expert만 ping/pong prefetch로 stream-in(`moe.cpp:304-436`). 16 expert 전체 온칩 상주 불필요 → 논문 제목의 "Memory-Efficient" 실현.
- **단일 공유 linear 엔진**: QKV/proj/MLP/expert가 한 엔진 시분할(`ViT_compute.cpp:128`) → DSP 절감, 코드 재사용 극대화. 플래그 다형성(direct/indirect/gelu/score)으로 모든 케이스 처리.
- **online-softmax 일관 적용**: gating(`moe.cpp:197`)·attention(`attention.cpp:127`) 동일 패턴 → 수치 안정 + 메모리 절약. attention은 확률 행렬 미저장, 통계만 흘리고 ·V에서 복원.
- **HLS 친화 DSL**: `FOR_BLOCK/FOR_OFFSET` 매크로(`util.hpp`)로 모든 루프 II=1 파이프라인을 일관되게 작성 → 가독성+성능 양립.
- **블록 패킹**: 256-bit AXI 정렬 + SIMD-8 벡터를 자료형에 내장(`hardware.hpp`) → 대역폭·병렬도 동시 확보.
- **단계별 검증**: `debug_id` 게이트로 중간 텐서 비교 가능(`ViT_compute.cpp`) → 디버깅/검증 용이.

### 한계
- **모델 형상 하드코딩**: model.hpp의 constexpr로 모든 크기 고정(192/768/384/16 expert/top-2 등). 다른 ViT로 재타깃 시 재합성 필수.
- **단일 엔진 시분할**: `allocation limit=1`은 면적 절감이지만 처리량 상한이 낮음 → **고처리량/저지연 동시 만족은 어려움**(파이프라인 병렬 없음, 레이어 순차).
- **activation 32-bit fixed**: weight는 16-bit이나 fm_t가 32-bit(`datatypes.hpp:8`) → BRAM/대역폭 부담. 더 공격적 양자화 여지.
- **호스트/합성 스크립트 부재**: csim testbench만 있고 보드 런타임 코드·합성 tcl 미포함 → 재현성 일부 "확인 불가".
- **attention이 완전 fused 아님**: `attn`/`attn_softmax_info`를 DRAM tmp로 왕복(`ViT_compute.cpp:161,167`) → FlashAttention 대비 외부 메모리 트래픽 잔존.
- **단일 배치/이미지 위주**: testbench는 num_images=1(`e2e.cpp:26`). 배치 파이프라이닝 미증명.

---

## 9. 우리 프로젝트(PRJXR-HBTXR) 시사점

우리 프로젝트 추정: **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적**.

1. **단일 엔진 vs 파이프라인의 정반대 트레이드오프**: Edge-MoE는 `allocation limit=1`로 면적 최소화·시분할(저처리량)을 택했다. HG-PIPE 계열(레이어별 전용 PE를 펼쳐 파이프라인하는 high-throughput 지향)과는 **설계 철학이 반대**다. 우리는 Edge-MoE의 "엔진 재사용"이 아니라 "레이어 펼침"을 따르되, Edge-MoE의 **블록 패킹/online-softmax/GELU LUT**는 그대로 차용 가능. (근거: `ViT_compute.cpp:128` vs HG-PIPE의 dataflow 펼침)
2. **online-softmax 재계산 패턴(`attention.cpp:127-213, 387-470`)**: attention 확률 행렬을 저장하지 않고 통계만 흘린 뒤 ·V에서 복원 → XR 시선추적처럼 **저지연·저메모리가 중요한 엣지**에서 매우 유용. 우리 attention 블록의 기준 구현으로 채택 검토.
3. **GELU LUT 보간(`gelu.cpp:6-211`)**: `relu(x) - delta(|x|)` 분해 + 184-entry LUT는 DSP 없이 비선형 근사. 우리 가속기의 활성함수 블록에 바로 이식 가능(시선추적 모델의 MLP에도 적용).
4. **혼합정밀(weight16/act32, `datatypes.hpp`)**: 우리는 더 공격적 양자화(예: act INT8/12)로 차별화 여지. Edge-MoE는 정확도 안전을 위해 act 32-bit를 둠 → 우리 양자화 전략의 baseline 비교군으로 활용.
5. **MoE 게더/스캐터 indirect 엔진(`linear.cpp:246-385`)**: 만약 우리 XR 파이프라인에 태스크/조건부 분기(예: 시선 영역별 전문가)가 들어간다면, dense 엔진 위에 큐 인덱싱으로 sparse를 얹는 이 패턴이 참고가 된다. 단, 우리가 high-throughput이라면 큐 기반 동적 분기는 파이프라인 stall 위험 → 정적 라우팅 검토 필요.
6. **AXI 번들 분리(`ViT_compute.cpp:102-126`)**: 다채널 HBM 활용 패턴은 우리도 동일하게 적용(weight 채널 / feature 채널 분리)하면 대역폭 병목 완화.
7. **`debug_id` 단계별 검증(`ViT_compute.cpp`)**: 레이어/서브스텝별 중간 텐서를 참조와 비교하는 메커니즘은 우리 검증 인프라(특히 quantization 정확도 추적)에 그대로 도입할 가치가 높다.

---

## 부록: 근거 표기 요약

- **확인(라인 근거 명시)**: 모델 형상(`model.hpp`), 자료형/양자화(`datatypes.hpp`), 블록/병렬도(`hardware.hpp`), MoE 전 과정(`moe.cpp`), attention 2-pass + online softmax(`attention.cpp`), 공유 linear 엔진/indirect 게더(`linear.cpp`), 레이어 오케스트레이션·even/odd 분기(`ViT_compute.cpp`), LayerNorm/PatchEmbed/GELU/Add, testbench 파일 입출력·검증(`e2e.cpp`), 빌드(`Makefile`), 제외 항목(`.gitignore`).
- **추정**: 타깃 다운스트림 태스크(세그/깊이 등), PYNQ overlay 런타임, 멀티태스크 = 게이트 가중치 교체, HBM 다채널 매핑, RTL은 HLS 합성 산출물.
- **확인 불가**: 합성/구현 tcl 스크립트(미포함), 보드 호스트 코드(미포함), 타깃 FPGA 디바이스(.hwh 미파싱), 실제 정확도/처리량 수치(논문 본문 필요).
