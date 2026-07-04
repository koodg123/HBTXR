# ViT-Accelerator 정밀 분석

> 분석 대상: `REF/Transformer-Accel/ViT-Accelerator`
> 작성일: 2026-06-20 / 방식: 실제 소스 Read 후 파일:라인 근거 기반
> 제약: `host/ggml/`(외부 ggml 라이브러리)는 분석 제외(이름만 언급), `.git`/`*.so`/이미지(.jpg/.gif) 제외

---

## 1. 개요

- **무엇인가**: Vision Transformer(ViT)의 **Self-Attention 단계**(Q×Kᵀ → scaling → softmax)를 Xilinx FPGA에 **HLS(High-Level Synthesis)**로 매핑한 학생 과제(Final Project)형 가속기 데모.
- **한 줄 요약**: ViT-B의 단일 head 어텐션 스코어(197×64 · 64×197 → 197×197)를 **systolic-array 스타일 타일드 matmul + 다항식 근사 softmax**로 합성하는 HLS 커널과, 이를 호출할 호스트/래퍼 스켈레톤의 모음. **호스트 추론 본체(vit.cpp/main.cpp)는 미구현**.
- **출처·과제**: README(line 63) — C++ 호스트 코드는 `github.com/staghado/vit.cpp`에서 가져온 것(원저작자 크레딧 명시). HLS 커널과 study journal은 본 프로젝트 팀(Hao Chen, Sheng-Wei Huang, Hua-Shao Chu) 작성(README §6~§8, line 133~219).
- **타깃 보드·디바이스**:
  - HLS 합성 파트: **xcu50-fsvh2104-2-e**(Alveo U50 디바이스 파트), 클럭 **10 ns(100 MHz)** — `hls_source/scripts/run_hls.tcl:6,9`.
  - v++ 빌드 플랫폼: **xilinx_u50_gen3x16_xdma_201920_3**(Alveo U50) — README:30,35,41.
  - 런타임: **Xilinx XRT** + **OpenCL** — `host/CMakeLists.txt:37,45`.
  - 툴체인: **Vitis 2022.1** — README:15.

> 정리: 자체 작성 분량은 작지만(HLS 3파일 + 래퍼 2파일 + quantize.cpp + vit.h), "ViT 어텐션을 systolic array로 HLS 합성"이라는 핵심 패턴이 완결적으로 들어 있다. 다만 SW 통합(호스트, 멀티헤드, V 곱) 다수가 **TODO/미완** 상태다.

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (분석 대상)

```
ViT-Accelerator/
├─ README.md                         # 개요 + 사용법 + 3인 study journal
├─ host/                             # 호스트(CPU) 측
│  ├─ vit.h                          # ViT 추론 자료구조/함수 선언 (vit.cpp는 부재)
│  ├─ quantize.cpp                   # ggml 기반 가중치 양자화 도구 (단독 실행 main 포함)
│  └─ CMakeLists.txt                 # vit / quantize / benchmark 빌드 + XRT 링크
├─ hls_source/                       # FPGA HLS 커널
│  ├─ kernel.h                       # mmult, attention_kernel 선언 + 차원 매크로
│  ├─ kernel.cpp                     # attention_kernel(top) + exp_approx + softmax
│  ├─ q_matmul_k_function.cpp        # mmult: 16×16 타일드 systolic array matmul
│  ├─ testbench.cpp                  # C-sim 검증용 reference_attention + main
│  └─ scripts/
│     ├─ run_hls.tcl                 # 프로젝트/파트/클럭/csim·csynth·cosim·export
│     └─ synth_config.tcl            # pipeline/array_partition directive
└─ wrapper/                          # FPGA 호출 래퍼
   ├─ opencl_wrapper.cpp             # 순수 OpenCL(cl.hpp) 호출 래퍼
   └─ attention_wrapper.cpp          # pybind11 파이썬 바인딩 래퍼
```

### 2.2 제외 대상 (이름만 언급)

- `host/ggml/...` — **외부 ggml 라이브러리 전체**(GGML core, GGUF, whisper/gpt-2/sam/mnist 등 예제 다수). ViT 추론은 이 ggml 텐서 연산에 의존하나 본 분석 범위 밖.
- 빌드 산출물 `*.so`: `host/attention_wrapper.so`, `wrapper/attention_wrapper.so` — pybind11 컴파일 결과물(분석 제외).
- 이미지: `host/cat-resized.jpg`(추론 데모 입력), `image/ViT.gif`(README 삽화) — 제외.
- `Final Project Presentation.pdf` — Glob 결과에 노출되지 않음(현 트리에 미존재로 추정). **PDF 직접 확인 불가**.

> **중요한 부재 파일(확실, 코드 근거)**: `host/CMakeLists.txt:52-53`은 `main.cpp`, `vit.cpp`를 빌드 소스로 지정하나 Glob(`host/*.cpp`)에는 `quantize.cpp`만 존재 → **vit.cpp / main.cpp 부재**. README:27도 "The actual host code has not done yet." 명시. 또한 `wrapper/attention_wrapper.cpp:3`이 include하는 `../hls_source/attention_kernel.h` 역시 부재(Glob `hls_source/attention_kernel.h` → No files found).

---

## 3. 핵심 모듈 정밀 분석 (★)

### 3.1 차원 규약 (`hls_source/kernel.h`)

```
REAL_Q_ROW 197   REAL_Q_COL 64   REAL_K_ROW 64   REAL_K_COL 197   (kernel.h:4-7)
```

- 197 = 패치 토큰 196개(224/16=14 → 14×14, 단 hparams patch_size=8과는 별개로 README/저널은 16px patch·196 patch 가정) + CLS 토큰 1개.
- 64 = 단일 head 차원(d_head). ViT-B는 hidden 768 / 12 head → head당 64.
- 따라서 본 커널은 **단일 head 1개분의 Q·Kᵀ**만 처리(멀티헤드 12개는 미구현, README §7.1 line 184 명시).

선언(`kernel.h:9-15`):
```cpp
extern "C" void mmult(float A[197][64], float B[64][197], float C[197][197]);
extern "C" void attention_kernel(volatile float* q, volatile float* k, volatile float* attention_score);
```
- 주목: 선언에 **V(value)도, 멀티헤드 인자도 없다.** 출력은 attention **score**(softmax 결과)까지이며 `softmax·V`는 커널 밖.

---

### 3.2 `attention_kernel` — Top-level HLS 커널 (`hls_source/kernel.cpp:20-115`)

이 커널이 합성 top(`run_hls.tcl:11 set_top attention_kernel`). 동작은 5단계 파이프라인.

#### (1) AXI/제어 인터페이스 (kernel.cpp:26-33)
```cpp
#pragma HLS INTERFACE m_axi port=q              offset=slave bundle=gmem0 depth=12608
#pragma HLS INTERFACE m_axi port=k              offset=slave bundle=gmem1 depth=12608
#pragma HLS INTERFACE m_axi port=attention_score offset=slave bundle=gmem2 depth=38809
#pragma HLS INTERFACE s_axilite port=q/k/attention_score/return bundle=control
```
- 입력 q,k를 별도 AXI 마스터 번들(gmem0/gmem1)로, 출력은 gmem2로 분리 → 동시 메모리 접근 대역폭 확보.
- depth 정량: q/k = **12608 = 197×64**(요소 수), 출력 = **38809 = 197×197**. (kernel.cpp:26-28, 정확히 일치)
- 제어는 `s_axilite ... bundle=control` → XRT/OpenCL이 인자 세팅·기동.

#### (2) 온칩 버퍼 + BRAM 바인딩 (kernel.cpp:35-43)
```cpp
static float localQ[197][64];          #pragma HLS BIND_STORAGE variable=localQ type=RAM_2P impl=BRAM latency=2
static float localK[64][197];          #pragma HLS BIND_STORAGE ... localK ... BRAM latency=2
static float localAttention[197][197]; #pragma HLS BIND_STORAGE ... BRAM latency=2
static float scores[197][197];         // (바인딩 미지정)
static float softmax_scores[197][197]; // (바인딩 미지정)
```
- 3개 주요 버퍼를 **RAM_2P(2-port) BRAM, latency=2**로 명시 바인딩. `localAttention`은 선언되나 실제로는 미사용(데드)이며 실제 계산은 `scores`/`softmax_scores` 사용.

#### (3) 입력 로드 (kernel.cpp:46-65) — `readQ`, `readK`
```cpp
readQ: for(loc=0,i=0,j=0; loc<197*64; loc++,j++){
  #pragma HLS PIPELINE II=1
  if(j==64){i++;j=0;} localQ[i][j]=q[loc]; }
readK: 동일 구조, 64*197, j==197 시 행 증가.
```
- 1D AXI 스트림을 2D 버퍼로 언플래튼. **II=1**로 매 사이클 1요소 적재.

#### (4) score 0초기화 + matmul + scaling (kernel.cpp:67-84)
```cpp
init_scores: scores[i][j]=0.0f;          (PIPELINE II=1)              // 67-73
mmult(localQ, localK, scores);           // 75  ← 서브함수 호출(systolic)
scailing: scores[i][j] = scores[i][j] / 8.0f; (SCAILING_FACTOR=8, PIPELINE II=1) // 78-84
```
- 주의: **스케일 인자가 SCAILING_FACTOR=8**(kernel.cpp:5)로 하드코딩. 이론상 √d=√64=8과 일치 → 본 head dim 64에 한해 정확. 그러나 매크로/상수로만 고정되어 다른 head dim에선 부정확(한계).

#### (5) Softmax 근사 (kernel.cpp:87-102) — `softmax`
행 단위로:
```cpp
float sum[8]={0};
for(j<197){ #pragma HLS PIPELINE II=1   sum[j%8] += exp_approx(scores[i][j]); }  // 90-93  부분합 8뱅크
for(j=1;j<8;j++){ #pragma HLS PIPELINE II=7  sum[0]+=sum[j]; }                    // 94-97  뱅크 환원
for(j<197){ #pragma HLS PIPELINE II=1   softmax_scores[i][j]=scores[i][j]/sum[0]; } // 98-101
```
- **핵심 최적화**: 누산 의존성(load→store) II 위반을 피하려고 `sum[j%8]` 8-뱅크 부분합으로 분산 후 환원(README §8.1 line 203~211 설명과 일치). 환원 루프는 II=7.
- **버그성 주의 1**: 정규화 분자가 `exp_approx(scores)`가 아니라 **raw `scores[i][j]`** → 즉 `softmax = score / Σexp(score)`. 표준 softmax(`exp(score)/Σexp(score)`)가 아님. testbench의 reference(softmax_scores[i][j]=e_val 후 /sum_exp, testbench.cpp:65,69)와도 **불일치** → cosim에서 max diff가 커질 소지(테스트 임계 2e-2, testbench.cpp:114).
- **버그성 주의 2**: max-subtraction(수치안정화) 없음 + 다항식 exp 근사(`exp_approx`)는 음수·큰 입력에서 음수/발산 가능.

#### (6) `exp_approx` (kernel.cpp:11-14)
```cpp
float exp_approx(float x){ return 1 + x + x*x/2 + x*x*x/6; }  // e^x 3차 테일러
```
- 하드웨어 친화 다항식 근사(곱·합만). hls::exp() 대안은 README §8.1(line 196~201)에서 논의(INTERNAL-INFO fexp 경고).

#### (7) 출력 라이트백 (kernel.cpp:104-112) — `writeAttention`
```cpp
for(loc<197*197){ #pragma HLS PIPELINE II=1  attention_score[loc]=softmax_scores[i][j]; }
```
- 2D → 1D AXI 버스트, II=1.

> **정량 요약(attention_kernel)**: 입력 II=1 로드 12608+12608 사이클, score init 38809, scaling 38809, softmax ≈197×(197+7+197), 라이트백 38809 + mmult 비용(아래). 클럭 100 MHz 기준 단일-head 1회 수십만~백만 사이클대. 멀티헤드/배치 병렬화는 없음.

---

### 3.3 `mmult` — 16×16 타일드 Systolic Array Matmul (`hls_source/q_matmul_k_function.cpp:18-175`)

attention_kernel이 호출하는 Q×Kᵀ 본체. **본 repo에서 HW적으로 가장 정교한 부분.**

#### 차원·패딩·타일 (q_matmul:6-16)
```
REAL_A_ROW197 A_COL64 B_ROW64 B_COL197
PAD_A_ROW 208  (197 + (16-5) = 208, 16배수 정렬)   // 12,13
PAD_B_COL 208
M 16  (타일 크기)
```
- 197을 16배수(208)로 올림 → 13×13 타일(`PAD/M=13`, line 35-36 주석 "0..12").

#### 온칩 버퍼 + 파티션 (q_matmul:22-31)
```cpp
float subA[16][64];  #pragma HLS ARRAY_PARTITION variable=subA dim=2 complete  // A_COL 완전분할
float subB[64][16];  #pragma HLS ARRAY_PARTITION variable=subB dim=1 complete  // B_ROW 완전분할
float inC[16][16];   #pragma HLS ARRAY_PARTITION variable=inC dim=0 complete   // 전체 256 레지스터화
```
- inC를 dim=0 complete로 완전 분할 → 16×16 PE 누산을 레지스터로 병렬화.

#### 타일 이중 루프 (q_matmul:34-167)
- `tile_outer_loop`: tileRow 0..12 × tileCol 0..12 = 169 타일.
- 각 타일 내부:
  1. **init_inC**(42-47, PIPELINE): inC=0.
  2. **read_subA**(54-66, PIPELINE): A의 16행×64열 블록 적재, `globalRow≥197`이면 0 패딩.
  3. **read_subB**(68-80, PIPELINE): B의 64행×16열 블록 적재, `globalCol≥197`이면 0 패딩.
  4. **systolic_tiling**(87-145): 핵심.
     - `inA[16][16]`, `inB[16][16]` 시프트 레지스터(둘 다 dim=0 complete, line 90,92).
     - 셔플 길이 `r < (64 + 2*16 - 2) = 94`(line 104, 주석 "94"). 각 r에서 PIPELINE(line 105):
       - inA 오른쪽 시프트(108-111), inB 아래쪽 시프트(114-118),
       - inA 좌열·inB 상행에 `subA[i][r-i]`/`subB[r-j][j]` 신규 주입(대각 스큐, 122-137),
       - PE 누산 `inC[i][j] += inA[i][j]*inB[i][j]`(140-144) — 256개 MAC 병렬.
  5. **store_tileC**(152-164, PIPELINE II=1): 유효 197×197 범위만 `C[gr][gc] += inC[i][j]`.
- **정량**: 169 타일 × (94-스텝 systolic + read/init/store) → 단순 추정 169×~94+오버헤드 ≈ 2만 사이클대(파이프라인 효율 가정). 단, README §7.2(line 188~190)는 "고정 크기만 처리, sub-buffer 직결 미적용" 한계 명시.

> **주의(정확성)**: 본 systolic 구현은 표준 출력-스테이셔너리 systolic의 `inC[i][j] += inA[i][col]*inB[row][j]`가 아니라 시프트 후 **요소별 inA[i][j]*inB[i][j]** 누산 형태(line 142). 대각 스큐 주입과 시프트로 시간축에서 부분곱을 정렬하는 의도지만, 표준 systolic 정합성은 검증 필요(testbench cosim에 의존). C는 `+=`로 누적(line 161)되며, 호출 전 attention_kernel이 scores를 0초기화(kernel.cpp:67-73)하므로 외부 초기화 의존.

---

### 3.4 `testbench.cpp` — C-sim/Co-sim 검증 (`hls_source/testbench.cpp`)

- `exp_approx_ref`(15-19): 커널과 동일 3차 테일러.
- `reference_attention`(22-79): 순수 SW로 Q×K(36-45) → /8 스케일(48-52) → softmax. 단 reference는 **표준 softmax**(`softmax_scores=e_val`(65) 후 `/sum_exp`(69)) → **3.2(5)에서 지적한 커널의 raw-score 분자와 불일치**.
- `main`(81-134): Q,K 난수(0~9.9, 90-95) → reference 계산 → `attention_kernel(q,k,attention_score)`(103) → 절대오차 비교, **임계 2e-2**(114). 불일치 누적 시 Fail.
- **시사**: 커널 softmax 분자 불일치(score vs exp(score)) 때문에 이 testbench는 실제로는 통과하기 어려울 가능성(추정). 적어도 의도(표준 softmax)와 구현이 다름.

---

### 3.5 `quantize.cpp` — ggml 기반 가중치 양자화 (`host/quantize.cpp`)

ViT 추론 본체와 별개의 **독립 실행 도구**(자체 main 포함, 358-409).

- `vit_hparams`(17-30): ViT-B 기본값 — hidden 768, layers 12, heads 12, classes 1000, **patch_size 8**, img 224, eps 1e-6, interpolation bicubic.
- `vit_model_quantize`(33-352):
  - itype→ggml_type 매핑(37-57): 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0.
  - GGUF magic 검증(76-86), hparams 직렬화 R/W(91-116), id2label 맵 R/W(121-150).
  - 텐서 루프(167-327): **이름이 `.*weight`이고 2D인 텐서만 양자화**(207-222). f16이면 fp32 승격 후(226-235), `ggml_quantize_q4_0/q4_1/q5_0/q5_1/q8_0`(272-302) 호출, 히스토그램 누적·크기 리포트(307-345).
- `main`(358-409): `usage: quantize in.gguf out.gguf type`, ggml f16 테이블 초기화(372-376), 타이밍 리포트.
- **연계**: 양자화 출력 gguf를 host 추론(`./bin/vit -m ...gguf`)이 로드(README:57). 단 추론 본체 미구현이라 end-to-end 미완.

---

### 3.6 `vit.h` — ViT 추론 자료구조/인터페이스 (`host/vit.h`)

`vit.cpp`가 부재하므로 **선언만** 존재(구현 분석 불가).

- include(2-6): ggml, ggml-alloc, **stb_image**(이미지 로드), 그리고 **XRT** 헤더 `experimental/xrt_kernel.h`, `ert.h` → 호스트가 XRT C++ API로 FPGA 호출하도록 설계됨.
- 자료구조:
  - `vit_block`(40-54): LayerNorm1/2, **qkv_w/qkv_b**(합쳐진 QKV proj), attn proj, MLP lin1/lin2 → 표준 ViT 인코더 블록.
  - `vit_image_encoder`(64-71): pos-embed `pe`, `cls_token`, patch proj `proj_w/b`, `layers`(블록 벡터).
  - `classifier_head`(56-62): norm + head(분류기).
  - `vit_state`(73-85): **`xrt::kernel krnl_attention;`**(79) — FPGA 어텐션 커널 핸들을 상태에 보관 → 추론 중 호출 의도(확실: 선언).
  - `vit_params`(110-118): seed, n_threads(≤4), topk=5, model/이미지 경로, eps.
- 함수 선언(120-129): `load_image_from_file`, `vit_image_preprocess`(전처리), `vit_model_load`, **`vit_encode_image`**(인코더 그래프 빌드), **`vit_predict`**(top-k 예측), arg 파서.
- **시사**: 설계 의도는 "ggml로 ViT forward 그래프 구성 + 어텐션만 XRT로 FPGA offload"지만, `vit_encode_image`/`vit_predict` **구현(vit.cpp) 부재**로 실제 offload 결선은 미완(확실).

---

### 3.7 Wrapper — FPGA 호출 래퍼 2종 (`wrapper/`)

#### (a) `opencl_wrapper.cpp` — 순수 OpenCL
```cpp
run_attention_kernel(float* Q,K,V,Output, size_t size)              // 6
  xclbin 바이너리 로드(8-10)
  cl::Context(CL_DEVICE_TYPE_ACCELERATOR), Queue, Program           // 13-15
  cl::Kernel(program,"attention_kernel")                            // 18
  q/k/v_buf(READ_ONLY|COPY_HOST_PTR), out_buf(WRITE_ONLY)           // 21-24
  setArg(0..3) → enqueueTask → enqueueReadBuffer(blocking)          // 27-34
```
- **불일치(확실)**: 래퍼는 인자 4개(**Q,K,V,Output**)로 설정하지만 실제 `attention_kernel`(kernel.cpp:21-24)은 **3인자(q,k,attention_score), V 없음**. → 이 래퍼는 실제 합성 커널과 인터페이스 불일치(미정합/구버전 추정).

#### (b) `attention_wrapper.cpp` — pybind11 파이썬 바인딩
```cpp
#include "../hls_source/attention_kernel.h"                        // 3  (파일 부재)
run_attention_kernel(py::array_t<float> Q,K,V,Output)              // 7-10
  num_tokens=Q.shape(0); d_head=Q.shape(1);                        // 19-20
  attention_kernel(q,k,v,out, num_tokens,d_head);                  // 22  (6인자!)
PYBIND11_MODULE(attention_wrapper, m){ m.def("run_attention_kernel",...) } // 25-27
```
- **파이썬 바인딩 존재 확인(확실)**: pybind11로 `attention_wrapper` 모듈 노출 → 빌드 산출물이 `attention_wrapper.so`(host/, wrapper/).
- **불일치(확실)**: 호출 시그니처 `attention_kernel(q,k,v,out,num_tokens,d_head)`(6인자, V·동적크기 포함)는 실제 커널(3인자, V없음, 고정크기)과 **완전 불일치**. include하는 `attention_kernel.h`도 부재. → 파이썬 경로는 **설계 스케치/미완**(추정).

> **정리(wrapper)**: 두 래퍼 모두 "Q,K,V,Output 일반 어텐션"을 가정해 만들어졌으나, 실제 합성된 커널은 "Q,K → score(softmax)까지"만 처리. **래퍼-커널 인터페이스가 서로 다른 세대**로 보이며 현 상태로는 직접 연결 불가(미완).

---

## 4. 데이터플로우 (의도 설계 vs 실제 구현)

```
[이미지(cat-resized.jpg)]
   │  load_image_from_file / vit_image_preprocess (vit.h:123-124, 구현부재)
   ▼
[224×224×3 정규화 텐서]
   │  patch_embed(proj_w) + cls_token + pos-embed (vit_image_encoder, 구현부재)
   ▼
[197 토큰 × 768]  (197 = 196 patch + 1 CLS)
   │  per-layer: LN1 → QKV proj → (head별 Q,K,V) ──────────────┐
   │                                                            │ (어텐션만 offload 의도)
   │   ┌──────────────── FPGA (HLS) ────────────────┐          │
   │   │ attention_kernel(q,k,score) [kernel.cpp]    │          │
   │   │   readQ/readK → mmult(Q×Kᵀ)[q_matmul] →     │          │
   │   │   /8 scaling → softmax근사 → score(197×197) │          │
   │   └─────────────────────────────────────────────┘          │
   │   (단일 head만; ×V·멀티헤드 12·proj는 호스트/미구현) ◄──────┘
   ▼
[attention score 197×197]  →  (×V, proj, +residual, LN2, MLP)  ← 호스트(ggml) 의도, 미구현
   ▼
[CLS 토큰 → classifier_head] → top-k 예측 (vit_predict, 구현부재)
```

- **실제로 합성·검증되는 구간**: `readQ/K → mmult → scaling → softmax근사 → writeAttention`(kernel.cpp, q_matmul, testbench)만. 그 앞(patch/embed)과 뒤(×V/MLP/classifier)는 **미구현**(vit.cpp 부재).
- 호스트↔FPGA 데이터: AXI m_axi(gmem0/1/2)로 q,k in / score out(kernel.cpp:26-28). OpenCL은 `enqueueTask`+`enqueueReadBuffer`(opencl_wrapper:33-34), XRT 경로는 `xrt::kernel`(vit.h:79).

---

## 5. HW/SW 매핑

| 계층 | 구현체 | 파일:라인 | 상태 |
|---|---|---|---|
| 이미지 전처리/패치/임베드 | ggml 텐서 그래프(예정) | vit.h:123-126 (선언만) | **미구현** |
| QKV projection / 멀티헤드 분리 | 호스트 ggml(예정) | vit.h:43-44 qkv_w/b | **미구현** |
| **Q×Kᵀ matmul (단일 head)** | HLS `mmult` (16×16 타일 systolic) | q_matmul:18-175 | 합성 대상(검증 의도) |
| **scaling /√d** | HLS scailing 루프(/8 고정) | kernel.cpp:78-84 | 합성됨 |
| **softmax 근사** | HLS softmax + exp_approx | kernel.cpp:11-14,87-102 | 합성됨(분자 불일치 의심) |
| score × V, attn proj, residual, LN2, MLP | 호스트 ggml(예정) | vit.h:45-53 | **미구현** |
| classifier / top-k | 호스트 `vit_predict` | vit.h:127 (선언만) | **미구현** |
| 가중치 양자화 | `quantize.cpp`(ggml) | host/quantize.cpp 전체 | 구현됨(독립 도구) |
| FPGA 호출 (C++/XRT) | `xrt::kernel krnl_attention` | vit.h:79 | 핸들만, 호출부 부재 |
| FPGA 호출 (OpenCL) | `run_attention_kernel` | opencl_wrapper.cpp:6-35 | 인터페이스 불일치 |
| FPGA 호출 (Python) | pybind11 모듈 | attention_wrapper.cpp:7-27 | 인터페이스 불일치/헤더 부재 |

- **핵심 대응**: 호스트 C++(ggml forward) ↔ HLS 커널(attention만) ↔ wrapper(OpenCL/pybind11) ↔ 외부 ggml. 다만 세 래퍼/커널/호스트가 **서로 다른 인터페이스 가정**으로 작성되어 정합성 미확보(아래 한계).

---

## 6. 빌드·실행

### 6.1 HLS 합성 (`hls_source/scripts/run_hls.tcl`)
```tcl
open_project hls_attention
add_files kernel.cpp; add_files q_matmul_k_function.cpp; add_files testbench.cpp -tb   // 2-4
set_part {xcu50-fsvh2104-2-e}          // 6  (Alveo U50)
create_clock -period 10 -name default  // 9  (100 MHz)
set_top attention_kernel               // 11
source ./scripts/synth_config.tcl      // 14
csim_design; csynth_design; cosim_design; export_design -rtl verilog -format ip_catalog  // 16-19
```
실행: `cd hls_source && vitis_hls -f scripts/run_hls.tcl`(README:23-25).

### 6.2 합성 directive (`hls_source/scripts/synth_config.tcl`)
```tcl
config_compile -pipeline_loops 1                 // 1
config_rtl -reset all                            // 2
set_directive_pipeline -II 1 "attention_kernel"  // 3
set_directive_array_partition -type cyclic -factor 4 -dim 2 "attention_kernel" Q   // 4
... K / V / Output 동일 (cyclic factor 4, dim 2)  // 5-7
```
- **불일치(주의)**: directive가 참조하는 변수명 **Q/K/V/Output**은 실제 커널 인자명(`q`/`k`/`attention_score`)·로컬명(localQ/localK)과 다르며 **V/Output은 존재하지 않음** → 이 array_partition directive들은 실제 커널에 적용 안 됨(무효 추정). 즉 합성 시 의도한 cyclic 분할이 미반영될 소지.

### 6.3 v++ 커널 빌드 + 링크 (README:29-46)
```
v++ -c -t hw --platform xilinx_u50_gen3x16_xdma_201920_3 -k attention_kernel kernel.cpp -o attention_kernel.xo
v++ -c -t hw --platform ... -k mmult q_matmul_k_function.cpp -o mmult.xo
v++ -l -t hw --platform ... attention_kernel.xo mmult.xo -o vit_accel.xclbin
```

### 6.4 호스트 CMake (`host/CMakeLists.txt`)
- `-O3 -march=native`(17-18), ggml `add_subdirectory`(26).
- `option(USE_FPGA ... OFF)`(31) — ON이어도 코드상으론 항상 XRT 경로 링크(37-46, 조건 분기 없이 include/link). XRT include `/opt/xilinx/xrt/include`, lib `xrt_coreutil OpenCL pthread`(45).
- 타깃: `vit`(main.cpp+vit.cpp, **소스 부재**), `quantize`(quantize.cpp, ON), `benchmark`(OFF).
- 실행 예: `./bin/vit -t 4 -m ../ggml-model-f16.gguf -i ../cat-resized.jpg`(README:57).

---

## 7. 의존성

- **Vitis / Vitis HLS 2022.1**(README:15) — csim/csynth/cosim/export, 파트 xcu50.
- **OpenCL** — `wrapper/opencl_wrapper.cpp:1 <CL/cl.hpp>`, CMake 링크(45).
- **XRT** — `host/vit.h:5-6`(experimental/xrt_kernel.h, ert.h), CMake `/opt/xilinx/xrt`(37-42).
- **ggml (외부)** — 추론/양자화 텐서 백엔드. `host/ggml/`(분석 제외), quantize.cpp·vit.h가 의존.
- **pybind11 + numpy** — `wrapper/attention_wrapper.cpp:1-2`(파이썬 바인딩).
- **stb_image** — 이미지 로드(vit.h:4).
- Python 3.11 + `host/requirements.txt`(README:13-14) — 모델 변환/실행 보조(추정).

---

## 8. 강점·한계

### 강점
1. **어텐션의 HW 매핑 패턴이 명확**: Q×Kᵀ를 16×16 타일 + 208 제로패딩 + 시프트레지스터 systolic으로 구현(q_matmul:34-145). HLS 입문/교육용으로 가독성 높음.
2. **AXI/제어 인터페이스 정량 정확**: m_axi depth(12608/38809)가 차원과 정확 일치, 멀티 번들로 대역폭 분리(kernel.cpp:26-33).
3. **softmax II 위반 회피 기법**: 8-뱅크 부분합으로 누산 의존성 분산(kernel.cpp:89-97) — 실전 HLS 최적화 사례.
4. **C-sim 검증 하니스 존재**(testbench.cpp) + cosim 흐름(run_hls.tcl).
5. 양자화 도구(quantize.cpp)가 ggml Q4/Q5/Q8 다형식 지원.

### 한계 (대부분 코드 근거 확인)
1. **호스트 추론 본체 부재**: vit.cpp/main.cpp 없음(CMake:52-53 참조 vs Glob 부재; README:27). end-to-end 추론 불가.
2. **단일 head만**: 멀티헤드 12개·배치 미구현(README §7.1 line 184). V곱·proj·MLP·classifier 전부 미구현.
3. **softmax 정확성 의심**: 정규화 분자가 exp(score)가 아닌 raw score(kernel.cpp:100) → testbench reference(표준 softmax, testbench.cpp:65,69)와 불일치, max-subtraction 부재로 수치 불안정.
4. **다항식 exp 근사**: 큰/음수 입력에서 부정확(kernel.cpp:12).
5. **인터페이스 정합성 붕괴(3중 불일치)**:
   - synth_config.tcl의 array_partition 대상 변수(Q/K/V/Output)가 실제 커널에 없음 → 분할 directive 무효(추정).
   - opencl_wrapper / attention_wrapper는 4~6인자(V·동적크기) 가정 vs 실제 커널 3인자 고정 → 미연결.
   - attention_wrapper가 include하는 attention_kernel.h 파일 부재.
6. **고정 크기·sub-buffer 미직결**: 임의 크기 미지원, 글로벌→로컬→서브버퍼 다단 복사로 메모리 지연(README §7.2 line 188-190).
7. **클럭 100 MHz·병렬도 낮음**: cyclic 분할 미반영 시 처리량 한계. 멀티-PE/멀티-head 복제 없음.
8. **SCAILING_FACTOR=8 하드코딩**: head dim 64 전용(kernel.cpp:5).

---

## 9. 우리 프로젝트 시사점 (우리: 고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적, 추정)

1. **반례로서의 교훈**: 본 repo는 "어텐션 score까지만 단일-head 합성"한 교육용 수준 → 우리 목표(고처리량·전체 파이프라인·멀티헤드)와는 격차가 큼. **무엇이 빠지면 실사용이 안 되는지**를 보여주는 체크리스트로 유용: (a) ×V·proj·MLP·LN까지 온칩, (b) 멀티헤드/배치 병렬, (c) 호스트-커널-바인딩 인터페이스 일관성, (d) 표준·수치안정 softmax.
2. **softmax 설계 직접 참고**: 8-뱅크 부분합 II=1 누산(kernel.cpp:89-97)은 우리 LayerNorm/Softmax reduction 데이터패스 설계에 즉시 차용 가능. 단 **max-subtraction + exp(score) 분자**를 반드시 추가(본 repo의 버그를 회피).
3. **타일드 systolic + 제로패딩(208) 패턴**: 비-16배수 시퀀스 길이(197) 처리 방식은 우리도 토큰 수가 16배수가 아닐 때 동일 전략 적용 가능. 단 출력-스테이셔너리 표준형으로 재설계 권장(본 repo는 요소별 누산형, 정합성 검증 필요).
4. **HG-PIPE 대비 갭**: HG-PIPE류는 레이어 간 파이프라인·온칩 상주·고II 병렬을 추구. 본 repo는 정반대(레이어 분리, 호스트 왕복, II 미최적). 우리는 **레이어 융합 + 어텐션 전체(QKV·score·×V·proj) 온칩**을 목표로 해야 함.
5. **XR 시선추적 관점**: ViT-B(197 토큰) 자체는 무겁다. 시선추적용 경량 ViT(작은 입력/패치, fewer tokens)라면 본 repo의 고정크기 가정이 오히려 단순화에 유리하나, 멀티헤드·V·classifier까지 HW화가 필수.
6. **인터페이스 일관성 거버넌스**: 본 repo의 가장 큰 실패가 "커널/래퍼/directive 시그니처 3중 불일치". 우리 프로젝트는 **단일 커널 헤더(SoT)**를 두고 호스트·pybind·tcl이 모두 거기서 파생되도록 관리할 것.

---

## 10. 근거/한계 표기

- **확실 (코드 직접 확인)**:
  - HLS 커널 동작·pragma·차원·depth(kernel.cpp:5-115, q_matmul:6-175, kernel.h:4-15).
  - 합성 타깃 xcu50/100MHz/set_top(run_hls.tcl:6,9,11), directive 변수 불일치(synth_config.tcl:4-7).
  - 호스트 vit.cpp/main.cpp 및 attention_kernel.h 부재(Glob 결과 + CMake:52-53 + README:27).
  - 래퍼-커널 인자 불일치(opencl_wrapper.cpp:27-34, attention_wrapper.cpp:22 vs kernel.cpp:21-24).
  - 양자화 타입·흐름(quantize.cpp:37-57,272-302), vit 자료구조·XRT 핸들(vit.h:5-6,79).
  - softmax 분자가 raw score(kernel.cpp:100) vs testbench 표준 softmax(testbench.cpp:65,69).
- **추정 (코드 정황 기반)**:
  - 래퍼/directive는 구버전·미완 스케치라는 판단; testbench cosim 실패 가능성; mmult systolic 정합성(표준형과 다름); USE_FPGA OFF여도 XRT 링크되는 점이 빌드 의도와 어긋날 가능성.
- **확인 불가**:
  - `Final Project Presentation.pdf` — 현 트리에서 미발견(Glob 미노출). 발표자료 정량(리소스 사용량 LUT/FF/DSP/BRAM, 실측 latency/throughput) **확인 불가**.
  - csynth/cosim 실측 리포트(합성 산출물 부재) — 성능·자원 수치 **확인 불가**.
  - `host/ggml/` 내부(분석 제외) — ggml 양자화/텐서 연산 세부 미검토.
```