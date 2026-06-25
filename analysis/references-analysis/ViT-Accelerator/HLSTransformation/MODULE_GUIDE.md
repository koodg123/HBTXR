# HLSTransformation (TinyLlama2 단일-커널 forward) 모듈 통합 가이드

> 1차 요약(맥락): [`../HLSTransformation.md`](../HLSTransformation.md)
> 소스 루트: `REF/ViT-Accelerator/HLSTransformation`. 구현 전체가 **Vitis HLS C++** (단일 커널 `forward`) + **XRT C++ 호스트** + CPU(llama2.c 파생)/GPU(PyTorch) **전력·지연 벤치마크**. 자체 RTL 소스 없음(합성은 Vitis GUI 프로젝트로 수행, 산출물 미동봉).
> 표기 규약: 라인으로 직접 확인한 사실은 단정, 코드 정황 기반은 "추정", 코드/문서에 없으면 "확인 불가".
> 제외물(이름만): `*.bin/*.model`(체크포인트·토크나이저), `cpu_benchmarks/benchmark_stories/*.txt`(스토리 텍스트 수백 개), `gpu_benchmarks/llama/`(Meta llama 원본 third-party), `gpu_benchmarks/stories110M.pt`·`emissions.csv`·`log_*.txt`(데이터/로그), Vitis 생성물(`.ip_user_files`, build 산출물).

---

## 0. 문서 머리말

### 0.1 대표 케이스 선정
HLSTransformation은 **단일 HLS 커널**(`forward`)이 12-layer TinyLlama2 디코더 전체 + 최종 분류기를 **한 토큰(1 position)** 단위로 시분할 실행한다(`forward.cpp` L269, `main_forward_loop` L309). Edge-MoE처럼 레이어 분기는 없으므로 대표 케이스를 **데이터패스 핫스팟 1개 + 시퀀스 의존 1개**로 잡는다.

- **GEMV 핫스팟 대표**: **`matmul<N,D>` (int8 그룹 양자 행렬·벡터 곱)** — `forward.cpp` L191~L267. 주석 그대로 "by far the most amount of time is spent inside this little function"(L195). 레이어당 7회(Q/K/V/Wo/W1/W3/W2) + 분류기 1회 호출되며, 대표 인스턴스는 **FFN W1 = `matmul<dim=768, hidden_dim=2048>`**(L450). 단, 내부 `matmul1~4`의 `UNROLL`이 **전부 주석 처리**되어(L238·246·255·260) 곱셈-누산이 직렬 — 반면교사(§N+3).
- **시퀀스 의존 대표**: **멀티헤드 어텐션 `iterate`/`acc` + `softmax<257>`** — `forward.cpp` L375~L430, L51~L96. KV 캐시를 `t=0..pos`로 순회(`loop_tripcount max=257` L389·415)하므로 사이클이 position에 따라 가변. KV 캐시는 외부 DRAM(m_axi master)이라 랜덤 액세스 레이턴시가 노출된다.

선정 근거: (1) `matmul`이 전체 연산량의 압도적 다수(아래 §N+1 scalar MAC 합산)이자 최적화 미완 지점이라 우리 가속기의 "절대 켜야 할 노브"를 정량 대조할 단위, (2) 어텐션은 fp32 + 동적 KV 캐시 DRAM 접근이라 dataflow/온칩화의 효과를 보여주는 단위.

### 0.2 수치 표기 규약
- **MAC lanes**: 한 사이클(II=1 가정) 동시 곱셈기 수 = unroll/벡터 차원의 곱. **본 설계의 `matmul`은 내부 UNROLL이 전부 주석화되어 누산이 직렬 → MAC lanes = 1(int8×int8 곱 1개/cyc)로 명시**(`forward.cpp` L255-261). 외부 `i` 루프만 `PIPELINE`(L227). 데드 코드 `matmul_old`(L98-189)는 `UNROLL`이 살아있으나 `#pragma HLS PIPEPLINE` 오타(L152)로 미적용 + `w_buffer[N*D]` BRAM 폭증 → 합성 불가 추정.
- **scalar MACs**: 대표 연산의 (out_dim D)×(in_dim N) 곱(GEMV는 패치 차원 없음, 토큰 1개). 양자화 누산은 int8×int8→int32, 그룹 끝에서 fp32 환산.
- **loop trips**: `matmul`은 외부 D회 × (그룹 N/GS회 × GS회) = D×N 곱셈. 어텐션은 head(12) × (pos+1) × head_size(64).
- **memory size (payload bit)**: 온칩 버퍼 깊이×폭(bit). 활성화는 전부 `static` on-chip 상주, 가중치만 `matmul` 내부에서 **행 단위**(`w_buffer[N]`)로 DRAM→온칩 스트리밍.

### 0.3 운영 경로 (소스 ↔ export ↔ XRT ↔ 디바이스)
```
[학습/export]   PyTorch TinyLlama2(stories110M 규모) → export.py quantize_q80
        │       (그룹 GS=64 대칭 int8 [-127,127], magic "ak42"=0x616b3432, version 2)  export.py L46-64
[적재]          호스트 build_transformer→read_checkpoint(mmap .bin)→memory_map_weights
        │       int8 q + fp32 scale 분리(init_quantized_tensors), 토큰 임베딩만 fp32 dequantize  llama2.cpp L97-208
[XRT 셋업]      xrt::device(0)→load_xclbin→xrt::kernel("forward")
        │       BO 5개: transformer(gid0), key(gid3)/value(gid4) cache, out(gid5)  llama2.cpp L701-714
[전송]          transformer_buffer.write + sync(TO_DEVICE) — 전체 모델 1회 전송  llama2.cpp L717-719
[토큰 루프]     run=kernel(...) / run.set_arg(1,token),(2,pos) / run.start()/wait()
        │       out_buffer.sync(FROM_DEVICE)→read(logits)→sample()  llama2.cpp L728-796
[FPGA forward]  forward 커널 1회 = 12 디코더 레이어 + 최종 rmsnorm + 분류기  forward.cpp L309-487
```
근거: `export.py` L46-64, `llama2.cpp` L97-208·L701-804, `forward.cpp` L269-487, `*.prj` L2.

### 0.4 타깃 / 데이터타입 / 양자화 정책
- **타깃**: **AWS F1 (UltraScale+ VU9P, 16nm 데이터센터 FPGA)**. `llama_xrt_kernels.prj` L2: platform `xilinx_aws-vu9p-f1_shell-v04261818_201920_3`, platformUID `xilinx:aws-vu9p-f1:shell-v04261818:201920.3`. 빌드 타깃 3종: `sw_emu`/`hw_emu`/`hw`(`.prj` L3·25·47). **Versal/HBM 아님** — VU9P + DDR(m_axi). 합성 PPA 리포트는 리포에 미동봉 → **확인 불가**.
- **데이터타입**: **fp32 연산 + int8 가중치 저장**. `ap_fixed`/`ap_int` 미사용(`typedefs.h` L14-19에서 ap_int 후보가 전부 주석 처리, "TODO: replace with HLS types"). 활성화·중간텐서·scale은 전부 `float`. 즉 **fixed-point가 아니라 fp32 datapath + int8 weight + 동적 int8 활성화**.
- **모델 하이퍼파라미터**(`config.h` L4-11): `dim=768, hidden_dim=2048, n_layers=12, n_heads=12, n_kv_heads=12, vocab_size=32000, seq_len=1024, GS=64`. `n_heads==n_kv_heads`이므로 표준 MHA(코드는 MQA 일반형 `kv_mul=n_heads/n_kv_heads=1` 유지, L302). head_size = 768/12 = 64.
- **양자화 정책**: **그룹(GS=64) 대칭 int8**. 오프라인 가중치 = `export.py:quantize_q80` ("symmetric quantization into int8, range [-127,127]", `scale=wmax/127.0` L49·58). 런타임 활성화 = `quantize<S>()`(`forward.h` L15-64): 그룹별 `|x|`max → `scale=wmax/Q_MAX(127)` → `q=(int8_t)round(x/scale)`. **명시적 clamp 없음**(L57, round만) → overflow 보호 미흡 가능. `dequantize`(L6-13)는 토큰 임베딩 복원에만 호스트에서 사용(`llama2.cpp` L114).

---

## 1. Repo / Layer 개요

| 레이어 | 경로 | 역할 |
|---|---|---|
| **llama_xrt_kernels/src/** | `forward.cpp` + `config.h`·`typedefs.h`·`forward.h` | ★ FPGA에 합성되는 HLS 커널 본체. rmsnorm/softmax/matmul/attention/RoPE/SwiGLU + `extern "C" forward`. |
| **llama_xrt/src/** | `llama2.cpp` + 동일 헤더 4종 | XRT 호스트(CPU측 SW): 모델 적재·토크나이저(BPE)·샘플러·토큰 단위 enqueue 루프·tok/s 측정. |
| **cpu_benchmarks/** | `runq.c`(골든)·`run.c` + `runq_*.c` 변형 + toolchain | llama2.c 파생 CPU 참조구현. `runq.c`가 커널 수학의 골든 모델. 학습/양자화 toolchain(`export.py` 등). |
| **gpu_benchmarks/** | `tinyllama2.py`(지연)·`tinyllama2_power_consumption.py`(전력) | Meta llama fork로 동일 stories110M 추론. codecarbon 에너지 측정. |
| **(빌드)** | `llama_xrt.prj`·`llama_xrt_kernels.prj` | Vitis GUI 프로젝트(AWS F1 platform, sw_emu/hw_emu/hw). v++ 스크립트/cfg는 부재. |

- HLS 커널 자체 소스: **단일 파일 `forward.cpp`**(488줄) + 헤더 3종(`config.h`·`typedefs.h`·`forward.h`). 함수 모듈 단위로는 7개(quantize/dequantize, rmsnorm, softmax, matmul[+matmul_old 데드코드], RoPE, multihead_attention, SwiGLU)가 `forward` 안에 인라인/호출된다.
- include 관계: `forward.cpp` → `forward.h`(quantize/dequantize 템플릿 + extern 선언) → `typedefs.h`(Config/QuantizedTensor/TransformerWeights/Transformer) + `config.h`(하이퍼파라미터 constexpr). 순환 없음. 호스트 헤더 4종은 커널측과 동일 정의(`config.h`·`typedefs.h`·`forward.h`).

### 모듈 인스턴스 계층 (top → leaf)
```
forward  (extern "C" HLS 커널, transformer/out=m_axi gmem0/1, key/value_cache=master, token/pos=scalar)  [forward.cpp L269]
├─ memcpy token_embedding_table[token] → x (on-chip static)  [L306]
└─ main_forward_loop (l=0..11):  [L309]
   ├─ rmsnorm<dim>(xb, x, rms_att_weight)             [L313]  §2
   ├─ quantize(&xq, xb, GS)                            [L316]  §6 (forward.h)
   ├─ matmul<dim,dim>(q, ...wq) / <dim,kv_dim>(k,v)    [L317-319]  §4 ★핫스팟
   ├─ RoPE  rotation1(q,k) / rotation2(q)             [L323-363]  §5  (cosf/sinf/powf 런타임)
   ├─ memcpy k,v → key_cache/value_cache[pos]  (DRAM write)  [L364-370]
   ├─ multihead_attention (h=0..11):                  [L375-430]  §3
   │   ├─ iterate: q·k dot (t=0..pos) → att          [L385-403]
   │   ├─ softmax<257>(att+off, pos+1)                [L406]  §2
   │   └─ acc: att·value → xb                          [L412-429]
   ├─ quantize → matmul<dim,dim>(xb2,...wo) → x+=xb2   [L433-442]  §4 + residual
   ├─ rmsnorm<dim>(xb,x,rms_ffn_weight)               [L445]  §2
   ├─ quantize → matmul<dim,hidden_dim>(hb=w1, hb2=w3) [L449-451]  §4
   ├─ swi_glu: silu(hb)*hb2 → hb                        [L454-466]  §5
   └─ quantize → matmul<hidden_dim,dim>(xb,...w2) → x+=xb  [L469-478]  §4 + residual2
   (loop 끝)
   rmsnorm<dim>(x,x,rms_final_weight)                  [L482]  §2
   quantize → matmul<dim,vocab_size>(out,...wcls) → logits[32000]  [L485-486]  §4
```
- **dataflow pragma 없음** — 함수들이 순차 실행(레이어 간/연산 간 task-level 파이프라이닝 부재). 모든 활성화 텐서는 `static float` on-chip 상주(L277-300).

---

## 2. RMSNorm / Softmax — 정규화 데이터패스 (`forward.cpp` L13-96)

### 2.1 역할 + 상위/하위
RMSNorm(`rmsnorm<S>`, L13-49)은 LLaMA 표준 `o = weight * x / sqrt(mean(x²)+1e-5)`. 어텐션 전(L313)·FFN 전(L445)·최종 logits 전(L482, in-place `x,x`)에서 3회/레이어+1 호출. Softmax(`softmax<MAXSIZE>`, L51-96)는 수치안정 softmax(max-subtract)로 head별 어텐션 점수에 적용(L406). 상위: `forward`. 하위: 없음(`sqrtf`/`expf` 표준 math).

### 2.2 데이터플로우
```
RMSNorm:  x,weight → memcpy 로컬 → [sum_of_squares 루프] → ss=1/sqrt(mean+1e-5) → [norm_and_scale 루프] → memcpy o
Softmax:  x → [max] → [exp: buffer=exp(x-max)] → [sum] → inv_sum=1/sum → [norm: x=buffer*inv_sum]
```

### 2.3 function call stack
`forward` → `rmsnorm<dim>` (memcpy + 2 루프) / `softmax<257>` (4 루프). 둘 다 leaf.

### 2.4 대표 코드 위치
`forward.cpp` L24-48(rmsnorm 로컬화 + 2-pass), L57-95(softmax 4-pass).

### 2.5 대표 코드 블록

(1) **RMSNorm — 제곱합 큰 언롤 + 정규화** (`forward.cpp` L27~L47)
```cpp
sum_of_squares:
  for (int j = 0; j < S; j++) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 128 skip_exit_check
    float x_j = x_buff[j];  ss += x_j * x_j; }
  ss /= S; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
norm_and_scale:
  for (int j = 0; j < S; j++) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 64
    out_buff[j] = weight_buff[j] * (ss * x_buff[j]); }
```
→ `x_buff` `ARRAY_PARTITION cyclic factor=128`, `weight/out_buff` factor=64(L21-23)로 언롤을 메모리 포트로 뒷받침. **RMSNorm은 실제로 언롤이 살아있는 유일 핵심부**(matmul과 대비).

(2) **Softmax — 1/sum 사전계산 후 곱셈 정규화** (`forward.cpp` L70~L94)
```cpp
exp:  for (...) {#pragma HLS PIPELINE / #pragma HLS UNROLL factor = 16
        buffer[i] = expf(x[i] - max_val); }
sum:  for (...) sum += buffer[i];
  const float inv_sum = 1.0 / sum;
norm: for (...) {#pragma HLS PIPELINE / #pragma HLS UNROLL factor = 16
        x[i] = buffer[i] * inv_sum; }
```
→ 모든 루프 `#pragma HLS loop_tripcount min=0 max=257 avg=129`(L60·73·83·91) — `size=pos+1`이 런타임 가변이라 레이턴시 추정용. `MAXSIZE=257`로 인스턴스화(L406).

### 2.6 마이크로아키텍처 + 정량
- **MAC lanes**: RMSNorm은 곱셈기 위주(MAC 아님). sum_of_squares UNROLL 128 + norm_and_scale UNROLL 64 → 곱셈 ~128/64 lane 목표(II=1 가정, 합성 II 확인 불가). Softmax exp/norm UNROLL 16.
- **loop trips**: RMSNorm S=768(2-pass). Softmax size=pos+1(가변, max 257).
- **메모리(payload bit)**: rmsnorm 로컬 `x_buff/weight_buff/out_buff` 각 S×32b = 768×32 ≈ **24.6 Kb/버퍼**. softmax `buffer[257]`×32b ≈ **8.2 Kb**.
- **병목**: rmsnorm은 잘 파이프라인됨. softmax는 `expf` 런타임 + size 가변 trip. 둘 다 dataflow 없이 순차(메모리 memcpy in/out 오버헤드).

---

## 3. 멀티헤드 어텐션 — `forward.cpp` L375-430 (시퀀스 의존 대표)

### 3.1 역할 + 상위/하위
head별로 (1) q·k dot-product로 어텐션 점수 산출 → (2) `softmax<257>`(§2) → (3) att 가중치로 value 누산 → `xb`. KV 캐시는 직전에 현재 pos의 k,v를 `key_cache`/`value_cache`(외부 DRAM)에 memcpy(L364-370). 상위: `forward`(L375). 하위: `softmax<257>`(L406).

### 3.2 데이터플로우
```
(직전) k,v → memcpy → key_cache/value_cache[loff + pos*kv_dim]   (DRAM write)
multihead_attention (h):
  iterate(t=0..pos): score = Σ_i q[i+q_off]*key_cache[i+key_off]; score/=sqrt(64); att[t+att_off]=score
  softmax<257>(att+att_off, pos+1)
  acc(t=0..pos): xb[i+xb_off] += att[t+att_off] * value_cache[i+v_off]   (head_size 내부 unroll)
```

### 3.3 function call stack
`forward` → `multihead_attention`(인라인, L375) { `iterate`(L385) → `softmax<257>`(L406) → `acc`(L412) }. KV write는 루프 직전 memcpy(L369-370).

### 3.4 대표 코드 위치
`forward.cpp` L385-403(q·k dot), L406(softmax), L411-429(att·value 누산).

### 3.5 대표 코드 블록

(1) **q·k dot-product — head_size 내부 unroll, scale** (`forward.cpp` L386~L402)
```cpp
iterate:
  for (int t = 0; t <= pos; t++) {
#pragma HLS PIPELINE
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
    const int key_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
#pragma HLS unroll
      score += q[i + q_offset] * key_cache[i + key_offset]; }   // ★ key_cache = DRAM 랜덤 액세스
    score /= sqrtf(head_size);
    att[t + att_offset] = score; }
```
→ 내부 head_size(64) `#pragma HLS unroll`은 살아있음(matmul과 대조). 단 `key_cache`가 m_axi(DRAM)라 t별 랜덤 액세스 레이턴시 노출. `kv_mul=1`이라 `h/kv_mul=h`.

(2) **att·value 누산 — head_size unroll** (`forward.cpp` L412~L428)
```cpp
acc:
  for (int t = 0; t <= pos; t++) {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
    float a = att[t + att_offset];
  acc_inner:
    for (int i = 0; i < head_size; i++) {
#pragma HLS unroll
      xb[i + xb_offset] += a * value_cache[i + v_offset]; } }   // value_cache = DRAM
```

### 3.6 마이크로아키텍처 + 정량
- **MAC lanes**: q·k dot = head_size(64) unroll → 곱셈 64 lane(목표 II=1). att·value 누산 = head_size 64 lane. (matmul 대비 어텐션은 unroll이 실제 적용된 점이 대조적).
- **scalar MACs(대표, 1 토큰)**: q·k = n_heads(12) × (pos+1) × head_size(64). att·value 동일. pos=256일 때 각 ≈ 12×257×64 ≈ **0.197 M/패스**, 어텐션 합 ≈ **0.39 M**(matmul 핫스팟 대비 작음 — 토큰 1개라 GEMM보다 작다).
- **메모리(payload bit)**: `att[n_heads*seq_len]` = 12×1024×32b ≈ **393 Kb**(on-chip static, `ARRAY_PARTITION cyclic 16`). KV 캐시는 DRAM: `n_layers*seq_len*kv_dim = 12*1024*768` fp32 ≈ **36 MB**(L709 `cache_dim`) → 온칩 불가.
- **병목**: **KV 캐시 DRAM 랜덤 액세스**(L398·427) — `iterate`/`acc`의 t-루프가 매 t마다 다른 캐시 행을 m_axi로 읽음. dataflow/burst 최적화 없음 → 외부 메모리 레이턴시가 어텐션 지연을 지배할 가능성(추정).

---

## 4. matmul — int8 그룹 양자 GEMV (`forward.cpp` L191-267) ★핫스팟

### 4.1 역할 + 상위/하위
`W(D,N) @ x(N,) → xout(D,)`. 주석대로 "by far the most amount of time"(L195). 레이어당 7회(Q/K/V/Wo/W1/W3/W2) + 분류기(wcls). 입력 x·가중치 W 둘 다 int8 양자화(그룹 scale fp32). 상위: `forward`(L317-486). 하위: 없음(int8×int8→int32 누산 + fp32 환산).

### 4.2 데이터플로우
```
xq,xs(입력 벡터) → x_buffer/xs_buffer 로컬 복사(static)
for i in [0,D):  #pragma HLS PIPELINE
   w_buffer[N] ← wq[i*N ..]      (matmul1: 행 한 줄 적재)   ← DRAM 행 스트리밍
   ws_buffer[N/GS] ← ws[...]      (matmul2)
   for j (그룹 GS step):           (matmul3)
      int32 ival = Σ_k x_buffer[j+k]*w_buffer[j+k]   (matmul4)
      val += ival * ws_buffer[j/GS] * xs_buffer[j/GS]
   xout[i] = val
```

### 4.3 function call stack
`forward` → `matmul<N,D>`(템플릿, L191) { x_buff/xs_buff 복사 → i-루프(matmul1 → matmul2 → matmul3 → matmul4) }. `matmul_old<N,D>`(L98-189)는 호출되지 않는 데드 코드.

### 4.4 대표 코드 위치
`forward.cpp` L204-222(입력 로컬화), L225-248(행 단위 적재 matmul1/2), L252-264(그룹 누산 matmul3/4), L98-189(데드코드 matmul_old).

### 4.5 대표 코드 블록

(1) **행 단위 스트리밍 + 그룹 int32 누산 — 그러나 UNROLL 전부 주석화** (`forward.cpp` L235~L265)
```cpp
  for (i = 0; i < D; i++) {
#pragma HLS PIPELINE                       // ★ 외부 루프만 파이프라인
    float val = 0.0f;
    int8_t w_buffer[N];  float ws_buffer[N / GS];
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 32
    const int in = i * N;
  matmul1: for (int j = 0; j < N; j++) {
      // #pragma HLS UNROLL factor       ← 주석 처리 (적용 안 됨)
      w_buffer[j] = wq[j + in]; }          // 행 하나만 온칩 적재 (DRAM→BRAM)
  matmul3: for (j = 0; j <= N - GS; j += GS) {
      // #pragma HLS UNROLL              ← 주석 처리
      int32_t ival = 0;
    matmul4: for (int k = 0; k < GS; k++) {
        // #pragma HLS UNROLL            ← 주석 처리
        ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[j + k]); }
      val += ((float)ival) * ws_buffer[j / GS] * xs_buffer[j / GS]; }  // 그룹 끝 fp32 환산
    xout[i] = val; }
```
→ **수학은 runq.c와 동일**(group-of-GS int8 dot + scale, `runq.c` L332-336). 그러나 `matmul1`(가중치 적재)·`matmul3`(그룹)·`matmul4`(그룹 내 곱)의 `UNROLL`이 **전부 주석 처리**(L238·246·255·260)되어 **내부가 완전 직렬**. 외부 `i` 루프의 `#pragma HLS PIPELINE`(L227)만 유효하나, 내부 가변 트립(N=768~2048)이 큰 직렬 본문을 감싸므로 실효 II가 매우 클 것(추정).

(2) **데드 코드 matmul_old — 전체 가중치 BRAM 적재 시도 + 오타** (`forward.cpp` L114, L152)
```cpp
  int8_t w_buffer[N * D];                 // ★ N*D (예: 768×2048) 전체 BRAM → 폭증
  float ws_buffer[N * D / GS];
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 128
  ...
  for (i = 0; i < D; i++) {
#pragma HLS PIPEPLINE                     // ★ 오타("PIPEPLINE") → pragma 미인식
```
→ 초기 버전. `N*D` 전체를 온칩에 올리려다 BRAM 폭증 + pragma 오타로 폐기. 현재 `matmul`은 **행 단위(`w_buffer[N]`) 스트리밍**으로 바꿔 온칩 메모리를 행 하나로 줄인 것이 유일 개선.

### 4.6 마이크로아키텍처 + 정량
- **MAC lanes**: **1** (matmul4의 int8×int8 곱이 UNROLL 주석화로 직렬, L260-261). 외부 i-루프만 PIPELINE. → 토큰당 1 곱셈기로 전 GEMV 처리(워스트). 데드코드 matmul_old는 matmul3/4 UNROLL이 살아있으나 미합성.
- **scalar MACs(1 토큰, 레이어당)**: Q = dim×dim = 768×768 = **0.59 M**, K=V = dim×kv_dim = 768×768 = 0.59 M 각, Wo = 0.59 M, W1=W3 = dim×hidden = 768×2048 = **1.57 M** 각, W2 = hidden×dim = 2048×768 = 1.57 M. **레이어 합 ≈ 0.59×4 + 1.57×3 = 7.07 M MAC**. 12 레이어 = **84.8 M**. 분류기 wcls = dim×vocab = 768×32000 = **24.6 M**. **토큰 1개 forward 총 ≈ 109 M int8 MAC** (대부분 matmul). → MAC lanes=1이면 이 전부가 직렬.
- **메모리(payload bit)**: 입력 로컬 `x_buffer[N]`×8b + `xs_buffer[N/GS]`×32b. 행 버퍼 `w_buffer[N]`×8b = 최대 2048×8 = **16.4 Kb**, `ws_buffer[N/GS]`×32b = 2048/64×32 = **1.0 Kb**. **행 단위라 온칩 weight 풋프린트는 작음**(개선점) — 대신 D회 행 재적재 DRAM 트래픽.
- **병목(★)**: 내부 UNROLL 미적용으로 **GEMV가 직렬화** — 설계 전체 처리량을 좌우. §N+3에서 정량 노브 제시.

---

## 5. RoPE + SwiGLU — 위치인코딩 / FFN 활성 (`forward.cpp` L323-363, L454-466)

### 5.1 역할 + 상위/하위
RoPE(`rotation1`/`rotation2`, L323-363): q,k에 복소 회전 위치인코딩. SwiGLU(`swi_glu`, L454-466): `silu(w1(x))*w3(x)`, silu=`x·σ(x)`. 상위: `forward`. 하위: `cosf/sinf/powf`(RoPE), `expf`(SwiGLU silu) 표준 math.

### 5.2 데이터플로우
```
RoPE: for i in [0,kv_dim) step 2:  freq=1/10000^(i%head_size/head_size); val=pos*freq
        (fcr,fci)=(cos,sin); q[i],q[i+1] 회전; k[i],k[i+1] 회전   (rotation1)
      for i in [kv_dim,dim) step 2: q만 회전                       (rotation2)
SwiGLU: for i in [0,hidden_dim): val=hb[i]; val*=1/(1+exp(-val)); val*=hb2[i]; hb_out[i]=val
```

### 5.3 function call stack
`forward` → `rotation1`/`rotation2`(인라인 루프, L323/L347) / `swi_glu`(인라인 루프, L454). 모두 leaf.

### 5.4 대표 코드 위치
`forward.cpp` L325-346(q·k 회전), L347-363(q만 회전), L454-465(silu*w3).

### 5.5 대표 코드 블록

(1) **RoPE — 위치마다 삼각함수 런타임 계산** (`forward.cpp` L329~L345)
```cpp
#pragma HLS UNROLL factor = UNROLL_FACTOR   // 16
#pragma HLS PIPELINE
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);   // ★ powf 런타임
    float val = pos * freq;
    float fcr = cosf(val);  float fci = sinf(val);                      // ★ cosf/sinf 런타임
    float v0_q = q[i], v1_q = q[i+1];
    q[i] = v0_q*fcr - v1_q*fci;  q[i+1] = v0_q*fci + v1_q*fcr;          // 복소 회전 (q)
    float v0_k = k[i], v1_k = k[i+1];
    k[i] = v0_k*fcr - v1_k*fci;  k[i+1] = v0_k*fci + v1_k*fcr;          // 복소 회전 (k)
```
→ `kv_dim==dim`(n_heads==n_kv_heads)이라 `rotation2`(L347, q만, PIPELINE만)는 거의 미동작. **`cosf/sinf/powf`를 위치별로 매번 호출** — LUT/CORDIC/사전테이블 미사용. (export.py는 freqs_cos/sin을 직렬화하나 커널은 미활용 → §N+3 반례.)

(2) **SwiGLU — silu(x)*w3(x)** (`forward.cpp` L459~L464)
```cpp
#pragma HLS UNROLL factor = 4
#pragma HLS PIPELINE
    float val = hb[i];
    val *= (1.0f / (1.0f + expf(-val)));   // silu = x·σ(x)
    val *= hb2[i];                          // ⊙ w3(x)
    hb_out[i] = val;
```
→ `hb_out` `ARRAY_PARTITION cyclic factor=16`(L453). UNROLL 4 + PIPELINE.

### 5.6 마이크로아키텍처 + 정량
- **MAC lanes**: RoPE UNROLL 16(2-원소 회전쌍 8개 병렬, 회전당 4 곱+2 합). SwiGLU UNROLL 4.
- **loop trips**: RoPE rotation1 = kv_dim/2 = 384 반복. SwiGLU = hidden_dim = 2048.
- **메모리**: RoPE는 q/k(on-chip static, dim×32b) in-place. SwiGLU `hb_out[hidden_dim]`×32b = 2048×32 ≈ **65.5 Kb**.
- **병목**: RoPE의 `powf/cosf/sinf`는 FPGA에서 비싼 비선형 — 위치마다 384쌍×3함수 호출. 사전 테이블화 시 대폭 절감 여지(미적용). SwiGLU `expf` 1회/원소는 짧음.

---

## 6. 양자화 / XRT 인터페이스 — `forward.h` L15-64, `forward.cpp` L269-303

### 6.1 역할 + 상위/하위
`quantize<S>()`(forward.h L15-64): 매 matmul 직전 활성화를 그룹(GS=64) 동적 int8 양자화. `forward` 커널 인터페이스(forward.cpp L269-303): m_axi/scalar 포트 정의 + 활성화 텐서 static on-chip 선언. 상위: `forward`. 하위: 없음.

### 6.2 데이터플로우
```
quantize:  for group in [0,S/64): wmax = max|x[group]|; scale=wmax/127;
             for i in GS: q = round(x[base+i]/scale)  → quantized_buffer
           memcpy → qx->q, qx->s
인터페이스: transformer(gmem0, m_axi) / out(gmem1, m_axi) / key·value_cache(master) / token·pos(scalar)
           활성화 x,xb,xb2,hb,hb2,xq,hq,q,k,v,att = static + ARRAY_PARTITION cyclic 16 (on-chip)
```

### 6.3 function call stack
`forward` → `quantize(&xq, xb, GS)` (7회/레이어 + 1, forward.h L16) → memcpy. 인터페이스 pragma는 `forward` 본문 진입부(L271-300).

### 6.4 대표 코드 위치
`forward.h` L27-60(그룹 양자화 루프), `forward.cpp` L271-272(m_axi 번들), L277-300(활성화 static + 파티션).

### 6.5 대표 코드 블록

(1) **그룹 대칭 양자화 — clamp 없이 round** (`forward.h` L28~L58)
```cpp
main_loop: for (int group = 0; group < num_groups; group++) {
#pragma HLS UNROLL factor = 8
#pragma HLS PIPELINE
    float wmax = 0.0;  int base_idx = group * GS;
    max: for (int i = 0; i < GS; i++) {#pragma HLS PIPELINE
          float val = fabs(x[base_idx + i]); if (val > wmax) wmax = val; }
    float scale = wmax / Q_MAX;  scale_buffer[group] = scale;   // Q_MAX=127
    for (int i = 0; i < GS; i++) {#pragma HLS PIPELINE
      float quant_value = x[base_idx + i] / scale;
      int8_t quantized = (int8_t)round(quant_value);            // ★ round만 (명시적 clamp 없음)
      quantized_buffer[base_idx + i] = quantized; } }
```
→ `quantized_buffer` `ARRAY_PARTITION cyclic factor=64`, `scale_buffer` factor=16(L23-24). 그룹 루프 UNROLL 8 + PIPELINE.

(2) **커널 인터페이스 — m_axi 2번들 + 활성화 on-chip** (`forward.cpp` L271~L300)
```cpp
#pragma HLS INTERFACE m_axi port = transformer offset = slave bundle = gmem0   // 전체 int8 가중치
#pragma HLS INTERFACE m_axi port = out         offset = slave bundle = gmem1   // logits
  static float x[config.dim]; ... static float att[config.n_heads*config.seq_len];
#pragma HLS ARRAY_PARTITION variable = q cyclic factor = UNROLL_FACTOR   // 16
  ... (x,xb,xb2,hb,hb2,xq,hq,q,k,v,att 전부 cyclic 16)
```
→ `key_cache`/`value_cache`/`out`/`transformer`는 `.prj`에서 master(=m_axi) 등록(`llama_xrt_kernels.prj` L6-11). `token`/`pos`는 스칼라(자동 s_axilite). **dataflow 미사용** — 함수 순차.

### 6.6 마이크로아키텍처 + 정량
- **MAC lanes**: quantize는 MAC 아님(max + 나눗셈 + round). 그룹 UNROLL 8.
- **loop trips**: quantize = S/64 그룹 × (max GS + quant GS). 호출당 S = dim(768)→12그룹 또는 hidden(2048)→32그룹.
- **메모리(payload bit)**: 활성화 on-chip 합: x/xb/xb2 각 768×32b ≈ 24.6 Kb, hb/hb2 각 2048×32b ≈ 65.5 Kb, att ≈ 393 Kb, q/k/v 각 768×32b. xq.q(int8)+xq.s(fp32), hq 동일. **중간텐서 전부 on-chip 상주**, 가중치만 DRAM(가중치 풋프린트는 §4 행 버퍼).
- **병목**: `transformer` 전체(int8 기준 수십~수백 MB)를 m_axi 단일 번들 gmem0로 묶음 → 가중치 대역폭 경합. quantize는 매 matmul 직전 동기 실행(오버랩 없음). 정확도: clamp 부재(L57) → 극단값 overflow 가능(추정).

---

## 7. 호스트 XRT enqueue + 벤치마크 매핑 — `llama2.cpp`, `cpu_benchmarks`, `gpu_benchmarks`

### 7.1 역할 + 상위/하위
호스트(`llama2.cpp`)는 모델 적재·BPE 토크나이저·샘플러·**토큰 단위 enqueue 루프**·tok/s 측정. CPU/GPU 벤치는 동일 모델의 지연·전력 베이스라인. 상위: 사용자 CLI. 하위: XRT 런타임(`xrt::device/kernel/bo`).

### 7.2 데이터플로우 (호스트 주도 autoregressive)
```
prompt → encode(BPE) → tokens[]
모델 1회 전송: transformer_buffer.write + sync(TO_DEVICE)            llama2.cpp L717-719
첫 토큰:  run = kernel(transformer,token,pos,key,value,out); run.wait()  L728-729
          out.sync(FROM_DEVICE) → read(logits)                         L735-736
이후 토큰: run.set_arg(1,token); set_arg(2,pos); run.start(); run.wait()  L760-763
          out.sync(FROM_DEVICE)→read→sample()→decode 출력             L769-795
tok/s 측정: time_in_ms() 첫 토큰 이후 구간                            L799-804
```

### 7.3 function call stack
`main` → `build_transformer`→`read_checkpoint`(mmap)→`memory_map_weights`→`init_quantized_tensors` → `build_tokenizer`/`build_sampler` → `generate`(L682) { xrt 셋업 → BO 할당 → 전송 → 토큰 루프(`kernel`/`set_arg`/`start`/`wait`/`sync`/`read`/`sample`) }.

### 7.4 대표 코드 위치
`llama2.cpp` L701-714(BO group_id 매핑), L717-719(모델 전송), L728-736(첫 토큰), L758-796(토큰 루프), L799-804(tok/s).

### 7.5 대표 코드 블록

(1) **BO group_id ↔ 커널 인자 매핑 + 모델 1회 전송** (`llama2.cpp` L707~L719)
```cpp
auto out_buffer = xrt::bo(device, vocab_size*sizeof(float), kernel.group_id(5));   // arg5 out
auto transformer_buffer = xrt::bo(device, sizeof(*transformer), kernel.group_id(0)); // arg0
auto key_buffer   = xrt::bo(device, cache_dim*sizeof(float), kernel.group_id(3));   // arg3
auto value_buffer = xrt::bo(device, cache_dim*sizeof(float), kernel.group_id(4));   // arg4
transformer_buffer.write(transformer, sizeof(*transformer), 0);
transformer_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);   // 전체 모델 1회만 전송
```
→ `cache_dim = n_layers*seq_len*((dim*n_kv_heads)/n_heads)`(L709) ≈ 9.4 M float = 36 MB/캐시.

(2) **무거운 버퍼 재사용 + 스칼라 인자만 갱신** (`llama2.cpp` L760~L763)
```cpp
run.set_arg(1, token);   // token만
run.set_arg(2, pos);     // pos만
run.start();
run.wait();
```
→ 모델·KV 버퍼는 재사용, 스칼라 2개만 바꿔 재실행. 토큰당 1회 디바이스 왕복(동기 `wait`).

### 7.6 마이크로아키텍처 + 정량 (벤치 매핑)
| 구성 | 위치 | 역할 | 측정 |
|---|---|---|---|
| HLS 커널 | `forward.cpp` | FPGA forward 본체 | FPGA 추론 |
| XRT 호스트 | `llama2.cpp` | enqueue 루프 | FPGA tok/s(L803) |
| CPU 골든 | `runq.c` matmul(L317-342)/rmsnorm(L282)/softmax(L297) | int8 추론, **커널 수학 동일** | 정확도·tok/s 기준 |
| CPU 변형 | `runq_power_consumption.c`·`runq_latency_256/1024.c`·`*_ppl.c` | 출력/EOS 억제 + 강제 루프 | CPU 지연/전력/PPL |
| GPU 벤치 | `tinyllama2.py`(지연)·`tinyllama2_power_consumption.py`(전력, codecarbon) | Meta llama fork | GPU 지연/에너지 |

- **검증 경로**: HLS `matmul`(그룹 int32 누산 + scale)과 `runq.c matmul`(L332-336)이 **동일 알고리즘** → runq.c가 FPGA 출력의 골든 모델. export.py `quantize_q80`(L46-64)이 가중치 양자화 골든 소스.
- **합성 PPA(LUT/FF/DSP/BRAM/주파수/지연)**: 리포트 미동봉 → **확인 불가**.

---

## N+1. 모듈 한눈 요약 표

| # | 모듈 | 파일:라인 | 핵심 역할 | MAC lanes(목표) | 대표 scalar MACs(1 토큰) | 주 메모리(추정) | 핵심 병목 |
|---|---|---|---|---|---|---|---|
| 2 | RMSNorm/Softmax | `forward.cpp` L13-96 | 정규화 데이터패스 | 곱 128/64(rms), 16(sm) | — (곱셈) | rms 로컬 ~24.6Kb, att buf 8.2Kb | 순차(dataflow 없음) |
| 3 | MHA | `forward.cpp` L375-430 | q·k/softmax/·V | 64(q·k), 64(·V) | 어텐션 합 ~0.39M | att 393Kb(온칩), KV 36MB(DRAM) | KV DRAM 랜덤 액세스 |
| 4 | matmul ★ | `forward.cpp` L191-267 | int8 그룹 GEMV | **1 (UNROLL 주석화)** | 레이어 7.07M / 분류기 24.6M / 총 ~109M | 행버퍼 ~16.4Kb(작음) | **내부 UNROLL 미적용 직렬** |
| 5 | RoPE/SwiGLU | `forward.cpp` L323-363,454-466 | 위치인코딩/FFN활성 | 16(RoPE), 4(SwiGLU) | — | hb_out 65.5Kb | powf/cosf/sinf 런타임 |
| 6 | quantize/인터페이스 | `forward.h` L15-64, `forward.cpp` L269-303 | 동적 int8 + m_axi | 8(그룹) | — | 활성화 on-chip 상주 | gmem0 단일번들 대역폭, clamp 부재 |
| 7 | XRT 호스트/벤치 | `llama2.cpp` L682-804 외 | enqueue + 3-플랫폼 벤치 | — | — | KV/out BO(DRAM) | 동기 wait 직렬 |

---

## N+2. 읽기·코드추적 순서 (권장)

1. **상수/타입**: `config.h`(하이퍼파라미터 L4-11) → `typedefs.h`(QuantizedTensor·TransformerWeights L33-64) → `forward.h`(quantize/dequantize L6-64).
2. **양자화 이해**: `forward.h` quantize(L15-64) ↔ `cpu_benchmarks/export.py` quantize_q80(L46-64, 골든) ↔ `cpu_benchmarks/runq.c` matmul(L317-342, 골든 수학).
3. **커널 골격**: `forward.cpp` 인터페이스(L269-303) → `main_forward_loop`(L309-487, 레이어 1회 흐름).
4. **핫스팟**: `forward.cpp` matmul(L191-267) — UNROLL 주석화 라인(L238·246·255·260) 확인이 핵심. 데드코드 matmul_old(L98-189)와 대조.
5. **데이터패스**: rmsnorm(L13-49) → softmax(L51-96) → RoPE(L323-363) → SwiGLU(L454-466).
6. **어텐션**: MHA iterate/softmax/acc(L375-430) + KV write(L364-370).
7. **호스트/벤치**: `llama2.cpp` generate(L682-807) → CPU `runq.c`/`runq_power_consumption.c` → GPU `tinyllama2*.py` → `gpu_benchmarks/README.md`.

---

## N+3. 병목 후보 & 병렬도 노브 (★matmul UNROLL 미적용 정량)

| 노브 | 위치 | 현재값 | 효과 | 리스크 |
|---|---|---|---|---|
| **matmul4 내부 UNROLL** | `forward.cpp` L260 | 주석화(=1) | 켜면 그룹 내 int8 곱 GS=64 병렬 | DSP/LUT↑, w_buffer 포트 폭↑ |
| **matmul3 그룹 UNROLL** | `forward.cpp` L255 | 주석화(=1) | 켜면 그룹 병렬 누산 | int32 누산기 다수, fanout↑ |
| **matmul1 가중치 적재 UNROLL** | `forward.cpp` L238 | 주석화(=1) | 켜면 행 적재 대역폭↑ | m_axi 포트/burst 설계 필요 |
| matmul i-루프 PIPELINE | `forward.cpp` L227 | 적용(유일) | II 유지 핵심 | 내부 직렬이라 실효 II 큼 |
| RoPE 삼각함수 | `forward.cpp` L330·353 | powf/cosf/sinf 런타임 | 사전테이블화 시 비선형 제거 | LUT/BRAM 소모, export freqs 활용 |
| dataflow(레이어/연산) | `forward.cpp` 전체 | 미사용 | task-level 파이프라인 | stream 재설계, 면적↑ |
| KV 캐시 위치 | `forward.cpp` L398·427 | DRAM(m_axi) | 온칩/타일링 시 레이턴시↓ | 36MB → 전체 온칩 불가, 타일 필요 |
| m_axi 번들 분리 | `forward.cpp` L271-272 | gmem0(transformer)/gmem1(out) | 가중치 다채널 분리 시 대역폭↑ | 번들/SLR 배치 설계 |
| quantize clamp | `forward.h` L57 | round만(clamp 없음) | clamp 추가 시 overflow 방어 | 미미한 추가 로직 |

**핵심 병목 진단(정량)**: HLSTransformation은 **레이턴시 중심 단일-호출** 설계로, 토큰당 1회 forward를 순차 실행한다. 결정적 병목은 **`matmul`의 내부 데이터패스 직렬화**다 — `matmul1/matmul3/matmul4`의 `UNROLL`이 모두 주석 처리(`forward.cpp` L238·246·255·260)되어 **MAC lane = 1**. 토큰 1개 forward의 총 int8 MAC ≈ **109 M**(레이어 7.07M×12 + 분류기 24.6M, §4.6)이 사실상 단일 곱셈기로 직렬 처리된다(외부 i-루프 PIPELINE은 큰 직렬 본문을 감싸 실효 II가 커짐). matmul4의 UNROLL만 켜도(GS=64) 그룹 내 64 곱 병렬 → 곱셈 수준 64배 잠재 향상(합성 자원 한도 내). 부차 병목: (1) RoPE의 `powf/cosf/sinf` 위치별 런타임 호출(384쌍×3, L330·353) — export.py가 freqs_cos/sin을 직렬화함에도 커널은 미활용, (2) 어텐션 KV 캐시 36MB DRAM 랜덤 액세스(L398·427), (3) dataflow 부재로 연산·메모리 비중첩. 합성 PPA(DSP/BRAM/주파수 달성·지연)는 리포트 미동봉으로 **확인 불가** — csynth 실행 또는 논문 본문 필요.

---

## 부록. submission 형제 동일 여부

`REF/Transformer-Accel`에는 **두 개의 형제 트리**가 존재한다: `Transformer-Accel/HLSTransformation/...` 와 `Transformer-Accel/submission/...`. 본 가이드 대상 `ViT-Accelerator/HLSTransformation`까지 **3중 사본**이다.

- **핵심 1~2파일 비교 결과(byte-동일 확인)**:
  - `llama_xrt_kernels/src/forward.cpp` — `matmul`(L191-267) + 파일 헤더(L1-14)를 세 트리에서 비교: **완전히 동일**(UNROLL 주석화 라인 L238·246·255·260, 데드코드 matmul_old 포함). `Transformer-Accel/HLSTransformation`·`Transformer-Accel/submission`·`ViT-Accelerator/HLSTransformation`의 matmul 영역 라인이 1:1 일치.
  - `extern "C" forward` 시그니처(L269) 및 m_axi pragma 동일.
- **결론**: 세 트리는 **동일 코드베이스의 사본(byte-동일)**으로 판단(핵심 GEMV·인터페이스 라인 일치 근거). `Transformer-Accel/submission`은 ViT-Accelerator/HLSTransformation과 같은 TinyLlama2 단일-커널 forward이며, 본 가이드의 모든 분석(§2~N+3)이 그대로 적용된다. (전 파일 전수 바이트 해시 비교는 미수행 — 핵심 파일 라인 일치로 동일 판정.)
