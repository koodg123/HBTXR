# ViT-Accelerator / HLSTransformation 정밀 분석

> 분석 대상 repo: `REF/ViT-Accelerator/HLSTransformation`
> 분석 방식: 실제 소스 Read 후 `파일명:라인` / 함수명 근거 기반. (bash 미사용, Glob/Grep/Read 만)
> 작성일 기준 데이터/로그 파일(`*.bin/*.model/*.chk/*.csv/log_*.txt`), `cpu_benchmarks/benchmark_stories/*.txt`(스토리 텍스트 수백 개), `gpu_benchmarks/llama`(Meta llama 원본 third-party)은 본문 분석 대상에서 제외하고 이름만 언급한다.

---

## 1. 개요

- **무엇인가**: Karpathy의 `llama2.c`(int8 양자화 추론) 계열의 **TinyLlama2 (stories110M 규모) Transformer 디코더**를, Vitis / Vitis HLS로 작성한 단일 HLS 커널(`forward`)로 FPGA에 가속하고, XRT(Xilinx Runtime) C++ 호스트로 구동하는 프로젝트. 부수적으로 **동일 모델의 CPU 참조 구현(llama2.c 파생)과 GPU(PyTorch) 구현으로 전력·지연을 측정하는 벤치마크 묶음**을 포함한다.
- **한 줄 요약**: "int8 그룹 양자화된 TinyLlama2 forward pass 전체를 하나의 m_axi HLS 커널로 합성하고, XRT 호스트가 토큰 단위로 enqueue하며 autoregressive 생성을 수행하는 FPGA Transformer 추론 + CPU/GPU 전력·지연 벤치마크 레퍼런스".
- **타깃 디바이스 (코드 근거 확정)**:
  - `llama_xrt/llama_xrt.prj:2`, `llama_xrt_kernels/llama_xrt_kernels.prj:2` 모두 platform = `xilinx_aws-vu9p-f1_shell-v04261818_201920_3`, platformUID = `xilinx:aws-vu9p-f1:shell-v04261818:201920.3`.
  - 즉 **AWS F1 (Alveo급 UltraScale+ VU9P)** 가속 카드. Versal이 아니라 **UltraScale+ (16nm) 데이터센터 FPGA**다.
  - 빌드 타깃은 `sw_emu` / `hw_emu` / `hw` 3종 (`*.prj`의 configuration 3개).
- **모델 하이퍼파라미터** (`llama_xrt_kernels/src/config.h:4-11`, 호스트 `llama_xrt/src/config.h`와 완전히 동일):
  - `dim=768, hidden_dim=2048, n_layers=12, n_heads=12, n_kv_heads=12, vocab_size=32000, seq_len=1024, GS=64`.
  - `n_heads == n_kv_heads`이므로 MQA/GQA가 아닌 **표준 MHA**(다만 코드는 MQA 일반형 `kv_mul = n_heads/n_kv_heads`를 유지). head_size = 768/12 = 64. group size(양자화) GS=64.

---

## 2. 디렉토리 구조

```
HLSTransformation/
├─ llama_xrt/                 # XRT 호스트 (CPU측 SW)
│  ├─ src/llama2.cpp          # main, 토크나이저, 샘플러, XRT enqueue 루프
│  ├─ src/config.h            # 모델 하이퍼파라미터 (커널과 동일)
│  ├─ src/typedefs.h          # Config / QuantizedTensor / TransformerWeights / Transformer
│  ├─ src/forward.h           # forward() extern 선언 + quantize/dequantize 템플릿
│  ├─ src/tokenizer.bin       # (데이터, 제외)
│  └─ llama_xrt.prj           # Vitis 호스트 프로젝트 (AWS F1 platform)
├─ llama_xrt_kernels/         # HLS 커널 (★ FPGA에 합성되는 부분)
│  ├─ src/forward.cpp         # ★ rmsnorm/softmax/matmul/attention/RoPE/SwiGLU + extern "C" forward()
│  ├─ src/config.h            # 동일 하이퍼파라미터
│  ├─ src/typedefs.h          # 동일 구조체 정의
│  ├─ src/forward.h           # quantize/dequantize 템플릿 (HLS pragma 포함)
│  └─ llama_xrt_kernels.prj   # HW 커널 프로젝트, kernel "forward", 인자 5개 + master 버퍼
├─ cpu_benchmarks/            # CPU 참조구현 + 학습/양자화 toolchain (llama2.c 파생)
│  ├─ run.c                   # fp32 비양자화 참조 추론 (Karpathy 원본 파생)
│  ├─ runq.c                  # ★ int8 그룹 양자화 참조 추론 (커널 수학의 골든 모델)
│  ├─ runq_power_consumption.c, runq_latency_256.c, runq_latency_1024.c  # 벤치 변형
│  ├─ runq_ppl.c, run_ppl.c   # perplexity 측정 변형
│  ├─ test.c / test_all.py    # 검증
│  ├─ model.py, train.py, export.py, configurator.py, tokenizer.py, sample.py, tinystories.py
│  ├─ Makefile                # gcc -Ofast 빌드 타깃 (run/runq/벤치/win64 등)
│  └─ benchmark_stories/*.txt # (데이터 수백 개, 제외)
└─ gpu_benchmarks/            # GPU(PyTorch) 측정 (Meta llama fork)
   ├─ tinyllama2.py                     # 지연 측정 (latency)
   ├─ tinyllama2_power_consumption.py   # 전력/에너지 측정 (codecarbon)
   ├─ llama/                            # Meta llama 원본 (third-party, 제외)
   ├─ README.md, requirements.txt, emissions.csv(데이터), log_*.txt(데이터)
   └─ stories110M.pt 등 (데이터, 제외)
```

루트에 통합 README는 없음. `gpu_benchmarks/README.md`만 존재(아래 §5·§6에서 인용).

---

## 3. 핵심 모듈 정밀 분석 ★

가장 중요한 파일은 **`llama_xrt_kernels/src/forward.cpp`** — TinyLlama2의 forward pass 전체를 하나의 HLS 커널로 합성한다. 데이터 타입은 거의 전부 `float`(fp32)이며, **가중치만 int8 그룹 양자화**(scale은 fp32)다. `ap_fixed`/`ap_int`는 사용하지 않는다(`typedefs.h:14-19`에서 ap_int 후보가 주석 처리됨 — "TODO: replace with HLS types"). 즉 **fixed-point가 아니라 fp32 연산 + int8 weight 저장** 방식이다.

### 3.0 데이터 타입 / 양자화 스킴 (config.h / typedefs.h / forward.h)

- `QuantizedTensor<SIZE>` (`typedefs.h:33-38`): `int8_t q[SIZE]` (양자화 값) + `float s[SIZE]` (그룹별 scale). 실제로 scale은 그룹당 1개라 `SIZE/GS`만 유효.
- `TransformerWeights<...>` (`typedefs.h:41-64`): 토큰 임베딩(`q_tokens` + 디퀀타이즈된 fp32 `token_embedding_table`), rmsnorm 가중치(fp32), wq/wk/wv/wo·w1/w2/w3·wcls(전부 `QuantizedTensor` = int8), final rmsnorm(fp32).
- `quantize<S>()` (`forward.h:15-64` 커널측, 호스트측 동일): 그룹(GS=64) 단위 **대칭 양자화**. 그룹 내 `|x|` 최대값 → `scale = wmax/127`, `q = round(x/scale)`. Q_MAX=127.0f. HLS pragma: `quantized_buffer`에 `ARRAY_PARTITION cyclic factor=64`, `scale_buffer` factor=16, 그룹 루프 `UNROLL factor=8 + PIPELINE`(`forward.h:23-31`).
- `dequantize<S>()` (`forward.h:6-13`): `x[i] = q[i] * s[i/GS]`. 토큰 임베딩 테이블 복원에만 호스트에서 사용(`llama2.cpp:114`).
- 양자화 정의의 골든 소스는 `cpu_benchmarks/export.py:46 quantize_q80()` — "symmetric quantization into int8, range [-127,127]", `w/(maxabs/127)` (export.py:49,54-64). 체크포인트 magic=`0x616b3432`("ak42"), version 2 (host `llama2.cpp:150,161`).

### 3.1 `rmsnorm<S>()` — RMS 정규화 (`forward.cpp:13-49`)

- 역할: `o = weight * x / sqrt(mean(x^2)+1e-5)`. 표준 LLaMA RMSNorm.
- 구현: 입력 `x`, `weight`를 로컬 버퍼로 `memcpy` 후(`:24-25`) 2개 루프.
  - `sum_of_squares`(`:27-34`): 제곱합. `#pragma HLS PIPELINE` + `UNROLL factor=128 skip_exit_check`.
  - `norm_and_scale`(`:39-47`): 정규화·스케일. `PIPELINE` + `UNROLL factor=64`.
- pragma: 로컬 버퍼 `x_buff`(`ARRAY_PARTITION cyclic factor=128`), `weight_buff`/`out_buff`(factor=64) (`:21-23`) — 큰 언롤 팩터를 메모리 포트로 뒷받침.
- 호출처: 어텐션 전(`:313`), FFN 전(`:445`), 최종 logits 전(`:482`, in-place `x,x`).

### 3.2 `softmax<MAXSIZE>()` — 어텐션 점수 정규화 (`forward.cpp:51-96`)

- 역할: 수치 안정 softmax(max 빼고 exp/합/정규화). 어텐션 head별 점수에 적용.
- 4단계 루프: `max`(`:57-67`), `exp`(`:70-78`, `UNROLL factor=16`), `sum`(`:80-85`), `norm`(`:88-95`, `UNROLL factor=16`).
- pragma 특징: 모든 루프에 `#pragma HLS loop_tripcount min=0 max=257 avg=129` — `pos`가 런타임에 따라 가변(시퀀스 위치)이므로 합성 시 트립카운트를 명시해 레이턴시 추정. `MAXSIZE=257`로 인스턴스화(`:406`).
- `1.0/sum`을 미리 계산해 곱셈으로 정규화(`:87,94`).

### 3.3 `matmul<N,D>()` — int8 양자 행렬·벡터 곱 (`forward.cpp:191-267`)

- 역할: `W(D,N) @ x(N,) -> xout(D,)`. 주석대로 "by far the most amount of time" — **핵심 연산 핫스팟**. 입력 x와 가중치 W 둘 다 int8 양자화.
- 시그니처: `matmul(float* xout, int8_t* xq, float* xs, int8_t* wq, float* ws)` — 양자화 값 + 그룹 scale을 받음.
- 구조:
  - 입력 벡터 `xq`/`xs`를 정적 로컬 버퍼로 복사(`x_buff`/`xs_buff` 루프, `UNROLL 16/4`, `:211-222`).
  - 외부 출력 루프 `for i in [0,D)` `#pragma HLS PIPELINE`(`:225-227`).
  - 행마다 가중치 한 줄 `w_buffer[N]`(int8) + `ws_buffer[N/GS]`을 로컬로 적재(`matmul1`,`matmul2`, `:235-248`), `ARRAY_PARTITION cyclic factor=32`(`:231-232`).
  - 그룹(GS=64) 단위 누산: `matmul3`(그룹 루프) → `matmul4`(그룹 내 int8×int8 → int32 누산, `:257-263`), 그룹 끝에서 `val += int32 * w_scale * x_scale`로 fp32 환산. 즉 **그룹별 int8 dot-product를 int32로 정확히 누산 후 scale 곱셈** (llama2.c runq의 행렬곱과 동일 수학, `runq.c:317-336` 비교 시 동일).
- `matmul_old`(`:98-189`)는 **데드 코드**: 전체 가중치 행렬 `w_buffer[N*D]`을 한 번에 BRAM에 올리려는 초기 버전. `N*D`가 거대(예: 768×2048)해 array_partition factor=128로도 BRAM이 폭증하고, `#pragma HLS PIPEPLINE`(오타)로 실제 적용도 안 됨. 현재 `matmul`은 **행 단위 스트리밍**으로 바꿔 on-chip 메모리를 행 하나(`w_buffer[N]`)로 줄인 것이 핵심 개선.
- 한계: 활성화된 `matmul`은 `matmul1`~`matmul4` 내부 UNROLL이 **전부 주석 처리**됨(`:238,246,255,260`). 즉 외부 `i` 루프만 PIPELINE되고 내부 곱셈-누산은 직렬. 성능 최적화가 미완 상태로 남아 있음(§8 참조).

### 3.4 RoPE — 회전 위치 인코딩 (`forward.cpp:323-363`)

- `rotation1`(`:323-346`): `i < kv_dim` 구간에서 q,k 둘 다 회전. `head_dim=i%head_size`, `freq=1/10000^(head_dim/head_size)`, `val=pos*freq`, `(fcr,fci)=(cos,sin)`로 2D 복소 회전.
  - pragma: `#pragma HLS UNROLL factor=16` + `PIPELINE`.
- `rotation2`(`:347-363`): `i >= kv_dim` 구간에서 q만 회전. `PIPELINE`만.
- `n_heads==n_kv_heads`라 실제로 `kv_dim==dim`이라 rotation2는 거의 동작하지 않지만 일반형 코드를 유지.
- `cosf/sinf/powf`를 런타임 호출 — LUT/CORDIC 미사용. 위치별로 매번 삼각함수를 계산(미리 테이블화하지 않음).

### 3.5 멀티헤드 어텐션 (`forward.cpp:372-430`)

- KV 캐시 갱신: 현재 위치 `pos`의 k,v를 `key_cache`/`value_cache`에 `memcpy`(`:364-370`). 캐시는 커널 인자(외부 DRAM, §3.7)다.
- `multihead_attention`(`:375-430`): head별 루프.
  - `iterate`(`:385-403`): t=0..pos에 대해 q·k dot-product(head_size 내부 `#pragma HLS unroll`), `score/=sqrt(head_size)`, `att[]`에 저장. `loop_tripcount max=257`.
  - `softmax<257>(att+att_offset, pos+1)`(`:406`).
  - `acc`(`:412-429`): att 가중치로 value를 누산해 `xb`에 기록(`acc_inner` head_size `#pragma HLS unroll`).
- 어텐션 점수와 value 누산 모두 `key_cache`/`value_cache`를 직접 인덱싱 — DRAM(m_axi) 랜덤 액세스가 발생(레이턴시 리스크).

### 3.6 FFN + SwiGLU + residual (`forward.cpp:432-479`)

- 어텐션 출력 매트멀: `quantize(&xq,xb)` → `matmul<dim,dim>(xb2,...wo)`(`:433-434`) → residual `x += xb2`(`:437-442`, `UNROLL factor=64`).
- FFN: rmsnorm(`:445`) → `quantize` → `matmul<dim,hidden_dim>`로 `hb=w1(x)`, `hb2=w3(x)`(`:450-451`).
- `swi_glu`(`:454-465`): `val = silu(hb)*hb2`, silu = `x*sigmoid(x)` = `x/(1+exp(-x))`(`:461`). `UNROLL factor=4` + `PIPELINE`, `hb_out` `ARRAY_PARTITION factor=16`.
- 최종 FFN 매트멀: `quantize(&hq,hb)` → `matmul<hidden_dim,dim>(xb,...w2)`(`:469-470`) → residual2 `x += xb`(`:473-478`).
- 최종: `rmsnorm`(`:482`) → `quantize` → `matmul<dim,vocab_size>(out,...wcls)`로 32000개 logits 생성(`:485-486`).

### 3.7 XRT 인터페이스 — 커널 시그니처와 pragma (`forward.cpp:269-303`)

```cpp
extern "C" void forward(Transformer<...>* transformer, int token, int pos,
        float key_cache[...], float value_cache[...], float* out)
#pragma HLS INTERFACE m_axi port=transformer offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out        offset=slave bundle=gmem1
```

- **`transformer`(전체 모델: config+가중치)**, **`out`(logits)**: `m_axi`로 DRAM 매핑. `transformer`는 gmem0, `out`는 gmem1로 번들 분리.
- `key_cache`/`value_cache`도 포인터 인자(`.prj`에서 master=true). 명시적 `#pragma HLS INTERFACE`는 없지만 `.prj`(`llama_xrt_kernels.prj:9-11`)에서 `key_cache`,`value_cache`,`out`,`transformer`가 master(=m_axi)로 등록됨. `token`/`pos`는 스칼라(자동 s_axilite/제어).
- 커널 내부 활성화 버퍼(`x,xb,xb2,hb,hb2,xq,hq,q,k,v,att`)는 전부 `static float`(`:277-287`) + 대규모 `ARRAY_PARTITION cyclic factor=16`(`:288-300`). 즉 **모든 중간 텐서는 on-chip(BRAM/레지스터)에 상주**, 가중치만 DRAM에서 행 단위로 스트리밍.
- 주의: `transformer` 한 구조체가 전체 가중치를 담음(int8 기준 수십~수백 MB). 호스트가 통째로 DRAM 버퍼에 write(`llama2.cpp:717`)하고, 커널은 m_axi로 읽음. **dataflow pragma는 사용하지 않음** — 함수 호출 순차 실행(레이어 루프 `main_forward_loop`, `:309`).

### 3.8 호스트측 커널 enqueue 흐름 (`llama_xrt/src/llama2.cpp`)

- 표준 llama2.c의 토크나이저(BPE, `encode/decode`, `:241-491`)·샘플러(argmax/multinomial/top-p, `:512-666`)를 그대로 사용(전형적 llama2.c 파생).
- 모델 적재: `build_transformer`→`read_checkpoint`(`:135-208`)에서 `mmap`으로 .bin을 매핑하고 `memory_map_weights`(`:97-133`)로 `TransformerWeights` 구조체를 채움. `init_quantized_tensors`(`:79-95`)로 int8 q + fp32 scale을 분리 적재. 토큰 임베딩만 `dequantize`로 fp32 복원(`:114`).
- **XRT 구동 (`generate`, `:681-807`)**:
  1. `xrt::device(0)` → `load_xclbin(kernelpath)` → `xrt::kernel(device,uuid,"forward")` (`:701-703`).
  2. 버퍼 할당: `out_buffer`(vocab×4B, group_id(5)), `transformer_buffer`(sizeof(*transformer), group_id(0)), `key_buffer`/`value_buffer`(cache_dim×4B, group_id(3)/(4)) (`:707-714`). group_id 번호는 커널 인자 순서와 매핑.
  3. `transformer_buffer.write(...)` → `sync(... BO_TO_DEVICE)`로 전체 모델을 한 번만 디바이스로 전송(`:717-719`).
  4. **첫 토큰**: `run = kernel(transformer_buffer, token, pos, key_buffer, value_buffer, out_buffer)` → `run.wait()` (`:728-729`).
  5. logits 회수: `out_buffer.sync(BO_FROM_DEVICE)` → `read(logits,...)` (`:735-736`).
  6. **이후 토큰 루프**(`:758-796`): `run.set_arg(1,token)`, `set_arg(2,pos)`만 갱신 → `run.start()` → `run.wait()` → logits read → `sample()`. 즉 **무거운 모델·KV캐시 버퍼는 재사용하고, 스칼라 인자만 바꿔 재실행**.
  7. 토큰/초 측정: `time_in_ms()`로 첫 토큰 이후 구간 측정 후 `achieved tok/s` 출력(`:799-804`).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 Autoregressive 토큰 생성 루프 (호스트 주도)

```
prompt → encode(BPE) → tokens[]
for pos in 0..steps:
   set_arg(token,pos) → kernel.start() → kernel.wait()      # FPGA forward 1회
   out_buffer.sync(FROM_DEVICE) → logits[vocab=32000]
   next = (prompt 소진 전? 강제 토큰 : sample(logits))       # argmax/multinomial/top-p
   decode(next) 출력, token=next
```
- 한 step = `forward` 커널 1회 호출 = 12개 디코더 레이어 + 최종 분류기. 토큰당 1회 디바이스 왕복(인자 갱신 + 결과 read).
- 종료: `pos>=steps` 또는 `next==1`(BOS 구분자, `:786`).

### 4.2 커널 내부 1-스텝 데이터플로우 (`forward.cpp`)

```
token → token_embedding_table[token] → x (on-chip)
main_forward_loop (l=0..11):
  rmsnorm(xb,x)            # §3.1
  quantize(xb)→xq
  matmul→q,k,v            # §3.3 int8
  RoPE(q,k)               # §3.4
  KV캐시[pos] ← k,v        # DRAM write
  multihead_attention:    # q·KV → softmax → ·V → xb   §3.5
  quantize→matmul(wo)→xb2; x += xb2     # residual
  rmsnorm→quantize→matmul(w1,w3)→hb,hb2
  SwiGLU(hb,hb2)→hb       # §3.6
  quantize→matmul(w2)→xb; x += xb       # residual2
rmsnorm(x); quantize; matmul(wcls)→out[32000]   # logits
```

### 4.3 메모리 계층

- **DRAM(m_axi)**: `transformer`(전체 int8 가중치, gmem0), `out`(logits, gmem1), `key_cache`/`value_cache`(KV, 별도 master 버퍼). 가중치는 `matmul` 내부에서 **행 단위로만** on-chip에 적재.
- **On-chip (BRAM/레지스터)**: 모든 활성화 텐서(`x,xb,hb,q,k,v,att,xq,hq`)가 `static` + `ARRAY_PARTITION`으로 상주. 행렬곱 입력 벡터/현재 행도 로컬 버퍼.
- KV 캐시 크기: `n_layers*seq_len*kv_dim = 12*1024*768` fp32 ≈ **36MB** → DRAM 상주 필수.

### 4.4 병렬화

- 루프 레벨: `PIPELINE` + `UNROLL`(rmsnorm 64/128, softmax/RoPE/SwiGLU 4~16, residual 16/64) + `ARRAY_PARTITION cyclic`으로 메모리 포트 확보.
- 단, **레이어 간/연산 간 `dataflow`는 없음** → 함수들이 순차 실행. matmul 내부 누산도 현재 직렬(UNROLL 주석화). 따라서 병렬화는 "벡터 elementwise 연산" 위주이고 핵심 GEMV는 아직 부분 최적화 상태.

### 4.5 양자화

- 가중치: 오프라인 int8 그룹(GS=64) 대칭 양자화(`export.py:46`). scale은 fp32.
- 활성화: 런타임에 매 matmul 직전 `quantize()`로 int8 동적 양자화 → int8×int8→int32 누산 → fp32 환산. RMSNorm/softmax/RoPE/SwiGLU/residual은 fp32.

---

## 5. HW/SW 매핑 (벤치 ↔ HLS 커널 ↔ XRT 호스트)

| 구성 요소 | 위치 | 역할 | 측정 목적 |
|---|---|---|---|
| **CPU 참조(양자화)** | `cpu_benchmarks/runq.c` | llama2.c 파생 int8 추론. `matmul`(`:317-336`)·`rmsnorm`(`:282`)·`softmax`(`:297`)·RoPE(`:372`)·SwiGLU(`:448-457`)가 HLS 커널과 동일 수학의 **골든 모델** | 정확도·tok/s 기준선 |
| CPU 참조(비양자화) | `cpu_benchmarks/run.c` | fp32 버전. runq와의 차이는 가중치 int8 양자화 유무뿐(매트멀이 fp32 dot-product) | 양자화 영향 비교 |
| CPU 벤치 변형 | `runq_power_consumption.c`, `runq_latency_256.c`, `runq_latency_1024.c`, `runq_ppl.c`, `run_ppl.c` | runq에서 **출력/EOS 종료를 끄고 전체 루프를 강제 실행**(`runq_power_consumption.c:882,886-887`)해 일정 토큰 수 동안 tok/s 측정(`:891,896-898`). seq 길이 256/1024로 분기, ppl은 perplexity | **CPU 지연/전력/PPL** |
| CPU toolchain | `model.py, train.py, export.py, tokenizer.py, configurator.py, sample.py, tinystories.py` | TinyStories 학습·체크포인트→.bin export·양자화(`export.py:46 quantize_q80`) | 모델 준비 |
| **HLS 커널** | `llama_xrt_kernels/src/forward.cpp` | forward pass 전체를 FPGA에 합성 (§3) | **FPGA 추론 본체** |
| **XRT 호스트** | `llama_xrt/src/llama2.cpp` | xclbin 로드·버퍼 관리·토큰 단위 enqueue·tok/s 측정(`:799-804`) | FPGA 구동/측정 |
| **GPU 벤치** | `gpu_benchmarks/tinyllama2.py` (지연), `tinyllama2_power_consumption.py` (전력) | Meta llama fork로 stories110M(`.pt`) 추론. `gen_length=1024, num_runs=100`(`tinyllama2.py:20-22`). 전력은 **codecarbon `@track_emissions`**(`..._power_consumption.py:18,61`) | **GPU 지연/에너지** |

- **벤치마크의 목적은 명시적으로 전력·지연 비교**다: `gpu_benchmarks/README.md:3` "benchmarking gpu energy and latency metrics for Llama 2", `:6` codecarbon 설치 후 `tinyllama2_power_consumption.py`(에너지)와 `tinyllama2.py`(지연)를 각각 실행. CPU측은 `runq_*` 변형 + `time_in_ms` tok/s. 세 플랫폼(CPU/GPU/FPGA)에서 **동일 TinyLlama2 stories110M**을 돌려 지연·전력을 비교하는 구도.
- 수학적 정합성: HLS `matmul`(그룹 누산 + scale)과 `runq.c matmul`은 동일 알고리즘이므로 runq.c가 FPGA 결과의 검증 기준이 된다.

---

## 6. 빌드 / 실행

- **CPU 벤치**(`cpu_benchmarks/Makefile`): `make runfast`(`-Ofast run.c/runq.c`, `:26-29`), `make runfast_benchmark`(전력/지연/ppl 변형 일괄, `:31-38`), `make runomp`(OpenMP), `make win64`/`win64_benchmarks`(mingw 크로스, `:50-59`). 실행: `./runq model.bin -n <steps> -i "<prompt>"` (옵션 `-t/-p/-s/-n/-i/-z/-m`).
- **GPU 벤치**(`gpu_benchmarks/README.md:6`): Llama2 의존성 설치 → stories110M.pt(HuggingFace karpathy/tinyllamas) → `pip install codecarbon` → `torchrun tinyllama2_power_consumption.py`(에너지), `torchrun tinyllama2.py`(지연).
- **FPGA**: Vitis 프로젝트 2개(`llama_xrt.prj` 호스트, `llama_xrt_kernels.prj` 커널). 타깃 `sw_emu/hw_emu/hw`. 플랫폼 AWS F1 VU9P. 호스트 실행은 `run <checkpoint> -k <xclbin>` 형태(`llama2.cpp:916-919`에서 `-k` = kernelpath). XRT 헤더 `xrt_bo.h/xrt_device.h/xrt_kernel.h`(`llama2.cpp:16-18`).
- 빌드 스크립트(Makefile/tcl/cfg)는 FPGA 측엔 없고 Vitis GUI 프로젝트(`.prj/.cproject/.project`) 기반.

---

## 7. 의존성

- **FPGA**: Vitis / Vitis HLS, XRT(`xrt::device/kernel/bo`), AWS F1 aws-fpga 플랫폼(`xilinx_aws-vu9p-f1_...`). HLS 수학(`sqrtf/expf/cosf/sinf/powf/fabs/round`)은 표준 `<math.h>`(`forward.h:3`) — 전용 HLS math 미사용("TODO: include HLS math package", `forward.cpp:5`).
- **CPU**: gcc/clang, `-lm`, 옵션 OpenMP, Windows는 `win.c`/`win.h`(mmap shim).
- **GPU**: PyTorch(CUDA HalfTensor), `fairscale`(model parallel), Meta `llama` 패키지(`gpu_benchmarks/llama`, third-party), `codecarbon`, `tqdm`, `numpy`. (`requirements.txt` — 데이터로 미열람, README로 충분히 식별).
- 모델 데이터: `stories110M.pt`(GPU), `model.bin`(CPU/FPGA, `export.py` 산출), `tokenizer.bin/.model`.

---

## 8. 강점 / 한계 / 리스크

### 강점
- **단일 커널에 forward 전체를 통합** — 레이어/연산을 개별 커널로 쪼개지 않아 호스트-디바이스 왕복을 토큰당 1회로 최소화(`llama2.cpp` enqueue 루프). 무거운 모델/KV 버퍼를 재사용하고 스칼라 인자만 갱신(`set_arg`).
- **int8 그룹 양자화로 가중치 대역폭·메모리 절감**, 누산은 int32로 정확. CPU(runq.c)와 수학이 동일해 **골든 모델 검증 경로가 명확**.
- 활성화 텐서 전부 on-chip 상주 + `ARRAY_PARTITION`/`PIPELINE`/`UNROLL`로 벡터 연산 병렬화. RMSNorm/softmax/RoPE/SwiGLU는 적절히 파이프라인됨.
- CPU/GPU/FPGA 3-플랫폼 동일 모델 벤치(지연/전력) 구도로 비교 평가 인프라가 갖춰짐.

### 한계
- **핵심 GEMV(`matmul`)가 미최적화**: 내부 `matmul1~4`의 `UNROLL`이 전부 주석 처리(`forward.cpp:238,246,255,260`)되어 곱셈-누산이 직렬. 외부 `i` 루프만 PIPELINE. 매트멀이 "가장 오래 걸리는 함수"인데도 데이터패스 병렬화가 빠져 있어 실효 처리량이 낮을 것.
- **dataflow 미적용**: 레이어/연산 간 task-level 파이프라이닝 없음 → 함수 순차 실행. KV 캐시·가중치 DRAM 접근이 연산과 겹쳐지지 못함.
- **삼각함수 런타임 계산**: RoPE의 `cosf/sinf/powf`를 위치마다 호출(`:330,353`) — FPGA에서 비싼 연산, LUT/사전계산 테이블 미사용(GPU export는 `freqs_cos/sin`을 직렬화하는데(`export.py:118-119`) 커널은 활용 안 함).
- `matmul_old`(`:98-189`)는 합성 시도 흔적이 남은 데드 코드(오타 `PIPEPLINE` 포함). 정리 필요.
- 정확도/수치: fp32 활성화 + 동적 int8 양자화. clamp가 명시적이지 않음(`(int8_t)round(...)`만, `forward.h:57`) — overflow 보호 미흡 가능.

### 리스크
- **자원/타이밍**: `transformer` 전체를 m_axi 단일 번들(gmem0)로 묶음 → 가중치 대역폭 경합. KV 캐시 랜덤 액세스(어텐션 t-루프)로 m_axi 레이턴시 노출.
- **빌드 재현성**: FPGA측 빌드 스크립트(Makefile/cfg) 부재, AWS F1 특정 shell에 고정(`shell-v04261818`). 다른 보드 이식 시 platform 교체 필요.
- 호스트 `generate`에서 `logits` malloc이 루프 밖/안 중복 흔적(`:734,767 주석`) — 메모리 관리 정리 여지.

---

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 — HG-PIPE 계열 + XR 시선추적) 관점 시사점

우리는 **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열)** + **XR 시선추적**을 지향한다. 이 repo는 LLaMA(디코더)지만 **Transformer 블록(LayerNorm/RMSNorm, QKV GEMV, MHA, softmax, FFN, residual)이라는 골격이 ViT 인코더와 거의 동형**이라 재사용 가치가 크다. 다만 설계 철학은 정반대(이 repo=레이턴시 중심 단일 호출, 우리=throughput 중심 파이프라인)임을 전제로 취사선택한다.

### 재사용 가능한 HLS 커널 구조
- **그룹 양자화 GEMV 패턴**(`matmul<N,D>` + `QuantizedTensor` + 행 단위 스트리밍): int8 weight + fp32 그룹 scale + int32 누산 구조를 그대로 ViT의 patch-embed/QKV/MLP 투영에 차용 가능. **단 우리는 내부 UNROLL을 실제로 켜고(여기선 주석화됨) 시스토릭/output-stationary MAC 어레이로 확장**해야 HG-PIPE의 처리량을 낼 수 있다.
- **on-chip 상주 활성화 + `ARRAY_PARTITION cyclic` 전략**(`forward.cpp:288-300`): 활성 텐서를 BRAM에 두고 가중치만 스트리밍하는 분리는 우리 가속기의 메모리 계획 출발점으로 적절.
- **softmax/RMSNorm/SwiGLU 파이프라인 템플릿**(§3.1·3.2·3.6): max-subtract softmax, 1/sum 사전계산, silu 구현은 ViT의 LayerNorm·GELU·softmax 데이터패스 설계에 바로 참고. 우리는 여기에 LUT 기반 exp/GELU와 reduction tree를 추가.
- **반례로 배울 점**: dataflow 미사용·GEMV 직렬화·삼각함수 런타임 계산은 **HG-PIPE의 핵심(전 레이어를 dataflow로 파이프라인, 연산 fold/unfold)** 과 정반대. 우리는 (1) 레이어 간 `#pragma HLS DATAFLOW` + stream, (2) RoPE/위치인코딩·삼각함수의 사전 테이블화, (3) GEMV의 PE 어레이화를 반드시 적용해야 함을 이 repo의 한계가 역으로 시사.

### 재사용 가능한 XRT 호스트 패턴
- **모델 1회 전송 + 스칼라 인자만 set_arg 재실행 루프**(`llama2.cpp:717-796`): 가중치를 디바이스에 한 번 올리고 프레임/토큰마다 가벼운 인자만 바꿔 재시작하는 패턴은, **XR 시선추적의 프레임 스트림 추론**(매 프레임 입력만 갱신, 모델 상주)에 그대로 이식 가능. 우리 시선추적 파이프라인의 호스트 골격으로 채택 가치 높음.
- `xrt::bo` group_id ↔ 커널 인자 매핑, `sync(TO/FROM_DEVICE)` 타이밍, `run.start()/wait()` 비동기 enqueue 구조는 저지연 호스트 루프의 표준 레퍼런스.
- 개선 방향: 우리는 입력 프레임 BO를 더블 버퍼링하고 `wait()` 대신 비동기 콜백/파이프라인으로 호스트-디바이스 오버랩을 추가해야 함(이 repo는 동기 `wait`로 직렬).

### 재사용 가능한 벤치마크 방법론
- **3-플랫폼(CPU/GPU/FPGA) 동일 모델 지연·전력 비교 프레임**(§5): CPU `runq_*`(`time_in_ms` tok/s) + GPU `codecarbon @track_emissions`(에너지) + FPGA 호스트 tok/s. 우리 ViT 가속기/시선추적도 **동일 입력으로 CPU(PyTorch/ONNX)·GPU·FPGA의 지연(ms/frame, FPS)과 전력(W, mJ/inference)을 나란히** 측정하는 동일 구도를 채택하면 논문 평가표를 바로 만들 수 있다.
- **codecarbon 기반 에너지 측정**은 GPU 베이스라인 전력 비교에 그대로 쓸 수 있는 실용 도구. FPGA측은 XRT 전력 API/보드 센서로 대응.
- **골든 모델(runq.c) 대조 검증** 방식: 양자화 커널 출력과 CPU 정밀 참조를 비교해 수치 정확도를 보증하는 흐름은 우리 ViT 양자화 가속기의 검증 파이프라인 설계에 그대로 적용.

---

## 10. 근거 / 한계 표기

### 직접 Read한 파일 (라인 근거 사용)
- `llama_xrt_kernels/src/forward.cpp` (전체, 488줄) — §3 전반
- `llama_xrt_kernels/src/config.h`, `llama_xrt_kernels/src/typedefs.h`, `llama_xrt_kernels/src/forward.h` — 하이퍼파라미터/구조체/양자화
- `llama_xrt/src/llama2.cpp` (전체, 967줄) — 호스트 XRT 흐름
- `llama_xrt/src/config.h`, `forward.h` (커널측과 동일 확인), `typedefs.h`(host run-state 주석)
- `llama_xrt/llama_xrt.prj`, `llama_xrt_kernels/llama_xrt_kernels.prj` — 타깃 디바이스/커널 인자
- `cpu_benchmarks/Makefile`, `cpu_benchmarks/runq.c`(Grep), `cpu_benchmarks/runq_power_consumption.c`(부분), `cpu_benchmarks/export.py`(Grep)
- `gpu_benchmarks/tinyllama2.py`(부분), `gpu_benchmarks/tinyllama2_power_consumption.py`(전체), `gpu_benchmarks/README.md`

### 제외(이름만 언급, 미분석)
- `cpu_benchmarks/benchmark_stories/*.txt`(스토리 텍스트 수백 개) — 입력 데이터
- `gpu_benchmarks/llama/`(Meta llama 원본 third-party)
- `*.bin / *.model / *.chk / *.csv / log_*.txt / *.DS_Store` — 데이터/로그
- `cpu_benchmarks/run*.c`는 Karpathy `llama2.c` 파생 **CPU 참조구현**으로 다룸. runq(양자화) vs run(비양자화)의 차이는 "가중치 int8 그룹 양자화 + matmul의 int8 dot-product 경로" 유무뿐(나머지 토크나이저/샘플러/생성 루프 동일).

### 미완 분석 부분과 이유
- **합성 결과(자원/주파수/레이턴시 리포트)**: HLS 합성 산출물(`*.rpt`, xclbin)이 repo에 없어(데이터 제외 + 미생성) **실측 성능·자원 수치는 코드 구조 기반 정성 추정에만 의존**. 정량 평가 불가.
- **FPGA 빌드 절차 정밀화**: Vitis GUI 프로젝트(`.prj`)만 있고 v++ 빌드 스크립트/cfg가 없어 정확한 합성 명령·connectivity(sp/slr 배치)는 미확인.
- `cpu_benchmarks/model.py`/`train.py`/`tokenizer.py` 등 학습·전처리 toolchain은 §5에서 역할만 식별(원본 llama2.c 파생으로 자명)했고 라인 단위 분석은 생략 — 본 분석의 초점(HLS 커널/XRT/벤치 매핑)에서 벗어남.
- `runq_latency_256/1024.c`, `*_ppl.c`는 `runq_power_consumption.c`와 동일 골격(출력 억제 + 강제 루프 + tok/s)에 seq 길이/PPL 계산만 다른 변형으로 판단(파일명·Makefile `:31-38`·power 변형 본문 근거). 각 파일 전수 라인 비교는 미수행.
