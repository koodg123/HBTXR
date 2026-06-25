# Transformer-Accel / `submission` 코드베이스 정밀 분석

분석 대상: `REF/Transformer-Accel/submission`
분석 일자: 2026-06-20
근거 표기 규칙: 라인 근거 있는 사실은 `파일:라인` 명시 / 불확실한 것은 "추정" 또는 "확인 불가" 명시

---

## 1. 개요

이 repo는 **Karpathy의 `llama2.c` (int8 그룹 양자화 추론) 포트를 Xilinx Vitis HLS + XRT로 옮겨 AWS F1 FPGA(VU9P)에서 가속**하는 프로젝트다. 핵심은 단일 HLS 커널 `forward()`로, Transformer(LLaMA2 계열) 한 토큰의 전체 forward pass(임베딩 → 12개 레이어의 RMSNorm/QKV matmul/RoPE/멀티헤드 어텐션/FFN(SwiGLU) → 최종 RMSNorm → classifier)를 FPGA에서 수행한다. 호스트(`llama2.cpp`)는 체크포인트 mmap·토크나이저(BPE)·샘플러(argmax/multinomial/top-p)를 담당하고, XRT 버퍼로 커널을 토큰마다 호출한다.

- 정체 확정 근거:
  - README 제목 "HLS Implementation of Llama 2" (`README.md:1`), AWS FPGA Developer AMI / EC2 z1d / F1 인스턴스 빌드 절차 (`README.md:4-26`).
  - 커널 시그니처가 `extern "C" void forward(Transformer<...>*, token, pos, key_cache, value_cache, out)` 로 FPGA 커널 형태 (`llama_xrt_kernels/src/forward.cpp:269`).
  - 호스트가 `xrt/xrt_bo.h`, `xrt_device.h`, `xrt_kernel.h` 사용 (`llama_xrt/src/llama2.cpp:16-18`), `xrt::device(0)`, `device.load_xclbin`, `xrt::kernel(device,uuid,"forward")` (`llama2.cpp:701-703`).
  - 모델 형상: dim=768, hidden_dim=2048, n_layers=12, n_heads=12, n_kv_heads=12, vocab=32000, seq_len=1024, GS(group size)=64 (`llama_xrt_kernels/src/config.h:4-11`). 이는 stories110M(TinyLlama 110M)급. CPU/GPU 벤치마크가 `stories110M.pt`/`stories110M.bin`을 쓰는 것과 일치 (`gpu_benchmarks/tinyllama2.py:40`, `cpu_benchmarks/README.md:8`).

- 비교 벤치마크 2종 포함:
  - **CPU**: Karpathy `runq.c` 기반 int8 양자화 스칼라 추론 + 지연/전력 측정 변형 (`cpu_benchmarks/`).
  - **GPU**: Meta의 `llama` 스택 위에서 stories110M을 fp16으로 추론, codecarbon으로 탄소/전력 측정 (`gpu_benchmarks/`).

### HLSTransformation(형제 repo)과의 관계 — 판정: **유사(거의 동일 계열) / 정밀 diff는 확인 불가**

작업 지시상 같은 부모의 `HLSTransformation` repo가 형제이며 둘 다 `llama_xrt / gpu_benchmarks / cpu_benchmarks` 구조를 가진다고 명시됐다. 본 `submission`도 정확히 그 3분할 구조 + Vitis 프로젝트 4종(`llama_xrt`, `llama_xrt_kernels`, `llama_xrt_system`, `llama_xrt_system_hw_link`)을 가진다. 따라서 **동일 프로젝트의 제출본(submission) 스냅샷으로 강하게 추정**된다. 다만 `HLSTransformation` 경로가 본 세션의 연결 폴더 밖이라 Glob/Read로 직접 대조가 **불가**했다(파일별 라인 동일성은 확인 불가). 결론: 구조·기능·모델 형상 기준 **동일/유사 계열**, `submission`은 별도 디렉토리이므로 본 문서로 자체 분석을 완결한다.

---

## 2. 디렉토리 구조

### 2.1 자체 소스(분석 대상)

```
submission/
├─ README.md                         # 빌드/실행 절차 (AWS F1)
├─ llama_xrt_kernels/                # HLS 커널 프로젝트
│   └─ src/
│       ├─ forward.cpp               # ★핵심 HLS 커널 (forward + rmsnorm/softmax/matmul/quantize)
│       ├─ forward.h                 # 커널 선언 + dequantize/quantize 템플릿
│       ├─ config.h                  # 모델 하이퍼파라미터(constexpr)
│       └─ typedefs.h                # Config/QuantizedTensor/TransformerWeights/Transformer 구조체
├─ llama_xrt/                        # 호스트(SW) 프로젝트
│   └─ src/
│       ├─ llama2.cpp                # ★호스트: mmap 가중치/토크나이저/샘플러/XRT 호출/생성 루프
│       ├─ forward.h, config.h, typedefs.h  # 커널과 공유 헤더(동일 사본)
│       └─ tokenizer.bin            # (생성물, 이름만)
├─ llama_xrt_system/                 # Vitis 시스템 프로젝트 메타(.sprj 등)
├─ llama_xrt_system_hw_link/         # HW 링크 설정(.prj): CU/메모리 바인딩
├─ cpu_benchmarks/                   # CPU 비교 벤치마크 (Karpathy runq.c 계열)
│   ├─ runq.c                        # 기준 int8 양자화 CPU 추론
│   ├─ runq_latency_1024.c / runq_latency_256.c  # 지연 측정 변형
│   ├─ runq_power_consumption.c      # 전력/탄소 측정 변형
│   ├─ model.py / train.py / export.py(추정) / tokenizer.py / tinystories.py
│   ├─ test.c / test_all.py / benchmark_results.py
│   ├─ win.c / win.h                 # Windows 호환 레이어(mman/clock)
│   ├─ accuracy_benchmark*.bash      # perplexity 벤치 스크립트
│   └─ build_msvc.bat, requirements.txt, README.md
├─ gpu_benchmarks/                   # GPU 비교 벤치마크 (Meta llama 스택)
│   ├─ tinyllama2.py                 # tok/s 측정(100 runs)
│   ├─ tinyllama2_power_consumption.py # codecarbon 전력/탄소 측정
│   ├─ example_text_completion.py / example_chat_completion.py
│   ├─ setup.py / requirements.txt / download.sh
│   └─ llama/__init__.py             # (Meta llama 패키지 진입점)
└─ LICENSE, .gitignore
```

### 2.2 제외(third-party / 원본 / 생성물 / 문서) — 이름만 언급

- **Meta LLaMA 원본 스택**: `gpu_benchmarks/llama/` (Meta `Llama`, `ModelArgs`, `Transformer`, `Tokenizer` import; `tinyllama2.py:1-9`), `gpu_benchmarks/{MODEL_CARD.md, USE_POLICY.md, CODE_OF_CONDUCT.md, CONTRIBUTING.md, UPDATES.md, LICENSE}`.
- **Karpathy `runq.c` 원본 계열**: `cpu_benchmarks/runq.c`(원본 거의 그대로) 및 그 파생 .c 들 — 자체 수정분만 §3.5에서 분석.
- **생성물/바이너리**: `cpu_benchmarks/runq.exe`, `cpu_benchmarks/runq_latency_1024`(ELF), `cpu_benchmarks/tokenizer.bin`, `llama_xrt/src/tokenizer.bin`, `gpu_benchmarks/{tokenizer.model, tokenizer_checklist.chk, .DS_Store}`.
- **로그/측정 결과**: `gpu_benchmarks/{log_256.txt, log_1024.txt, log_4096.txt, emissions.csv}`.
- **VCS/IDE 메타**: `.git/`, 각 프로젝트의 `.project/.cproject/.settings/*`, `*.prj/.sprj`(단 HW 링크 .prj는 §5에서 매핑 근거로 사용).

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

### 3.0 자료형·메모리 레이아웃 (typedefs.h / config.h)

- `Config`: dim/hidden_dim/n_layers/n_heads/n_kv_heads/vocab_size/seq_len/GS 8개 int 필드 (`typedefs.h:21-31`).
- **`QuantizedTensor<SIZE>`**: `int8_t q[SIZE]` + `float s[SIZE]` (`typedefs.h:33-38`). 주의: 스케일 배열 `s`가 `SIZE` 전체 크기로 선언됨(실제 그룹 수 SIZE/GS만 유효). 즉 컴파일타임 정적 배열로, 동적 포인터(원본 llama2.c의 `int8_t* q; float* s;`)와 달리 **모든 가중치를 정적 멤버로 구조체에 in-line**한다 → FPGA에서 단일 `m_axi` 슬레이브로 통째로 매핑하기 위함(추정).
- **`TransformerWeights<...>`** (`typedefs.h:41-64`): 전 가중치를 정적 배열로 보유. `q_tokens`(임베딩), `token_embedding_table`(dequant 사본), `rms_att_weight`/`rms_ffn_weight`/`rms_final_weight`, `wq/wk/wv/wo`(레이어별 QuantizedTensor 배열), `w1/w2/w3`(FFN), `wcls`(classifier). dim==n_heads*head_size 가정 명시(`typedefs.h:51`).
- **`Transformer<...>`** (`typedefs.h:69-78`): `Config config` + `TransformerWeights weights`. 원본의 mmap fd/data/file_size 상태는 주석 처리됨(`typedefs.h:74-77`).
- 형상 상수는 `kernels/src/config.h`와 `llama_xrt/src/config.h`가 **완전히 동일**(둘 다 dim=768…GS=64; `config.h:4-22` 양쪽 동일) → 커널/호스트가 같은 ABI 공유.

### 3.1 양자화/역양자화 (forward.h)

- **`dequantize<S>`** (`forward.h:6-13`): `x[i] = q[i] * s[i/GS]`. 그룹당 1개 스케일을 GS개 원소에 적용. 단순 루프, HLS pragma 없음(호스트에서 임베딩 dequant에 주로 사용).
- **`quantize<S>`** (`forward.h:15-64`): 그룹(GS=64)별 대칭 양자화.
  - `num_groups = S/64` 하드코딩, `Q_MAX=127` (`forward.h:18-19`).
  - 그룹 단위 max-abs 탐색 → `scale = wmax/127` (`forward.h:36-48`) → `q = round(x/scale)` (`forward.h:52-58`).
  - HLS 최적화: `scale_buffer`/`quantized_buffer` `ARRAY_PARTITION cyclic`(`forward.h:23-24`), 외곽 그룹 루프 `UNROLL factor=8 + PIPELINE`(`forward.h:30-31`), 내부 루프 `PIPELINE`(`forward.h:39,55`). 결과를 `memcpy`로 `qx->q`, `qx->s`에 기록(`forward.h:62-63`).
  - 한계: clamp가 round만 있고 [-127,127] 포화가 명시 없음(`forward.h:57`) — 정상 입력에선 |q|≤127 보장되나 경계 round에서 ±128 가능성(추정, 미세 정확도 이슈).

### 3.2 HLS 커널 빌딩블록 (forward.cpp)

#### (a) `rmsnorm<S>` (`forward.cpp:13-49`)
- 표준 RMSNorm: 제곱합 → `ss = 1/sqrt(mean+1e-5)` → `o = weight * (ss*x)`.
- 입력/가중치/출력을 로컬 버퍼에 `memcpy` 후 처리(`forward.cpp:24-25,48`).
- HLS: `x_buff cyclic factor=128`, `weight_buff/out_buff factor=64` (`forward.cpp:21-23`); 제곱합 루프 `PIPELINE + UNROLL 128`(`forward.cpp:30-31`), 정규화 루프 `PIPELINE + UNROLL 64`(`forward.cpp:42-43`).
- 주의: `ss += x_j*x_j` 가 UNROLL 128과 함께 단일 누산기에 더해짐 → 부동소수 누산 의존성으로 II가 펼쳐진 만큼 줄지 않을 수 있음(reduction 트리 미구성)(추정).

#### (b) `softmax<MAXSIZE>` (`forward.cpp:51-96`)
- max 탐색 → `exp(x-max)` → 합 → 정규화. 4개 루프.
- 가변 길이 `size`에 대해 `loop_tripcount min=0 max=257 avg=129` 지정(`forward.cpp:60,73,83,91`) — pos+1(최대 seq 길이 관련, 어텐션용). `MAXSIZE=257`로 인스턴스화(`forward.cpp:406`).
- exp/norm 루프 `PIPELINE + UNROLL 16`(`forward.cpp:74-75,92-93`). 합산 루프는 단일 누산(`forward.cpp:80-85`).

#### (c) `matmul<N,D>` (현행, `forward.cpp:191-267`) — 핵심 연산
- 의미: `W(D,N) @ x(N,) -> xout(D,)`, **입력 x와 가중치 W 모두 int8 그룹 양자화** 상태에서 정수 MAC 후 그룹 스케일 복원.
- 입력 벡터를 정적 `x_buffer[N]`, 스케일 `xs_buffer[N/GS]`에 적재(`forward.cpp:204-222`), 각각 `cyclic factor=16/4`(`forward.cpp:208-209`).
- 출력 행 루프 `for i in D` 에 `#pragma HLS PIPELINE`(`forward.cpp:227`). 행마다 로컬 `w_buffer[N]`/`ws_buffer[N/GS]`를 `cyclic factor=32`로 파티션(`forward.cpp:229-232`)하고 wq/ws에서 적재(`forward.cpp:235-248`).
- 핵심 누산: GS(=64) 그룹 단위로 `int32_t ival += x_buffer[j+k]*w_buffer[j+k]` (`forward.cpp:257-262`), 그룹 끝에서 `val += ival * ws_buffer[j/GS] * xs_buffer[j/GS]` (`forward.cpp:263`). 즉 **정수 MAC → float 스케일 보정**의 그룹별 dot product.
- **주목**: `matmul3`/`matmul4` 내부 루프의 `#pragma HLS UNROLL`이 전부 주석 처리됨(`forward.cpp:255,260`) → 현행 matmul은 사실상 미언롤(스칼라) 파이프라인. 가장 비용 큰 연산(`forward.cpp:194-195` 주석 "by far the most amount of time")인데 병렬화 pragma가 비활성 → 성능 병목 가능성(추정, 라인 근거 있음).

#### (d) `matmul_old<N,D>` (`forward.cpp:98-189`) — 사용 안 함(레거시)
- 전체 W를 `w_buffer[N*D]`로 한꺼번에 적재하고 `UNROLL factor=128/32`로 공격적 언롤(`forward.cpp:114-147`), 내부 `matmul3/4`에 `UNROLL`(`forward.cpp:177,182`).
- 단, 출력 루프에 `#pragma HLS PIPEPLINE`(오타, 무효) (`forward.cpp:152`), `w_buffer[N*D]`는 거대 BRAM 요구 → 합성 비현실적. 현행 `matmul`로 대체됨(호출처 없음).
- 시사: 저자가 "전체 적재 + 풀 언롤"(자원 폭발) ↔ "행 단위 적재 + 파이프라인"(자원 절약) 사이에서 후자를 택한 설계 trade-off의 흔적.

### 3.3 `forward()` 최상위 커널 (forward.cpp:269-487) — 단일 토큰 전체 forward

시그니처/인터페이스:
- `extern "C" void forward(Transformer*, int token, int pos, float key_cache[...], float value_cache[...], float* out)` (`forward.cpp:269`).
- AXI 인터페이스: `transformer` → `m_axi ... bundle=gmem0`, `out` → `m_axi ... bundle=gmem1` (`forward.cpp:271-272`). key_cache/value_cache는 명시 pragma 없음(기본 m_axi로 추정; HW 링크 .prj에서 master로 바인딩됨 §5).

내부 상태 버퍼 및 파티션(`forward.cpp:277-300`):
- `x/xb/xb2`(dim), `hb/hb2`(hidden_dim), `xq`(dim 양자화), `hq`(hidden 양자화), `q`(dim), `k/v`(kv_dim), `att`(n_heads*seq_len). 전부 `static`(BRAM 상주).
- 다수 `ARRAY_PARTITION cyclic factor=UNROLL_FACTOR(=16)` (`forward.cpp:288-300`).
- 상수: `kv_dim=(dim*n_kv_heads)/n_heads`=768, `kv_mul=n_heads/n_kv_heads`=1(MHA, MQA 아님), `head_size=dim/n_heads`=64 (`forward.cpp:301-303`).

연산 흐름 (레이어 루프 `main_forward_loop`, `forward.cpp:309-479`):
1. **임베딩 로드**: `memcpy(x, token_embedding_table + token*dim, ...)` (`forward.cpp:306`).
2. **Attn RMSNorm**: `rmsnorm<dim>(xb, x, rms_att_weight + l*dim)` (`forward.cpp:313`).
3. **QKV**: `quantize(&xq, xb, GS)` 후 `matmul<dim,dim>(q,...wq)`, `matmul<dim,kv_dim>(k,...wk)`, `matmul<dim,kv_dim>(v,...wv)` (`forward.cpp:316-319`).
4. **RoPE**: `rotation1`(i<kv_dim: q,k 동시 회전, `UNROLL 16+PIPELINE`, `forward.cpp:323-346`), `rotation2`(kv_dim≤i<dim: q만, `PIPELINE`, `forward.cpp:347-363`). freq/cos/sin을 `powf/cosf/sinf`로 매 원소 계산(`forward.cpp:330-333`) — 미리계산 테이블 없음(자원↓ 지연↑, 추정).
5. **KV 캐시 쓰기**: `loff=l*seq_len*kv_dim`, `memcpy(key_cache_row,k,...)`, `value_cache_row` (`forward.cpp:365-370`).
6. **멀티헤드 어텐션** (`multihead_attention`, `forward.cpp:375-430`):
   - 헤드별로 t=0..pos 점수 계산: `score = Σ q[i+q_off]*key_cache[i+key_off]` (head_size 내부 `#pragma HLS unroll`, `forward.cpp:395-399`), `/sqrt(head_size)` (`forward.cpp:400`).
   - `softmax<257>(att+att_off, pos+1)` (`forward.cpp:406`).
   - 가중합: `xb[i+xb_off] += a*value_cache[i+v_off]` (acc, 내부 `unroll`, `forward.cpp:412-429`).
   - key/value 오프셋에 `h/kv_mul` 사용(MQA 일반화, 여기선 kv_mul=1) (`forward.cpp:392,420`).
7. **Attn 출력 + 잔차**: `quantize(&xq,xb)` → `matmul<dim,dim>(xb2,...wo)` (`forward.cpp:433-434`), `x[i]+=xb2[i]` (`residual`, `UNROLL 64`, `forward.cpp:437-442`).
8. **FFN RMSNorm**: `rmsnorm<dim>(xb, x, rms_ffn_weight+l*dim)` (`forward.cpp:445`).
9. **FFN(SwiGLU)**: `quantize(&xq,xb)` → `matmul<dim,hidden_dim>(hb,...w1)`, `matmul<dim,hidden_dim>(hb2,...w3)` (`forward.cpp:449-451`); `swi_glu` 루프: `val = silu(hb[i])*hb2[i]` (silu=`x*sigmoid(x)`, `UNROLL 4+PIPELINE`, `forward.cpp:454-465`); `quantize(&hq,hb)` → `matmul<hidden_dim,dim>(xb,...w2)` (`forward.cpp:469-470`).
10. **FFN 잔차**: `x[i]+=xb[i]` (`residual2`, `UNROLL 16`, `forward.cpp:473-478`).
11. **최종 RMSNorm + classifier**: `rmsnorm<dim>(x,x,rms_final_weight)` (`forward.cpp:482`), `quantize(&xq,x)` → `matmul<dim,vocab_size>(out,...wcls)` (`forward.cpp:485-486`). out=32000 logits.

설계 특징:
- **토큰 단위 호출** 구조(autoregressive 1-token-per-call). KV 캐시는 호스트가 보유한 외부 DDR 버퍼(`forward.cpp:269` 인자), 커널은 pos에 읽고/쓴다 → 호스트 루프가 pos를 증가시키며 재호출(§3.4).
- 전 가중치를 `transformer` 단일 AXI 슬레이브로 받음 → forward 1회 = 110M급 가중치를 (정적 멤버를 통해) 접근. 단, 가중치를 BRAM에 캐시하지 않고 매 matmul마다 gmem에서 적재(`matmul`의 `w_buffer` 적재가 `wq` 포인터 = gmem) → 대역폭 바운드일 가능성(추정).

### 3.4 호스트 (llama2.cpp) 정밀 분석

가중치 적재:
- **`init_quantized_tensors<SIZE>`** (`llama2.cpp:79-95`): 파일 포인터에서 int8 q(size_each) → float s(size_each/GS) 순으로 `memcpy`, 포인터 전진. 양자화 직렬 포맷 해석.
- **`memory_map_weights<...>`** (`llama2.cpp:97-133`): fp32 rmsnorm 가중치 먼저 → q_tokens → `dequantize`로 임베딩 테이블 복원(`llama2.cpp:114`) → wq/wk/wv/wo → w1/w2/w3 → shared_classifier면 wcls=q_tokens 공유(`llama2.cpp:125-132`).
- **`read_checkpoint<...>`** (`llama2.cpp:135-208`): magic `0x616b3432`("ak42") 검증(`:150`), **version==2 강제**(`:161-165`), header 256B(`:166`), `mmap(PROT_READ)`(`:195`)로 파일 매핑 후 헤더 스킵하고 `memory_map_weights` 호출(`:201-202`), 이후 `munmap`(`:206`). → quantized v2 export 포맷 전용.

토크나이저(BPE, `llama2.cpp:218-491`):
- `build_tokenizer`(vocab/score 적재, byte_pieces 초기화 `:241-288`), `decode`(BOS 후 선행 공백 strip, `<0x..>` raw byte 처리 `:301-317`), `encode`(UTF-8 코드포인트 단위 lookup → byte_fallback +3 → score 기반 best-pair greedy merge `:350-491`). 원본 llama2.c BPE와 동일 로직.

샘플러(`llama2.cpp:497-666`):
- `sample_argmax`(`:512`), `sample_mult`(CDF, `:528`), `sample_topp`(nucleus: cutoff 프루닝 + qsort + 누적 truncate, `:555-603`), xorshift RNG(`:620-631`), `sample` 디스패치(temp==0→argmax, else softmax 후 topp/mult, `:633-666`).

XRT 실행 + 생성 루프(`generate<...>`, `llama2.cpp:681-807`):
- 프롬프트 encode(`:693`).
- 커널 로드: `xrt::device(0)`→`load_xclbin(kernelpath)`→`xrt::kernel(...,"forward")` (`:701-703`).
- 버퍼 할당: `out_buffer`(vocab*4B, group_id 5), `transformer_buffer`(sizeof transformer, group_id 0), `key_buffer`/`value_buffer`(cache_dim*4B, group_id 3/4) (`:707-714`). cache_dim=`n_layers*seq_len*kv_dim` (`:709`).
- `transformer_buffer.write(...)` → `sync(TO_DEVICE)` (`:717-719`).
- **첫 실행**: `run = kernel(transformer_buffer, token, pos, key_buffer, value_buffer, out_buffer)` → `run.wait()` (`:728-729`), `out_buffer.sync(FROM_DEVICE)` → `read(logits)` (`:735-736`).
- **메인 루프**(`while pos<steps`, `:758-796`): `run.set_arg(1,token)`, `set_arg(2,pos)`, `start()`, `wait()` → logits 읽기 → 프롬프트 forcing 또는 sample → decode/print. next==BOS(1)면 종료(`:786-789`). 인자 0/3/4/5(버퍼)는 재바인딩 안 함(첫 호출 바인딩 유지) → **KV 캐시가 호출 간 persist**.
- tok/s 측정: `time_in_ms`로 첫 토큰 이후 측정(`:756,800-804`).

`main`(`llama2.cpp:843-967`): poor-man argparse(`-t/-p/-s/-n/-i/-z/-m/-y/-k`, `:868-924`; `-k`=kernel xclbin 경로 `:916-918`), `build_transformer`/`build_tokenizer`/`build_sampler` 후 `generate` 호출(`:937-953`). mode="generate"만 구현(chat 미구현, `:951-959`).

### 3.5 CPU 벤치마크 (cpu_benchmarks/)

- `runq.c`: Karpathy `llama2.c`의 int8 양자화 추론 원본(동적 포인터 `QuantizedTensor{int8_t* q; float* s;}` `runq_latency_1024.c:35-39`, 글로벌 `int GS` `:19`). HLS판과 알고리즘 동치이나 **자료구조가 포인터/calloc 기반**(FPGA판은 정적 in-line) — 두 코드의 본질 차이.
- `runq_latency_256.c`/`runq_latency_1024.c`: 생성 길이 256/1024 고정 + 지연 측정 변형(추정: steps 하드코딩 및 타이밍 출력 추가).
- `runq_power_consumption.c`: 동일 추론 + codecarbon 연동/전력 측정용 변형(`README.md:5-7`의 `pip install codecarbon`/`emissions.csv` 절차와 대응).
- 보조: `model.py`/`train.py`/`tokenizer.py`/`tinystories.py`(학습·토크나이저·데이터셋, Karpathy 계열), `test.c`/`test_all.py`/`benchmark_results.py`(검증/집계), `win.c`/`win.h`(Windows에서 mmap/clock 대체), `accuracy_benchmark*.bash`(양자화/비양자화 perplexity).
- 결론: CPU 측은 **비교 기준선**이며 본 repo의 자체 기여(HLS)와는 독립. 자체 수정은 latency/power 측정 hook에 국한(추정).

### 3.6 GPU 벤치마크 (gpu_benchmarks/)

- `tinyllama2.py`: Meta `llama` 스택(`Llama/ModelArgs/Transformer/Tokenizer`)으로 stories110M(`.pt`)을 fp16 CUDA HalfTensor로 로드(`:40-53`), `gen_length=1024`, `num_runs=100`(`:21-22`)로 `text_completion` 반복하며 tok/s 평균·표준편차 산출, `log_{len}.txt` 저장(`:60-84`).
- `tinyllama2_power_consumption.py`: 동일 구성에 `@track_emissions()`(codecarbon) 래핑으로 전력/탄소 측정(`:18,61-79`).
- `example_*_completion.py`, `download.sh`, `setup.py`는 Meta 원본 유틸. 결론: GPU도 **비교 기준선**(fp16, 비양자화)으로, HW(FPGA)·CPU(int8) 대비 정확도/전력/속도 3축 비교용.

---

## 4. 데이터 플로우

```
[checkpoint .bin(v2,int8 group-quant)] --mmap--> memory_map_weights --> Transformer{Config, Weights} (호스트 RAM, 정적)
[tokenizer.bin] --> Tokenizer (BPE vocab/scores)
prompt --encode--> prompt_tokens[]

(호스트) transformer_buffer.write + sync(TO_DEVICE)   # 전 가중치 1회 DMA
loop pos = 0..steps-1:
    set_arg(token,pos); run.start(); run.wait()        # 커널 forward 1토큰
        (커널) embed -> [×12 layers: RMSNorm->QKV matmul->RoPE->KVcache->MHA->Wo+res->RMSNorm->FFN(SwiGLU)->res]
               -> final RMSNorm -> wcls matmul -> out[vocab]
    out_buffer.sync(FROM_DEVICE).read(logits)
    next = (prompt forcing) ? prompt_tokens[pos+1] : sample(logits)  # argmax/mult/topp
    decode(next) -> stdout
```

- 가중치 DMA는 루프 밖 1회(`llama2.cpp:717-719`). 토큰마다 token/pos만 갱신(`:760-761`). KV 캐시는 device 버퍼에 누적(호출 간 유지) → autoregressive incremental decoding.
- logits만 FROM_DEVICE 동기화(vocab*4B=128KB/토큰) (`:735,769-770`).

---

## 5. HW/SW 매핑

| 항목 | SW(호스트, llama_xrt) | HW(커널, llama_xrt_kernels) |
|---|---|---|
| 가중치 적재/mmap/dequant | `read_checkpoint`/`memory_map_weights` (llama2.cpp:135-208) | — |
| 토크나이즈(BPE) | `encode`/`decode` (llama2.cpp:350-491) | — |
| 샘플링(argmax/topp) | `sample*` (llama2.cpp:512-666) | — |
| forward(임베딩→레이어→logits) | XRT 호출만 (llama2.cpp:728,760-763) | `forward()` (forward.cpp:269-487) |
| RMSNorm/Softmax/Matmul/RoPE/MHA/SwiGLU | — | forward.cpp 내부 함수/루프 |
| 양자화 | 가중치는 오프라인(export), 임베딩 dequant 호스트 | 활성값 `quantize` (forward.h:15-64) |

- 가속 경계: **"한 토큰 전체 forward"가 단일 CU(`forward_1`)로 오프로드**. HW 링크 설정에서 CU 인자 마스터/메모리 바인딩 확인 — `transformer/key_cache/value_cache/out`이 `master="true"`(AXI master), `token/pos`는 스칼라 (`llama_xrt_system_hw_link.prj:6-14, 64-73`).
- 플랫폼: **AWS F1, Xilinx VU9P** (`...hw_link.prj:2` platform=`xilinx_aws-vu9p-f1_shell-v04261818_201920_3`). 빌드 타깃 sw_emu/hw_emu/hw 3종(`.prj:3,33,61`).
- 메모리 인터페이스: `transformer→gmem0`, `out→gmem1` (`forward.cpp:271-272`); key/value_cache는 .prj에서 master 바인딩(기본 gmem). 단일 슬레이브로 전 가중치 노출.
- **주의(설정 불일치)**: hw_link.prj의 Emulation-HW `lastBuildOptions`(`:48-59`)는 CU 인자에서 key_cache/value_cache가 빠진 4-인자 형태로 기록됨(다른 config는 6-인자). 빌드 이력 잔재로 보이며 실제 커널 시그니처(6-인자, forward.cpp:269)와 어긋남 → 재빌드 시 혼동 가능(라인 근거 있음, 영향은 추정).

---

## 6. 빌드·실행

- **HW 빌드**: AWS FPGA Developer AMI 세팅 → Vitis IDE에서 프로젝트 오픈 → SW Emulation으로 검증 → System Project Debug 런 구성에 인자 `weights.bin -z tokenizer.bin -t 0.8 -n 256 -i "{prompt}" -k` (`README.md:9-14`). HW 빌드 ~12시간(`README.md:17`), `.xclbin`→AWS AFI→`.awsxclbin` 변환 후 F1에 복사(`README.md:18-19`).
- **런타임**: `source vitis_setup.sh`/`vitis_runtime_setup.sh`, devtoolset-9/g++-9 필요(`README.md:22-27`).
- **호스트 컴파일**: `g++ -O3 -std=c++17 src/llama2.cpp -o llama2 -I$XILINX_XRT/include -L$XILINX_XRT/lib -lxrt_coreutil -lpthread -lrt` (`README.md:30`).
- **실행**: `./llama2 {weights} -z {tokenizer} -t {temp} -n {steps} -i {prompt} -k {xclbin}` (`README.md:33`, argparse `llama2.cpp:868-924`).
- **CPU 벤치**: codecarbon 설치 후 벤치 스크립트, `stories110M.bin` 다운로드/`export.py`로 quantized 생성(`cpu_benchmarks/README.md:5-8`), MSVC는 `build_msvc.bat`.
- **GPU 벤치**: `requirements.txt`/`setup.py`, `tinyllama2.py` 실행(stories110M.pt + tokenizer.model 필요).

---

## 7. 의존성

- **HW/호스트**: Xilinx Vitis HLS, XRT(`xrt_bo/device/kernel`), AWS FPGA SDK(F1/VU9P xpfm), g++-9/C++17, libxrt_coreutil. HLS math: `math.h`의 `sqrtf/expf/powf/cosf/sinf/fabs/round`(`forward.cpp`, `forward.h`) — 단 `typedefs.h:4`, `forward.cpp:5` 주석상 "HLS types/math 패키지로 교체 TODO" 미완료(ap_int 등 미사용, float 연산).
- **CPU 벤치**: C(gcc/MSVC), codecarbon, `win.h`(Windows mmap/clock 시뮬), Python(model/train/tokenizer).
- **GPU 벤치**: PyTorch(CUDA), fairscale(model parallel), Meta `llama` 패키지, tqdm, numpy, codecarbon.

---

## 8. 강점 · 한계

### 강점
- LLaMA2 전체 forward를 **단일 자기완결 HLS 커널**로 구현(임베딩~classifier, RMSNorm/RoPE/MHA/SwiGLU/그룹양자 matmul 전부 포함) — 라인 근거 명확(`forward.cpp:269-487`).
- 호스트는 표준 llama2.c 자산(BPE/샘플러/mmap) 재사용으로 신뢰성↑, FPGA는 연산만 담당하는 깔끔한 SW/HW 분리(`llama2.cpp` vs `forward.cpp`).
- CPU(int8)·GPU(fp16) **3-way 비교 벤치마크**(속도·전력/탄소) 동봉 → 가속 효과 정량화 프레임 보유.
- 그룹 양자화(GS=64) 정수 MAC + 그룹 스케일 복원으로 BRAM/대역폭 절감 의도(`forward.cpp:257-263`).

### 한계
- **현행 `matmul`의 핵심 내부 루프 UNROLL pragma가 주석 처리**(`forward.cpp:255,260`) → 가장 무거운 연산이 미병렬. 성능 미최적(라인 근거).
- 가중치를 BRAM에 상주 캐시하지 않고 매 matmul마다 gmem 포인터 적재(`forward.cpp:235-248`) → 토큰마다 전 가중치 재독출, **메모리 대역폭 바운드** 가능성(추정).
- RoPE의 cos/sin/powf를 매 호출 런타임 계산(`forward.cpp:330-333,352-356`) — LUT/테이블화 미적용(자원↓ 지연↑).
- `matmul_old` 레거시 잔존, `PIPEPLINE` 오타(`forward.cpp:152`), hw_link.prj의 인자 불일치(`...hw_link.prj:48-59`), `quantize` 포화 클램프 부재(`forward.h:57`) 등 정리 미완 흔적.
- chat 모드 미구현(`llama2.cpp:956-959`), 단일 토큰/단일 CU(배치·다중 CU 병렬 없음).
- HLS 고정소수/ap_int 미도입(전부 float; `typedefs.h:4` TODO) → DSP/면적 최적 여지 큼.

---

## 9. 우리 프로젝트(PRJXR-HBTXR) 시사점

우리 프로젝트는 **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적(eye-tracking)** 방향으로 추정된다. 본 repo와의 연결점:

1. **Transformer 빌딩블록 HLS 레퍼런스로 직접 재사용**: RMSNorm(`forward.cpp:13-49`), Softmax(가변 길이/tripcount 처리 `:51-96`), 그룹양자 정수 matmul(`:191-267`), RoPE(`:323-363`), SwiGLU(`:454-465`)는 ViT의 LayerNorm/Softmax/Linear/MLP와 구조적으로 동형 → ViT 인코더 블록 HLS 설계의 출발점.
2. **반면교사(성능)**: HG-PIPE류 "고처리량/완전 파이프라인"을 목표로 한다면, 본 repo의 (a) matmul 미언롤(`:255,260`), (b) 가중치 BRAM 비상주, (c) 토큰 단위 직렬 호출은 **피해야 할 패턴**. 우리는 가중치 온칩 스테이징 + dataflow/systolic + 멀티 CU/배치로 처리량을 끌어올려야 함을 보여주는 대조군.
3. **SW/HW 분리 + XRT 호스트 템플릿**: `llama2.cpp`의 XRT 버퍼 할당/`set_arg`/sync 루프(`:701-770`)는 AWS F1 또는 임베디드 XRT 기반 호스트 작성 시 참고 가능. XR 시선추적 추론도 "전처리(호스트) + 모델 forward(커널) + 후처리(호스트)" 분리에 동일 패턴 적용 가능.
4. **그룹 양자화 전략**: GS=64 int8 + 그룹 스케일은 ViT/시선추적 모델 경량화 시 정확도-자원 trade-off 레퍼런스. 단, on-device(XR)라면 본 repo의 큰 외부 DDR 가중치 모델보다 더 공격적 압축/온칩 상주가 필요(시사).
5. **벤치마크 프레임 차용**: CPU/GPU/FPGA 속도+전력(codecarbon)+정확도(perplexity) 3축 비교 구성(`cpu_benchmarks`/`gpu_benchmarks`)은 우리 XR 가속기의 평가 프로토콜로 그대로 이식 가능.

(주의: 위 "우리 프로젝트 = HG-PIPE 계열 ViT 가속 + XR 시선추적"은 부모 디렉토리 명/지시 기반 **추정**이며 본 repo 코드로 직접 증명되지 않음.)

---

## 10. 근거 표기 요약

- **확인(라인 근거)**: 모델 형상(config.h:4-11), 커널 시그니처/AXI(forward.cpp:269-272), forward 연산 순서(forward.cpp:306-486), matmul 양자 MAC(forward.cpp:257-263), matmul UNROLL 주석화(forward.cpp:255,260), quantize 로직(forward.h:15-64), 호스트 XRT 루프(llama2.cpp:701-770), 체크포인트 v2/magic(llama2.cpp:150-165), AWS F1 VU9P 플랫폼·CU 바인딩(hw_link.prj:2-14), CPU 포인터 자료구조(runq_latency_1024.c:35-39), GPU fp16/stories110M(tinyllama2.py:40-53).
- **추정**: matmul 대역폭 바운드, RoPE 테이블화 부재의 지연 영향, CPU 벤치 자체 수정 범위(measurement hook), quantize 포화 이슈, hw_link.prj 인자 불일치의 빌드 영향, "우리 프로젝트=HG-PIPE+XR" 매핑.
- **확인 불가**: `HLSTransformation` 형제 repo와의 파일별 동일성(경로가 연결 폴더 밖 → 직접 대조 불가). 구조·기능상 **동일/유사 계열**로 판정하되 라인 단위 diff는 미수행.
