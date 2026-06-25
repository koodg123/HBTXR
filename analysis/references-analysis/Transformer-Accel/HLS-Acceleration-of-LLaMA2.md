# HLS-Acceleration-of-LLaMA2 정밀 분석

> 대상 repo: `REF/Transformer-Accel/HLS-Acceleration-of-LLaMA2`
> 분석 일자: 2026-06-20
> 근거 표기 규칙: **확인** = 실제 코드 라인 근거(파일명:라인), **추정** = 코드/문맥 기반 합리적 추론, **확인 불가** = repo 내 근거 없음

---

## 1. 개요

LLaMA2 디코더-온리 Transformer의 **단일 토큰 forward pass**를 Vitis HLS 커널로 구현하여 AMD Zynq UltraScale+ MPSoC **ZCU106** 보드의 PL(FPGA)에서 가속하는 프로젝트다. (확인: `README.md:1-4`)

- **소프트웨어 기반**: Andrej Karpathy의 `llama2.c`를 거의 그대로 채택한 호스트 코드다. `main.cpp:1` 주석 "Inference for Llama-2 Transformer model in cpp", 그리고 토크나이저(BPE)/샘플러(argmax·multinomial·top-p)/생성 루프/채팅 루프 구조가 llama2.c와 동일하다. (확인: `main.cpp:209-560`, `main.cpp:574-743`)
- **가속 핵심**: 원본 llama2.c의 `forward()` 함수 본체(layer loop 전체)를 통째로 FPGA 커널 `kernel_forward()`로 오프로드했다. 호스트의 `forward()`는 RoPE 테이블만 CPU에서 만들고 커널을 호출(`kernel(...)`)한 뒤 logits를 받아온다. (확인: `main.cpp:179-207`, `kernel_forward.cpp:294-386`)
- **타깃 모델**: `stories15M.bin` (TinyStories 15M). 커널에서 하이퍼파라미터를 컴파일 타임 상수로 하드코딩: dim=288, hidden_dim=768, n_layers=6, n_heads=6, vocab=32000, seq_len=256, head_size=48, kv_dim=288. (확인: `kernel_forward.cpp:3-13`)
- **성능 보고치**: PS-only 1.75 tok/s → FPGA 가속 8.71 tok/s, 약 5배 향상. (확인: `README.md:71-78`)
- **양자화**: **없음**. 전 구간 `float`(FP32) 연산이다. (확인: 전 파일에 `float`만 사용, INT/fixed/ap_int/ap_fixed 미등장)

**한 줄 요약(추정)**: "llama2.c forward를 FP32 그대로 단일 HLS 커널로 감싸 XRT로 ZCU106에 올린 교과서적 첫-가속(first-cut) 구현"으로, 알고리즘 정확도는 보존하되 데이터 타입/메모리 최적화는 미적용 상태다.

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (분석 대상)

```
HLS-Acceleration-of-LLaMA2/
├── kernel_forward.cpp   # Vitis HLS 커널: Transformer forward 전체 (387 라인)
├── main.cpp             # XRT 호스트 프로그램: llama2.c 포팅 + 디바이스 관리 (939 라인)
└── README.md            # 빌드/실행/보드 부팅/성능 문서
```

repo 전체가 위 3개 파일이 전부다. **HLS 커널과 호스트가 각각 단일 파일**이며, 별도 헤더/모듈 분할이 없다. (확인: Glob 결과 — `.git` 외 트래킹 파일은 위 3개뿐)

### 2.2 제외 항목 (이름만 언급)

- **빌드 시스템 없음**: `Makefile` / `CMakeLists.txt` / `*.tcl`(Vitis cfg, run_hls.tcl 등) **부재**. (확인 불가 — repo에 없음. 빌드는 Vitis GUI로 수행한 것으로 추정)
- **생성물/바이너리 부재**: `kernel_forward.xclbin`, `binary_container_1.bin`(README가 언급하나 미포함), 모델 체크포인트 `stories15M.bin`, `tokenizer.bin` 모두 미포함. (README가 SD카드 복사 대상으로만 언급 — 확인: `README.md:21-26`, `README.md:68`)
- **버전관리 메타**: `.git/`(HEAD, objects/pack, hooks 샘플 등) — 분석 제외.

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

본 절은 FPGA 커널(`kernel_forward.cpp`)을 함수 단위로 라인 근거와 함께 정밀 분석한다. 커널은 LLaMA2 블록 6요소(RMSNorm, QKV matmul, RoPE, attention+KV cache, FFN/SwiGLU, residual)를 별도 함수로 분해하고 `kernel_forward()` 톱레벨에서 layer loop로 조립한다.

### 3.1 톱레벨 커널 `kernel_forward()` — 인터페이스와 데이터플로우

(확인: `kernel_forward.cpp:294-386`)

**인터페이스 구성 (m_axi / s_axilite)**:
- 17개 포인터 인자를 4개 AXI 메모리 번들로 분산하여 대역폭을 병렬화한다. (확인: `kernel_forward.cpp:298-313`)
  - `gmem0`: W_table, W_att, W_ffn, W_final, W_wcls (임베딩/RMSNorm/분류기) — `:298-302`
  - `gmem1`: W_wq, W_wk, W_wv, W_wo, table (어텐션 가중치 + RoPE 테이블) — `:303-306, :313`
  - `gmem2`: W_w1, W_w2, W_w3 (FFN 가중치) — `:307-309`
  - `gmem3`: S_key_cache, S_value_cache, S_logits (KV 캐시 + 출력, R/W) — `:310-312`
- 모든 메모리 포트에 `max_read_burst_length=64`, KV/logits에는 `max_write_burst_length=64` 설정으로 버스트 전송을 유도한다. (확인: `kernel_forward.cpp:298-313`)
- 제어 인자(pos, token)와 모든 포인터 베이스주소, return은 `s_axilite ... bundle=control`로 묶어 호스트가 단일 컨트롤 인터페이스로 기동한다. (확인: `kernel_forward.cpp:315-333`)

**온칩 상태 버퍼**: forward 1회분의 중간 텐서를 BRAM에 상주시킨다. `S_x, S_xb, S_xb2`(각 dim=288), `S_hb, S_hb2`(hidden=768), `S_q`(288), `S_att`(n_heads*seq_len=6*256=1536). 모두 `ARRAY_PARTITION cyclic factor=32 dim=1`로 분할해 32-way 병렬 접근을 가능케 한다. (확인: `kernel_forward.cpp:335-342`)

**실행 순서 (라인 근거)**:
1. 토큰 임베딩 lookup: `S_x[i] = W_table[token*dim + i]` (확인: `:344-348`)
2. RoPE 테이블 온칩 적재: `TABLE[i] = table[i]` (호스트가 cos/sin을 미리 채워 보냄) (확인: `:350-357`)
3. `layer` 루프 (l=0..5, 확인: `:359-382`):
   - `rmsnorm(S_xb, S_x, W_att + l*dim)` — 어텐션 전 정규화 (`:365`)
   - `matmul_dim_dim(S_q, ...)`, `matmul_dim_kvdim(S_k/S_v, ...)` — Q/K/V 투영 (`:366-368`)
   - K/V는 캐시의 현재 pos 슬롯에 직접 쓴다: `S_k = S_key_cache + loff + pos*kv_dim` (`:362-363`)
   - `RoPE(TABLE, S_q, S_k)` (`:370`)
   - `attention(...)` (`:371`)
   - `matmul_dim_dim(S_xb2, S_xb, W_wo)` + `residual(S_x, S_xb2)` — 출력 투영 후 잔차 (`:372-373`)
   - `rmsnorm` → `matmul_dim_hiddendim` x2(w1,w3) → `SwiGLU` → `matmul_hiddendim_dim`(w2) → `residual` — FFN 블록 (`:375-381`)
4. 최종 `rmsnorm(S_x, S_x, W_final)` + `matmul_dim_vocabsize(S_logits, ...)` — 분류기 logits (확인: `:384-385`)

> 주의(추정): KV 캐시 인덱싱은 `loff = l*seq_len*kv_dim`(`:361`)이고 `S_k = ... + pos*kv_dim`(`:362`)인데, kv_dim==dim==288이므로 본 모델에선 정상 동작하나 GQA(n_kv_heads<n_heads)로 일반화하면 `attention()` 내 head 인덱싱(`h*P_HEAD_SIZE`, `:223,:257`)이 어긋날 수 있다 — 현 코드는 MHA(kv_dim=dim) 전제.

### 3.2 RMSNorm `rmsnorm()`

(확인: `kernel_forward.cpp:15-44`)

- 가중치 `w`를 온칩 `W[P_DIM]`으로 복사(`:19-23`, `UNROLL factor=8`)한 뒤 사용 → 외부 메모리 재접근 제거.
- **제곱합 reduction을 4-way 부분합으로 분할**: `partial[i%4] += x[i]*x[i]`, `PIPELINE II=1` (확인: `:25-32`). 단일 누산기 의존성 체인을 끊어 II=1을 달성하는 전형적 기법.
- 정규화 계수: `ss *= INV_P_DIM(=1/288)`, `+1e-5`, `hls::rsqrt(ss)` (확인: `:34-37`). 원본 llama2.c의 `1.0f/sqrtf`(`main.cpp:138`)를 HLS 친화적 `rsqrt`로 치환.
- 출력: `o[i] = W[i]*(ss*x[i])`, `UNROLL factor=8` (확인: `:39-43`).
- `W` 배열은 `ARRAY_PARTITION cyclic factor=32`(`:17`)지만 read 루프는 factor=8 unroll — 분할(32)과 언롤(8) factor 불일치는 의도적이거나 미세 비효율로 볼 수 있다(추정).

### 3.3 Softmax `softmax()`

(확인: `kernel_forward.cpp:46-78`)

- 수치 안정화 max 탐색(`:49-56`, II=1), `hls::expf(x[i]-max_val)`와 합 누적을 4-way partial로(`:61-67`), 역수 곱 정규화(`:72-77`, `UNROLL factor=32`).
- `LOOP_TRIPCOUNT min=1 max=257`(`:51,:63,:74`)로 가변 길이(pos+1, 최대 seq_len+1) 루프의 레이턴시 추정을 컴파일러에 제공.
- attention 내부에서 `att`(score 벡터)에 대해 호출된다(`:239`).

### 3.4 Matmul 커널군 (5종)

5개의 행렬-벡터 곱 함수가 **shape별로 하드코딩**되어 있다. 모두 동일 패턴: 출력 차원 i 외부 루프(`loop_flatten off`) + 입력 차원 j 내부 루프(`PIPELINE II=1`) + 4-way partial 누산. (확인: `kernel_forward.cpp:80-168`)

| 함수 | 출력×입력 | 가중치 인덱싱 | 용도 | 라인 |
|------|-----------|---------------|------|------|
| `matmul_dim_dim` | 288×288 | `w[i*P_DIM+j]` | Wq, Wo 투영 | `:80-96` |
| `matmul_dim_kvdim` | 288×288 | `w[i*P_DIM+j]` | Wk, Wv 투영 | `:98-114` |
| `matmul_dim_hiddendim` | 768×288 | `w[i*P_DIM+j]` | FFN w1, w3 | `:116-132` |
| `matmul_hiddendim_dim` | 288×768 | `w[i*P_HIDDEN_DIM+j]` | FFN w2 | `:134-150` |
| `matmul_dim_vocabsize` | 32000×288 | `w[i*P_DIM+j]` | 분류기 wcls | `:152-168` |

- `matmul_dim_dim`과 `matmul_dim_kvdim`은 본 모델(kv_dim==dim==288)에서 **완전히 동일한 코드**다(확인: `:80-96` vs `:98-114` 바이트 동일). GQA 일반화를 염두에 둔 이름 분리이나 현재는 중복(추정).
- 내부 루프만 II=1 파이프라인이고 외부 출력 루프는 `loop_flatten off`(`:83 등`)로 평탄화를 막아 partial 배열 초기화/리덕션을 출력 원소마다 깨끗이 수행한다.
- **벡터화/언롤 미적용**: 내부 곱은 4-way partial로 의존성만 끊었을 뿐, j 방향 데이터 병렬(예: 8/16-lane MAC)은 없다. 가장 큰 비용인 `matmul_dim_vocabsize`(32000×288 ≈ 9.2M MAC)도 동일하게 스칼라 파이프라인 — 주요 병목 후보(추정).

### 3.5 RoPE `RoPE()` (회전 위치 임베딩)

(확인: `kernel_forward.cpp:170-210`)

- 호스트가 위치별 cos/sin을 미리 계산해 `TABLE`로 전달(`main.cpp:189-197`): `freq=1/10000^(head_dim/head_size)`, `val=pos*freq`, `TABLE[i]=cos(val)`, `TABLE[i+1]=sin(val)`. → **삼각함수를 PL에서 계산하지 않고 PS에서 LUT화**한 설계 (확인: `main.cpp:189-195`).
- 커널은 S_k를 온칩 `S_K`로 복사(`:174-178`) 후 2칸씩(complex 쌍) 회전 적용, `UNROLL factor=16`(`:181-182`).
- 회전식: `out0 = v0*fcr - v1*fci`, `out1 = v0*fci + v1*fcr` (확인: `:189-190` Q, `:194-195` K).
- `if (i < P_KV_DIM)`(`:186`)로 Q와 K를 함께 회전, 그 외엔 Q만 회전 — kv_dim==dim이라 본 모델에선 항상 두 가지를 함께 회전(else 분기 미실행, 추정). 결과를 `S_k`로 다시 write-back(`:205-209`).

### 3.6 Attention (멀티헤드 + KV 캐시) `attention()`

(확인: `kernel_forward.cpp:212-274`) — **본 커널에서 가장 복잡한 모듈**.

- 헤드 루프 `h=0..5`(`:213-214`). 헤드별 q 포인터 `S_q + h*head_size`, att 포인터 `S_att + h*seq_len`(`:215-216`).
- **Score 계산(QK^T)**: t=0..pos 각 과거 토큰에 대해 `k = S_key_cache + loff + h*head_size + t*kv_dim`(`:223`)에서 키를 읽어 q·k 내적을 4-way partial+II=1로 계산(`:228-234`), `score *= 1/sqrt(head_size)=0.14433757`(`:235`, 상수 `INV_SQRT_P_HEAD_SIZE`), `att[t]=score`(`:236`).
- **Softmax**: `softmax(att, pos+1)` — 가변 길이(현재까지 본 토큰 수) (`:239`).
- **가중합(attn·V)**: `acc[head_size]`(`ARRAY_PARTITION cyclic 32`, `:243-244`)를 0 초기화(`:246-251`) 후, t=0..pos에 대해 `v = S_value_cache + loff + h*head_size + t*kv_dim`(`:257`), `acc[i] += att[t]*v[i]`를 II=1로 누산(`:261-265`). 최종 `xb[i]=acc[i]`(`:268-272`).
- **KV 캐시 활용**: K/V는 커널 진입 전 layer loop에서 현재 pos 슬롯에 이미 기록(`main` 흐름상 `matmul_dim_kvdim`이 `S_k`(=캐시의 pos 슬롯)에 직접 쓴 뒤 RoPE로 갱신, `kernel_forward.cpp:367-370`). attention은 0..pos 전체 캐시를 스트리밍 read만 한다 — **autoregressive 디코딩의 O(seq) 누적 read 패턴** (확인: `:219,:254`의 `t<=pos`).
- 가변 루프엔 `LOOP_TRIPCOUNT min=1 max=257`(`:221,:255`).

### 3.7 SwiGLU FFN `SwiGLU()`

(확인: `kernel_forward.cpp:285-292`)

- `S_hb[i] *= 1/(1+exp(-S_hb[i]))` → SiLU(=x·sigmoid(x)) 게이트 (확인: `:289`).
- `S_hb[i] *= S_hb2[i]` → gate(w1·x) ⊙ (w3·x) 의 게이팅 곱 (확인: `:290`). `UNROLL factor=8`(`:288`).
- FFN 전체 흐름: `h1=w1·x`, `h3=w3·x`, `SwiGLU(h1,h3)`, `out=w2·(silu(h1)⊙h3)` (확인: `kernel_forward.cpp:376-380`). 이는 표준 LLaMA SwiGLU FFN과 일치.

### 3.8 Residual `residual()`

(확인: `kernel_forward.cpp:277-283`) — `S_x[i] += S_xb[i]`, `UNROLL factor=32`. 어텐션 출력/ FFN 출력 두 번 적용(`:373,:381`).

### 3.9 호스트측 핵심 (`main.cpp`)

- **모델 로딩**: `read_checkpoint()`가 .bin을 `mmap`하고 `memory_map_weights()`로 포인터를 레이아웃에 맞춰 슬라이싱(확인: `main.cpp:63-114`). RoPE용 freq_cis 영역은 skip(`:89-90`) — 본 구현은 RoPE를 런타임 계산하므로 미사용.
- **XRT 디바이스/커널 셋업**: `xrt::device(0)` → `load_xclbin("kernel_forward.xclbin")` → `xrt::kernel(...,"kernel_forward")` (확인: `main.cpp:769-771`).
- **버퍼 할당/복사**: 12개 가중치 `xrt::bo`를 `kernel.group_id(N)`(N=2..13)로 메모리 뱅크에 배치, `map`→`memcpy`→`sync(TO_DEVICE)`로 1회 적재(확인: `main.cpp:850-900`). KV캐시/logits/table 버퍼는 group_id 14..17(`:906-914`).
- **forward 1스텝**: 호스트가 RoPE 테이블 갱신→`bo_table.sync(TO_DEVICE)`→`kernel(pos,token,...)`→`run.wait()`→`bo_logits.sync(FROM_DEVICE)`(확인: `main.cpp:189-204`). 가중치는 매 스텝 재전송하지 않고 table/logits만 동기화 — 효율적.
- **토크나이저/샘플러/생성·채팅**: llama2.c와 동일한 BPE encode(`:298-417`), argmax/multinomial/top-p sample(`:436-560`), generate/chat 루프(`:574-743`). (확인)

---

## 4. 데이터 플로우

```
[PS/호스트 main.cpp]
  .bin mmap → memory_map_weights → xrt::bo memcpy → sync(TO_DEVICE)  // 가중치 1회 적재
  generate loop (pos=0..steps):
    RoPE table = cos/sin(pos)  → bo_table.sync(TO_DEVICE)             // 매 스텝
    kernel_forward(pos, token, ...) ──────────────► [PL/FPGA 커널]
                                                       token embed lookup
                                                       for layer 0..5:
                                                         RMSNorm→QKV matmul→(K/V→KV cache)
                                                         RoPE→attention(read KV cache)→Wo→residual
                                                         RMSNorm→w1,w3→SwiGLU→w2→residual
                                                       final RMSNorm→wcls matmul→S_logits
                                       ◄────────────── bo_logits.sync(FROM_DEVICE)
    sample(logits) → next token → decode → print
```

- **온칩 상주**: 활성화 텐서 전부(S_x/xb/hb/q/att)는 BRAM 상주, 가중치는 매번 DDR에서 m_axi 버스트 read. (확인: `kernel_forward.cpp:335-342` 활성화, `:298-313` 가중치 외부 read)
- **KV 캐시 위치**: DDR(`gmem3`)에 상주하며 layer×seq×kv_dim 크기로 호스트가 할당(`main.cpp:903-907`). 커널이 직접 read/write. (확인)
- **PS↔PL 스텝당 통신**: table(288 float) write + logits(32000 float) read만. 가중치 재전송 없음. (확인: `main.cpp:197-203`)

---

## 5. HW/SW 매핑

| 기능 | 위치 | 근거 |
|------|------|------|
| 모델 .bin mmap, 가중치 레이아웃 | PS (CPU) | `main.cpp:63-114` |
| BPE 토크나이즈/디토크나이즈 | PS | `main.cpp:230-417` |
| 샘플링(argmax/top-p/multinomial) | PS | `main.cpp:436-560` |
| RoPE cos/sin 테이블 생성 | PS | `main.cpp:189-195` |
| 토큰 임베딩 lookup | PL | `kernel_forward.cpp:344-348` |
| RMSNorm / Softmax | PL | `:15-78` |
| QKV / Wo / FFN / 분류기 matmul | PL | `:80-168` |
| RoPE 회전 적용 | PL | `:170-210` |
| Multi-head attention + KV cache R/W | PL | `:212-274` |
| SwiGLU / residual | PL | `:277-292` |
| 커널 기동/대기, BO 동기화 | PS↔PL (XRT) | `main.cpp:199-204` |

**경계 설계 요약(추정)**: 제어 흐름·문자열·동적 메모리(토크나이저/샘플러)는 PS, 정형 수치 연산(GEMV/elementwise)은 PL이라는 깔끔한 분리. 삼각함수(RoPE)만 PL 자원 절약을 위해 PS로 끌어올린 의도적 선택.

---

## 6. 빌드 · 실행

- **개발 환경**: Ubuntu 22.04, Vitis/Vivado 2024.2 (확인: `README.md:8-12`).
- **실행 환경**: ZCU106 + PetaLinux 2024.2 (확인: `README.md:14-17`).
- **빌드 스크립트 부재**: Makefile/tcl 없음 → Vitis GUI 빌드로 추정(확인 불가). 호스트는 XRT 헤더(`xrt/xrt_bo.h` 등, `main.cpp:15-17`) 링크 필요. 커널은 `hls_math.h`(`kernel_forward.cpp:1`)만 의존.
- **배포 절차**(확인: `README.md:19-69`): SD카드에 host 실행파일(`Llama2_host`)·커널 바이너리(`binary_container_1.bin`)·모델(`stories15M.bin`)·`tokenizer.bin` 복사 → SD 모드 부팅 → UART 115200bps 모니터 → `./Llama2_host stories15M.bin` 실행.
  - 단, 코드상 기본 경로는 `model/stories15M.bin`, `model/tokenizer.bin`이고 xclbin 이름은 `kernel_forward.xclbin`이다(확인: `main.cpp:770,778-779`). README의 `binary_container_1.bin`/실행 인자와 코드 기본값이 **불일치** — 빌드 산출물 이름·실행 시 인자 조정 필요(추정).
- **CLI 옵션**: `-t/-p/-s/-n/-i/-c/-z/-m/-y`(temperature, topp, seed, steps, prompt, ckpt, tokenizer, mode, system) (확인: `main.cpp:750-801`).

---

## 7. 의존성

- **HLS 커널**: `hls_math.h`(`hls::rsqrt`, `hls::expf`)만 사용 — 외부 라이브러리 무의존 (확인: `kernel_forward.cpp:1,37,65,289`).
- **호스트**: 표준 C/C++ (`stdio/stdlib/math/string/fcntl/mman` 등) + **XRT C++ API** (`xrt_bo.h`, `xrt_device.h`, `xrt_kernel.h`) (확인: `main.cpp:3-17`).
- **외부 데이터(미포함)**: `stories15M.bin`(TinyStories Llama2 15M 가중치), `tokenizer.bin`(SentencePiece/llama2.c 포맷), 합성된 `*.xclbin`. (확인 불가 — repo 미포함, README 참조)
- **third-party/vendor 디렉토리**: 없음.

---

## 8. 강점 · 한계

### 강점
- **알고리즘 충실도**: llama2.c 수식을 그대로 보존(softmax 안정화, top-p, BPE 등)하여 정확도 검증이 쉬움 (확인: `main.cpp` 전반이 원본과 동형).
- **깔끔한 HW/SW 분리**: 수치 연산만 PL로, 제어/문자열/RNG는 PS로. RoPE 삼각함수를 PS로 빼 PL 자원 절약 (확인: §5).
- **메모리 번들 분할**: gmem0~3으로 동시 read 대역폭 확보, burst length 명시 (확인: `kernel_forward.cpp:298-313`).
- **검증된 파이프라인 패턴**: 4-way partial로 reduction 의존성 제거 + II=1 달성, 활성화 BRAM 상주 (확인: 모든 reduction 루프).
- **가중치 1회 적재**: 스텝당 통신 최소화(table/logits만) (확인: `main.cpp:889-900` vs `:197-203`).

### 한계
- **양자화/저정밀 미적용**: 전 구간 FP32. INT8/4 또는 fixed-point 미사용 → 대역폭·DSP·BRAM 비효율, 대형 모델 확장 곤란 (확인: 전 파일 float).
- **matmul 데이터 병렬 부족**: 내부 곱이 스칼라 4-way partial뿐, j-방향 다중 lane MAC 부재. 최대 부하 `matmul_dim_vocabsize`(9.2M MAC)도 동일 → 병목 가능 (확인: `:152-168`).
- **하이퍼파라미터 하드코딩**: dim/layers/heads 등이 `#define`(`:3-13`). 다른 모델은 재합성 필요. matmul도 shape별 함수 5종으로 중복(`matmul_dim_dim`≡`matmul_dim_kvdim`) (확인: `:80-114`).
- **단일 토큰·단일 배치**: forward가 토큰 1개씩(autoregressive), prefill 배칭 없음 → 시퀀스 길이에 따라 attention read 선형 증가 (확인: `:219,:254`).
- **MHA 전제**: GQA(n_kv_heads<n_heads) 일반화 시 KV 인덱싱 보정 필요(추정, §3.1 주의).
- **레이어 함수 dataflow 미적용**: layer loop 내 함수들이 순차 호출(task-level pipelining/`DATAFLOW` 없음) → 함수 간 오버랩 없음 (확인: `:359-382` 순차 호출, DATAFLOW pragma 미등장).
- **문서·코드 산출물 이름 불일치**: README의 `binary_container_1.bin` vs 코드 `kernel_forward.xclbin`, 경로 기본값 차이 (확인: §6).

---

## 9. 우리 프로젝트 시사점 (PRJXR-HBTXR: 고처리량 ViT/Transformer FPGA 가속기[HG-PIPE 계열] + XR 시선추적, 추정)

> 본 절은 우리 프로젝트 방향(추정)에 대한 함의이며, 본 repo 코드만으로 확인 가능한 부분과 추정을 구분한다.

1. **대조군/베이스라인으로서의 가치(추정)**: 이 repo는 "최적화 거의 없는 단일 커널 FP32 first-cut"의 전형이다. HG-PIPE류가 지향하는 **레이어 파이프라인(`DATAFLOW`)·저정밀 양자화·systolic/output-stationary MAC 어레이**와 정면 대비되어, 우리 설계의 개선 포인트를 수치로 보여주는 베이스라인(8.71 tok/s, 5배)이 될 수 있다 (확인: 성능 `README.md:71-78`, 미적용 최적화 §8).

2. **PS/PL 경계 분리 패턴은 재사용 가능(확인 기반)**: 토크나이즈·샘플링·삼각함수 LUT를 PS로, 정형 GEMV/elementwise를 PL로 보내는 분리는 XR 파이프라인에도 적용 가능. 단 XR 시선추적은 **스트리밍 영상 입력(ViT patch embedding)**이 추가되므로, 본 repo의 "토큰 임베딩 lookup→layer loop"를 "패치 임베딩→encoder block loop"로 치환하는 매핑이 자연스럽다(추정).

3. **반드시 넘어서야 할 한계(추정)**:
   - **데이터플로우/파이프라이닝**: 본 repo는 함수 순차 호출(§8). 고처리량 ViT는 블록 간 `DATAFLOW`로 오버랩 필수.
   - **양자화**: 본 repo FP32. XR 엣지 추론은 INT8/fixed 필요 — on-device 제약(레이턴시·전력)상 양자화가 핵심 차별점.
   - **공간적 MAC 병렬**: 본 repo 스칼라 MAC. ViT의 큰 GEMM은 systolic/PE 어레이로 공간 전개해야 처리량 확보.
4. **KV cache vs ViT 비인과(추정)**: 본 repo의 autoregressive KV 캐시(`:212-274`)는 디코더 특화다. ViT 인코더는 전체 시퀀스 동시 처리(causal mask 없음)이므로 KV 캐시 로직은 재사용 대상이 아니며, 대신 **전체 attention 행렬 타일링**이 관건(추정).

5. **RoPE 테이블 PS 사전계산 트릭(확인 기반)**: 위치 인코딩을 PS에서 LUT화해 PL 삼각함수 자원을 아낀 설계는, XR에서 위치/시간 임베딩이나 카메라 보정 LUT를 PS로 빼는 데 직접 응용 가능 (확인: `main.cpp:189-195`).

---

## 부록 A. 모델 하이퍼파라미터 (stories15M, 확인: `kernel_forward.cpp:3-13`)

| 파라미터 | 값 | 비고 |
|----------|----|------|
| P_DIM | 288 | 모델 차원 |
| P_HIDDEN_DIM | 768 | FFN 내부 차원 |
| P_N_LAYERS | 6 | 디코더 블록 수 |
| P_N_HEADS | 6 | 어텐션 헤드 |
| P_HEAD_SIZE | 48 | = dim/heads |
| P_KV_DIM | 288 | = dim (MHA) |
| P_VOCAB_SIZE | 32000 | SentencePiece |
| P_SEQ_LEN | 256 | 최대 시퀀스 |
| INV_P_DIM | 1/288 | RMSNorm 상수 |
| INV_SQRT_P_HEAD_SIZE | 1/√48 | attention 스케일 |

## 부록 B. 함수 인덱스

| 함수 | 파일:라인 | 역할 |
|------|-----------|------|
| `rmsnorm` | kernel_forward.cpp:15-44 | RMS 정규화 (HW) |
| `softmax` | kernel_forward.cpp:46-78 | 가변길이 softmax (HW) |
| `matmul_*` (5종) | kernel_forward.cpp:80-168 | GEMV shape별 (HW) |
| `RoPE` | kernel_forward.cpp:170-210 | 회전 위치 임베딩 (HW) |
| `attention` | kernel_forward.cpp:212-274 | MHA + KV cache (HW) |
| `residual` | kernel_forward.cpp:277-283 | 잔차 (HW) |
| `SwiGLU` | kernel_forward.cpp:285-292 | FFN 게이트 (HW) |
| `kernel_forward` | kernel_forward.cpp:294-386 | 톱레벨 커널 (HW) |
| `memory_map_weights`/`read_checkpoint` | main.cpp:63-114 | 가중치 로딩 (SW) |
| `forward` | main.cpp:179-207 | RoPE 테이블+커널 호출 (SW) |
| `encode`/`decode` | main.cpp:263-417 | BPE 토크나이저 (SW) |
| `sample*` | main.cpp:436-560 | 샘플링 (SW) |
| `generate`/`chat` | main.cpp:574-743 | 생성/채팅 루프 (SW) |
| `main` | main.cpp:767-938 | XRT 셋업 + 디스패치 (SW) |
