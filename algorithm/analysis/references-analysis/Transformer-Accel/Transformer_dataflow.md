# Transformer_dataflow 정밀 분석

분석 대상: `REF/Transformer-Accel/Transformer_dataflow`
작성일: 2026-06-20
근거 표기 규칙: 확인된 사실은 `파일명:라인`으로 표기. 코드에 없는 추론은 "추정", 확인 불가능한 항목은 "확인 불가"로 명시.

---

## 1. 개요

Transformer_dataflow는 **Transformer Encoder(BERT/ViT 계열) 전체를 HLS C++(Vitis HLS) 템플릿 헤더 라이브러리로 구현한 단일 가속기 IP**다. 최상위 커널은 `accel`(`Accel.cpp:4`, 프로토타입 `Definitions.h:19`)이며, `csim`(C 시뮬레이션) → `csynth`(합성) → `cosim`(코시뮬) → `export_design`(IP 카탈로그) 흐름으로 빌드된다(`run_hls.tcl:10-15`).

핵심 특징:
- **순수 템플릿 헤더 기반 모듈러 구조**: 모든 연산(matmul, linear, softmax, layernorm, attention 등)이 `template<typename T, int ...>` 형태의 헤더 전용 함수로 작성되어, 파라미터(시퀀스 길이, 토큰 차원, 헤드 수)를 컴파일 타임에 주입한다(`Layers/*.h`).
- **함수 호출 계층 = 데이터플로우 그래프**: 상위 함수가 하위 함수를 순차 호출하고 중간 결과를 로컬 배열로 전달하는 구조. Encoder → MultiHeadAtt → AttHead → ScaledDotAtt → MatMul/SoftMax 식의 트리(`Encoder.h`, `MultiHeadAtt.h`, `AttHead.h`, `ScaledDotAtt.h`).
- **데이터 타입은 현재 `double`**: `Definitions.h:13-14`에서 `idata_t`/`odata_t`가 `double`로 정의. `ap_fixed.h`/`hls_math.h`를 include하지만(`Definitions.h:2-3`) 고정소수점은 미사용. 즉 **정밀도 검증용 골든 모델 성격이 강하고, 양자화/고정소수점화는 아직 적용 전 상태**(추정).
- **모듈 단위 단위테스트 인프라**: `Tests/Test*.h` 16종 + PyTorch 골든 생성기(`TestFilesGenerator.py`)로 각 레이어를 독립 검증. `Transformer_dataflow.cpp:9-27`의 `Test_types` enum이 16개 테스트를 분기.

중요한 한계: 이름은 "dataflow"지만 **현재 코드에는 `#pragma HLS DATAFLOW`, `#pragma HLS PIPELINE`, `hls::stream`, 배열 파티셔닝이 거의 없다.** UNROLL 프라그마만 일부 존재(`Scale.h:13`, `Transpose.h:6,9`, `VecAdd.h:6`). 즉 "dataflow"는 **아키텍처 목표/명명 의도**이며, 현 시점 구현은 알고리즘적 골든 레퍼런스 단계(추정).

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (분석 대상)

```
Transformer_dataflow/
├─ Accel.cpp                  최상위 커널 accel() 래퍼 (set_top 대상)
├─ Definitions.h              전역 파라미터/타입/accel 프로토타입(모드별 #ifdef)
├─ Transformer_dataflow.cpp   호스트/테스트 디스패처 main() — 16종 테스트 enum 분기
├─ TestBenchAccel.cpp         HLS 테스트벤치(-tb) — Encoder 모드 csim 드라이버
├─ TestFilesGenerator.py      PyTorch 골든 데이터 생성기 (레이어별 입력/정답)
├─ run_hls.tcl               Vitis HLS 빌드 스크립트
├─ run_vitis_command _line.sh  vitis_hls 실행 셸 래퍼
├─ Transformer_dataflow.sln / .vcxproj / .filters  Visual Studio 프로젝트(csim 디버깅용)
├─ Layers/                    HLS 연산 커널(헤더 전용 템플릿)
│   ├─ Encoder.h             Encoder 블록(MHA+residual+LN+FF+residual+LN)
│   ├─ MultiHeadAtt.h        멀티헤드 어텐션(헤드 루프 + concat + 출력 linear)
│   ├─ AttHead.h             단일 헤드(Q/K/V linear + scaled-dot-att)
│   ├─ ScaledDotAtt.h        scaled dot-product attention(fused matmul-scale-mask-softmax)
│   ├─ MatMul.h              matmul / transpose_matmul / matmul_transpose_scale
│   ├─ Linear.h              linear = matmul + bias(vecadd)
│   ├─ FF.h                  position-wise FFN(linear→relu→linear)
│   ├─ LayerNorm.h           layer normalization
│   ├─ SoftMax.h             softmax / masked_softmax / fused matmul_scale_masked_softmax
│   ├─ Activations.h         relu / gelu / erf (활성함수)
│   ├─ Scale.h               요소별 스칼라 나눗셈(scale)
│   ├─ Mask.h                요소별 마스크 곱
│   ├─ Transpose.h           행렬 전치
│   ├─ Concat.h              concat_cols / concat_rows
│   ├─ MatAdd.h              행렬 덧셈(residual용)
│   └─ VecAdd.h              벡터 덧셈(bias add용)
└─ Tests/                     단위테스트 하니스(헤더 전용 템플릿)
    ├─ TestUtils.h           load_arr / compare_vec / compare_mat (파일 I/O + 검증)
    ├─ TestConstants.h       테스트 파라미터(TYPE=float, ROWS=COLS=HIDDEN=10 등)
    ├─ Test.h                테스트 집합 인클루드 허브(추정; 내용 미확인)
    └─ TestEncoder.h, TestMultiHeadAtt.h, TestAttHead.h, TestScaledDotAtt.h,
       TestMatMul.h, TestLinear.h, TestFF.h, TestLayerNorm.h, TestSoftMax.h,
       TestActivations.h, TestScale.h, TestMask.h, TestConcat.h, TestMatAdd.h,
       TestVecAdd.h, TestTranspose.h, TestConstants.h
```

### 2.2 제외 항목 (third-party/vendor/생성물 — 이름만)

- `.git/` 전체 (버전관리 메타데이터, pack 객체 등)
- `.gitignore`, `.gitattributes` (빌드 산출물 제외 규칙; `vitis_hls.log`, `Transformer_accel/` 합성 디렉토리, `*.txt` 데이터 파일 무시 — `.gitignore:12,366,368`)
- `*.txt` (input1~12.txt, golden_result.txt, log.txt) — `TestFilesGenerator.py`가 생성하는 런타임 데이터(저장소에 미커밋, `.gitignore:12`)
- `Transformer_accel/` — Vitis HLS 합성 프로젝트 산출 디렉토리(`.gitignore:368`, 미존재)
- Vitis HLS 표준 라이브러리: `ap_fixed.h`, `hls_math.h`(`Definitions.h:2-3`) — 벤더 제공, 분석 제외

---

## 3. 핵심 모듈 정밀 분석

> 호출 트리(확인): `accel`(`Accel.cpp:20`) → `encoder`(`Encoder.h:7`) → {`multi_head_att`(`MultiHeadAtt.h:7`), `matadd`, `layer_norm`, `ff`} ; `multi_head_att` → `att_head`(`AttHead.h:6`) → {`linear`×3, `scaledotatt`} ; `scaledotatt`(`ScaledDotAtt.h:9`) → {`matmul_scale_masked_softmax`, `matmul`} ; `ff`(`FF.h:6`) → {`linear`, `activation`, `linear`}.

### 3.1 최상위 커널 `accel` (Accel.cpp)

- `Definitions.h:16`에서 `#define ENCODER`로 빌드 모드를 고정. `#ifdef ENCODER` 분기(`Accel.cpp:3-37`)만 컴파일된다. 다른 모드(MULTIHEAD/ATTHEAD/FDFRWRD/LAYERNORM/DOTPRODATT/LINEAR/MATMUL/SOFTMAX/ACTIVATION)는 `Definitions.h:38-114`, `Accel.cpp:39-180`에 동일 패턴으로 대기.
- Encoder 모드 인자(`Accel.cpp:5-17`): head 가중치/바이어스(4D/3D), 출력 linear 가중치/바이어스, FF 가중치/바이어스 2세트, layernorm `gamma`/`beta` 2세트, `input[SEQ_LEN][TOKEN_LEN]`, `input_mask[SEQ_LEN][SEQ_LEN]`, `result[SEQ_LEN][TOKEN_LEN]`.
- `accel`은 로컬에 `epsilon[NUM_LAYER_NORM]={EPSILON,EPSILON}`(`Accel.cpp:19`)를 만들고 `encoder<...>(...)`를 호출하는 **얇은 어댑터**다. 즉 합성 대상은 `accel`이지만 실제 로직은 전부 `Encoder.h` 이하 템플릿.
- 주의: 다른 모드 일부에 컴파일 에러 소지가 있는 코드 잔존 — `Accel.cpp:69`의 `result[...],`(trailing comma), `Accel.cpp:113`의 `layer_norm(..., epsilon, ...)`에서 `epsilon`이 선언 없이 사용됨(`Definitions.h`/`Accel.cpp`의 LAYERNORM 분기에 `epsilon` 인자 없음). ENCODER 모드만 빌드되므로 현재는 무해하나, **모드 전환 시 수정 필요**(확인: `Accel.cpp:69`, `Accel.cpp:104-117`).

### 3.2 `encoder` — Transformer Encoder 블록 (Encoder.h:7-48)

표준 Post-LN Transformer 인코더 1개 레이어를 순차 데이터플로우로 구성. 확인된 단계(`Encoder.h:24-47`):

1. `multi_head_att(input,input,input, mask, head_w, head_b, lin_w, lin_b, tmp1)` — self-attention (Q=K=V=input, `Encoder.h:25-33`).
2. `matadd(input, tmp1, tmp2)` — residual add (`Encoder.h:36`).
3. `layer_norm(tmp2, epsilon[0], gamma[0], beta[0], tmp3)` — 1차 LN (`Encoder.h:39`).
4. `ff(tmp3, ff_w1, ff_b1, ff_w2, ff_b2, tmp4)` — position-wise FFN (`Encoder.h:42`).
5. `matadd(tmp3, tmp4, tmp5)` — 2차 residual add (`Encoder.h:45`).
6. `layer_norm(tmp5, epsilon[1], gamma[1], beta[1], result)` — 2차 LN, 최종 출력 (`Encoder.h:47`).

특징: 각 단계 사이 중간 텐서 `tmp1`~`tmp5`가 **별도 로컬 배열**로 잡힘(`Encoder.h:24,35,38,41,44`). 이는 HLS에서 모두 BRAM/레지스터로 매핑되며, `#pragma HLS DATAFLOW`가 없으므로 **순차 실행**(단계 간 파이프라인 오버랩 없음, 추정). 잔차 연결이 `input→tmp2`, `tmp3→tmp5`로 정확히 Post-LN 구조와 일치(PyTorch 골든도 동일: `TestFilesGenerator.py:94-100`).

### 3.3 `multi_head_att` — 멀티헤드 어텐션 (MultiHeadAtt.h:7-32)

- `multi_head_att_loop`(`MultiHeadAtt.h:19-28`): `num_heads`만큼 `att_head`를 호출, 헤드별 결과를 `tmp1[i]`에 누적. **헤드 루프는 unroll/pipeline 프라그마 없음** → 순차(추정). 현재 `NUM_HEADS=1`(`Definitions.h:5`)이라 루프 1회.
- `concat_cols(tmp1, tmp2)`(`MultiHeadAtt.h:30`): 헤드별 `[seq][head_len]` 결과를 열 방향으로 이어 붙여 `[seq][head_len*num_heads]` 생성(`Concat.h:4-15`).
- `linear(tmp2, linear_weights, linear_bias, result)`(`MultiHeadAtt.h:31`): 출력 projection(W_O).

### 3.4 `att_head` — 단일 어텐션 헤드 (AttHead.h:6-26)

- Q/K/V를 각각 별도 `linear`로 생성(`AttHead.h:17,20,23`), 가중치 인덱스 `weights[0/1/2]`로 Q/K/V projection 구분. 출력 차원은 `head_token_length`(헤드당 차원).
- `scaledotatt(Q,K,V,mask,result)` 호출(`AttHead.h:25`)로 attention 계산. Q/K/V 각각 `[seq][head_len]` 크기의 로컬 배열(`AttHead.h:16,19,22`).

### 3.5 `scaledotatt` + 융합 커널 `matmul_scale_masked_softmax` — Attention 핵심 (ScaledDotAtt.h, SoftMax.h)

가장 중요한 데이터플로우 최적화 지점.

- `scaledotatt`(`ScaledDotAtt.h:9-20`):
  1. `matmul_scale_masked_softmax(Q, K, SCALE_FACTOR, mask, softmax_att)` — **QKᵀ → /√d → mask → softmax**를 단일 함수로 융합(`ScaledDotAtt.h:17`).
  2. `matmul(softmax_att, value, result)` — attention weight × V (`ScaledDotAtt.h:19`).
- 융합 커널 `matmul_scale_masked_softmax`(`SoftMax.h:32-53`):
  - 외부 행 루프 `matmul_transpose_scale_row_loop`(`SoftMax.h:39`)에서 각 쿼리 행 `i`에 대해:
  - 내부 열 루프(`SoftMax.h:42-50`): `sum += A[i][k]*B[j][k]`로 **B를 전치된 형태로 접근**(QKᵀ를 명시적 전치 없이 계산, `SoftMax.h:46-48`), `tmp[j] = sum/scale_factor`로 즉시 스케일(`SoftMax.h:49`).
  - 행 단위로 `masked_sofmax(tmp, input_mask[i], result[i])` 호출(`SoftMax.h:51`).
  - **핵심 설계**: QKᵀ 한 행을 계산하자마자 그 행에 softmax를 적용 → 전체 `[seq][seq]` score 행렬을 만든 뒤 별도 softmax 패스를 도는 대신 **행 단위 융합**. 이는 dataflow/스트리밍 친화적 패턴(중간 score를 행 단위로만 보관, 추정상 BRAM 절감 의도).
- `masked_sofmax`(`SoftMax.h:16-29`): `mask[i]?exp(input):0`로 masked exponent 계산(`SoftMax.h:22`) 후 정규화. **수치 안정화용 max 차감(`x-max`)이 없음** → `double`이라 현재는 무난하나 저정밀/고정소수점 전환 시 overflow 위험(확인: `SoftMax.h:17-28`, max 차감 부재).
- 별도 `transpose_matmul`/`matmul_transpose_scale`(`MatMul.h:17-46`)도 존재하나, attention 경로는 융합 버전(`SoftMax.h`)을 사용. `MatMul.h`의 전치 버전은 미사용 가능성(추정).

### 3.6 `matmul` — 기본 행렬곱 (MatMul.h:2-15)

- 3중 루프(`matmul_row_loop`/`matmul_col_loop`/`matmul_result_loop`, `MatMul.h:4-13`). `result[i][j]=0` 초기화 후 `+= A[i][k]*B[k][j]` 누산.
- **프라그마 전혀 없음** → systolic/MAC array 아님. 순수 순차 누산 루프(확인: `MatMul.h:2-15`). 합성 시 II 최적화/언롤은 합성기 디폴트 또는 후속 작업에 의존(추정). 이름 라벨(`matmul_*_loop`)은 추후 directive 부착 지점 표식(추정).

### 3.7 `linear` / `ff` — Linear & FFN (Linear.h, FF.h)

- `linear`(`Linear.h:6-17`): `matmul` 후 행별 `vecadd`로 bias 추가(`Linear.h:13-16`). 가중치 레이아웃 `weights[hidden][cols]`로 받음 → PyTorch에서 전치 저장(`TestFilesGenerator.py:153` `transpose(weight)`)과 호환.
- `ff`(`FF.h:6-19`): `linear(input,W1,b1)` → `activation(relu)` → `linear(W2,b2)`(`FF.h:15-18`). 중간 차원 `hidden`. 현재 활성함수는 `relu` 하드코딩(`Activations.h:50`).

### 3.8 `layer_norm` (LayerNorm.h:5-32)

- 행(채널)별로 mean(`LayerNorm.h:14-19`) → variance(`LayerNorm.h:20-26`) → 정규화 `((x-mean)*gamma)/sqrt(var+eps)+beta`(`LayerNorm.h:29`). `hls::sqrt` 사용.
- 3패스 순차 구조(평균/분산/정규화), 프라그마 없음(확인: `LayerNorm.h:12-31`).

### 3.9 보조 커널

- `Activations.h`: `relu`(`:6-9`), `_gelu`/`gelu`(`:11-20`, `#ifdef GELU`), `_erf`/`erf`(Abramowitz-Stegun 근사, `:22-44`, `#ifdef ERF`). `activation()`는 relu 고정(`:50`). GELU/ERF는 컴파일 스위치로 대기(미사용, 추정).
- `Scale.h`: 요소별 `A OP scale_factor`(`OP` 기본 `/`, `:4-6`), UNROLL(`:13`).
- `Mask.h`: 요소별 곱 `input*mask`(`:12`).
- `Transpose.h`: 완전 UNROLL 전치(`:6,9`).
- `Concat.h`: `concat_cols`(헤드 결합, `:4-15`) / `concat_rows`(`:17-29`).
- `MatAdd.h`/`VecAdd.h`: residual/bias 덧셈(`MatAdd.h:9`, `VecAdd.h:7` UNROLL).

### 3.10 테스트 인프라 (Tests/, TestBenchAccel.cpp, TestFilesGenerator.py)

- `TestUtils.h`: `load_arr`(텍스트 파일에서 평탄화 배열 로드, `:7-17`), `compare_vec`(요소별 비교 + 상대오차%/미스매치 카운트 로깅, `:20-54`), `compare_mat`(행렬을 평탄화해 `compare_vec` 호출, `:56-59`).
- `TestBenchAccel.cpp`: HLS `-tb`로 등록되는 csim 드라이버(`run_hls.tcl:4`). `#ifdef ENCODER` 분기(`:23-78`)에서 12개 input 파일 로드 → `accel()` 호출 → `compare_mat`로 골든 대조. **입력 경로가 `/home/carlos/...`로 하드코딩**(`:6-21`) → 다른 환경에서 경로 수정 필요(확인: `TestBenchAccel.cpp:6-21`).
- `TestFilesGenerator.py`: PyTorch `nn.functional`로 각 레이어의 입력/정답을 생성(`encoder()`는 `:63-113`에서 전체 인코더를 PyTorch로 재현). `int_or_float` 플래그로 정수/실수 데이터 선택(`:17`). 가중치는 HLS 레이아웃에 맞춰 전치 저장(`:102,105,107,109`).
- `Transformer_dataflow.cpp`: VS 빌드용 main, 16종 테스트 enum 디스패치(`:9-124`). 현재 `Test_Encoder` 선택(`:48`).

---

## 4. 데이터 플로우

엔드투엔드(Encoder 1레이어, 확인된 호출 순서 기준):

```
input[SEQ][TOK], mask[SEQ][SEQ]
      │
      ▼  multi_head_att (MultiHeadAtt.h:7)
   ┌─ for head i: att_head (AttHead.h:6)
   │     Q = linear(input,Wq) ─┐
   │     K = linear(input,Wk) ─┤→ matmul_scale_masked_softmax (SoftMax.h:32)
   │     V = linear(input,Wv) ─┘    (QKᵀ → /√d → mask → row-wise softmax)
   │                                 → matmul(att, V)  (ScaledDotAtt.h:19)
   └─ concat_cols heads → linear(W_O)            → tmp1
      │
      ▼  matadd(input, tmp1)                      → tmp2   (residual, Encoder.h:36)
      ▼  layer_norm(tmp2, γ0, β0)                 → tmp3   (Encoder.h:39)
      ▼  ff: linear(W1)→relu→linear(W2)           → tmp4   (FF.h, Encoder.h:42)
      ▼  matadd(tmp3, tmp4)                       → tmp5   (residual, Encoder.h:45)
      ▼  layer_norm(tmp5, γ1, β1)                 → result (Encoder.h:47)
```

데이터플로우 특성(확인/추정):
- **온칩 버퍼링**: 모든 중간 텐서(`tmp1`~`tmp5`, Q/K/V, `softmax_att`)가 함수 로컬 배열 → 합성 시 BRAM/레지스터. 외부 DRAM 스트리밍 인터페이스(AXI master)나 `hls::stream`은 코드에 없음(확인: `Layers/*.h` 전반에 stream/AXI 부재).
- **융합 스트리밍 후보**: `matmul_scale_masked_softmax`의 행 단위 융합(`SoftMax.h:39-51`)이 유일하게 명시적 dataflow 친화 패턴. 나머지는 풀(full) 중간버퍼 후 다음 단계 진행.
- **파이프라인 오버랩 없음**: `#pragma HLS DATAFLOW`/`PIPELINE` 부재로 단계 간 task-level 병렬화는 미구현. "dataflow"는 목표 아키텍처 명명(추정).

---

## 5. HW/SW 매핑

| 구분 | 항목 | 근거 |
|---|---|---|
| HW(합성 대상) | `accel` 최상위 커널 | `run_hls.tcl:2` `set_top accel`, `Accel.cpp:4` |
| HW 디바이스 | `xck24-ubva530-2LV-c` (Versal/Kria 계열 part로 추정) | `run_hls.tcl:6` |
| HW 클럭 | 5ns 주기(=200MHz), uncertainty 2ns | `run_hls.tcl:7-8` |
| HW 흐름 | csim→(csynth→cosim→export IP) | `run_hls.tcl:10-16` (`hls_exec==2`일 때만 합성/export) |
| HW 인터페이스 | 명시적 AXI/INTERFACE 프라그마 없음 → 합성기 디폴트(배열 인자 = BRAM/ap_memory 추정) | `Accel.cpp:5-17`(프라그마 부재) |
| SW(테스트벤치) | `TestBenchAccel.cpp` (HLS csim 드라이버) | `run_hls.tcl:4` |
| SW(골든 생성) | `TestFilesGenerator.py` (PyTorch) | `TestFilesGenerator.py:2-4` |
| SW(VS 디버깅) | `Transformer_dataflow.cpp` + `.sln/.vcxproj` | `Transformer_dataflow.cpp:29` |
| 파라미터 주입 | 컴파일 타임 `#define` + 템플릿 인자 | `Definitions.h:5-12`, 템플릿 호출 `Accel.cpp:20` |

연산→HW 매핑(추정): matmul/linear → MAC 누산 루프(현재 directive 미부착, 합성기가 LUT/DSP 매핑), softmax/layernorm → `hls::exp`/`hls::sqrt`(DSP+LUT), 중간 텐서 → BRAM. **MAC array/systolic 구조는 코드 레벨에서 미구현**(확인: 명시적 PE 배열·체계적 데이터 재사용 코드 없음).

---

## 6. 빌드 · 실행

1. **골든 데이터 생성(SW)**: `python TestFilesGenerator.py Test_Encoder [float]` → `input1.txt`~`input12.txt`, `golden_result.txt` 생성(`TestFilesGenerator.py:290-334`). 인자로 테스트 종류 선택, 2번째 인자 `float`면 실수 데이터(`:289`).
2. **HLS csim/합성**: `sh "run_vitis_command _line.sh"` → 내부에서 `vitis_hls run_hls.tcl`(`run_vitis_command _line.sh:2`).
   - `run_hls.tcl`: project `Transformer_accel`, top `accel`, src `Accel.cpp`(`-ILayers`), tb `TestBenchAccel.cpp`(`-ITests`), part `xck24-ubva530-2LV-c`, clk 5ns → `csim_design` 실행(`run_hls.tcl:1-10`).
   - `csynth/cosim/export`는 `hls_exec==2`로 바꿔야 활성화(현재 1로 csim만, `run_hls.tcl:11-16`).
3. **빌드 모드 전환**: `Definitions.h:16`의 `#define ENCODER`를 다른 매크로로 바꿔 단위 커널 합성 가능(MULTIHEAD/ATTHEAD/FDFRWRD/LAYERNORM/DOTPRODATT/LINEAR/MATMUL/SOFTMAX/ACTIVATION). 단 §3.1의 컴파일 잔존 이슈 수정 필요.
4. **VS(Windows) csim 디버깅**: `Transformer_dataflow.sln` 열어 `Transformer_dataflow.cpp`의 `Test_types test` 변경 후 실행(`:48`).

주의: TestBench의 입력 경로가 `/home/carlos/Transformer_dataflow/`로 하드코딩(`TestBenchAccel.cpp:6-21`) → 실행 환경에 맞게 수정 필수.

---

## 7. 의존성

- **Vitis HLS 표준 헤더**: `ap_fixed.h`, `hls_math.h`(`Definitions.h:2-3`), `hls::exp`(`SoftMax.h:7`), `hls::sqrt`(`LayerNorm.h:29`). → Vitis HLS 툴체인 필수.
- **C++ STL**: `<iostream>`, `<fstream>`, `<sstream>`, `<cmath>`, `<vector>`, `<functional>`(`TestUtils.h:2-6`, `Mask.h:2-3`, `Activations.h:2-3`).
- **Python(SW 골든)**: `numpy`, `torch`(`nn`), `typing`(`TestFilesGenerator.py:2-5`). PyTorch가 정답 생성의 레퍼런스.
- **내부 include 그래프**(확인): `Accel.cpp`→`Encoder.h`→{`MultiHeadAtt.h`,`MatAdd.h`,`LayerNorm.h`,`FF.h`}; `MultiHeadAtt.h`→{`AttHead.h`,`Concat.h`,`Linear.h`}; `AttHead.h`→{`Linear.h`,`ScaledDotAtt.h`}; `ScaledDotAtt.h`→{`MatMul.h`,`Transpose.h`,`Mask.h`,`SoftMax.h`,`Scale.h`}; `Linear.h`→{`VecAdd.h`,`MatMul.h`}; `FF.h`→{`Linear.h`,`Activations.h`}.
- 외부 빌드시스템(CMake/Makefile) 없음 — tcl + 셸 + VS 프로젝트만 존재.

---

## 8. 강점 · 한계

### 강점
- **명료한 모듈러 템플릿 설계**: 연산별 단일 책임 헤더, 컴파일 타임 파라미터화로 토큰/시퀀스/헤드 차원 변경이 용이(`Definitions.h:5-12`).
- **함수 트리 = 알고리즘 구조**: Encoder→MHA→Head→Attention의 호출 계층이 Transformer 수식과 1:1 대응, 가독성·검증성 우수(`Encoder.h`, `MultiHeadAtt.h`).
- **레이어별 단위테스트 + PyTorch 골든**: 16종 테스트와 자동 상대오차 비교로 정확성 검증 체계가 갖춰짐(`TestFilesGenerator.py`, `TestUtils.h:20-54`).
- **Attention 융합 커널**: QKᵀ·scale·mask·softmax 행 단위 융합으로 중간버퍼 절감 패턴 확보(`SoftMax.h:32-53`).

### 한계
- **HLS 최적화 미적용**: `#pragma HLS DATAFLOW/PIPELINE/ARRAY_PARTITION`, `hls::stream` 부재. UNROLL만 산발적(`Scale.h:13`, `Transpose.h`, `VecAdd.h`). 현 상태로는 고처리량 미달성(추정) — "dataflow"는 목표 명칭.
- **데이터 타입 `double`**: 고정소수점/양자화 미적용(`Definitions.h:13-14`). FPGA 리소스/속도 측면에서 비현실적, 골든 레퍼런스 단계로 판단(추정).
- **MAC/systolic array 부재**: matmul이 평범한 3중 루프(`MatMul.h:2-15`), 명시적 PE 배열·데이터 재사용 구조 없음.
- **수치 안정성**: softmax에 max 차감 없음(`SoftMax.h:17-28`) → 저정밀 전환 시 위험.
- **이식성 이슈**: TestBench 절대경로 하드코딩(`TestBenchAccel.cpp:6-21`), 비-ENCODER 모드 컴파일 잔존 버그(`Accel.cpp:69,113`).
- **AXI 인터페이스 미정의**: 외부 메모리 스트리밍/DMA 경로 부재 → 실제 SoC 통합 단계 추가 필요.
- **단일 인코더 레이어**: `encoder()`가 1레이어만 구성(`Encoder.h`), 다층 스택/임베딩/포지셔널 인코딩·분류 헤드는 미포함.

---

## 9. 우리 프로젝트(PRJXR-HBTXR) 시사점

전제(추정): 본 프로젝트는 **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적** 방향.

1. **알고리즘 골든 레퍼런스로 직접 활용 가능**: 본 repo는 비트정확 골든이 아닌 `double` 기능 레퍼런스이나, Encoder 전 단계(MHA/LN/FFN/residual)가 PyTorch와 교차검증되어 있어(`TestFilesGenerator.py`) 우리 양자화/고정소수점 가속기의 **정확성 회귀 기준선**으로 재사용 가치 높음.
2. **HG-PIPE식 파이프라인화의 출발점**: 현 코드는 `#pragma HLS DATAFLOW`/`PIPELINE`/`ARRAY_PARTITION`이 비어 있어, HG-PIPE의 핵심인 **전(全)레이어 데이터플로우 + on-chip 스트리밍**을 적용할 명확한 삽입 지점(`matmul_*_loop` 라벨, `Encoder.h`의 단계 경계)이 표식되어 있음. 우리 가속기에서 이 골격에 stream/dataflow를 입히면 HG-PIPE 구조로 진화 가능(추정).
3. **Attention 융합 패턴 차용**: `matmul_scale_masked_softmax`의 행 단위 융합(`SoftMax.h:32-53`)은 score 행렬 BRAM 절감에 유효 — 우리 ViT attention 블록(특히 작은 시퀀스의 시선추적 토큰)에 적합.
4. **XR 시선추적 적용 시 조정 필요**: 현 파라미터(`SEQ_LEN`, `TOKEN_LEN`)는 `Definitions.h`에서 10(소형)·테스트는 128/384(`Transformer_dataflow.cpp:6-7`)로 불일치. 시선추적용 경량 ViT(작은 패치/시퀀스, 저지연)에 맞춰 차원 재설정과 고정소수점화·MAC array 도입이 선결 과제.
5. **보완해야 할 격차**: (a) 다층 인코더 스택·임베딩/포지셔널 인코딩, (b) AXI/DMA 외부메모리 인터페이스, (c) 양자화(ViT-Quantization repo와 연계), (d) MAC/systolic array — 본 repo는 이들이 없으므로 우리 통합 설계에서 채워야 함.

---

## 근거 표기 요약

- 확인(라인 근거): §3~§7의 모든 `파일명:라인` 인용은 실제 Read한 소스 기준.
- 추정: dataflow 미구현 사유, double=골든단계, MAC array 부재 해석, HW 매핑 세부, HG-PIPE 진화 가능성 등 — "추정" 명시.
- 확인 불가: `Tests/Test.h`(인클루드 허브로 추정, 내용 미열람), 실제 합성 리소스/타이밍 수치(합성 산출물 `.gitignore`로 미커밋), 디바이스 part `xck24-ubva530`의 정확한 시리즈 — "확인 불가".
