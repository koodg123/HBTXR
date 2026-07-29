# HG-PIPE `src/` + `case/` 병렬도 튜닝 치트시트

이 문서는 [SRC-CASE-MODULE-GUIDE.md](./SRC-CASE-MODULE-GUIDE.md) 를 읽은 뒤, 실제 resource fit / latency trade-off 관점에서 어떤 병렬도 knob를 먼저 만져야 하는지 빠르게 판단하기 위한 운영 문서다.

중요한 전제:

- 이 문서는 공통 `workspace/hardware/src/` 구조를 기준으로 설명한다.
- 실제 적용은 `workspace/hardware/case_*` 또는 `workspace/hardware/case/` 안의 concrete case 상수에 반영된다.
- 아래 영향도는 코드 구조 기반 추정이며, 최종 수치는 target / bitwidth / HLS binding / post-synth 결과에 따라 달라질 수 있다.

## 1. 먼저 봐야 할 병렬도 상수

### 전역적으로 반복 등장하는 상수

- `TP`
  - token 방향 병렬도
- `CIP`
  - input channel parallelism
- `COP`
  - output channel parallelism
- `CP`
  - quant / split / small stage의 channel parallelism
- `CHAP`
  - `MLP` hidden-channel adapter parallelism

### 대표 위치

- `PATCH_EMBED`
  - `workspace/hardware/case/PATCH_EMBED.cpp`
- `ATTN`
  - `workspace/hardware/case/ATTN.cpp.template`
  - generated layer case: `workspace/hardware/case/ATTN0.cpp` .. `ATTN11.cpp`
- `MLP`
  - `workspace/hardware/case/MLP.cpp.template`
  - generated layer case: `workspace/hardware/case/MLP0.cpp` .. `MLP11.cpp`

## 2. 코드 기준 cost model

실제 자원 증가는 대부분 `../workspace/hardware/src/matmul.h` 의 아래 구조에서 결정된다.

### 2.1 `Matmul`의 기본 scaling

- MAC lane 수:
  - `TP * COP * CIP`
- 누산 register tile 크기:
  - `TP * COP`
- input window scratchpad 폭:
  - `TP * CI`, 단 실제 LUTRAM pressure는 `CIP` unroll과 강하게 연결
- 주요 loop trip:
  - `TT * COT * CIT`
  - 여기서 `COT = CO / COP`, `CIT = CI / CIP`

즉:

- `CIP`를 줄이면
  - MAC lane 수 감소
  - input-side LUTRAM / window buffer pressure 감소
  - `CIT` 증가로 latency 증가
- `COP`를 줄이면
  - MAC lane 수 감소
  - output accumulator / adder / bias fanout 감소
  - `COT` 증가로 latency 증가
- `TP`를 줄이면
  - 거의 모든 branch에서 병렬도가 줄어 latency 타격이 가장 큼

### 2.2 왜 `COP`가 자주 첫 번째 knob인가

`../workspace/hardware/src/matmul.h` 를 보면:

- `weight_arr`는 `COP`, `CIP` 둘 다에 따라 array reshape 된다.
- `bias_arr`는 `COP` 방향으로 reshape 된다.
- `vec_o[TP*COP]`가 output-stationary partial sum bank 역할을 한다.

따라서 `COP`를 줄이면:

- partial sum register 수
- bias broadcast fanout
- output-side adder tree pressure

가 함께 줄어드는 경향이 있다.

실무적으로는 `LUT as Logic`이 크게 넘을 때 `COP`가 `CIP`보다 먼저 듣는 경우가 많다.

## 3. 모듈별 병렬도 민감도

## 3.1 `PATCH_EMBED`

대표 코드:

- `../workspace/hardware/src/patch_embed.h`

핵심 상수:

- `TP`
- `CIP`
- `CIAP`
- `COP`

구조 포인트:

- `step1_cache_window()`의 `wb[TP][CI]`는 `LUTRAM`에 bind 되어 있다.
- weight는 `HGPIPE_PATCH_EMBED_WEIGHT_STORAGE_STYLE`에 따라 `BRAM/URAM/LUTRAM`이 갈린다.

튜닝 해석:

- `CIP` 감소
  - input window `wb` pressure 감소
  - 곱셈 lane 감소
  - latency 증가
- `COP` 감소
  - output stationary register / bias fanout 감소
  - logic fit에 더 직접적으로 듣는 편
- `CIAP` 감소
  - adapter packing trip 증가
  - compute보다 I/O packing 영향이 큼

권장 순서:

1. `COP`
2. `CIP`
3. `CIAP`
4. `TP`

## 3.2 `MLP`

대표 코드:

- `../workspace/hardware/src/mlp.h`

핵심 상수:

- `CHAP`
- `M1_CIP`, `M1_COP`
- `M2_CIP`, `M2_COP`

구조 포인트:

- `MLP`는 FC1, GeLU, FC2의 순차 dataflow다.
- FC1, FC2 둘 다 `Matmul` 기반이라 resource의 대부분을 먹는다.
- residual FIFO는 `resi_sm depth=512`로 고정돼 있고, 필요 시 `URAM`으로 보낼 수 있다.

튜닝 해석:

- `CHAP` 감소
  - hidden path stream 폭 자체를 줄인다
  - 보통 deadlock 위험은 낮고 fit 효과는 꽤 좋다
- `M1_COP`, `M2_COP` 감소
  - logic/register pressure 감소
- `M1_CIP`, `M2_CIP` 감소
  - LUTRAM / input window pressure 감소
- `TP` 감소
  - 전체 latency 타격이 큼

권장 순서:

1. `CHAP`
2. `M1_COP`, `M2_COP`
3. `M1_CIP`, `M2_CIP`
4. `TP`

## 3.3 `ATTN`

대표 코드:

- `../workspace/hardware/src/attn.h`
- `../workspace/hardware/src/head_split.h`
- `../workspace/hardware/src/reshaper.h`

핵심 상수:

- `MATMUL_QKV_CIP`, `MATMUL_QKV_COP`
- `MATMUL_R_CIP`, `MATMUL_R_COP`
- `MATMUL_A_CIP`, `MATMUL_A_COP`
- `MATMUL_O_CIP`, `MATMUL_O_COP`
- 각종 head FIFO depth

구조 포인트:

- `Q/K/V` projection은 3개 `Matmul`이 완전히 복제된다.
- `QK`와 `RV`도 head별로 3개씩 복제된다.
- `Reshaper::reorder()`는 읽기와 쓰기 phase가 순차라 burst/backpressure에 민감하다.

튜닝 해석:

- `MATMUL_O_*`
  - output projection만 건드리므로 상대적으로 안전
  - deadlock 위험이 낮다
- `MATMUL_QKV_*`
  - Q/K/V 세 branch의 균형을 바꾼다
  - 과하게 줄이면 split/reshape/downstream과 균형이 깨질 수 있다
- `MATMUL_A_*`
  - `RQ -> VQ -> RV` 경로의 병목과 직접 연결
  - deadlock 민감도가 높다
- `MATMUL_R_*`
  - `QQ -> KQ -> softmax` 경로와 직접 연결
  - deadlock 민감도가 높다

권장 순서:

1. `MATMUL_O_CIP`, `MATMUL_O_COP`
2. `MATMUL_QKV_COP`
3. `MATMUL_QKV_CIP`
4. `MATMUL_A_*`, `MATMUL_R_*`
5. `TP`

중요:

- `ATTN`은 병렬도만 줄이는 것보다 `deadlock risk`를 같이 봐야 한다.
- `KQ/VQ reshape`, `softmax`, `merge`는 stream elasticity에 민감하다.
- 따라서 `FIFO depth`를 크게 줄인 상태에서 `QKV/A/R`까지 같이 줄이는 건 가장 위험한 조합이다.

## 3.4 `HEAD`

대표 코드:

- `workspace/hardware/case/HEAD.cpp`
- `workspace/hardware/src/head.h`

해석:

- 전체 모델 대비 비중이 작다.
- `HEAD`는 fit 최적화의 첫 타깃이 아니다.

권장 순서:

1. 다른 모듈을 먼저 줄인다.
2. 정말 끝까지 안 맞을 때만 건드린다.

## 4. 메모리 스타일과 병렬도의 관계

병렬도만 줄이는 것과 별개로, 이 코드베이스는 `LUTRAM` 사용량이 커질 수 있는 구조가 여러 군데 있다.

### 즉시 확인해야 할 위치

- `../workspace/hardware/src/matmul.h`
  - `wb`는 `LUTRAM`
  - `bias_arr`도 `LUTRAM`
  - `WEIGHT_RAM_STYLE`에 따라 weight는 `BRAM/LUTRAM`
- `../workspace/hardware/src/reshaper.h`
  - `buffer`가 `lutram`
- `../workspace/hardware/src/patch_embed.h`
  - `wb`가 `LUTRAM`
  - weight는 `HGPIPE_PATCH_EMBED_WEIGHT_STORAGE_STYLE`로 제어

해석:

- `LUT as Memory`가 넘는 경우:
  - `CIP` 축소
  - `Reshaper` buffer를 BRAM/URAM 선택형으로 변경
  - `QK/RV` weight storage를 `LRAM -> BRAM`
  - `PatchEmbed` weight storage를 BRAM/URAM으로 고정

## 5. 실제 튜닝 우선순위

### 5.1 `LUT as Logic`이 먼저 넘는 경우

1. `PATCH_EMBED COP`
2. `MLP CHAP`
3. `MLP M1_COP / M2_COP`
4. `ATTN MATMUL_O_CIP / O_COP`
5. `PATCH_EMBED CIP`
6. `MLP M1_CIP / M2_CIP`

### 5.2 `LUT as Memory`가 먼저 넘는 경우

1. `PatchEmbed` / `Matmul` window buffer 관련 `CIP`
2. `Reshaper` buffer storage style 변경
3. `QK/RV` weight RAM style 변경
4. head FIFO depth 조정

### 5.3 `BRAM`이 먼저 넘는 경우

1. 큰 FIFO depth 정리
2. 불필요한 BRAM binding을 LUTRAM/URAM 쪽으로 재분배
3. 그다음에 병렬도 감소

## 6. 저위험 / 고위험 knob

### 저위험

- `PATCH_EMBED COP`
- `MLP CHAP`
- `MLP M1_COP`, `M2_COP`
- `ATTN MATMUL_O_CIP`, `MATMUL_O_COP`

### 중간 위험

- `PATCH_EMBED CIP`
- `MLP M1_CIP`, `M2_CIP`
- `ATTN MATMUL_QKV_CIP`, `MATMUL_QKV_COP`

### 고위험

- `ATTN MATMUL_A_CIP`, `MATMUL_A_COP`
- `ATTN MATMUL_R_CIP`, `MATMUL_R_COP`
- `TP`
- `ATTN`의 큰 FIFO depth를 병렬도 축소와 동시에 크게 줄이는 조합

## 7. 구조 변경이 필요한 경우

병렬도 숫자만 만져서는 안 풀릴 때는 다음이 더 큰 효과를 낼 수 있다.

- `Q/K/V` projection fusion
  - 현재는 3개 `Matmul`이 완전히 복제된다.
- head serialization 또는 2-way head parallelism
  - 현재는 `QK`, `RV`가 3-head full replication 구조다.
- `Reshaper` buffer storage style 분리
  - 현재 `lutram` 고정이라 LUTRAM pressure가 크다.

이 셋은 효과는 크지만, 코드 영향 범위도 커서 “숫자 튜닝” 다음 단계로 보는 것이 맞다.

## 8. 보드별 현재 profile 스냅샷

이 섹션은 현재 `case_*`에 반영된 대표 값 기준의 빠른 스냅샷이다.

주의:

- 여기 값은 representative template 기준이다.
- 실제 layer별 case가 이미 concretize 되어 있으면 `ATTN0..11.cpp`, `MLP0..11.cpp`도 함께 봐야 한다.
- `zu15eg`는 deadlock 대응 때문에 `ATTN`이 한 번 rollback / restore를 거친 상태다.

### 8.1 `zu15eg`

- `PATCH_EMBED`
  - `CIP=8`, `CIAP=1`, `COP=8`
- `ATTN`
  - `QKV_CIP=6`, `QKV_COP=12`
  - `R_CIP=4`, `R_COP=7`
  - `A_CIP=7`, `A_COP=4`
  - `O_CIP=4`, `O_COP=4`
  - `O_USE_DSP=true`
  - `RESI_FIFO_DEPTH=4096*3`
  - `QQ_HEAD_FIFO_DEPTH=8000`
  - `KQ_RESHAPE_HEAD_FIFO_DEPTH=512`
  - `VQ_TRANSPOSE_HEAD_FIFO_DEPTH=512`
- `MLP`
  - `CHAP=1`
  - `M1_CIP=4`, `M1_COP=8`, `M1_USE_DSP=true`
  - `M2_CIP=8`, `M2_COP=4`, `M2_USE_DSP=true`

해석:

- `ATTN`의 `QKV/A/FIFO`는 deadlock 회피를 위해 baseline에 가깝게 유지했다.
- 대신 `MLP`와 `ATTN O-path`를 더 줄이고, MLP/O-path 곱셈을 DSP로 보내 LUT logic pressure를 낮추는 fit-C 상태다.

### 8.2 `zcu102`

- `PATCH_EMBED`
  - `CIP=8`, `CIAP=1`, `COP=4`
- `ATTN`
  - `QKV_CIP=4`, `A_CIP=4`, `O_CIP=6`
  - `RESI_FIFO_DEPTH=1024`
  - `QQ_HEAD_FIFO_DEPTH=2048`
  - `KQ/VQ reshape depth=256`
- `MLP`
  - `CHAP=1`
  - `M1_CIP=8`, `M1_COP=12`
  - `M2_CIP=12`, `M2_COP=8`

해석:

- 이미 꽤 공격적으로 줄어든 상태다.
- 다음 단계는 `storage style`과 `ATTN O-path`의 추가 정리가 더 중요하다.

### 8.3 `zcu104`

- `PATCH_EMBED`
  - `CIP=4`, `CIAP=1`, `COP=4`
- `ATTN`
  - `QKV_CIP=3`, `A_CIP=3`, `O_CIP=4`
  - `RESI_FIFO_DEPTH=1024`
  - `QQ_HEAD_FIFO_DEPTH=1024`
  - `KQ/VQ reshape depth=128`
- `MLP`
  - `CHAP=1`
  - `M1_CIP=6`, `M1_COP=12`
  - `M2_CIP=12`, `M2_COP=8`

해석:

- full-model 유지 조건에서 이미 낮은-throughput profile에 들어와 있다.
- 숫자만 더 줄이기보다 memory binding과 구조 변경 쪽 검토 가치가 더 커진다.

### 8.4 `ultra96v2`

- `PATCH_EMBED`
  - `CIP=4`, `CIAP=1`, `COP=4`
- `ATTN`
  - `QKV_CIP=3`, `A_CIP=3`, `O_CIP=4`
  - `QK_MATMUL_WEIGHT_RAM_STYLE=BRAM`
  - `RV_MATMUL_WEIGHT_RAM_STYLE=BRAM`
  - `RESI_FIFO_DEPTH=512`
  - `QQ_HEAD_FIFO_DEPTH=1024`
  - `KQ/VQ reshape depth=128`
- `MLP`
  - `CHAP=1`
  - `M1_CIP=4`
  - `M2_CIP=8`

해석:

- feasibility-oriented profile에 가깝다.
- 숫자 축소보다 storage style / 구조 변경 위주로 보는 편이 낫다.

## 9. 보드별 다음 fit recipe

이 섹션은 “현재 값에서 다음으로 무엇을 줄일까”를 빠르게 선택하기 위한 운영 메모다.

### 9.1 `zu15eg fit-D / 3ns resource-fit`

현재 적용 상태:

1. ZU15EG 전용 design / target config로 `clock_period_ns=3.0`을 사용한다.
2. `ATTN O-path`: `O_CIP/O_COP 6 -> 4`, `O_USE_DSP false -> true`
3. `MLP`: `M1_CIP 8 -> 4`, `M1_COP 12 -> 8`, `M1_USE_DSP false -> true`
4. `MLP`: `M2_CIP 16 -> 8`, `M2_COP 8 -> 4`, `M2_USE_DSP false -> true`
5. `ATTN` 계산 병렬도와 FIFO depth는 유지하고, `Reshaper buffer`, `resi_sm`, `qq_sm_head*`를 ZU15EG에서 URAM으로 이동한다.
6. `PATCH_EMBED` weight storage는 ZU15EG에서 URAM path를 사용한다.

지키는 원칙:

- `ATTN QKV/A/FIFO`는 그대로 둔다.
- deadlock을 다시 만들 수 있는 `KQ/VQ/softmax` 경로는 건드리지 않는다.
- 숫자 병렬도를 더 줄이기 전에 storage binding으로 `LUT as Memory`와 BRAM pressure를 낮춘다.
- 3ns timing closure는 resource fit 이후 별도 이슈로 분리한다.

목표:

- `LUT logic`보다 먼저 `LUTRAM/BRAM` pressure를 URAM으로 재분배한다.
- `step4-synth`의 resource DRC를 먼저 통과한다.
- `ATTN8` deadlock을 재발시키지 않는다.

다음 fallback:

1. URAM이 초과되면 `QQ_HEAD_FIFO` URAM binding을 먼저 끈다.
2. `LUT as Memory`만 남으면 `QK/RV weight RAM style: LRAM -> URAM`을 검토한다.
3. BRAM만 남으면 `KQ/VQ reshape head FIFO`의 URAM binding macro를 추가한다.
4. 그래도 안 맞으면 `ATTN` head serialization 또는 block reuse를 검토한다.

### 9.2 `zcu102 fit-C`

추천 순서:

1. `QK/RV weight RAM style: LRAM -> BRAM`
2. `ATTN MATMUL_O_COP 6 -> 4`
3. `Reshaper buffer: LUTRAM -> BRAM/URAM 선택형`
4. 필요하면 `PATCH_EMBED CIP 8 -> 6`

지키는 원칙:

- 이미 `CHAP=1`이라 hidden-path 쪽 여유가 적다.
- 다음 타격은 memory style과 output-side 정리에서 찾는 편이 낫다.

### 9.3 `zcu104 fit-C`

추천 순서:

1. `QK/RV weight RAM style: LRAM -> BRAM`
2. `Reshaper buffer: LUTRAM -> BRAM/URAM 선택형`
3. 필요하면 `PATCH_EMBED CIP 4 -> 3`
4. 그다음은 구조 변경 검토

지키는 원칙:

- `QKV/A`는 이미 많이 줄었으므로 더 줄이면 deadlock/latency 손실 대비 효과가 작을 수 있다.

### 9.4 `ultra96v2 fit-C`

추천 순서:

1. `Reshaper buffer: LUTRAM -> BRAM/URAM 선택형`
2. 구조 변경 검토
   - `QKV fusion`
   - head serialization / 2-way head parallelism

지키는 원칙:

- 숫자만 더 줄이는 것은 latency 손실이 너무 커질 수 있다.
- 이 구간부터는 구조 변경의 효과가 더 크다.

## 10. 문서와 함께 읽을 것

병렬도 변경이 실제 board fit에 어떤 영향을 주는지는 아래 문서와 함께 봐야 한다.

- [SRC-CASE-MODULE-GUIDE.md](./SRC-CASE-MODULE-GUIDE.md)
- [MULTI-BOARD-VALIDATION.md](./MULTI-BOARD-VALIDATION.md)

`SRC_CASE_MODULE_GUIDE.md`는 구조를 설명하고, 이 문서는 “어디를 먼저 건드릴지”를 설명한다.
