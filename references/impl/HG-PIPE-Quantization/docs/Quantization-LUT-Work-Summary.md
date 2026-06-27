# HG-PIPE Quantization 및 LUT 작업 정리

작성일: 2026-06-24

이 문서는 `HG-PIPE-Quantization`에서 지금까지 수행한 quantization, LUT approximation, calibration, hardware pass 분석 및 구현 내용을 정리한다. 기준으로 사용한 공개 HG-PIPE 자료는 다음과 같다.

- `../ICCAD24-HG-PIPE`
- `../ICCAD24-HG-PIPE/case/refs`
- `../ICCAD24-HG-PIPE/src`
- `../Vision Transformer Acceleration with Hybrid-Grained Pipeline.pdf`

주 작업 디렉토리:

- `HG-PIPE-Quantization`

## 작업 범위

이번 작업은 공개 HG-PIPE 코드와 `case/refs`에 저장된 배포 artifact를 기반으로 inference quantization path를 재구성하고, 추가적인 LUT 생성/캘리브레이션 코드를 구현한 것이다.

중요한 제한 사항:

- 공개 checkout에는 원래 논문 실험에 사용된 전체 QAT/training-time calibration/export pipeline이 포함되어 있지 않다.
- 따라서 새로 구현한 LUT calibration 코드는 공개 artifact의 hardware contract를 따르는 재현 가능한 replacement generator이지, 원본 private QAT 코드와 동일하다고 주장하지 않는다.

포함된 내용:

- Group-vector quantization
- Linear operator의 scale/zero-point 정책
- Dyadic scaling factor
- KL-divergence 기반 calibration
- LayerNorm, Softmax, GeLU용 nonlinear LUT generation/calibration
- `cursor = (x + b) >> s` 기반 hardware LUT address 계산
- LayerNorm, Softmax, GeLU hardware pass 동작
- LUT entry 개수와 value precision 정리
- HG-PIPE `case/refs` 기준 bit-exact 검증 결과

## 구현 파일

| 파일 | 내용 |
|---|---|
| `hgpipe_quantization/lut_calibration.py` | ReQuant, GeLU-ReQuant, LayerNorm RSQRT, Softmax EXP/RECIP LUT와 scalar 생성 |
| `hgpipe_quantization/quantization_scheme.py` | Group-vector quantization, KL dyadic scale calibration, nonlinear LUT wrapper, hardware index helper |
| `hgpipe_quantization/cli.py` | group-vector, dyadic calibration, LUT calibration CLI 추가 |
| `hgpipe_quantization/__init__.py` | 새 quantization/LUT API export |
| `hgpipe_quantization/paper_equivalence.py` | replacement LUT generator가 원본 QAT equivalence 체크를 잘못 통과하지 않도록 제외 |
| `tests/test_lut_calibration.py` | LUT calibration 및 CLI output 테스트 |
| `tests/test_quantization_scheme.py` | group-vector, dyadic scale, nonlinear LUT wrapper, hardware index 테스트 |
| `tests/test_cli.py` | 새 CLI command 노출 테스트 |

## CLI 추가 사항

### Group-Vector Quantization

```bash
python -m hgpipe_quantization.cli \
  --source ../ICCAD24-HG-PIPE \
  quantize-group-vector-npy \
  --input-npy samples.npy \
  --tensor-role activation \
  --bits 8 \
  --group-size 192 \
  --quantized-npy quantized.npy \
  --scales-npy scales.npy \
  --json quantization.json
```

동작:

- Activation/X: last dimension 기준 per-token 또는 token-group quantization
- Weight/W: output channel 기준 per-channel quantization

### Linear Dyadic Scale Calibration

```bash
python -m hgpipe_quantization.cli \
  --source ../ICCAD24-HG-PIPE \
  calibrate-linear-dyadic-npy \
  --input-npy samples.npy \
  --bits 8 \
  --histogram-bins 2048 \
  --json dyadic_scale.json
```

동작:

1. percentile clip 후보를 탐색한다.
2. 각 후보로 quantize/dequantize를 수행한다.
3. reference histogram과 candidate histogram의 KL divergence를 계산한다.
4. KL divergence가 가장 작은 clip을 선택한다.
5. 최종 scale을 dyadic form으로 근사한다.

```text
effective_scale = multiplier / 2^shift
```

### Nonlinear LUT Calibration

```bash
python -m hgpipe_quantization.cli \
  --source ../ICCAD24-HG-PIPE \
  calibrate-lut-npy \
  --kind softmax \
  --input-npy samples.npy \
  --entries 32 \
  --recip-entries 64 \
  --json softmax_lut.json \
  --txt-dir generated_refs \
  --stem attn_0_softmaxq
```

지원하는 `--kind`:

- `requant`
- `gelu-requant`
- `rsqrt`
- `softmax`

## Quantization Scheme

### Granularity

구현한 group-vector quantization 정책은 다음과 같다.

| Tensor | Granularity | Scale shape |
|---|---|---|
| Activation `X` | Per-token 또는 token-group | `[tokens, channel_groups]` |
| Weight `W` | Per-output-channel | `[out_channels]` |

Activation은 token별 feature vector를 group 단위로 나누어 scale을 잡고, weight는 linear layer의 output channel별 row에 대해 scale을 잡는다.

### Linear Operator의 Scale과 Zero-Point

Linear unit, convolution, SMU, RMU에 대한 기본 정책:

- Weight scale: per-output-channel
- Activation scale: per-token 또는 group-vector
- 기본 zero-point: signed symmetric path에서는 `0`
- output/requant scale: floating-point affine quantizer가 아니라 integer/dyadic scalar 또는 LUT table로 표현

일반적인 linear accumulator requant 관계:

```text
y_int ~= round((Sx * Sw / Sy) * acc_int)
```

HG-PIPE에서는 이를 hardware-friendly하게 표현하기 위해 다음 중 하나로 변환한다.

- `multiplier / 2^shift`
- table lookup scalar tuple
- fused LUT output

### Dyadic Quantization

Dyadic scale은 다음 형태를 쓴다.

```text
scale ~= multiplier / 2^shift
```

이 방식은 hardware에서 general division을 shift 기반 연산으로 바꾸기 위한 것이다.

Calibration 절차:

1. calibration tensor를 수집한다.
2. 여러 percentile clip 후보를 만든다.
3. 각 후보에 대해 quantize/dequantize를 수행한다.
4. reference distribution과 candidate distribution의 KL divergence를 계산한다.
5. 가장 작은 KL divergence를 갖는 clip을 선택한다.
6. 선택된 scale을 `multiplier / 2^shift`로 근사한다.

## LUT Address 계산

HG-PIPE LUT의 핵심 address 계산은 다음이다.

```text
cursor = (x + b) >> s
cursor = clamp(cursor, 0, bound)
output = table[cursor]
```

해석:

- LUT index 값 자체를 power-of-two로 만드는 것이 아니다.
- affine address scale factor를 `2^-s` shift 형태로 근사하는 것이다.
- 목적은 LUT address 계산에서 multiplier/divider를 제거하는 것이다.

원래 affine index가 다음과 같다면:

```text
idx ~= floor(alpha * x + beta)
```

HG-PIPE는 이를 다음처럼 근사한다.

```text
idx ~= (x + b) >> s
```

여기서 `b`는 offset, `s`는 power-of-two shift scale, `bound`는 최대 LUT index이다.

## Nonlinear LUT Calibration

### GeLU-ReQuant

목적:

- GeLU와 output quantization을 하나의 LUT로 fuse한다.
- runtime에서 tanh, erf, polynomial GeLU hardware를 사용하지 않는다.

Global artifact:

```text
scalars = [b, s, bound]
geluq_table[64]
```

Calibration 과정:

1. GeLU input sample을 수집한다.
2. percentile 기반 clipping range를 선택한다.
3. LUT entry 수를 정한다. 공개 HG-PIPE artifact는 64 entries를 사용한다.
4. `b`, `s`, `bound`를 계산한다.
5. 각 entry index에 대응하는 input coordinate를 만든다.
6. 해당 coordinate에서 GeLU 값을 계산한다.
7. output quantization range로 clamp/round한다.
8. table과 scalar를 저장한다.

Runtime:

```cpp
cursor = (x + b) >> s;
cursor = clamp(cursor, 0, bound);
y = geluq_table[cursor];
```

### LayerNorm RSQRT

목적:

- `1 / sqrt(var + eps)` 계산을 직접 hardware로 구현하지 않고 RSQRT LUT로 근사한다.
- variance는 layer 전체 global 값이 아니라 runtime token row에서 계산되는 local 값이다.

Global artifact:

```text
scalars = [C_1_m, C_1_s, b, s1, bound, s2, clamp_bits]
lnw[C]
lnb[C]
rsqrt_table[128]
```

Local runtime 값:

```text
mean[token]
var_sum[token]
rsqrt[token]
```

Calibration 과정:

1. LayerNorm 입력에서 token별 variance-sum sample을 수집한다.
2. variance-sum distribution에 대해 clipping range를 정한다.
3. RSQRT LUT entry 수를 정한다. 공개 HG-PIPE artifact는 128 entries이다.
4. `var_sum -> cursor` mapping을 위한 `b`, `s1`, `bound`를 계산한다.
5. 각 entry에 대해 fixed-point RSQRT 값을 계산한다.
6. table과 scalar를 저장한다.

Runtime:

```cpp
mean = round(sum(x) / C);
var_sum = sum((x - mean) * (x - mean));
cursor = clamp((var_sum + b) >> s1, 0, bound);
rsqrt = rsqrt_table[cursor];
y = ((x - mean) * rsqrt * lnw[c] + lnb[c]) >> s2;
y = signed_clamp(y, clamp_bits);
```

### Softmax EXP / RECIP

목적:

- exponential과 reciprocal을 runtime hardware로 직접 계산하지 않는다.
- row-local max와 row-local sum을 사용해 numerically stable softmax를 수행한다.

Global artifact:

```text
exp_table[32]
recip_table_one[64]
recip_table_two[64]
scalars = [
  b1, s1, bound1,
  b2_one, s2_one, bound2_one, b3_one, s3_one,
  b2_two, s2_two, bound2_two, b3_two, s3_two,
  clamp_bits
]
```

Local runtime 값:

```text
max_val[row]
exp_score[row][col]
acc_val[row] = sum(exp_score)
recip_val[row]
segment_select[row]
```

Calibration 과정:

1. attention score sample을 수집한다.
2. `max - x` distribution을 기준으로 EXP LUT range를 잡는다.
3. `max - x -> exp` table을 만든다.
4. `sum(exp)` distribution을 수집하거나 추정한다.
5. reciprocal은 dynamic range가 크므로 table one/two 두 segment로 나눈다.
6. segment별 reciprocal table과 requant scalar `b3/s3`를 저장한다.

Runtime:

```cpp
max_val = max(row);

minus = max_val - x;
cursor1 = clamp((minus + b1) >> s1, 0, bound1);
exp_score = exp_table[cursor1];
acc_val += exp_score;

cursor_one = (acc_val + b2_one) >> s2_one;
if (cursor_one > bound2_one) {
    cursor_two = clamp((acc_val + b2_two) >> s2_two, 0, bound2_two);
    recip = recip_table_two[cursor_two];
    b3 = b3_two;
    s3 = s3_two;
} else {
    cursor_one = clamp(cursor_one, 0, bound2_one);
    recip = recip_table_one[cursor_one];
    b3 = b3_one;
    s3 = s3_one;
}

y = (exp_score * recip + b3) >> s3;
y = unsigned_clamp(y, clamp_bits);
```

## Global 값과 Local 값

| Operator | Global calibration/export 값 | Local runtime 값 |
|---|---|---|
| GeLU | `b`, `s`, `bound`, `geluq_table` | per-element `cursor` |
| LayerNorm | `C_1_m`, `C_1_s`, `b`, `s1`, `bound`, `s2`, `clamp_bits`, `lnw`, `lnb`, `rsqrt_table` | per-token `mean`, `var_sum`, `rsqrt` |
| Softmax | EXP/RECIP tables, 14 scalar values | per-row `max_val`, `acc_val`, `recip_val`, segment select |

LayerNorm과 Softmax는 runtime에 전체 row/tensor tile의 local statistic을 계산한 뒤 LUT를 사용한다. GeLU는 각 element가 독립적이므로 local reduction이 필요 없다.

## Hardware Pass 동작

### GeLU

Pass 수: 1

1. input stream에서 `TP * CP` lane을 읽는다.
2. 각 lane에 대해 cursor를 계산한다.
3. `geluq_table[cursor]`를 읽는다.
4. output stream으로 쓴다.

```cpp
__cursor_t cursor = (vec_i[tp*CP + cp] + b) >> s;
cursor = clamp(cursor, 0, bound);
vec_o[tp*CP + cp] = table[cursor];
```

### LayerNorm

Pass 수: 3

1. Pass 0: input을 읽고 buffer에 저장하면서 `sum(x)`를 누적하고 mean을 계산한다.
2. Pass 1: buffer를 다시 읽고 `sum((x - mean)^2)`를 계산한 뒤 RSQRT LUT를 lookup한다.
3. Pass 2: normalization, affine, shift, signed clamp를 수행하고 output stream으로 쓴다.

```cpp
// pass 0
acc[tp] += vec_i[tp*CP + cp];
mean[tp] = ((acc[tp] * C_1_m) + (1 << (C_1_s - 1))) >> C_1_s;

// pass 1
diff = buffer[tp][c] - mean[tp];
sum[tp] += diff * diff;
cursor = clamp((sum[tp] + b) >> s1, 0, bound);
st_sqrt[tp] = rsqrt_table[cursor];

// pass 2
val = diff * st_sqrt[tp] * lnw[c] + lnb[c];
rel = val >> s2;
out = quantize_clamp(rel, clamp_bits, true);
```

### Softmax

Pass 수: 3

1. Pass 0: row를 읽고 buffer에 저장하면서 row max를 계산한다.
2. Pass 1: `max - x`로 EXP LUT를 lookup하고 `sum(exp)`를 누적한 뒤 reciprocal LUT를 lookup한다.
3. Pass 2: `exp * reciprocal`을 계산하고 segment-specific requant를 수행한 뒤 output stream으로 쓴다.

```cpp
// pass 0
buffer[tp][c] = x;
max_val[tp] = max(max_val[tp], x);

// pass 1
minus = max_val[tp] - buffer[tp][c];
cursor1 = clamp((minus + b1) >> s1, 0, bound1);
exp_score[tp][c] = exp_table[cursor1];
acc_val[tp] += exp_score[tp][c];

// recip segment selection
cursor_one = (acc_val[tp] + b2_one) >> s2_one;
if (cursor_one > bound2_one) {
    cursor_two = clamp((acc_val[tp] + b2_two) >> s2_two, 0, bound2_two);
    recip_val[tp] = recip_table_two[cursor_two];
} else {
    cursor_one = clamp(cursor_one, 0, bound2_one);
    recip_val[tp] = recip_table_one[cursor_one];
}

// pass 2
val = exp_score[tp][c] * recip_val[tp];
rel = (val + selected_b3) >> selected_s3;
out = quantize_clamp(rel, clamp_bits, false);
```

## LUT Entry 개수와 Value Precision

공개 HG-PIPE artifact 기준 table 개수:

| Artifact group | Count |
|---|---:|
| `*_geluq_table_m.txt` | 12 |
| `*_rsqrt_table_m.txt` | 25 |
| `*_exp_opp_table_m.txt` | 12 |
| `*_recip_scaled_table_m_one.txt` | 12 |
| `*_recip_scaled_table_m_two.txt` | 12 |

Transformer block 0-11은 동일한 entry/value precision 구성을 쓴다.

| Layer | GeLU | LayerNorm | Softmax |
|---:|---|---|---|
| 0-11 | 64 entries, `uint3` value | `attn_i_lnq`, `mlp_i_lnq`: 128 entries, `uint12` RSQRT value | EXP: 32 entries, `uint16`; RECIP1/2: each 64 entries, `uint8` |

추가 head:

| Block | LayerNorm |
|---|---|
| `head_lnq` | 128 entries. HLS 선언은 `uint11`, 저장된 refs 값은 최대 2130이라 raw integer 기준 12bit 필요 |

### Value Precision 기준

`Value Precision`은 LUT address bit-width가 아니라 `table[cursor]`로 읽혀 나오는 ROM output value의 bit-width이다.

| Operator | Entries | Address bits | Value precision |
|---|---:|---:|---:|
| GeLU | 64 | 6 | 3-bit unsigned |
| LayerNorm RSQRT | 128 | 7 | 12-bit unsigned |
| Softmax EXP | 32 | 5 | 16-bit unsigned |
| Softmax RECIP | 64 | 6 | 8-bit unsigned |

예를 들어 Softmax EXP는 entry가 32개라 address는 5bit면 충분하지만, table value가 32768 수준까지 나오므로 value precision은 16bit이다.

## SOFTMAX_1X2, SOFTMAX_2X1, SOFTMAX_2X2

이 세 파일은 Softmax 알고리즘 차이가 아니라 parallelization 설정 차이다.

| Variant | TP | CP | 의미 |
|---|---:|---:|---|
| `SOFTMAX_1X2` | 1 | 2 | 한 row를 처리하면서 row 내부 원소 2개를 병렬 처리 |
| `SOFTMAX_2X1` | 2 | 1 | row 2개를 병렬 처리하면서 각 row당 원소 1개 처리 |
| `SOFTMAX_2X2` | 2 | 2 | row 2개를 병렬 처리하고 각 row당 원소 2개 처리 |

`TP`는 token/query-row 병렬도이고, `CP`는 channel/key-column 병렬도이다. `CP`가 커져도 Softmax 통계 범위는 바뀌지 않는다. row max와 sum은 여전히 전체 row에 대해 계산된다.

## Python Reconstruction 코드 흐름

`hgpipe_quantization/ops.py`는 HLS quantization kernel을 Python으로 bit-exact하게 재구성한다.

### Table Quantization 및 GeLU

```python
def table_quantize(inputs, scalars, table):
    b, s, bound = scalars
    return [table[clamp((x + b) >> s, 0, bound)] for x in inputs]
```

### LayerNorm

```python
mean = ((sum(row) * C_1_m) + (1 << (C_1_s - 1))) >> C_1_s
var_sum = sum((x - mean) * (x - mean) for x in row)
cursor = clamp((var_sum + b) >> s1, 0, bound)
rsqrt = rsqrt_table[cursor]
y = ((x - mean) * rsqrt * lnw[c] + lnb[c]) >> s2
y = quantize_clamp(y, clamp_bits, signed=True)
```

### Softmax

```python
max_val = max(row)

for x in row:
    cursor1 = clamp((max_val - x + b1) >> s1, 0, bound1)
    exp_value = exp_table[cursor1]
    acc_val += exp_value

cursor_one = (acc_val + b2_one) >> s2_one
if cursor_one > bound2_one:
    cursor_two = clamp((acc_val + b2_two) >> s2_two, 0, bound2_two)
    recip = recip_table_two[cursor_two]
    b3, s3 = b3_two, s3_two
else:
    cursor_one = clamp(cursor_one, 0, bound2_one)
    recip = recip_table_one[cursor_one]
    b3, s3 = b3_one, s3_one

y = (exp_value * recip + b3) >> s3
y = quantize_clamp(y, clamp_bits, signed=False)
```

## 검증 결과

기존 verification report 기준:

```text
Cases: 97/97 passed
Elements checked: 5,899,008
Total mismatches: 0
```

요약:

| Kind | Cases | Elements | Mismatches |
|---|---:|---:|---:|
| `gelu_requant_table` | 12 | 1,806,336 | 0 |
| `layernorm_rsqrt_table` | 25 | 903,360 | 0 |
| `requant_table` | 48 | 1,806,336 | 0 |
| `softmax_segmented_table` | 12 | 1,382,976 | 0 |

구현 중 실행한 테스트:

```bash
python -m unittest tests.test_lut_calibration tests.test_quantization_scheme
python -m unittest tests.test_quantization_scheme tests.test_cli
python -m py_compile hgpipe_quantization/quantization_scheme.py hgpipe_quantization/lut_calibration.py hgpipe_quantization/cli.py hgpipe_quantization/__init__.py
python -m unittest discover -s tests
```

전체 discovery 결과:

```text
Ran 88 tests
OK
```

## 한계 및 주의점

- 원본 QAT/training-time calibration pipeline은 공개 repo에 포함되어 있지 않다.
- 새 LUT calibration 코드는 공개 HLS contract와 artifact 형식을 따르는 replacement generator이다.
- accuracy drop 분석은 operator bit-exact 검증만으로 충분하지 않고 full validation dataset 평가가 필요하다.
- `head_lnq`는 HLS type 선언과 raw saved table value range 사이에 차이가 있다. 재생성 또는 재합성 시 별도 확인이 필요하다.

## 핵심 결론

1. HG-PIPE의 nonlinear operator는 일반 affine quantizer가 아니라 table/scalar contract로 배포된다.
2. `cursor = (x + b) >> s`는 multiplier-free LUT address 계산이다.
3. LayerNorm과 Softmax는 local runtime statistic을 먼저 계산한 뒤 LUT를 사용한다.
4. GeLU는 local reduction 없이 1-pass fused GeLU-ReQuant lookup으로 처리된다.
5. LUT entry 수는 address precision을 결정하고, value precision은 ROM output bit-width를 결정한다.
6. value precision을 높이면 table value quantization error는 줄일 수 있지만, index binning, clipping, calibration error, downstream quantization error는 별도로 남는다.
