> **작성** 2026-07-31 · **갱신** 2026-08-03
> **상태** active — **적용 완료** (2026-08-03). §6 이 그 결과입니다
> **소유** algorithm/quantization

# requant 승수가 33비트까지 옵니다 — 그리고 조이는 지점이 shift 가 아닙니다

`dyadic_params` 가 고르는 `(M, n)` 이 하드웨어 requant 유닛의 폭을 그대로 정합니다.
HLS 재작성(S1)이 골든을 뽑으면서 실측했더니 **`M` 이 최대 33비트**였고, `ap_int<20>` 누산기와
곱하면 **53비트 곱**입니다. DSP48E2 한 개로 안 됩니다.

측정 대상: `hardware-new` 브랜치 `hardware/tools/export_hls_golden.py` 가 낸 골든의
requant 쌍 **1,932개** (search 프리셋 1벌 기준).

---

## 1. 실측

| 프리셋 | `M` 최대 | 폭 | `n` 범위 | `acc·M` |
|---|---:|---:|---|---:|
| search-a4 | 2,468,372,009 | **32b** | 2 – 31 | **52b** |
| search-a8 | 4,461,753,799 | **33b** | 20 – 31 | **53b** |

원인은 탐색 방식입니다. [`i_ops.py:26`](../../quantization/i_ops.py) 이 `shift ∈ [1, 31]` 을
훑으며 **오차가 가장 작은 것**을 고르는데, 오차는 shift 가 커질수록 단조 감소합니다.
그래서 **언제나 `shift_max` 가 이깁니다.** 그러면 `M = round(ratio·2³¹)` 이고,
비율이 1을 넘는 엣지는 `M` 이 2³¹ 을 넘습니다.

예: `e02_ln1_to_qkv` 는 비율 1.149 → `M = 2,468,372,009`, `n = 31`.

## 2. 구현이 셋이고, cap 이 서로 다릅니다

| 위치 | `shift_max` | 비고 |
|---|---:|---|
| [`i_ops.dyadic_params`](../../quantization/i_ops.py) | **31** | 정수 골든. `i_block.rescale` 이 인자 없이 호출 |
| [`ilayers/int_functional._dyadic_params`](../../quantization/ilayers/int_functional.py) | **31** | **의도적 사본** — `ilayers` 가 `i_ops` 를 import 하지 않는다는 규칙 때문 |
| [`spec._dyadic_scale`](../../quantization/spec.py) | **24** | torch fake-quant 경로 (`scale_type="dyadic"`) |

**fake-quant 경로는 이미 24로 잘라 쓰고 있고 정수 경로만 31입니다.** 의도된 차이라는 근거를
어느 docstring 에서도 못 찾았습니다.

추가로 [`int_calibrate.py:491`](../../quantization/int_calibrate.py) 이 LayerNorm 의
`c_1_m / 2^c_1_s ≈ 1/C` 에도 같은 함수를 씁니다. cap 을 바꾸면 requant 뿐 아니라
**LayerNorm scalars 도 같이 움직입니다.**

---

## 3. `shift_max` 를 조이는 것은 **틀린 손잡이**입니다

처음엔 `shift_max=17` 이면 `M` 이 18비트에 들어가고 상대오차는 무시할 만하다고 봤습니다.
**한 엣지만 보면 맞고, 전체를 보면 틀립니다.**

| `shift_max` | `M` 최대 | `acc·M` | **최대** 상대오차 | 골든이 바뀌나 |
|---:|---:|---:|---:|---|
| **31** (현재) | 32b | 52b | — | — |
| 24 | 25b | 45b | 1.12e-04 | **바뀜** (9개 중 7개 벡터) |
| 20 | 21b | 41b | 1.90e-03 | **바뀜** (9/9) |
| 17 | 18b | 38b | **1.58e-02** | **바뀜** (9/9) |
| 14 | 14b | 34b | **1.11e-01** | **바뀜** (9/9) |

**비율이 작은 엣지가 큰 shift 를 필요로 하기 때문입니다.** `M = max(1, round(ratio·2ⁿ))` 이라
`ratio ≈ 1e-6` 인 엣지는 `n=17` 에서 `M` 이 1 로 바닥을 치고 유효 비율이 7.6e-6 이 됩니다 —
7배 오차입니다. 누산기 격자에서 출력 격자로 내려가는 엣지가 전부 이런 모양입니다.

`shift_max=14` 의 1.11e-01 은 **4비트 출력의 1 LSB(1/7 = 1.4e-01)와 같은 자릿수**입니다.

---

## 4. 옳은 손잡이는 **`M` 의 폭**입니다

shift 를 자르는 대신 **`M` 이 `W` 비트에 들어가는 shift 중 오차가 가장 작은 것**을 고릅니다.
비율이 작으면 큰 shift 가 허용되고(`M` 이 작으니까), 1 근처면 shift 가 자동으로 잘립니다.

```python
def dyadic_params(scale, *, shift_min=1, shift_max=31, wbits=None):
    best = None
    for shift in range(shift_min, shift_max + 1):
        m = max(1, round(scale * (1 << shift)))
        if wbits is not None and m.bit_length() > wbits:
            continue                      # 이 shift 는 승수가 안 들어간다
        err = abs(m / (1 << shift) - scale)
        if best is None or err < best[0]:
            best = (err, m, shift, m / (1 << shift))
    ...
```

| `M ≤ W` | `acc·M` | `n` 범위 | 최대 상대오차 |
|---:|---:|---|---:|
| 24b | 44b | 2 – 31 | 5.95e-08 |
| 20b | 40b | 2 – 31 | 9.18e-07 |
| **18b** | **38b** | 2 – 30 | 3.81e-06 |
| 16b | 36b | 2 – 28 | 1.50e-05 |

`shift` 범위가 여전히 2~28 인 것이 요점입니다 — **작은 비율은 큰 shift 를 그대로 씁니다.**
잘리는 것은 shift 가 아니라 승수뿐입니다.

#### 「무손실」의 정확한 범위 — 최초 주장은 너무 넓었습니다

처음 측정은 **`search-a4` 한 프리셋의 값 벡터 9개**였고, 리포트는 그걸 "골든 비트 동일"로
적었습니다. 적용 시점에 **프리셋 9벌 775개 파일 전수**로 다시 재니 그렇지 않습니다.

| 설정 | 값 벡터 |
|---|---|
| `search-a4` · `track-a4` · `tiny-a4` · `tiny-a8` · **실제 모델 `model-hbtxr-w4s4h8`** | **비트 동일** |
| `search-a8` · `track-a8` (순수 8비트 블록) | **바뀜** — `block_y` 포함 |

**4비트 출력 격자가 `3.8e-06` 의 비율 오차를 흡수하고 8비트는 못 합니다.** 격자가 16배
촘촘하니 어딘가에서 양자화 경계를 넘고, 넘는 순간 생성기의 캘리브레이션 walk 가 관측값을
다시 읽어 페이로드까지 연쇄로 달라집니다.

**8비트에서 비트 동일을 쫓는 것은 의미가 없습니다.** 폭 스윕이 **단조가 아닙니다**:

| `M ≤ W` | search-a8 |
|---:|---|
| 18b · 20b · 22b | 바뀜 |
| **24b** | **동일** |
| **26b** | **바뀜** |
| 제한 없음 | 동일 |

반올림이 어디 떨어지느냐의 우연이지 폭이 늘면 좋아지는 성질이 아닙니다. `-a8` 블록 프리셋은
혼합정밀을 오라클이 표현 못 해서 두는 **우회용이지 배포 설정이 아닙니다.**

### 하드웨어에서 의미하는 것

DSP48E2 는 `27×18` 입니다. **`M ≤ 18b` 면 B 포트에 그대로 들어가고**, 곱은 38비트라
누산기 하나에 담깁니다. 현재의 32비트 승수는 곱이 52비트라 DSP 를 쪼개 붙여야 합니다.

---

## 5. 바꾼다면 건드리는 곳

| | |
|---|---|
| [`i_ops.dyadic_params`](../../quantization/i_ops.py) | `wbits` 인자 추가 + 기본값 결정 |
| [`ilayers/int_functional._dyadic_params`](../../quantization/ilayers/int_functional.py) | **의도적 사본이라 같이** 고쳐야 합니다. 안 고치면 골든과 배포 커널이 갈립니다 |
| [`i_block.rescale`](../../quantization/i_block.py) | 호출부. 기본값을 쓰면 수정 불필요 |
| [`int_calibrate.py:491`](../../quantization/int_calibrate.py) | LayerNorm `c_1_m` — 폭 제한을 **적용할지 말지 따로 판단**해야 합니다 |
| [`spec._dyadic_scale`](../../quantization/spec.py) | fake-quant 경로. cap 24 와의 관계를 정리 |
| 테스트 | pytest **594개**. `test_int_graph.py` · `test_int_layernorm.py` 가 `shift_max` 를 직접 참조 |

### 결정해야 할 것

1. `wbits` 기본값 — **18 을 권합니다** (DSP48E2 B 포트, 16까지 무손실이므로 2비트 여유)
2. `int_calibrate` 의 LayerNorm `c_1_m` 에도 적용할지 — requant 와 성격이 다릅니다
3. `spec._dyadic_scale` 의 24 를 맞출지, 다른 이유가 있는지

**결정 전까지 하드웨어는 `REQ_M_BITS` 를 traits 파라미터로 두고 진행합니다.**
값이 바뀌어도 설계가 아니라 상수 하나가 바뀝니다.

---

## 재현

```bash
# hardware-new 워크트리
sh hardware/build/make_golden.sh search-4
# 그 뒤 emit() 을 i_block.dyadic_params 패치와 함께 재실행하고 .txt 를 diff
```

측정 스크립트는 이 리포트에 인라인된 `by_width` 가 전부입니다 — 별도 파일을 두지 않았습니다.

---

## 6. 적용 — 2026-08-03

`DYADIC_MULT_BITS = 18` 을 `i_ops` 에 두고 `ilayers/int_functional` 의 **의도적 사본**도 같이
바꿨습니다. `test_dyadic_copy_matches_golden` 이 두 값이 갈리지 않도록 고정합니다.

| | |
|---|---|
| **pytest** | 594 → **604 passed** (신규 테스트 4개·10 케이스). 기존 594 중 실패 0 |
| **`acc·M`** | 53b → **38b**. `M` 최대 4,461,753,799 → **262,111** |
| **`n` 범위** | 2 – 31 그대로 |
| **하드웨어** | `REQ_M_BITS` 33 → 18. tb **21개 실행 전부 PASS** |

`spec._dyadic_scale`(fake-quant, cap 24)은 **건드리지 않았습니다** — requant 승수가 아니라
스케일 표현이고 학습 경로입니다. 무손실이 측정된 둘과 안 측정된 하나를 같이 바꾸면 회귀가
나도 원인을 못 짚습니다. 별건입니다.

`int_calibrate` 의 LayerNorm `c_1_m` 은 같은 함수를 쓰므로 **자동으로 따라갔고**,
`acc·c_1_m` 이 36 → 30비트가 됩니다. 평균의 상대오차는 1.6e-06 으로 1 LSB 한참 아래입니다.

### 폭 제한이 드러낸 것 둘

둘 다 **제한이 없을 때는 거대한 승수를 조용히 받아내서** 아무도 못 보던 것입니다.

1. **`_stem_peak` 의 프로브가 `1e-12` 격자로 requant 를 통과**시키고 있었습니다. 비율 5.5e6,
   승수 24비트 — 측정 장치일 뿐 데이터패스가 아닌데 `dyadic_params` 를 지나고 있었습니다.
   누산기 피크를 직접 계산하도록 바꿨습니다 (그쪽이 더 짧습니다).
2. **`_grid` 가 전부 0 인 누산기에 `1e-12` 바닥값**을 내고 있었습니다. `model-tiny` 에서
   비율 **1.73e11**, 승수 **39비트** — 의미 없는 하드웨어인데 그 프리셋을 보는 tb 가 없어
   드러나지 않았습니다. 관측할 값이 없으면 생산자의 격자를 그대로 쓰도록(항등) 고쳤습니다.
