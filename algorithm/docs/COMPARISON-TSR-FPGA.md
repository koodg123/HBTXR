> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** reference — 조사 완료, 실행 항목은 §7
> **소유** repo

# TSR_FPGA (SICDL) — quantization/integer analysis, and what it says about HBTXR

**조사 대상**: `C:\Users\User\Downloads\SICDL\TSR_FPGA` @ `.git` 기준 스냅샷 (외부 저장소)

An external, **completed-through-silicon** INT8 CNN accelerator project, read end to end and
compared against this repo's quantization stack. It is the closest thing available to a
finished version of where HBTXR is heading, so the comparison is worth having on record —
including the places where its conclusions contradict ours.

Everything below was read from source. Where a claim is a measurement it says so; where it
is an inference from code structure it says that too.

---

## 1. What TSR_FPGA is

GTSRB traffic-sign recognition on a **ZC706** FPGA. Models are ResNet11 (A/B/C/D variants)
and a small ZynqNet. ~6200 files, of which the quantization/integer code is ~2900 lines.

```
FP32 학습 → BN folding → QAT → 정수 추론 → .bin 추출
          → HLS → Vivado → Petalinux → Vitis 드라이버 → UART 보드 테스트
```

| stage | 내용 |
|---|---|
| stage1 | FP32 학습/추론, BN folding 스크립트 |
| stage2 | **QAT** — LSQ / TQT / LLTQ 양자화기, YAML 설정 |
| stage3 | **정수 추론** — `QConv2d` / `QLinear` / `QConv2d_Branch` |
| stage4 | `.bin` 추출 (이미지, fp32/int8 weight) |
| stage5 | HLS + Vivado + Petalinux + Vitis, 4개 PE 구성 × fp32/int8 |

**결정적 사실: 비선형이 ReLU뿐입니다.** LayerNorm도 Softmax도 GELU도 없습니다. ReLU는
정수에서 `clamp(x, 0, ·)`로 정확히 표현되므로 **LUT이 한 개도 필요 없습니다.** 이것이 두
프로젝트의 난이도 지형을 완전히 갈라놓습니다 — 자세히는 §6.

---

## 2. QAT 계층 (stage2)

### 양자화기

`quan/quantizer/quant_module.py`에 7종: `LSQuan` · `TQTQuan` · `LLTQuan` · `LLSQuan` ·
`SFQuan` · `SPOTQuan` · `FXQuan` · `LinearQuan`. YAML 기본값은 `lltq`.

STE 관용구는 이 repo와 **동일**합니다:

```python
def round_pass(x): y = x.round(); return (y - x).detach() + x
def ceil_pass(x):  y = x.ceil();  return (y - x).detach() + x
```

핵심은 **TQT/LLTQ가 스케일을 학습**한다는 점입니다:

```python
ceil_log2_t = ceil_pass(self.log2_t)      # log2 임계값이 학습 파라미터
threshold   = pow(2.0, ceil_log2_t)        # → 2의 거듭제곱
pot_scale   = threshold / self.q_levels_per_sign
x = t.clamp(round_pass(x / pot_scale), self.thd_neg, self.thd_pos) * pot_scale
```

`log2_t`가 gradient로 학습되고 `ceil`이 STE로 통과되므로, 스케일이 **학습된 2의 거듭제곱
(PoT)** 으로 수렴합니다. 이 repo의 `apply_scale_type(..., "pot")` — 캘리브된 스케일을
사후에 `2^k`로 반올림하는 것 — 과는 **다른 연산**입니다. §6①이 이 차이를 다룹니다.

### BN folding

`quan/func.py` 372줄 중 ~200줄이 `QuanConv2dBN` 한 클래스입니다. YAML로 제어:

| 옵션 | 의미 |
|---|---|
| `bn.fold` | `param` \| `layer` — folding 방식 |
| `bn.correction` | batch std vs running std 보정항 사용 |
| `bn.freeze` | 에폭 리스트. `[200]` 단일 또는 `[40,80,120,160,200]` 반복 freeze/unfreeze |
| `bn.adaptive` | adaptive BN |

`correction=true`일 때 forward가 batch 통계로 conv를 돌린 뒤
`out * (mv_std/batch_std) + bn_weight*(running_mean/mv_std - batch_mean/batch_std)`로
running 통계에 맞춰 되돌립니다 — QAT 중 BN 통계 이동이 양자화 스케일을 흔드는 문제에
대한 표준 대응입니다.

### 첫/마지막 레이어 고정

`excepts:`로 `conv1`과 `classifier`를 8비트로 못박고, `hold: first|last`면 활성 스케일을
**`0.00390625 = 2⁻⁸`로 강제**합니다. 입력이 uint8 이미지라 `1/256` 격자가 공짜로 정확하기
때문입니다.

---

## 3. 정수 계층 (stage3) — 스케일 대수

모든 스케일이 PoT이므로 **requant가 곱셈 없이 시프트 하나**입니다.
`log2` 지수만 정수로 들고 다니고 (`int_scale = -log2(scale)`), 엣지마다:

```
s1 = sa + sw - sb        # bias를 누산기 격자로 올리는 시프트
s2 = sa + sw - sa_out    # 누산기를 출력 격자로 내리는 시프트
```

```python
bias = (self.bias << self.s1)        # 누산기 격자
out  = out + bias                    # 누산기에서 더함
out  = out >> self.s2                # 산술 시프트 = floor
out  = t.clamp(out, thd_neg, thd_pos)
```

HLS가 이것을 그대로 구현합니다 (`processing_element.cpp`) — `scaling_factor = ap_int<6>`,
`data_t = char`(int8), `idata_t = int`(int32 누산):

```cpp
idata_t up_scaled_bias     = (((idata_t)tmp_bias) << (s1));
biased                     = raw + up_scaled_bias;
rectified                  = (biased < 0) ? 0 : biased;      // ReLU
idata_t down_scaled_result = tmp_result >> (s2);
```

conv는 `F.conv2d`에 **int32 텐서**를 그대로 넣습니다 (확인: torch가 int32 conv2d를 CPU에서
지원). bias를 누산기에서 더하는 것은 이 repo의 `_IntLinear.bias_int`와 같은 원칙입니다.

### "fully integer"에 float이 두 군데 남아 있습니다

**잔차 덧셈** (`stage3/model/resA_int.py: adderv3`):

```python
data_1 = ((data_1 / scale_1a_out) * scale_bp).type(t.int32)   # 호출당 float 나눗셈·곱셈
data_2 = ((data_2 / scale_2a_out) * scale_bp).type(t.int32)
out    = data_1 + data_2
out    = out / scale_bp * scale_2a_out
qout   = out.type(t.int32)     # 반올림 아님 — 0 방향 절단
# qout = out.round()           # ← 주석 처리되어 남아 있음
```

**Global average pool** (`resA_int.py: forward`):

```python
out = out.type(t.float32); out = F.avg_pool2d(out, out.size()[3]); out = out.type(t.int32)
```

이 둘은 이 repo의 A1이 제거한 것과 정확히 같은 부류이고, `IPool`이 int64 정확 평균으로
만든 바로 그 지점입니다. 덧셈의 `.type(t.int32)` 절단은 주석 처리된 `.round()` 때문에
**의도적 선택인지 미완인지 코드만으로는 판정 불가**합니다. HLS의 PE는 시프트만 하므로
파이썬 정수 모델과 HLS가 이 지점에서 갈라질 수 있고, **그것을 확인하는 장치가 없습니다.**

---

## 4. 검증

| 층위 | 방법 | 성격 |
|---|---|---|
| QAT vs 정수 | Top-1 정확도 비교. 체크포인트 파일명에 인코딩: `..._qat_98.994_int_99.07.pth.tar` | 통계량 |
| 레이어별 | `debug/stage2_hook`, `debug/stage3_hook` 활성화 훅 | 육안 대조 |
| HW vs SW | 보드에서 UART로 Top-5 출력, CPU C++ 결과와 비교 | end-to-end |
| HLS 단위테스트 | **fp32 설계에만** (`unittests.cpp`: MemoryController / WeightsCache / ImageCache) | 인프라, 수치 아님 |

**자동 테스트 스위트 없음.** `stage1/model/test_model.py`는 테스트가 아니라 모델 정의입니다.
**순수 파이썬 골든 없음.** stage3이 HLS의 기준이지만, **stage3 자체의 기준은 없습니다.**

그 대가가 로그에 남아 있습니다 (`stage5/report/run_zc706_log/`):

```
szynq2s5_int8_pe1_cpu_fail.txt   szynq2s5_int8_pe1_fpga_fail.txt
szynq2s5_int8_pe2_cpu_fail.txt   szynq2s5_int8_pe2_fpga_fail.txt
szynq2s5_int8_pe4_cpu_fail.txt   szynq2s5_int8_pe4_fpga_fail.txt  → v2, v3 재시도 후 성공
szynq2s5_int8_pe8_cpu_fail.txt   szynq2s5_int8_pe8_fpga_fail.txt
```

```
TestBench Result: FAILURE              TestBench Result: SUCCESS
Actual: 93.33, Expected: 100.00        100.00%: class 2
    93.33%: class 2 (output 3527.0)        (output -287018.8)
     4.65%: class 5 (output 3524.0)
```

**int8 결함이 11단계 플로우의 맨 끝, FPGA 보드 위에서 발견됐습니다.** fp32 구성은 모두 한
번에 통과했고, `_fail` 접미사가 붙은 것은 int8 8개뿐입니다. 정확도 지표로는 "93.33% vs
100%"라는 약한 신호로만 보입니다 — 어느 레이어에서 무엇이 틀렸는지는 말해주지 않습니다.

---

## 5. 비교표

| | TSR_FPGA | HBTXR |
|---|---|---|
| 모델 | CNN (ResNet11 / ZynqNet) | ViT (transformer) |
| 비선형 | **ReLU만** — LUT 0개 | LayerNorm · Softmax · GELU — 정수 LUT 3종 |
| 주 경로 | **QAT 우선** | **PTQ 우선** (QAT 지원) |
| 스케일 결정 | **학습된 PoT** (`ceil(log2_t)`, gradient) | 관측 캘리브 (minmax/percentile/MSE/KL) + float/dyadic/pot |
| requant | **시프트만** `>> s2` | **dyadic** `(acc·m + rnd) >> s` |
| 반올림 | floor (기본), `round` 옵션 | round-half-up (상수 `1<<(s-1)`) |
| 입도 | per-tensor 위주 | per-tensor / channel / group / block |
| bias | 누산기에서 `<< s1` | 누산기에서 정수 bias |
| BN folding | **정교함** (correction / freeze 스케줄 / adaptive) | 해당 없음 (ViT는 LayerNorm) |
| 잔차 덧셈 | **float 비율 + 절단** | dyadic 정렬 + 클램프 (그래프 티어는 빌드타임 상수) |
| 풀링 | **float32 `avg_pool2d`** | int64 정확 평균, round-half-away-from-zero |
| 골든 | **없음** | `i_ops.py` 임의정밀도 순수 파이썬, 원소별 `==` |
| 조합 오라클 | **없음** | `i_block.replay_block_int` / `replay_model_int` |
| 테스트 | **없음** | 594개 + 뮤테이션 게이트 필수 |
| 하드웨어 | **비트스트림 · 리눅스 · 드라이버 · 보드 실측 완료** | **없음** (export 아티팩트까지) |
| 학습 체크포인트 | 있음 (GTSRB 98~99%) | **없음** (A2 — 모든 수치가 랜덤 초기화) |

---

## 6. 실질적으로 중요한 차이

### ① PoT 결론이 서로 다른 이유 — 같은 실험이 아닙니다

이 repo의 Part B 리포트:

> **`pot` is ~6× worse** (8.7% vs 1.5%) — rounding the whole scale to 2^k is coarse;
> use dyadic, not pot, for HW.

TSR_FPGA: PoT 스케일로 **QAT 98.994% → int8 99.07%**, 손실 없음 (오히려 +0.08pp).

**모순이 아닙니다.** 이 repo는 캘리브된 스케일을 **사후에** `2^k`로 반올림했고(PTQ 경로),
TSR_FPGA는 `log2_t`를 **gradient로 학습**했습니다(QAT 경로). 사후 반올림은 PoT 격자가
강제하는 오차를 그대로 받지만, 학습은 네트워크가 그 격자에 **적응**합니다.

이것이 이번 조사에서 가장 실행 가능한 발견입니다. **"pot 6배 악화"는 PTQ 경로에 대해서만
참이고, 학습된-PoT QAT에는 전이되지 않을 수 있습니다.** 성립한다면 requant에서 곱셈기가
통째로 사라집니다 — dyadic의 `m` 없이 시프트만 남고, 이것은 DSP 슬라이스에 직접 반영되는
차이입니다. §7①.

### ② TSR_FPGA의 "fully integer"에도 float이 남아 있습니다

§3 참조. 이 repo가 A1에서 제거한 것과 같은 부류이며, 잔차 덧셈의 절단/반올림 미결정은
파이썬 모델과 HLS가 갈라질 수 있는 지점입니다. **이 repo의 A1이 과잉이 아니었다는 외부
증거**로 읽는 것이 맞습니다.

### ③ 검증 문화가 정반대이고, 대가도 정반대입니다

TSR_FPGA는 정확도(top-1)와 보드 실행으로 정합성을 세웁니다. 빠르고, 실물까지 갑니다.
대신 int8 결함이 파이프라인 맨 끝에서 드러났습니다 (§4).

이 repo는 커널마다 임의정밀도 골든과 원소별 `==`로 잡습니다. 최근 한 세션만 해도 float32
누산 부정확(2^24 초과), qkv를 Q 그리드로 눌러 3.2% 오차, metrics가 export까지 도달하지
않음 — 전부 커널/조합 수준에서 잡혔습니다. **대신 하드웨어에서 한 번도 돌아본 적이
없습니다.** TSR_FPGA가 겪은 종류의 결함(메모리 레이아웃, DRAM 정렬, PE 스케줄링)은 이
repo가 아직 마주치지도 않았습니다.

두 프로젝트는 서로의 사각지대를 정확히 채웁니다.

### ④ TSR_FPGA는 이 repo의 어려운 부분에 대해 선례가 아닙니다

ReLU-only이므로 정수 LayerNorm의 분할 rsqrt LUT, 정수 Softmax의 분할 reciprocal, PoT
인덱스 GELU 테이블 — **이 repo가 D3에서 10~12.6배 개선을 측정한 그 영역 전체에 대해
이 레포에는 참고할 것이 없습니다.** transformer 가속기 선례는 다른 곳에서 찾아야 합니다.

---

## 7. HBTXR가 가져올 것 / 가져오지 말 것

### 가져올 것

1. **학습된 PoT 스케일 측정** — A2(학습 체크포인트) 이후. `log2_t`를 학습 파라미터로 두고
   `ceil_pass`로 STE 통과시키는 방식. 성립하면 requant 곱셈기가 사라집니다. TSR_FPGA가
   존재 증명입니다. **A2 의존.**
2. **입력 그리드 `2⁻⁸` 고정** — 이미지가 uint8이면 `1/256` 격자가 공짜로 정확합니다. 현재
   patch-embed conv의 활성 스케일을 관측으로 캘리브하는데, 입력만큼은 격자가 이미 정해져
   있습니다. `QuantScheme.overrides`로 표현 가능하고 기본값이 될 만합니다. **A2 무관, 즉시 가능.**
3. **첫/마지막 레이어 8비트 고정을 기본 정책으로** — 기능(`overrides`)은 있으나 정책이
   없습니다. TSR_FPGA는 `excepts`로 항상 이렇게 합니다.
4. **stage4 → stage5 배치를 참고** — `.bin` 추출 → HLS → 블록디자인 → Petalinux → 드라이버
   → UART 대조. 이 repo가 아직 밟지 않은 경로의 작동하는 사례입니다.

### 가져오지 말 것

- **정수 티어의 float 누출** (잔차 덧셈, avg_pool) — 이미 A1에서 제거함
- **floor requant** — 이 repo는 round-half-up. floor는 계통 편향(bias)을 만듭니다
- **검증을 정확도 지표로만 세우는 것** — §6③
- **transformer 비선형 설계** — 여기에 없음 (§6④)

---

## 8. TSR_FPGA에 부족한 것 (역방향 관찰)

기록해 두는 이유: 이 repo가 하드웨어로 갈 때 같은 함정을 피하기 위해서입니다.

- 순수 파이썬 골든과 원소별 검증 부재 — "약간 틀림"과 "구조적으로 틀림"을 구분 못 함
- 자동 테스트 부재
- 정수 티어 float 누출 2건
- 잔차 덧셈의 절단 vs 반올림이 결정인지 미완인지 불명확
- int8 HLS 설계에 단위테스트 없음 (fp32에는 있음)
- 파이썬 정수 모델 ↔ HLS 사이의 자동 대조 없음
