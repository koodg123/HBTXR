> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active
> **소유** hardware

# module — HLS 구현

```bash
sh hardware/build/run_tb.sh              # 전부
sh hardware/build/run_tb.sh rmu          # 하나
```

## 있는 것

| | |
|---|---|
| `include/hbtxr_requant.hpp` | dyadic requant. **`i_ops.py:requant` 와 비트 단위로 같습니다** |
| `include/hbtxr_rmu.hpp` | **RMU** — 가중치 상주(런타임 적재), 활성만 스트림 |
| `include/hbtxr_smu.hpp` | **SMU** — **두 피연산자 다 스트림.** B 를 유닛이 전치 |
| `include/hbtxr_lut.hpp` | PoT 커서 + GeLU. **커서는 signed·클램프 전 범위** |
| `include/hbtxr_layernorm.hpp` | 정수 LayerNorm (7 scalar, rsqrt 1세그먼트) |
| `include/hbtxr_softmax.hpp` | 정수 Softmax (14 scalar, reciprocal 2세그먼트) |
| `include/hbtxr_mha_core.hpp` | **MHA Core** — 9단계 체인 |
| `tb/hbtxr_probe.hpp` | 스테이지 프로브 — 비교 · **배수 단언** · **골든 재충전**. tb 전용 |
| `tb/hbtxr_golden.hpp` | 골든 `.txt` 리더 + 비교 + 음성 대조. **tb 전용** |
| `tb/tb_{requant,gelu,layernorm,rmu,smu,softmax,mha}.cpp` | V1 테스트벤치 7개 |

`src/` 는 S4(코어)부터입니다. `golden/` 은 쓰지 않습니다 — 골든은 생성물이라
`hardware/workspace/golden/` 에 있습니다.

## RMU 와 SMU 는 왜 별개인가

**데이터패스는 같고 두 번째 피연산자의 수명이 다릅니다.** RMU 의 가중치는 블록 간에 상주하고,
SMU 의 `K` 는 토큰 세트마다 바뀌므로 상주할 수 없습니다. `Q×Kᵀ` 는 **`K` 가 전부 도착하기
전에 점수 행렬의 0행도 못 냅니다** — SMU 안의 버퍼가 그 제약을 코드로 만든 것이고,
MHA 코어가 텐서 전체 residual FIFO 를 요구하는 이유도 같습니다 ([SPEC §5](../docs/SPEC.md)).

## 세 숫자가 일치해야 합니다

```cpp
hls::vector<act_t, TP*CIP>                                    // 스트림 폭
#pragma HLS unroll                                            // 언롤
#pragma HLS array_reshape variable=w cyclic factor=CIP dim=2  // reshape
```

**`array_partition` 이 아니라 `array_reshape`** 입니다. 그리고 **리덕션은 최내곽** —
부분합이 `TP*COP` 플립플롭이 되어 II=1 이 공짜로 나옵니다.

## 비트 레이아웃 — 생산자·소비자가 합의해야 합니다

```
RMU in   v[p*CIP + c] = 토큰 t0+p, 입력채널   i0+c
RMU out  v[p*COP + c] = 토큰 t0+p, 출력채널   o0+c
SMU a,b  v[p*CIP + c] = 행   t0+p, 리덕션채널 k0+c
SMU out  v[p*COP + c] = 행   t0+p, 열         j0+c
```

## `ap_int::operator*` 는 피연산자 폭의 **합**을 냅니다

넓은 타입으로 먼저 캐스팅하고 곱하면 **그 넓은 타입들의** 합만큼 곱셈기를 요구합니다.
5곳에서 걸렸습니다 — requant(54×54→108), LN 의 mean·variance·affine, softmax 의 `e*recip`.
**항상 좁은 피연산자끼리 곱하고 결과 타입을 유도**합니다.

## 프로브가 첫 실패 지점을 특정합니다

각 스테이지 경계에서 **비교 → 스트림이 비었는지 단언 → 골든으로 재충전**을 합니다.
재충전이 핵심입니다: 앞 단계 오류가 뒤로 전파되면 프로브가 전부 빨개져서 **어디가 원인인지
알 수 없습니다.**

`tb_mha` 가 이걸 고장 주입으로 확인합니다 — `e02` 승수를 2배로 만들면 **7개 중 1개**
(`qkv_x`)만 발화하고 위아래는 통과합니다.

## 완료 조건

**"컴파일된다"가 아니라 "골든과 원소별로 같다"** 입니다. 모든 tb 가 셋을 합니다 —
원소별 `==` · **스트림 배수 확인**(과생산은 값 비교로 못 잡습니다) · **음성 대조**.
