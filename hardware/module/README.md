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
| `tb/hbtxr_golden.hpp` | 골든 `.txt` 리더 + 비교 + 음성 대조. **tb 전용** |
| `tb/tb_{requant,rmu,smu}.cpp` | V1 테스트벤치 |

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

## 완료 조건

**"컴파일된다"가 아니라 "골든과 원소별로 같다"** 입니다. 모든 tb 가 셋을 합니다 —
원소별 `==` · **스트림 배수 확인**(과생산은 값 비교로 못 잡습니다) · **음성 대조**.
