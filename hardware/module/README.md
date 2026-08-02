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
| `include/hbtxr_mlp_core.hpp` | **MLP Core** — 6단계. SMU·score buffer·reorder **없음** |
| `include/hbtxr_patch_embed.hpp` | **Patch Embedding** — line buffer + windower + PE 배열 |
| `include/hbtxr_backbone.hpp` | **Backbone** — 컨트롤러 · global buffer · prefetcher · 코어 순회 |
| `include/hbtxr_head.hpp` | 토큰 풀링 + 2층 회귀 헤드. **RMU 아니라 dense 루프** (197은 안 나뉨) |
| `include/hbtxr_top.hpp` | **top** — stem · 백본 · terminal decode, 모드 하나로 라우팅 |
| `tb/hbtxr_model.hpp` | 모델 스코프 골든 적재 공용 (`tb_backbone`·`tb_top`) |
| `tb/hbtxr_load.hpp` | 페이로드 적재 공용. tb 3개가 같은 블록을 읽습니다 |
| `tb/hbtxr_probe.hpp` | 스테이지 프로브 — 비교 · **배수 단언** · **골든 재충전**. tb 전용 |
| `tb/hbtxr_golden.hpp` | 골든 `.txt` 리더 + 비교 + 음성 대조. **tb 전용** |
| `tb/tb_{requant,gelu,layernorm,rmu,smu,softmax,patch,mha,mlp,block,backbone,top}.cpp` | V1 테스트벤치 12개 |

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

## 모드 의존인 것은 정확히 넷입니다

stem · **최종 norm 진입 requant** · 헤드(+anchor) · **출력 requant**. 가운데 둘이 놓치기 쉽습니다 —
두 경로가 서로 다른 블록에서 나와 **같은 norm 하나**로 들어가고, 두 헤드의 마지막 linear 는
**서로 다른 누산기 격자**를 갖습니다.

출력 requant 를 배열 하나로 뒀더니 **누산기 골든은 계속 통과**하면서(누산기가 그 위에 있으니까)
두 번째로 돈 모드의 고정소수점 값만 조용히 틀렸습니다. `tb_top` 의
**search → track → search 재현성 검사**가 그걸 잡았습니다.

## 순회가 곧 더블 버퍼링입니다

코어쌍이 **2개**고 TRB 8개가 그 위를 돕니다. `i mod 2` 쌍이 TRB i 를 계산하는 동안
`(i+1) mod 2` 쌍은 놀고 있으므로 **가중치를 그때 갈아끼웁니다** — prefetcher 가 자기 버퍼를
따로 들 필요가 없습니다. 참조가 하드웨어 인스턴스를 12개 만들어야 했던 이유(ROM 초기화
가중치, SPEC §4)가 여기서 2개가 됩니다.

Global buffer 가 필요한 것도 **코어를 재사용하기 때문**입니다 — 블록마다 코어가 따로 있으면
잔차 스트림은 dataflow FIFO 에 머물고 저장될 일이 없습니다.

## 런타임 길이와 컴파일 최대치를 헷갈리면 한쪽 토큰 수에서만 보입니다

S×V 의 리덕션이 `kdim`(런타임 = 토큰 수)이 아니라 `K`(컴파일 최대 = 64)까지 돌고 있었습니다.
**search 는 `kdim == K` 라 완벽히 가려지고**, 64토큰 실행 뒤의 16토큰 실행만 stale 48열을
읽습니다. `tb_backbone` 이 **search → track → search** 를 도는 이유가 이것입니다.

## 패치 임베딩의 PE 배열은 **RMU 입니다**

`kernel == stride == patch` 라 윈도우가 겹치지 않고, **im2col 이 순수 주소 계산**입니다.
그래서 conv 가 `[tokens, Cin·K·K] × [D, Cin·K·K]ᵀ` matmul 이고, SPEC §5-3 의
"Do 개 PE, PE당 K·K·Ui 곱셈기" 가 그 matmul 의 PE 배열입니다. **스템 고유는 windower 뿐**입니다.

공유하지 않는 것 둘:

- **누산기.** 8비트 피연산자 × `Cin·K·K` 탭 = **25비트** 로 코어의 20비트에 안 들어갑니다.
  골든 실측은 20비트에 **들어가고**, 그게 함정입니다 — 측정값으로 타입을 잡으면 오늘 모든
  테스트를 통과하고 실데이터에서 wrap 합니다. `HbtxrRmu` 의 `static_assert` 가 실제로
  좁히기를 **컴파일 단계에서 거부**하는 것을 확인했습니다
- **입력 격자가 비대칭.** 진짜 0 이 zero-point 라 창마다 `- zp·Σw` 가 붙는데, 출력채널별
  상수라 **RMU 가 이미 더하는 bias 에 접힙니다** — 상주 숫자 하나, 런타임 뺄셈 없음

## 두 코어 사이에는 이음새가 없습니다

`tb_block` 이 처음으로 그걸 시험하고, 답은 **requant 가 없다**입니다 — MLP 코어의 입력 포트가
곧 attention residual 의 출력 격자라 `mlp_x == mha_y` 입니다. 여기에 브리지를 넣으면
최적화가 아니라 **버그**이고, 골든이 그렇게 말합니다.

## 프로브가 첫 실패 지점을 특정합니다

각 스테이지 경계에서 **비교 → 스트림이 비었는지 단언 → 골든으로 재충전**을 합니다.
재충전이 핵심입니다: 앞 단계 오류가 뒤로 전파되면 프로브가 전부 빨개져서 **어디가 원인인지
알 수 없습니다.**

`tb_mha` 가 이걸 고장 주입으로 확인합니다 — `e02` 승수를 2배로 만들면 **7개 중 1개**
(`qkv_x`)만 발화하고 위아래는 통과합니다.

## 완료 조건

**"컴파일된다"가 아니라 "골든과 원소별로 같다"** 입니다. 모든 tb 가 셋을 합니다 —
원소별 `==` · **스트림 배수 확인**(과생산은 값 비교로 못 잡습니다) · **음성 대조**.
