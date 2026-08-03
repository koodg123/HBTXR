> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active
> **소유** hardware

# config — 파라미터 계약

`design/hbtxr_config.hpp` **하나**입니다. **매크로 0개, traits 클래스 하나.**

```cpp
template <class CFG> struct HbtxrRmu { ... };      // config 는 타입으로 전달
struct HbtxrCfgTrack : HbtxrCfgBase { static constexpr int N = 16; };
```

## 왜 이 형태인가

| | |
|---|---|
| **매크로 0개** | 참조는 12 레이어를 각각 펼쳐서 한 레이어 배치 실패용 탈출구가 필요했고, 그 탈출구 하나에 `attn.h` 1,400줄이 딸려 있습니다. HBTXR 은 **코어 공유가 아키텍처**라 지울 멤버가 없습니다 |
| **타입으로 전달** | **C++20 class-type non-type 템플릿 파라미터를 HLS 프론트엔드가 받지 않습니다.** `constexpr` 객체 값으로 넘기면 안 됩니다 |
| **이름 있는 인자만** | 참조는 파라미터 126개 중 76개가 `int` 라 슬롯이 바뀌어도 **컴파일·csim 을 통과**하고 cosim 에서만 드러납니다 |
| **폭은 유도** | `mac_width(w,a,ci)` · `clog2()` 로 계산하고 `static_assert` 로 고정합니다. 참조가 캘리브레이션 값을 전사해서 LayerNorm `var` 가 27비트였습니다 (필요 36) |

`HbtxrCfgCheck<CFG>` 가 config 전역 불변식을, 각 유닛이 자기 `CI` 로 유도한 누산기 폭을
검사합니다.

## `REQ_M_BITS` — 18

requant 승수 포트의 폭입니다. `algorithm` 의 `i_ops.DYADIC_MULT_BITS` 와 **같은 계약의
양 끝**이고, tb 로더가 안 맞는 승수를 **적재 시점에 거부**하므로 어긋나면 조용한 절단이
아니라 실패입니다.

33이었던 이유는 결정이 아니라 부작용입니다 — dyadic 탐색이 단조라 **언제나 `shift_max`
에 착지**해서, 비율이 1을 넘는 엣지가 전부 33비트를 요구했습니다. `acc(20b)·M` 이 53비트라
DSP48E2 에 안 들어갑니다. 18이면 **38비트**이고 B 포트에 그대로 들어갑니다.

**조인 것은 shift 가 아니라 승수입니다** — shift 범위는 2~31 그대로 남습니다.
근거·측정·무손실 범위: `algorithm/docs/reports/2026-07-31-requant-multiplier-width.md`
(브랜치 `rewrite/flat-functional`).

구 `config.h`(256×256·256토큰)는 논문과 다릅니다. `board/` 는 아직 비어 있습니다.
