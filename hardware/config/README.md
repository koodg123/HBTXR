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

## `REQ_M_BITS` — 지금 33입니다

requant 승수 폭입니다. **`dyadic_params` 의 성질이 아니라 설계가 제시하는 비율의 성질**입니다:
비율 1.15 는 shift 31 에서 33비트, 비율 4.5 는 34비트를 요구합니다. 그래서 tb 로더가
**들어가지 않는 승수를 거부**합니다 — 안 그러면 `ap_uint<REQ_M_BITS>` 가 조용히 자릅니다.

18로 내리면 DSP48E2 의 B 포트에 들어가고 곱이 38비트가 됩니다. **16비트까지 골든이 비트
동일**이라는 것이 실측돼 있습니다 —
근거는 `algorithm/docs/reports/2026-07-31-requant-multiplier-width.md`
(브랜치 `rewrite/flat-functional`. 이 워크트리에는 없습니다 — 두 브랜치가 합쳐지면 링크가 됩니다).
algorithm 쪽 변경이라 대기 중이고, **하드웨어는 기다리지 않습니다**: 상수 하나입니다.

구 `config.h`(256×256·256토큰)는 논문과 다릅니다. `board/` 는 아직 비어 있습니다.
