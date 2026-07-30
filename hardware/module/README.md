> **작성** 2026-07-29 · **갱신** 2026-07-30
> **상태** active — **M2 이관 완료 (96/96). 합성 검증은 안 됐습니다** (아래)
> **소유** hardware

# module — HLS 소스와 테스트벤치

```
module/
├── include/  19   템플릿 라이브러리·공통 헤더
├── src/      19   top + 모듈 구현
├── tb/       12   테스트벤치 (tb_*.cpp)
└── golden/   46   골든 벡터 17 + 참조 계약·스펙 19 + inputs/outputs/vref_p0/weights 10
```

출처 `archive/hardware/hls/` 67 + `archive/hardware/refs/` 29 — **96개 전부, 해시 대조 미이관 0.**
파일명·디렉토리명 무변경 ([M1과 같은 이유](../config/README.md) — 기계가 읽습니다).

## 구현이 셋입니다 — 트리만 봐서는 안 보입니다

census 판정이고 재구성 전에 알아야 합니다
([census §3.5](../docs/reports/2026-07-29-hardware-census.md)):

| top | 형태 | 비트스트림 | 판정 |
|---|---|---:|---|
| `hgtxr_e2e_axis_top` | `include/hgtxr_e2e_vit.hpp` **단일 헤더 4,503줄** | **6** | **정본** |
| `hgtxr_top` | 모듈 조립 (top + `src/*.cpp` 15) | 1 | 살아 있음 |
| `hgtxr_search_profile_top` / `hgtxr_track_profile_top` | `src/hgtxr_mode_profile_top.cpp`(파일) 안의 **top 둘**. `HGTXR_MODE_PROFILE_TOP` 환경변수가 고름 | 2 | 살아 있음 |

> **`hgtxr_mode_profile_top`이라는 심볼은 없습니다** — 파일명입니다. 실제 top은
> `src/hgtxr_mode_profile_top.cpp:446`과 `:471`의 두 함수이고,
> `run_mode_profile_q4w8a_csynth.tcl:15-17`이 환경변수로 하나를 골라 `set_top`합니다.
> 선택된 이름이 패키징 IP의 VLNV가 되므로 **비트스트림 2개는 별개 IP 코어**입니다.

**정본인 `hgtxr_e2e_vit.hpp`는 `include/hgtxr_cyclic_*.hpp` 템플릿 라이브러리의
`cyclic_transformer_block`(106줄)을 0회 사용하고 같은 일을 4,503줄로 재구현합니다.**
분해는 [계획 §5](../docs/plans/active/2026-07-29-hardware-reconstruction.md)의 P7이고,
합성 결과를 검증할 보드가 없어 **착수 금지**입니다.

## ⚠️ 이 디렉토리는 합성 검증되지 않았습니다

`archive/hardware/tests`의 **426 passed는 전부 Python 도구 테스트**입니다.
**HLS 빌드 입력을 검증하는 테스트는 0건입니다.** 이관이 옳다는 것은 `.cpp`/`.hpp`가
archive와 **바이트 동일**하다는 것까지만 증명하며, **컴파일되는지는 증명하지 않습니다** —
Vitis HLS가 없습니다.

## 전수 분석 결과 (2026-07-30)

전문: [docs/reports/2026-07-30-module-code-analysis.md](../docs/reports/2026-07-30-module-code-analysis.md).
45건 제기 · 21 확정 · 8 기각 · **16 미검증**(세션 한도). 요약:

| | |
|---|---|
| **csim 기본이 `float`** | `include/fixed_types.h`는 `HGTXR_HLS_FIXED_CSIM` 없으면 `typedef float`. **"csim이 golden을 통과"는 float 참조모델을 검증한 것**이고 양자화·포화·비트폭에 대해 말하는 바가 없습니다 |
| **`quant.h` 클램프가 죽어 있음** | ±31/32은 `ap_fixed<16,6>` 범위인데 배포 설정은 `BIT_WIDTH 8` → `ap_fixed<8,4>` → **±8**. ±8에서 이미 넘친 값을 ±31에서 막습니다 |
| **`hgtxr_data_to_axis` 부호확장 없음** | `BIT_WIDTH<16`에서 상위 비트가 0으로 남는데 호스트는 16비트 부호로 디코드 → **음수 출력 전부 오독** |
| **cyclic 라이브러리 8개 헤더가 dark** | `hgtxr_top.cpp:57`의 매크로를 세우는 `run_cyclic_*.tcl` 9개를 호출하는 셸이 없습니다. 마지막 증거는 2026-06-08 수동 실행 |
| **LUT 48개 호출 0건** | `hgtxr_cyclic_math.hpp:192-756`. LayerNorm·softmax는 레이어별 캘리브레이션을 쓰는데 Q/K/V·attn 출력은 안 씁니다. **삭제 금지** — `golden/hgpipe_lut_math_contract.json`이 미러링을 강제 |
| **테스트벤치 12→6만 빌드에 걸림** | 그리고 **가장 잘 만든 3개가 안 걸린 쪽**입니다 (`tb_cyclic_{head_attention,s2_projection,primitives}.cpp`) |

**위 결함들은 고치지 않았습니다** — Vitis HLS가 없어 고친 결과를 판정할 수 없습니다.
특히 `fixed_types.h`를 고치면 기존 golden이 전부 실패할 수 있고, 그게 정상인지 회귀인지
구분할 방법이 없습니다.

### 2026-07-30 삭제 2건

`tb/tb_mlp.cpp` · `tb/tb_attention.cpp` (각 8줄). 어떤 tcl·셸·py도 참조하지 않고(전수 확인),
초기화 없는 버퍼로 스테이지를 한 번 호출한 뒤 결과를 확인하지 않고 `return 0`합니다.
**DUT가 빈 함수여도 통과**하므로 없는 것보다 나쁩니다.

### M3에서 갚아야 하는 부채 — golden 분리

테스트벤치가 golden을 `#include "e2e_axis_vector_*_golden.hpp"`로 **무수식 상대 경로**
참조합니다. M2에서 `tb/`와 `golden/`을 나눴으므로 **새 트리에서는 이 include가 실패**합니다
(archive는 같은 디렉토리라 무해, 빌드도 아직 archive를 읽습니다).
분리 자체는 옳지만 공짜가 아니었고, **M3에서 tcl에 `-I ../golden`을 추가**해야 합니다.

## 알려진 결함 — 이관하며 확인, **고치지 않음**

### `golden/weights/` 매니페스트 3개는 짝 바이너리가 없습니다

`cyclic_weights_s2_block_software_initial{,_q4,_q4_head64}_manifest.json` 이 선언하는

```json
"binary": "hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_head64.bin"
```

**그 `.bin`이 저장소에 없습니다.** `.gitignore`도 없으니 커밋되지 않은 것입니다. 그리고
`tb/tb_cyclic_s2_block_vector.cpp:19`가 바로 그 파일을 읽습니다 — **이 테스트벤치는 돌 수
없습니다.** 테스트 스위트는 이걸 못 잡습니다 (관련 테스트 0건).

**지우지 않습니다.** 레이아웃 기록으로 값이 있고, 체크포인트가 생기면 재생성 검증의
기준이 됩니다.

### `golden/weights/e2e_m_axi_*.bin` (12MB)은 원칙상 산출물인데 재생성 불가입니다

`tools/export_e2e_m_axi_weights.py`가 만드는 파일이고 `check_third_goal_preflight.py`가
게이트 입력으로 읽습니다. 그런데 생성에 필요한 `software_initial_weights.pt`가 저장소에
없습니다. **다시 만들 수 없는 산출물은 실질적으로 골든**이라 `workspace/`(gitignore)가
아니라 여기 둡니다 — 옮겼으면 유일본이 사라집니다.

### `include/common.h`의 미호출 선언 3개

`matmul`(22행) · `layernorm_stage`(41) · `softmax_stage`(42). `hgtxr_top.cpp` 호출 0회.

**파일을 편집하지 않았습니다 — 주석조차 달지 않았습니다.** 주석을 달면 archive와 해시가
달라져 "내용 동일 이관"을 증명하는 `check_migration_manifest.py`가 무력해집니다. 그리고
계획 §3이 *"census는 삭제 근거가 아니다, 컴파일러로 확인한 뒤"*라고 적어 뒀는데 그 컴파일러가
없습니다. **코드가 실제로 바뀌는 것은 P7입니다.**

## ⚠️ 아직 살아 있지 않습니다

**사본입니다.** 빌드 스크립트와 도구는 여전히 `archive/hardware/hls/`·`refs/`를 읽습니다.
전환은 **M5**입니다. 그 전에 경로를 바꾸면 426 기준선이 깨집니다. **여기를 고치지 마십시오.**

---

계획: [docs/plans/active/2026-07-29-hardware-reconstruction.md](../docs/plans/active/2026-07-29-hardware-reconstruction.md) ·
대장: [docs/plans/active/2026-07-29-code-migration-manifest.md](../docs/plans/active/2026-07-29-code-migration-manifest.md)
