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
| `hgtxr_mode_profile_top` | 모드별 프로파일링 | 2 | 살아 있음 |

**정본인 `hgtxr_e2e_vit.hpp`는 `include/hgtxr_cyclic_*.hpp` 템플릿 라이브러리의
`cyclic_transformer_block`(106줄)을 0회 사용하고 같은 일을 4,503줄로 재구현합니다.**
분해는 [계획 §5](../docs/plans/active/2026-07-29-hardware-reconstruction.md)의 P7이고,
합성 결과를 검증할 보드가 없어 **착수 금지**입니다.

## ⚠️ 이 디렉토리는 합성 검증되지 않았습니다

`archive/hardware/tests`의 **426 passed는 전부 Python 도구 테스트**입니다.
**HLS 빌드 입력을 검증하는 테스트는 0건입니다.** 이관이 옳다는 것은 `.cpp`/`.hpp`가
archive와 **바이트 동일**하다는 것까지만 증명하며, **컴파일되는지는 증명하지 않습니다** —
Vitis HLS가 없습니다.

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
