> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — **M1 이관 완료 (33/33). 아직 아무도 읽지 않습니다** (아래)
> **소유** hardware

# config — 보드·설계 파라미터

**사람이 쓴 입력만** 둡니다. 생성된 설정은 [`../workspace/`](../workspace/README.md)로 갑니다.

## 구조

```
config/
├── board/     타깃·툴체인 핀              3
│   ├── zcu104.yaml       part xczu7ev-ffvc1156-2-e · clock 5.0ns
│   ├── vck190.yaml
│   └── vitis_hls.yaml    HLS 프로젝트 — top_function · 모델 차원 · part
├── design/    HLS 설계 파라미터           11
│   ├── zcu104_*_defines.h            9 — s0 · s1_weight · s2_{attn,block,mlp,qkv} · e2e_q4w8a
│   ├── quant_int8.yaml
│   └── sweeps/                       DSE 스윕
└── xr_accel/  정규화 매니페스트 체계      19
    ├── schema.json                   `hbtxr.xr-accel-manifest/v1`
    └── designs/ experiments/ models/ targets/
```

## 이관 기록 (M1, 2026-07-29)

출처 `archive/hardware/configs/` — **33개 → 33개. 내용 해시 대조로 미이관 0건.**

| 결정 | 근거 |
|---|---|
| **파일명을 바꾸지 않았습니다** | 문서 규약 §3은 **문서**에만 적용됩니다. `zcu104_cyclic_s0_defines.h`는 Vivado tcl이 `#include`하고 `static_validate_hgtxr.py`가 경로 리터럴로 읽습니다. 개명하면 그냥 깨집니다 |
| `xr_accel/` 하위 구조 유지 | `schema.json`이 `designs/ experiments/ models/ targets/` 배치를 전제합니다. 소비자는 `tools/xr_accel_config.py`와 `test_xr_accel_config_schema.py`입니다 |
| `sweeps/`가 파일 1개인데 유지 | 스윕은 **쌓일 예정**입니다 (계획 §6의 DSE). 늘어날 디렉토리는 냄새가 아닙니다 |
| `vitis_hls.yaml`을 `board/`에 | 모델 차원도 담아 애매하지만 `part`를 핀하는 타깃 설정입니다. 쪼개면 한 파일을 두 곳에서 봐야 합니다 |

## 무엇이 실제로 읽히나 (2026-07-30 조사 · **판정 아님, 기록만**)

`archive/hardware/` 전수 텍스트 대조. **"사용/미사용" 2칸으로는 안 나뉩니다.**

| 분류 | 수 | 소비자 |
|---|---:|---|
| **① 도구가 이름으로 읽음** | 11 | `static_validate_hgtxr.py`(defines 8 + sweep) · `check_third_goal_preflight.py`(e2e_q4w8a) · `generate_hls_config.py`(vitis_hls) |
| **② 디렉토리 열거로 읽힘** | 19 | `xr_accel_config.py`가 `configs/{category}/*.json` 을 glob |
| **③ 빌드만 씀 — 지금 확인 불가** | ①과 겹침 | Vivado tcl이 `-define_file …_defines.h` 로 전달. **Xilinx 미설치라 실행 검증 불가** |
| **④ 참조 0건** | **3** | `board/zcu104.yaml` · `board/vck190.yaml` · `design/quant_int8.yaml` |

### ②는 코드가 스스로 완전성을 단언합니다

```python
inventory = {f"configs/{category}/{path.name}"
             for category in ("experiments","models","designs","targets")
             for path in (root/category).glob("*.json")}
if reached != inventory:      # 도달 못 한 파일이 하나라도 있으면 에러
```

`xr_accel/` 19개는 **파일명이 코드에 없어도 전부 쓰입니다.** 하나라도 고아가 되면
`normalize_all_experiments`가 실패합니다. 이름으로만 grep하면 19개를 통째로
미사용으로 오분류합니다 — 1차 측정이 실제로 그랬습니다.

### ④ 3개 — 미사용 "후보"일 뿐입니다

코드·스크립트·tcl 어디에도 참조가 없고 문서에서만 각 3회 언급됩니다. 그런데
`zcu104.yaml`의 내용은 읽히는 `vitis_hls.yaml`과 겹칩니다 (둘 다 `part: xczu7ev-ffvc1156-2-e`,
`clock_period_ns: 5.0`). **`zcu104.yaml`에만 있는 것은 `board: xilinx.com:zcu104:part0:1.1`**이고,
그 값을 쓸 만한 코드는 **M3에서 옮길 `vivado/scripts/` 31개**입니다.

**그래서 판정하지 않습니다. M3 이후에 결정합니다.**

### 이 조사가 못 보는 것

1. **동적 경로 조립** — ②가 그 사례. 측정을 **세 번** 고쳐서야 잡았습니다
   (stem 오탐 → 슬래시 앞 경계 오류 → 알려진 참 사례로 자기검증 추가).
2. **Xilinx 실행 경로** — tcl이 넘긴 defines를 실제로 소비하는지는 합성을 돌려야 압니다.
3. **사람이 수동 실행** — `--config configs/vck190.yaml` 같은 사용은 흔적이 없습니다.

## ⚠️ 아직 살아 있지 않습니다

**이 디렉토리는 사본이고, 도구는 여전히 `archive/hardware/configs/`를 읽습니다.**
전환은 **M5**(`tools/` 이관)에서 일어납니다 — 그 전에 도구 경로를 바꾸면
`archive/hardware/tests`의 **426 passed** 기준선이 깨져 회귀를 구분할 수 없습니다.

그때까지 **여기를 고치지 마십시오.** 두 벌이 갈라지면
`python scripts/check_migration_manifest.py`가 잡습니다.

## 이관 중 발견 — P8로 넘김

`tools/export_weights.py`와 `tools/export_s2_pytorch_golden.py`가 `--config` 기본값으로
**`software/configs/base.yaml`**을 씁니다. HBTXR에 `software/`는 없습니다 (알고리즘 트리는
`algorithm/`). 원본 저장소 레이아웃의 잔재이고 **환경 결합**이라 P8에서 처리합니다.

---

계획: [docs/plans/active/2026-07-29-hardware-reconstruction.md](../docs/plans/active/2026-07-29-hardware-reconstruction.md) ·
대장: [docs/plans/active/2026-07-29-code-migration-manifest.md](../docs/plans/active/2026-07-29-code-migration-manifest.md)
