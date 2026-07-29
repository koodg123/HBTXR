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
