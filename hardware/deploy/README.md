> **작성** 2026-07-29 · **갱신** 2026-07-30
> **상태** active — **M4 완료 (9/9). 새 트리 테스트는 M5 이후에 돕니다** (아래)
> **소유** hardware

# deploy — 보드 런타임과 호스트

```
deploy/hgtxr/                       파이썬 패키지 — 이름 유지 (import 가 의존)
├── hgtxr_overlay.py            209  베이스 오버레이 · fixed16 변환 · hwh 파싱
├── e2e_axis_dma_overlay.py     278  AXIS DMA 경로
├── e2e_m_axi_overlay.py        153  M-AXI 경로
├── e2e_m_axi_weights.py        223  가중치 적재
├── run_e2e_axis_dma_smoke.py   280  ┐
├── run_e2e_axis_dma_hybrid_smoke.py  274  ├ 보드 스모크 러너
├── run_e2e_m_axi_smoke.py      123  ┘
├── test_hgtxr_overlay.py       582  ★ 아래
└── __init__.py
```

출처 `archive/hardware/pynq/hgtxr/` — **`.py` 9개 전부, 해시 대조 미이관 0.**

## 재지정이 필요 없었습니다

`.bit`/`.hwh` 경로를 **전부 인자·설정으로 받습니다** (`args.bit` `config.hwh` `self.hwh`).
tcl과 달리 트리 경로가 하드코딩돼 있지 않아 M3-2 같은 작업이 없었습니다.

## `.bit` / `.hwh` 18개는 안 옮겼습니다

산출물이라 `workspace/`(gitignore) 소관인데, 옮기면 **커밋된 유일본이 사라집니다.**
`archive/hardware/pynq/hgtxr/`에 그대로 둡니다 — `module/golden/weights`의
`e2e_m_axi_*.bin` 때와 같은 판단입니다.

## ★ `test_hgtxr_overlay.py` — 회귀 기준선 밖에 있던 24개

**보드가 필요 없습니다.** `unittest.mock` 기반이고 어서션 129개입니다.

```
archive 사본 : 24 passed          ← 실제로 통과합니다
새 트리 사본 : ModuleNotFoundError: export_e2e_m_axi_weights
```

`tests/`가 아니라 `pynq/`에 있어서 **아무도 세지 않았습니다.** 이 저장소의 회귀 기준선은
426이 아니라 **450**입니다:

```bash
python -m pytest archive/hardware/tests archive/hardware/pynq/hgtxr/test_hgtxr_overlay.py -q
# 49 failed, 450 passed
```

새 트리 사본이 안 도는 이유는 `TOOLS_DIR = HARDWARE_ROOT/"tools"`가 아직 비어 있어서입니다
(`hardware/tools/`, M5). **M5가 끝나면 새 트리 사본도 돕니다.**

## 보드가 필요한 것 / 아닌 것

| | |
|---|---|
| 보드 불필요 | `test_hgtxr_overlay.py` 24개 (mock) |
| **보드 필요** | `run_*_smoke.py` 3개 — ZCU104 물리 접근 ([STATUS Blocked](../docs/STATUS.md)) |

---

계획: [docs/plans/active/2026-07-29-hardware-reconstruction.md](../docs/plans/active/2026-07-29-hardware-reconstruction.md) ·
대장: [docs/plans/active/2026-07-29-code-migration-manifest.md](../docs/plans/active/2026-07-29-code-migration-manifest.md)
