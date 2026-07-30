> **작성** 2026-07-29 · **갱신** 2026-07-31
> **상태** active — **M5-1 완료 (147/147 복사). 재지정은 실패해서 되돌렸습니다** (아래)
> **소유** hardware

# tools — 감사·검증·패키징 자동화

```
tools/         87  argparse 진입점
└── tests/     60  + conftest.py (새로 쓴 유일한 파일)
```

출처 `archive/hardware/tools/` 87 + `archive/hardware/tests/` 60 — **147개, 해시 대조 미이관 0.**
역할별 재배치(`_lib/ audit/ validate/ check/ package/ discover/`)는 **P2**입니다.
빈 골격 디렉토리는 지웠습니다 — 채울 때 만듭니다.

6개 디렉토리 어디에도 안 들어가서 7번째로 둡니다. 이들은 빌드도 배포도 아닌
**프로세스 게이팅**입니다.

## ⚠️ 이 사본으로는 테스트가 기준선에 못 미칩니다

```
archive  48 failed / 437 passed     ← 회귀 기준선
새 트리  67 failed / 418 passed     ← 통과 19개 감소
```

**이관 규칙대로 멈췄습니다.** `tools/`는 아직 **archive 사본이 정본**입니다.

### 원인 — 도구와 테스트가 레이아웃을 **함께** 인코딩합니다

M3-2가 `hardware/hls` → `module`, `configs` → `config/design`,
`refs` → `module/golden` 으로 바꿨습니다. 그런데:

| | |
|---|---|
| 도구 **27개** | `ROOT / "hls" / "include" / …` 형태로 **구 경로를 조립**합니다 |
| 그 테스트들 | 같은 구 경로로 픽스처를 만들고 **그 경로를 단언**합니다 |

**한쪽만 고치면 더 나빠집니다.** 실제로 도구 29개를 재지정했더니 67 → **83 failed**가
됐습니다. 되돌렸습니다.

> M3-2 때 tcl 재지정이 기계적 sed로 끝난 것과 다릅니다. tcl은 **경로를 소비만** 했고,
> 여기는 도구와 테스트가 **같은 레이아웃 가정을 양쪽에서** 들고 있습니다.

→ 도구와 테스트를 **함께** 재지정해야 하고, 그건 **P2(패키지화)**의 일입니다.
P2가 `sys.path` 조작 70곳을 없애면서 경로 계산을 한 곳으로 모읍니다.

### `conftest.py` — 새로 쓴 유일한 파일

테스트가 `parents[1] / "tools"` 로 도구를 찾습니다. 구 트리에서 `tests/`와 `tools/`가
**형제**였기 때문입니다. 새 트리는 `tests/`가 `tools/` **안**이라 그 경로가 빗나갑니다.
**60개 파일을 고치는 대신** conftest 한 줄로 넣습니다 — 파일들은 archive와 바이트 동일을
유지하고, `check_migration_manifest.py`의 증명이 살아 있습니다.

`test_xr_accel_config_schema.py`는 `collect_ignore`입니다. 모듈 최상위에서
`REPO/"hardware"/"tools"/…`를 조립해 **파일 경로로 import**하는데, 깊이가 한 단계 달라
`hardware/hardware/…`가 되고 **수집 단계에서 전체 실행을 막습니다.**
archive 사본에서는 수집은 되고 1건 실패합니다.

## M5에 남은 것 — 전부 한 덩어리입니다

| | 왜 아직 안 했나 |
|---|---|
| 도구+테스트 경로 재지정 | 위 — **P2** |
| `generated/` → `workspace/` (93곳) | 도구가 **339번** 읽습니다. 위와 같은 커밋이어야 갈라지지 않습니다 |
| `.sh` 106개 CRLF 정규화 | `.gitattributes` 에 `*.sh text eol=lf` + `git add --renormalize`. 이관 검증과 섞으면 원인 분리가 안 됩니다 |
| `deploy/hgtxr/test_hgtxr_overlay.py` 새 트리 복구 | `TOOLS_DIR`가 여기를 봅니다. 위가 풀리면 같이 풀립니다 |

---

계획: [docs/plans/active/2026-07-29-hardware-reconstruction.md](../docs/plans/active/2026-07-29-hardware-reconstruction.md) ·
대장: [docs/plans/active/2026-07-29-code-migration-manifest.md](../docs/plans/active/2026-07-29-code-migration-manifest.md)
