> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 안내
> **소유** hardware

# plans — 디렉토리가 상태를 말합니다

- `active/` : 실행 중
- `done/` : 완료 — **삭제하지 않고 이동**합니다

파일을 열지 않아도 상태를 알 수 있어야 합니다
([규약 §7](../../../docs/governance/DOC-CONVENTIONS.md)).

## active

| 계획 | 내용 |
|---|---|
| [2026-07-29-hardware-reconstruction.md](active/2026-07-29-hardware-reconstruction.md) | 재구성 전체 — 생사 대장·M1~M5·P1~P8. **문서 완료, 코드 이관 진행 중.** §2 목표 구조는 채택되지 않았습니다 (정정 고지 부착) |
| [2026-07-29-code-migration-manifest.md](active/2026-07-29-code-migration-manifest.md) | **P0 이관 대장** — 625행 전수 판정. `.csv`가 데이터, `scripts/check_migration_manifest.py`가 검증 |

## done

| 계획 | 결과 |
|---|---|
| [2026-06-16-third-goal/](done/2026-06-16-third-goal/) | [experiments/2026-06-16-third-goal-audit/](../experiments/2026-06-16-third-goal-audit/) |
| [2026-06-17-directory-layout.md](done/2026-06-17-directory-layout.md) | **대체됨** — 정의한 24개 디렉토리가 비어 있었음 |

> **`2026-07-15-xr-accel-execution-plan.md`는 2026-07-29에 삭제했습니다.** 우리 계획이 아니라
> ViT_Accel의 [`PLAN.md`](../references/vit-accel/PLAN.md)에 저장소 이름
> (`koodg123/ViT_Accel`→`koodg123/XR_Accel`) 3곳을 치환한 사본이었습니다 — 128줄 중 실질
> 차이 4줄. 실행 기록도 결과를 가리킬 실험 디렉토리도 없었습니다.
> 원본은 `archive/hardware/docs/`에 남아 있습니다.
| [2026-07-29-docs-migration-map.md](done/2026-07-29-docs-migration-map.md) | [STATUS.md 문서 이관 결과](../STATUS.md) — archive md 159개 해시 대조, 미이관 0건 |
