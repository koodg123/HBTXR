> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 안내
> **소유** hardware

# reports — 분석 (개정 가능)

**결과가 아니라 해석**입니다. 결과는 [../experiments/](../experiments/)에 불변으로 있습니다
([규약 §6](../../../docs/governance/DOC-CONVENTIONS.md)).

| 리포트 | 내용 |
|---|---|
| [2026-07-29-hardware-census.md](2026-07-29-hardware-census.md) | **전수 semantic 검사** — 642 파일·285,261줄, 세 구현·생사 판정 |

**다중 보드 검증**은 여기 없습니다 →
[references/vit-accel/MULTI-BOARD-VALIDATION.md](../references/vit-accel/MULTI-BOARD-VALIDATION.md).
우리가 수행한 검증이 아니라 **선행 프로젝트의 것**입니다. 2026-07-29까지 이 표에
`2026-06-16-multi-board-validation.md`로 올라와 있었는데, 그 파일은 ViT_Accel 문서에
`ViT_Accel`→`XR_Accel` 한 줄을 치환한 사본이었습니다 (248줄 중 실질 차이 2줄).
| [2026-06-16-deit-cyclic-verification.md](2026-06-16-deit-cyclic-verification.md) | DeiT cyclic 검증 |
| [2026-06-17-cleanup.md](2026-06-17-cleanup.md) | 디렉토리 정리 기록 — **예외**, 아래 |

## `2026-06-17-cleanup.md`가 왜 여기 있나

이 문서는 실험 해석이 아니라 **파일 이동 기록**입니다. 원래대로면
[../track/CHANGELOG.md](../track/CHANGELOG.md)에 들어갈 내용입니다.

옮기지 않는 이유: `archive/hardware/docs/`에서 **무변경 복사**한 frozen 문서이고,
내용을 CHANGELOG의 한 항목으로 접으면 그건 이관이 아니라 재작성입니다. 그리고
2026-06-17 시점에는 hardware에 `CHANGELOG.md`가 없었습니다 — 변경 이력이
`PROGRESS.md`와 `log.md`에 섞여 있었고 이 문서가 그 증거입니다.

**2026-07-29 이후의 정리 기록은 `track/CHANGELOG.md`로 갑니다.**
