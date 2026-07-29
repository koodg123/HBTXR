> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 채우는 중
> **소유** hardware

# workspace — 산출물

## 들어가는 것

- HLS 프로젝트·합성 리포트
- 비트스트림·hwh·PYNQ 번들
- 서명(signoff) 아티팩트

## 들어가지 않는 것

- **`.gitignore` 대상입니다.** 예외는 `release/` — 승격된 산출물만 커밋합니다.
- 소스는 절대 여기 두지 않습니다.

## 이관 출처

구 트리의 `generated/`(코드가 339회 참조하나 저장소에 없었음) + `pynq/hgtxr/*.bit` + `artifacts/` + `reports/`

> 이 README는 **이관 체크리스트**입니다. 구 트리(`archive/hardware/`)가 기준이고,
> 계획은 [docs/plans/active/2026-07-29-hardware-reconstruction.md](../docs/plans/active/2026-07-29-hardware-reconstruction.md)에 있습니다.

---

## gitignore 정책

```gitignore
# hardware/workspace/.gitignore
*
!.gitignore
!README.md
!release/
!release/**
```

`workspace/` 전체를 무시하되 **`release/`만 예외**입니다 — 승격된 산출물(보드에서 검증된
비트스트림, 최종 리포트)만 커밋합니다.

**왜 전부 무시하지 않는가**: 구 트리는 비트스트림 9개를 커밋해 두었고, 그것이 보드나
Vivado 없이도 과거 결과를 확인할 수 있는 유일한 경로였습니다. 전부 무시하면 그 경로가
사라집니다. **왜 전부 커밋하지 않는가**: 합성 산출물은 크고 매 실행마다 바뀌어 이력을
오염시킵니다.
