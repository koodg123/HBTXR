> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 채우는 중
> **소유** hardware

# tools — 감사·검증·패키징 자동화

## 들어가는 것

- `_lib/` : 공용 유틸. 구 트리에서 `load_json`이 22곳, `sha256_file` 11곳, `normalize_roots` 10곳에 복제돼 있었습니다
- `audit/` `validate/` `check/` `package/` `discover/` : 역할별
- `tests/` : 이 도구들의 파이썬 테스트

## 들어가지 않는 것

- **HLS 테스트벤치**는 `module/tb/`로. 여기 `tests/`는 파이썬 도구용입니다.
- 6개 디렉토리(config/module/build/deploy/docs/workspace) 어디에도 들어가지 않아 7번째로 둡니다 — 이들은 빌드도 배포도 아닌 **프로세스 게이팅**입니다.

## 이관 출처

`archive/hardware/tools/` (87) + `archive/hardware/tests/` (60)

> 이 README는 **이관 체크리스트**입니다. 구 트리(`archive/hardware/`)가 기준이고,
> 계획은 [docs/plans/active/2026-07-29-hardware-reconstruction.md](../docs/plans/active/2026-07-29-hardware-reconstruction.md)에 있습니다.
