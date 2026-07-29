> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 채우는 중
> **소유** hardware

# module — HLS 소스와 테스트벤치

## 들어가는 것

- `include/` : 템플릿 라이브러리·공통 헤더
- `src/` : top + 모듈 구현
- `tb/` : 테스트벤치 (`tb_*.cpp`)
- `golden/` : 골든 벡터·참조 계약·가중치 매니페스트

## 들어가지 않는 것

- **빌드 스크립트**는 `build/`로.
- **생성된 골든 헤더**(`*_golden.hpp`)는 `golden/`에 두되 생성물임을 표시합니다.

## 이관 출처

`archive/hardware/hls/` (67) + `archive/hardware/refs/` (29)

> 이 README는 **이관 체크리스트**입니다. 구 트리(`archive/hardware/`)가 기준이고,
> 계획은 [docs/plans/active/2026-07-29-hardware-reconstruction.md](../docs/plans/active/2026-07-29-hardware-reconstruction.md)에 있습니다.
