> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 채우는 중
> **소유** hardware

# config — 보드·설계 파라미터

## 들어가는 것

- `board/` : 보드별 설정 (`zcu104.yaml` `vck190.yaml` `vitis_hls.yaml`)
- `design/` : HLS 설계 파라미터 헤더 (`zcu104_*_defines.h`)
- 양자화·스윕 설정

## 들어가지 않는 것

- **생성된 설정**은 `workspace/`로. 여기 있는 것은 사람이 쓴 입력뿐입니다.

## 이관 출처

`archive/hardware/configs/` (33개)

> 이 README는 **이관 체크리스트**입니다. 구 트리(`archive/hardware/`)가 기준이고,
> 계획은 [docs/plans/active/2026-07-29-hardware-reconstruction.md](../docs/plans/active/2026-07-29-hardware-reconstruction.md)에 있습니다.
