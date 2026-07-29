> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 재구성 중, 대부분 비어 있음
> **소유** hardware

# hardware — HBTXR 가속기

**2026-07-29에 처음부터 다시 구성하는 중입니다.** 구 트리는
[`archive/hardware/`](../archive/hardware/)에 그대로 보존되어 있고 **기준(reference)이자
동작하는 코드**입니다. 새 트리가 완성될 때까지 실제로 도는 것은 archive 쪽입니다.

- 왜 이렇게 하는가 · 무엇을 어디로 : [docs/plans/active/2026-07-29-hardware-reconstruction.md](docs/plans/active/2026-07-29-hardware-reconstruction.md)
- 구 트리 전수 조사 : [docs/reports/2026-07-29-hardware-census.md](docs/reports/2026-07-29-hardware-census.md)
- 지금 무엇이 진행 중인가 : [docs/STATUS.md](docs/STATUS.md)

## 구조 — 한 역할은 한 디렉토리

| 디렉토리 | 담는 것 | 이관 |
|---|---|---|
| [`config/`](config/README.md) | 보드 설정 · HLS 설계 파라미터 | ⬜ |
| [`module/`](module/README.md) | HLS 소스 · 테스트벤치 · 골든 벡터 | ⬜ |
| [`build/`](build/README.md) | 합성 · P&R · 비트스트림 | ⬜ |
| [`deploy/`](deploy/README.md) | PYNQ 오버레이 · 호스트 | ⬜ |
| [`tools/`](tools/README.md) | 감사 · 검증 · 패키징 자동화 | ⬜ |
| [`docs/`](docs/STATUS.md) | 문서 | ✅ **완료 (D1~D5)** — 상태는 [docs/STATUS.md](docs/STATUS.md) |
| [`workspace/`](workspace/README.md) | 산출물 (**gitignore**) | ⬜ |

## 왜 `tools/`가 7번째인가

87개 스크립트가 읽는 대상이 전방위입니다 — `generated` 339회 · `docs` 208 · `pynq` 206 ·
`hls` 128 · `vivado` 64 · `refs` 52 · `configs` 28. 이들이 하는 일은 **서명·게이팅 프로세스
자동화**이고, `build/`나 `deploy/` 안에 넣으면 그 디렉토리의 정의가 틀려집니다.

## 각 README는 이관 체크리스트입니다

구 트리에는 `.gitkeep`만 든 **빈 디렉토리가 24개** 있었습니다 — 설계됐지만 아무도 채우지
않았고, 무엇이 들어가야 하는지 적힌 곳이 없었기 때문입니다. 그래서 이번에는 디렉토리마다
**무엇이 들어가고, 무엇이 들어가지 않으며, archive의 어디서 오는지**를 README에 적습니다.

## archive/hardware/ 를 지우지 않는 이유

- **동작하는 유일한 구현입니다.** `pytest archive/hardware/tests` → 426 passed
  (49 failed는 Xilinx 설치 경로·형제 저장소를 단언하는 환경 결합, 이동 전과 동일).
- 비트스트림 9개와 합성 리포트가 거기에만 있습니다.
- 새 트리의 각 항목은 archive의 무엇을 옮긴 것인지 추적되어야 합니다.

재구성이 끝나고 검증되기 전까지 archive는 **읽기 전용 기준**입니다.
