# HBTXR

Hybrid event-camera pupil/gaze model (IEEE JETCAS) — algorithm, quantization, and FPGA
accelerator, assembled from the three source branches in `HBTXR-Pool`.

## 문서를 찾으신다면

**[docs/INDEX.md](docs/INDEX.md)** — 저장소 전체 문서 색인 (자동 생성).
날짜·상태·소유가 한 표에 있습니다.

바로 가는 길:

| 알고 싶은 것 | 문서 |
|---|---|
| 지금 무엇이 진행 중이고 무엇이 막혀 있나 | [docs/STATUS.md](docs/STATUS.md) |
| 양자화/정수 그래프 현황 | [algorithm/docs/STATUS.md](algorithm/docs/STATUS.md) |
| 하드웨어 가속기 현황 | [hardware/docs/STATUS.md](hardware/docs/STATUS.md) |
| 문서를 어디에 써야 하나 | [docs/governance/DOC-CONVENTIONS.md](docs/governance/DOC-CONVENTIONS.md) |

## Layout

| 디렉토리 | 내용 | 문서 |
|---|---|---|
| `algorithm/` | 학습·평가·양자화·정수 그래프 (Python) | `algorithm/docs/` |
| `hardware/` | HLS·RTL·Vivado·PYNQ 가속기 | `hardware/docs/` |
| `docs/` | 횡단·거버넌스·외부 조사 | — |
| `references/` | 논문 참조와 보존된 레거시 코드베이스 | — |
| `third/` | 벤더링된 서드파티 (중첩 저장소 메타데이터 제외) | — |

문서는 **자신을 거짓으로 만들 수 있는 코드가 있는 트리**에 삽니다. 규칙은
[DOC-CONVENTIONS.md](docs/governance/DOC-CONVENTIONS.md)에 있습니다.

## Provenance

- Baseline branch: `HBTXR-etri-server/HBTXR`
- Retina/ERVT result packaging additions: `HBTXR-etri-desktop/HBTXR`
- Annotation and frame/crop experiment additions: `HBTXR-home/HBTXR`

Large checkpoints, row-level prediction tables, HDF5 files, and local run outputs are
intentionally excluded or routed to artifact storage paths — see
[docs/Artifact-Policy.md](docs/Artifact-Policy.md).

> **2026-07-29 정정.** 이 파일은 이전까지 루트에 `quantization/` 디렉토리가 있다고
> 안내했으나 **존재하지 않습니다** — Q7에서 `references/hardware/hg-pipe-quantization/`으로
> 이동했습니다. 또한 `docs/`를 전혀 언급하지 않아 저장소를 처음 여는 사람이 문서에 도달할
> 경로가 없었습니다.
