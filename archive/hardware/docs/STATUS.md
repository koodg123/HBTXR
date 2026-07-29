> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** hardware

# hardware STATUS

**이 파일이 "지금 무엇이 진행 중인가"의 단일 출처입니다.**
상세는 [track/TODO.md](track/TODO.md) · [track/PROGRESS.md](track/PROGRESS.md).

> **주의 — 이 STATUS는 기존 추적 문서에서 옮겨 적은 것이며, 마지막 하드웨어 커밋은
> 2026-07-15입니다.** 그 이후 하드웨어 쪽 작업이 없었으므로 아래 내용은 그 시점 상태
> 그대로입니다. 하드웨어 작업을 재개할 때 **가장 먼저 이 파일을 실제와 대조**해야 합니다.

---

## Active

2026-07-15 이후 하드웨어 커밋이 없습니다. 아래는 그 시점에 진행 중이던 항목입니다.

## Blocked

**모든 Blocked 항목은 "누가 풀 수 있나"를 반드시 적습니다.**

| 항목 | 무엇이 막고 있나 | 누가 풀 수 있나 |
|---|---|---|
| C3b AXIS/DMA physical smoke | ZCU104 **보드 접근** 필요 (`tools/run_zcu104_c3b_smoke_remote.py`) | 사용자 — 보드 접근 승인 |
| `VREF-P0-02` QKV cache URAM 승격 | HLS CSim/csynth·IP·overlay timing·PYNQ 번들 전부 있음. **physical smoke만 미완** | 사용자 — 보드 접근 |
| AQ2 board-runtime histogram | 보드 반복 측정 필요. 현재 mean/median/P95/P99는 **HLS 결정론적 추정치**이지 측정치가 아님 | 사용자 — 보드 접근 |
| AQ2 전력 분해 (DDR/센서 I/O rail, 블록별 dynamic) | Vivado 리포트가 top-level BD 계층까지만 분리 | 도구 제약 — 별도 조사 필요 |
| XR-VITs sibling 참조 정책 | 정확한 sibling 또는 승인된 대체 정책 미확정 | 사용자 결정 |

**하드웨어 쪽 Blocked는 대부분 "보드 접근"입니다.** 알고리즘 쪽의 A2(학습)와 대칭입니다 —
양쪽 다 사용자 승인이 있어야 풀립니다.

## Next (P1 이하, 착수 전)

- nonlinear 병목 확인 후 TATAA mixed-precision 경로 판단
- search/track expert ablation — bounded/static routing으로 한정
- ViTALiTy/ViTCoD ablation용 exact-attention fallback 준비

## Done — 최근

| 날짜 | 내용 |
|---|---|
| 2026-07-15 | XR accelerator 구성 검증 |
| 2026-07-05 | 저장소 상태 캡처 + 문서화 |
| 2026-06-30 | AQ2 Search/Track 측정 — Search 300MHz WNS 0.000ns 통과, Track WNS -0.017ns (사용자 허용 임계 내), 자원·전력 실측 |

## 알고리즘 서브시스템과의 미해결 접점

**`hardware/hls/include/hgtxr_cyclic_math.hpp`와 `algorithm/quantization/i_ops.py`가 같은
HG-PIPE 커널의 두 구현인데 교차 참조가 0건입니다.** 그리고 수치 규약이 다릅니다:

| | `algorithm/quantization/` | `hardware/hls/` |
|---|---|---|
| 데이터패스 | int8 affine + dyadic requant | `ap_fixed<16,6>`, 누산 `ap_fixed<32,12>` |
| LayerNorm | 정수 mean/var + 2세그먼트 rsqrt LUT | `identity`/`mean_center`/`affine` — **rsqrt 없음** |
| attention 스케일 | `1/√d`를 dyadic requant에 접음 | `ap_int<8> score_scale_shift` (PoT) |
| softmax 출력 | int8 | `clamp(rel, 0, 7)` — **3비트** |

**의도된 분기인지 드리프트인지 적힌 곳이 없습니다.** → `docs/contracts/NUMERICS-CONTRACT.md`
(미작성). `i_ops.table_quantize`에 HLS의 `(b, s, bound)`와 테이블을 먹여 원소별 대조가
가능합니다.
