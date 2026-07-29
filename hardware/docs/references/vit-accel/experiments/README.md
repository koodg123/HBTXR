> **HBTXR 주석 (2026-07-29)** — 아래 원문은 **평탄화 전 구조**를 설명합니다. 파일 10개가
> 디렉토리 8개에 4단계로 흩어져 있었고 그중 4개는 디렉토리 하나에 파일 하나였습니다.
> 경로가 담던 정보를 파일명으로 옮겼고, 원래 경로는 아래 표에 보존합니다.
> 평탄화 전 원본: `archive/hardware/docs/references/vit_accel/experiments/`.

## 원래 경로 ↔ 지금 파일명

| 원래 경로 | 지금 |
|---|---|
| `pnr/vck190_step4_ooc_status_20260408.md` | [2026-04-08-vck190-step4-ooc-status-pnr.md](2026-04-08-vck190-step4-ooc-status-pnr.md) |
| `pnr/zcu102_step4_bitstream_status_20260408.md` | [2026-04-08-zcu102-step4-bitstream-status-pnr.md](2026-04-08-zcu102-step4-bitstream-status-pnr.md) |
| `pnr/deit_tiny_full_multi_board_status_20260410.md` | [2026-04-10-deit-tiny-full-multi-board-status-pnr.md](2026-04-10-deit-tiny-full-multi-board-status-pnr.md) |
| `pnr/vck190_deit_tiny_full_status_20260410.md` | [2026-04-10-vck190-deit-tiny-full-status-pnr.md](2026-04-10-vck190-deit-tiny-full-status-pnr.md) |
| `pnr/zcu102_deit_tiny_full_status_20260410.md` | [2026-04-10-zcu102-deit-tiny-full-status-pnr.md](2026-04-10-zcu102-deit-tiny-full-status-pnr.md) |
| `pnr/zu15eg_deit_tiny_full_status_20260410.md` | [2026-04-10-zu15eg-deit-tiny-full-status-pnr.md](2026-04-10-zu15eg-deit-tiny-full-status-pnr.md) |
| `deit_tiny_baseline/full_deit_tiny_fit/step2_hls_resource_summary.md` | [deit-tiny-baseline-step2-hls-resource-summary.md](deit-tiny-baseline-step2-hls-resource-summary.md) |
| `deit_tiny_baseline_ooc_rerun/full_deit_tiny_fit/step2_hls_resource_summary.md` | [deit-tiny-baseline-ooc-rerun-step2-hls-resource-summary.md](deit-tiny-baseline-ooc-rerun-step2-hls-resource-summary.md) |
| `zu15eg/deit_tiny_baseline/full_deit_tiny_fit/step2_hls_resource_summary.md` | [zu15eg-deit-tiny-baseline-step2-hls-resource-summary.md](zu15eg-deit-tiny-baseline-step2-hls-resource-summary.md) |
| `vck190/deit_tiny_baseline/full_deit_tiny_fit/step5_vck190_pnr_summary.md` | [vck190-deit-tiny-baseline-step5-pnr-summary.md](vck190-deit-tiny-baseline-step5-pnr-summary.md) |

**baseline과 ooc-rerun의 step2 요약은 바이트 단위로 동일합니다.** 한쪽을 지우지 않은 이유:
"OOC로 다시 돌렸는데 자원 요약이 한 글자도 안 바뀌었다"가 그 실험의 결과이고, 파일을 하나로
합치면 그 사실이 사라집니다. `zu15eg` 것은 내용이 다릅니다.

---

*아래는 ViT_Accel 원문입니다. 무편집 — 위 표가 설명하는 "평탄화 전" 구조입니다.*

# Experiment Reports

This directory collects markdown reports that were produced and curated during multi-board experiments.

Current organization:

- `pnr/`
  - board implementation and bitstream status reports
- `vck190/`
  - VCK190-specific experiment summaries such as Step5 handoff reports
- `zu15eg/`
  - ZU15EG HLS summary reports
- `deit_tiny_baseline/`
  - baseline Step2 HLS summaries
- `deit_tiny_baseline_ooc_rerun/`
  - rerun or alternate Step2 HLS summaries

These files were moved from `workspace/artifacts/reports/` so that experiment findings live under `docs/` together with the rest of the project documentation.
