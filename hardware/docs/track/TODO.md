> **작성** 2026-07-05 · **갱신** 2026-07-05
> **상태** active — 내용은 2026-07-15 구 트리에서 이어받음, 재구성 진행분을 여기에 이어 씁니다
> **소유** hardware

# HGTXR Hardware TODO

> ### ⚠️ 아래 항목의 경로는 **구 트리 기준**입니다 (2026-07-29 확인)
>
> 이 항목들은 2026-07-05에 쓰였고, 그 뒤 구 트리가 `archive/hardware/`로 옮겨졌습니다.
> **지시대로 실행하면 실패합니다.** 지금의 대응은:
>
> | 항목에 적힌 것 | 지금 어디 |
> |---|---|
> | `tools/run_zcu104_c3b_smoke_remote.py` | `archive/hardware/tools/run_zcu104_c3b_smoke_remote.py` |
> | `pynq/hgtxr/` | 없음. 런타임 코드는 `deploy/`로 갑니다 (**아직 이관 전**) |
> | `generated/` | **이관되지 않았습니다.** `archive/`에도 없습니다 — 생성물이라 커밋되지 않았습니다 |
>
> 본문을 고치지 않는 이유: 항목의 내용(무엇을 해야 하나)은 여전히 유효하고, 경로는
> 코드 이관(계획 §6 P0~P8)이 끝나면 새 위치로 한 번에 갱신됩니다. 그 전에 개별로 고치면
> 두 번 고치게 됩니다. **어차피 전부 보드 접근 대기이므로 지금 실행할 수 없습니다**
> ([../STATUS.md](../STATUS.md) Blocked).

## P0
- Capture/import C3b AXIS/DMA physical smoke JSON.
- Resolve exact XR-VITs sibling or approved replacement policy.
- Run/import ZCU104 physical smoke JSON for `softmax_input_x2` `dsp_mixed_stream` using either `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md` or `python3 tools/run_zcu104_c3b_smoke_remote.py --profile vref-p0-softmax-input-x2-dsp-mixed-stream --host <zcu104-ip-or-host> --user xilinx --execute`.
- Copy successor board result back to `pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json` so preflight/projection can promote it.
- Keep C3b as protected baseline until board-smoke and XR-VITs reference gates are closed.
- For additional non-current `VREF-P0-01` scale successors, regenerate a dedicated golden header and pass CSim before promotion.
- Keep `VREF-P0-02` QKV cache URAM promotion as a successor-only experiment; HLS CSim/csynth, HLS IP package, routed overlay timing, and PYNQ bundle/session artifacts are present; physical smoke remains pending.

## P1
- Check nonlinear bottleneck before TATAA mixed-precision path.
- Prepare search/track expert ablation only as bounded/static routing.
- Prepare exact-attention fallback for ViTALiTy/ViTCoD ablations.

## P2/P3
- Keep FlexLLM-style generator as tooling-only.
- Keep LUT-heavy/ternary methods as negative controls.
