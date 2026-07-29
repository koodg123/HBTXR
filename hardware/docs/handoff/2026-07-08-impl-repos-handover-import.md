> **작성** 2026-07-08 · **갱신** 2026-07-29
> **상태** consumed — 실행 지침으로 따르지 마십시오
> **소유** hardware

> ## ⚠️ 2026-07-29 만료 고지
>
> 이 핸드오프는 **소비되었습니다.** 아래 본문은 2026-07-08 시점의 기록이며, 그 이후
> 하드웨어 트리가 `archive/hardware/`로 이동하고 재구성이 시작되었습니다.
> 본문의 경로·상태는 당시 것입니다.
>
> **현재 상태**: [../STATUS.md](../STATUS.md)
>
> 본문은 편집하지 않습니다 — 어느 시점의 기록을 지금 참이 되도록 고쳐 쓰면
> 그것이 존재한 이유가 사라집니다 ([규약 규칙 G](../../../docs/governance/DOC-CONVENTIONS.md)).

---

# IMPL_REPOS And HANDOVER Hardware Import - 2026-07-08

## Scope

This pass imported hardware-side materials identified during the `IMPL_REPOS`
and `HANDOVER` review. The import keeps HBTXR-specific architecture docs,
ZCU104 config metadata, validation reports, and compact reference materials.
Generated Vivado/Vitis projects, bitstreams, large binary artifacts, and cache
trees were not imported.

## Imported Sources

### IMPL_REPOS/HW/XR_Accel

Source root:

- `/mnt/d/dataset/EV_Eye/paper_works/IMPL_REPOS/HW/XR_Accel`

Imported to:

- `hardware/docs/architecture/xr_accel/`
- `hardware/docs/status/xr_accel/`
- `hardware/docs/validation/xr_accel/`
- `hardware/configs/xr_accel/`

High-value content:

- `HBTXR_ARCH_FREEZE.md`
- `HBTXR_CYCLIC_IMPLEMENTATION.md`
- `ZCU104_CYCLIC_MAXPERF_HANDOVER.md`
- `ZCU104_CYCLIC_MAXPERF_PROGRESS.md`
- `DEIT_CYCLIC_VERIFICATION.md`
- `MULTI_BOARD_VALIDATION.md`
- HBTXR option/design/experiment JSON configs for ZCU104 cyclic and streaming
  paths.

Reason:

- XR_Accel is the most directly HBTXR-specific external hardware workspace.
  It documents search/track mode routing, cyclic architecture, DMA boundaries,
  Global Buffer, NoC/interconnect, Weight Prefetcher, controller-visible status,
  and ZCU104 experiment naming.

### HANDOVER/HGTXR

Source root:

- `/mnt/d/dataset/EV_Eye/paper_works/HANDOVER/HGTXR/docs/resources`

Imported to:

- `hardware/docs/resources/hgtxr_handover_additional/`

Reason:

- Preserve compact audit, validation, resource, PYNQ-smoke, VREF, XR-ViTS, and
  final-unblock evidence that was not already present in
  `hardware/docs/resources/hgtxr_final_evidence/`.

### IMPL_REPOS/HW/ViT_Accel

Source root:

- `/mnt/d/dataset/EV_Eye/paper_works/IMPL_REPOS/HW/ViT_Accel/docs`

Imported to:

- `hardware/docs/references/vit_accel/`

Reason:

- Keep multi-board runbooks, deployment docs, pipeline overview, and ViT
  accelerator reference notes without mixing ViT_Accel implementation code into
  the active HBTXR hardware source tree.

### IMPL_REPOS/HGPIPE/ICCAD24-HG-PIPE

Source root:

- `/mnt/d/dataset/EV_Eye/paper_works/IMPL_REPOS/HGPIPE/ICCAD24-HG-PIPE`

Imported to:

- `references/legacy-codebase/hardware/hgpipe_iccad24_docs/`

Reason:

- Preserve README-level context and HLS header contracts as legacy HG-PIPE
  reference material without importing the large generated case/build trees.

## Quantization Check

`IMPL_REPOS/HGPIPE/HG-PIPE-Quantization/hgpipe_quantization` was compared
against `quantization/src`. The only meaningful source difference found in this
pass was `eval/imagenet_eval.py`, where HBTXR intentionally uses the package
name `src` instead of `hgpipe_quantization`. The file was not overwritten.

## Excluded

- Vivado/Vitis generated projects, `.Xil`, `generated`, build instances, and
  synthesized RTL/report trees.
- Large binary and numeric tensors: `*.npz`, `*.f32bin`, bitstreams, tarballs.
- One 32 MB raw timing metrics CSV was removed from the imported handover
  resource set; compact reports and JSON/Markdown evidence were kept.
- Full XR_Accel/ViT_Accel automation code was not activated. It remains a
  future porting candidate after API and directory mapping.

## Follow-Up

- Decide whether XR_Accel configs should remain under `hardware/configs/xr_accel`
  or be normalized into the active config schema.
- Map XR_Accel architecture terms to the current HBTXR HLS module names before
  promoting any automation scripts.
- Keep large generated hardware evidence in external artifact storage or Git
  LFS if it needs to be preserved.
