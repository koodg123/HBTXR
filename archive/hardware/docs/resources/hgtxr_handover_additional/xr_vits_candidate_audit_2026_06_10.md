# XR-VITs Candidate Audit

- status: `candidate-found`
- requested_path: `/home/kjm26/project/PRJXR/XR-VITs`
- approved_replacement: `False`
- policy: Do not treat any candidate as XR-VITs until the user approves the replacement or the exact requested path is restored.

## Recommendation

- `candidate-xr-accel` score=`99` path=`/home/kjm26/project/PRJXR/XR-VIT/XR_Accel`
- reason: highest score for ZCU104/cyclic/DeiT/HLS evidence

## Candidates

### candidate-xr-accel
- path: `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel`
- exists: `True`
- score: `99`
- files: `1315`
- hls_files: `97`
- zcu104_configs: `12`
- cyclic_deit_configs: `8`
- primitives: `layernorm:4, gelu:8, softmax:8, quant:2`
- zcu104 samples:
  - `configs/designs/deit_tiny_cyclic_zcu104.json`
  - `configs/designs/hbtxr_cyclic_zcu104.json`
  - `configs/designs/hbtxr_streaming_zcu104.json`
  - `configs/experiments/zcu104_deit_tiny_baseline_cyclic.json`
  - `configs/experiments/zcu104_full_deit_tiny_fit.json`

### candidate-vit-accel
- path: `/home/kjm26/project/PRJXR/XR-VIT/ViT_Accel`
- exists: `True`
- score: `90`
- files: `2916`
- hls_files: `229`
- zcu104_configs: `5`
- cyclic_deit_configs: `3`
- primitives: `layernorm:8, gelu:8, softmax:8, quant:8`
- zcu104 samples:
  - `configs/experiments/zcu104_deit_tiny_variant_fit.json`
  - `configs/experiments/zcu104_full_deit_tiny_fit.json`
  - `configs/models/deit_tiny_variant_zcu104.json`
  - `configs/targets/zcu104.json`
  - `workspace/hardware/vivado/boards/zcu104.tcl`

### candidate-analysis-xr-accel
- path: `/home/kjm26/project/PRJXR/XR-VIT/analysis/XR_Accel`
- exists: `True`
- score: `0`
- files: `6`
- hls_files: `0`
- zcu104_configs: `0`
- cyclic_deit_configs: `0`
- primitives: `layernorm:0, gelu:0, softmax:0, quant:0`
