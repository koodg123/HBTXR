# Training, Evaluation, and Metrics

This repository is primarily a hardware generation and validation project, not a model-training repository.

## 1. What This Repository Does Not Train

`ViT_Accel` does not include a DeiT training loop in the usual deep-learning sense.

There is no native pipeline here for:

- dataset ingestion for ImageNet training
- PyTorch or TensorFlow optimizer loops
- checkpoint training and fine-tuning
- top-1 / top-5 accuracy sweeps

Instead, the repository assumes that the model structure, quantization widths, and reference data have already been chosen, then focuses on generating and validating the accelerator implementation.

## 2. What Serves as the Model Definition

The model assumptions come from configuration and checked-in statistics.

Important inputs:

- `configs/models/deit_tiny_baseline.json`
- `configs/designs/full_deit_tiny_fit.json`
- `workspace/hardware/statistics/type.npy`
- `workspace/hardware/case/refs` or `refs.7z`

These files drive:

- layer count
- input size
- patch size
- embedding width
- attention/MLP quantization widths
- stream tiling
- reference vectors used by simulation and comparison flows

## 3. What “Evaluation” Means In This Repository

Evaluation here is hardware-flow validation rather than ML accuracy benchmarking.

The main evaluation layers are:

- HLS functional validation
- HLS timing and utilization analysis
- Spinal block latency measurement
- Vivado batch success or failure
- board preparation and bitstream-generation readiness

## 4. HLS Functional Metrics

The first metric family comes from Step1/Step2.

Important checks:

- `CSIM` completion
- `COSIM_PASS`

The current checked report snapshot shows:

- instances: `26`
- `COSIM_PASS`: `26/26`

Source:

- `workspace/artifacts/reports/<target>/deit_tiny_baseline/full_deit_tiny_fit/step2_hls_resource_summary.md`

## 5. HLS Timing and Utilization Metrics

Step2 aggregates the main synthesis metrics per block.

Tracked metrics:

- target clock period
- estimated clock period
- slack
- cycle count
- initiation interval
- LUT
- FF
- DSP
- BRAM18K
- URAM

From the current report snapshot:

- worst timing: `PATCH_EMBED` at `3.397 ns`
- max LUT: `ATTN0` with `78460`
- max FF: `ATTN0` with `40907`
- max DSP: `PATCH_EMBED` with `428`
- max BRAM18K: `ATTN0` with `123`
- max URAM: `PATCH_EMBED` with `18`

These are currently the most actionable optimization metrics in the repository.

## 6. Spinal Runtime Metrics

The Step3 flow is used to measure and validate the block-level runtime path.

Step3 provides:

- block staging into the Spinal runtime
- per-block simulation
- block latency reporting

This gives a functional integration view across the sequence:

- `PATCH_EMBED`
- `ATTN*`
- `MLP*`
- `HEAD`

## 7. Vivado Metrics

The Vivado stages are evaluated more coarsely.

Typical evaluation signals:

- batch run success or failure
- validation success
- synthesis completion
- implementation completion
- bitstream generation completion

Today, the repository is strongest up to:

- `step4-export`
- `step4-prepare`

The remaining hard boundary is final implementation completion for the heavy board flows.

## 8. Board-Level Runtime Metrics

Board runtime details live in target configuration files.

Examples from `configs/targets/vck190.json`:

- DMA base address
- GPIO base address
- clock base address
- accelerator base address
- LPDDR base addresses

These are deployment-oriented metrics and interfaces rather than performance measurements by themselves.

## 9. How To Interpret “Training and Evaluation” For This Project

The safest way to read the project scope is:

- training: external or pre-resolved
- model structure: captured in configs and statistics
- evaluation: hardware-centric

That means the meaningful metrics to monitor in this repository are:

- `COSIM_PASS`
- timing estimate and slack
- cycles and II
- LUT / FF / DSP / BRAM / URAM
- block latency
- Vivado implementation success
- board bitstream success

## 10. Current Metric Gap

If end-to-end ML accuracy needs to be documented later, it will have to be linked from an upstream training repository or external experiment record.

This repository currently does not produce:

- classification accuracy
- loss curves
- confusion matrices
- calibration statistics

Those are outside the present codebase boundary.
