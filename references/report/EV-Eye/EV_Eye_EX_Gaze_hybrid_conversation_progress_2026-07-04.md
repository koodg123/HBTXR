# EV-Eye and EX-Gaze Hybrid Conversation Progress

Date: 2026-07-04

## Scope

This note records the current conversation decisions and execution state for the
planned EV-Eye and EX-Gaze adapted hybrid experiments.

The target experiment condition is:

- EV-Eye on GPU0
- EX-Gaze on GPU1
- frame input: `1 x 128 x 128`
- event input: `2 x 64 x 64`
- subject-independent split aligned to HBTXR:
  - train subjects 1-32
  - validation subjects 33-36
  - test subjects 37-48
- evaluation package aligned to:
  - `/home/kjm26/project/PRJXR/HBTXR/analysis/RESULTS/HBTXR_subject37_48_error_distribution`

## Decisions So Far

### EX-Gaze Weight Availability

The EX-Gaze repository includes model weights:

- `references/codebase/software/EX-Gaze/model_weight/frame_based_model.pth`
- `references/codebase/software/EX-Gaze/model_weight/event_based_model.pth`

The frame-based model is used for initialization/relocalization. The event-based
model is a sparse event-patch tracking model.

The EX-Gaze event tracker input contract is:

```text
B x 8 x 2 x 16 x 16
```

The original EX-Gaze patch extraction is not from the `1 x 160 x 256` frame
crop. It is from the full original EV-Eye/DAVIS event map:

```text
2 x 260 x 346 -> 8 x 2 x 16 x 16
```

For the HBTXR-Hybrid protocol, this must be adapted to:

```text
2 x 64 x 64 -> 8 x 2 x 16 x 16
```

with ellipse coordinates in the `64 x 64` event-grid coordinate system.

### EX-Gaze Paper Baseline Interpretation

The EX-Gaze paper states that competing methods were trained using the same
EV-Eye training set as EX-Gaze and implemented on Jetson Orin Nano 8GB for
runtime measurements.

Therefore, the EV-Eye and Swift-Eye numbers in the EX-Gaze paper should be
treated as EX-Gaze-reported re-trained or re-implemented baseline results, not
as direct copies from the original EV-Eye or Swift-Eye papers.

However, the local EX-Gaze repository does not include the competing baseline
training code, weights, or raw benchmark result files for EV-Eye, Swift-Eye,
3ET, and Retina.

### Hybrid Comparison Rule

Original paper numbers should not be directly compared to local HBTXR results
without a protocol caveat because the input resolutions, coordinate systems,
fusion strategy, and splits differ.

The fair local comparison should use adapted baselines:

- `EV-Eye_Hybrid_frame128_event64`
- `EX-Gaze_Hybrid_frame128_event64`

Both are adapted under the same HBTXR-Hybrid protocol.

### Derived Dataset Requirement

No new raw data collection is required.

New model-facing derived datasets are required because the HBTXR dataset,
EV-Eye, and EX-Gaze expect different tensor contracts.

Recommended derived dataset roots:

```text
/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent
/home/kjm26/project/dataset/XR/EV_Eye/target_data/EX_Gaze_Hybrid_frame128_event64_subject_independent
```

The source dataset remains:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent
```

### EV-Eye Adaptation

EV-Eye original local code is a frame U-Net segmentation benchmark. It is not a
direct event-frame center/ellipse tracker.

For this experiment, EV-Eye should be reported as:

```text
EV-Eye adapted under HBTXR-Hybrid frame128/event64 protocol
```

The adapted EV-Eye dataset needs:

- frame `1 x 128 x 128`
- pupil mask `1 x 128 x 128`
- event `2 x 64 x 64`
- ellipse label in common evaluation coordinates
- subject/session/frame metadata
- motion label and blink flag

If a real mask is unavailable, the mask can be rasterized from the ellipse label.

### EX-Gaze Adaptation

EX-Gaze should keep the sparse patch tracker contract, but the patch source
changes from full `2 x 260 x 346` event maps to local `2 x 64 x 64` event maps.

The adapted EX-Gaze dataset needs:

- frame `1 x 128 x 128`
- event map `2 x 64 x 64`
- previous state `pre_state = [x, y, a, b, angle]`
- event patches `8 x 2 x 16 x 16`
- sample regions for patch provenance
- current ellipse label
- subject/session/frame metadata
- motion label and blink flag

At test time, previous ground-truth ellipse should not be used at every step for
the main result, because that would be an oracle setting. Main evaluation should
use previous prediction after initialization.

## Documented Plan

The full shared plan is saved in both model report directories:

- `references/report/EV-Eye/EV_Eye_EX_Gaze_hybrid_frame128_event64_plan_2026-07-04.md`
- `references/report/EX-Gaze/EV_Eye_EX_Gaze_hybrid_frame128_event64_plan_2026-07-04.md`

## Current Status

Completed:

- EX-Gaze weights inspected.
- EX-Gaze event patch input contract identified.
- EX-Gaze paper split and baseline wording analyzed.
- Hybrid comparison fairness rule established.
- Derived dataset requirement established.
- Shared EV-Eye/EX-Gaze hybrid plan created.
- Plan documents moved from the FACET report directory into the EV-Eye and
  EX-Gaze report directories.

Pending:

- Create derived dataset generation scripts.
- Run dataset preflight against the source `DeanDataset_full_unet_subject_independent`.
- Generate or index frame `128 x 128` and event `2 x 64 x 64` tensors.
- Generate EV-Eye masks or ellipse-rasterized pseudo masks.
- Generate EX-Gaze `pre_state` and patch extraction metadata.
- Add one-batch smoke tests for both model adapters.
- Launch EV-Eye on GPU0 and EX-Gaze on GPU1 only after smoke tests pass.
- Produce HBTXR-style test packages under `analysis/RESULTS`.

## Next Execution Step

The next practical step is Phase 0/1 preflight:

1. Inspect the source dataset manifest and cached tensor layout.
2. Verify frame/event tensor availability and shapes.
3. Verify label coordinate system.
4. Write a reusable derived-dataset generator or adapter script.
5. Run a dry-run that emits only manifests and sample-count summaries before
   writing large tensor caches.

## Reporting Phrase

Use this wording in future reports:

```text
We compare against adapted EV-Eye and EX-Gaze hybrid baselines under a unified
HBTXR-Hybrid protocol using 128x128 frame inputs and 2x64x64 event inputs.
Original EX-Gaze and EV-Eye reported numbers are not directly comparable because
their input resolutions, fusion strategies, coordinate systems, and evaluation
protocols differ.
```

