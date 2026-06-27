# Conversation History

## 2026-03-25

### Request 1

- Build a new project in `FECET-HBTXR` using FACET as the base reference.
- Reuse the proposed model structure and scheduler from `HBTXR_v3_0`.
- Analyze the FACET code/paper and simplify the final repository structure.

### Accepted decision

- Keep FACET-style event representation and ellipse supervision.
- Replace the original FACET project complexity with a flatter package layout centered on `dataset.py`, `model.py`, `trainer.py`, and `scheduler.py`.
- Use the `HBTXR_v3_0` hybrid tracker, 2-stage training flow, and host-side Search/Track FSM as the main runtime contract.

### Request 2

- Use raw EV-Eye data from `E:\WSL\Shared\dataset\Eye\EV_Eye\raw_data\Data_davis`.
- Use Grounded-SAM from `E:\WSL\Shared\ETRI_SYNC\HBTXR\annotation_tools\Grounded-Segment-Anything-main`.
- Do not train a UNet for annotation.
- Generate FACET-style dataset preparation scripts.
- Do not damage either input path.

### Accepted decision

- Implement Grounded-SAM based annotation support in `tools/prepare_facet_gsam_dataset.py`.
- Keep `csv_only`, `csv_then_gsam`, and `grounded_sam` modes.
- Reject output paths nested inside the raw dataset root or the Grounded-SAM root to preserve both inputs.

### Request 3

- Make the project runnable on Linux with `sh`.
- Create `requirements.txt` and `.gitignore`.
- Update the `README` with structure and detailed execution instructions.
- Add a `doc` directory to manage update history and conversation history.

### Accepted decision

- Add POSIX `sh` wrapper scripts under `scripts/`.
- Expand the README around Linux-first execution.
- Track implementation history and decision history in `doc/UPDATE_HISTORY.md` and `doc/CONVERSATION_HISTORY.md`.

### Request 4

- Add overlay preview code and an execution script.
- Randomly select data per user/session.
- Show selected data with the corresponding annotation drawn on the visualization plot.

### Accepted decision

- Add `tools/overlay_preview.py` and `scripts/overlay_preview.sh`.
- Generate one overlay contact sheet per session from canonical annotation rows.
- Render ROI box, pupil box, ellipse contour, mask overlay, and annotation metadata on the original frame.

### Request 5

- Update the implementation history and conversation summary documents.
- Review progress against the integration plan.
- Present the result as a checklist.

### Accepted decision

- Keep `UPDATE_HISTORY.md` for repository-level changes.
- Keep `CONVERSATION_HISTORY.md` for request/decision summaries.
- Add `PROGRESS_CHECKLIST.md` for plan-vs-progress review, including completed, partial, and pending items.

### Request 6

- Use Grounded-SAM annotations to support quantitative comparison against the original FACET baseline.
- Generate FACET training, evaluation, and inference scripts.
- Generate FECET-HBTXR training, evaluation, and inference scripts for the same comparison flow.

### Accepted decision

- Add a dataset bridge that converts canonical/manifests data into the FACET reference sample-cache format.
- Add FACET wrapper tools for train/eval/infer without modifying the reference repository itself.
- Add comparison-oriented FECET-HBTXR shell wrappers and a dedicated workflow document.
