> **작성** 2026-07-14 · **갱신** 2026-07-23
> **상태** active
> **소유** repo

# Conversation Context

## 2026-07-14 Decisions

- User confirmed commit scope is only
  `D:\dataset\EV_Eye\paper_works\HBTXR-Pool\HBTXR`.
- User requested that commit work proceed and that all code under
  `HANDOVER` be analyzed in detail for concrete HBTXR applicability.
- User requested a plan before execution.
- The planning pass stopped before staging/commit/integration. Push was not
  included.
- User approved sequential execution of the plan. Local commits and analysis may
  proceed; push and active HANDOVER code promotion remain excluded.
- Sequential execution produced C0-C3 local commits and a C4 analysis package.
- No remote push, branch merge, active software/hardware port, board run, or
  external artifact import was authorized or performed.

## 2026-07-15 Decisions

- User requested a concrete plan for additional work after the HANDOVER
  integration analysis.
- Plan scope remains only the nested HBTXR repository; HANDOVER is read-only.
- Planning does not authorize dependency installation, active adaptation,
  experiment execution, FPGA tools, board access, artifact moves/uploads,
  Spec-Kit initialization, commits, or push.
- The canonical follow-up plan is
  `docs/aegis/plans/2026-07-15-hbtxr-follow-up-implementation.md`.
- User then narrowed execution to code/file integration only and explicitly
  excluded installation, reproducibility verification, and experiment work.
- After dependency/experiment scope review, active tasks are T-010, T-020,
  T-100, T-110, and T-210. T-120, T-220, and T-490 were removed; push remains
  excluded.
- The user requested resumption from the stopped point; execution resumed at T-110
  and continued sequentially through T-210 and final verification.
- The selected tasks were implemented as local semantic commits in HBTXR only;
  HANDOVER stayed read-only and existing XR reference JSON remained unchanged.
- Installation, reproducibility runs, experiments, synthesis, board access, merge,
  and push remained excluded.

## 2026-07-15 Algorithm And Reference Decisions

- The user confirmed that `algorithm/event`, `algorithm/frame`, and `algorithm/hybrid` are intentional modality surfaces and requested the plan be updated accordingly.
- The user wants `references/**` retained for comparison experiments.
- The revised plan preserves the current algorithm layout, applies default package cleanup only to Quantization, and treats Hybrid namespace migration as conditional with a default NO-OP.
- Reference deletion, rename, deduplication, and external-pointer replacement are excluded.

## 2026-07-16 Handover Decision

- The user requested a durable HANDOVER containing current progress, conversation decisions, and performed work, plus a prompt that a separate session can use to resume.
- The HANDOVER is documentation only; it does not expand authority to commit, push, install, experiment, delete, clean the worktree, or start an unselected implementation task.

## 2026-07-21 Algorithm Naming And Pools Decision

- The user requested removal of the `facet` name from the algorithm tree except reference-facing material, and requested removal of `pools`.
- The plan interprets the exclusion as preserving reference/archive/analysis/provenance/history and exact FACET reference paths, filenames, citations, and project identity.
- Active paths/contracts use existing semantic config keys or neutral CLI/API names; no `HBTXR_*` replacement environment namespace or permanent old-name alias is planned.
- `algorithm/hybrid/src/pools` is planned for complete removal. Unused facades are not recreated; LR scheduler moves to `src.optim.lr_schedulers`.
- `algorithm/hybrid/src/optim/pool.py` is explicitly retained because it implements optimizer candidate/report behavior rather than the removed generic facade.
- This decision revised planning documents only and did not authorize implementation, tests, deletion, commit, push, or cleanup.
- The user subsequently rejected the proposed `HBTXR_*` replacement names. This is scoped to replacement controls; it does not authorize a global rename of existing HBTXR project/model/config/reference identity.

## 2026-07-21 Algorithm Modular Ownership Decision

- The user requested a concrete plan based on the architecture analysis.
- `analysis`, `archive`, `artifacts`, `requirements`, `README.md`, and
  `pyproject.toml` must remain in place.
- The active target roots are `common`, `dataset`, `utils`, `engine`, `docs`,
  `configs`, `tests`, `frame`, `event`, and `hybrid`.
- Hybrid retains its special fusion/search-track behavior. Shared extraction is
  conditional on a real second consumer.
- `pools` is removed, but loss, optimizer, LR scheduler and runtime scheduler
  APIs are retained in their Hybrid domain packages; `optim/pool.py` remains.
- The user requested the target code tree and the meaning/content of each owner.
- This decision produced plans/docs only and did not authorize implementation.
- Independent T-727 review approved the concrete plan after its blocking
  findings were corrected; implementation remains gated on a separate selection.
