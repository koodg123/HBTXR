# TODO

## Decision Gates

- Decide whether Spec-Kit scaffolding may be initialized after a clean checkpoint.
- Decide Git LFS versus external artifact storage if large artifacts are required.
- Approve any future push separately.
- Completed: approve and execute the T-010/T-020/T-100/T-110/T-210 code/file-only slice.
- Approve dependency installation before T-000 creates the test environment.

## Analysis Backlog

- [x] Classify the 10 changed/missing HANDOVER/HBTXR files.
- [x] Analyze HGTXR recovery, ensemble, support-adaptive, teacher, and metric flows.
- [x] Identify HGTXR regression/golden-vector adaptation priorities.
- [x] Analyze XR_Accel cyclic/ZCU104 automation and config schema.
- [x] Implement data and evaluation contract gates as T-100 and T-110.
- [ ] Implement checkpoint compatibility and blending as T-120 after environment approval.
- [ ] Adapt HGTXR Stage-1/Stage-2 regression fixtures behind current interfaces.
- [x] Implement XR cyclic configuration validation as T-210.
- [ ] Implement the T-220 XR automation dry-run planner after a new selection.
- [ ] Map licenses for ERVT, TENNs-Eye, TDTracker, BRAT, FECET, and Retina sources.
- [ ] Audit CRLF/tooling governance and record decisions as T-600.
- [x] Write the concrete follow-up implementation plan with Task Cards and gates.
- Scope Fence: HBTXR listed paths; HANDOVER read-only; no external mutation.
- Baseline Lock: verify branch, HEAD, tools, licenses, and inputs before each wave.
- Approved Behavior: the selected five-task slice is complete; every deferred task needs a new selection and its applicable gate.
- Compatibility: current defaults, output ABI, checkpoint load, HLS top, and XR references stay stable.
- Retirement: remove conditional adapters or never add them when characterization proves them unnecessary.
- Execution: sequential waves with implementer, specification review, and quality review.
- Drift: stop on owner, source, license, tool, split, or board drift and revise the plan.
- Evidence: preserve commands, staged manifests, reviews, commit SHAs, and uncovered risks.
- Advisory boundary: branch-local completion is not merge, release, or push authority.
