# Documentation Policy

This directory stores lightweight project history for `SWIFT-HBTXR`.

## Files

- `UPDATE_HISTORY.md`: dated implementation history and repository changes
- `CONVERSATION_HISTORY.md`: concise decision log distilled from the user-assistant thread
- `PROGRESS_CHECKLIST.md`: plan-vs-progress status, validation scope, and remaining work
- Git publication status is recorded in `UPDATE_HISTORY.md` and `PROGRESS_CHECKLIST.md`
- End-to-end smoke analysis artifacts live under `../runs/smoke_full_20260327/` and are referenced from the dated history instead of copied here
- `../runs/interpolate/prepared_summary_real.json`: latest real-session `timelens` input-preparation result outside this directory

## Update Rule

- Append new dated entries instead of rewriting older decisions.
- Keep the conversation log concise. Record requests, constraints, decisions, and outcomes rather than full transcripts.
- Keep the progress checklist explicit about what is implemented, what is test-validated only, and what is still pending on the real dataset.
- When the repository is published or re-published, record the branch, remote, and commit identifier in `UPDATE_HISTORY.md`.
- Do not store raw dataset contents, generated manifests, checkpoints, or temporary experiment outputs in this directory.
