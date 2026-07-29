# Evidence Bundle Draft

## Baseline

- Branch: `refactor/hbtxr-structure`
- HEAD: `087912c2b27f63b13600c20af6f5180c01c7aeb7`
- Before planning files: 2 modified, 545 untracked, 0 staged.
- Import audit: 105 JSON, 81 Python, and 26 shell files passed static checks.

## Pending Evidence

- C0-C4 commit SHAs and staged manifests.
- Analysis artifact completeness checks.
- Final clean worktree and drift decision.

## C0

- Commit: `9535d24 docs(plan): Define HBTXR handover analysis workflow`.
- Scope: 17 approved planning/checkpoint files; 897 insertions.
- Gates: staged allowlist PASS, diff check PASS, spec review PASS, quality PASS.

## C1

- Commit: `918d0fd chore(algorithm): Archive FECET and SWIFT porting sources`.
- Scope: 120 inactive archive files; 14,157 insertions.
- Gates: Python 71/71 PASS, shell 26/26 PASS, staged allowlist PASS,
  prohibited-artifact scan PASS, spec/quality reviews PASS.
- Waiver: 25 inherited EOF blank lines, restricted to this provenance archive.

## C2

- Commit: `0901a96 chore(analysis): Import EV-Eye and EX-Gaze handover artifacts`.
- Scope: 223 files; 21,071 insertions; exact five-path allowlist passed.
- Gates: Python 10/10 PASS, JSON 17/17 PASS, CSV headers 187/187 PASS,
  maximum size/prohibited-artifact checks PASS, spec/quality reviews PASS.
- Waiver: 16,743 inherited CRLF findings restricted to imported CSV evidence;
  zero Python, JSON, or Markdown diff-check findings.

## C3

- Commit: `6b880d5 chore(hardware): Import accelerator handover evidence`.
- Scope: 204 files; 50,306 insertions and 1 deletion; exact allowlist passed.
- Gates: JSON 88/88 PASS, Markdown UTF-8 98/98 PASS, active-source isolation,
  size/binary/mode/prohibited-artifact/secret checks PASS, reviews PASS.
- Waiver: exactly 60 inherited formatting findings restricted to enumerated
  ICCAD24, ViT, HGTXR, and XR reference/evidence files.

## C4 Draft

- Six required analysis documents created under `docs/analysis/`.
- Inventory records six repository revisions and byte/path overlap evidence.
- Software and hardware audits define adapter/test gates without active promotion.
- License/artifact audit blocks unclear sources and externalizes large/generated data.
- Matrix and recommendations provide priorities, dependencies, stop conditions,
  G0-G10 validation gates, and a bounded next implementation proposal.
- Final spec review: PASS after exact-path and S8 execution-contract remediation.
- Final documentation quality review: PASS; runtime/board behavior remains unverified.
