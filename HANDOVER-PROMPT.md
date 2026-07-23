# HBTXR Algorithm Refactor Resume Prompt

Read first: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/HANDOVER.md`

Authority plan: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md`

First action: 편집하기 전에 `/tmp/hbtxr-algorithm-modular-ownership-exec`가
존재하는지 확인하고 branch, HEAD, status, `git diff HEAD --stat`, 계획 SHA-256,
preserved-zone drift를 `HANDOVER.md`의 명령으로 검증하라. 이 `/tmp` worktree가
없으면 case-only `EvEye`/`eveye` 상태를 재구성하지 말고 즉시 중단하여 사용자에게
보고하라.

현재 권위 상태는 AM-000/010 commit 완료, AM-020/030 승인·미커밋 완료이며
다음 작업은 AM-040 common/engine/event owner 분리다. `20-checkpoint.md`의 오래된
“Next AM-030” 한 줄은 무시한다.

주 작업 트리의 의도적인 planning/recon dirty 상태를 reset, clean, stash, stage,
commit하지 말라. Linux `/tmp`를 강제하고 AM-060에서 `EvEye`가 제거될 때까지
dual-case migration state를 commit하지 말라. push/merge/tag/PR/cleanup은 금지다.
