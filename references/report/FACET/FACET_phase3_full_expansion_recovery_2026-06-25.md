# FACET Phase 3 Full Expansion Recovery

Date: 2026-06-25

## Summary

Phase 3의 `DeanDataset_full_unet` 생성 중 첫 장기 실행이 `exit code 137`로 종료되었다. 디스크 부족은 아니었고, 당시 swap이 가득 찬 상태였기 때문에 장시간 실행 중 메모리 피크 또는 외부 kill 가능성이 높다.

이에 `build_full_dean_dataset_with_unet.py`를 session 단위 재개 가능 구조로 수정했다. 현재 full expansion은 `tmux` 세션 `facet_full_expansion`에서 백그라운드로 계속 실행 중이다.

## U-Net Accuracy Snapshot

현재 full expansion에 사용하는 U-Net checkpoint:

`/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/RGBUNet_local_subset/version_1/checkpoints/epoch=02-val_mean_distance=0.4997.ckpt`

라벨이 있는 `Data_davis_labelled_with_mask` 기반 local subset validation에서 확인된 best checkpoint 기준:

- `val_mean_distance`: `0.4997 px`
- `val_IoU`: `0.921`
- `val_p10_acc`: `1.000`
- `val_p5_acc`: `1.000`
- `val_p3_acc`: `1.000`
- `val_p1_acc`: `0.957`

주의: 위 수치는 labelled subset validation 기준이다. 전체 `Data_davis`에는 GT mask/ellipse가 없으므로, full expansion 품질은 현재 U-Net 추론 성공률, skip count, sample overlay 육안 확인, 이후 EPNet metric으로 간접 검증해야 한다.

## Failure And Recovery

첫 full expansion 장기 실행:

- 진행: 약 `240/384` session까지 tqdm 출력 확인
- 종료: `exit code 137`
- GPU 프로세스: 종료 후 남지 않음
- 디스크: 여유 있음
- partial output: manifest 없음
- 원인 판단: 디스크 부족 아님. 메모리 피크, swap pressure, 또는 외부 kill 가능성.

기존 스크립트 한계:

- `manifest.json`은 전체 완료 후에만 생성됨
- ellipse memmap도 `writer.close()` 시점에 최종 생성됨
- 장기 실행 중 죽으면 partial output을 최종 dataset으로 사용할 수 없음

수정 내용:

- `--resume` 옵션 추가
- `progress_state.json` session 단위 저장
- split writer가 session마다 event batch를 flush
- `ellipse_records.npy`를 중간 저장하여 resume 시 ellipse record 복구
- GPU cache 정리와 `gc.collect()` 추가
- 완료 전에도 현재까지의 `completed_session_count`, writer count, totals 확인 가능

## Current Background Run

실행 스크립트:

`/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/run_full_expansion_resume_2026-06-25.sh`

tmux 세션:

`facet_full_expansion`

현재 확인 시점:

- `progress_state.json`: 전체 기준 `24/384` session 완료
- train valid samples: `91830`
- skipped frames: `2134`
- output size: `14G`
- GPU process: `/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python`
- GPU memory: 약 `2916 MiB`

상태 확인 명령:

```bash
tmux capture-pane -pt facet_full_expansion -S -10
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
python3 - <<'PY'
import json
from pathlib import Path
p = Path('/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet/progress_state.json')
d = json.loads(p.read_text())
print(d['completed_session_count'])
print(d['writer_counts'])
print(d['totals'])
print(d['updated_at_utc'])
PY
```

## Resume Command

tmux 세션이 죽었지만 `progress_state.json`이 남아 있으면 다음 명령으로 이어갈 수 있다.

```bash
tmux new-session -d -s facet_full_expansion \
  /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/run_full_expansion_resume_2026-06-25.sh
```

## Next Gates

1. `DeanDataset_full_unet/manifest.json` 생성 확인.
2. `num_train`, `num_val`, `valid_ellipse_count`, `skip_no_ellipse_count`, `skip_no_events_count` 확인.
3. `DavisEyeEllipseDataset` train/val loader smoke 검증.
4. `FACET_reproduction_status_2026-06-25.{md,json}` 재생성.
5. full EPNet training으로 Phase 4 진행.

## Progress Update

Update time: 2026-06-26 00:04 KST

Current state:

- tmux session `facet_full_expansion` is alive.
- GPU process exists: `/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python`.
- `progress_state.json` exists.
- Completed sessions: `166/384`.
- Train valid samples: `629707`.
- Skipped frames: `14705`.
- Skip no ellipse: `14694`.
- Skip no events: `11`.
- Output size: `94G`.
- Disk free under raw data mount: `683G`.
- `manifest.json`: not yet generated.

Interpretation:

- The run is slow but still advancing.
- Because `progress_state.json` is updated session-by-session, a future interruption can be resumed with `run_full_expansion_resume_2026-06-25.sh`.
- Do not start Phase 4 full EPNet training until `manifest.json` exists and train/val loader smoke passes.

## Completion Update

Update time: 2026-06-26 07:51 KST

Current state:

- `DeanDataset_full_unet/manifest.json` exists.
- `progress_state.json` reports `384/384` completed sessions.
- Train valid samples: `1165260`.
- Val valid samples: `292560`.
- Total frames scanned: `1490548`.
- Valid ellipse count: `1457820`.
- Skipped frames: `32728`.
- Skip no ellipse: `32681`.
- Skip no events: `47`.

Loader issue found after completion:

- The resumable full-expansion writer flushes event batches at session boundaries.
- As a result, `events_indices_*.npy` files are variable-length.
- `load_event_segment()` previously assumed fixed 5000-sample batches and failed during full training with `IndexError`.

Resolution:

- `EvEye/utils/cache/MemmapCacheStructedEvents.py` now resolves event batches by cumulative lengths from the actual `events_indices_*.npy` files.
- Train/val smoke checks passed for first, boundary, and last indices.
- Phase 4 EPNet and HBTXR training were launched after this fix.
