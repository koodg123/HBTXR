# FACET Phase 3 세션 모니터링 보고서

- 점검 시각: 2026-06-25 17:39:18 KST 기준
- 대상 실행: FACET Phase 3 `DeanDataset_full_unet` 확장 생성
- FACET repo: `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET`
- 출력 경로: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet`
- 점검 방식: 실행 프로세스, GPU compute process, 출력 파일, manifest, 디스크 여유 공간을 읽기 전용으로 확인했다. 실행 중인 프로세스에는 종료/중단/시그널 전송을 하지 않았다.

## 확인 명령과 증거

| 항목 | 명령 | 결과 요약 |
|---|---|---|
| 프로세스 존재 | `pgrep -af "build_full_dean_dataset_with_unet.py|DeanDataset_full_unet|FACET_DISABLE_CUDNN"` | PID `761772`가 예상 스크립트 `EvEye/utils/scripts/build_full_dean_dataset_with_unet.py`를 실행 중이다. 인자도 `--output-root .../DeanDataset_full_unet`, `--device cuda:0`, `--inference-batch-size 32`, `--train-ratio 0.8`, `--sample-count 10`, `--overwrite`와 일치한다. |
| GPU compute process | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` | PID `761772`, 프로세스 `.facet-train-venv/bin/python`, GPU 메모리 `5850 MiB` 사용 확인. |
| GPU 전체 상태 | `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader` | 한 차례 `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`가 발생했다. 다만 compute-apps 쿼리는 전후로 성공했으므로 GPU 프로세스 존재 증거는 확보했다. |
| 출력 용량 | `du -sh .../DeanDataset_full_unet` | 17:38 전후 `6.0G`, 이후 `6.7G`, 최종 재확인 시 `7.5G`로 증가했다. |
| 파일 수 | `find .../DeanDataset_full_unet -type f | wc -l` | 최종 재확인 시 파일 `70`개. |
| 샘플 파일 | `find .../DeanDataset_full_unet -maxdepth 3 -type f | sort | head -80` | `samples/` 아래 `sample_000`부터 `sample_009`까지 각 `frame.png`, `mask.png`, `overlay.png`, `label.txt`가 존재한다. 샘플 파일 수는 `40`개다. |
| 최신 생성 파일 | `find ... -type f -printf ... | sort | tail -20` | `train/cached_data/events_batch_9.memmap`가 2026-06-25 17:39:56에 생성되었고, 대응하는 `events_batch_info_9.txt`, `events_indices_9.npy`도 생성되었다. |
| manifest | `find ... \( -iname '*manifest*' -o -iname '*.json' -o -iname '*.jsonl' -o -iname '*.csv' \) -type f ...` | manifest/json/jsonl/csv 파일은 아직 발견되지 않았다. 실행 중간 단계에서는 완료 후 메타데이터가 마지막에 기록될 수 있으므로, 현 시점에서는 단독 이상 징후로 보지 않는다. |
| 디스크 여유 공간 | `df -h .../DeanDataset_full_unet /tmp /home/kjm26/project/PRJXR/HBTXR` | `/dev/nvme0n1p2` 기준 전체 `3.6T`, 사용 `2.7T`, 가용 `772G`, 사용률 `78%`. 즉시 공간 부족 징후는 없다. |

## 진행 상황 추론

- 실행 프로세스 PID `761772`와 GPU compute process가 확인되어, FACET Phase 3 데이터셋 확장 작업은 모니터링 시점에 계속 실행 중인 것으로 판단된다.
- 출력 크기가 `6.0G -> 6.7G -> 7.5G`로 증가했고, 최신 파일이 `events_batch_9.*`까지 생성되었으므로 파일 생성은 정체되지 않았다.
- 현재 생성물은 주로 `train/cached_data`에 집중되어 있다. `val/cached_data`에는 아직 파일이 보이지 않았고, manifest도 아직 없다.
- `samples/`에는 10개 샘플 세트가 정상적으로 생성되어 있어, UNet 기반 마스크/오버레이 샘플 출력 단계는 최소 10개 샘플에 대해 통과한 것으로 보인다.

## 이상 징후 평가

- 치명적 이상 징후: 발견되지 않음.
- 진행 정체 징후: 발견되지 않음. 최신 파일 생성 시각이 점검 시각과 가깝고 출력 용량이 증가 중이다.
- 리소스 위험: 디스크 가용 공간 `772G`로 현재 출력 증가율 기준 즉시 위험은 낮다. GPU 메모리 사용량 `5850 MiB`는 확인되었으나, GPU 전체 utilization/temperature 쿼리는 한 차례 실패해 해당 지표는 제한적으로만 확인 가능하다.
- 완료 여부: 완료로 판단할 수 없다. manifest/json/csv 계열 메타데이터가 아직 없고 `val/` 캐시 산출물도 아직 비어 있다.

## 리스크

- manifest가 아직 없으므로, 지금 중단되거나 비정상 종료되면 데이터셋 완결성 검증이 어렵다.
- `nvidia-smi` 전체 GPU 상태 쿼리가 한 차례 실패했으므로, GPU 온도/전체 사용률은 이번 점검에서 확정하지 못했다. compute-apps 쿼리로 프로세스와 메모리 사용은 확인했다.
- `ps -p 761772`와 `/proc/761772` 조회가 짧은 시점에서 빈 결과를 보인 적이 있으나, 이후 `pgrep`과 `nvidia-smi --query-compute-apps`에서 같은 PID가 재확인되었다. 샌드박스/프로세스 조회 타이밍 차이 가능성이 있어, 단독 이상으로 보지 않는다.

## 권장 다음 모니터링

1. 5-10분 후 `pgrep -af "build_full_dean_dataset_with_unet.py|DeanDataset_full_unet"`와 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`를 재확인한다.
2. `du -sh /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet`와 최신 파일 `find ... -printf ... | sort | tail`을 비교해 용량과 mtime이 계속 증가하는지 본다.
3. `val/cached_data`에 파일이 생성되기 시작하는지 확인한다. 현재는 `train/cached_data` 중심 진행으로 보인다.
4. 실행 종료 후 manifest/json/csv 계열 파일 생성 여부를 확인하고, 없으면 생성 스크립트의 완료 로그 또는 예외 출력을 별도로 확인해야 한다.
5. 디스크 사용률이 85-90%에 접근하면 출력 크기 증가 속도를 기준으로 중단 없는 정리 계획을 세운다. 현재 78%로 즉시 조치는 필요하지 않다.
