# TimeLens-XL Fine-Tuning Workflow

이 문서는 `Third/FI`에 통합된 `TimeLens-XL` repo root가 존재한다고 가정하고, 현재 `HBTXR_v3_0` 작업 공간에서 `TimeLens-XL` fine-tuning을 수행하는 실험 흐름을 정리한다.

## 1. 구현 범위

현재 프로젝트에 추가된 `TimeLens-XL` fine-tuning 표면은 아래 두 단계로 나뉜다.

1. HBTXR target-FPS surface를 `TimeLens-XL`이 읽을 수 있는 dataset surface로 export
2. native `Third/FI/run_network.py` trainer를 HBTXR launcher로 실행

외부 repo 코드는 `src/` 안으로 넣지 않는다. `TimeLens-XL`은 `Third/FI`에 upstream repo root 형태로 통합하고, HBTXR는 export/launcher/docs를 제공한다.

## 2. 추가된 실행 표면

실험용 script:

- `exps/scripts/export_timelens_xl_finetune_dataset.py`
- `exps/scripts/finetune_timelens_xl.py`
- `exps/scripts/run_export_timelens_xl_finetune_dataset.sh`
- `exps/scripts/run_finetune_timelens_xl.sh`

공통 구현 모듈:

- `src/hbtxr/preprocess/timelens_xl_finetune.py`

## 3. 입력 조건

필수 조건:

- `Third/FI` integrated root 존재
- `workspace/target_data/fps_<tag>/...` target-FPS dataset이 이미 materialized 상태
- target-FPS session store가 `frame_storage_mode=materialized_target_frames`

현재 export는 `lazy_source_frames` 세션 store는 지원하지 않는다.

## 4. dataset export contract

export 결과는 아래처럼 `TimeLens-XL`용 flat sequence surface를 만든다.

```text
workspace/timelens_xl_finetune_dataset/
  train/
    user01_left_session_101/
      images/
        000000.png
        000001.png
        ...
      events/
        000000.npz
        000001.npz
        ...
      sequence_meta.json
  val/
    ...
  export_summary.json
```

핵심 규칙:

- `images/*.png`
  - grayscale target frame을 3채널 RGB PNG로 저장
- `events/*.npz`
  - key: `data`
  - shape: `[H, W]`
  - 의미: 두 target frame 사이 event packet의 signed accumulation map

이 contract는 `TimeLens-XL`의 `loader_timelens` dataloader에 맞춘다.

## 5. export 단계

예시:

```bash
sh exps/scripts/run_export_timelens_xl_finetune_dataset.sh \
  --target-fps 2000 \
  --overwrite
```

주요 옵션:

- `--target-root`
  - 기본값: `<project_root>/workspace/target_data`
- `--target-fps`
  - 필수
- `--output-root`
  - 기본값: `<project_root>/workspace/timelens_xl_finetune_dataset`
- `--split-map`
  - JSON 파일로 train/val session_key를 직접 지정
- `--val-every-nth-sequence`
  - split map이 없을 때 deterministic val 분리
  - `0`이면 모두 train
- `--overwrite`
  - 기존 export surface 삭제 후 재생성

Python 직접 실행 예시:

```bash
PYTHONPATH=src .venv/bin/python exps/scripts/export_timelens_xl_finetune_dataset.py \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --target-fps 2000 \
  --output-root workspace/timelens_xl_finetune_dataset \
  --overwrite
```

## 6. native fine-tuning 단계

예시:

```bash
sh exps/scripts/run_finetune_timelens_xl.sh \
  --dataset-root workspace/timelens_xl_finetune_dataset \
  --model-pretrained /abs/path/to/timelens_xl_checkpoint.pth \
  --max-epoch 27 \
  --batch-size 2
```

이 launcher는 내부적으로:

1. `Third/FI`를 import path에 추가
2. HBTXR dataset root를 읽어 native `PARAM_REGISTRY`에 HBTXR 전용 param을 동적 등록
3. `run_network.py`를 기존 CLI contract 그대로 호출

## 7. fine-tuning 주요 옵션

- `--dataset-root`
  - export된 dataset root
- `--output-root`
  - run output root
  - 기본값: `workspace/timelens_xl_finetune_runs`
- `--param-name`
  - native registry에 등록할 HBTXR param 이름
- `--model-name`
  - 기본값: `TimeLens`
- `--dataloader`
  - 기본값: `loader_timelens`
- `--model-pretrained`
  - pretrained/native checkpoint
  - 명시하지 않으면 `workspace/third_party_checkpoints/FI/TimeLens-XL/` 아래의 최신 checkpoint를 먼저 찾고, 없으면 upstream 기본 상대 경로를 본다
- `--crop-size`
  - `0`이면 crop 비활성화
- `--batch-size`
- `--num-workers`
- `--interp-ratio`
  - 현재 기본은 `2`
- `--rgb-sampling-ratio`
  - 기본 `1`
- `--random-t` / `--no-random-t`
- `--lr`
- `--milestones`
  - 예: `12,24`
- `--gamma`
- `--max-epoch`
- `--training-stage`
  - 기본값: `tuning`
- `--skip-training`
  - validation-only 실행
- `--clear-previous`
- `--extension`
- `--dry-run`
  - native launch 전에 resolved payload와 argv만 출력

`--dry-run` 예시:

```bash
PYTHONPATH=src .venv/bin/python exps/scripts/finetune_timelens_xl.py \
  --paths-config configs/paths/ev_eye_groundedsam_paths.json \
  --dataset-root workspace/timelens_xl_finetune_dataset \
  --model-pretrained /abs/path/to/checkpoint.pth \
  --dry-run
```

## 8. 권장 실행 순서

1. `build_target_fps_dataset.py`로 target-FPS surface materialize
2. `export_timelens_xl_finetune_dataset.py`로 `images/events` surface export
3. `finetune_timelens_xl.py --dry-run`으로 native argv 확인
4. `finetune_timelens_xl.py` 실제 실행
5. 생성된 checkpoint를 HBTXR mode2 interpolation backend에 다시 연결

## 9. 현재 제약

- export는 현재 `materialized_target_frames` 세션 store만 지원
- event packet은 signed accumulation map으로 export한다
- first implementation은 `TimeLens-XL`의 `loader_timelens` 경로를 기준으로 맞춰져 있다
- `TimeLens` 본체 fine-tuning은 아직 별도 구현 대상이다

## 10. 검증 포인트

최소 검증:

- export 결과에 `images = N`, `events = N-1` 규칙이 맞는지
- `events/*.npz`가 `data` key를 갖는지
- `finetune_timelens_xl.py --help`와 `--dry-run`이 정상 동작하는지
- native run 전에 `Third/FI` dependency가 실제 환경에 설치돼 있는지
