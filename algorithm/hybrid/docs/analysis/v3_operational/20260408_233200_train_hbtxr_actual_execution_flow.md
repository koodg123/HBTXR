# train_hbtxr.py 기준 실제 실행 순서도

최종 갱신: 2026-04-08 KST

Role: `학습 파이프라인 분석가`, `실행 흐름 문서 작성자`

## 목적

이 문서는 `scripts/train_hbtxr.py`를 기준으로 실제 학습이 어떤 순서로 흘러가는지, 함수 단위까지 내려가며 설명한다.

분석 기준 파일:

- [scripts/train_hbtxr.py](../../scripts/train_hbtxr.py)
- [scripts/_config.py](../../scripts/_config.py)
- [src/hbtxr/training/trainer.py](../../src/hbtxr/training/trainer.py)
- [src/hbtxr/optim/registry.py](../../src/hbtxr/optim/registry.py)
- [src/hbtxr/models/export_pruned.py](../../src/hbtxr/models/export_pruned.py)

## 1. 한눈에 보는 실행 순서

```text
CLI
-> parse_args()
-> load_config()
-> apply_config_overrides()
-> resolve_resume_root()
-> resolve_run_contract()
-> resolve_training_entry()
-> write_run_artifacts()
-> train()
   -> make_loader()
   -> build_model()
   -> build_optimizer()
   -> _build_scheduler()
   -> _maybe_apply_pretrained()
   -> _create_teacher_model()
   -> epoch loop
      -> _epoch_loop()
         -> _forward_once()
            -> model(batch)
            -> compute_stage1_losses() or compute_stage2_losses()
            -> compute_metrics()
         -> backward / optimizer.step()
      -> validation _epoch_loop()
      -> checkpoint / history / early stop
   -> export_structural_student() optional
```

## 2. train_hbtxr.py 단계별 흐름

### 2.1 CLI 파싱

`main()`은 먼저 `parse_args()`를 호출해 아래 입력을 받는다.

- `--config`
- `--mode`
- `--stage`
- `--train-manifest`
- `--val-manifest`
- `--output`
- `--init-checkpoint`
- `--resume`
- `--experiment-name`
- `--device`
- `--override`

이 시점에는 아직 실제 training config가 완성되지 않았다.

### 2.2 YAML 로딩과 override 적용

다음 순서로 config가 materialize된다.

1. `load_config(args.config)`
2. `apply_config_overrides(...)`

여기서 일어나는 일:

- `extends` 체인 병합
- `data.mode`, `training.stage` override 반영
- explicit device, experiment name, manifest, checkpoint override 반영
- dotted override 반영

즉 “학습이 어떤 조건으로 돌아갈지”가 이 단계에서 확정된다.

### 2.3 run contract와 resume contract 해석

이후 아래 두 함수가 실행 run의 identity를 정한다.

- `resolve_resume_root(PROJECT_ROOT, experiment_name, args.resume)`
- `resolve_run_contract(...)`

의미:

- `--resume auto`면 최신 run root를 찾는다
- 새 실행이면 timestamped run name을 materialize한다
- `train_dir`, `eval_dir`, `infer_dir`, `vis_dir`, `hypers_dir`를 생성 규칙으로 계산한다

### 2.4 training entry 해석

`resolve_training_entry(...)`는 실제 training 입력을 확정한다.

확정되는 값:

- `stage`
- `mode`
- `canonical_name`
- `manifest_name`
- `experiment_name`
- `train_manifest`
- `val_manifest`
- `output_dir`
- `init_checkpoint`

그 다음 `_validate_path()`로 manifest/checkpoint 존재를 확인한다.

### 2.5 run artifact 기록

`write_run_artifacts(...)`는 `hypers/` 아래에 실행 계약을 저장한다.

대표 산출물:

- `resolved_config.yaml`
- `resolved_config.json`
- `cli_args.txt`
- `overrides.txt`
- `device.txt`
- `run_contract.json`

즉 학습 시작 전에 재현성 metadata를 먼저 남긴다.

## 3. optimizer_pool 분기

`training.optimizer_pool.enabled=true`인 경우에는 일반 학습 대신 candidate sweep으로 들어간다.

실행 순서:

1. `expand_optimizer_pool_candidates(cfg)`
2. candidate별 config clone
3. candidate별 run contract 작성
4. candidate별 `train(...)` 호출
5. `write_optimizer_pool_report(...)`

이 경로는 single run이 아니라 `runs/<exp>/pool/<candidate>/train` 구조를 만든다.

즉 `train_hbtxr.py`는 “단일 학습 엔트리”이면서 동시에 “optimizer benchmark 엔트리”도 겸한다.

## 4. 일반 학습 경로

optimizer pool이 아니면 마지막에 아래 호출 하나로 넘어간다.

```python
train(
    cfg=cfg,
    train_manifest=train_manifest,
    val_manifest=val_manifest,
    output_dir=run_contract["train_dir"],
    stage1_checkpoint=init_checkpoint,
    resume_checkpoint=resume_checkpoint,
)
```

실제 복잡도는 대부분 `trainer.train()` 안에 있다.

## 5. trainer.train() 실제 순서

### 5.1 초기 준비

가장 먼저 수행되는 일:

1. `normalize_compression_cfg(cfg)`
2. `set_seed(...)`
3. output dir 생성
4. history/log path 준비
5. `_ConsoleLogger` 초기화

### 5.2 loader 생성

다음으로 dataloader를 만든다.

1. `make_loader(train_manifest, cfg, shuffle=True)`
2. `make_loader(val_manifest, cfg, shuffle=False)`

이 단계에서 `data.mode`에 맞는 dataset class가 결정된다.

### 5.3 model / optimizer / scheduler 생성

다음 순서로 학습 core 객체를 만든다.

1. `build_model(cfg, role="student")`
2. `_resolve_device_and_wrap(model, training.device)`
3. `build_optimizer(model, cfg)`
4. `_build_scheduler(...)`
5. `_write_optimizer_reports(...)`
6. AMP scaler 생성

이 시점에 model topology, device wrapping, optimizer family, modifier, scheduler까지 모두 고정된다.

### 5.4 checkpoint / pretrained / teacher 경로

이후 checkpoint 관련 분기가 이어진다.

- `stage == "stage2"` and `stage1_checkpoint`
  - Stage1 checkpoint를 init weight로 불러옴
- `_maybe_apply_pretrained(...)`
  - external pretrained load
- `resume_checkpoint`
  - optimizer / scheduler / scaler / history / best_metrics 복구
- `_create_teacher_model(...)`
  - distillation teacher 생성

즉 학습 시작 직전 상태는 아래 세 경로가 합쳐진 결과다.

- stage warm start
- external pretrained
- interrupted-run resume

## 6. epoch loop 실제 흐름

### 6.1 상위 epoch loop

`for epoch in range(start_epoch, total_epochs + 1):`

각 epoch에서는 아래 순서가 실행된다.

1. optimizer training mode 설정
2. warmup lr 반영
3. `train_stats = _epoch_loop(..., phase="train")`
4. 필요 시 `val_stats = _epoch_loop(..., phase="val")`
5. scheduler step
6. history 저장
7. `last.pt` 저장
8. best checkpoint 저장
9. early stopping 판단

### 6.2 _epoch_loop() 내부

`_epoch_loop()`는 실제 batch iteration 엔진이다.

핵심 구조:

```text
for batch in loader:
  batch = move_to_device(batch, device)
  width_ratio = _resolve_pruning_width(...)
  losses, metrics, loss_total = _forward_once(batch, width_ratio)
  backward()
  optimizer.step()
  teacher ema update
  running stat accumulate
```

추가 분기:

- exact MARS optimizer multi-pass path
- sampled-parameter optimizer path
- AMP / non-AMP path
- grad accumulation path
- hessian refresh path

즉 `_epoch_loop()`는 일반 PyTorch training loop보다 optimizer family별 특수 경로가 더 많이 들어가 있다.

## 7. _forward_once() 실제 의미

이 함수가 batch 한 번의 핵심 계산을 담당한다.

순서:

1. 필요 시 teacher forward
2. `outputs = model(batch, width_ratio=width_ratio)`
3. stage에 따라
   - `compute_stage1_losses(...)`
   - `compute_stage2_losses(...)`
4. 필요 시
   - `compute_distillation_losses(...)`
   - `compute_regularization_ssl_losses(...)`
   - pruning regularizer
5. `loss_total` 합산
6. `metrics = compute_metrics(batch, outputs)`
7. `metric_active_width` 기록

즉 `_forward_once()`는 “forward + stage loss + optional auxiliary losses + metrics”의 묶음이다.

## 8. checkpoint 저장 규칙

항상 저장되는 것:

- `last.pt`

stage별 best checkpoint:

- Stage1
  - `best_search_p10.pt`
  - `best_search_p5.pt`
- Stage2
  - `best_track_p10.pt`
  - `best_track_p5.pt`

추가 규칙:

- `best_metric_name`에 해당하는 checkpoint는 `best.pt`로도 저장
- teacher 사용 시
  - `teacher_last.pt`
  - `teacher_best.pt`

## 9. export 경로

학습 종료 후 structural pruning export가 켜져 있으면 아래가 실행된다.

- `export_structural_student(...)`

산출물:

- `export/student_export.pt`
- `export/student_export_config.json`
- `export/student_export_report.json`

즉 export는 별도 script로도 가능하지만, pruning export가 켜져 있으면 training 종료 시 자동으로도 일어난다.

## 10. stage별 차이

### Stage1

- 일반적으로 `event=false`, `track=false`, `mask=true`
- ranking metric은 보통 `metric_search_p10_pct`
- warm start checkpoint 없이 시작하는 경우가 많음

### Stage2

- 일반적으로 `event=true`, `track=true`, `mask=false`
- ranking metric은 보통 `metric_track_p10_pct`
- Stage1 best checkpoint를 init checkpoint로 사용

## 11. 실무적으로 봐야 할 디버깅 포인트

문제가 생겼을 때 가장 먼저 볼 지점은 아래 순서가 효율적이다.

1. `hypers/run_contract.json`
2. `hypers/resolved_config.yaml`
3. manifest path 존재 여부
4. `check_dataloader.py` 결과
5. `trainer.py`의 `build_model()`과 `_epoch_loop()`
6. `history.jsonl`
7. 저장된 `best_*.pt`, `last.pt`

## 12. 요약

`train_hbtxr.py`는 얇은 launcher이고, 실제 학습의 중심은 `trainer.train()`이다.

정확히는 아래처럼 역할이 나뉜다.

- `train_hbtxr.py`
  - 실행 계약 확정
- `scripts/_config.py`
  - config, manifest, run identity 해석
- `trainer.train()`
  - 실제 학습 orchestration
- `_epoch_loop()`
  - batch iteration
- `_forward_once()`
  - model + loss + metrics 핵심 계산
