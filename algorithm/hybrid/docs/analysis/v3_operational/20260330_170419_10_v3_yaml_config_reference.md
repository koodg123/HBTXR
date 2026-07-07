# HBTXR_v3_0 YAML 설정 참조

최종 갱신: 2026-03-28 KST

Role: `설정 아키텍트`, `데이터 파이프라인 분석가`, `학습 시스템 엔지니어`

## 요약

이 문서는 `HBTXR_v3_0`의 현재 YAML 설정 체계를 설명한다.

다루는 범위:

- `base.yaml`과 preset이 어떻게 합쳐지는지
- CLI override가 어떻게 동작하는지
- 어떤 top-level 섹션이 존재하는지
- 어떤 옵션이 코드에서 실제 소비되는지
- 어떤 값이 코드 기준으로 유효한 것으로 확인되었는지

주 코드 기준:

- [scripts/_config.py](../../scripts/_config.py)
- [train_hbtxr.py](../../scripts/train_hbtxr.py)
- [loader.py](../../src/hbtxr/data/loader.py)
- [dataset.py](../../src/hbtxr/data/dataset.py)
- [trainer.py](../../src/hbtxr/training/trainer.py)
- [pruning.py](../../src/hbtxr/models/pruning.py)
- [export_pruned.py](../../src/hbtxr/models/export_pruned.py)

## 1. 설정 시스템 개요

### 1.1 Base + preset 구조

설정 시스템은 아래 구조를 기본으로 한다.

- [base.yaml](../../configs/base.yaml)
- `extends: base.yaml`를 선언하는 stage / mode별 preset

대표 main preset:

- [mode1_stage1.yaml](../../configs/mode1_stage1.yaml)
- [mode1_stage2.yaml](../../configs/mode1_stage2.yaml)
- [mode2_stage1.yaml](../../configs/mode2_stage1.yaml)
- [mode2_stage2.yaml](../../configs/mode2_stage2.yaml)

대표 experimental preset:

- [stage1_train_a.yaml](../../exps/configs/stage1_train_a.yaml)
- [stage2_hybrid_v3.yaml](../../exps/configs/stage2_hybrid_v3.yaml)
- [mode0_stage1.yaml](../../exps/configs/mode0_stage1.yaml)
- [mode0_stage2.yaml](../../exps/configs/mode0_stage2.yaml)
- [mode2_stage2_lazy_2000fps.yaml](../../exps/configs/mode2_stage2_lazy_2000fps.yaml)

메모:

- 공식 실행 표면은 `configs/` 아래의 `base + mode1/2 stage1/2` preset이다.
- 실험 preset은 `exps/configs/`로 이동했다.
- 과거 `configs/...` experimental YAML 경로는 [scripts/_config.py](../../scripts/_config.py)가 한시적으로 alias로 받아준다.

### 1.2 Merge semantics

`scripts/_config.py`는 YAML을 recursive deep merge로 불러온다.

- parent config를 먼저 로드
- child config가 동일 key를 override
- nested dictionary는 통째 교체가 아니라 merge

예를 들어 아래 override는 `model` 전체를 다시 쓰지 않고 `model.heads.event`만 바꾼다.

```yaml
model:
  heads:
    event: false
```

### 1.3 CLI override

학습 entry script가 지원하는 대표 인자:

- `--config`
- `--mode mode1|mode2`
- `--stage stage1|stage2`
- `--device ...`
- `--experiment-name ...`
- 반복 가능한 `--override key=value`

예시:

```bash
python scripts/train_hbtxr.py \
  --config configs/mode2_stage2.yaml \
  --override training.epochs=5 \
  --override data.mode2.synthetic_event_builder.policy=interval_all
```

`--override`는 dotted key와 YAML parsing을 사용하므로:

- `true` -> boolean
- `[1, 2]` -> list
- `null` -> `None`

## 2. Top-level 설정 섹션

| Section | 의미 | 주요 소비 코드 |
|---|---|---|
| `seed` | random seed | `trainer.py` |
| `model` | backbone, heads, student, pretrained | `trainer.py`, `hybrid_tracker.py` |
| `data` | mode, canonical, manifest, event builder | `loader.py`, `dataset.py` |
| `training` | stage, epochs, optimizer, scheduler, device | `trainer.py` |
| `experiment` | manifest path, output root, init checkpoint | `_config.py`, script |
| `run` | materialized run naming metadata | `_config.py` |
| `distillation` | teacher-student KD / RKD | `losses.py`, `trainer.py` |
| `regularization_ssl` | stage-aware self-regularization | `losses.py` |
| `ssl` | legacy compatibility surface | `pruning.py`, report |
| `pruning` | structural student 및 export 설정 | `pruning.py`, `export_pruned.py` |
| `loss` | supervised loss weight | `losses.py` |
| `runtime` | inference FSM threshold | `runtime/tracker.py` |

## 3. `model` 섹션

### 3.1 핵심 구조

| Option | Type | 의미 |
|---|---|---|
| `embed_dim` | int | backbone embedding width |
| `depth` | int | transformer depth |
| `num_heads` | int | attention head count |
| `mlp_ratio` | float | transformer MLP expansion |
| `patch_size` | int | patch embedding patch size |
| `input_size` | `[H, W]` | 모델 기대 입력 크기 |
| `dropout` | float | dropout 비율 |
| `aux_classes` | int | auxiliary class 수 |

### 3.2 Head gating

| Option | Type | 의미 |
|---|---|---|
| `model.heads.eye` | bool | `eye_head` 활성화 |
| `model.heads.search` | bool | `search_head` 활성화 |
| `model.heads.event` | bool | `event_head` 활성화 |
| `model.heads.track` | bool | `track_head` 활성화 |
| `model.heads.mask` | bool | `mask_head` 활성화 |
| `model.heads.aux` | bool | auxiliary head 활성화 |

현재 권장 정책:

- `Stage1`
  - `event: false`
  - `mask: true`
- `Stage2`
  - `event: true`
  - `mask: false`
- deployment export
  - `event_head`, `mask_head` 제거

### 3.3 Student / structural model shape

| Option | Type | 의미 |
|---|---|---|
| `student.width_ratio` | float | 목표 student width ratio |
| `student.embed_dim` | int | 명시적 student embedding dim |
| `student.depth` | int | student depth |
| `student.num_heads` | int | student attention heads |
| `student.mlp_ratio` | float | student MLP expansion |
| `student.patch_size` | int | student patch size |
| `student.adapter_hidden_dim` | int | adapter hidden dim |
| `student.prev_state_hidden_dim` | int | prev-state encoder hidden dim |
| `student.head_hidden_dim` | int | 기본 head hidden dim |
| `student.track_head_hidden_dim` | int | tracking head hidden dim |
| `student.mask_hidden_dim` | int | mask head hidden dim |

### 3.4 Patch embedding / pretrained

| Option | Type | 의미 |
|---|---|---|
| `patch_embed.variant` | string | patch embed 구현 variant |
| `patch_embed.use_mode_affine` | bool | mode-conditioned affine modulation |
| `patch_embed.flatten_tokens` | bool | token flatten 여부 |
| `patch_embed.cache_policy` | string | patch-embed caching 정책 |
| `patch_embed.search_use_pseudo_edges` | bool | search path pseudo-edge feature 사용 |
| `patch_embed.warmup_epochs` | int | patch-embed warmup phase |
| `patch_embed.alpha`, `patch_embed.beta` | float | patch-embed mixing 계수 |
| `pretrained.enabled` | bool | pretrained loading 활성화 |
| `pretrained.path` | path or null | pretrained checkpoint 경로 |
| `pretrained.source` | string | source format 힌트 |
| `pretrained.strict_shape` | bool | strict tensor-shape match |
| `pretrained.load_stage` | string | stage gate for loading |

메모:

- 현재 코드는 위 key를 실제 소비한다.
- `variant`, `cache_policy`, `source`의 전체 허용 enum은 이번 패스에서 완전 검증되지는 않았다.

## 4. `data` 섹션

### 4.1 공통 옵션

| Option | Type | 의미 |
|---|---|---|
| `data.mode` | `mode1` or `mode2` | 데이터 계열 선택 |
| `canonical_name` | string | canonical dataset 이름 |
| `manifest_name` | string | manifest 이름 |
| `canonical_root` | path or null | canonical root override |
| `input_size` | `[H, W]` | dataset output size |
| `resize_policy` | string | spatial resize policy |
| `cache_root` | path or null | dataset cache root |
| `use_cache` | bool | dataset cache 사용 여부 |
| `per_channel_normalize` | bool | 채널별 정규화 여부 |

### 4.2 Mode 기본값

| `data.mode` | Default Canonical | Default Manifest | Default Frame Source |
|---|---|---|---|
| `mode1` | `canonical1` | `manifest1` | `original` |
| `mode2` | `canonical2` | `manifest2` | `interpolated` |

## 5. Event builder 옵션

### 5.1 핵심 event window 옵션

| Option | 알려진 값 | 의미 |
|---|---|---|
| `policy` | `fixed_count`, `time_bin`, `interval_all` | event 선택 규칙 |
| `time_bin_us` | int | `time_bin`용 시간 창 길이 |
| `event_count_target` | int | `fixed_count`용 event 수 |
| `accumulation` | `plain`, `causal_linear`, `causal_linear_ori`, `fast_causal_linear` | event accumulation 방식 |
| `causal_weight_power` | float | causal weighting 강도 |
| `fast_causal_limit` | float | fast causal saturation 한계 |
| `polarity_split` | bool | polarity 분리 여부 |

### 5.2 Policy 의미

| Policy | 의미 |
|---|---|
| `fixed_count` | 끝 시점 이전의 최근 N개 event 사용 |
| `time_bin` | `[t-end, t]` 고정 시간 창 사용 |
| `interval_all` | `[prev_t, cur_t]` 전체 interval 사용 |

## 6. `mode2.synthetic_event_builder` 추가 옵션

| Option | 알려진 값 | 의미 |
|---|---|---|
| `generation_strategy` | `raw_window`, `source_pair_average`, `target_fps_session_store` | synthetic event 생성 전략 |
| `source_pair_scope` | `anchor_window`, `pair_local` | source pair 범위 |
| `average_weighting` | `mean`, `alpha` | source pair mixing 방식 |

## 7. `data.mode2.interpolation`

핵심 옵션:

- `enabled`
- `backend`
- `model`
- `target_fps`
- `fixed_insert`
- `count_policy`
- `max_insert`
- `alpha_legacy`
- `timelens_root`
- `timelens_checkpoint`
- `timelens_device`
- `cache_root`
- `quality_default`
- `overlap_policy`

## 8. `data.mode2.execution`

| 값 | 의미 |
|---|---|
| `materialized` | target frame를 파일로 materialize |
| `lazy_target_fps` | source frame pair에서 on-the-fly 합성 |

## 9. `training` 섹션

### 9.1 핵심 학습 옵션

대표 옵션:

- `stage`
- `batch_size`
- `num_workers`
- `grad_accum_steps`
- `epochs`
- `lr`
- `weight_decay`
- `amp`
- `device`
- `pin_memory`
- `best_metric_name`
- `grad_clip_norm`
- `log_every`
- `checkpoint_every`

현재는 아래 optimizer surface도 포함한다.

- `training.optimizer`
- `training.optimizer_modifiers`
- `training.optimizer_pool`

### 9.2 Scheduler

| Option | 의미 |
|---|---|
| `type` | `none`, `cosine`, `step`, `plateau` |
| `warmup_epochs` | warmup 길이 |
| `min_lr` | cosine minimum lr |
| `metric_name` | plateau monitor metric |
| `factor`, `patience`, `threshold` | plateau 파라미터 |
| `step_size`, `gamma` | step scheduler 파라미터 |

### 9.3 Early stopping

핵심 key:

- `enabled`
- `patience`
- `min_delta`
- `start_epoch`
- `metric_name`

## 10. `experiment`와 `run`

### 10.1 `experiment`

| Option | 의미 |
|---|---|
| `experiment.name` | 실험명 |
| `train_manifest` | train manifest override |
| `val_manifest` | val manifest override |
| `test_manifest` | test manifest override |
| `output_dir` | output root override |
| `init_checkpoint` | 초기 checkpoint |

### 10.2 `run`

| Option | 의미 |
|---|---|
| `run.root` | materialized run root |
| `materialized_experiment_name` | timestamped run 이름 override |

## 11. `distillation`

핵심 옵션:

- `enabled`
- `teacher_student`
- `ema_decay`
- `feature_similarity`
- `feature_weight`
- `state_similarity`
- `state_weight`
- `prediction_similarity`
- `prediction_weight`
- `mask_similarity`
- `mask_weight`
- `teacher_init_checkpoint`
- `force_teacher_from_student`

stage-aware target key:

- `feature_keys`
- `state_keys`
- `prediction_keys`
- `mask_keys`
- `logits_keys`
- `rkd_keys`

KD / RKD:

- `kd.enabled`
- `kd.temperature`
- `kd.weight`
- `rkd.enabled`
- `rkd.distance_weight`

## 12. `regularization_ssl`와 `ssl`

`regularization_ssl`가 active modern path이고, `ssl`은 legacy compatibility surface다.

`regularization_ssl` 핵심 key:

- `enabled`
- `force_with_teacher`
- `feature_similarity`
- `feature_weight`
- `state_similarity`
- `state_weight`
- `prediction_similarity`
- `prediction_weight`
- `cross_modal_contrastive`
- `contrastive_weight`
- `self_consistency`
- `self_consistency_weight`

## 13. `pruning`

핵심 key:

- `enabled`
- `scheme`
- `mode`
- `method`
- `slimmable_backbone`
- `width_candidates`
- `sample_strategy`
- `student_width`
- `student.width_ratio`
- `width_loss_weight`
- `patch_size_mutable`
- `export.enabled`
- `export.filename`
- `export.report_filename`

코드 기준 알려진 scheme:

- `structural_width`
- `implicit_masking_legacy`
- `implicit_width_slicing`
- `legacy_masking`

## 14. `loss`

| Group | Options |
|---|---|
| eye | `eye_weight`, `eye_conf_weight` |
| mask | `mask_weight` |
| search | `search_xy_weight`, `search_ab_weight`, `search_trig_weight`, `search_geo_weight`, `search_conf_weight` |
| event | `event_xy_weight`, `event_ab_weight`, `event_trig_weight`, `event_geo_weight`, `event_conf_weight` |
| track | `track_xy_weight`, `track_ab_weight`, `track_trig_weight`, `track_geo_weight`, `track_conf_weight`, `track_quality_weight` |
| regularizer | `consistency_weight`, `constraint_center_weight`, `constraint_center_radius`, `aux_weight` |

## 15. `runtime`

| Option | 의미 |
|---|---|
| `search_conf_threshold` | minimum search confidence |
| `track_conf_threshold` | minimum track confidence |
| `track_quality_threshold` | minimum track quality |
| `similarity_threshold` | runtime switching용 similarity gate |
| `density_threshold` | event density gate |
| `relocalize_cooldown` | relocalization cooldown 길이 |

## 16. 권장 운영 preset

| Preset | 주 용도 |
|---|---|
| `mode1_stage1.yaml` | original-frame search foundation |
| `mode1_stage2.yaml` | `interval_all` event policy를 쓰는 original-frame hybrid tracking |
| `mode2_stage1.yaml` | interpolated-frame search foundation |
| `mode2_stage2.yaml` | `source_pair_average + pair_local + alpha`를 쓰는 materialized mode2 tracking |
| `exps/configs/mode2_stage2_lazy_2000fps.yaml` | session-store 기반 event range를 쓰는 lazy target-FPS Stage2 experimental preset |

## 17. 실무 권장 읽기 순서

1. [base.yaml](../../configs/base.yaml)
2. 선택한 preset
3. [scripts/_config.py](../../scripts/_config.py)
4. [dataset.py](../../src/hbtxr/data/dataset.py)
5. [trainer.py](../../src/hbtxr/training/trainer.py)

## 18. 실무적 해석

YAML 파일은 단순 hyperparameter 파일이 아니다. 아래 실행 계약 전체를 정의한다.

- 어떤 mode contract를 활성화할지
- 어떤 frame / event generation path를 사용할지
- 어떤 head를 켤지
- 어떤 training stage를 사용할지
- 어떤 distillation / export rule을 적용할지

가장 영향이 큰 key:

- `data.mode`
- `data.mode2.execution`
- `event_builder.policy`
- `synthetic_event_builder.generation_strategy`
- `training.stage`
- `model.heads.*`
- `pruning.scheme`

## 19. 코드 참조

- config load / merge / override
  - `scripts/_config.py`
- train CLI
  - `scripts/train_hbtxr.py`
- mode-aware loader contract
  - `src/hbtxr/data/loader.py`
- event / frame generation contract
  - `src/hbtxr/data/dataset.py`
- training / model construction
  - `src/hbtxr/training/trainer.py`
- pruning normalization
  - `src/hbtxr/models/pruning.py`
- deployment export stripping
  - `src/hbtxr/models/export_pruned.py`
