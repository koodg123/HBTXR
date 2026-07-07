# Tracker And Dataset Output Contracts

## 목적

- 리팩터링 이후에도 `dataset -> model -> loss` 경로의 key / shape / 의미가 변하지 않도록 기준 계약을 남긴다.
- `Head` 분리, `Branch` 분리, `Dataset` 분리 후에도 테스트와 디버깅 기준점을 유지한다.
- 설명용 기준 문서는 이 문서이고, 테스트용 source of truth는 `tests/snapshots/output_contracts_v3.json` 이다.
- 스냅샷 검증 테스트는 `tests/test_output_contract_snapshot_v3.py` 에서 수행한다.
- `loss/stage.py` facade 와 `stage1.py`, `stage2.py`, `metrics.py` 분리 동일성은 `tests/test_loss_stage_module_split_v3.py` 에서 검증한다.
- CI guard 는 `.github/scripts/check_snapshot_guard.py` 와 `.github/workflows/ci.yml` 에서 수행한다.

## snapshot 변경 규칙

- snapshot 파일:
  - `tests/snapshots/output_contracts_v3.json`
- 계약 설명 문서:
  - `docs/prj/20260408_190000_tracker_dataset_output_contracts.md`

운영 규칙:

- snapshot 이 바뀌면 계약 설명 문서도 같이 갱신한다.
- CI 에서는 `HBTXR_UPDATE_SNAPSHOTS=1` 을 허용하지 않는다.
- CI 는 snapshot test 이후 snapshot 파일이 dirty 상태로 남아 있으면 실패한다.
- 즉 snapshot 변경은 단순 테스트 결과가 아니라 의도적인 ABI 변경으로 취급한다.

## 1. Dataset Sample Contract

기준 클래스:
- `EVEyeHBTXRDataset`
- `Mode0Dataset`
- `Mode1Dataset`
- `Mode2Dataset`

주요 반환 key:

| key | shape / type | 의미 |
|---|---|---|
| `sample_id` | `str` | manifest sample 식별자 |
| `frame` | `torch.float32 [1,H,W]` | transform 이후 grayscale frame |
| `event` | `torch.float32 [2,H,W]` | transform 이후 polarity-split event voxel |
| `mask_target` | `torch.float32 [1,H,W]` | pupil mask supervision |
| `eye_target` | `torch.float32 [5]` | `[cx, cy, w, h, valid]` eye region supervision |
| `pupil_region_target` | `torch.float32 [5]` | `[cx, cy, w, h, valid]` pupil ROI bbox supervision |
| `prev_state` | `torch.float32 [6]` | 이전 state `xyabuv` |
| `cur_state` | `torch.float32 [6]` | 현재 state `xyabuv` |
| `pupil_search_target` | `torch.float32 [7]` | `[x, y, a, b, u, v, conf]` search/event head target |
| `pupil_track_target` | `torch.float32 [8]` | `[dx, dy, dlog_a, dlog_b, du, dv, conf, quality]` track head target |
| `constraint_center` | `torch.float32 [2]` | eye region center constraint |
| `annotation_quality` | scalar tensor | supervision quality weight |
| `similarity_target` | scalar tensor | runtime / auxiliary similarity target |
| `event_density` | scalar tensor | selected event count normalized by ROI area |
| `closed_eye_flag` | scalar tensor | closed-eye supervision flag |
| `mask_valid` | scalar tensor | mask supervision validity |
| `valid_track` | scalar tensor | track supervision validity |
| `aux_target` | `torch.long` scalar | auxiliary class target |
| `meta` | `dict` | 디버깅/실행 계약 메타데이터 |

## 2. Tracker Output Contract

기준 클래스:
- `HBTXRTracker`

search branch:

| key | shape | 의미 |
|---|---|---|
| `search/pooled` | `[B,D]` | frame branch pooled token |
| `search/eye` | `[B,5]` 또는 detector-specific contract | eye detector primary output |
| `search/pupil` | `[B,7]` | `[x, y, a, b, u, v, conf]` |
| `search/state` | `[B,6]` | `search/pupil[..., :6]` |
| `search/pupil_bbox` | `[B,5]` | axis-aligned bbox auxiliary output |
| `search/pupil_obb` | `[B,6]` | rotated bbox auxiliary output |
| `search/mask_logits` | `[B,1,H,W]` | dense mask logits |
| `search/mask_coarse_logits` | `[B,1,H,W]` | coarse mask logits when enabled |
| `search/aux` | `[B,C]` | auxiliary class logits |

event branch:

| key | shape | 의미 |
|---|---|---|
| `event/pooled` | `[B,D]` | event branch pooled token |
| `event/pupil` | `[B,7]` | event search state prediction |
| `event/state` | `[B,6]` | `event/pupil[..., :6]` |
| `event/pupil_bbox` | `[B,5]` | event bbox auxiliary output |
| `event/pupil_obb` | `[B,6]` | event rotated bbox auxiliary output |
| `event/aux` | `[B,C]` | event auxiliary logits |

track branch:

| key | shape | 의미 |
|---|---|---|
| `track/event_pooled` | `[B,D]` | track path event pooled token |
| `track/prev_feat` | `[B,D]` | encoded previous state |
| `track/fused` | `[B,2D]` | pooled + prev_state feature |
| `track/pupil` | `[B,8]` | `[dx, dy, dlog_a, dlog_b, du, dv, conf, quality]` |
| `track/state` | `[B,6]` | decoded current `xyabuv` state |

공통 pruning/runtime:

| key | shape | 의미 |
|---|---|---|
| `pruning/active_dim` | `[B]` | active embedding width |
| `pruning/active_width` | `[B]` | active width ratio |
| `runtime/state` | scalar-like | search/track scheduler state |
| `runtime/reason` | scalar-like | runtime transition reason |
| `runtime/cache_valid` | `[B]` | patch cache validity flag |

## 3. Loss Contract

stage1 기준 입력:
- `search/pupil`
- `search/state`
- `search/eye`
- `search/mask_logits`
- optional: `search/pupil_bbox`, `search/pupil_obb`, `search/aux`

stage2 기준 입력:
- stage1 입력 전체
- `event/pupil`
- `event/state`
- `track/pupil`
- `track/state`

핵심 원칙:
- loss variant는 output contract를 바꾸지 않는다.
- `CIoU`, `rotated_bbox`, `alpha weighting`, `mask mode`는 bundle 전략 옵션이다.
- 따라서 `Head` 개수는 loss 옵션 개수와 1:1로 늘리지 않는다.

## 4. YAML Component Selection Contract

dataset selector:

```yaml
data:
  components:
    reader: { variant: split_v1 }
    transform: { variant: split_v1 }
    event_builder: { variant: split_v1 }
    roi: { variant: split_v1 }
    targets: { variant: split_v1 }
    assembler: { variant: split_v1 }
```

tracker selector:

```yaml
model:
  components:
    encoder: { variant: split_v1 }
    head_factory: { variant: split_v1 }
    search_refiner: { variant: split_v1 }
    search_branch: { variant: split_v1 }
    event_branch: { variant: split_v1 }
    track_branch: { variant: split_v1 }
    runtime_policy: { variant: split_v1 }
    track_codec: { variant: split_v1 }
```

현재 `default`와 `split_v1`는 같은 구현을 가리킨다.
이 이름을 남겨둔 이유는 이후 `split_v2`, `fast_io_v2`, `runtime_light` 같은 대체 구현을 YAML만 바꿔 선택할 수 있게 하기 위해서다.
