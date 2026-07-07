# Swift-Eye 및 TimeLens 파이프라인 표

최종 갱신: 2026-03-27 KST

## 목적

이 문서는 Swift-Eye와 TimeLens의 실제 코드 기준으로 다음 4개 파이프라인을 표 형태로 정리합니다.

1. TimeLens 보간 파이프라인
2. Swift-Eye Detection 학습
3. Swift-Eye Temporal Fusion / Tracking 학습
4. Swift-Eye 추론

## 전제

- TimeLens 샘플 해상도: `260 x 346`
- Swift-Eye 입력 해상도: pad 후 `288 x 352`
- Swift-Eye backbone 첫 feature map: `72 x 88`
- TimeLens voxel time-bin: `5`

## 1. TimeLens 보간 파이프라인

| 단계 | 객체 | shape / 구조 | 비고 |
|---|---|---|---|
| Raw dataset | `images/*.png` | `[260,346]` grayscale PNG | 로딩 시 RGB `[3,260,346]` 사용 |
| Raw dataset | `images/timestamp.txt` | `[N_frames]` | boundary frame timestamp |
| Raw dataset | `events/*.npz` | `keys=(x,y,t,p)` | 개념적으로 `[E,4]` stream |
| Iterator build | boundary frame pair | `(Frame(t0), Frame(t1))` | 연속 frame pair |
| Iterator build | interframe event sequence | `[E_interval,4]` | 두 frame 사이 전체 이벤트 |
| Offline split | target timestamps | 구간당 `199`개 | inserted frame 수 |
| Offline split | `left_events` | `[E_left,4]` | `t0 ~ t*` |
| Offline split | `right_events` | `[E_right,4]` | `t* ~ t1` |
| Transform | `before.rgb_image_tensor` | `[3,260,346]` | `Frame(t0)` |
| Transform | `after.rgb_image_tensor` | `[3,260,346]` | `Frame(t1)` |
| Transform | `before.reversed_voxel_grid` | `[5,260,346]` | left events reverse voxel |
| Transform | `after.voxel_grid` | `[5,260,346]` | right events voxel |
| Collate | example dict | `{'before','middle','after'}` | 예제 스크립트는 사실상 `B=1` |
| Collate 결과 | `before.rgb_image_tensor` | `[B,3,260,346]` |  |
| Collate 결과 | `after.rgb_image_tensor` | `[B,3,260,346]` |  |
| Collate 결과 | `before.reversed_voxel_grid` | `[B,5,260,346]` |  |
| Collate 결과 | `after.voxel_grid` | `[B,5,260,346]` |  |
| Model forward | Warp 입력 | `[2B,5,260,346]` | before/after concat |
| Model forward | Warp flow 출력 | `[2B,2,260,346]` | chunk 후 각각 `[B,2,260,346]` |
| Model forward | Warped images | `[2B,3,260,346]` | before/after 각각 |
| Model forward | Fusion 입력 | `[B,16,260,346]` | `5+3+5+3` 채널 |
| Model forward | Fusion 출력 | `[B,3,260,346]` | synthesis branch |
| Model forward | Refine 입력 | `[B,9,260,346]` | `3+3+3` |
| Model forward | Residual flow 출력 | `[B,4,260,346]` | before/after residual flow |
| Model forward | Attention 입력 | `[B,14,260,346]` | `2+3+2+3+3+1` 채널 |
| Model forward | Attention score | `[B,3,260,346]` | before / after / fusion weight |
| Model forward | 최종 interpolated frame | `[B,3,260,346]` | 저장 시 `[260,346,3]` |
| Offline output | `interpolated_frames/*.png` | `[N_interp,260,346,3]` | Swift-Eye 입력 |

## 2. Swift-Eye Detection 학습

| 단계 | 객체 | shape / 구조 | 비고 |
|---|---|---|---|
| Raw dataset | `detection_train.pickle` row | `{'image_path','poly'}` | `poly=[x1,y1,...,x4,y4]` |
| Dataset `__getitem__` | raw image | `[260,346,3]` | frame 1장 |
| Dataset `__getitem__` | raw poly | `[8]` | pupil polygon |
| Dataset `__getitem__` | `gt_bboxes` | `[1,5]` | polygon -> oriented bbox |
| Dataset `__getitem__` | `gt_labels` | `[1]` | 클래스는 `pupil` 하나 |
| Dataset `__getitem__` | pipeline 후 image | `[3,288,352]` | normalize + pad + CHW |
| DataLoader | batch before collate | sample dict list, 길이 `B` | MMCV `DataContainer` 사용 |
| Collate 결과 | `img` | `[B,3,288,352]` | 예: `B=8` |
| Collate 결과 | `gt_bboxes` | list 길이 `B`, 각 `[1,5]` | stack=False |
| Collate 결과 | `gt_labels` | list 길이 `B`, 각 `[1]` |  |
| Collate 결과 | `img_metas` | list 길이 `B` | ori/img/pad shape 포함 |
| Model forward 입력 | `x` | `[B,3,288,352]` | backbone+neck 입력 |
| Backbone+Neck | first feature | `[B,256,72,88]` | 주요 feature map |
| Detection head 입력 | `roi_features` | `tuple([ [B,256,72,88] ])` | tuple 입력 |
| Detection head 출력 | rbbox result | image별 class list, det는 `[x,y,w,h,a,score]` | simple_test 기준 |

## 3. Swift-Eye Temporal Fusion / Tracking 학습

| 단계 | 객체 | shape / 구조 | 비고 |
|---|---|---|---|
| Raw dataset | `tracking_train_dataset.pickle` row | `{'template_path','origin_poly','search_path','occlusion_poly'}` | template-search pair |
| Dataset `__getitem__` | template image | `[3,288,352]` | template frame |
| Dataset `__getitem__` | search image | `[3,288,352]` | search frame |
| Dataset `__getitem__` | template bbox | `[1,5]` | `origin_poly -> OBB` |
| Dataset `__getitem__` | search bbox | `[1,5]` | `occlusion_poly -> OBB` |
| Dataset `__getitem__` | 동기 augmentation | 동일 flip / 동일 rotate | pair geometry 유지 |
| 1차 collate | sample 내부 pair | `[2,3,288,352]` | template 1장 + search 1장 |
| 1차 collate | `gt_bboxes` | list 길이 `2`, 각 `[1,5]` | pair 단위 |
| 2차 collate | outer batch image | `[2B,3,288,352]` | 예: `B=8`이면 `[16,3,288,352]` |
| 2차 collate | outer batch `gt_bboxes` | list 길이 `2B`, 각 `[1,5]` | template/search 번갈아 배치 |
| 2차 collate | outer batch `img_metas` | list 길이 `2B` |  |
| Model forward 입력 | `x` | `[2B,3,288,352]` | backbone+neck 입력 |
| Backbone+Neck | whole feature | `[2B,256,72,88]` | even=template, odd=search |
| Feature crop | `template_feature` | `[B,256,13,13]` | `template_size=13` |
| Feature crop | `search_feature` | `[B,256,33,33]` | `search_size=33` |
| Tracking head 입력 | template/search | `[B,256,13,13]` + `[B,256,33,33]` | correlation 후 tracking head |
| Tracking head 출력 | rbbox result | search image별 `[x,y,w,h,a,score]` | simple_test 기준 |

## 4. Swift-Eye 추론

| 단계 | 객체 | shape / 구조 | 비고 |
|---|---|---|---|
| Raw dataset | `interpolated_frames/*.png` | `[260,346,3]` | TimeLens 출력 frame sequence |
| Test pipeline | 단일 frame | `[3,288,352]` | Compose + pad 후 |
| Collate 결과 | `img` | `[1,3,288,352]` | 프레임 1장씩 순차 처리 |
| Model forward(첫 frame) | `whole_features` | `[1,256,72,88]` | detection mode 시작 |
| Model forward(첫 frame) | detection result | `[1 image][1 class][N,6]` | pupil 초기화 |
| Model state | cached template | `[1,256,13,13]` | open extent 충분 시 저장 |
| Model forward(이후 frame) | search ROI | `[1,256,33,33]` | last prediction 중심 crop |
| Model forward(이후 frame) | tracking result | `[1 image][1 class][N,6]` | tracking mode |
| Auxiliary forward | mask UNet 입력 | `[1,1,260,346]` | grayscale eye image |
| Auxiliary forward | mask UNet 출력 | `[1,2,260,346]` | pupil / non-pupil segmentation |
| Decision logic | `open_extent` | scalar | detection / tracking / interpolation 전환 |
| 최종 출력 | `pred_ep` | `[5]` | `(x, y, w, h, angle)` |

## 메모

- 위 표는 Swift-Eye 저장소와 함께 제공된 TimeLens 코드 구조를 기준으로 재구성한 것입니다.
- TimeLens는 raw event를 직접 Swift-Eye detector/tracker에 넣지 않고, 먼저 dense interpolated frame을 생성합니다.
- Swift-Eye temporal fusion 학습의 핵심은 pair-synchronized augmentation과 2단계 collate 구조입니다.

## 주요 코드 참조

- `references/timelens/tests/run_attention.py`
- `references/timelens/timelens/common/transformers.py`
- `references/timelens/timelens/common/representation.py`
- `references/timelens/timelens/warp_network.py`
- `references/timelens/timelens/fusion_network.py`
- `references/timelens/timelens/refine_warp_network.py`
- `references/timelens/timelens/attention_average_network.py`
- `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_without_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py`
- `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py`
- `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/pipelines/collate.py`
- `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/train_with_temporal_fusion_component.py`
- `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/swift_eye/model.py`
