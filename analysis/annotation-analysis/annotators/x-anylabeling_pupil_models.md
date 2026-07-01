# X-AnyLabeling — Pupil Mask Segmentation / Detection 모델 리스트

`third/X-AnyLabeling`(v4.0.0-beta.11)에서 **실제 번들된**(`anylabeling/configs/models.yaml` + `configs/auto_labeling/*.yaml`) 모델 중 eye-tracking 동공 **마스크 분할/검출**에 쓸 수 있는 것. 전체 분석은 [x-anylabeling.md](x-anylabeling.md). 마스크는 polygon으로 변환되어 **COCO/YOLO-seg/이진 mask PNG** export 가능.

## A. Promptable SAM 계열 — 마스크(클릭/박스로 동공 지정) · 클래스 무관 → 도메인갭 최소
| 모델 | config | 프롬프트 | 자동배치 | 동공 적합 |
|---|---|---|---|---|
| **SAM-HQ** (ViT-B/L/H ±quant) | `sam_hq_vit_*.yaml` | point/box | ❌ | ⭐ 작은 동공 경계 최상, 수동 gold-standard |
| **SAM 2.1** (tiny/small/base/large) | `sam2_hiera_*.yaml` | point/box | ❌ | ⭐ 강한 마스크 |
| SAM (ViT-B/L/H ±quant) | `segment_anything_vit_*.yaml` | point/box | ❌ | 좋음 |
| MobileSAM | `mobile_sam_vit_h.yaml` | point/box | ❌ | 경량 |
| EfficientViT-SAM (l0/l1) | `efficientvit_sam_l*.yaml` | point/box | ❌ | 빠름 |
| EdgeSAM | `edge_sam.yaml` | point/box | ❌ | 매우 빠름 |
| SAM-Med2D (ViT-B, 256px) | `sam_med2d_vit_b.yaml` | point/box | ❌ | 의료튜닝(256px, IR 전이 가능성) |
| SAM 2 Video | `sam2_hiera_*_video.yaml` | point/box(+text) | ✅(비디오) | 프레임 추적 |

## B. Text→마스크 (grounded, "pupil"/"black pupil" 입력) · 폴더 자동배치 가능
| 모델 | config | 비고 |
|---|---|---|
| **GroundingSAM2** (GroundingDINO-SwinT + SAM2.1-L) | `groundingdino_swint_sam2_large.yaml` | ⭐ **우리 GSAM2 파이프라인 직접 아날로그**. text→box→mask, 배치. 단 우리 스택과 계열 겹쳐 독립성 약함 |
| GroundingSAM (GroundingDINO-SwinB + SAM-HQ) | `groundingdino_swinb_attn_fuse_sam_hq_vit_l_quant.yaml` | text→HQ 마스크 |
| SAM 3 (ViT-H) | `sam3_vit_h.yaml` | text 개념→마스크, 배치. 단 client ONNX는 text-only+느림(~5GB), `third/SAM3-I`와 중복 |
| Open Vision (BERT + SAM2.1-L) | `open_vision.yaml` | text(카운팅), BERT 수동 준비 필요 |

## C. Text→박스 (검출만, 마스크 없음)
| 모델 | config | 비고 |
|---|---|---|
| **Grounding DINO** (SwinT/SwinB) | `groundingdino_swint_ogc*.yaml` | text "pupil"→bbox. 마스크 원하면 SAM과 결합(=B) |
| YOLOE | `yoloe_*.yaml` | 프롬프트→box/mask |

## D. 닫힌 어휘(COCO) 분할 — 동공엔 재학습 필요
`yolov8/v5/11/26-seg`, `rfdetr_seg`, `hyper_yolo_seg` (+bytetrack/botsort 추적 변형). COCO 80클래스라 "pupil" 없음 → **커스텀 학습(도구의 Ultralytics 학습 패널)** 시에만 자동 동공 분할기.

## 동공/eye-tracking 실전 선택
- **대량 자동 마스크**: **GroundingSAM2**(text `"black pupil."`, 배치·자동다운로드) — 단 우리 GSAM2와 계열 겹침(약한 독립성).
- **소수 프레임 gold-standard**: **SAM-HQ / SAM 2.1** 수동 클릭/박스(동공은 고대비 검은 원 → SAM 이상적 타깃, 폴더 자동모드 없음).
- **검출(box)만**: **Grounding DINO** text 프롬프트.
- **완전 독립 자동 분할기 원하면**: YOLO-seg 커스텀 학습(별도 워크스트림).
- 근적외 grayscale 주의: text-grounded 모델은 저대비에서 under-fire 가능 → `box_threshold/conf_threshold`↓, 프롬프트 `"black pupil."`/`"dark circle."`. 입력은 3채널 RGB로 자동 로드되어 grayscale도 OK.

> 참고: X-AnyLabeling 가중치는 in-repo가 아니라 최초 사용 시 자동 다운로드(GitHub-release ONNX). 그래서 `weights/`엔 미수집([../weights/ANNOTATOR_WEIGHTS.md](../weights/ANNOTATOR_WEIGHTS.md)).
