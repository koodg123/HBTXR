# Annotator pre-trained weights (수집본)

annotator 서베이([../annotators/](../annotators/)) 도구 중 **가중치를 in-repo로 제공**하는 것들을 모델명별로 복사. 2026-07-01. 원본은 `third/<repo>/`.

> `.git_ok` 확장자 = PyTorch state_dict를 `.pt/.pkl` gitignore 회피용으로 rename한 것(실제로는 정상 PyTorch 체크포인트). 각 도구가 그 경로 그대로 로드함(예: EllSeg `--loadfile ...all.git_ok`).

| 폴더 | 파일 | 모델/용도 | 원본 | 라이선스 |
|---|---|---|---|---|
| `EllSeg/` | `all.git_ok`(권장) + LPW/pupilnet/openeds/riteyes/Fuhl/nvgaze | DenseElNet(=ritnet_v3), 동공+홍채 mask+타원. 데이터셋별 학습본 | `third/EllSeg/weights/` | MIT |
| `RITnet/` | `best_model.pkl`·`ritnet_400400.pkl`·`ritnet_pupil.pkl` | RITnet DenseNet 4클래스(bg/sclera/iris/pupil) | `third/Pupil-Labs-Core-RITnet-Plugins/ritnet/` | MIT |
| `RITnet/` | `ellseg_v2_pretrained.git_ok` | EllSeg-v2 사전학습 | `.../ritnet/Ellseg_v2/pretrained/` | MIT |
| `DeepVOG/` | `DeepVOG_weights.h5` | Keras/TF U-Net 동공 seg(+3D gaze 파이프라인) | `third/DeepVOG/deepvog/model/` | ⚠️ GPLv3 |
| `Edge-Guided/` | `baseline_edge_16.pkl` | DenseNet2D seg+ellipse(동공+홍채) | `third/Edge-Guided-.../` | ⚠️ 없음(인용만) |
| `Edge-Guided/` | `gen_00000016.pt` | BDCN 엣지 검출기(edge-guided 입력) | `third/Edge-Guided-.../` | ⚠️ 없음 |

**미수집(가중치 미제공):**
- **X-AnyLabeling**: 가중치를 최초 사용 시 자동 다운로드(GitHub-release ONNX, `~/xanylabeling_data/models/`). in-repo 아님 → 필요 시 도구가 받음.
- **SAM3-I**: base(Meta SAM3)=HF **gated**, 파인튜닝본=Google Drive → 자동 획득 불가.
- GSAM2용 SAM2.1/GroundingDINO는 이미 `gsam2/`에 있음. HBTXR는 `hbtxr/`.

**적용 메모:** EllSeg/RITnet은 근안 전용·MIT·가중치 있음 → GSAM2 외 **독립 audit 소스** 최우선(상세 [[annotator-survey]] / `../annotators/README.md`). 08 계약(프레임→mask→ellipse-fit center, 346×260) 미러링해 붙이면 `y_orig/y_unet/y_gsam2/y_pred` 옆 5번째 소스로 사용 가능. DeepVOG(GPLv3)·Edge-Guided(라이선스 없음)는 사용 전 라이선스 확인.
