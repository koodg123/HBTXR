# Eye/Pupil Annotation Tool Survey — for EV-Eye near-IR pupil MASK annotation & audit

생성: 2026-07-01. 대상: EV-Eye(DAVIS346, 근적외 grayscale 근안 프레임, 동공 = 346×260 안의 ~40–60px 검은 원)의 **동공 마스크 주석**과 **독립 audit 소스** 확보.

## 배경 — 왜 이 조사인가
현재 파이프라인의 라벨 소스: **사람 GT(VIA 타원, sparse)** · **U-Net dense 의사라벨** · **GSAM2(GroundingDINO+SAM2) audit**. 리뷰어 대응(label noise / uncertainty)을 위해 **서로 독립적인 동공 검출기**가 많을수록 좋다. GSAM2는 근적외 도메인 갭 + ~3% 오검출(하단 마커) 한계가 있어, **근안 전용 모델**을 추가 audit 소스로 검토한다. 각 도구 상세는 아래 링크 파일 참조.

## 비교표

| 도구 | 방법/계보 | 동공 출력 | 근적외 도메인 적합 | 가중치 in-repo | 라이선스 | GSAM2 대비 독립성 | 통합난이도 |
|---|---|---|---|---|---|---|---|
| [EllSeg](ellseg.md) | DenseNet U-Net (RITnet 계보), ellipse seg | 동공+홍채 **mask+타원+center** | ✅ 근적외 320×240 | ✅ ~10MB (`.git_ok`) | **MIT** | ✅ 높음 | **Low–Med** |
| [RITnet-Plugins](ritnet-plugins.md) | RITnet 4클래스 + EllSeg/v2 (Pupil Core 플러그인) | 동공 **mask+타원+center**+지름 | ✅ 근적외 | ✅ 1–12MB | **MIT** | ✅ 높음 | **Low–Med** (~1일) |
| [DeepVOG](deepvog.md) | Keras/TF1 U-Net + 3D 안구모델 | 동공 mask→타원→center (+gaze) | ✅ 근적외 grayscale | ✅ 99MB (`.h5`) | GPLv3 | ✅ 높음 | Med (레거시 TF env) |
| [Edge-Guided](edge-guided-near-eye.md) | BDCN edge + DenseNet2D (EllSeg 변형, ISMAR'21) | 동공+홍채 **mask+타원** | ✅ 근적외 320×240 | ✅ 13+63MB | ⚠️ 없음(인용만) | ✅ 높음 | Low–Med (구버전 deps, avi/스테레오 수정) |
| [X-AnyLabeling](x-anylabeling.md) | GUI 오토라벨러; SAM 계열 + GroundingSAM2 | mask→polygon → COCO/mask PNG | ⚠️ 텍스트모델 갭; SAM 클릭은 OK | 자동 다운로드 | GPL-3.0 | GroundingSAM2 약함 / SAM-HQ(수동) 높음 | Low(GUI)/Med(헤드리스) |
| [SAM3-I](sam3-i.md) | Meta SAM3 명령튜닝(adapter/LoRA), text-only | text→인스턴스 mask(RLE) | ⚠️ RGB 학습, IR 갭 큼 | ❌ gated/GDrive | ⚠️ 불명확 | ✅ 높으나 불확실 | High |

> 독립성 = "GSAM2(GroundingDINO+SAM2)와 다른 모델 계보인가". EllSeg/RITnet/DeepVOG/Edge-Guided는 근안 전용 세그 모델이라 계보가 완전히 다름(강한 독립 audit). X-AnyLabeling의 GroundingSAM2는 우리 스택과 같은 계열이라 약함.

## 도구별 요지

- **EllSeg** — 근안 타원 세그의 표준. 동공/홍채를 **가림에도 타원 전체**를 예측. 가중치 in-repo(MIT), CPU 가능, 입력 320×240으로 346×260 정합 후 원해상도 환산. 걸림돌은 image-folder 모드 부재(≈30줄 래퍼) + 구버전 PyTorch env. → **최우선 후보**.
- **RITnet-Plugins** — RITnet(bg/sclera/iris/pupil 4클래스) + EllSeg/v2를 Pupil Core 플러그인으로. **추론 코어(`ritnet/`)는 Pupil Core 없이 독립 실행 가능**, 가중치 전부 in-repo(MIT). OpenCV4 findContours arity 등 소소한 수정. → **최우선 후보(EllSeg와 사실상 동일 계보 → 둘 중 하나)**.
- **DeepVOG** — 근안 동공 U-Net + 3D 안구/시선. 가중치 in-repo(99MB)라 획득 불필요, 도메인 적합 강함. 단 **GPLv3**(벤더링 말고 subprocess 격리) + **레거시 TF1/Keras/Py≤3.7** 필요 + **비디오 전용 I/O**(PNG 폴더 어댑터 필요). → 별도 env 감수 시 좋은 3번째 표.
- **Edge-Guided** — 엣지 유도 세그(EllSeg 변형), 동공+홍채 타원. 가중치 in-repo. 단 **라이선스 파일 없음**(사용 전 확인), `evaluate.py`가 **스테레오 640폭 하드코딩 + avi 전용** → 단안·프레임 입력 수정 필요, 구버전 deps(torch1.2/CUDA9.2).
- **X-AnyLabeling** — 도구 자체는 세그 모델 다수 번들: **SAM/SAM2/SAM3/SAM-HQ/MobileSAM/EfficientViT-SAM/EdgeSAM/SAM-Med2D + GroundingSAM2 + YOLO*-seg**. Mask→polygon→**COCO/YOLO-seg/mask PNG** export. 근적외 동공엔: (1) **GroundingSAM2**=우리 파이프라인의 직접 아날로그(편리하나 약한 독립성), (2) **SAM-HQ/SAM2 수동 클릭**=도메인 갭 최소, gold-standard 스팟체크용(폴더 자동 모드 없음), (3) 텍스트 모델은 IR 저대비에서 under-fire 위험(threshold 조정 필요).
- **SAM3-I** — Meta SAM3 위 명령튜닝. 진짜 독립적이나 **가중치 gated(HF)+GDrive**로 미포함, **RGB 학습→IR 갭**, 단일이미지 데모 없음, 라이선스 불명확. → 비용·불확실성 높음(후순위).

## 권고 (audit 소스 추가 우선순위)
1. **EllSeg 또는 RITnet-Plugins**(둘 중 하나; 동일 RITnet 계보) — MIT·가중치 in-repo·근안 전용·Low–Med. GSAM2와 독립적인 **4번째 표(y_ellseg)**로 즉시 유력.
2. **DeepVOG** — 레거시 TF env를 별도로 감수할 수 있으면 독립성 강한 추가 표.
3. **X-AnyLabeling SAM-HQ 수동** — 소수 프레임 **gold-standard 스팟체크**(사람 클릭)로, 사람 재주석 대체 정밀도 검증에 유용.
4. Edge-Guided(라이선스 확인 후), SAM3-I(가중치·도메인 해결 시) — 후순위.

> 이들은 모두 GSAM2와 동일한 계약(프레임→mask→ellipse-fit center)으로 붙일 수 있어, `y_orig`/`y_unet`/`y_gsam2` 옆에 **동일 좌표계(346×260) center**를 추가해 label-noise·uncertainty 교차검증에 바로 투입 가능. 통합 시 `scripts/08_run_gsam2.py`의 출력 계약(`label/{key}/*.json` + per-frame center/mask/bbox)을 그대로 미러링 권장.
