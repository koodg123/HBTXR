# 07 · GSAM2 audit 결과 (08_run_gsam2 실행)

실행: 2026-07-01. Claude가 세션 내에서 직접 실행(과거엔 사용자 tmux 예정이었으나 WSL 네이티브 CLI라 bash 동작).
산출: `samples/label/*/gsam2.json` (483 anchor). 다음: 09(pred) + 10(eval).

## 1. 실행 환경 (재현 레시피)
- **GPU venv**: `analysis/annotation-analysis/.venv-gsam2` (분석용 `.venv`와 분리 — torch 미포함 경량 env 유지).
- torch 2.5.1+cu121 / torchvision 0.20.1+cu121 (RTX 4070 Ti, sm_89).
- 코드: `third/Grounded-SAM-2`(clone) → `pip install -e .`(sam2, `SAM2_BUILD_CUDA=0`) + `pip install -e grounding_dino`.
  - **CUDA 확장 빌드**: gcc가 13이라 nvcc 12.0과 불일치 → `CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=/usr/bin/g++-12 CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST=8.9`로 g++-12를 호스트 컴파일러로 지정해 성공(2분).
  - **transformers는 4.44.2**로 다운그레이드(5.x는 `BertModel.get_head_mask` 제거로 GroundingDINO 실패). 그 외: supervision, addict, yapf, timm, pycocotools, opencv-python-headless.
- **실행 시 필수**: `cd scripts` + `PYTHONPATH=<Grounded-SAM-2 repo root>` (이 fork 코드가 `grounding_dino.groundingdino.*` 네임스페이스로 내부 import). `--gdino-cfg`는 절대경로. 가중치는 `weights/gsam2/`.

## 2. 실행 커맨드
```bash
cd analysis/annotation-analysis/scripts
PYTHONPATH=<repo>/third/Grounded-SAM-2 ../.venv-gsam2/bin/python 08_run_gsam2.py \
  --out ../samples \
  --sam2-ckpt ../weights/gsam2/sam2.1_hiera_large.pt --sam2-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --gdino-ckpt ../weights/gsam2/groundingdino_swint_ogc.pth \
  --gdino-cfg <repo>/third/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --prompt "black pupil." --box-thr 0.25 --text-thr 0.20 --max-box-frac 0.55 \
  --repeats 4 --tta --anchors-only
```

## 3. 프롬프트 수정 (핵심)
- `"pupil."`(원 기본값)은 **top-1 박스가 이미지 전체**(det~0.42), 진짜 동공은 2등(det~0.32) → argmax가 전체이미지 선택 → 실패(중심이 이미지 중앙).
- 진단(프롬프트 6종×3모션): `"black pupil."`/`"black circle."`/`"dark circle."`는 **top-1이 정확히 동공**(det 0.46–0.50, box ~43×42, GT에 sub-px 일치).
- 08 스크립트 보강: `gdino_box`에 **과대박스 거부 필터**(`--max-box-frac 0.55`; 폭·높이 > 55%*이미지 → 제거) 추가 → 전체이미지·눈전체 박스 안전 제거.

## 4. 결과 (초기 483 anchor, anchors-only)

> **[2026-07-01 확장]** 이후 `--save-masks --overwrite`로 **전 4,669 프레임** 재실행: valid **4,666/4,669 (99.9%)**, gsam2 마스크 4,666 저장(`label/{key}/gsam2_masks/{stem}_mask.png`). per-frame 트리·오버레이·annotator 서베이는 [08](08_perframe_dataset.md). 아래는 초기 anchor-only 분석(수치 동일 경향).
- **valid rate: 483/483 = 100%**.
- **‖y_gsam2 − y_orig‖ (label-noise proxy, px)**: 전체 median **0.75**, p90 1.38, p95 1.66, mean 3.21, max 124.9.
  - 모션별 median: fixation 0.88 / saccade 0.71 / smooth 0.68.
- **GSAM2 precision proxy (repeat-std, box 지터×4 + h-flip TTA)**: median **0.46px**, p95 0.53 — 매우 일관.
- **오검출 tail**: err>10px = **14/483 (2.9%)**, err>5px = 15/483(3.1%). 원인: 동공이 눈꺼풀에 가려진 어려운/블링크 프레임에서 **하단 캘리브레이션 마커 기둥**을 오검출.
  - 오검출 특징(정상 대비): det 0.32 vs 0.45, area 269 vs 777, cy 189(하단) vs 123(중앙) — 뚜렷이 분리됨.

## 5. 해석 (리벗 관점)
- 독립 검출기(GSAM2)가 **사람 EV-Eye 라벨과 median 0.75px(정상프레임 p95 ~1.5px) 일치** → 이것이 **label-noise floor**의 강한 근거. 모델이 보고한 0.18px(dense 라벨 기준)보다 **약 4× 큼** → 사람 GT 기준 floor가 dense-label 기준 값과 **다른 reference**임을 보여줌(누수로 단정하지 않음 — 사용자 정정).
- 정밀도 proxy 0.46px는 결정론 검출기에 섭동을 준 재현 산포(사람 정밀도 아님 — 리벗에 "automated proxy" 명시).

## 6. 미결(결정 필요) — 오검출 tail 처리
- ~3% 오검출은 **라벨노이즈가 아니라 GSAM2 검출실패**. label-noise floor 산정에서 제외 필요.
- 권장: **U-Net dense center와 교차검증 게이트**(`‖y_gsam2 − y_unet‖ > ~15px`면 GSAM2 invalid) — 사람 GT 미사용이라 audit 독립성 유지하며 오검출만 제거. 단순 det+area 게이트는 정상 57개도 버려 부적합.
- 대안: median 기반 통계(robust)로 그대로 두고 "GSAM2 mis-detection rate ~3%"를 별도 보고. → 10_eval 설계 시 확정.
