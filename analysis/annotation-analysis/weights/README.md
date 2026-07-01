# Grounded-SAM-2 weights

이 폴더에 Grounded-SAM-2(= Grounding DINO + SAM 2) 가중치를 받습니다.
(제가 세션에서 직접 못 받는 이유: 셸 비활성 + 파일도구는 대용량 바이너리 기록 불가. 그래서 스크립트로 드립니다.)

## 받는 법
```bash
cd /home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis/weights
bash download_weights.sh          # 기본: SAM2.1-large + GDINO Swin-T (~1.5GB)
# bash download_weights.sh fast   # SAM2.1-tiny + GDINO Swin-T (빠름/작음)
# bash download_weights.sh all    # 모든 변형(~4GB+)
```

## 받아지는 파일
| 파일 | 역할 | 대략 크기 | 출처 |
|---|---|---|---|
| `groundingdino_swint_ogc.pth` | 텍스트→박스 검출("pupil") | ~660MB | IDEA-Research/GroundingDINO release v0.1.0-alpha |
| `sam2.1_hiera_large.pt` | 박스→마스크 분할 | ~900MB | Meta segment_anything_2 (092824) |

## 설정(config) 파일은 여기 없음 — 코드 패키지에 들어있음
가중치(.pt/.pth)만 여기 두고, **config는 설치한 파이썬 패키지에서** 옵니다:
- SAM 2.1: `sam2` 패키지의 `configs/sam2.1/sam2.1_hiera_l.yaml`
- Grounding DINO: `groundingdino` 패키지의 `config/GroundingDINO_SwinT_OGC.py`

설치(별도 GPU env 권장):
```bash
pip install "git+https://github.com/IDEA-Research/Grounded-SAM-2.git"  # 또는 repo clone 후 설치
# 내부적으로 sam2 + groundingdino 설치됨
```

## 08_run_gsam2.py 가 기대하는 경로
```
--gdino-ckpt  .../weights/groundingdino_swint_ogc.pth
--sam2-ckpt   .../weights/sam2.1_hiera_large.pt
--sam2-cfg    configs/sam2.1/sam2.1_hiera_l.yaml           (패키지 상대경로)
--gdino-cfg   <groundingdino>/config/GroundingDINO_SwinT_OGC.py
```

## 주의
- 이 가중치(~1.5GB)는 git에 올리지 마세요 → 프로젝트 `.gitignore`에 `analysis/annotation-analysis/weights/*.pt`, `*.pth` 추가.
- 다운로드가 중단되면 스크립트를 다시 실행하면 이어받습니다(`wget -c` / `curl -C -`).
