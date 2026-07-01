# 08 · per-frame 트리 + 오버레이 + annotator 서베이

생성: 2026-07-01. 윈도우 단위 `samples/label/*`를 **프레임(이미지) 단위** 트리로 재구성하고, QA 오버레이와 추가 annotator 도구 조사를 정리.

## 1. per-frame 트리 (`samples/perframe/`)
빌드 스크립트: `scripts/11_build_perframe.py` (`--out ../samples`, CPU 전용, 재실행 안전). 소스: `samples/frame/`, `samples/label/{key}/{gt,unet_dense,gsam2,pred}.json`, `unet_masks/*.gif`, `gsam2_masks/*.png`.

```
samples/perframe/{key}/
├─ window.json                         # 윈도우 메타(manifest_windows 행)
└─ {idx6}_{ts}/                        # 이미지 이름별 디렉토리
   ├─ frame.png            # 원본 APS 프레임 복사(346×260)
   ├─ meta.json            # key,idx,ts,role(anchor|neighbor),motion,user,eye,session
   ├─ gt.json              # 사람 GT 타원          (anchor만)
   ├─ gt_bbox.json         # {xyxy, source} 타원→AABB (anchor만)
   ├─ unet/  center.json · mask.png(0/255) · bbox.json{xyxy,source:unet_mask}
   ├─ gsam2/ center.json · mask.png(0/255) · bbox.json{xyxy:GDINO박스, mask_xyxy, source}
   └─ pred/  center.json   # 09 이후 자동 합류
samples/perframe/index.csv             # 프레임별 소스 존재 플래그
```
- **center.json**: unet=`{cx,cy,area,valid}`; gsam2=`{cx,cy,area,det_score,sam_score,repeats,valid}`.
- **마스크**: 전부 **0/255 이진 PNG(346×260)**. unet은 gif→png 변환, gsam2는 08 `--save-masks` 산출 복사.
- **bbox**: unet=마스크 tight box; gsam2=`xyxy`(GDINO 검출박스)+`mask_xyxy`(마스크 tight box); gt=타원 AABB.

### 빌드 결과 (2026-07-01)
- frames=**4,669**, windows=483. `has_gt=483`(anchor 전부), `has_unet=4,659`, `has_unet_mask=4,659`, `has_gsam2=4,666`, `has_gsam2_mask=4,666`, `has_pred=0`(09 대기).
- 미보유 소수: U-Net 무효 10, GSAM2 무효 3(블링크/가림). anchor에만 gt/gt_bbox.

### 로드 예
```python
import json, cv2, csv, os
PF="samples/perframe"
rows=list(csv.DictReader(open(f"{PF}/index.csv")))
d=f"{PF}/{rows[0]['key']}/{rows[0]['stem']}"
gc=json.load(open(f"{d}/gsam2/center.json"))          # {cx,cy,...}
gm=cv2.imread(f"{d}/gsam2/mask.png",0)                # 0/255
```

## 2. 오버레이 QA (`scripts/12_overlay.py`)
`--perframe samples/perframe --overlay-out ../overlay`. 프레임 위에 **GT(초록 타원+center+bbox) / U-Net(파랑 mask윤곽+center+bbox) / GSAM2(빨강 …)**를 그려 저장(×3 확대, cv2 필요 → `.venv-gsam2`로 실행).
- 선택: 모션별 최저오차 anchor(정상) + 전체 최고오차 anchor(오검출). 파일명 `{good|bad}_{motion}_{gsam2err}px_{key}__{stem}.png`.
- 산출: `analysis/annotation-analysis/overlay/`에 15장. 육안 확인 완료 — 정상은 3소스 동공 정합(err 0.06–0.17px), 오검출은 GSAM2가 하단 마커로 이탈(err 89–125px)함이 시각적으로 확인됨.

## 3. annotator 도구 서베이 (`annotators/`)
GSAM2 외 **독립 동공 검출/주석 도구** 6종 조사(각 상세 md + 비교표). 상세: [`../annotators/README.md`](../annotators/README.md).
- 추가 audit 소스 최우선: **EllSeg / RITnet-Plugins**(MIT·가중치 in-repo·근안 전용·Low–Med). 다음 **DeepVOG**(GPLv3·레거시 TF).
- **X-AnyLabeling**: SAM 계열 다수(SAM/SAM2/SAM3/SAM-HQ/…)+GroundingSAM2, mask→COCO/PNG export. **SAM3-I**: text 명령튜닝이나 가중치 gated+RGB갭(후순위).
- 통합 시 08 출력 계약(프레임→mask→ellipse-fit center, 346×260) 미러링 → `y_orig/y_unet/y_gsam2` 옆 4번째 center로 투입.

## 4. 재현
```bash
# per-frame 빌드
python scripts/11_build_perframe.py --out samples          # (cwd=scripts면 --out ../samples)
# 오버레이(cv2 필요)
.venv-gsam2/bin/python scripts/12_overlay.py --perframe samples/perframe --overlay-out overlay
```
