# 15 · 독립 Annotator 도구 정확도 하네스 (EllSeg/RITnet/Edge-Guided/DeepVOG/YOLOE/SAM3-I)

2026-07-01. GSAM2·U-Net 외의 `third/` 도구들을 **독립 audit 소스**로 붙여 동공 center 정확도를
사람 GT anchor와 비교. "바로 실험 가능"하게 런너·환경·집계기를 준비하고 스모크까지 검증.

## 1. 목적 / 계약
- 각 도구: 프레임(346×260) → **동공 이진 mask** → (annlib) **균일 ellipse-fit** → center(x,y) @346×260.
- 08 계약 미러링. 지표: `err=‖center−GT‖`(px), overall+모션별 mean/median/p95/p99/std, valid율,
  `radius_ratio = r_equiv/GT r_equiv`(>1.5 = 홍채/과분할 의심 — center는 [[docs/14]]대로 맹점).
- **공정성**: 모든 도구에 대해 mask→동일 ellipse-fit(annlib.mask_to_center) 적용 → 차이는 검출기 자체.

## 2. 구성 파일
| 파일 | 역할 |
|---|---|
| `scripts/annotators/annlib.py` | 공통: GT/anchor 로드, mask→center(최대CC+fitEllipse), CSV, `run_tool()` 하네스, numpy-2 shim |
| `scripts/annotators/run_ellseg.py` | EllSeg(ritnet_v3 DenseNet2D), enc→dec, seg class2=pupil |
| `scripts/annotators/run_ritnet.py` | RITnet 4-class, gamma+CLAHE, class3=pupil |
| `scripts/annotators/run_edge_guided.py` | BDCN edge + RITnet_v2, enc(img)+enc(edge)concat→dec, class2=pupil |
| `scripts/annotators/run_deepvog.py` | DeepVOG Keras U-Net, softmax ch1=pupil>0.5 |
| `scripts/annotators/run_yoloe.py` | ultralytics YOLOE(open-vocab), text "black pupil"→mask/box |
| `scripts/annotators/run_sam3i.py` | SAM3-I text-prompt(**GPU-deferred**, 9.8GB) |
| `scripts/annotators/_deepvog_model.py` | DeepVOG 모델정의 벤더링(1줄 keras2 호환 패치, third/ 미수정) |
| `scripts/annotators/run_all_annotators.sh` | 5개 full-eval 순차 실행 + 집계 |
| `scripts/15_eval_annotators.py` | 도구 CSV + U-Net/GSAM2 베이스라인 → 통합 비교표(`results/annotators/summary.md`) |

## 3. 환경 (2개 venv)
- **torch 4종(EllSeg/RITnet/Edge-Guided/YOLOE)** = 기존 `.venv-gsam2`(torch 2.5.1) + 추가 설치:
  `scikit-image`(EllSeg/Edge utils import), `scikit-learn`(동), `ultralytics`(YOLOE, 최초 mobileclip/가중치 자동 DL).
- **DeepVOG** = 신규 `.venv-deepvog`(**py3.11 + tensorflow-cpu 2.15.1**, `import keras`=keras2.15, numpy<2, opencv-headless).
  레거시 TF1/py3.7 불가(uv에 3.7 없음) → TF2/keras2로 우회.
- **SAM3-I** = `.venv-gsam2`+SAM3 deps이나 **실행 보류**: ckpt 9.8GB, 추론 ~14–18GB GPU 필요(현재 FACET 학습 점유).

## 4. 실행 함정(해결됨)
1. `.git_ok` 파일들 = 실제 PyTorch 체크포인트(이름만 회피용). 그대로 로드.
2. `.venv-gsam2`엔 **skimage/sklearn 없음** → EllSeg/Edge-Guided `utils.py`가 모듈 로드시 import → 설치.
3. **numpy 2.0**: `np.int`(EllSeg/Edge getSizes), `np.in1d`(Edge 후처리) 제거됨 → annlib에서 런타임 shim(int/float/bool/in1d).
4. **RITnet**: DenseNet2D 4-downsample → 입력 /16 필수. 네이티브 346×260 부적합 → **640×480(4:3,/16)** 리사이즈.
5. **EllSeg/Edge-Guided**: 전체 forward는 GT-dummy·center-of-mass·loss 계산이 keras/numpy2와 충돌 →
   **enc→dec만 직접 호출**해 seg 로짓만 취득(loss/ellReg tail 우회).
6. **DeepVOG/keras2.15**: 모델정의 `if X_jump==0`가 심볼릭 텐서면 깨짐(옛 keras는 bool 반환) →
   `_deepvog_model.py`로 벤더링 + `isinstance(X_jump,int) and ==0` 패치. 가중치는 명시 경로 로드, 패키지 __init__(skvideo) 회피.
7. **YOLOE**: `.pt`/`mobileclip_blt.ts`(572MB) 최초 자동 DL(네트워크 필요, 이후 캐시).

## 5. 스모크 검증 (앵커 2개: 정상 + GSAM2-홍채혼동 케이스)
| 도구 | median err(px) | radius_ratio | 비고 |
|---|---|---|---|
| RITnet | 0.57 | 0.98 | 가장 정확 |
| DeepVOG | 1.10 | 0.98 | conf~1.0 |
| EllSeg | 1.11 | 0.98 | |
| Edge-Guided | 1.20 | 0.99 | cross-dataset |
| YOLOE | 1.375 | 1.08 | box→ellipse, 약간 큼 |
| SAM3-I | — | — | GPU 대기(보류) |

**교차검증 발견**: GSAM2가 **홍채로 오분할**(ratio 2.72)한 프레임에서 **전용 도구 5종 모두 동공 정확 검출**
(ratio 0.97–1.08) → 홍채혼동은 GSAM2 text-prompt 방식 고유 문제. 전용 seg 도구는 4클래스 구조로 pupil을 직접 분리.

## 6. 실행 방법 (자동스케줄 없음 — 수동 실행)
```bash
cd analysis/annotation-analysis
# 개별 (예: RITnet 전체 483앵커, CPU)
.venv-gsam2/bin/python scripts/annotators/run_ritnet.py --device cpu
# 스모크: --keys "<key1>,<key2>"  전체가중치 대안: env RITNET_W=.../ritnet_pupil.pkl
# 5종 일괄 + 집계
bash scripts/annotators/run_all_annotators.sh cpu      # 또는 cuda(여유 시)
# 집계만
.venv-gsam2/bin/python scripts/15_eval_annotators.py   # -> results/annotators/summary.md
# DeepVOG(별도 env)
.venv-deepvog/bin/python scripts/annotators/run_deepvog.py --device cpu
# SAM3-I (>=20GB GPU 여유 시)
.venv-gsam2/bin/python scripts/annotators/run_sam3i.py --device cuda --prompt "pupil"
```
공통 옵션: `--device cpu|cuda`, `--limit N`, `--good-only`, `--keys k1,k2`, `--save-masks`, `--out path.csv`.

## 7. 결과물
- `results/annotators/<tool>.csv` (도구별 per-anchor: err/radius_ratio/valid/…)
- `results/annotators/summary.md` (통합 비교표: 전체+모션별, U-Net/GSAM2 베이스라인 포함)
- `results/annotators/masks/<tool>/…`(옵션)

## 8. 한계·주의
- 도구 전부 **off-domain**(EllSeg/RITnet/Edge-Guided=타 근적외 세트, DeepVOG=타 근적외, YOLOE/SAM3=RGB) → **cross-dataset audit**, EV-Eye 튜닝 아님.
- center는 홍채혼동 맹점 → **radius_ratio 병행 필수**([[docs/14]]).
- DeepVOG=GPLv3, Edge-Guided=라이선스 없음 → 인용/사용 시 확인.
- 공정 비교 위해 native ellipse-head(EllSeg/Edge/DeepVOG) 대신 **균일 mask-fit** 사용 — 도구 강점 일부 미반영 가능(후속 옵션).
