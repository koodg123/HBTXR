# 12 · GSAM2 검출 harness 설계 (U-Net fallback 제거, 제대로 검출)

2026-07-01. 목표: GSAM2가 ~3% 실패 프레임에서도 **U-Net에 의존하지 않고** 동공을 스스로 검출. 데이터 근거 + 문헌 조사(하단 Sources) 종합.

## 1. 문제 재정의 (데이터 근거)
- 실패 모드: **반쯤 감긴/가려진 동공** → `"black pupil"` 텍스트가 확신할 동공 박스를 못 찾고 **하단 캘리브레이션 마커**(어두운 기둥)를 검출. det↓(~0.32)·area↓(~266).
- **공간 분리(핵심)**: 동공 center **cy ≤ 177**(p99 162), 마커 center **cy ≥ 161**(median 189). 동공 cx 85–265, cy 61–177. → **하단 밴드를 시야에서 제거하면 마커 오검출 원천 차단**.

## 2. 문헌 요약 (Sources)
- **Grounding DINO**: 단일 객체 프롬프트 최선, **작은 박스 size-threshold 제거**, box/text/NMS threshold 튜닝. (learnopencv, roboflow, GD1.5)
- **SAM2 프롬프트**: **box ≫ point**(정확도), **negative point로 오영역 배제**, 박스 과확장 시 배경노이즈↑. (RP-SAM2, SAMAug, Ultralytics)
- **Staged SAM pupil (MDPI Electronics 2025)**: `SAM-BaseIris → SAM-RefinedIris(자동 bbox) → SAM-RefinedPupil` = **계층 iris→pupil**.
- **SAM2 zero-shot pupil, 14M (arXiv 2410.08926)**: 동공중심 근처 **point 1개** → IoU 90–93%. 실패는 수동 보정.
- **SAM eye features zero-shot (arXiv 2311.08077)**: **automatic mask + mask-matching(기하 기준 선택)**, BBOX+pos/neg point 조합, pupil IoU 93%.

## 3. Harness 후보 (다각도)
| # | harness | 방식 | 마커 배제 | 독립성(비U-Net) | 비용 | 근거 |
|---|---|---|---|---|---|---|
| **A. Eye-ROI 크롭** | 검출 전 눈 영역만 크롭 | A1 고정 ROI(데이터 분포) / A2 GDINO "eye"→크롭 | ✅ 강함(밴드 제거) | ✅ | 낮음 | 데이터(마커 cy≥161) |
| **B. 계층 eye→iris→pupil** | 큰 구조로 좁혀 검출 | GDINO eye→iris박스→내부 pupil | ✅ | ✅ | 중 | MDPI staged SAM |
| **C. 기하 선택** | argmax 대신 위치·어둡기·둥글기·크기로 박스 선택 | min+max size 필터 + ROI내 darkest/round | ✅(마커=하단·소형) | ✅ | 낮음 | GD size-thr, mask-matching |
| **D. SAM2 auto-mask + 선택** | ROI에 AutomaticMaskGenerator → 동공(어둡+둥글+면적) 선택 | ✅ | ✅(텍스트 도메인갭 우회) | 중 | arXiv 2311.08077 |
| **E. Negative prompt** | SAM2에 ROI밖/마커에 negative point | 보강 | ✅ | 낮음 | SAM2 best practice |
| **F. Point prior(비U-Net)** | ROI내 darkest-blob/iris중심 → SAM2 point | — | ✅ | 낮음 | arXiv 2410.08926 |

## 4. 권장 harness (cascade — 독립 검출 + 내재적 복구)
U-Net을 전혀 안 쓰고 실패를 내부에서 복구:
```
1) Eye-ROI 크롭  : 고정 ROI(예: x[30,320] y[15,195], 전 동공 포함·하단 마커 제외)
                   또는 A2(GDINO "eye"→박스+마진)로 적응적 크롭
2) GDINO "black pupil." (ROI 내) : box_thr/text_thr↓ + **min&max size 필터**
                   (tiny=마커, whole=눈전체 제거) → 후보 박스들
3) 기하 선택(C)  : ROI중앙 근접 + 내부 평균밝기 낮음(어두움) + 종횡비~1 로 점수화(argmax conf 대신)
4) SAM2 box→mask + negative point(ROI 경계) → ellipse-fit center
5) 내재적 복구(비U-Net): (2)가 비면 → SAM2 auto-mask(D)에서 어둡+둥근 blob 선택,
                   또는 iris 먼저 검출(B) 후 그 안에서 pupil
```
- **검증**: 기존 anchor gold(사람 GT)로 재평가 — 목표: mislabel 14/14 회복(U-Net fallback 없이), 정상 프레임 정확도 유지(median ~0.75px).
- **비교**: 현재(fallback) vs 새 harness의 `‖y_gsam2−y_orig‖` 및 valid rate.

## 5. 정직한 한계
- ~3% 중 일부는 **거의 감긴 눈(near-blink)**으로 동공이 물리적으로 거의 안 보임 → "정확 검출"이 ill-defined. 이런 프레임은 **blink 정책(G6)으로 분류·제외**가 더 타당할 수 있음(강제 검출 대신). harness는 "가려졌지만 동공이 보이는" 케이스를 회복하는 게 목표.
- 고정 ROI는 이 near-eye 고정 카메라(DAVIS346)에 유효. 카메라/피험자 정렬이 크게 바뀌면 A2(적응적) 필요.

## 6. 구현 우선순위 (제안)
1. **A1(고정 ROI 크롭) + C(min-size 필터·기하 선택)** — 가장 싸고 효과 큼(마커 원천 차단). 먼저 구현·검증.
2. 부족하면 **D(auto-mask 복구)** 또는 **B(iris→pupil)** 추가.
3. `08_run_gsam2.py`에 `--roi`, `--min-box`, `--geom-select`, `--automask-fallback` 옵션으로 단계적 추가.

## Sources
- Grounding DINO: learnopencv.com/fine-tuning-grounding-dino, blog.roboflow.com/grounding-dino-zero-shot-object-detection, arXiv:2405.10300
- SAM2 prompts: arXiv:2504.07117(RP-SAM2), arXiv:2307.01187(SAMAug), docs.ultralytics.com/models/sam-2
- Pupil/eye + SAM: MDPI Electronics 14(9):1850(Adapting SAM for pupil), arXiv:2311.08077(zero-shot eye features), arXiv:2410.08926(SAM2 pupil 14M), arXiv:2410.06131(unsupervised eye-region)
