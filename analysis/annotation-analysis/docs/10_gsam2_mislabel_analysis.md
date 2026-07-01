# 10 · GSAM2 mislabeling 분석 (비율·원인·해결책)

2026-07-01. 전 4,669 프레임(마스크 포함) GSAM2 결과 대상. 스크립트 `scripts/13_gsam2_mislabel.py`(분석) + `scripts/14_flag_gsam2_mislabel.py`(게이트 적용). 시각: `overlay/misdetect/`.

측정 방식: **anchor(483)는 사람 GT 대비**(gold), **전 프레임은 U-Net dense center 대비**(audit-vs-audit proxy). 둘 다 동공에서 멀면 gross mislabel.

## (1) Mislabeling 비율
| 기준 | 지표 | 비율 |
|---|---|---|
| **Gold** (anchor vs 사람 GT) | `e_orig > 10px` | **14 / 483 = 2.90%** |
| **Proxy** (전 프레임 vs U-Net) | `e_unet > 15px` | **160 / 4,657 = 3.44%** |
| 게이트 적용 후 플래그 | mislabel | 161 / 4,666 = 3.45% |
- gold 14건은 전부 `e_orig>30px` = **완전히 다른 물체**를 잡은 gross 오검출(부분 오차 아님).
- **proxy 검증**: anchor에서 `e_unet>15`가 `e_orig>10`을 **완벽 예측**(TP=14, FP=0, FN=0 → recall/precision 1.00). ⇒ 전 프레임 3.44% 추정은 신뢰 가능.
- 나머지 **~96.6%는 정상**(‖gsam2−사람GT‖ median 0.75px, [07](07_gsam2_audit_results.md)).

## (2) Mislabeling 원인
**정량 특징 (mislabel vs good, anchor):**
| | det_score | mask area | y-position(cy) | box_w |
|---|---|---|---|---|
| MISLABEL(n=14) | 0.317 | 266 | 188.6 | 15.5 |
| GOOD(n=469) | 0.445 | 779 | 122.8 | 31.8 |
- **14/14가 tiny(area<400) AND lowdet(det<0.35)** — 작고 저신뢰. 위치는 중앙(cy~123)보다 아래(cy~189).

**분포:**
- 모션: **fixation 6.8%(11/161)** ≫ saccade 0.6%(1) · smooth 1.2%(2). → 거의 fixation.
- 피험자: **user8(6) + user7(5) = 14건 중 11건**. 소수 피험자 집중.

**근본 원인 (육안 확인, `overlay/misdetect/`):**
- **눈이 거의 감긴/눈꺼풀에 심하게 가려진 프레임**에서 동공이 얇은 조각만 보임 → `"black pupil"` 프롬프트가 **확신할 동공 박스를 못 찾고**, 프레임 내 다른 어두운 물체 = **하단 캘리브레이션 마커 기둥**(또는 속눈썹/눈꺼풀 그림자)을 대신 검출.
- U-Net(EV-Eye 도메인 학습)은 같은 프레임에서 조각 동공을 정확히 찾음(uNet err ~2px) → **도메인 갭**이 근본. 일반 목적 GSAM2는 근적외 반쯤 감긴 눈에 취약.
- user7/8은 **좁은 검열(palpebral fissure)**로 이런 반가림 프레임이 많아 집중됨. fixation은 저속도 anchor 선정 특성상 반가림/정지 프레임 비율이 높음.

## (3) 해결책
**게이트 평가 (anchor gold, mislabel=`e_orig>10`):**
| 게이트 | mislabel 제거 | good 오제거 |
|---|---|---|
| **U-Net 교차검증 `e_unet>15`** | **14/14** | **0/469** ⭐ |
| y-pos `cy>200` | 2/14 | 0/469 |
| `det<0.35` | 14/14 | 25/469 |
| `area<400` | 14/14 | 40/469 |
| `det<0.35 & area<400` | 14/14 | 7/469 |

- **채택: U-Net 교차검증 게이트** — `‖y_gsam2 − y_unet‖ > 15px`면 mislabel. **오검출 전부 제거 + 정상 0개 오제거**(단순 det/area 게이트는 정상 25~40개도 버림). 사람 GT 미사용이라 **audit 독립성 유지**. U-Net 무효 프레임엔 fallback(`det<0.35 & area<400`).
- **적용됨**: `14_flag_gsam2_mislabel.py`로 `gsam2.json`·perframe `gsam2/center.json`에 `mislabel`/`mislabel_reason`/`e_unet` 필드 추가(비파괴). 결과 **161건(3.45%) 플래그**(unet_xcheck 160 + lowconf_tiny 1). → **10_eval에서 label-noise floor 산정 시 mislabel=True 제외**.

**보조 완화책(선택):**
- (a) 하단 마커 **ROI 마스킹**(프레임 하단 밴드 검출 제외) — 마커 오검출 원천 차단.
- (b) **시간적 일관성**: 윈도우 내 이웃 median center에서 크게 튀는 프레임 제외.
- (c) **SAM2 point-prompt fallback**: 박스 검출 실패 시 U-Net center를 point로 SAM2 실행(단 GSAM2 독립성↓ → audit 대신 보정용).
- (d) 반가림/blink 프레임은 **blink 정책(G6)으로 별도 제외**(mislabel과 구분해 리포트).
- (e) 프롬프트 네거티브("not a marker")는 GroundingDINO에서 효과 제한적 → 비권장.

## 결론
GSAM2 mislabel은 **~3%**, 전부 **반가림 동공 → 마커/어두운 blob 오검출**(도메인 갭). **U-Net 교차검증 게이트**로 정상 손실 0으로 완전 분리·플래그 완료. audit floor(median 0.75px)는 이 3%를 제외하면 더욱 견고.
