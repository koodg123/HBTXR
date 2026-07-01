# 02 · 결정사항 로그 (Decision Log)

각 결정은 "무엇을/왜"로 기록. 시간순.

## D1. 세 개념을 분리 보고한다
- **결정**: Annotation Precision(반복정밀도) / Label Noise(GT의 진실 대비 오차) / Label Uncertainty(sample별 GT 신뢰도)를 각각 다른 표로.
- **이유**: 리뷰어가 세 용어를 구분해 요구. 섞으면 재지적됨.

## D2. Test GT = 원래 EV-Eye 사람 라벨로 동결
- **결정**: 평가 GT는 VIA 타원/마스크(9,011 키프레임)로 고정. U-Net·Grounded-SAM2는 **audit 전용**(GT 교체 금지).
- **이유**: U-Net 예측을 GT로 쓰면 순환(U-Net은 사람 마스크로 학습됨). label noise = "사람 GT 대비 U-Net/GSAM2 불일치".

## D3. 0.1812는 dense 라벨 기준 결과로 규정, 사람 GT로 재계산해 나란히 제시
- **결정**: 0.1812(=DeanDataset_full_unet dense 의사라벨 기준)를 main에서 내리고, 원래 사람 GT 기준으로 재계산해 나란히 보고.
- **이유**: 0.1812는 dense(U-Net)라는 특정 reference 기준 값이고 sub-pixel은 사람 정밀도보다 작음. 사람 GT라는 **다른 reference**로 재계산. ⚠️ **누수로 단정하지 않음**(사용자 정정 2026-07-01).

## D4. 사람 재주석 불가 → 자동 정밀도 proxy
- **결정**: 사람 inter/intra-annotator precision은 측정 불가. 대신 ① **fixation 구간 F2F center jitter**(참 운동≈0 → 변동=측정노이즈) ② **GSAM2 perturbation(박스 지터/TTA) 산포** 로 대체하고 한계 명시.
- **이유**: 사용자가 사람 재주석 불가라고 확정.

## D5. GT-앵커 연속 윈도우로 수집
- **결정**: 각 윈도우를 사람 GT 키프레임에 **중앙 앵커**, 양옆 K-1 연속 프레임. anchor=사람 GT, 이웃=U-Net/GSAM2/pred(예측만).
- **이유**: 사람 GT가 희소 → 연속 GT 시퀀스는 불가. 앵커로 진실 비교 + 윈도우로 시간 jitter 둘 다 확보.

## D6. 윈도우 길이 K
- **결정**: fixation/smooth K=11, saccade K=7.
- **이유**: 25fps에서 saccade는 1~3프레임뿐 → 짧게. 연속성은 이벤트 스트림에서.

## D7. 모션 층화 + saccade 균형
- **결정**: fixation/saccade/smooth 3종으로 층화(세션이 아닌 **U-Net dense center 속도**로 분류), **saccade 가용량에 맞춰 균등**(모션당 161).
- **이유**: label noise는 모션/블링크에 좌우. saccade가 희소해 균형 기준으로 삼음.

## D8. 모션 분류 robust화
- **결정**: 블링크 프레임 제외(U-Net 마스크 면적 붕괴) + fixation은 **median 속도**, saccade는 max 속도, 임계 `vfix=1.2/vsac=6.0`.
- **이유**: 초기(vfix0.6/max) 결과가 fixation18/saccade74로 역전 → 블링크·지터가 saccade로 오분류. 수정 후 fixation306/saccade161로 정상화.

## D9. 좌표계 통일(346×260)
- **결정**: 모든 소스 center를 원본 APS **346×260 px**로 통일. 모델은 grid(=img_size/patch_size)에서 예측 → `x*346/G, y*260/G` 환산.
- **이유**: 형상차·해상도차가 noise로 위장되는 것 방지.

## D10. 이벤트 포맷 수정
- **결정**: `events.txt`는 `ts x y pol` **4필드**(맨 앞이 timestamp). 파서/슬라이서 수정, 07b로 이벤트만 재생성.
- **이유**: 이전에 5필드(맨 앞 index)로 오해 → Read 도구의 줄번호 접두어를 데이터로 착각. n_events=0 버그의 원인.

## D11. GSAM2 audit-only + 정밀도 proxy 겸용
- **결정**: Grounding DINO("pupil.")→SAM2→ellipse center = y_gsam2. `--repeats`(박스 지터)+`--tta`로 정밀도 proxy.
- **이유**: 독립 audit + 결정론 모델의 정밀도 측정을 위한 변동 주입.

## D12. subject-independent ckpt로만 E_i
- **결정**: 09는 users 1–10을 **test로 제외한** ckpt로 실행.
- **이유**: 학습에 포함된 ckpt면 E_i가 낙관 편향.

## D13. 산출물 저장 위치
- **결정**: 최초 요청 `E:\DATASET\codes\annotation-analysis`는 마운트 불가 → 프로젝트 내 `analysis/annotation-analysis/`에 저장.
- **이유**: 세션의 UNC 마운트 문제로 E:\DATASET\codes 마운트 실패.

## D14. GSAM2 프롬프트 `"black pupil."` + 과대박스 필터 (2026-07-01)
- **결정**: 08의 프롬프트를 `"pupil."`→`"black pupil."`로, `gdino_box`에 `--max-box-frac 0.55`(폭·높이>55%*이미지 박스 제거) 추가.
- **이유**: `"pupil."`은 GroundingDINO top-1 박스가 **이미지 전체**(동공은 2등)라 argmax가 실패. `"black pupil."`은 top-1이 정확히 동공(det~0.47). 필터는 전체이미지·눈전체 박스 안전 제거. → valid 100%, median 0.75px.

## D15. GSAM2 오검출 tail 처리 (미확정 — 10_eval에서 결정)
- **상황**: 483 중 ~3%(14개)가 err>10px 오검출(눈꺼풀 가림/블링크 프레임에서 **하단 마커 기둥** 오검출). label-noise가 아니라 검출실패.
- **권장**: `‖y_gsam2 − y_unet‖ > ~15px`면 GSAM2 anchor invalid 처리(사람 GT 미사용 → audit 독립성 유지). 단순 det+area 게이트는 정상 57개도 버려 부적합.
- **이유**: floor 산정에서 검출실패를 배제해야 label-noise가 과대추정 안 됨. median 통계는 robust하나 mis-detection rate는 별도 보고.
