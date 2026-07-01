# 01 · 진행상황 & 향후 계획

갱신: 2026-07-01

## 파이프라인 단계별 상태

```mermaid
flowchart LR
  A[분석·설계] --> B[07 수집] --> C[07b 이벤트수정] --> D[08 GSAM2] --> E[09 HBTXR pred] --> F[10 평가] --> G[리벗]
```

| 단계 | 스크립트 | 상태 | 비고 |
|---|---|---|---|
| 데이터셋·코드 분석 | 보고서 01~03 | ✅ 완료 | v2e + EV-Eye + 라벨 파이프라인 |
| 평가 방법론 설계 | COLLECTION_PLAN_v2 | ✅ 완료 | GT 동결·audit 분리·3개념 분리 |
| 샘플 수집 | `07_collect_samples.py` | ✅ 완료 | 483 윈도우(모션당 161), users 1–10 |
| 이벤트 재슬라이스 | `07b_reslice_events.py` | ✅ 완료 | 4필드 파싱 버그 수정, n_events 채워짐 |
| GSAM2 audit | `08_run_gsam2.py` | ✅ **완료(2026-07-01)** | 483 anchor valid 100%·median **0.75px**, **전 4,669 valid 99.9%**+마스크저장. prompt `"black pupil."`+과대박스필터. 상세 [07](07_gsam2_audit_results.md) |
| per-frame 트리·오버레이 | `11_build_perframe.py`·`12_overlay.py` | ✅ **완료** | `samples/perframe/`(4,669 프레임별 gt/unet/gsam2 center·mask·bbox) + `overlay/`. 상세 [08](08_perframe_dataset.md) |
| annotator 도구 서베이 | `annotators/` | ✅ **완료** | 6종(EllSeg·RITnet·DeepVOG·Edge-Guided·X-AnyLabeling·SAM3-I) 분석+비교. 추가 audit 후보 |
| HBTXR 예측 주입 | `09_inject_pred.py` | ✅ **완료(2026-07-01)** | subject-independent **img64** ckpt(G=16, pre_process 64×64 수정). 전 4,669 valid 89.7%. **E_orig median 5.70px** vs 사람 GT(보고 0.18px의 ~30배). 상세 [09](09_hbtxr_pred_results.md) |
| 평가(Label Noise/Uncertainty) | `10_label_noise.py` | 🔨 **부분 완료** | GT-vs-U-Net/GSAM2 mean/median/p95/p99/std(전체+모션별) → `results/label_noise_*.md`+`plots/`. E_i/U_i(pred)·Spearman 미작성 |
| GSAM2 mislabel 분석·게이트 | `13_gsam2_mislabel.py`·`14_flag_gsam2_mislabel.py` | ✅ **완료** | ~3% 오검출(반가림→하단마커) 규명, U-Net 교차검증 게이트로 mislabel 플래그. [10](10_gsam2_mislabel_analysis.md) |
| EV-Eye 데이터셋 provenance | (조사) | ✅ **완료** | U-Net=EV-Eye 확인, data/label 개수, 세션 커버리지(프레임6/predict4/사람3), predict 개수 불일치(user30 중복) 규명. [11](11_eveye_dataset_provenance.md) |
| 리벗 작성 | — | ⛏ 대기 | 10 결과로 표·문구 조립 |

## 지금 당장 할 것 (Now)
1. ✅ 08 GSAM2 전 프레임 + per-frame 트리(`samples/perframe/`) + 오버레이(`overlay/`)
2. ✅ **09 HBTXR 전 프레임 완료** → `pred.json`, **E_orig median 5.70px** vs 사람 GT(label-noise 0.75px보다 훨씬 큼)
3. 🔨 **10_eval 진행 중** — Label Noise/Uncertainty(GT vs U-Net/GSAM2, 전체+모션별) 산출 완료(`results/label_noise_*`). 남음: E_i/U_i(pred)·Spearman·CDF/scatter·subject-level cluster-bootstrap CI
4. ✅ GSAM2 오검출 ~3% → **U-Net 교차검증 게이트 적용**(mislabel 플래그, [10](10_gsam2_mislabel_analysis.md)). 미결: 0.1812 원 config/mode 재계산, 추가 audit(EllSeg/RITnet — `weights/`에 수집됨) 투입

## 다음 (Next)
- ✅ 08 전 프레임 확장 완료. 09 전 프레임 확장은 subject-independent ckpt 확보 후
- 10으로 3표 + `E_i/U_i`/Spearman + CDF/scatter + subject-level cluster-bootstrap CI
- Corrected primary accuracy(frame/event/hybrid) 재계산
- 리벗 단락·표·그림 패키징

## 나중 (Later / 정밀도 향상)
- **정확한 dense-label 대비 수치**: E: 사본 대신 학습머신의 **DeanDataset_full_unet 실제 라벨**을 anchor 프레임에 붙여 `y_unet` 교체(현재는 공식 Data_davis_predict를 proxy로 사용)
- saccade 균형(161)이 작으면 users 확대(현재 1–10) 또는 `--vsac` 조정으로 표본 증대
- (선택) 사람 재주석이 가능해지면 Annotation Precision을 자동 proxy → 사람 inter/intra로 승격

## 미해결/확인 필요
- ✅ (해결) 09 ckpt=`hbtxr-imgsz64`(subject-independent, G=16); 08 valid rate 99.9%; y_unet=EV-Eye U-Net 확인(11).
- **0.1812 원 config/mode 확정**(frame/event/hybrid) → 그 ckpt로 E_orig 재계산(비교 완결)
- 정확한 dense-label 대비: 학습머신 `DeanDataset_full_unet` 실제 라벨로 y_unet 교체(현재 공식 Data_davis_predict proxy)
- 데이터 gotcha: predict `user30` 중복·`1_0_1` 미완성은 **분석 제외만, 원본 미삭제**(11)
