# 03 · 세션 로그 & 대화 흐름

세션 요약(2026-06-30~07-01). 질문→판단→조치 순.

## A. 분석 착수
- 요청: `third/` 코드베이스 + `E:\DATASET\eveye` 데이터셋 심층 분석, 산출물은 `E:\DATASET\codes\annotation-analysis`.
- 발견: `third/`엔 **v2e 하나**(video→DVS 이벤트 시뮬레이터). `eveye`는 **EV-Eye**(48명×양안×4세션, DAVIS346, Tobii GT).
- 조치: 보고서 3종(01 v2e / 02 EV-Eye / 03 평가) + 정량 툴킷(scripts 01~06) 작성. **환경 제약**으로 bash 불가 → 스크립트로 제공.
- 저장위치 이슈: `E:\DATASET\codes` 마운트 불가 → 프로젝트 폴더로 저장. 사용자가 `analysis/annotation-analysis/`로 이동.

## B. 리뷰어 코멘트 대응 설계
- 사용자가 리뷰어 코멘트(annotation precision/label noise) + 초안 플랜 제시 → 평가.
- 판단: 초안의 3결함(① U-Net을 GT로 씀=순환, ② 결정론 모델 5회반복=정밀도 아님, ③ 단위 불일치) 지적.
- 사용자가 **더 정교한 플랜**(GT 동결·audit 분리·3개념 분리·0.1812 재계산) 제시 → 이게 더 우수하다고 확인, 8개 보강점 추가(anchoring bias, center 추출 단일화, human subset 필수, cluster bootstrap, blink 정책, split 명시, 헤드라인=median E vs σ, EV-Eye 제약).

## C. 확정된 전제
- 사용자 답변: **① 0.1812는 Grounded-SAM/U-Net dense 라벨 기준, ② 사람 재주석 불가.**
- → GT=원래 EV-Eye 사람 라벨 동결, precision은 자동 proxy로 대체.

## D. 데이터 실체 규명
- 라벨 위치 논의 → `Data_davis`의 `user_N.csv` 직접 확인: 프레임당 1행이나 **키프레임만 타원**(나머지 `region_count=0`).
- `DeanDataset_full_unet`/`DavisWithMaskDataset_labelled_subset` 정체 규명(코드·리포트 grep): full_unet=**전 프레임 U-Net 의사라벨(dense)**, labelled_subset=**9,011 사람 키프레임(sparse)**. → 0.1812는 dense 라벨 기준 값으로 규명.

## E. 수집 설계 & 실행
- COLLECTION_PLAN → v2(라벨 3-스트림: y_orig/y_unet/y_gsam2/y_pred). 모션·윈도우·앵커·이벤트 정의.
- 파라미터 확정: K=11(saccade 7), 모션당 목표 45/45/70 → **saccade 균형**, test-users **1–10**, GT 키프레임 중앙 앵커, 윈도우 내 전 이벤트 수집.
- dry-run(3명) → 모션 역전 발견 → **robust 분류**(블링크 제외·median 속도·임계 상향)로 수정.
- **본수집 실행** → 483 윈도우(모션당 161). 단 **n_events=0** 발견.
- 진단: `events.txt`가 4필드(ts x y pol)인데 파서가 5필드로 가정 → 전 줄 skip. **07b로 이벤트만 재슬라이스** → 정상(예: 7,945/8,910/11,712).

## F. 후속 파이프라인 준비
- GSAM2 가중치: 다운로드 스크립트(`weights/download_weights.sh`) 제공 → 사용자 다운로드 완료.
- HBTXR 추론: FACET 코드(`references/codebase/software/FACET`) 조사 → `event_to_frame→pre_process→model→post_process→center(64그리드)→346×260` 경로 확인.
- `08_run_gsam2.py`, `09_inject_pred.py` 작성 완료. 사용자가 GSAM2/HBTXR를 **tmux/WSL**에서 실행하기로(Claude는 셸 불가).

## G. 환경 논의
- 왜 bash 불가? → 연결 폴더가 `\\wsl.localhost\` **UNC**라 샌드박스가 마운트 실패. 다른 대화가 되는 건 그쪽이 정상 경로라서.
- 네이티브 경로로 쓰려면 **WSL 터미널에서 `claude`(CLI) 직접 실행**(GUI 대신 CLI). 또는 프로젝트를 윈도우 드라이브로 복사.

## H. GSAM2 실행 (2026-07-01, Claude 직접)
- 환경 재점검: 현재 세션은 **WSL 네이티브 CLI라 bash 동작**(과거 docs의 "bash 불가"는 Cowork UNC 세션 한정). GPU RTX 4070 Ti 확인.
- `third/`가 v2e 빼고 **전부 빈 디렉토리** 발견 → 사용자 요청으로 6개 repo clone(Grounded-SAM-2, SAM3-I=`debby-0527/SAM3-I`, timelens=`ztysdu/timelens`, TimeLens-XL, X-AnyLabeling).
- GSAM2 설치: `.venv-gsam2`(torch cu121) + sam2 + groundingdino. gcc13/nvcc12.0 불일치 → **g++-12를 nvcc 호스트로** 지정해 CUDA 확장 빌드 성공. transformers 5.x 실패 → **4.44.2**로.
- 외부 repo `setup.py` 실행이 auto-mode 차단 → 사용자 승인 후 진행.
- 스모크에서 **동공 국소화 실패**(프롬프트 `"pupil."`이 이미지 전체 박스 선택) 발견 → 진단(프롬프트×모션) → `"black pupil."`+과대박스필터로 수정(D14).
- 483 anchor 실행 → **valid 100%, median 0.75px, precision 0.46px, 오검출 ~3%**(D15). 상세 [07](07_gsam2_audit_results.md).

## I. 전 프레임 GSAM2 + per-frame 트리 + annotator 서베이 (2026-07-01, Claude)
- 08에 `--save-masks` 추가 → **전 4,669 프레임** 재실행: valid 4,666(99.9%), gsam2 마스크 저장.
- `11_build_perframe.py` 작성 → `samples/perframe/{key}/{stem}/`에 프레임별 frame·meta·gt/gt_bbox·unet(center/mask/bbox)·gsam2(center/mask/bbox)·pred 재구성(소스별 하위폴더, PNG 이진 마스크) + `index.csv`. 4,669 dir.
- `12_overlay.py` → `overlay/`에 GT/U-Net/GSAM2 오버레이 15장(정상+오검출), 육안 검증 완료.
- 사용자 요청 도구 조사: `third/`에 DeepVOG·EllSeg·Edge-Guided·RITnet-Plugins clone(+기존 X-AnyLabeling·SAM3-I). 6종 분석 → `annotators/`(개별 md + 비교 README). 추가 audit 후보 최우선=EllSeg/RITnet(MIT·가중치in-repo).

## J. 09 HBTXR 예측 + annotator 가중치 수집 (2026-07-01, Claude)
- ckpt `hbtxr-imgsz64-epoch66-pe0.5401`(subject-independent, img64) ↔ config `..._subject_independent_img64_patch4.yaml`, G=16.
- FACET env를 `.venv-gsam2`에 구축(albumentations·lightning·thop·natsort). **09 정확성 수정**: `Predict.pre_process`(256 하드코딩)가 DeiT PatchEmbed(정확 img_size 강제)와 충돌 → 09에 img_size(64) resize pre_process 추가. interp는 `causal_linear`만 유효.
- 전 4,669 실행 valid 89.7% → **E_orig=‖pred−사람GT‖ median 5.70px**(보고 0.18px의 ~30배, label-noise 0.75px의 ~7배 → **label-noise-limited 아님**). perframe `pred/`·오버레이(magenta) 합류. 상세 [09](09_hbtxr_pred_results.md).
- 사용자 요청: annotator 가중치 in-repo 제공분 `weights/`에 모델명별 수집(EllSeg·RITnet·DeepVOG·Edge-Guided → `weights/ANNOTATOR_WEIGHTS.md`). X-AnyLabeling 동공 모델 리스트 `annotators/x-anylabeling_pupil_models.md`.

## K. Label Noise 측정 + 0.1812 중립화 + GSAM2 mislabel 게이트 + 데이터셋 provenance (2026-07-01, Claude)
- **0.1812 프레이밍 중립화**(사용자 정정): "누수 확정" 철회 → 전 docs/메모리를 "dense(U-Net) 라벨 기준 결과"로 중립화. 5.70px는 sparse 사람 GT 기준(별개 reference).
- **GSAM2 mislabel 분석**(`13_gsam2_mislabel.py`): 오검출 ~3%(anchor 14/483, 전 프레임 3.44%). 원인=반가림 동공→하단 마커 오검출(det↓·area↓·fixation·user7/8). **U-Net 교차검증(`e_unet>15`)이 14/14 완벽 분리(정상 0 손실)** → `14_flag_gsam2_mislabel.py`로 gsam2.json에 mislabel 플래그(161건). [10](10_gsam2_mislabel_analysis.md).
- **Label Noise/Uncertainty**(`10_label_noise.py`): GT-vs-U-Net median **1.575px**(대부분 계통 bias |Δ|=1.53px, U-Net gross오검출 0), GT-vs-GSAM2(failure→U-Net fallback) median **0.751px**. 전체+모션별 mean/median/p95/p99/std → `results/label_noise_*.md`+`plots/`. U-Net은 사람 라벨 학습→비독립(GSAM2가 더 가까움).
- **EV-Eye 데이터셋 provenance**(`/mnt/e` 접근, 원본 읽기만): U-Net=EV-Eye 확인. 개수 Data_davis(frame 1,506,387/label 9,012)·labelled_with_mask(9,011)·predict(정상 1,436,776/raw 1,622,918). 세션 프레임6/predict4/사람3. predict 불일치=**user30 중첩중복(+186,142)**+1_0_1 미완성(−53,772)+session_3 미추론(−15,839). 1:1 predict↔frame=1,436,776. **원본 미삭제·분석 제외만**(사용자 결정). [11](11_eveye_dataset_provenance.md).

## 현재 지점
- 08·09·per-frame·오버레이·annotator·가중치·**Label Noise(GT vs U-Net/GSAM2)·mislabel 게이트·데이터셋 provenance** 완료.
- **다음**: 10_eval 나머지(E_i/U_i(pred)·Spearman·CDF/scatter·subject-level cluster-bootstrap CI) + 리벗. 미결: 0.1812 원 config/mode 재계산.
