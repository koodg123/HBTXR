# EV-Eye 라벨/주석 분석 툴킷 — 실행 가이드

이 폴더의 스크립트는 **본 세션에서 직접 계산 불가했던 정량 지표**(h5 마스크, .mat 추적결과, 수억 이벤트 통계, 라벨 품질)를 사용자의 Python 환경에서 산출하기 위한 것입니다. 모두 `--root`(EV-Eye 루트)와 `--out`(결과 폴더)만 맞추면 동작합니다.

## 설치
```bash
cd <이 폴더>            # .../annotation-analysis/scripts
python -m pip install -r requirements.txt
```
> `01`(이벤트/프레임 통계)과 `03`(Tobii gaze)은 **표준 라이브러리만으로도** 실행됩니다. `h5py`는 `02`(마스크)·`04`(v7.3 .mat), `scipy`는 `04`(≤v7 .mat), `opencv-python`은 `06`·타원적합, `matplotlib`은 `05` 그림에 필요합니다.

## 전체 실행
```bash
python run_all.py --root E:/DATASET/eveye --out ../results
# 빠른 스모크(3명만): --max-users 3
# 정확한 이벤트수(느림, 27억개 전수): --exact
# 예측 마스크 GIF 인벤토리 포함: --predict
```
결과 종합은 `../results/annotation_quality_report.md` 를 여세요.

## 개별 스크립트
| 스크립트 | 기능 | 핵심 산출물 |
|---|---|---|
| `01_analyze_dataset.py` | 구조·세션별 프레임수·이벤트수/율·VIA커버리지·processed_data 존재점검 | `01_session_stats.csv`, `01_summary.json` |
| `02_analyze_labels.py` | h5 마스크 레이아웃·마스크 기하(면적/중심/타원), VIA 타원 커버리지·지터, 세션결손 | `02_h5_layout.json`, `02_mask_stats.csv`, `02_via_coverage.csv` |
| `03_analyze_tobii_gaze.py` | Tobii gaze 통계·샘플레이트·유효율·동공직경, TTL 동기 offset | `03_tobii_gaze.csv`, `03_tobii_sync.csv` |
| `04_analyze_tracking_results.py` | `.mat` 추적결과 스키마/궤적, (옵션)예측마스크 인벤토리 | `04_mat_layout.json`, `04_track_summary.json` |
| `05_eval_annotation_quality.py` | 01–04 종합 → Q1~Q4 라벨품질 + 4대 벤치마크 매핑 + 그림 | `annotation_quality_report.md`, `plots/*.png` |
| `06_v2e_synth_check.py` | (옵션) v2e로 합성한 이벤트 vs 실제 이벤트 분포 비교(시뮬레이터 평가) | `06_v2e_vs_real.json` |
| `evlib.py` | 공용 파서/유틸(events.txt·VIA·h5·mat·마스크기하) | — |

## 평가 지표 매핑 (EV-Eye 공식 4축 + 라벨진단 Q1~Q4)
- **Q1 마스크↔타원 일관성**(02+05): 마스크는 타원의 결정론적 함수 → IoU≈1 기대, 이탈=생성결함.
- **Q2 라벨 노이즈**(02+05): 타원중심 시계열 지터 → PE 하한 추정.
- **Q3 커버리지·균형**(01+02+05): 사용자/세션/양안 라벨밀도, `session_1_0_1` 마스크 결손.
- **Q4 클럭 동기**(03+05): DAVIS µs UNIX ↔ Tobii 상대초, TTL offset.
- **① IoU/F1 ② 프레임PE ③ 이벤트PE ④ 시선DoD**: 보고서 `03_annotation_tool_evaluation.md` §3 참조.

## 알려진 데이터 특이점(스크립트가 처리함)
- `events.txt`는 인덱스 뒤 TAB + 이후 공백 혼합 구분자.
- 마스크 라벨은 `session_1_0_1` 부재(후3세션만).
- `Data_davis_predict`는 `userN/userN`(이중) 디렉터리 중첩.
- `.mat`은 v7/v7.3 혼재 → `evlib.load_mat_any`가 scipy→h5py 자동 폴백.
- h5 내부 스키마 미상 → `02`는 가장 큰 2D/3D 배열을 마스크 스택으로 휴리스틱 처리(레이아웃을 먼저 `02_h5_layout.json`으로 확인 권장).

## 주의
- `--exact`는 전체 이벤트(≈27억) 전수 스캔으로 매우 느립니다. 기본은 파일크기 기반 추정 + 표본 통계입니다.
- `06`은 v2e 설치가 필요합니다: `pip install -e <.../third/v2e>` 후 `--v2e <.../third/v2e/v2e.py>`.
