# Precision / Label-Noise 실험 패키지

HBTXR(JETCAS 리비전) 리뷰어 응답용 **주석 정밀도 / 라벨 노이즈** 실험의 코드·산출물·보고서 번들.
재훈련·신규 인간주석 없이 추론 결과와 기존 라벨만 사용. 프레임 **346×260(주)** + **64×64(모델)** 병기.

## 구성
```
packages/
├── README.md                 # (이 파일) 색인
├── INTERPRETATION.md         # ★ 결과별 상세 해석 (La/Lb/Lc/STEP5/그림/종합)
├── reports/
│   ├── precision_full_report.md      # 종합 보고서 (EN)
│   ├── precision_full_report_ko.md   # 종합 보고서 (KO)
│   ├── precision_report.md           # 요약 budget 리포트
│   └── e0_schema_report.md           # STEP 0 스키마 확인
├── code/                     # 재현 파이프라인 (src/ 미러)
│   ├── io_schema.py          # E0: 로더/상수/mask-h5 매처
│   ├── align.py              # E0.2: 통일 center 테이블
│   ├── repeatability.py      # La.1/La.2/La.3
│   ├── reproducibility.py    # Lb.1/Lb.2(3CH)/Lb.3
│   ├── accuracy_view.py      # Lc.1 P_n
│   ├── corrected_error.py    # STEP5 (y_pred 전용)
│   ├── frame64.py            # 64×64 스칼라 재계산 (비등방)
│   ├── figures.py            # fig_3ch / fig_budget
│   ├── tables_out.py         # tables/*.csv + precision_report.md
│   └── run_all.py            # DAG 러너 (E0→La→Lb→Lc→STEP5→figures)
├── figures/
│   ├── fig_3ch.png           # 3CH σ 분해
│   └── fig_budget.png        # 라벨노이즈 budget vs 정정오차
├── tables/
│   ├── align_centers.csv     # 통일 소스 center (long)
│   ├── per_subject.csv       # subject별 E_orig
│   ├── per_motion.csv        # 모션별 E_orig
│   ├── Pn_contrast.csv       # P_n (gsam2 vs pred)
│   ├── repeats_crosscheck.csv# Lb.3 교차검증
│   └── pairwise_agreement.csv# Lb.1 RMS/BA/IoU
└── scalars/
    ├── precision_summary.json# 전 스칼라 (346×260)
    ├── frame64_scalars.json  # 64×64 스칼라
    └── run_gate_log.txt      # 단계별 [DONE/SKIP/FALLBACK/OPTIMISTIC]
```

## 헤드라인 수치 (346×260 / 64×64)
| 양 | 346×260 | 64×64 |
|---|---|---|
| σ_human (3CH) | 0.55 | 0.12 |
| σ_human bracket | [0.55, 0.86] | [0.12, 0.19] |
| 보고 0.1812 (환산) | ≈0.85 | 0.181 (원) |
| 정정 E_orig (median, ⚠train subj) | 5.70 | 1.17 |
| GSAM2 vs human RMS / IoU | 0.86 / 0.914 | 0.19 / — |

**결론**: 라벨노이즈 floor ≈0.12px(64)/0.55px(346); 보고 0.1812는 그 floor에 위치(dense 라벨 아티팩트);
사람 GT 대비 정정오차는 floor 위(실제 모델오차, 누수 반증). 상세는 `INTERPRETATION.md`.

## 재현
```bash
cd analysis/annotation-analysis
.venv-gsam2/bin/python src/run_all.py      # 전 DAG (346×260) → results/precision/, fig/, tables/
.venv-gsam2/bin/python src/frame64.py      # 64×64 스칼라
```
env `.venv-gsam2`(torch2.5.1 + h5py/scipy/matplotlib/cv2). 입력: `samples/label/*/*.json`,
`results/annotators/*.csv`, mask h5 `/mnt/e/.../Data_davis_labelled_with_mask`.

## 한계 (필독)
- samples = users 1-10 = HBTXR **학습 subject**(test=37-48) → **STEP5 E_orig은 낙관치**(subject-independent
  아님). La/Lb/Lc(라벨 품질)는 split 무관.
- 평가 ckpt `pe0.5401` ≠ 0.1812 생성 모델.
- mask = 타원 rasterize **파생** → 표현 floor는 rasterization floor.
- 진짜 subject-independent 수치는 **test 37-48 재수집** 후 동일 파이프라인 재실행 필요(사람 GT+U-Net 존재 확인됨).
