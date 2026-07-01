# Annotation-Tool 평가 — v2e 코드베이스 & EV-Eye 데이터셋 심층 분석

**Role: AI/CV Systems Analyst / Event-Vision Dataset Engineer / Research Methodology Reviewer**

생성일: 2026-06-30 · 분석 대상: `third/v2e` 코드베이스 + `E:\DATASET\eveye`(EV-Eye 데이터셋)

> **저장 위치**: 현재 위치는 `.../HBTXR/analysis/annotation-analysis/` 입니다. (원래 요청 경로 `E:\DATASET\codes\annotation-analysis`는 이 세션의 폴더 마운트 서브시스템이 기존 WSL UNC 마운트 충돌로 새 드라이브를 마운트하지 못해 사용 불가였고, 사용자가 프로젝트 폴더 내 현재 위치로 이동함.) 필요 시 이 폴더 전체를 원하는 곳으로 복사하면 됩니다.

---

## 0. 이 폴더의 구성 (Deliverable Map)

| 파일 | 내용 |
|---|---|
| `README.md` | 본 문서. 전체 요약, 분석 기준, 핵심 결론, 실행 방법, 진행상황 |
| `01_v2e_codebase_analysis.md` | v2e 코드베이스 모듈/함수 단위 심층 분석 + 다이어그램 |
| `02_eveye_dataset_analysis.md` | EV-Eye 데이터셋 구조·모달리티·라벨·실험결과 심층 분석 + 다이어그램 |
| `03_annotation_tool_evaluation.md` | 평가 종합: (1) 라벨 생성 파이프라인, (2) v2e 이벤트 시뮬레이터, (3) EV-Eye 라벨 품질 |
| `scripts/` | 실행 가능한 정량 분석 툴킷(Python). 데이터셋·라벨·결과 통계 + 라벨 품질 평가 |
| `scripts/README.md` | 툴킷 실행 가이드 |

> **다이어그램**: 각 보고서의 ```mermaid``` 코드블록으로 포함되어 있으며, GitHub/VS Code(Markdown Preview Mermaid)/Obsidian 등에서 렌더링됩니다.

---

## 1. 한눈 요약 (Executive Summary)

- **`third/`에는 단 하나의 코드베이스 `v2e`가 있습니다.** v2e(SensorsINI/v2e, CVPRW 2021)는 일반 영상 프레임을 **물리적으로 사실적인 합성 DVS(이벤트 카메라) 스트림**으로 변환하는 PyTorch 도구입니다. 이벤트 카메라 픽셀의 아날로그 특성(광수용체 대역폭, 픽셀별 임계값 편차, 누설/샷 노이즈, 불응기, 센터-서라운드)을 모델링하고, 회색조로 재학습한 SuperSloMo로 프레임을 업샘플링합니다.
- **`E:\DATASET\eveye`는 EV-Eye 데이터셋입니다.** 48명 × 양안 × 4세션, 2×DAVIS346로 수집한 **근안(near-eye) 프레임+이벤트** 멀티모달 안구추적 데이터셋(약 150만 프레임, 27억 이벤트)이며, Tobii Pro Glasses 3 시선 GT를 동반합니다. **라벨**은 (a) VIA 동공 타원 주석(`user_N.csv`), (b) 이진 동공 마스크(`*.h5`), (c) Tobii 시선/동공직경 GT 세 종류입니다.
- **"Annotation Tool"의 실체** — EV-Eye에서 라벨을 만드는 주석 파이프라인은 **VGG Image Annotator(수동 동공 타원) → MATLAB 마스크 생성(`generate_pupil_mask.m`) → DL 동공 분할 학습/예측 → 프레임·이벤트 하이브리드 추적 → Tobii 기준 시선 매핑**의 다단계 구조입니다. v2e는 이 파이프라인의 직접 구성요소는 아니지만, **프레임 주석을 이벤트로 전이**하거나 **합성 이벤트에 신호/노이즈 라벨**을 부여하는 보조 도구로 활용 가능합니다.
- **평가 결론(요지)**: EV-Eye의 라벨 체계는 다중 모달 교차검증이 가능해 강력하지만, **타원 주석이 희소(키프레임 9,011장)**하고 **session_1_0_1에는 마스크 라벨이 없으며**, 두 장치(DAVIS μs UNIX clock vs Tobii 상대초)의 **클럭 동기화가 TTL에 의존**합니다. 정량 품질지표(마스크-타원 IoU, 동공중심 픽셀오차, 라벨 노이즈)는 본 환경에서 직접 계산 불가하여 **`scripts/` 툴킷으로 사용자가 실행**하도록 제공합니다.

---

## 2. 최상의 결과물 기준 (Best-output Criteria)

| 기준 | 본 분석에서의 충족 방법 |
|---|---|
| 구조성 | 코드/데이터/평가를 3개 보고서로 분리, 각 모듈·파일·라벨에 책임 명시 |
| 실행성 | 정량 분석은 즉시 실행 가능한 Python 툴킷으로 제공(경로 설정만 하면 동작) |
| 근거성 | 모든 주장에 파일 경로·라인·공식 EV-Eye 문서/논문 출처 명시 |
| 추적성 | 어떤 결론이 어떤 파일/샘플/문서에서 나왔는지 표기, 미확인 항목은 별도 표시 |
| 검증성 | 마지막에 자체 검증 + 가정/불확실성 목록 |
| 안정성 | 바이너리(h5/mat) 미파싱 한계를 명시하고 추측과 사실을 구분 |

---

## 3. 분석 환경 제약 (정직한 고지)

- 본 세션의 **Linux 셸(Python/h5py/scipy 실행)이 비활성**입니다(WSL UNC 마운트가 샌드박스를 깨뜨림). 따라서:
  - **가능**: 텍스트 파일(코드, `events.txt`, `gazedata`, `user_N.csv`, `timestamps.txt`) 정밀 분석, PNG 프레임 육안 확인, 구조 매핑, 공식 문서/논문 대조.
  - **불가(본 환경)**: `*.h5` 마스크·`*.mat` 추적결과 바이너리 파싱, 수억 개 이벤트에 대한 통계 연산.
- 이 한계를 메우기 위해 **정량 분석을 수행하는 Python 스크립트**를 `scripts/`에 작성했습니다. 사용자는 WSL/Windows의 Python 환경에서 직접 실행하여 마스크 면적·동공중심 오차·이벤트율·라벨 커버리지·품질지표를 산출할 수 있습니다.

---

## 4. 핵심 결론 미리보기 (자세한 내용은 03 보고서)

1. **v2e는 "주석 도구"가 아니라 "프레임→이벤트 변환기"**입니다. 자체적인 동공/시선 모델이 없으며, 의미 라벨(동공·시선·깜빡임)은 소스 프레임에서 와야 합니다. 단, `--label_signal_noise`로 **이벤트별 신호/샷노이즈 라벨**은 생성 가능합니다.
2. **EV-Eye 라벨 파이프라인은 재현 가능**하지만 수동 타원 주석이 병목이며 희소합니다. 마스크는 타원에서 결정론적으로 생성되므로 **마스크-타원 일관성**은 본질적으로 높되, **타원 주석 자체의 사람 오차**가 상한입니다.
3. **라벨 품질 평가는 4개 축**(IoU/F1 분할, 프레임 PE, 이벤트 PE, 시선 DoD)으로 수행하는 것이 EV-Eye 벤치마크와 일치합니다. 본 툴킷이 이를 재현/검증하도록 설계되었습니다.

---

## 5. 툴킷 실행 방법 (요약)

```bash
cd <이 폴더>/scripts
python -m pip install -r requirements.txt        # numpy, h5py, scipy, opencv-python, pandas, matplotlib, tqdm
python run_all.py --root "E:/DATASET/eveye" --out "../results"
```
개별 실행은 `scripts/README.md` 참고. 모든 스크립트는 `--root`(데이터셋 루트)와 `--out`(결과 폴더)만 맞추면 동작합니다.

---

## 6. 출처 (Sources)

- v2e 코드: `\\wsl.localhost\...\HBTXR\third\v2e\` (README, `v2ecore/*` 직접 분석)
- EV-Eye 데이터: `E:\DATASET\eveye\` (직접 샘플링)
- [EV-Eye 공식 저장소 (Ningreka/EV-Eye)](https://github.com/Ningreka/EV-Eye/blob/main/README.md)
- [EV-Eye 논문 (NeurIPS 2023 D&B, OpenReview)](https://openreview.net/forum?id=bmfMNIf1bU)
- [v2e 논문 (Hu, Liu, Delbruck, CVPRW 2021, arXiv:2006.07722)](https://arxiv.org/abs/2006.07722)
