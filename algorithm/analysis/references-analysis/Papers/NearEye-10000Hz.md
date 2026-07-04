# Event-Based Near-Eye Gaze Tracking Beyond 10,000 Hz

> **중요 한계**: 본 PDF는 파일 크기 20MB 초과 + 본 세션의 PDF 페이지 추출 도구(pdftoppm) 부재 + bash 금지 제약으로 **원문 직접 판독 불가(확인 불가)**. 아래 내용은 (a) 본 세션에서 직접 읽은 6편 논문(3ET, Retina, Swift-Eye, E-Track, SynUnlabeled-PupilTrack, AdaptiveSSM)의 **참고문헌·본문 인용에서 교차확인된 사실**과 (b) 일반 지식 기반 **추정**을 구분 표기함. 정밀 수치/수식은 원문 재판독 필요.

## 1. 서지정보
- **제목**: Event-Based Near-Eye Gaze Tracking Beyond 10,000 Hz
- **저자**: Anastasios N. Angelopoulos, Julien N.P. Martel, Amit P. Kohli, Jörg Conradt, Gordon Wetzstein (Stanford 등) — *교차인용 확인*
- **연도/게재처**: IEEE Transactions on Visualization and Computer Graphics (TVCG) 27(5):2577–2586, 2021, DOI 10.1109/TVCG.2021.3067784. (초기 버전: IEEE VR 2020, arXiv:2004.03577) — *여러 논문 참고문헌에서 교차확인*
- **원본파일명**: Event-Based Near-Eye Gaze Tracking Beyond 10,000 Hz.pdf
- **판독 상태**: 원문 본문 **확인 불가** (파일>20MB).

## 2. 문제정의·배경 (교차인용 기반)
- 프레임 기반 시선추적은 카메라 프레임율(보통 ≤300Hz)에 묶여 saccade/microsaccade 등 초고속 안구운동을 충분히 포착 못함.
- 이벤트 카메라는 μs 시간해상도·sparse·저전력으로 kHz급 추적 가능 → **상용 트래커에 필적하는 정확도로 10,000Hz(10kHz) 이상 업데이트율** 달성을 목표/주장 (3ET[5], Retina, AdaptiveSSM, E-Track[10] 등이 일관 인용).
- **이벤트 기반 시선추적의 최초(2020) 대표 연구**로 후속 연구들이 baseline·동기로 광범위 인용 (Retina 논문: "The first event-based eye-tracking work was published in 2020 [Angelopoulos]").

## 3. 핵심기여 (교차인용 기반)
- **하이브리드 프레임-이벤트 near-eye gaze tracking 시스템** 제안: 저프레임율 grayscale 프레임 + 고시간해상도 이벤트 스트림 결합.
- **parametric pupil model 기반**: 프레임에서 동공 모델을 초기화하고 **이벤트로 모델을 갱신(pupil-fitting)**하여 프레임 사이를 보간 → >10kHz 업데이트율 (Retina 논문 Table 1: "Model-based, ≥10kHz, 0.45°–1.75°", Swift-Eye: "EVBEYE ... pupil region ... model-fitting and updating").
- 후속 이벤트 시선추적 데이터셋·연구의 기준점 제공(여러 데이터셋이 이 연구 또는 그 셋업과 비교).

## 4. 방법론 (교차인용 기반 + 추정, 정밀내용 확인 불가)
- **이벤트 표현/모델**: parametric pupil model(타원/원형 동공 파라미터)을 이벤트로 갱신하는 model-based 접근 — *교차인용 확인* (Retina: "based on a parametric pupil model and utilizes the event to update the model with a pupil-fitting method").
- **하드웨어**: DAVIS346b 이벤트 카메라(346×260, frame+event 동시 출력) 사용 — *교차인용 확인* (SynUnlabeled "EB-NEG" 설명: "two DAVIS 346b event cameras (346×260 pixels)", head 고정 headrest, random saccade + smooth pursuit 태스크, 화면 자극좌표 GT).
- **민감도 한계**: 센서 noise에 민감 — 모델 갱신에 유용한 이벤트 선택이 어려움 (Retina 논문 언급) — *교차인용 확인*.
- 구체적 수식·파이프라인 단계·캘리브레이션 절차는 **확인 불가** (원문 재판독 필요).

## 5. 실험 (교차인용 기반)
- **데이터셋**: 자체 수집(24명, headrest 머리 고정, random saccade + smooth pursuit, 화면 1920×1080 ~64°×96° 시야, GT는 화면 stimulus 좌표) — *SynUnlabeled의 EB-NEG 설명에서 교차확인*. 이후 "Angelopoulos dataset / EB-NEG"로 후속 연구에서 벤치마크로 사용.
- **메트릭/수치**: gaze 정확도 **0.45°–1.75°** (각도 오차), 업데이트율 **≥10kHz** — *Retina 논문 Table 1에서 교차확인*. (단, 화면좌표 GT라 동공중심 px 메트릭과 직접 비교 제한 — SynUnlabeled가 지적.)
- 상세 ablation·baseline 비교는 **확인 불가**.

## 6. 강점/한계 (교차인용 기반 + 추정)
- **강점**: 이벤트로 프레임 사이를 보간해 매우 높은 시간해상도(>10kHz) 달성, 상용 트래커급 각도 정확도. 이벤트 시선추적 분야를 개척한 선구적 연구.
- **한계 (교차인용 기반)**: (1) 하이브리드(프레임+이벤트)로 프레임 센서 의존 → 순수 이벤트보다 전력/대역폭 불리 (Retina 지적). (2) parametric model 갱신이 센서 noise에 민감(유용 이벤트 선택 난제). (3) head-fixed(headrest) 통제환경 수집 → in-the-wild 일반화 제약(INI-30/Ini-30가 이를 개선 동기로 제시). (4) GT가 화면좌표라 동공 위치 라벨 부재(E-Track이 별도 분할로 동공 타원 생성해 사용).

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **직접 연관 (foundational)**: 본 연구는 이벤트 기반 고주파 시선추적의 원점 - 우리 프로젝트가 추구하는 "고주파·저지연 XR 시선추적"의 정당화 근거이자 정확도/주파수 비교 기준(>10kHz, 0.45°–1.75°)(추정).
- **추정**: parametric pupil model + 이벤트 갱신 방식은 NN 추론 빈도를 낮추는 model-based 보간으로, FPGA에서 무거운 NN 가속 없이 경량 fitting 회로로 프레임 사이를 채우는 설계(E-Track RoI와 유사 철학)에 응용 가능. 단 noise 민감성은 HW 필터링 필요.
- **추정**: 하이브리드(프레임+이벤트)는 프레임 센서·이벤트 센서 듀얼 입력 → FPGA 입력단 동기화/버퍼링 설계 고려사항. 순수 이벤트(E-Track, Retina) 대비 전력 trade-off.
- 양자화·딥러닝 가속과 직접 연관은 약함(model-based 접근) — 우리 ViT 양자화(Q-HyViT) 라인과는 상보적(추정).

## 8. 근거표기
- **원문 본문 직접 판독: 확인 불가** (PDF >20MB, pdftoppm 부재, bash 금지).
- 1·2·3장 및 5장 수치(0.45°–1.75°, ≥10kHz, DAVIS346b, 24명, headrest)는 본 세션 6편 논문의 참고문헌/본문에서 **교차인용 확인**.
- 서지정보(저자/TVCG 2021/DOI/arXiv:2004.03577)는 다수 논문 참고문헌에서 **교차확인**.
- 4장 방법 세부·6/7장 시사점은 **추정** 또는 교차인용 기반 요약. 정밀 수식·아키텍처·전체 실험표는 원문 재판독 시 보강 필요.
