# E-Track 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\E-Track`
> 도구: Glob / Grep / Read (실제 코드 라인 직접 확인 기반)
> 근거표기 원칙: 코드/주석에서 확인된 사실은 "확인", 명시 안 됐으나 코드 정황으로 합리적 추론은 "추정", 코드에 단서가 없는 것은 "확인 불가"로 명기.

---

## 1. 개요

### 1.1 목적
- **이벤트 카메라(event camera) 기반 XR용 시선/동공 추적(eye/pupil tracking) 알고리즘** 구현체. (확인: `README.md:1`, `e_track.py:296` `e_track()` 함수)
- 원논문: **"E-Track: Eye Tracking with Event Camera for Extended Reality (XR) Applications"**, Nealson Li, Ashwin Bhat, Arijit Raychowdhury, **AICAS 2023** (IEEE 5th Int. Conf. on AI Circuits and Systems). DOI 10.1109/AICAS57966.2023.10168551. (확인: `README.md:5-15`)
- 저자 소속 정황상 Georgia Tech 계열의 **AI 회로/시스템(AICAS)** 학회 발표 작업 → 알고리즘 자체가 저전력/저지연 on-device 가속을 염두에 둔 설계. (추정: 학회 성격 AICAS + RoI 기반 이벤트 절감 메커니즘)

### 1.2 입력 / 출력
- **입력 (두 종류 스트림을 시간순으로 병합):** (확인: `eye_dataset.py:14-16`, `dataset.__getitem__` 로직 `eye_dataset.py:73-91`)
  - **이벤트(Event)**: `(polarity, row, col, timestamp, label)` namedtuple. DAVIS 계열 센서로 추정되는 `.aerdat` 바이너리에서 파싱(`packet_format='BHHI'`, 9바이트/이벤트). (확인: `eye_dataset.py:26-45`)
  - **프레임(Frame)**: `(row, col, img, timestamp)`. 그레이스케일 PNG/JPG 이미지로, 여기서 `(col,row)`는 **응시 자극 위치(gaze stimulus label)** 역할도 겸함. (확인: `eye_dataset.py:73-91`, `e_track.py:77` `frame_label = [col, row]`)
- **센서 해상도:** 이벤트 `346 x 260` (DAVIS346 추정). (확인: `e_track.py:62-65`)
- **출력:**
  - **U-Net 학습/추론 경로(`e_track_unet.py`)**: 입력 이벤트 프레임 → **2-class 픽셀 세그멘테이션(동공 이벤트 vs 배경)** 확률맵 `(352, 256, 2)`. (확인: `e_track_unet.py:51-58` `num_classes=2`, `e_track_dataset.py:37` 2채널 라벨)
  - **전체 E-Track 알고리즘(`e_track.py`)**: 세그멘테이션 후 **타원 피팅(ellipse fitting)** 으로 **동공 타원 파라미터** `center=[x0,y0], width, height, phi` 산출 → 동공 중심 좌표가 최종 시선/동공 위치. (확인: `e_track.py:447-452`, `util/ellipse.py:134-197`)
- 요약: **입력 = 이벤트 스트림 + (자극 라벨 포함) 프레임 / 출력 = 동공 세그멘테이션 마스크 + 동공 타원(중심 좌표)**.

### 1.3 데이터셋
- 베이스는 **"Event-Based Eye Tracking" 공개 데이터셋(Kohli/Martel/Angelopoulos, 2020)** 을 차용. (확인: `eye_dataset.py:3-4` 저자 크레딧, `e_track.py:37-38, 515-516`)
- `setup.sh`가 berkeley.box.com에서 user1~user27 raw tar를 다운로드. 논문은 **이전 연구와의 일관성을 위해 subject 4~27만 사용**(subject 1~3은 다른 셋업). (확인: `README.md:28`, `setup.sh:8-115`, `e_track_unet.py:81` `range(4, 28)`)
- 학습용은 raw `.aerdat`가 아니라 전처리된 **`.tfrec`(GZIP TFRecord)** 사용: `tfrecord_0`(train) / `tfrecord_1`(valid) / `tfrecord_2`(test). (확인: `e_track_unet.py:82-87`, `e_track_dataset.py:43-44`)

---

## 2. 디렉토리 구조 (자체 소스 vs 제외 대상)

```
E-Track/
├── README.md                      [자체] 목적/논문/실행법
├── requirements.txt               [자체] pip 의존성 (TF-GPU 2.6, unet @ git)
├── requirements_conda.txt         [자체] conda 환경 스냅샷
├── setup.sh                       [자체] raw 데이터 다운로드 스크립트
├── e_track.py                     [자체·핵심] 전체 E-Track 추론 루프 (536 lines)
├── e_track_unet.py                [자체·핵심] U-Net 학습/추론 진입점 (146 lines)
├── dataset/
│   ├── eye_dataset.py             [자체·핵심] raw aerdat/frame 로더 (122 lines)
│   └── e_track_dataset.py         [자체·핵심] TFRecord → tf.data 파이프라인 (48 lines)
├── util/
│   └── ellipse.py                 [자체·차용] 최소제곱 타원 피팅 (238 lines, bdhammel 차용)
├── data/                          [제외] tfrecord_0/1/2, eye_data/ — 대용량 데이터
│   ├── tfrecord_0/ *.tfrec        [제외] train split
│   ├── tfrecord_1/ *.tfrec        [제외] valid split
│   └── tfrecord_2/ *.tfrec        [제외] test split
├── trained_model/2023-01-24T00-11_42   [제외] 학습된 Keras SavedModel 체크포인트
├── img/                           [제외] README용 시스템 흐름도 이미지
└── .git/                          [제외] VCS 메타데이터
```

### 제외 근거 (명시)
- **`unet` 패키지**: 코드베이스 내부에 소스 없음. `requirements.txt:58`에서 `unet @ git+https://github.com/jakeret/unet.git@f557a51b...` 로 **외부 GitHub 패키지(jakeret/unet)** 핀 고정 → `e_track_unet.py:21,24`의 `import unet`, `unet.build_model`, `unet.finalize_model`, `unet.Trainer`, `from unet import custom_objects, utils`가 모두 외부 라이브러리. (확인: glob 결과 `unet/` 디렉토리에 `.py` 없음 + requirements 핀)
- **`util/ellipse.py`**: 내부 파일이긴 하나 **bdhammel/least-squares-ellipse-fitting v2.0.0 차용**(라인 1-3 크레딧). 알고리즘은 Halir & Flusser 직접 최소제곱법. 자체 작성이 아니라 "차용 모듈"로 분류하되 핵심 경로라 분석에 포함.
- **`data/`, `trained_model/`, `img/`, `.git/`**: 데이터·산출물·메타로 정밀 분석 제외.

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

### 3.1 데이터 로더 ① — `dataset/eye_dataset.py` (raw 스트림 → 시간순 이벤트/프레임)

- **`read_aerdat(filepath)` (`:26-45`)**: 이벤트 바이너리를 `struct` 언팩. `packet_format='BHHI'` → `pol(uint8), x(uint16), y(uint16), t(uint32)` = 9바이트/이벤트(`:33` 주석). 끝의 잉여 바이트 제거 후 전체를 한 리스트로 언팩하고 **`reverse()`** (`:43`) — 이후 `pop()`이 가장 이른 타임스탬프를 꺼내도록 스택 LIFO를 시간순으로 맞춤.
- **`get_path_info(path)` (`:48-57`)**: 프레임 파일명 `idx_row_col_stimType_timestamp.ext` 파싱. **`row,col`이 곧 응시 자극(gaze) 라벨**로 쓰임. (주의: `:52` `path.split('\\')` — Windows 구분자 하드코딩, POSIX 경로에선 깨질 위험. 한계로 후술.)
- **`EyeDataset` 클래스 (`:60-121`)**:
  - `collect_data(eye)` (`:93-100`): 프레임 스택과 이벤트 스택을 각각 로드.
  - `__getitem__` (`:73-91`): **이벤트 스택의 4번째 뒤 원소(=다음 이벤트 timestamp, `:75` `event_stack[-4]`)와 프레임 timestamp를 비교**해 더 이른 것을 반환 → 두 스트림을 시간순 병합. 이벤트면 `pop()` 4회로 `(pol,row,col,ts)` 복원하고 현재 프레임의 `(col,row)`를 `label`로 붙임(`:80-86`). 프레임이면 PIL로 그레이스케일 로드(`:89`).
  - 설계 의도(확인): **단일 정렬 이벤트-프레임 시퀀스**를 만들어 `e_track.py`가 `for ... in eye_dataset` 한 줄로 순회하도록 함.

### 3.2 데이터 로더 ② — `dataset/e_track_dataset.py` (TFRecord → tf.data, U-Net 학습용)

- **`_read_tfrecord(example)` (`:17-40`)**:
  - 이미지: `decode_raw(uint8)` → reshape `(346,260,3)` → `/255.0` 정규화 → `resize_with_crop_or_pad(352,256)`. (확인 `:25-29`)
  - 라벨: `decode_raw(bool)` → `(346,260,1)` → crop/pad → **`concat([logical_not(label), label])`로 2채널 one-hot** 생성(`:37`) → float32. 즉 채널0=배경, 채널1=동공.
  - **핵심 설계 결정**: 입력을 346→**352(=11×32)**, 260→**256(=2^8)** 로 패딩/크롭. U-Net이 layer_depth=3로 2배씩 3회 다운샘플하므로 8의 배수 정렬이 필요 → 32 정렬은 추가 마진. (추정: 입력을 2의 거듭제곱·32 배수로 맞추는 것은 다운/업샘플 정합 목적)
- **`load_data(tfrecs)` (`:43-47`)**: GZIP TFRecordDataset, `num_parallel_reads=AUTO`, `experimental_deterministic=False`(`:5-6`)로 처리량 우선. (주의: `.batch/.shuffle/.prefetch` 미적용 — 배치는 호출부 `trainer.fit(batch_size=8)`에서 처리. 확인 `e_track_unet.py:101`)

### 3.3 모델(백본·넥·헤드) — U-Net (외부 `unet`), 빌드부 `e_track_unet.py`

- **아키텍처: 표준 U-Net 인코더-디코더(완전 합성곱)**. backbone/neck/head가 명시적으로 분리된 detection류가 아니라 **세그멘테이션 U-Net 단일체**. (확인: `e_track_unet.py:51-58`)
- `unet.build_model(nx=352, ny=256, channels=3, num_classes=2, layer_depth=3, filters_root=32, padding="same")` (`:51-58`):
  - **입력**: 3채널(이벤트→RGB 프레임), `352×256`.
  - **백본(인코더)**: `layer_depth=3` → 3단계 다운샘플(`filters_root=32` 시작, 통상 32→64→128→256 채널). (추정: jakeret/unet 표준 구현 기준. 정확한 채널 진행은 외부 패키지라 "확인 불가", build 인자만 확인)
  - **넥/스킵**: U-Net의 스킵 커넥션(인코더↔디코더 concat) — U-Net 정의상 존재(확인: U-Net 사용), 세부 구현은 외부.
  - **헤드**: `num_classes=2` → 마지막 1×1 conv + softmax로 2채널 픽셀 분류맵. (확인: `num_classes=2`; 활성화 세부는 외부 "확인 불가")
- `unet.finalize_model(...)` (`:60-65`): 컴파일 단계. **loss=가중 categorical CE**, `learning_rate=1e-3`(`:30`), `auc=False`, `epsilon=1e-6`.
- **모델 규모/파라미터 수는 `model.summary()` 호출(`:67`)로 런타임에만 출력 → 정적 코드상 "확인 불가"**(외부 패키지 의존).

### 3.4 손실함수 — Weighted Categorical Cross-Entropy (`e_track.py:267-278` = `e_track_unet.py:36-47`, 동일 정의 중복)

```python
def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)      # 확률 정규화
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon()) # 수치 안정화(log(0) 방지)
        loss_wcc = y_true * K.log(y_pred) * weights           # 클래스별 가중
        loss_wcc = -K.sum(loss_wcc, -1)
        return loss_wcc
    return loss
```
- **가중치 `[0.1, 0.9]`** 적용(`e_track_unet.py:61`, `e_track.py:318`) → **배경 0.1, 동공 0.9**. (확인) 이유: 동공 이벤트 픽셀은 전체 대비 극소수 → **클래스 불균형 보정**으로 소수 클래스(동공)에 9배 가중. (추정: 불균형 보정 목적, 코드 정황상 명확)
- 동일 함수가 두 파일에 복붙됨 → 유지보수 측면 중복(코드 스멜, 후술).
- 학습 메트릭: `EarlyStopping(monitor="mean_iou", mode="max", patience=3, restore_best_weights=True)` (`e_track_unet.py:69-77`) → **mIoU 기반 조기종료**. (확인)

### 3.5 학습 루프 — `e_track_unet.py:train()` (`:50-103`)

1. `build_model` → `finalize_model` → `summary` (`:51-67`).
2. EarlyStopping 콜백(`:69-77`), `unet.Trainer(name="pupil_event", callbacks=[...])` (`:79`) — Trainer는 외부 패키지.
3. **데이터 split 구성(`:81-87`)**: `users=range(4,28)`; `tf.io.gfile.glob`으로 tfrecord_0/1/2를 각각 train/valid/test로. `load_data`로 tf.data 변환.
4. `trainer.fit(model, train, valid, test, epochs=40, batch_size=8, verbose=2)` (`:95-101`).
- **버그 의심(High)**: `:82-84`에서 `tf.io.gfile.glob(f"...{user}..." for user in users)` — **제너레이터를 glob에 직접 전달**. `gfile.glob`은 단일 문자열/리스트를 받으므로 제너레이터 입력은 의도대로 동작하지 않음(각 유저별 glob 결과를 모으려면 리스트 컴프리헨션 후 flatten 필요). 또한 f-string 안의 `{user}`는 제너레이터 컴프리헨션 변수라 실제 문자열 포매팅과 어긋남. → 학습 데이터 경로 수집이 깨질 가능성. (확인: 코드 라인; 동작 여부는 실행 환경 의존이라 "추정"으로 표기)

### 3.6 추론 루프 — `e_track_unet.py:predict()` (`:106-136`) & **전체 알고리즘 `e_track.py:e_track()` (`:296-511`, 핵심 중의 핵심)**

#### 3.6.1 `predict()` (단순 U-Net 추론·벤치)
- `load_model('trained_model/2023-01-24T00-11_42', custom_objects)` (`:108-109`). custom_objects에 loss 주입(`:107`).
- warmup 추론 후 `predict`/`evaluate`로 결과·지연시간(`timeit`) 측정(`:119-133`). `with tf.device('/cpu:0')`(`:144`)로 **CPU 추론 시간** 측정 — 저지연 on-device 관심을 시사. (확인 `:144`; on-device 가속 의도는 "추정")

#### 3.6.2 `e_track()` — 풀 파이프라인 (E2F → U-Net → RoI → 타원피팅)
이 함수가 논문의 시스템 흐름(Event-to-Frame Converter + Pupil Event U-Net + Event-Based RoI Mechanism)을 직접 구현. (확인: `README.md:39-43`, 함수 본문)

- **(0) 모델 로드 (`:317-321`)**: 가중 CE loss 주입 후 SavedModel 로드.
- **(1) 스트림 순회 (`:323`)**: `eye_dataset` 순회. `Frame`이면 timestamp/label 갱신(`:324-329`), `Event`면 본처리.
- **(2) do_unet 스위칭 상태머신 (`:331-341`)**: **U-Net을 매번 돌리지 않고, 필요할 때만 호출**하는 핵심 절전 로직.
  - `first_unet` / `abort_unet` / `abort_fit` 상태에 따라 `do_unet` 결정.
  - `do_unet=True`이면 큰 버퍼(`buf_thsld = 4000` 초기 또는 `opt.buffer=2000`)로 이벤트 모음 → 프레임 변환 후 U-Net.
  - `do_unet=False`이면 **작은 버퍼 `buf_thsld=256`**(`:339`)로 빠르게 타원 RoI만 갱신 → **저연산 추적 모드**.
- **(3) 좌표 오프셋 보정 (`:343-344`)**: `(col-col_offset, row-row_offset)`로 눈 영역을 이미지 중심으로 이동. 오프셋은 최초 1회 `get_eye_offset`로 계산(`:371-380`).
- **(4) Event-Based RoI Mechanism (`:346-352`)**: `do_unet=False`일 때만. 현재 타원 기준 **바깥 타원(+4)과 안쪽 타원(-4) 사이 링 영역의 이벤트만 통과**시킴(`get_ell_roi_points` 안/밖 마스크 조합) → 동공 경계 근처 이벤트만 처리해 연산 절감. (확인 `:347-352`, `get_ell_roi_points :118-122`)
- **(5) 버퍼 채워지면(`len==buf_thsld`) 처리 (`:360-...`)**:
  - 센서 이슈 제거 `handle_sensor_issue`(같은 좌표가 15회 초과 반복 시 제거, `:252-264`).
  - **최초 그룹: `get_eye_offset`로 눈 영역 1회 보정**(`:371-380`).
  - 라벨이 `[0,0]`(블링크/무자극)이면 스킵(`:382-387`).
- **(6) ROI 데이터 필터링 (`:392-423`)**:
  - **U-Net 경로(`:392-413`)**: `event_to_frame`으로 이벤트→RGB 프레임 → `unet_model.predict` → **확률 > 0.9 픽셀을 동공 이벤트로 채택**(`tf.where(prediction[0,:,:,1] > 0.9)`, `:404`), 좌표 보정 `+[-2,4]`(`:405`). 결과 없으면 `abort_unet`.
  - **RoI 경로(`:415-423`)**: U-Net 생략, 링 필터된 이벤트(`e_buf`) 그대로 타원 피팅 입력.
- **(7) abort 조건 (`:426-439`)**: ROI 내 이벤트 ≤30이면 `abort_fit`. abort 시 버퍼 비우고 다음 그룹.
- **(8) Ellipse Fitting (`:441-471`)**:
  - 좌표에 미세 노이즈(`noise/10`) 더해 수치 특이 방지(`:445-446`).
  - `LsqEllipse().fit(measurements)` → `as_parameters()`로 `center,width,height,phi`(`:447-452`).
  - **피팅 품질 검증(`:454-462`)**: `get_ell_fit_score`(평균 |r²-1|)가 0.19 초과거나, 첫 피팅의 비정상 종횡비/회전, 또는 직전 대비 width/height 변화율이 임계(17%) 초과·width>40이면 **롤백**(직전 파라미터 복원)하고 abort. → **시간적 안정성(temporal smoothing) 휴리스틱**. (확인)
- **(9) 통계/시각화 (`:473-510`)**: matplotlib로 프레임+이벤트+피팅 타원 오버레이(옵션). `unet_ell_cnt`/`roi_ell_cnt`로 U-Net 경로 vs RoI 경로 성공 카운트 → **연산 절감 효과 측정 지점**.
- **주의(중요): `e_track()`은 결과를 누적·반환하지 않고 `return None`(`:511`)**. 그런데 `main()`은 `target_event_sets, target_event_set_labels = e_track(...)`(`:530`)로 **2-튜플 언패킹** → `None`을 언패킹하면 `TypeError`. 시각화/카운트는 함수 내부에서만 소비되고 최종 정량 출력은 코드상 미완. (확인: `:511, :530` — 명백한 불일치 버그)

### 3.7 보조 알고리즘 — `e_track.py` 기하 유틸 & `util/ellipse.py`

- **`event_to_frame` (`:281-293`)**: 이벤트 누적 히스토그램 방식 프레임화. NEG(pol=0)→R 채널, POS→G 채널, 양쪽 모두 B에 누적(+10/이벤트), `clip(0,255)/255`. **고정 좌표 시프트 `+3, -1 / -2, -2`** 하드코딩(센서-프레임 정렬 보정 추정). → **이벤트 표현 = 2-극성 누적 프레임(accumulated event frame)**, voxel grid/time-surface 아님. (확인)
- **`get_eye_offset` (`:148-249`)**: 이벤트→이진 이미지 → OpenCV 모폴로지(medianBlur, MORPH_CLOSE, dilate/erode, disk 구조요소) → 행 히스토그램(bin=5)으로 최대 클러스터 탐색 → 눈 바운딩박스·중심·오프셋 산출. **CPU 시간 측정(`timeit`, `:158-168`)** → 경량 전처리 지연 관심. (확인)
- **`get_rad_from_ell_c`/`get_ell_roi_points`/`get_ell_fit_score` (`:104-127`)**: 점-타원 정규화 반경(r²) 계산 → RoI 마스킹 및 피팅 점수.
- **`util/ellipse.py` `LsqEllipse` (`:11-237`)**: **Halir & Flusser 수치 안정 직접 최소제곱 타원 피팅**. `fit`(`:56-121`)에서 설계행렬 D1/D2, 산란행렬 S1/S2/S3, 제약행렬 C1, 일반화 고유값 문제로 계수 산출. `as_parameters`(`:134-197`)로 일반형 계수→`center,width,height,phi` 변환(장축/단축 정렬 포함). (확인)

---

## 4. 알고리즘 · 데이터 표현

### 4.1 이벤트 표현
- **누적 이벤트 프레임(accumulated 2-polarity event frame)** 방식. (확인: `event_to_frame :281-293`)
  - U-Net 입력은 3채널 RGB이지만 실질적으로 R=NEG, G=POS, B=공통의 **극성 분리 누적 히스토그램**.
  - **voxel grid / time-surface / SNN 표현은 사용 안 함**(코드 확인). 시간정보는 프레임 단위 버퍼링(`buf_thsld`)으로만 양자화.
- 라벨도 이벤트 좌표를 이진 마스크로 만든 픽셀맵(TFRecord 단계에서 생성, raw 변환부는 이 repo에 "확인 불가").

### 4.2 시간 모델링 / 시퀀스 처리
- **ConvLSTM / SSM / Transformer / RNN 전혀 사용하지 않음.** (확인: 전 파일 grep 범위 내 해당 레이어 없음, 모델은 순수 U-Net)
- 대신 **명시적 상태머신 + RoI 추적 + 타원 파라미터 롤백 휴리스틱**으로 시간적 일관성 확보. (확인: `e_track() :331-471`)
- 즉 **"무거운 U-Net은 가끔, 가벼운 이벤트 RoI 타원 추적은 자주"** 의 하이브리드 시간 처리. (확인: do_unet 스위칭 `:332-339`)

### 4.3 후처리
1. **U-Net 확률맵 임계(>0.9)** → 동공 이벤트 포인트셋. (`:404`)
2. **노이즈 주입 후 최소제곱 타원 피팅** → 동공 타원. (`:445-447`)
3. **피팅 스코어·변화율 기반 검증/롤백** → 이상 프레임 제거 및 평활화. (`:454-462`)
4. **Event-Based RoI 링 마스크**로 다음 프레임 이벤트 사전 필터. (`:347-350`)

---

## 5. 학습 · 평가

### 5.1 데이터셋 / split
- train=`data/tfrecord_0`, valid=`data/tfrecord_1`, test=`data/tfrecord_2`. (확인: `e_track_unet.py:82-87`)
- subject 4~27 사용. raw는 `setup.sh`로 box.com에서 받음. (확인)

### 5.2 손실 / 옵티마이저 / 하이퍼파라미터
- Loss: weighted categorical CE, weights `[0.1, 0.9]`. (확인 `:61`)
- LR=`1e-3`, epochs=40, batch_size=8, layer_depth=3, filters_root=32, padding="same". (확인 `:30,51-58,101`)
- 옵티마이저 종류는 `unet.finalize_model` 내부(외부 패키지) → **"확인 불가"**(코드상 Adam 추정이나 단정 불가).

### 5.3 메트릭
- **세그멘테이션: mean IoU(mIoU)** — EarlyStopping monitor 및 `evaluate` 기준. (확인 `:70`, `:123`)
- **추론 지연: `timeit` CPU 시간** — `get_eye_offset`(`:158-168`) 및 `predict()`(`:127-133`). (확인)
- **동공 위치 오차(p-error 등) 정량 지표는 코드에 미구현**: `e_track()`이 결과를 반환하지 않고(`:511`) 시각화·카운트만 함. 논문 본문 메트릭(예: pixel error)은 이 repo 코드만으로는 **"확인 불가"**. (확인: 코드 부재)

### 5.4 실행 명령어 (확인: `README.md:21-43`)
```bash
# 환경
conda create -n e_track python=3.8 && conda activate e_track
python -m pip install -r requirements.txt
# 데이터
bash setup.sh
# U-Net 학습/추론 (e_track_unet.py 내 train()/predict() 토글, :140-145)
python e_track_unet.py
# 풀 E-Track 알고리즘
python e_track.py            # 예: --subject 22 --eye left --buffer 2000
```

---

## 6. 의존성

- **핵심 프레임워크: TensorFlow-GPU 2.6.0 / Keras 2.6.0** (확인 `requirements.txt:19,52`). Python 3.8, CUDA 10.1 권장(`README.md:19`).
- **U-Net 본체: 외부 `unet` (jakeret/unet, git 커밋 `f557a51b` 핀)** (확인 `:58`). 모델·Trainer·utils 전부 여기 의존.
- **이미지/수치: numpy 1.19.5, scipy 1.4.1, scikit-image 0.19, opencv-python 4.9, pillow, matplotlib 3.5** (확인 `:27,29,31,45,46`).
- **타원 피팅: 내부 `util/ellipse.py`(bdhammel v2.0.0 차용)**, numpy.linalg만 사용.
- 표준 라이브러리: pickle, struct, argparse, collections, multiprocessing 등.

---

## 7. 강점 · 한계

### 강점
- **연산 효율 중심 설계**: U-Net을 항상 돌리지 않고 do_unet 상태머신 + Event-Based RoI 링 필터로 **대부분 프레임을 경량 타원 추적**으로 처리(`e_track() :332-350`). XR on-device에 적합한 구조.
- **시간적 안정성 휴리스틱**: 피팅 스코어/변화율 기반 롤백(`:454-462`)으로 RNN 없이도 추적 안정성 확보.
- **명확한 단계 분리**: E2F 변환 / U-Net 세그 / RoI / 타원 피팅이 함수 단위로 모듈화.
- **재현성 배려**: seed 고정(`e_track.py:31-34`), conda/pip 의존성 핀 고정.

### 한계 / 리스크
- **(High) `train()`의 glob 제너레이터 오용(`e_track_unet.py:82-84`)** — 데이터 경로 수집이 의도대로 안 될 가능성. 리스트 컴프리헨션+flatten 필요.
- **(High) `e_track()` 반환/언패킹 불일치(`:511` vs `:530`)** — `None`을 2-튜플 언패킹 → 실행 시 TypeError 위험. 정량 출력 미완성.
- **(Medium) 경로 처리 비이식성**: `eye_dataset.py:52` `path.split('\\')` Windows 구분자 하드코딩 → Linux/WSL에서 프레임 파싱 실패 가능. (현재 repo가 WSL 경로에 있음에 유의)
- **(Medium) 코드 중복**: `weighted_categorical_crossentropy`가 `e_track.py:267`·`e_track_unet.py:36`에 동일 복붙. img 크기 상수(346/260, 352/256)도 세 파일에 산재.
- **(Medium) 매직 넘버 다수**: `+3,-1/-2,-2` 좌표 시프트(`:288-291`), 임계 0.9·0.19·17·40·30·15 등 — 근거 주석 부족, 튜닝 추적 어려움.
- **(Low) 평가 메트릭 미구현**: 동공 위치 오차 산출/저장 로직 없음(시각화만).
- **외부 `unet` 의존**: 모델 구조 변경·양자화·이식 시 외부 패키지 내부를 건드려야 함.

---

## 8. 우리 프로젝트 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)

> 가정: 우리 프로젝트 목표 = "XR 시선추적을 FPGA 등에서 저지연·저전력 on-device로 가속". (추정)

1. **하이브리드 연산 스케줄링을 HW 파이프라인으로 매핑**: E-Track의 "U-Net은 가끔 / RoI 타원 추적은 자주" 구조(`e_track() :332-350`)는 **이종 가속(무거운 CNN 가속기 + 경량 기하 연산 datapath)** 설계와 자연스럽게 대응. FPGA에서 U-Net은 PL의 systolic MAC array로, RoI/타원 피팅은 경량 FP 연산 블록으로 분리 가능. (추정)
2. **세그멘테이션 백본 경량화·양자화 1순위**: 현재 U-Net(filters_root=32, depth=3)이 유일한 무거운 모듈. **INT8 PTQ/QAT, depthwise 분리합성곱, 채널 프루닝**으로 FPGA BRAM/DSP 예산에 맞추는 것이 가장 효과적. (추정; 양자화 미구현 = 우리 작업 여지)
3. **이벤트 표현이 단순(누적 2극성 프레임)** → HW 친화적: voxel/SNN 대비 **고정 누적 버퍼 + 정수 increment**(`event_to_frame :286-291`)라 FPGA에서 간단한 누산기/히스토그램 블록으로 구현 가능. **E2F 변환 자체를 스트리밍 HW로** 옮기면 호스트-가속기 전송량 절감. (추정)
4. **타원 피팅의 행렬연산(3×3 고유값/역행렬)** (`ellipse.py:82-119`)은 **소형 고정크기 선형대수 → HLS로 완전 펼침(unroll) 가능**. 최소제곱 + 3×3 일반화 고유값은 FPGA 상수-크기 datapath로 결정적 지연 보장. (추정)
5. **저지연 검증 인프라 차용**: 코드에 이미 `timeit` 기반 CPU 지연 측정 훅(`:158-168, :127-133`)이 있어 **알고리즘↔HW 지연 비교 기준선**으로 활용 가능.
6. **시간 안정성을 RNN 없이 휴리스틱으로** 해결한 점은 HW에 유리(상태 레지스터 몇 개로 롤백 로직 구현). SSM/ConvLSTM 도입 부담 없이 저지연 유지. (확인된 설계 → HW 이식 용이성은 추정)
7. **개선 선결과제**: 위 §7의 glob 버그·반환 언패킹·경로 이식성·평가 메트릭 미구현은 **우리가 HW 매핑 전 알고리즘을 정상 동작/정량화하려면 먼저 패치**해야 함.

---

## 9. 근거표기 요약

| 항목 | 상태 | 근거 |
|---|---|---|
| 목적/논문/입출력 | 확인 | README.md:1-43, e_track.py 본문 |
| 이벤트 표현=누적 2극성 프레임 | 확인 | event_to_frame e_track.py:281-293 |
| U-Net 2-class 세그 | 확인 | e_track_unet.py:51-58, e_track_dataset.py:37 |
| loss=weighted CE [0.1,0.9] | 확인 | e_track_unet.py:36-61, e_track.py:267-318 |
| do_unet 절전 스위칭 + RoI 링필터 | 확인 | e_track.py:332-350 |
| 타원피팅(Halir-Flusser) | 확인 | util/ellipse.py:56-197 |
| 메트릭=mIoU + timeit 지연 | 확인 | e_track_unet.py:70,123,127-133 |
| 동공 위치오차(p-error) 산출 | 확인 불가(미구현) | e_track.py:511 return None |
| U-Net 정확한 채널/옵티마이저 | 확인 불가 | 외부 jakeret/unet 패키지 |
| AICAS=on-device 가속 의도 | 추정 | 학회 성격 + CPU 지연측정 + RoI 절감 |
| glob 제너레이터 버그 동작 | 추정(High 리스크) | e_track_unet.py:82-84 |
| 반환 언패킹 TypeError | 확인(코드 불일치) | e_track.py:511 vs 530 |

---
*(작성: 실제 소스 5개 파일 + 의존성/스크립트 전수 Read 기반. 외부 `unet` 패키지 내부 및 raw→tfrec 변환부는 repo에 부재하여 해당 항목만 "확인 불가" 표기.)*
