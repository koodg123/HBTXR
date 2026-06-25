# event_based_gaze_tracking 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\event_based_gaze_tracking`
> 분석 도구: Glob / Grep / Read (bash 미사용)
> 작성일: 2026-06-20

---

## 0. 핵심 결론 (먼저 읽을 것)

**이 repo는 "모델/학습/추론 코드"가 아니라 "데이터셋 배포 + 시각화 데모" repo이다. (확정)**

전체 소스 파일은 단 4개뿐이다:

| 파일 | 역할 | LOC |
|------|------|-----|
| `README.md` | 논문/데이터셋/포맷 설명 | 110 |
| `setup.sh` | 27명 피험자 데이터셋 다운로드(Box 링크) | 116 |
| `visualize.py` | 이벤트/프레임 파싱·렌더링 데모 | 193 |
| `ebv-eye.yml` | conda 환경(=matplotlib, pillow뿐) | 9 |

따라서 **backbone / neck / head / loss / dataset(torch) / train / infer 루프는 이 repo에 "존재하지 않는다"(확인됨, 없음).** 사용자 분석 절차에서 요구한 "모델 정의·loss·데이터로더·학습/추론·후처리" 분석 항목은 *코드 부재로 분석 불가*이며, 본 문서에서는 (1) 실제로 존재하는 파싱/시각화 코드의 정밀 분석과 (2) 원논문 기준의 알고리즘 구조 *추정*을 명확히 구분하여 기술한다.

근거: `event_based_gaze_tracking\` 하위 `**/*.{py,ipynb,cpp,h,cu,sh,yml,yaml,cfg,txt,md}` Glob 결과 = README.md, setup.sh, visualize.py, ebv-eye.yml 4개만 매칭. 나머지는 `.git/`(빌드/메타, 제외)와 `misc/*.gif`(티저 영상, 제외)뿐.

---

## 1. 개요

### 1.1 목적
- 논문 **"Event Based, Near Eye Gaze Tracking Beyond 10,000 Hz"** (Angelopoulos*, Martel*, Kohli, Conradt, Wetzstein) 의 **공개 데이터셋**을 배포하고, 데이터 포맷을 정확히 파싱하는 최소 예제를 제공하는 것.
  - 근거: `README.md:1-3` 제목/저자, `README.md:100-108` BibTeX(`angelopoulos2020event`, arXiv:2004.03577).
- README가 명시적으로 밝힌 repo 범위: "This repository includes instructions for downloading and using our 27-person ... dataset." (`README.md:14`). 즉 데이터셋 + 사용법 중심이며 모델 코드는 범위 외(확정).

### 1.2 원논문 / 출처
- arXiv: https://arxiv.org/abs/2004.03577 (`README.md:2`).
- 저자 표기상 CVPR-W 2020 류 추정이나, repo 내부에는 "arXiv preprint"로만 적혀 있음 → 정확한 게재처는 이 repo만으로 **확인 불가(추정: 사용자 메모의 CVPR-W 2020과 일치 가능)**.

### 1.3 입력 (데이터 모달리티)
DAVIS 346b(iniVation) 센서로 동시 수집한 **좌/우 양안, IR 조명** 데이터 (`README.md:43-45`). 두 가지 모달리티:

1. **이벤트(event) 데이터** — `events.aerdat` raw binary.
   - 이벤트 1개당 필드: Polarity(극성), Timestamp(us), Row, Col (`README.md:56-60`).
   - **주의: README 본문 설명(`README.md:57-61`)과 실제 코드(`visualize.py:46`)의 바이트 포맷이 불일치**. (아래 3.2에서 상세)
2. **프레임(frame) 데이터** — `frames/` 디렉토리, ~25 FPS, 8-bit **346×260 px grayscale PNG** (`README.md:66-69`).
   - 파일명 포맷 `Index_Row_Column_Stimulus_Timestamp.png` (`README.md:70`).

### 1.4 출력 / 레이블
- 이 repo가 학습/추론을 하지 않으므로 "동공좌표/세그멘테이션 출력" 코드는 없음(확정).
- 단, **GT(정답) 신호**는 데이터에 내장: 프레임 파일명의 `Row`/`Column`이 *피험자가 그 순간 바라본 화면 자극점의 (행,열) 픽셀 좌표* = 시선(gaze) GT (`README.md:75-81`). 화면 중심은 (540, 960) (`README.md:77,81`).
- `Stimulus` 필드 = 자극 종류: `'s'`(saccade, 5초마다 랜덤 점프), `'p'`(smooth pursuit, 등속 이동), `'st'`(stop, 일시정지 심볼) (`README.md:83-86`).
- 즉 **출력 좌표계는 "동공 자체"가 아니라 "응시 화면 좌표(gaze point)"** 이다. 동공 좌표/세그먼트는 원논문 파이프라인의 중간 표현일 수 있으나 이 repo에는 없음(추정).

---

## 2. 디렉토리 구조 (자체 소스 vs 제외)

```
event_based_gaze_tracking/
├── README.md          [자체] 데이터셋/포맷/논문 문서
├── setup.sh           [자체] 27 subject 다운로드 스크립트
├── visualize.py       [자체] 유일한 실제 파이썬 소스 (파싱+시각화)
├── ebv-eye.yml        [자체] conda 환경(matplotlib, pillow)
├── LICENSE            [자체/제외] 라이선스
├── .gitignore         [자체/제외] eye_data/ 등 무시
├── misc/
│   └── github_event_based_eye_tracking_teaser.gif   [제외] 티저 영상(대용량 바이너리)
└── .git/              [제외] VCS 메타데이터 (hooks/objects/refs/logs ...)
```

- **자체 핵심 소스(분석 대상):** `visualize.py`, `setup.sh`, `README.md`, `ebv-eye.yml`.
- **제외:** `.git/`(VCS 산출물), `misc/*.gif`(대용량 미디어), `LICENSE`/`.gitignore`(부수 파일).
- `eye_data/`는 `.gitignore`에 등재(`.gitignore:1-4`)되어 repo에 미포함 — 실데이터는 `setup.sh`로 별도 다운로드. 현재 repo에는 실데이터 없음(확정).

---

## 3. 핵심 모듈 정밀 분석 (실제 존재 코드)

> NOTE: 요구된 backbone/neck/head/loss/train/infer는 코드 부재(섹션 0). 여기서는 **유일 소스 `visualize.py`의 모든 함수/클래스를 라인 근거로 정밀 분석**한다. 이것이 본 repo에서 "가장 중요한" 부분이다(데이터 I/O 규약 = 다운스트림 모델·HW 구현의 입력 계약).

### 3.1 자료형 정의
- `Event = namedtuple('Event', 'polarity row col timestamp')` (`visualize.py:26`)
- `Frame = namedtuple('Frame', 'row col img timestamp')` (`visualize.py:27`)
- `color = ['r', 'g']` — 극성 0→빨강, 1→초록 (`visualize.py:30`).

### 3.2 `read_aerdat(filepath)` — 이벤트 바이너리 파서 (가장 중요, 그리고 가장 함정)
`visualize.py:41-59`

- `packet_format = 'BHHI'` (`visualize.py:46`): `B`=uchar(polarity,1B), `H`=ushort(2B), `H`=ushort(2B), `I`=uint32(4B) → **패킷당 9바이트**.
- `packet_size = struct.calcsize('='+packet_format)` (`visualize.py:47`): `'='`(native byte order, no alignment)로 정확히 9바이트 계산.
- `num_events = len // packet_size`, 나머지 바이트 `extra_bits` 절삭 (`visualize.py:48-53`).
- 전체를 한 번에 `struct.unpack('=' + 'BHHI'*num_events, ...)` 로 언팩 → **flat list** 반환 (`visualize.py:56`). 즉 `[pol0,a0,b0,t0, pol1,a1,b1,t1, ...]` 형태(튜플 list가 아니라 평탄화된 단일 리스트).
- `event_list.reverse()` (`visualize.py:57`): 뒤에서 `pop()`으로 시간순 소비하기 위해 역순 정렬.

**불일치/함정 (Senior Reviewer 관점, High):**
1. README(`README.md:57-60`)는 필드 순서를 *Polarity, Timestamp, Row, Col* 및 타입을 (uchar, uint16, uint8, uint8)로 기술하지만, 실제 코드 포맷은 `BHHI`(=pol, ushort, ushort, uint32)로 **타입·순서가 다르다**. README의 "9 bytes" 결론(`README.md:61`)은 코드와 일치하나 중간 타입 서술은 코드와 모순. → 데이터 재현/HW 파서 작성 시 **반드시 코드(`BHHI`, 9B)를 진실의 원천으로 삼아야 함**(확정 근거: `visualize.py:46-47`).
2. `read_aerdat`는 (pol,a,b,t) 4-튜플로 묶지 않고 flat list를 반환하므로, 소비측(`__getitem__`)이 `pop()` 4회로 수동 디코딩한다(아래 3.4). 결합도 높고 오류 취약.

### 3.3 `get_path_info(path)` — 프레임 파일명 파서
`visualize.py:63-71`
- `path.split('/')[-1]` → 파일명만 추출 후 `.`로 확장자 제거, `_`로 split (`visualize.py:64-66`).
- 반환 dict: `index`(=part0), `row`(part1), `col`(part2), `stimulus_type`(part3), `timestamp`(part4) (`visualize.py:67-71`).
- 이 `row,col`이 곧 **gaze GT 좌표**, `timestamp`가 이벤트와의 **동기화 키**.
- **이식성 한계(Medium):** `split('/')` 하드코딩(`visualize.py:64`) → Windows 경로(`\`) 미지원. POSIX 전제.

### 3.4 `class EyeDataset` — 이벤트·프레임 시간정렬 컨테이너
`visualize.py:75-133`
- `__init__` (`:78-83`): `frame_stack`, `event_stack` 두 스택 보유.
- `__len__` (`:85-86`): 두 스택 길이 합(이벤트는 flat list라 실제 이벤트 수 아님 → 의미상 부정확, Info).
- `__getitem__` (`:88-105`): **이벤트/프레임 시간 병합(merge by timestamp)** 의 핵심.
  - `frame_timestamp = self.frame_stack[-1].timestamp` (`:90`), `event_timestamp = self.event_stack[-4]` (`:91`).
  - **주목:** `[-4]`로 인덱싱 — flat list에서 다음 이벤트의 timestamp가 끝에서 4번째라는 전제(역순 저장 + BHHI 4필드). 매우 암묵적·취약(Maintainability, Medium).
  - 이벤트가 더 이르면 `pop()` 4회로 polarity/row/col/timestamp 순서로 꺼내 `Event` 생성 (`:95-100`). **단, list는 역순이므로 `pop()`은 끝(=원래 앞)부터 꺼냄. 꺼내는 순서가 polarity,row,col,timestamp 인데 저장은 pol,a,b,t(BHHI) 순.** → row/col 의미 매핑은 데이터 생성 시 규약에 의존(코드만으로 row=H[0]/col=H[1] 확정 불가, 추정).
  - 프레임이 더 이르면 frame을 pop하고 `Image.open(...).convert("L")`로 grayscale 로드 후 `_replace(img=...)` (`:102-105`).
- `collect_data(eye)` (`:108-114`): 프레임→이벤트 순 로드, 개수 출력.
- `load_frame_data(eye)` (`:116-127`): `data_dir/user{N}/{eye}/frames/` 에서 이미지 glob → `index` 기준 정렬 후 `reverse()`(pop 소비용) → `Frame` list 구성. eye=0 좌, 1 우 (`README.md:48-49`).
- `load_event_data(eye)` (`:129-133`): `data_dir/user{N}/{eye}/events.aerdat` 를 `read_aerdat`로 로드.

### 3.5 `display_data(eye_dataset)` — 렌더 루프
`visualize.py:137-165`
- col/row/polarity 버퍼 축적 (`:138-140`), 빈 plot 핸들 `s` 초기화 (`:141`).
- 데이터 순회: `Frame`이면 `imshow`/`set_data`로 영상 갱신 (`:145-152`); `Event`면 버퍼에 누적, `len % opt.buffer == 0`마다 이전 scatter 제거 후 새 scatter 그림 (`:154-163`).
- `opt.buffer`(기본 1000, `:22`)가 한 번에 그리는 이벤트 묶음 크기 — 크면 빠르지만 blocky (`README.md:37`).
- **버그(Low):** `:147-148`에서 `init = True`로 오타(이후 분기 조건은 `init_img_axis`만 검사, `:146`) → `init_img_axis`가 갱신되지 않아 매 프레임 `imshow`를 다시 호출할 가능성(논리 결함). 또한 `:146-150` 들여쓰기가 탭/스페이스 혼용.

### 3.6 `main()` / CLI
`visualize.py:168-185`, 인자 `:16-23`
- `--subject`(기본 22), `--eye`(left/right), `--data_dir`(기본 `cwd/eye_data`), `--buffer`(1000).
- eye=left→`collect_data(0)`, right→`collect_data(1)` 후 `display_data` 호출.
- 실행 예: `python visualize.py --data_dir ./eye_data --subject 3 --eye left --buffer 1000` (`README.md:34`).

### 3.7 `setup.sh` — 데이터 획득
`setup.sh:1-116`
- `eye_data/` 생성 후 user1~user27 각각 Box static 링크에서 `wget`→`tar -xzf`→원본 `.tar.gz` 삭제 (`setup.sh:3-113`).
- 논문은 setup이 다른 subject 1-3을 제외하고 4-27만 사용(`README.md:25`)하지만 스크립트는 27명 전부 받음.
- 비결정성/링크 만료 위험: 외부 Box 정적 링크 의존(Reviewer, Medium). 체크섬 검증 없음.

---

## 4. 알고리즘 · 데이터 표현

### 4.1 실제 repo의 데이터 표현 (확정)
- **이벤트 표현:** raw AER 스트림(per-event tuple: polarity, x, y, t). **voxel/frame/time-surface 같은 텐서화는 이 repo에 없음(확정).** `visualize.py`는 시각화를 위해 단순 (col,row) scatter 누적만 수행(`:154-163`).
- **프레임:** 8-bit 346×260 grayscale PNG, ~25FPS (`README.md:67-68`).
- **동기화:** 이벤트 .aerdat의 us 타임스탬프와 프레임 파일명 us 타임스탬프가 DAVIS sync 커넥터로 양안·양모달 동기화(`README.md:88-90`). `EyeDataset.__getitem__`이 이 타임스탬프로 merge(`:90-94`).

### 4.2 원논문 기준 알고리즘 (이 repo에는 없음 — 전부 추정)
다음은 코드 근거가 아니라 논문 제목/일반 지식 기반 **추정**(repo에서 확인 불가):
- 프레임 기반 CNN으로 동공/시선 초기 추정 후, **이벤트로 프레임 사이를 고속(>10kHz) 보간/업데이트**하는 하이브리드 파이프라인일 가능성(추정).
- 동공 위치를 파라메트릭(예: 타원/2D 좌표)으로 추정하고 이벤트로 그 변화를 추적하는 model-based 접근 가능성(추정). ConvLSTM/SSM/Transformer 같은 시퀀스 백본 사용 여부는 **이 repo로 확인 불가**.
- 메트릭(p-error, IoU 등)·후처리 코드: 이 repo에 없음(확정). 논문 본문 별도 확인 필요.

---

## 5. 학습 · 평가

- **학습 코드:** 없음(확정). loss/optimizer/epoch 루프 부재.
- **평가 코드/메트릭:** 없음(확정). p-error/IoU/정확도 계산 코드 없음.
- **데이터셋:** 27인 양안 event+frame, DAVIS 346b, IR 조명; 논문은 subject 4-27 사용(`README.md:24-25,43-45`).
- **재현 명령어 (이 repo가 제공하는 전부):**
  1. 환경: `conda env create -f ebv-eye.yml` (`README.md:19`)
  2. 데이터: `bash setup.sh` (`README.md:27`)
  3. 시각화: `python visualize.py --data_dir ./eye_data --subject 3 --eye left --buffer 1000` (`README.md:34`)

---

## 6. 의존성
- `ebv-eye.yml:4-6`: **matplotlib, pillow** 뿐 (channels: defaults).
- `visualize.py` import: argparse, struct, glob, os, matplotlib.pyplot, PIL.Image, collections.namedtuple (`:8-14`).
- **딥러닝 프레임워크(torch/tensorflow) 의존성 전무** → 모델 코드 부재의 결정적 정황 근거.
- 외부 인프라: `setup.sh`의 `wget` + UC Berkeley Box 링크.

---

## 7. 강점 · 한계

### 강점
- 데이터 포맷 규약(.aerdat 9B 패킷, 프레임 파일명 스키마)이 명확히 문서화(`README.md:54-96`).
- 파싱 최소 예제 제공으로 데이터 진입장벽 낮음(`visualize.py`).
- 양안·이벤트·프레임 동기화된 27인 near-eye 데이터셋 자체가 희소 자원.

### 한계 (심각도 표기)
- **[High] 모델/학습/추론/메트릭 코드 전무.** 알고리즘 재현 불가 — 데이터만 제공.
- **[High] README 이벤트 타입 서술(`:57-60`)과 코드 포맷 `BHHI`(`:46`) 불일치.** 파서/HW 구현 시 코드 기준 필수.
- **[Medium] `__getitem__`의 `[-4]` 매직 인덱스, flat-list pop 디코딩** — 가독성/안전성 취약(`:91,95-98`).
- **[Medium] POSIX 경로 하드코딩 `split('/')`** (`:64`) — Windows 비호환.
- **[Low] `display_data` init 플래그 오타**(`:147-148`) — 프레임 축 재초기화 가능 버그.
- **[Medium] `setup.sh` 외부 정적 링크 의존·체크섬 없음** — 재현성 위험.

---

## 8. 우리 프로젝트 시사점 (PRJXR-HBTXR: "XR 시선추적 + FPGA 저지연 on-device 가속" — 추정)

> 본 repo는 *알고리즘이 아니라 데이터/입력 계약*을 제공. 따라서 우리 프로젝트에는 "모델 참조"가 아니라 "데이터 인터페이스 사양·HW 입력 파이프라인 설계 근거"로 활용.

1. **이벤트 입력 포맷 = HW 파서 사양 (직접 이식 가치 높음).**
   - DAVIS 9바이트 패킷(`BHHI`: pol 1B + x 2B + y 2B + t 4B, `visualize.py:46-47`)을 FPGA 입력 디코더(스트리밍 9B 정렬 파서)로 그대로 구현 가능. `extra_bits` 절삭 로직(`:49-53`)도 HW에서 잔여 바이트 처리로 대응.
   - 권고: README가 아닌 **코드 포맷을 HW 명세의 ground truth**로 채택.

2. **이벤트→텐서 표현은 우리가 직접 설계해야 함.**
   - 이 repo엔 voxel/time-surface 변환이 없으므로, 저지연 on-device용 **이벤트 누적 표현(time-surface/event-frame/voxel grid)** 을 우리가 정의하고 그 변환을 FPGA에 매핑해야 한다. scatter 누적 버퍼(`opt.buffer`, `:157`)는 "window/슬라이딩 누적" HW 버퍼의 개념적 출발점(경량 누적기).

3. **양모달 시간정렬을 HW 타임스탬프 머지 로직으로.**
   - `__getitem__`의 timestamp 비교 머지(`:90-94`)는 이벤트/프레임 두 스트림을 us 단위로 동기화하는 패턴. FPGA에서 두 FIFO를 타임스탬프 비교로 머지하는 저지연 스케줄러로 이식 가능.

4. **GT/메트릭은 외부에서 가져와야 함.**
   - gaze GT는 프레임 파일명(`row,col`, 화면중심 540/960)에 내장(`README.md:75-81`). 학습/평가 파이프라인(p-error 등)은 본 repo 밖(원논문/타 repo)에서 확보 필요.

5. **경량화·양자화·FPGA 이식 관점:**
   - 본 repo는 부동소수 모델이 없어 양자화 대상 코드가 없음. 우리 가속기 설계에서 이 데이터셋을 *벤치마크 입력*으로 쓰되, 모델은 별도(예: 논문 재구현 또는 다른 backbone)에서 가져와 INT8/경량화 후 FPGA에 올리는 구조가 현실적(추정).

---

## 9. 근거 표기 요약

| 주장 | 상태 | 근거 |
|------|------|------|
| 모델/학습/loss/메트릭 코드 없음 | 확정 | Glob 결과 4파일, torch 의존성 0 (`ebv-eye.yml:4-6`) |
| 이벤트 패킷 = 9B(BHHI) | 확정 | `visualize.py:46-47` |
| README 타입 서술 ≠ 코드 | 확정 | `README.md:57-60` vs `visualize.py:46` |
| gaze GT는 프레임 파일명에 내장 | 확정 | `README.md:70-81` |
| CVPR-W 2020 게재처 | 추정 | repo엔 "arXiv preprint"만 표기(`README.md:2`) |
| ConvLSTM/SSM/Transformer 백본 사용 | 확인 불가 | repo에 모델 코드 없음 |
| voxel/time-surface 표현 사용 | 확인 불가(repo엔 없음) | `visualize.py`는 raw scatter만 |
| >10kHz 하이브리드 보간 파이프라인 | 추정 | 논문 제목 기반, 코드 부재 |
