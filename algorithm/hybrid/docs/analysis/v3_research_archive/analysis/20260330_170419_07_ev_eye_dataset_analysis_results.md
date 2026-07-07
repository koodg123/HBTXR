# EV-Eye 데이터셋 분석 결과

최종 갱신: 2026-03-28 KST

Role: `데이터셋 분석 엔지니어`, `이벤트 카메라 비전 연구자`, `수치 검증 엔지니어`

## 요약

- 이 문서는 EV-Eye `Data_davis`에 대한 검증된 session / frame 집계를 정리한다.
- `valid 384`와 `clean 366`은 공식 split 이름이 아니라 운영 분석용 working set이다.
- `valid 384`는 raw inventory 집계에 사용하는 standard-layout readable set이다.
- `clean 366`은 frame-event alignment, interpolation planning, preprocessing 가정에 사용하는 interval-count-safe subset이다.

## 데이터셋 루트

- Raw dataset
  - `E:/WSL/Shared/dataset/Eye/EV_Eye/raw_data/Data_davis`

## 정의

### Total 388

- `user*/left/session_*`
- `user*/right/session_*`

아래 broad session 검색 규칙으로 찾은 전체 세션 집합이다.

### Valid 384

- 표준 레이아웃으로 직접 읽을 수 있는 세션 집합
  - `session/frames/timestamps.txt`
  - `session/events/events.txt`
- raw frame / session inventory 집계에 사용한다.

### Clean 366

- `valid 384` 중 frame-to-frame interval event counting이 정상 동작하는 subset
- 추가 제외 규칙
  - timestamp regression 세션
  - 모든 interval count가 0인 세션

## 상위 집계

| 집합 | 세션 수 | 프레임 수 | 의미 |
|---|---:|---:|---|
| Total | 388 | 1,506,387 | non-standard timestamp layout까지 포함한 broad 집계 |
| Valid | 384 | 1,490,548 | 표준 레이아웃으로 읽을 수 있는 세션 |
| Clean | 366 | 1,411,579 | `valid 384`에서 regression / all-zero interval 세션을 제외한 집합 |

## 집합 관계

```text
388 total
- 4 non-standard layout sessions
= 384 valid

384 valid
- 16 timestamp regression sessions
- 2 all-zero interval sessions
= 366 clean
```

## `valid 384`에서 제외된 non-standard layout 세션

이 세션들은 `events/events.txt`는 존재하지만 timestamp 파일이 표준 위치 `frames/timestamps.txt`에 없다. 대신 `events/frames/timestamps.txt` 아래에 존재한다.

| 세션 | 프레임 수 | 제외 사유 |
|---|---:|---|
| `user14/left/session_3_0_1` | 5,070 | non-standard timestamp layout |
| `user14/left/session_3_0_2` | 2,873 | non-standard timestamp layout |
| `user14/right/session_3_0_1` | 5,047 | non-standard timestamp layout |
| `user14/right/session_3_0_2` | 2,849 | non-standard timestamp layout |

합계:

- 세션: `4`
- 프레임: `15,839`

## `clean 366`에서 제외된 timestamp regression 세션

이 세션들은 `valid 384`에는 포함되지만, 전체 시퀀스 기준 frame timestamp가 단조 증가하지 않는다. 따라서 frame-interval event counting이 깨진다.

| 세션 | 프레임 수 | 제외 사유 |
|---|---:|---|
| `user12/left/session_1_0_2` | 2,792 | timestamp regression |
| `user12/right/session_1_0_2` | 2,768 | timestamp regression |
| `user23/right/session_1_0_1` | 4,966 | timestamp regression |
| `user27/left/session_1_0_1` | 5,039 | timestamp regression |
| `user27/right/session_1_0_1` | 5,009 | timestamp regression |
| `user29/left/session_1_0_1` | 4,999 | timestamp regression |
| `user32/left/session_1_0_1` | 5,032 | timestamp regression |
| `user32/right/session_1_0_1` | 5,003 | timestamp regression |
| `user41/right/session_2_0_1` | 4,977 | timestamp regression |
| `user43/left/session_2_0_2` | 2,834 | timestamp regression |
| `user43/right/session_2_0_2` | 2,813 | timestamp regression |
| `user44/left/session_1_0_1` | 4,990 | timestamp regression |
| `user44/right/session_1_0_1` | 4,963 | timestamp regression |
| `user46/left/session_2_0_1` | 5,035 | timestamp regression |
| `user46/right/session_2_0_1` | 5,011 | timestamp regression |
| `user48/left/session_2_0_1` | 4,989 | timestamp regression |

합계:

- 세션: `16`
- 프레임: `71,220`

## `clean 366`에서 제외된 all-zero interval 세션

이 세션들은 `valid 384`에는 포함되지만, 모든 frame interval의 event count가 0이다. 운영적으로는 frame timestamp와 event timestamp가 서로 다른 time band에 있어 interval counting이 의미를 잃는다.

| 세션 | 프레임 수 | 제외 사유 |
|---|---:|---|
| `user9/right/session_2_0_1` | 4,980 | all interval counts are zero |
| `user9/right/session_2_0_2` | 2,769 | all interval counts are zero |

합계:

- 세션: `2`
- 프레임: `7,749`

## 검증 등식

- `388 total - 4 non-standard = 384 valid`
- `384 valid - 16 regression - 2 all-zero = 366 clean`
- `1,490,548 - 71,220 - 7,749 = 1,411,579`

## interpolation 프레임 수 공식

한 세션의 raw frame 수가 `n`이고 target frame-rate multiplier가 다음과 같다면:

- `r = target_fps / 25`

해당 세션의 interpolated frame 수는 다음과 같다.

- `r * (n - 1) + 1`

전체 집합에 대해선:

- `Total = r * N_total - (r - 1) * S`

여기서:

- `N_total`은 전체 raw frame 수
- `S`는 세션 수

## 예시 총합

### Valid 384

| Target FPS | Multiplier `r` | 총 프레임 수 |
|---|---:|---:|
| 500 | 20 | 29,803,664 |
| 1000 | 40 | 59,606,944 |
| 2000 | 80 | 119,213,504 |

### Clean 366

| Target FPS | Multiplier `r` | 총 프레임 수 |
|---|---:|---:|
| 500 | 20 | 28,224,626 |
| 1000 | 40 | 56,448,886 |
| 2000 | 80 | 112,897,406 |

## 권장 사용처

- `valid 384`
  - raw dataset inventory
  - 저장공간 추정
  - standard-layout availability accounting
- `clean 366`
  - interval event statistics
  - interpolation planning
  - target-FPS 설계 분석
  - 시간축 일관성에 의존하는 preprocessing 가정

## 참고 메모

- `valid 384`, `clean 366`은 분석용 working set이다.
- 공식 EV-Eye split 이름이 아니다.
- `16 regression sessions`는 아래 레거시 기술 노트와도 정렬된다.
  - [16_ev_eye_dataset_v1_technical_note.md](../../legacy/docs/version2_plan/16_ev_eye_dataset_v1_technical_note.md)
