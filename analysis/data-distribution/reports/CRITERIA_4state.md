# 4-state 분류 기준 (Frame / Event) + Train/Val/Test 결과

States: **Fixation / Saccade / Smooth pursuit / Blink**.
Split: **Train = subjects 1–36, Val = {37,40,45,48}, Test = 37–48** (EX-Gaze config 기준; Val ⊂ Test 중첩 주의).
세션→운동: pattern 1(…_0_1)=saccade 세션(101/201), pattern 2(…_0_2)=smooth 세션(102/202).

## A. Frame 스트림 기준 (25Hz, 40ms)
동공 중심 = UNet 예측 마스크(타원 적합) 중심. 프레임별:
1. **Blink** = 동공 부재(UNet가 타원/마스크를 못 찾음, `skip_no_ellipse`). 전 프레임 기준 카운트(보간 없음).
2. **Saccade** = 눈뜸 AND 중심 속도 `speed > v_sacc = 493 px/s` (I-VT; v_sacc=saccade-세션 속도 90퍼센타일).
   - **25Hz에서는 saccade(30–80ms)가 프레임 사이에 끝나 거의 검출 안 됨 → ≈ 0%** (degenerate).
3. **Fixation** = 눈뜸 AND 비-saccade AND saccade-세션(101/201).
4. **Smooth** = 눈뜸 AND 비-saccade AND smooth-세션(102/202).
- 출처: `DeanDataset_full_unet/progress_state.json` (세션별 frames/valid/skip_no_ellipse).

## B. Event 스트림 기준 (원시 events.txt, 20ms bin)
빠른 운동=이벤트 급증. 20ms bin마다 `rate`(ev/s), 세션중앙 `m`:
1. **Blink** = `rate > 12·m` (극단 버스트; 눈꺼풀). *근사 — 이벤트만으론 과대추정 경향*.
2. **Saccade** = `4·m < rate ≤ 12·m` (버스트; μs~20ms 해상도라 saccade 포착).
3. **Fixation** = 비버스트(`rate ≤ 4·m`) AND saccade-세션.
4. **Smooth** = 비버스트 AND smooth-세션.
- 20ms bin 이유: saccade보다 짧고(검출), 노이즈보다 길고(안정), 프레임 40ms의 반주기(정합).

## C. 보간 후 (EMULATED — 실 보간 데이터 없음, 가정 명시)
실제 200Hz 프레임보간/이벤트보간 데이터가 디스크에 없어 **가정 기반 모델링**입니다:
- **가정**: 200Hz 프레임 보간 + 이벤트 densify ⇒ 프레임이 이벤트처럼 saccade를 해상.
- **규칙**: Blink = 프레임-UNet blink(불변, 신뢰) · 비-blink 구간을 **이벤트 비율(Fix:Sacc:Smooth)**로 재분배.
- ⚠️ 보간 프레임은 추정(ballistic saccade 평활화·오염 위험)이므로 GT/평가용 아님 → **입력 증강 한정**, 평가는 실데이터.

## D. Split별 결과 요약 (share %)
| 조건 | split | Fixation | Saccade | Smooth | Blink |
|---|---|--:|--:|--:|--:|
| Frame(전) | Train/Val/Test | 63.2/62.7/62.9 | **0.0/0.0/0.0** | 34.5/35.3/35.2 | 2.3/2.0/2.0 |
| Event(전) | Train/Val/Test | 54.0/51.8/51.6 | **8.4/8.7/8.6** | 28.5/30.8/30.2 | 9.1/8.7/9.6 |
| After(emul) | Train/Val/Test | 56.9/55.6/56.0 | **9.3/9.3/9.3** | 31.5/33.1/32.7 | 2.3/2.0/2.0 |

핵심: **Frame은 보간 전 Saccade≈0%(학습 불가)이나, 보간(가정) 후 ~9.3%로 학습 가능 수준**. Event는 보간 전부터 ~8.5%.

## 파일
표(비율 포함): `split_4state_{frame_before,event_before,after_emulated}.csv`, `split_4state_summary.csv`
그림: `fig_split_4state_{frame_before,event_before,after_emulated}.png`
코드: `src/event_4state_all.py`, `src/build_splits_4state.py` (+ event/frame 기준 스크립트)
