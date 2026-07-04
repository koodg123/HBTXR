# Search / Track Invocation — 산출 기준 (per-subject, Train/Val/Test)

하이브리드 search–track 추적기의 모드를 분석된 4-state 분포로 환산:
- **Track-mode (이벤트 연속추적)** = Fixation + Smooth  (동공 존재·완만/연속 운동 → 이벤트 트래커가 연속 처리)
- **Search-mode (재검출/재탐색)** = Saccade + Blink  (추적 손실·불안정 → 재검출 필요)
- Ratio = Track : Search (= Track_inv / Search_inv)

두 기준(basis):
- **Event basis**(권장): 측정된 event 4-state(20ms bin) — saccade·blink를 모두 포착. Search ≈ 17.5%, Track ≈ 82.5%, Track:Search ≈ 4.7.
- **Frame basis**: frame 4-state — saccade≈0이라 Search≈Blink만(~2%), Track:Search ≈ 43–50.

## 주의 (정직성)
- 이 수치는 **4-state 분포 기반 모드-워크로드 추정**이며, 실제 트래커의 invocation 로그가 아님. 실제 로그는 EX-Gaze end_to_end_tracking pickle이 user48/left/201 1세션만 디스크에 존재(img_detect=6, img_similarity=45, ev_tracking=919, ev_accum=3161 → 실제 frame 재검출(search)은 매우 드묾).
- 따라서 **Event basis(~17.5%)** = 추적이 어려워지는 상태의 시간점유(상한 워크로드), **Frame basis(~2%)** = 실제 frame 재검출 빈도에 가까운 하한. 실제 invocation은 이 사이.
- Split: Train 1–36 / Val {37,40,45,48}⊂Test / Test 37–48.
