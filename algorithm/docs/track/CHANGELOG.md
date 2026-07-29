> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** algorithm

# algorithm CHANGELOG

`algorithm/**`의 변경 기록. 최신이 위.

> **소급 기록 고지.** 2026-07-24 ~ 2026-07-29의 33개 커밋은 이 파일이 존재하지 않아
> 기록되지 못했습니다 (`algorithm/docs/`에 `track/`이 없었고, 저장소 수준
> `docs/track/CHANGELOG.md`는 2026-07-23에 멈춰 있었습니다). 아래는 커밋 로그에서
> 소급 복원한 것이며, 각 항목의 근거는 해당 커밋 메시지입니다.

---

## 2026-07-29

### R1–R4 — 문서가 코드와 어긋난 것을 고치고, 그 과정에서 결함 2건

- **R4** (`be61e8f`) rsqrt 세그먼트 3개는 **불필요**로 확정. 관측 분산 동적범위 2:1 미만,
  잔여 rsqrt 오차 ~1e-4, 3세그먼트는 1.5–1.7배 개선 — 전체 오차 2e-2보다 네 자릿수 아래.
  **부수 결함**: 리포트가 "payload가 metrics를 실어 오차가 보인다"고 했으나 실제로는 계산
  후 폐기되어 export에 도달하지 않았음 → `ILayerNorm.metrics` + export 반영. 첫 뮤테이션이
  생존(`rms=0.0` 조작이 모든 상한 통과) → `rms >= max/√rows` 하한 추가.
- **R2** (`cc61c6a`) 변환 커버리지와 배포 커버리지 분리. `ConversionReport.auxiliary`/
  `deployed`. 어느 head가 aux인지는 **모델이 선언**(`DirectPupilDetector.AUXILIARY_HEADS`).
- **R3** (`bc5deda`) `max_tokens` 가드가 그래프에 배선됐음을 테스트로 확인. 위치 임베딩이
  없어 128px 입력이 물리적으로 가능(토큰 16→64).
- **R1** (`e7b86ac`) 사용자가 제일 먼저 읽는 파일 4곳의 거짓 주장 정정 + 은퇴 문구 재등장
  금지 테스트.

### A1 remainder · B1 · B2 · B6

- **TSR_FPGA 분석** (`9e65e14`) 외부 INT8 CNN 가속기 전수 조사. **학습된 PoT 스케일이
  손실 없음**(QAT 98.994% → int8 99.07%)을 확인 — 이 repo의 "pot 6배 악화"는 PTQ 사후
  반올림에 대한 결론이라 전이되지 않을 수 있음. → `docs/reference/COMPARISON-TSR-FPGA.md`
- **B2** (`91cba64`) 하이브리드 공유 스택 측정. 두 모달리티 range 최대 2.21배 차이,
  좁은 쪽이 최대 1.14비트 손실, 대신 클리핑 감소. **순 비용은 부호가 안정되지 않음.**
- **A1 remainder + B1** (`5c2b60e`) `IPatchEmbed`/`IViTBackbone`/`IPooledMlpHead`/
  `IEllipseHead` + `IDirectPupilDetector`. 전체 모델 오라클과 원소별 일치. ICat/IPool 실사용.
- **B6** (`61e46c6`) conv 누산이 shipped 설정에서 이미 부정확(1728 taps → 2.79e7 > 2^24).
  int64 im2col matmul로 교체.

## 2026-07-28

- **A1 step 2** (`e957d9c`) `ilayers/vit.py` — 블록 forward에 float 텐서 0개.
  오라클과 768/768 정수 일치(6시드), per-op 대비 2 LSB.

## 2026-07-27

- **A1 step 1** (`e1d9833`) `i_block.replay_block_int` — 순수 파이썬 블록 전체 오라클.
  자체 초안 버그 2건 즉시 검출(bias 누락 0.90, qkv 그리드 공유 3.2%).
- **B3/B4/B5** (`0c0abdd`) 재로드를 견뎌야 하는 상태, 0으로 읽히던 크기.
- **D5+D6** (`096deed`) HG-PIPE txt 렌더, QAT→정수 검증, Part C/D 리포트.
- **D4** (`95c78bb`) `models/blocks/seams.py` — 인라인이라 도달 못하던 IMatMul/IAdd가
  production에서 살아남음. 리팩터 무해함을 worktree 리비전 대조로 증명.
- **D3** (`c06cfeb`) 분할 rsqrt(새 10-스칼라 골든). 10.0–12.6배 개선, 테이블 메모리 절반.
- **D2** (`86f3b58`) IConv2d padding(zero_point로), ISoftmax `max_tokens` 런타임 가드.
- **D1** (`9b3ca98`) **실제 버그 3건**: 5개 entrypoint 전부 import 불가 ·
  `dataset/preprocess/__init__.py`가 없는 모듈 import · 하이브리드 PTQ 사망.
- **D0** (`78779f4`) per-channel 스케일 절단, 정수 정확성, 낡은 문서.
- **Part D 계획** (`3b82321`) §9 격차 해소 계획 + 범위 산정 중 발견한 라이브 버그 3건.
- **Part C5+C6** (`7a7b8c5`) 전체 그래프 Q→I 변환 + replay 기반 export.
- **Part C3** (`4298073`) 완전정수 ILayerNorm + ISoftmax.

## 2026-07-25

- **Part C0–C5** (`6762e74` `97a38d2` `8f614d6` `3684414` `8b82838` `1d45582`)
  QTensor + 정수 프리미티브 골든 · ILinear/IConv2d · IMatMul · IGeLU + IAdd/ICat/IPool ·
  Q→I 변환.
- **Part B** (`d65dbcf` `1725ec8` `1d0f288`) 완전지정 config — spec/grouping/AffineObserver,
  레이어별 override, config-matrix 검증 + PTQ 리포트.
- **Part A** (`665ae48`) q/i 티어 재구조화 (동작 변경 없음).
- **전체 계획 확정** (`a5b33fb`) `QUANTIZATION-PLAN.md`.

## 2026-07-24

- **Q9+Q10** (`fe5ae11`) LayerNorm + Softmax LUT 비선형 (HW-friendly 세트 완성).
- **Q8** (`743dce8`) HW-friendly GeLU LUT 비선형.

---

## 이전 (2026-07-23 이전)

저장소 수준 `docs/track/CHANGELOG.md` 참조. Q0–Q7(양자화 초기), 평면 기능별 재작성,
알고리즘 모듈 재구조화가 거기에 있습니다.
