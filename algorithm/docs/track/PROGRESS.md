> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** algorithm

# algorithm PROGRESS

계획 대비 진행 상황. **항목의 상태는 [TODO.md](TODO.md)가 단일 출처이고, 이 파일은 그것을
요약합니다.** 두 파일이 어긋나면 TODO.md가 맞습니다.

관련: [STATUS.md](../STATUS.md) (Active/Blocked/Next) · [CHANGELOG.md](CHANGELOG.md) (무엇이 바뀌었나)

---

## 양자화 서브시스템 — 계획 대비

계획 문서: [plans/active/2026-07-25-quantization-build-plan.md](../plans/active/2026-07-25-quantization-build-plan.md) ·
[plans/done/2026-07-27-quantization-part-d.md](../plans/done/2026-07-27-quantization-part-d.md)

| 단계 | 범위 | 상태 |
|---|---|---|
| Q0–Q7 | fake-quant 기반, 정수 커널 골든, PTQ/QAT, entrypoint | ✅ 완료 |
| Q8–Q10 | HW-friendly LUT 비선형 (GeLU / LayerNorm / Softmax) | ✅ 완료 |
| Part A | q/i 티어 재구조화 | ✅ 완료 |
| Part B | 완전지정 config (입도·sym/asym·scale_type·calibration) | ✅ 완료 |
| Part C | 정수 그래프 커널 (C0–C6) | ✅ 완료 |
| Part D | 도달 가능성·정확성·export (D0–D6) | ✅ 완료 |
| A1 | float 전송 제거 — 블록 + 모델 전체 | ✅ 완료 |
| B1 | ICat/IPool 실사용 | ✅ 완료 |
| B2 | 하이브리드 공유 스칼라 측정 | ✅ 완료 |
| B6 | conv 누산 정확성 | ✅ 완료 |
| R1–R4 | 문서-코드 정합, 배포 범위, 가드 배선, rsqrt 세그먼트 | ✅ 완료 |
| **A2** | **학습 체크포인트** | ⛔ **차단** — [STATUS.md](../STATUS.md) 참조 |

**검증 규모**: pytest **594 passed** (Part D 종료 시점 507).

## 지금 서 있는 자리

정수 그래프는 입력 포트 1회 양자화부터 출력 1회 역양자화까지 **완성**되었고,
`__torch_function__`으로 float 텐서 부재가 기계적으로 증명되어 있습니다.

**학습(A2)을 제외하면 양자화 서브시스템에 미결 항목이 없습니다.** 남은 것은 전부
A2에 의존하거나(학습된 PoT 측정, 실제 분산 분포에서의 rsqrt 재검토), 저장소 수준
백로그(게이트 걸림)입니다.

## 지켜지지 않은 것 — 기록

2026-07-24 ~ 2026-07-29의 **33개 커밋이 어떤 추적 문서에도 기록되지 않았습니다.**
`algorithm/docs/`에 `track/`이 없었고, 저장소 수준 `docs/track/`은 2026-07-23에
멈춰 있었기 때문입니다. 이 디렉토리가 그 해법이며, [CHANGELOG.md](CHANGELOG.md)에
소급 복원했습니다.

**규칙**: `algorithm/**`에 커밋할 때마다 이 파일과 CHANGELOG를 갱신합니다.
