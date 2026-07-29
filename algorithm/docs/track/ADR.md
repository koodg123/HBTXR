> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** algorithm

# algorithm ADR — Architecture Decision Record

**추가만 합니다. 기존 항목은 수정하지 않습니다.** 결정이 뒤집히면 새 항목을 쓰고 옛 항목에
`superseded by ADR-NNN`만 붙입니다 — 결정을 지운 기록은 왜 그렇게 했는지도 지웁니다.

> **소급 고지.** ADR-001 ~ ADR-011은 2026-07-24 ~ 07-29에 내려진 결정을 커밋에서 소급
> 복원한 것입니다. 당시 이 파일이 없었습니다.

---

## ADR-001 — 모든 정수 커널은 임의정밀도 순수 파이썬 골든과 **원소별 `==`** 로 검증한다
**2026-07-24 · 상태 active**

`i_ops.py`가 각 커널을 파이썬 무한정밀도 정수로 재구현하고, torch 커널을 그것과 `==`로
비교합니다. 허용오차(`allclose`)를 쓰지 않습니다.

**왜**: 허용오차는 통과할 때까지 넓어지고, 그 안에 잘못된 lowering이 숨습니다. 이 서브
시스템에서 실제로 세 번 일어났습니다 — 피연산자 스케일 교환, 누락된 requant, per-channel
벡터의 element-0 절단.

## ADR-002 — 모든 신규 테스트는 그것이 지키는 뮤테이션 하에서 RED임을 보인 뒤 채택한다
**2026-07-24 · 상태 active**

**왜**: 실패할 수 없는 테스트가 여러 개 발견됐습니다. 뮤테이션 게이트 도입 후 실제로
생존한 뮤테이션이 4건 있었고(IGeLU float64 revert, QAT/PTQ swap, per-channel 붕괴,
`rms=0.0` 조작), 전부 테스트가 약했던 것입니다.

## ADR-003 — 하드웨어용 스케일은 `pot`이 아니라 `dyadic`
**2026-07-25 · 상태 active, 단 ADR-011로 조건부 재검토**

Part B 측정: `dyadic ≈ float`(1.5% vs 1.4%), `pot`은 8.7%로 ~6배 악화.

## ADR-004 — LayerNorm rsqrt 인덱스는 2세그먼트, 세그먼트당 64엔트리
**2026-07-27 · 상태 active**

7-스칼라 커서는 분기가 없어 2세그먼트를 표현할 수 없으므로 **새 10-스칼라 골든**
(`layernorm_quantize_segmented`)을 만들었습니다. 기존 7-스칼라 골든은 그대로 둡니다 —
이미 export된 payload의 기준이기 때문입니다.

**측정**: 10.0–12.6배 RMS 개선, 테이블 메모리는 절반(2×64 = 128워드 < 1×256).
**2026-07-29(R4) 재확인**: 3세그먼트는 불필요.

## ADR-005 — conv 패딩은 0이 아니라 **활성 zero-point**로 채운다
**2026-07-27 · 상태 active**

비대칭 활성 격자에서 실수 0을 나타내는 정수는 zero-point이지 0이 아닙니다.
**측정**: `zp=17`인 3×3/pad-1 conv에서 zp 패딩 1.8e-07 vs 나이브 zero-pad 0.24.
손상이 정확히 패딩 링에만 나타나므로 경계를 건드리지 않는 프로브로는 볼 수 없습니다.

## ADR-006 — 인라인 연산자를 파라미터 없는 seam 모듈로 승격한다
**2026-07-27 · 상태 active**

`models/blocks/seams.py`의 `MatMul`/`Add`/`Scale`. 변환 패스가 도달할 수 있게 하려는
목적이며, `state_dict` 키와 float 출력이 bitwise 동일함을 리비전 대조로 증명했습니다.

## ADR-007 — 그래프 lowering 전에 **조합 오라클을 먼저** 만든다
**2026-07-27 · 상태 active**

`i_block.replay_block_int` → 그 다음 `ilayers/vit.py`.

**왜**: QTensor 그래프는 per-op 모델과 의도적으로 다른 함수이므로 둘의 비교는 허용오차밖에
없습니다. 오라클은 즉시 값을 했습니다 — 자체 초안에서 bias 누락(0.90)과 qkv를 Q 그리드로
눌러 클램프 후 분할(3.2%)을 잡았고, 둘 다 손으로 쓸 어떤 허용오차로도 보이지 않습니다.

## ADR-008 — 모든 스케일 상수는 **빌드 타임**에 확정한다
**2026-07-28 · 상태 active**

정수 bias, 채널별 dyadic `(multiplier, shift)`, `1/√d` 접기 — 전부 `__init__`에서.

**왜**: 최적화가 아닙니다. 호출마다 계산하면 그게 데이터패스 위의 float 연산입니다.
첫 초안이 정확히 그랬고 `__torch_function__` 프로브가 잡았습니다.

## ADR-009 — 회귀 헤드의 **마지막 linear은 requant하지 않는다**
**2026-07-29 · 상태 active**

int32 누산기 + per-out-channel 스케일을 그대로 내보내고 host가 곱합니다.

**왜**: 다른 모든 엣지는 소비자가 원하는 스케일을 선언하지만 이것의 소비자는 host입니다.
캘리브된 출력 격자가 없으므로 requant는 아무도 측정하지 않은 격자로 답을 반올림할 뿐입니다.

## ADR-010 — conv 누산은 float 스케줄이 아니라 **int64 im2col matmul**
**2026-07-29 · 상태 active**

float conv를 2^24 안에 들어가는 조각으로 나누는 방법도 정확했고 실제로 구현했다가 버렸습니다.

**왜**: 정확성은 요구사항의 절반입니다. int8×int8 MAC 배열은 정수 하드웨어이고, float
유닛을 거쳐 정답에 도달하는 커널은 정확해도 그 물건의 모델이 아닙니다. 또한 datapath 검사에
산술이 가장 어려운 자리에 예외를 뚫습니다. 실측 비용은 장당 2.7–6.3 ms.

## ADR-011 — "pot은 hardware에 부적합"은 **PTQ 경로에 한한 결론**
**2026-07-29 · 상태 active · ADR-003을 한정**

TSR_FPGA(외부)는 `log2_t`를 gradient로 학습해 PoT 스케일에서 손실이 없습니다
(QAT 98.994% → int8 99.07%). 이 repo의 `pot` 측정은 **캘리브된 스케일의 사후 반올림**
이었으므로 학습된-PoT QAT에 전이되지 않습니다.

**결정**: ADR-003은 PTQ 기본값으로 유지하되, 학습된 PoT는 A2 이후 별도 측정.
성립하면 requant 곱셈기가 사라집니다.
근거: [COMPARISON-TSR-FPGA.md](../../../docs/reference/COMPARISON-TSR-FPGA.md)

## ADR-012 — 어느 head가 auxiliary인지는 **모델이 선언**한다
**2026-07-29 · 상태 active**

`DirectPupilDetector.AUXILIARY_HEADS = ("mask_head",)`. 양자화기가 이름을 하드코딩하지
않습니다.

**왜**: 모델링 사실(mask head는 Sec III-D.1 aux supervision)이고, `convert.py`에 박으면
드리프트 가능한 사본이 됩니다. 변환 패스 자체는 범용으로 유지하고 **리포트만** 배포/aux를
구분합니다.
