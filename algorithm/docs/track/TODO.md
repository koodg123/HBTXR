> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** algorithm

# algorithm TODO

`algorithm/**`의 미결 항목. **항목 상태의 단일 출처입니다** — [PROGRESS.md](PROGRESS.md)는
이것을 요약할 뿐이고, 어긋나면 이 파일이 맞습니다.

저장소 수준 백로그(T-/SI-/RC-/CL-/AM-)는 [../../../docs/track/TODO.md](../../../docs/track/TODO.md)에 있습니다.

---

## 차단됨 (Blocked)

- [ ] **A2 — 학습 체크포인트.** manifest 빌드 + 실제 학습 실행 필요. 레거시 HGTXR
      체크포인트는 state-dict 키가 하나도 일치하지 않고 `load_checkpoint`는 `strict=True`.
      **차단 해제: 사용자 결정.** 이것이 풀리기 전까지 이 저장소의 모든 정확도 수치는
      랜덤 초기화 모델 기준입니다.
- [ ] **학습된 PoT 스케일 측정.** `log2_t`를 학습 파라미터로 두고 `ceil` STE로 통과.
      성립하면 requant 곱셈기가 사라짐. **A2 의존.**
      근거: [COMPARISON-TSR-FPGA.md](../../../docs/reference/COMPARISON-TSR-FPGA.md) §6①
- [ ] **rsqrt 세그먼트 재검토.** 현재 2세그먼트로 충분함을 측정했으나(R4) 랜덤 가중치
      기준. 학습된 모델의 분산 동적범위가 넓으면 재개. **A2 의존.**

## 즉시 가능 (A2 무관)

- [ ] **입력 활성 그리드를 `2⁻⁸`로 고정.** 이미지가 uint8이면 `1/256` 격자가 공짜로
      정확합니다. 현재 patch-embed conv의 활성 스케일을 관측으로 캘리브하는데, 입력만큼은
      격자가 이미 정해져 있습니다. `QuantScheme.overrides`로 표현 가능.
- [ ] **첫/마지막 레이어 8비트 고정을 기본 정책으로.** 기능(`overrides`)은 있으나 정책 없음.

## 조사 필요

- [ ] **`i_ops` ↔ `archive/hardware/hls/include/hgtxr_cyclic_math.hpp` 대조.** 둘 다 HG-PIPE에서
      온 같은 커널(2세그먼트 reciprocal, dyadic requant, PoT 인덱스 테이블)인데 교차 참조가
      0건이고 수치 규약이 다릅니다(ap_fixed<16,6> vs int8 affine, HLS LayerNorm에 rsqrt 없음,
      softmax 출력 3비트 vs int8). `i_ops.table_quantize`에 HLS의 `(b,s,bound)`와 테이블을
      먹여 원소별 대조 가능. → `docs/contracts/NUMERICS-CONTRACT.md`로 산출

## 알려진 제약 (조치 불필요, 기록용)

- **mask head는 정수화하지 않습니다.** 추론 경로에 없는 stage-1 aux supervision이고
  `HybridModel`에는 존재하지도 않습니다. `IDirectPupilDetector.float_io_heads`에 명시.
- **conv 활성은 대칭 캘리브**이므로 `IConv2d`의 zero-point 보정·패딩 경로는 변환에서
  실행되지 않습니다. 비대칭 그리드가 캘리브되면 필요한 경로이고, 직접 테스트로 덮여 있습니다.
- **`Scale`은 의도적으로 float** — 정확한 상수 곱(shipped `head_dim=64`에서 `2⁻³`)이라
  양자화하면 오차만 늘어납니다.
