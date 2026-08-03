> **작성** 2026-07-05 · **갱신** 2026-07-05
> **상태** active — 내용은 2026-07-15 구 트리에서 이어받음, 재구성 진행분을 여기에 이어 씁니다
> **소유** hardware

# HGTXR Hardware TODO

## S10 — HLS rewrite, S9 다음 (2026-08-03 등록)

**우선순위는 S9 측정이 정했습니다** — 근거는 전부
[reports/2026-08-03-s9-csynth.md](../reports/2026-08-03-s9-csynth.md).
아래 P0~P3 절과 달리 이 항목들은 **보드 접근을 기다리지 않습니다.**

### S10-1 — `dataflow` 재구조화 + 복사 루프 제거 · **먼저**

지금 구성이 ZCU104 에 **안 들어갑니다**: 코어쌍 2 + 스템 2 + 헤드 2 = LUT **350%** ·
DSP **265%**, MHA 코어 **하나**가 LUT 98%.

- [ ] `push2d`/`pop2d` 제거. MHA 코어 DSP 1,355 중 **960개**가 이 복사 루프의 주소 곱셈.
      **국소 수정 2건은 이미 측정으로 기각**했으니 다시 하지 말 것 —
      행 베이스 hoist(DSP 불변) · `cols` 템플릿화(DSP −71, LUT **+68k**, 디바이스 초과).
      `hbtxr_mha_core.hpp` 주석에 기록되어 있습니다
- [ ] `HbtxrMhaCore::run` 을 dataflow 정규형으로. 지금 어기는 것 셋:
      헤드 루프가 `qkv_buf` 를 H=3 번 다시 읽음 · `x` 를 stage 1 과 9 가 둘 다 읽음(바이패스) ·
      헤드 루프 안에서 스트림 재선언
- [ ] FIFO 깊이 확정. **실측 최대 1,536 beat**(= MLP 은닉 F=768 전체, 24 KB). 이 숫자는
      dataflow 를 **쓰지 않을 때**의 비용이고, 겹쳐 돌면 훨씬 얕아집니다 (SPEC §5-1)

### S10-2 — DSP 패킹 / 병렬도

- [ ] RMU 하나가 `TP×CIP×COP = 256` MAC 에 DSP 160 (**1.6 MAC/DSP**). 4비트 곱 4개를
      DSP48E2 하나에 넣거나 전부 LUT 로 내리면 4배. SPEC §2-A 의 `TP`/`CIP`/`COP` 를
      줄이는 것도 같은 축 — **어느 쪽이든 S9 숫자가 근거입니다**

### S10-3 — 검증 층

- [ ] **V4 cosim.** 위 FIFO 깊이가 실제로 데드락하지 않는지 보는 **유일한 층**입니다.
      csim 은 `hls::stream` 이 무한이라 못 봅니다
- [ ] **V2 csim.** 툴 프런트엔드가 g++ 와 같은 값을 내는지. 가설이 아닙니다 — S9 에서
      2023.2 프런트엔드가 C++17 을 거부하는 실제 차이가 나왔습니다

### S10-4 — 보드 인터페이스

- [ ] `HbtxrTop`·`HbtxrBackbone` 합성. **합성 가능한 weight SOURCE 가 필요**하고 그건
      산술이 아니라 인터페이스 결정입니다 (M-AXI)
- [ ] 가중치 적재 경로. 지금 II=2 · qkv 적재 **221,186 사이클 = 0.74 ms**/블록 →
      8블록 **6 ms**. 8-wide 워드로 넣으면 1/16
- [ ] 잔여 II 미달 2건 — `smu_ctx` 의 B 전치 write (II=2, dim 2 reshape 된 메모리의 서로
      다른 행 8개에 씀) · `patch_f` windower (II=4, Conv-E 는 2)

### S10-5 — 결정 대기 (사용자)

- [ ] **엣지별 dtype.** `BlockSpec.dtype` 이 하나라 블록 안에서 **matmul 4비트 + 비선형
      16비트**를 동시에 표현 못 합니다. GeLU LUT 입력 알파벳이 16개로 붕괴.
      현재 `-a4`/`-a8` 두 벌로 우회 중이고 **S4 전부터 미결**입니다

### 범위에서 뺀 것 (기록용)

- **Table III 재현** — [SPEC §10](../SPEC.md)
- **보조 헤드 3종** (ROI + Reliability ×2). 배포 모델 `HybridModel` 에는 셋 다 있으므로
  나중에 붙여야 합니다. `ReliabilityHead` 의 sigmoid 는 **온칩에 둘 필요가 없습니다** —
  단조라 로짓을 `sigmoid⁻¹(threshold)` 와 비교하면 됩니다

---

> ### ⚠️ 아래 항목의 경로는 **구 트리 기준**입니다 (2026-07-29 확인)
>
> 이 항목들은 2026-07-05에 쓰였고, 그 뒤 구 트리가 `archive/hardware/`로 옮겨졌습니다.
> **지시대로 실행하면 실패합니다.** 지금의 대응은:
>
> | 항목에 적힌 것 | 지금 어디 |
> |---|---|
> | `tools/run_zcu104_c3b_smoke_remote.py` | `archive/hardware/tools/run_zcu104_c3b_smoke_remote.py` |
> | `pynq/hgtxr/` | 없음. 런타임 코드는 `deploy/`로 갑니다 (**아직 이관 전**) |
> | `generated/` | **이관되지 않았습니다.** `archive/`에도 없습니다 — 생성물이라 커밋되지 않았습니다 |
>
> 본문을 고치지 않는 이유: 항목의 내용(무엇을 해야 하나)은 여전히 유효하고, 경로는
> 코드 이관(계획 §6 P0~P8)이 끝나면 새 위치로 한 번에 갱신됩니다. 그 전에 개별로 고치면
> 두 번 고치게 됩니다. **어차피 전부 보드 접근 대기이므로 지금 실행할 수 없습니다**
> ([../STATUS.md](../STATUS.md) Blocked).

## P0
- Capture/import C3b AXIS/DMA physical smoke JSON.
- Resolve exact XR-VITs sibling or approved replacement policy.
- Run/import ZCU104 physical smoke JSON for `softmax_input_x2` `dsp_mixed_stream` using either `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md` or `python3 tools/run_zcu104_c3b_smoke_remote.py --profile vref-p0-softmax-input-x2-dsp-mixed-stream --host <zcu104-ip-or-host> --user xilinx --execute`.
- Copy successor board result back to `pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json` so preflight/projection can promote it.
- Keep C3b as protected baseline until board-smoke and XR-VITs reference gates are closed.
- For additional non-current `VREF-P0-01` scale successors, regenerate a dedicated golden header and pass CSim before promotion.
- Keep `VREF-P0-02` QKV cache URAM promotion as a successor-only experiment; HLS CSim/csynth, HLS IP package, routed overlay timing, and PYNQ bundle/session artifacts are present; physical smoke remains pending.

## P1
- Check nonlinear bottleneck before TATAA mixed-precision path.
- Prepare search/track expert ablation only as bounded/static routing.
- Prepare exact-attention fallback for ViTALiTy/ViTCoD ablations.

## P2/P3
- Keep FlexLLM-style generator as tooling-only.
- Keep LUT-heavy/ternary methods as negative controls.
