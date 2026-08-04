> **작성** 2026-08-03 · **갱신** 2026-08-03
> **상태** active — 검증됨
> **소유** hardware

# AXI 페이로드 계약 — 호스트 ↔ 커널

**[SPEC §8](../SPEC.md)이 포트를 정하고, 이 문서가 그 위에 무엇이 흐르는지를 정합니다.**

```cpp
void hbtxr_top(hls::stream<hbtxr_axis_t> &in_stream,     // AXIS   <- 이미지
               hls::stream<hbtxr_axis_t> &out_stream,    // AXIS   -> 결과 5개
               const volatile hbtxr_axi_word_t *weights, // M-AXI  <- 가중치 blob
               int mode, const hbtxr_anchor_t anchor[5], int *status);
```

| 끝 | 파일 |
|---|---|
| 호스트 | [`deploy/hbtxr/payload.py`](../../deploy/hbtxr/payload.py) |
| 커널 | [`module/include/hbtxr_payload.hpp`](../../module/include/hbtxr_payload.hpp) |
| 증명 | [`module/tb/tb_payload.cpp`](../../module/tb/tb_payload.cpp) — 러너에 연결됨 |

---

## 0. 레이아웃은 **파일이 들고 다닙니다**

blob 안에 **섹션 테이블**이 있고 양쪽이 그것을 읽습니다. **어느 쪽도 오프셋을 계산하지
않습니다.**

이 프로젝트가 단계 하나씩 써서 잡은 버그는 전부 *한쪽만 보고 판단한* 것이었습니다 —
툴에 닿은 적 없는 `array_reshape`, 자기가 재려던 데이터패스를 통과하던 측정, 두 번의
추론을 견디고도 틀렸던 DSP 귀속. **파일 안의 표는 자기 자신과 어긋날 수 없습니다.**

```
offset 0   magic  "HBTXRPL\0"        8 B
offset 8   version u32               = 1
offset 12  header_bytes u32          64 B 정렬
offset 16  n_sections u32
offset 20  reserved u32
offset 24  entry[n]                  각 48 B
             name[24]  offset u64  bytes u64  count i32  elem_bits i32
```

- 모든 섹션은 **64바이트 정렬** (512비트 M-AXI 한 beat)
- `count`와 `elem_bits`는 **읽는 쪽이 기대하는 값과 대조**합니다.
  `HbtxrBlobSource::expect()`가 불일치를 적재 시점에 잡습니다 — 데이터에서 발견되는
  폭 불일치는 아무도 귀속시킬 수 없습니다
- 이름은 `b3.fc1.rq` 꼴. 접두어 일치로는 안 되고 **저장된 이름도 거기서 끝나야** 합니다
  (`b1.fc1.w`가 `b1.fc1.wx`에 맞으면 안 됨)

**측정: 249 섹션 · 2,164,932 B.**

## 1. 원소 인코딩

| 무엇 | 폭 | 비고 |
|---|---|---|
| MHA·MLP 가중치 | **int4** | 바이트당 2개, **낮은 nibble 먼저**. 크기는 어느 쪽이든 같아서 순서 오류가 조용합니다 |
| 스템·헤드 가중치 | int8 | 논문 §V-B-2 |
| 누산기 bias | int32 | `acc_t`는 20비트지만 값이 아니라 **타입**이 폭을 정합니다 ([SPEC §3](../SPEC.md)) |
| `(mult, shift)` | **u32 하나** | `mult \| shift << 18`. 하나의 결정이라 함께 이동합니다 |
| LUT 엔트리 | int16 / **uint16** | `exp`·`recip`는 **부호 없음** — exp 최대값이 `1<<15 = 32768`이라 int16에 안 들어갑니다 |
| `lnb` | **int64** | 테이블 엔트리가 아니라 affine 누산 격자(실측 ~30비트 부호). wrap이 불가능하도록 넓게 |

`mult`가 18비트를 넘으면 **패커가 거부**합니다. `i_ops.DYADIC_MULT_BITS`가 계약의
반대쪽 끝이고, 조용히 잘린 승수는 그럴듯한 오답입니다.

## 2. 가중치 blob — **블록 우선**

```
stem.{search,track}.{w,b,rq}          스템 2개
b{0..7}.{qkv,proj,fc1,fc2}.{w,b,rq}   블록별 matmul 4개
b{0..7}.{ln1,ln2}.{sc,w,b,rs}
b{0..7}.sm.{sc,exp,r1,r2}
b{0..7}.gelu.{sc,t}
b{0..7}.edges                          per-tensor 12개
fnorm.{sc,w,b,rs}                      공유 최종 norm 1개
head.{search,track}.{fc1w,fc1b,fc1rq,gsc,gt,grq,fc2w,fc2b}
mode.{search,track}.{exit,feat,out}
mode.track.anchor
```

**블록 우선인 이유**: prefetcher가 한 코어쌍이 계산하는 동안 다른 TRB를 옮깁니다
([SPEC §6](../SPEC.md)). 블록 하나의 페이로드가 **연속 영역**이어야 하고, 텐서별로
인터리브하면 모든 prefetch가 scatter가 됩니다.

두 가지가 packing 단계에서 **접힙니다**:

- **스템의 zero-point 보정.** 입력 격자가 비대칭이라 진짜 0이 zero-point고 창마다
  `- zp·Σw`가 붙는데, 출력채널별 상수라 RMU가 이미 더하는 bias에 접힙니다. 런타임 뺄셈
  없음, 상주 숫자 하나
- **qkv의 세 소비자.** 누산기 하나가 서로 다르게 캘리브된 Q·K·V로 갑니다. **출력채널
  순서로 이어붙인** per-channel `(M,n)` 배열이 그것을 통째로 표현합니다 — 별도 하드웨어 없음

## 3. AXIS

| | |
|---|---|
| 입력 폭 | **256비트** = `AXIS_LANES(32) × uint8`. `stem_*_t::pix_beat_t`와 **정확히 같아서** 어댑터가 없습니다 |
| 입력 순서 | **픽셀 인터리브 `[y][x][c]`** — 골든의 채널 우선 `[c][y][x]`가 아닙니다 |
| beat 수 | search `512` · track `256` (실측 `hls::stream` 최대 깊이와 일치) |
| 출력 | `5 × int32`, **Q.16** (`HBTXR_OUTPUT_FRAC`) |

**전치가 DMA 디스크립터의 일입니다.** 채널 우선으로 들어오면 채널 0 전체가 채널 1보다
먼저 도착하므로, `Cin > 1`에서 K행 윈도우가 이미지 전체를 버퍼링해야 합니다. 라인 버퍼가
성립하려면 픽셀 인터리브여야 합니다.

**커널이 고정소수점을 보냅니다** — 누산기가 아닙니다. 호스트는 아무것도 곱하지 않습니다.
골든이 들고 있는 것은 누산기고 그건 이 위에 있습니다.

## 4. `status`

| 값 | 뜻 |
|---|---|
| 0 | ok |
| 1 | blob을 읽을 수 없음 (magic·version·테이블) |
| 2 | 섹션 부재 또는 모양 불일치 |
| 3 | **prefetch 스케줄 이탈** — 코어쌍이 다른 블록을 돌았음 |
| 4 | mode가 0도 1도 아님 |

3번이 값 비교로는 절대 안 잡히는 종류입니다: 어긋난 prefetch는 **틀린 블록에서 그럴듯한
숫자**를 냅니다.

## 5. 모드 의존은 정확히 넷

**스템 · 공유 최종 norm으로 들어가는 requant · 어느 헤드(+anchor) · 출력 requant.**

가운데 둘이 놓치기 쉽습니다. 두 경로가 서로 다른 블록에서 나와 **norm 하나**로 들어가고,
두 헤드의 마지막 linear는 **서로 다른 누산기 격자**를 갖습니다. 출력 테이블을 하나로 두면
**누산기 골든은 계속 통과**하면서 두 번째로 돈 모드의 고정소수점 값만 조용히 틀립니다.

anchor는 track 전용이고 **호스트 격자로 도착**하므로 concat 전에 pooled feature 격자로
requant해야 합니다 — `Linear`의 입력 격자는 하나뿐입니다.

## 6. 검증

```bash
python3 hardware/deploy/hbtxr/payload.py --golden hardware/workspace/golden/model-hbtxr-w4s4h8
sh hardware/build/run_tb.sh payload
python3 hardware/deploy/hbtxr/overlay.py --dry-run --golden <golden>
```

`tb_payload`가 **코어쌍 하나를 두 경로로 적재하고 멤버를 전수 비교**합니다 — 골든 `.txt`
경유(`HbtxrModelWeights`)와 blob 경유(`HbtxrBlobSource`). 둘 다 같은 `rmu.load(...)`에서
끝나므로 **다른 것은 packing뿐**이고, 라벨이 어느 섹션인지 말합니다.

> 모델을 blob 가중치로 끝까지 돌리는 것보다 **강한** 시험입니다. end-to-end는 "동공이
> 움직였다"를 말하고, 이건 "`b3.fc1.rq`가 채널 417에서 다르다"를 말합니다. 모델 자체는
> 이미 `tb_top`이 골든에 대고 증명했고, 증명 안 된 것은 **전송**이었습니다.

**측정: 8블록 × 32그룹 = 256그룹 전부 원소별 일치.** 음성 대조로 **nibble 하나 뒤집기가
거부**되는 것을 확인했습니다.

호스트 쪽 `payload.requant`가 커널의 `requant<HCFG, ap_int<32>>`를 재현하는지는 dry-run이
찍는 기댓값이 `tb_top`의 출력과 같은지로 확인됩니다:

```
search  0.0007 -0.0030 -0.0089 0.0255 0.0072
track   0.0226  0.0123  0.0088 0.0134 -0.0045
```

## 7. 아직 안 된 것

| | |
|---|---|
| **비트스트림** | `HbtxrTop`이 합성된 적 없습니다. 이 계약이 그 weight SOURCE를 정의하지만, 래퍼와 csynth는 **S10-4** |
| **자원** | S9 실측 LUT **350%** · DSP **265%**. 인터페이스가 정해져도 배치에서 실패합니다 — **S10-1** |
| **보드** | ZCU104 물리 접근 |
| **커널 쪽 스템·헤드 로더** | blob에는 들어 있고 `tb_payload`가 packing을 검증하지만, `HbtxrBlobSource`는 아직 **블록만** 적재합니다. 나머지는 S10-4에서 top 래퍼와 함께 |

`overlay.py`의 `--dry-run`이 PL이 필요 없는 것 전부를 지금 검사합니다. 나머지는 보드를
기다립니다 — 바이트 수 실수를 **예약해서 간 보드에서** 발견하는 것이 대안이라서 만들었습니다.
