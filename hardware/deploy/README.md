> **작성** 2026-07-31 · **갱신** 2026-08-03
> **상태** active
> **소유** hardware

# deploy — PYNQ·호스트

```bash
# 워크스테이션: 골든에서 보드가 먹을 파일을 만듭니다
python3 hardware/deploy/hbtxr/payload.py --golden hardware/workspace/golden/model-hbtxr-w4s4h8
python3 hardware/deploy/hbtxr/overlay.py --dry-run --golden hardware/workspace/golden/model-hbtxr-w4s4h8

# 보드: 실행
python3 -m hbtxr.run_smoke --bit hbtxr.bit --deploy . --golden model-hbtxr-w4s4h8
```

| 파일 | 무엇 | 상태 |
|---|---|---|
| `hbtxr/payload.py` | 가중치 blob · AXIS 스트림 · `status` 코드 · 기댓값 오라클 | ✅ **검증됨** — `tb_payload` |
| `hbtxr/overlay.py` | PYNQ 런타임 | 🟨 **초안.** `--dry-run` 만 실제로 돌려봤습니다 |
| `hbtxr/run_smoke.py` | 두 모드 실행 + 골든 대조 + 세션 JSON | 🟨 **초안.** 오라클은 검증, 실행 경로는 미검증 |

> **초안이라는 말의 뜻**: DMA 순서 · 레지스터 이름 · `AP_DONE` 폴링이 맞는지는
> **비트스트림이 나와야 알 수 있습니다.** `.hwh` 가 없으면 레지스터 이름을 확인할 방법이
> 없고, 지금 맞다고 적어두면 그건 측정이 아니라 희망입니다.

계약 정본: [../docs/contracts/AXI-PAYLOAD.md](../docs/contracts/AXI-PAYLOAD.md).
검증: `sh hardware/build/run_tb.sh payload`.

## 생성물

`hardware/workspace/deploy/` (git 에 없습니다 — 골든처럼 재생성됩니다)

| | 크기 |
|---|---|
| `hbtxr_weights.bin` | **2,164,932 B** · 249 섹션 |
| `hbtxr_image_search.bin` | 16,384 B = 512 AXIS beat |
| `hbtxr_image_track.bin` | 8,192 B = 256 beat |
| `hbtxr_anchor_track.bin` | 20 B |

## 한 설계, 두 모드

구 트리는 `hgtxr_mode_search_par32.bit` 와 `hgtxr_mode_track_par32.bit` 를 **따로** 냈습니다.
논문 구조가 아닙니다. 모드에 의존하는 것은 **정확히 넷**이고 (스템 · 공유 최종 norm 진입
requant · 어느 헤드 · 출력 requant) **넷 다 페이로드지 fabric 이 아닙니다.**

## 아직 못 도는 것 — 그리고 왜

**비트스트림이 없습니다.** `HbtxrTop` 은 합성된 적이 없고(S10-4, 합성 가능한 weight SOURCE
가 필요), S9 가 지금 데이터패스를 **LUT 350%** 로 실측했으므로 배치에서 실패합니다(S10-1).
보드도 없습니다.

그래서 여기 있는 것은 **계약**이고, 그게 마침 `HbtxrTop` 합성을 막고 있던 바로 그것입니다.
`tb_payload` 가 blob 을 커널 쪽 리더로 풀어 골든 경유 적재와 **원소별로 대조**하므로,
비트스트림이 생겼을 때 전송 계층은 이미 증명되어 있습니다.

`--dry-run` 은 PL 이 필요 없는 것을 전부 지금 검사합니다 — 바이트 수 실수를 **예약해서 간
보드에서** 발견하는 것이 대안이라 만들었습니다.

## 구 런타임에서 가져온 것

`archive/hardware/pynq/hgtxr/` 의 **배관만 조각으로** 가져왔습니다 (출처 주석 있음):
`.hwh` 레지스터 파싱 · Vitis 가 64비트 포인터를 `name_1`/`name_2` 로 나누는 것 ·
flush/invalidate · `AP_START`/`AP_DONE` 폴링 + 타임아웃.

**계약은 하나도 살아남지 않았습니다** — 구 런타임은 `FRAME_SHAPE=(256,256)` ·
`STATE_SHAPE=(6,)` · `DATA_FRAC_BITS=10` 이고 전부 M-AXI 였습니다. 새 설계는
`128×128` / `2×64×64` · **5개 출력** · **Q.16** · AXIS + M-AXI 입니다.

구 PYNQ 가이드 본문은 `archive/hardware/pynq/hgtxr/README.md` 에 있습니다 — 구 비트스트림의
파일 배치를 기술하므로 참조용입니다.
