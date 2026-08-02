> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active
> **소유** hardware

# hardware STATUS — `feat/hls-rewrite`

**이 브랜치는 논문 §IV 구조를 HLS로 새로 구현합니다.** 이관이 아닙니다.

> 이관 작업(M1~M5)과 구 코드 591개는 `rewrite/flat-functional` 브랜치에 있습니다.
> `archive/hardware/`는 두 브랜치 모두에 있고 **조각을 가져올 원본**입니다.

---

## Active

| 단계 | 내용 | 상태 |
|---|---|---|
| **S0** | [SPEC.md](SPEC.md) + [ViT_Accel 참조 분석](references/2026-07-31-vit-accel-hls-analysis.md) | ✅ **완료** |
| **S1** | 골든 생성기 [`tools/export_hls_golden.py`](../tools/export_hls_golden.py) | ✅ **완료** |
| **S2** | RMU · SMU | ✅ **완료** — tb 3개 PASS |
| **S3** | 비선형 LUT (RSQRT64·EXP32·RECIP128·GeLU32, 전부 16b) | ✅ **완료** — tb 3개 PASS |
| **S4** | MHA Core (9단계) | ✅ **완료** — 9단계 체인 + 스테이지 프로브 7개 |
| **S5** | MLP Core (6단계) | ✅ **완료** — 6단계 + **블록 전체** `block_y` 통과 |
| **S6** | Patch Embedding (Conv-F/Conv-E + shuffler) | ✅ **완료** — 두 스템 PASS |
| **S7** | Global Buffer · interconnect · Weight Prefetcher · Controller | ✅ **완료** — 8 TRB / 코어쌍 2개 |
| **S8** | 4코어 cyclic top + mode별 terminal decode | ⬜ **다음** |
| **S9** | csynth · 자원 리포트 | ⬜ |

S2~S7은 독립입니다. **S8이 처음으로 전체를 묶습니다.**

## 규칙

- **완료 조건은 "골든과 원소별 일치" + "러너에 연결"** 입니다. "컴파일된다"가 아닙니다
- **`hgtxr` 금지.** 새 코드는 `hbtxr` ([SPEC §0](SPEC.md))
- **파일 통째 복사 금지.** 조각만, 출처 주석과 함께
- **`HBTXR_FIXED_CSIM` 항상 켬.** float 폴백 없음

## 검증 환경 (WSL)

```bash
sh hardware/build/make_golden.sh                  # 골든 (생성물, git 에 없음)
sh hardware/build/run_tb.sh                       # V1  g++ + ap_int/hls::stream
vitis_hls -f hardware/build/hls/<blk>_csim.tcl    # V2  S8 부터
```

| tb | 대조 | 규모 |
|---|---|---|
| `tb_requant` | `i_ops.py:requant` 직접 | 3,733 케이스 · 출력 16/16 · 포화 35% |
| `tb_rmu` | `rmu_y` (출력 프로젝션 `[192,192]`) | 12,288 값 |
| `tb_smu` | `smu_y` (`Q×Kᵀ`, 3헤드) | 12,288 값 |
| `tb_gelu` | `gelu_y` (`table_quantize`) | 49,152 값 + 포화 직접 검증 |
| `tb_layernorm` | `ln1_y` (7-scalar 커널) | 12,288 값 |
| `tb_softmax` | `softmax_y` (14-scalar, 2세그먼트) | 12,288 값 · **두 세그먼트 다 사용** |
| `tb_mha` | `mha_y` (9단계 전체) | 12,288 값 · **프로브 7개** + 고장 주입 |
| `tb_mlp` | `mlp_y` (6단계 전체) | 12,288 값 · **프로브 5개** + 고장 주입 |
| `tb_patch` | `patch_y` (Conv-F · Conv-E) | 12,288 / 3,072 값 |
| `tb_block` | **`block_y` — 블록 하나 통째** | 12,288 값 · **프로브 12개**(두 코어) |
| `tb_backbone` | **`{search,track}_block_out` — 8 TRB** | 코어쌍 2개 재사용 · prefetch 태그 · 고장 주입 |

전부 **원소별 `==` + 스트림 배수 + 음성 대조**. 러너가 **모든 tb 를 두 토큰 수로** 돌립니다
(search `N=64` · track `N=16`, 같은 바이너리) — 공유 백본이 주장이지 한쪽만 되는 게 아니니까요.
클린 트리에서 **20개 실행 전부 PASS** (`tb_backbone` 은 자기 안에서 search·track·search 3회).

## Blocked

| 항목 | 무엇이 막나 | 누가 |
|---|---|---|
| 보드 실측 | ZCU104 물리 접근 | **사용자** |
| Table III 재현 | **범위 밖** ([SPEC §10](SPEC.md)) | — |

## 결정 대기 — 하드웨어를 막지는 않습니다

| | 무엇 | 상태 |
|---|---|---|
| **requant 승수 폭** | `M` 이 33비트, `acc·M` 이 53비트라 DSP48E2 하나에 안 들어감. **`M ≤ 16b` 까지 골든 비트 동일**이 실측됨 (`algorithm/docs/reports/2026-07-31-requant-multiplier-width.md`, 브랜치 `rewrite/flat-functional`) | algorithm 대기. 하드웨어는 `REQ_M_BITS=33` 으로 **이미 통과** — 바뀌면 상수 하나 |
| **엣지별 dtype** | `BlockSpec.dtype` 이 하나라 블록 안에서 **matmul 4비트 + 비선형 16비트**를 표현 못 함. GeLU LUT 입력 알파벳이 16개로 붕괴 | S4·S5 전. 지금은 `-a4`/`-a8` 두 벌로 우회 중 |
| **`anchor` 포트** | track 헤드가 호스트에서 **5차원 anchor state** 를 받습니다 (`197→197→5`). §8 인터페이스에 없었습니다 | S7 컨트롤러 · S8 top |

셋 다 [SPEC](SPEC.md) 에 근거와 실측이 있습니다 (§3 · §8).

## 범위에서 뺀 것

**보조 헤드 3종**(ROI · Reliability ×2)은 골든에 넣지 않았습니다 — 논문 Fig. 5 의 Pupil Box /
Pupil Ellipse 두 개만 냅니다. 배포 모델(`HybridModel`)에는 셋 다 있으므로 나중에 붙여야 합니다.
`ReliabilityHead` 의 `sigmoid` 는 **온칩에 둘 필요가 없습니다** — 스케줄러가 임계값과 비교할
뿐이고 sigmoid 는 단조라, 로짓을 `sigmoid⁻¹(threshold)` 와 비교하면 됩니다.

## Done

| 날짜 | 내용 |
|---|---|
| 2026-07-31 | **S7** Backbone — 8 TRB 가 **코어쌍 2개** 위를 돌며 가중치 교체. `bt` 리덕션이 런타임 길이 아닌 컴파일 최대치를 돌던 버그를 track 이 잡음 |
| 2026-07-31 | **S6** Patch Embedding — Conv-F·Conv-E PASS. PE 배열이 곧 RMU(kernel==stride 라 im2col 이 주소 계산), 스템만 `ap_int<27>` |
| 2026-07-31 | **S5** MLP Core — 6단계 PASS. 두 코어를 이어 **블록 하나 전체**가 `block_y` 와 비트 일치 |
| 2026-07-31 | **S4** MHA Core — 9단계 체인 PASS. 스테이지 프로브(비교·배수·재충전) 도입, 고장 주입으로 국소화 확인 |
| 2026-07-31 | **S3** 비선형 LUT — GeLU·LayerNorm·Softmax tb PASS. `exp`/`recip` 는 **unsigned**, `lnb` 는 30b 임을 실측 |
| 2026-07-31 | **S2** RMU · SMU · requant — tb 3개 PASS. 첫 HLS 구현이 골든을 통과했습니다 |
| 2026-07-31 | **S1+** 모델 전체 골든 — 두 스템·**공유** 블록 스택·두 헤드, 논문 비트폭. 공유 백본 제약 3건·`anchor` 포트 누락 발견 |
| 2026-07-31 | **S1** 골든 생성기 — 스테이지 10 · 프리셋 6벌 · 파일 84개/벌. stdlib 만, `M` 33비트 실측 |
| 2026-07-31 | **S0** SPEC — 파라미터 계약·traits 방식·유도 규칙·금지 관용구·검증 계약 |
| 2026-07-31 | ViT_Accel HLS 참조 분석 — 35건 제기·26 확정 (채택 9·회피 4·적응 8) |
| 2026-07-31 | worktree `hardware-new` 생성, 코드 6개 디렉토리 비움 (286 삭제) |
| 2026-07-31 | `.gitignore`의 `build/`가 `hardware/build/` 44개를 삼키던 것 수정 |
