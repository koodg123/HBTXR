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
| **S2** | RMU · SMU | ⬜ **다음** — 착수 전 `shift_max` 결정 필요 |
| **S3** | 비선형 LUT (RSQRT64·EXP32·RECIP128·GeLU32, 전부 16b) | ⬜ |
| **S4** | MHA Core (9단계) | ⬜ |
| **S5** | MLP Core (6단계) | ⬜ |
| **S6** | Patch Embedding (Conv-F/Conv-E + shuffler) | ⬜ |
| **S7** | Global Buffer · interconnect · Weight Prefetcher · Controller | ⬜ |
| **S8** | 4코어 cyclic top + mode별 terminal decode | ⬜ |
| **S9** | csynth · 자원 리포트 | ⬜ |

S2~S7은 독립입니다. **S8이 처음으로 전체를 묶습니다.**

## 규칙

- **완료 조건은 "골든과 원소별 일치" + "러너에 연결"** 입니다. "컴파일된다"가 아닙니다
- **`hgtxr` 금지.** 새 코드는 `hbtxr` ([SPEC §0](SPEC.md))
- **파일 통째 복사 금지.** 조각만, 출처 주석과 함께
- **`HBTXR_FIXED_CSIM` 항상 켬.** float 폴백 없음

## 검증 환경 (WSL)

```bash
sh hardware/build/make_golden.sh                  # 골든 6벌 (생성물, git 에 없음)
sh hardware/build/run_<blk>_tb.sh                 # V1  g++ + ap_int
vitis_hls -f hardware/build/hls/<blk>_csim.tcl    # V2  /tools/Xilinx/Vitis_HLS/2023.2
```

## Blocked

| 항목 | 무엇이 막나 | 누가 |
|---|---|---|
| 보드 실측 | ZCU104 물리 접근 | **사용자** |
| Table III 재현 | **범위 밖** ([SPEC §10](SPEC.md)) | — |

## S2 착수 전 결정할 것 — S1 이 남긴 것

| | 무엇 | 왜 지금 |
|---|---|---|
| **requant `shift_max`** | `dyadic_params` 가 거의 항상 `n=31` 을 골라 `M` 이 **33비트**, `acc·M` 이 **53비트** | RMU 의 requant 유닛 폭이 여기서 정해집니다. 조이려면 `i_block.rescale` 변경 = algorithm 쪽 작업 |
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
| 2026-07-31 | **S1+** 모델 전체 골든 — 두 스템·**공유** 블록 스택·두 헤드, 논문 비트폭. 공유 백본 제약 3건·`anchor` 포트 누락 발견 |
| 2026-07-31 | **S1** 골든 생성기 — 스테이지 10 · 프리셋 6벌 · 파일 84개/벌. stdlib 만, `M` 33비트 실측 |
| 2026-07-31 | **S0** SPEC — 파라미터 계약·traits 방식·유도 규칙·금지 관용구·검증 계약 |
| 2026-07-31 | ViT_Accel HLS 참조 분석 — 35건 제기·26 확정 (채택 9·회피 4·적응 8) |
| 2026-07-31 | worktree `hardware-new` 생성, 코드 6개 디렉토리 비움 (286 삭제) |
| 2026-07-31 | `.gitignore`의 `build/`가 `hardware/build/` 44개를 삼키던 것 수정 |
