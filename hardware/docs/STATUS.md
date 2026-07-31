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
| **S1** | 골든 생성기 `tools/export_hls_golden.py` | ⬜ **다음** |
| **S2** | RMU · SMU | ⬜ |
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
sh hardware/build/run_<blk>_tb.sh                 # V1  g++ + ap_int
vitis_hls -f hardware/build/hls/<blk>_csim.tcl    # V2  /tools/Xilinx/Vitis_HLS/2023.2
```

## Blocked

| 항목 | 무엇이 막나 | 누가 |
|---|---|---|
| 보드 실측 | ZCU104 물리 접근 | **사용자** |
| Table III 재현 | **범위 밖** ([SPEC §10](SPEC.md)) | — |

## Done

| 날짜 | 내용 |
|---|---|
| 2026-07-31 | **S0** SPEC — 파라미터 계약·traits 방식·유도 규칙·금지 관용구·검증 계약 |
| 2026-07-31 | ViT_Accel HLS 참조 분석 — 35건 제기·26 확정 (채택 9·회피 4·적응 8) |
| 2026-07-31 | worktree `hardware-new` 생성, 코드 6개 디렉토리 비움 (286 삭제) |
| 2026-07-31 | `.gitignore`의 `build/`가 `hardware/build/` 44개를 삼키던 것 수정 |
