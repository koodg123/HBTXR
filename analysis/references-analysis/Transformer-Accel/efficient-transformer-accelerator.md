# efficient-transformer-accelerator — Cross-Ref

> 대상 repo: `REF/Transformer-Accel/efficient-transformer-accelerator`
> 기준 분석: **`REF/Analysis/ViT-Accelerator/efficient-transformer-accelerator.md` 참조** (기존 .md 존재함, 269줄 정밀 분석)
> 작성일: 2026-06-20 / 도구: Glob·Grep·Read (정적, bash 미사용)

---

## 결론: 동일 (same repo, 다른 경로)

대상 repo는 기준 분석 대상(`REF/ViT-Accelerator/efficient-transformer-accelerator`)과 **동일한 저장소**다. 프로젝트명·아키텍처·파일 트리·핵심 소스가 일치한다. 별도 신규 정밀 분석 불필요하며, 위 기준 .md를 그대로 적용한다.

### 동일 근거 (파일/라인)

1. **readme.md 일치**: "Glide Accelerator", 32×16 = 512 MAC, INT8 + 32-bit 누산, Taylor degree-1/degree-2 softmax 근사, 200 MHz / 102.4 GOPS, LUT ~36% / DSP ~15%, end-to-end ~61 cycle, 64유닛 시분할 양자화 — 기준 분석 §1과 수치까지 동일 (`readme.md:1-21`). References도 ViTALiTy + AMD QTViT 동일 (`readme.md:44-47`).
2. **핵심 RTL 일치**: 주력 top `systolic_quant_32x16.sv`의 모듈명·포트·파라미터(`ROWS=32, COLS=16, DATA_WIDTH=8, ACC_WIDTH=32, QUANT_UNITS=64`)가 기준 분석 §3.7과 동일 (`systolic_quant_32x16.sv:17-46`). 헤더 주석의 "512 PEs / 64 shared quant units (8× time-multiplexed) / K+~40 cycles"도 일치 (`:6-14`).
3. **파일 트리 일치**: `hw/src/` 11개 .sv(pe, systolic_mac, systolic_top, systolic_mac_rect, accumulator_bank, quant, quant_top, quant_shared, systolic_quant_integrated, systolic_quant_32x16), `hw/tb/`·`hw/scripts/`·`hw/constraints/`, `models/degree_{1,2}_train`·`degree_2_quant/test.py`, `PRODUCTION_RELEASE_NOTES.md`까지 기준 분석 §2 디렉토리 구조와 동일.

### 차이점

- **경로만 상이**: `ViT-Accelerator/` → `Transformer-Accel/` 하위로 위치만 다름. 콘텐츠 차이는 발견되지 않음(정적 비교 기준). 동일 git 저장소의 복제/재배치로 추정.

### 내부 ViTALiTy 서브폴더 (1줄)

이 repo 안의 `ViTALiTy/` 서브폴더는 GaTech-EIC HPCA'23 원본 코드를 vendoring 한 것으로, **최상위 ViTALiTy repo(별도 분석 대상)와는 별개**이며 본 cross-ref 범위 밖이다.

---

## 시사점

기준 .md의 §9(HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적 시사점)가 그대로 유효하다: systolic+quant 통합 3-stage 데이터패스, N:1 시분할 requant, 직사각 systolic 차원 매핑, Taylor linear attention의 저지연 적용성, FPGA+ASIC 듀얼 제약 분리. 상세는 기준 .md 참조.
