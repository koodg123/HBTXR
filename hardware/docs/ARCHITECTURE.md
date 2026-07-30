> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 색인. 본문 설계는 아래 문서들이 담습니다
> **소유** hardware

# ARCHITECTURE — HBTXR 가속기 설계

**이 파일은 색인이고, 설계 본문은 이관된 문서들에 있습니다.** 새로 쓰지 않았습니다 —
census가 확인한 실제 설계 문서를 가리킵니다.

## Scope — 무엇을 하고 무엇을 안 하는가

**규약 규칙 F에 따라 Scope는 별도 문서가 아니라 여기 있습니다.** 분리하면 반드시
어긋납니다.

### 한다

- HBTXR ViT 가속기의 HLS 구현 (Vitis HLS → Vivado → ZCU104/VCK190)
- INT8 / Q4W8A 정수 데이터패스, HG-PIPE 계열 정수 LUT 비선형
- search / track 두 모드와 그 사이의 런타임 전환
- PYNQ 기반 보드 런타임과 스모크 테스트

### 하지 않는다

- **학습·양자화 캘리브레이션** — `algorithm/quantization/`이 담당합니다.
  두 서브시스템이 같은 HG-PIPE 커널을 각각 구현하고 있으며 **교차 참조가 0건**입니다
  ([STATUS.md](STATUS.md) 하단).
- **`hgtxr_e2e_vit.hpp` 4,503줄의 분해** — 합성 결과가 코드 형태에 민감하고 검증할 보드가
  없어 보류했습니다 ([재구성 계획 §5](plans/active/2026-07-29-hardware-reconstruction.md)).
- **보드 실측** — ZCU104 물리 접근이 필요합니다.

## ★ 논문 ↔ 코드 대응 — 여기부터 읽으십시오

[2026-07-31-paper-to-hardware-mapping.md](architecture/2026-07-31-paper-to-hardware-mapping.md)

논문(`PAPERS/JETCAS_REVISION1_FINAL_MAIN.pdf` §IV, Table III)이 정의한 블록과 코드를
대조한 문서입니다. **결론 셋:**

1. **두 구현이 논문의 서로 다른 절반을 갖고 있습니다.** 정본 `hgtxr_e2e_vit.hpp`에는
   논문의 핵심 프리미티브 **RMU/SMU가 0회**이고 **NoC·Weight Prefetcher도 없습니다.**
   그 셋은 비정본 `hgtxr_top`의 `src/*.cpp`에만 있습니다.
2. 반대로 **4개 cyclic core**(MHA0/1·MLP0/1)는 정본에만 있습니다 —
   `attn_unit<0/1>`·`mlp_unit<0/1>` 템플릿 인스턴스로.
3. **Table III의 수치를 산출하는 빌드가 저장소에 없습니다.** URAM 5배·클럭 1.5배·
   GOPS 10배 차이입니다.

## 설계 문서

| 문서 | 내용 |
|---|---|
| [architecture/2026-07-31-paper-to-hardware-mapping.md](architecture/2026-07-31-paper-to-hardware-mapping.md) | **논문 ↔ 코드 대응표 · 구현 계획 H1~H7** |
| [MODULE-GUIDE.md](MODULE-GUIDE.md) | **모듈별 상세** — 각 모듈의 분석·다이어그램·마이크로아키텍처를 한 섹션에 통합 |
| [architecture/2026-06-16-cyclic-streaming-accelerator.md](architecture/2026-06-16-cyclic-streaming-accelerator.md) | cyclic 스트리밍 가속기 개요 |
| [architecture/2026-06-16-hbtxr-arch-freeze.md](architecture/2026-06-16-hbtxr-arch-freeze.md) | 아키텍처 동결과 DeiT 등가성 |
| [SPEC.md](SPEC.md) | 사양 — 보드·양자화·파라미터 계약 |

## 구현이 셋이라는 사실

census가 확인한 것이며, 재구성 전에 알아야 합니다
([census §3.5](reports/2026-07-29-hardware-census.md)):

| | 형태 | 비트스트림 | 판정 |
|---|---|---:|---|
| `hgtxr_e2e_axis_top` | 단일 헤더 4,503줄 | **6** | **정본** |
| `hgtxr_top` | 모듈 조립 (top + 15 `.cpp`) | 1 | 살아 있음 |
| `hgtxr_mode_profile_top` | 모드별 프로파일링 | 2 | 살아 있음 |

`hgtxr_cyclic_*.hpp` 템플릿 라이브러리(10헤더, 순환 없는 의존 트리)를 `hgtxr_top`과
테스트벤치가 사용합니다. **정본인 `hgtxr_e2e_vit.hpp`는 이 라이브러리의
`cyclic_transformer_block`(106줄)을 0회 사용하고 같은 일을 4,503줄로 재구현합니다.**

## 디렉토리 레이아웃

현재 레이아웃은 [hardware/README.md](../README.md)가 정의합니다.
2026-06-17의 이전 레이아웃 결정은
[plans/done/2026-06-17-directory-layout.md](plans/done/2026-06-17-directory-layout.md)에
대체 고지와 함께 보존되어 있습니다.
