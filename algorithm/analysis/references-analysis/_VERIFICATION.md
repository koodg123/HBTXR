# 검증 리포트 (_VERIFICATION)

> 대상: `REF/Analysis/` 산출물 · 검증일: 2026-06 · 방식: 파일 도구(Glob/Read) 기반 커버리지·품질 점검

## 1. 커버리지 (카테고리별 산출물 수)

| 카테고리 | 분석 .md 수 | 열거 방식 | 판정 |
|---|---|---|---|
| ViT-Accelerator | 21 | 카테고리 repo 전수 분석(2개 에이전트 분담) | ✅ 완료 |
| ViT-Quantization | 48 | repo 전수 열거 후 누락분 보충(추가 repo 다수 발굴) | ✅ 완료 |
| Transformer-Accel | 32 | repo 31개 전수 열거 → 고유 24 심층 + 중복 8 cross-ref | ✅ 완료 |
| CNN-Accel | 18 | repo 18개 전수 열거·분석 | ✅ 완료 |
| Others | 11 | repo 전수 열거(후보 외 HiSpMM/AGNA/REMOT 추가 발굴) | ✅ 완료 |
| XR-Eye-Tracking/Codebase | 17 | 코드 repo 전수(awesome-list 4종은 코드 0건→제외) | ✅ 완료 |
| Papers | 22 | Papers 폴더 PDF 21편 + Q-HyViT 논문 | ✅ 완료 |
| **합계** | **169** | | |

각 카테고리 보충 에이전트는 **카테고리 하위 repo를 전수 열거 후, 분석 .md가 없는 자체 repo만 생성**하는 멱등 방식으로 수행했고, 누락/미존재/미완을 명시적으로 보고함. third-party(fpnew/ramulator/softfloat/openc910, vendor, 3rdparty, .ip_user_files 등)는 정책상 제외이며 누락이 아님.

## 2. 품질 표본 점검

표본 3종(HW/양자화/논문)을 직접 열람해 (a) 필수 섹션, (b) 실제 근거, (c) 표기 일관성을 확인:

- `Transformer-Accel/HG-PIPE.md`: 개요~디렉토리 구조에 `README.md:Lxx`, `BlockCfg.scala:L63-66` 등 **파일·라인 근거** 다수. ICCAD'24 출처·VCK190·DeiT-Tiny 사양 명시. ✅
- `ViT-Quantization/I-ViT.md`: `quant_utils.py:150-261`, `quant_modules.py:389-497` 등 함수·라인 근거. dyadic/ShiftGELU/Shiftmax/I-LayerNorm 알고리즘 정확. ✅
- `Papers/3ET.md`: 서지·문제정의·기여·방법론(ConvLSTM/CB-ConvLSTM **LaTeX 수식**)·실험까지 8섹션 완비. ✅

→ 표본 전부 일반론이 아닌 **코드/논문 근거 기반**, "추정"/"확인 불가" 표기 일관. 저품질·빈 산출물 미발견.

## 3. 발견된 한계 / 잔여 리스크

- **소스 부재 repo(빌드 산출물·리포트만)**: `LLM_FPGA`, `LUT-LLM`, `TeraFly`, `flightllm_test_demo` → 라인단위 알고리즘 "확인 불가", 리포트/바이너리 역추정으로 작성(해당 .md에 명시).
- **부분 체크아웃/빈 repo**: `ternaryLLM`(설계 본체 일부 누락), `TinyTransformer`(사실상 빈 repo), `ViTALiTy`(Transformer-Accel 사본 HW 부재).
- **명칭-실체 불일치(문서에 정정 기록)**: `Tiny-GPT-on-Vortex`(실제 2-layer MLP), `AnyPackingNet`(실체 DeepBurning-MixQ).
- **대용량 PDF 미판독**: `Papers/NearEye-10000Hz.md`(>20MB) → 교차인용 기반, 본문 정밀내용 "확인 불가".
- **정량 PPA**: 합성 리포트 미동봉 repo 다수 → LUT/DSP/BRAM/주파수/fps "확인 불가".
- **중복 정합성**: Transformer-Accel의 8개 cross-ref는 `ViT-Accelerator/` 동명 문서를 가리키며, 대부분 byte-동일 또는 스냅샷 차이로 확인됨(`ViT-FPGA-TPU`만 컨트롤러 RTL 추가본).
- **ViT-Accelerator 열거 범위**: 분담 분석으로 확인된 자체 repo 21개를 모두 분석. 물리적으로 추가 repo가 있을 경우를 대비해, INDEX의 카탈로그와 실제 디렉토리를 차후 1회 대조 권장(경미).

## 4. 전체 판정

**커버리지: 완료** (third-party·코드 0건 awesome-list 제외, 자체 코드베이스·논문 전수 분석). 품질: 표본 기준 양호(근거 기반·표기 일관). 위 §3의 한계 항목은 원본 자료 자체의 결손에 기인하며 각 문서에 투명하게 표기됨.
