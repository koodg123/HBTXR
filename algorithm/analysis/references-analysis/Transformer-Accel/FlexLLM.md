# FlexLLM 교차참조 (REF/Transformer-Accel/FlexLLM)

> 대상 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\Transformer-Accel\FlexLLM`
> 참조 분석: **`REF/Analysis/ViT-Accelerator/FlexLLM.md`** (기존 정밀 분석 존재함, Read 완료)
> 본 문서는 중복 정밀분석 대신 **동일여부 판정 + 차이점**만 기록한다. 라인 근거 기반.

---

## 판정: **동일 (identical) repo, 다른 REF 위치**

`REF/Transformer-Accel/FlexLLM`은 `REF/ViT-Accelerator/FlexLLM`과 **동일한 FlexLLM 저장소**다. 디렉토리 구조, README, 핵심 config, 핵심 HLS 모듈이 모두 일치한다. 별도 정밀분석 불필요 → 기존 분석 `REF/Analysis/ViT-Accelerator/FlexLLM.md` 전체가 그대로 적용된다.

### 동일성 근거 (대조 확인)

| 항목 | 기존(ViT-Accelerator) | 대상(Transformer-Accel) | 일치 |
|---|---|---|---|
| README 제목 | "FlexLLM: A Composable HLS Library for Rapid LLM Accelerator Design" | 동일 (README.md:1) | O |
| DOI 배지 | `10.5281/zenodo.18793354` | 동일 (README.md:2) | O |
| 자기정의 문구 | "composable High-Level Synthesis (HLS) library ... hybrid temporal–spatial accelerators" | 동일 (README.md:4) | O |
| 성능 요약 | 1.29× e2e, 1.64× decode, 3.14× energy / V80 4.71× / HMT 23.23×·64× | 동일 (README.md:21-35) | O |
| 모델 상수 | LAYER 16, HIDDEN 2048, KV_HIDDEN 512, HEAD_DIM 64, INTER 8192, VOCAB 128256 | 동일 (config_u280.h:19-45) | O |
| 병렬도 상수 | TOKEN_PARALLEL 8, PRE_QKVO 16, PRE_FFN 64/2, DEC_K/V 32, LOGITS_MAX_K 5 | 동일 (config_u280.h:49-75) | O |
| Modules/ 트리 | PE/Linear_Layer/MHA/Softmax/quant/RoPE/LayerNorm/Swish/FHT/Logits/data_io/HMT 등 17개 | 동일 17개 (Modules/*.h) | O |
| DSP 패킹 PE | `PE_i4xi4_pack_2x2_2D`, `DSP48E2: A27 B18 C48` 주석 | 동일 (PE.h:6-7, 373-413, pack_b 합성·sign 보정 로직 일치) | O |
| 변형 디렉토리 | SpinQuant_Llama_32_1B / _Ins / HMT_SpinQuant_Llama_32_1B | 동일 3변형 존재 | O |

→ 코드/주석 레벨에서 차이를 발견하지 못함. 두 위치는 같은 repo의 사본으로 판단.

---

## 차이점 (위치/맥락 차이만, 코드 차이 아님)

1. **REF 분류 위치만 다름**: 기존은 `ViT-Accelerator/` 하위, 대상은 `Transformer-Accel/` 하위. FlexLLM은 LLaMA(decoder-only LLM) 가속기이므로 분류상 **Transformer-Accel 위치가 더 정확**하다 (ViT가 아닌 LLM). 코드 내용 차이는 없음.
2. **빌드 산출물 포함 여부(사본 상태)**: 대상의 `SpinQuant_Llama_32_1B/run/` 아래 `bitstreams/*.xclbin` 4종, 컴파일된 호스트 실행파일, `u280_power_log.csv`, `gpu_answer.txt` 등 **실행 산출물이 함께 존재**함을 확인 (기존 분석에서는 이들을 "제외물"로만 언급). 이는 분석 대상이 아닌 생성물이며, 정밀분석 결론에는 영향 없음.
3. 그 외 소스 차이는 확인되지 않음.

---

## 결론
- 동일 repo. **별도 정밀분석 생성 생략, `REF/Analysis/ViT-Accelerator/FlexLLM.md` 참조로 갈음.**
- 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점도 기존 분석 §9를 그대로 적용 (TAPA dataflow+RapidStream floorplan 자동화, INT4/INT8 DSP 패킹 PE, INT8 시스톨릭 MHA, mem_opt HBM 채널 압축, Modules 라이브러리화 방법론).

### 근거 표기
- 직접 대조 확인: README.md:1-35, config_u280.h:19-75, Modules/PE.h:6-7·373-413, Modules/ 17파일 목록, 3변형 디렉토리 존재.
- 미수행(불필요): 기존 분석이 이미 라인 단위로 다룬 prefill/decode 태스크 그래프, FHT, mem_opt link config 등은 동일 repo이므로 재검증 생략.
