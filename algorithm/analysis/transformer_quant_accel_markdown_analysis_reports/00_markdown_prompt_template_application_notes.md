# Markdown Prompt / Template Application Notes

## 1. 적용한 Prompt/Template

본 문서 패키지는 업로드된 `Paper-Review-Summary-Prompt.md`와 `Paper-Review-Summary-Template.md`의 구조를 기준으로 작성하였다.

- Prompt의 9개 필수 섹션을 모든 논문별 문서에 동일하게 적용하였다.
- Template의 `Paper Information`, `Overview`, `Research Background`, `Related Works`, `Key Concepts` 구조를 확장하여 사용하였다.
- Template이 4.4까지만 제공되어 있으므로, Prompt의 지시사항에 따라 `Methodology`, `Experiments Results`, `Summary`, `Pros.`, `Cons.`를 추가하였다.

## 2. 품질 향상용 분석 프롬프트

```text
업로드된 Transformer quantization / ViT accelerator / integer-only inference 논문을 처음부터 끝까지 검토하라.
각 논문에 대해 다음을 구분하라.

1. 연구 문제가 algorithm 문제인지, hardware 문제인지, 또는 algorithm-hardware co-design 문제인지 판별한다.
2. low-bit quantization, integer-only arithmetic, non-linear approximation, sparse attention, systolic array, CUDA/Triton kernel, FPGA RTL/VHDL template 중 핵심 축을 분류한다.
3. 논문의 수치 결과를 full-network, module-level, estimated result, measured result로 구분한다.
4. accuracy metric과 efficiency metric이 동일 조건에서 비교되었는지 확인한다.
5. 제안 방법이 FPGA/NPU/ASIC/HLS/RTL 설계에 재사용 가능한지 분석한다.
6. 장점과 한계는 감상이 아니라 실험 근거, architecture assumption, precision assumption, hardware target에 기반하여 작성한다.
```

## 3. 검증 기준

| 항목 | 적용 여부 |
|---|---|
| 9개 필수 섹션 유지 | 적용 |
| Paper Information table 포함 | 적용 |
| 대표 결과 table 포함 | 적용 |
| Related works comparison table 포함 | 적용 |
| Key concepts 및 source 표시 | 적용 |
| Methodology 재현 관점 설명 | 적용 |
| Experiments setup/result 분리 | 적용 |
| Pros./Cons. 근거 기반 작성 | 적용 |
| 불확실성/미수행 항목 명시 | 적용 |

## 4. 범위와 한계

- 분석 대상은 현재 업로드된 7개 PDF이다.
- 각 문서는 제공된 PDF의 본문, 표, 그림, 수식, 알고리즘 정보를 기반으로 작성하였다.
- 외부 웹 검색, GitHub 코드 실행, FPGA 보드 실행, CUDA/Triton benchmark 재현, RTL/VHDL synthesis 재현은 수행하지 않았다.
- 수치 결과는 논문에 제시된 조건을 기준으로 정리했으며, 서로 다른 hardware, precision, model, operation scope 간 비교는 직접 비교에 주의가 필요하다.
