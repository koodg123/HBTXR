# HG-PIPE `src/` + `case/` 심층 분석

이 문서는 `main` 브랜치에서 추가된 SRC/case 분석 문서를 `ViT_Accel` 문서 세트로 가져오기 위한 entrypoint다.

현재 `ViT_Accel`에서는 분석 / 다이어그램 / 마이크로아키텍처 설명을 분리된 문서로 유지하되, 실제 읽기는 통합 가이드부터 시작하는 것을 권장한다.

권장 읽기 순서:

1. [SRC_CASE_MODULE_GUIDE.md](./SRC_CASE_MODULE_GUIDE.md)
2. 병렬도 / fit 관점이면 [SRC_CASE_PARALLELISM_TUNING.md](./SRC_CASE_PARALLELISM_TUNING.md)
3. 필요하면 [SRC_CASE_DIAGRAMS.md](./SRC_CASE_DIAGRAMS.md)
4. 필요하면 [SRC_CASE_MICROARCH.md](./SRC_CASE_MICROARCH.md)

통합 가이드에서 바로 보면 좋은 섹션:

- repo / case layer 개요
- primitive kernel 요약
- composite kernel 요약
- `PATCH_EMBED`, `ATTN`, `MLP`, `HEAD` 데이터흐름
- 그리고 실제 fit 조정은 `SRC_CASE_PARALLELISM_TUNING.md`

주의:

- 이 문서 세트는 현재 docs-first merge 단계 기준이다.
- 세부 코드 경로는 최종적으로 `workspace/hardware/src/` 와 `workspace/hardware/case/` 기준으로 정렬될 예정이다.
