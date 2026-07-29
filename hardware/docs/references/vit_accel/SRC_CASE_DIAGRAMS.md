# HG-PIPE `src/` + `case/` 다이어그램 인덱스

이 문서는 `main` 브랜치에서 추가된 다이어그램 문서의 역할을 `ViT_Accel` 문서 세트 안에서 이어받는다.

현재는 각 모듈의 전체 Mermaid 다이어그램을 별도 분산 문서로 늘리기보다, 통합 문서에서 핵심 흐름을 먼저 확인하는 방식으로 정리했다.

권장 시작점:

1. [SRC_CASE_MODULE_GUIDE.md](./SRC_CASE_MODULE_GUIDE.md)
2. `PATCH_EMBED`
3. `ATTN`
4. `MLP`
5. `HEAD`

다이어그램을 볼 때의 초점:

- stage 간 FIFO 연결
- matrix / vector streaming 경계
- layernorm / quant / reshape / split / merge 배치
- residual 및 wrapper 경계

이 문서는 index 역할만 담당하며, 실제 요약 다이어그램은 통합 가이드 안에 포함되어 있다.
