# HG-PIPE `src/` + `case/` 마이크로아키텍처 인덱스

이 문서는 `main` 브랜치의 마이크로아키텍처 설명을 `ViT_Accel`용으로 정리한 index다.

현재 권장 읽기 방식은 다음과 같다.

1. [SRC_CASE_MODULE_GUIDE.md](./SRC_CASE_MODULE_GUIDE.md)에서 모듈별 요약 확인
2. 각 모듈의 역할, 병렬성, 메모리 구조, 병목 후보를 함께 읽기

주요 관찰 포인트:

- `PATCH_EMBED`
  - 가장 큰 DSP 수요와 장주기 arithmetic hotspot
- `ATTN`
  - Q/K/V 생성, split, matmul, softmax, merge가 결합된 가장 복합적인 데이터흐름
- `MLP`
  - 두 번의 matmul과 GeLU/LayerNorm이 결합된 반복 구조
- `HEAD`
  - 최종 투영 및 adapter 경계

코드 merge가 진행되면 세부 참조 위치는 다음 기준으로 맞춰질 예정이다.

- `workspace/hardware/src/`
- `workspace/hardware/case/`
