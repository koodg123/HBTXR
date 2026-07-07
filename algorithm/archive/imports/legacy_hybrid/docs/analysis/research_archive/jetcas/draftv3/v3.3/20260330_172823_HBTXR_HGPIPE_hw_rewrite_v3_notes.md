# HBTXR HG-PIPE 기반 하드웨어 재작성 메모

## 반영 사항
- 하드웨어 섹션을 HG-PIPE의 핵심 아이디어(스테이지 오버랩, 온칩 weight residency, 비선형 함수 LUT 근사, stage-local control) 중심으로 다시 작성함.
- HG-PIPE 원문 용어를 직접 반복하지 않도록 HBTXR 전용 용어로 재정의함.
  - hybrid-grained pipeline -> dual-scale overlap chain
  - coarse/fine-grained execution -> set-synchronous pass / fragment-stream pass
  - FIFO / deep buffer류 설명 -> elastic lane / context reservoir
  - static / dynamic matmul -> stored-weight matrix engine / context-mixing matrix engine
  - feature reuse -> resident feature ring / resident-feature circulation
- CPU-FPGA partition, head gating, latency decomposition, stage cadence 모델을 추가해 hardware narrative를 강화함.
- 하드웨어 관련 표/그림 placeholder를 확장하여 논문 분량이 10페이지가 되도록 구성함.

## 구성 변경 포인트
- Section V (Proposed HBTXR Hardware)를 사실상 전면 재작성.
- Experimental Results에서도 hardware evaluation narrative를 하드웨어 용어에 맞춰 업데이트.
- Comparison with Other Accelerator 문단에서 HG-PIPE와 HBTXR의 차이를 명시.
- Ablation 항목 중 하드웨어 관련 표현을 resident-feature circulation 기준으로 수정.

## 산출물 상태
- 최종 PDF 페이지 수: 10 pages
- FPGA resource / board power 숫자는 원문 draft에 없으므로 TBD 유지
- figure는 placeholder 유지
