# hls-fpga-accelerators (Transformer-Accel) — Cross-Ref

> **참조**: `REF/Analysis/ViT-Accelerator/hls-fpga-accelerators.md` (기존 정밀 분석 존재함)
> 본 문서는 중복 정밀분석을 피하고 **교차참조(cross-ref)** 만 기록한다. 전체 모듈 분석은 위 문서를 볼 것.

## 동일 여부: **동일 (essentially identical)**

`REF/Transformer-Accel/hls-fpga-accelerators`는 기존 분석 대상인 `REF/ViT-Accelerator/hls-fpga-accelerators`와 **사실상 동일한 저장소**다 (동일 저자 Luis G. Leon-Vega / Luis Prieto Sibaja, 동일 커널 구성, 동일 빌드/시그니처). 별도 정밀분석 불필요.

## 근거 (파일/라인)

- **저자 동일**: README:167~171 (`Luis G. Leon Vega`, `Luis Prieto Sibaja`), config.h:2~3 (`Copyright 2022-2024 / Author: Luis G. Leon-Vega`).
- **커널 구성 동일**: matmul / elementwise / unary(+axc-math) / rmsnorm / softmax + `common/config.h` — 디렉토리·파일 목록이 기존 문서 §2와 일치 (Glob 확인).
- **공통 인프라 동일**: `common/config.h` — `kBusWidth=512` 기본(config.h:14~18), 패킷화 `kPackets = kBusWidth/kDataWidth`(config.h:41), `RawDataT=ap_uint<kBusWidth>`(config.h:43), FIXED16=Q6.10·FIXED8=Q4.4(config.h:29~39), `GET_NUMBER/GET_RAW` 매크로 패턴(config.h:64~70) — 기존 문서 §3.1과 라인 단위로 일치.
- **matmul 상수 동일**: `kARows=2` 고정, `kBCols/kCCols` 기본 32768(matmul.h:11~21) — 기존 문서 §3.2 한계 근거와 동일.
- **시그니처 동일**: `matmul(RawDataT*a,*b,*c,a_rows,b_cols,c_cols)`(README:29, matmul.h:24), `elementwise(...,size,op)` op 0=add/1=mult(README:61~69), `unary(...,size,op)` op 1=ReLU/2=SiLU + IMPLEXP=LUT/STD(README:96~102, 87~91), `rmsnorm`/`softmax`(README:128/159) — 기존 문서와 일치.
- **타깃·DSE 환경변수 동일**: U250(`xcu250-figd2104-2L-e`) 기본 + Kria K26, DATATYPE(FLOAT4~32/FIXED8/16)·BUS(64~2048) 파라미터화(README:18~24 등) — 기존 §1, §6과 일치.

## 차이점 (관찰된 미세 차이)

- **softmax DATATYPE 옵션**: 본 repo README는 softmax에서 `FLOAT4/8/16/32`만 노출(FIXED 제외, README:146). 기존 ViT-Accelerator 문서 §3.3은 "softmax/rmsnorm은 FLOAT 권장(README:154)"으로 서술 — 의미상 동일(softmax는 float 전제). FIXED 옵션 노출 여부의 표기 차이 수준이며 본질적 차이 아님.
- **기본 행렬 크기 표기**: 본 README의 matmul `B_COLS/C_COLS` 기본은 4096으로 표기(README:20~21)되나, 실제 헤더 기본값은 32768(matmul.h:13,18) — README 표와 코드 default 간 불일치는 양 저장소 공통 특성으로 추정(기존 문서 §2도 32768 기재).
- 그 외 알고리즘·데이터플로우(`load→compute→store` dataflow + 패킷 UNROLL + PIPELINE), softmax 2-pass(max-subtraction 부재), rmsnorm 2-pass(eps 하드코딩, γ곱 없음), LUT exp 등 **차이 없음**.

## 결론

두 사본은 동일 코드베이스다. 본 Transformer-Accel 사본에 대한 평가·강점·한계·우리 프로젝트(HG-PIPE/ViT) 시사점은 **기존 `REF/Analysis/ViT-Accelerator/hls-fpga-accelerators.md` §8~9를 그대로 적용**한다. (요지: 연산자 단위 라이브러리로 재사용 가치 높음 — config.h 패킷화/매크로 패턴·LUT exp·dataflow 골격 차용 가능; 단 softmax online화·matmul 타일링/상주가중치·레이어 융합은 우리가 추가 설계 필요.)
