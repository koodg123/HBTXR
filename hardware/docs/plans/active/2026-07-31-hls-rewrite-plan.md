> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active — 승인 대기
> **소유** hardware

# HLS 재작성 계획 — 논문 구조를 새로 구현

근거: [논문 대응표](../../architecture/2026-07-31-paper-to-hardware-mapping.md) ·
`PAPERS/JETCAS_REVISION1_FINAL_MAIN.pdf` §IV

## 0. 무엇을 하고 무엇을 안 하는가

**한다**: 논문 §IV 구조를 HLS로 **새로 구현**. 디렉토리 구조(`config module build deploy tools docs workspace`)는 **그대로**.

**안 한다**: 자원·전력·지연을 Table III에 맞추는 것. **구조가 목표이지 수치가 아닙니다.**
수치는 구조가 서고 난 뒤 별도 과제입니다.

**금지**: **파일 통째 복사.** 기존 코드에서 가져오는 것은 **조각 단위**이고, 가져올 때마다
출처와 이유를 남깁니다.

---

## 1. Spec 파라미터 — 전부 확정됐습니다

논문은 `D`·`H`·`F`를 기호로 두고 slimming이 정한다고 합니다. 실값의 정본은 **학습 모델**입니다.

| 항목 | 값 | 출처 |
|---|---|---|
| search 입력 | **128×128** (frame, Conv-F) | 논문 Table IV |
| track 입력 | **2×64×64** (event 2극성, Conv-E) | 논문 Table IV |
| patch | **16** | `algorithm/models/frame/model.py:29` |
| 토큰 `N` | search **64** (8×8) · track **16** (4×4) | 위 둘에서 유도 |
| 임베딩 `D` | **192** | `model.py:28` · `vitis_hls.yaml` |
| 헤드 `H` | **3** (`d = D/H = 64`) | `vitis_hls.yaml` |
| 깊이 | search **8 TRB** · track **4 TRB** (cut `c=4`) | 논문 Fig. 5 |
| 헤드 | search **Pupil Box** · track **Pupil Ellipse** | 논문 Fig. 5 |
| 양자화 | patch/head **8-bit** · MHA/MLP **4-bit W/A** · 비선형 **16-bit** | 논문 §V-B-2 |
| 비선형 LUT | RSQRT **64**/16b · EXP **32**/16b · RECIP **128**/16b · GeLU **32**/16b | 논문 §V-C |
| 클럭 목표 | 300 MHz | 논문 Table IV (**맞추지 않아도 됨**) |

> **`config.h`의 `HEIGHT/WIDTH 256`·`TOKENS 256`은 논문과 다릅니다.** 레거시입니다.
> `cyclic_config.h`(128/64, 64/16 토큰)가 맞고, 새 Spec은 후자를 따릅니다.

### Spec 문서가 담을 것

1. 위 표 (파라미터 계약)
2. 블록 인터페이스 — 각 블록의 입출력 타입·스트림 폭·핸드셰이크
3. 스테이지 체인 — MHA 9단계 / MLP 6단계, 각 단계의 버퍼와 FIFO 깊이
4. PE 조직 — `Nt×Do`, `Di` 곱셈기 등 논문 §IV-B의 수를 파라미터로
5. 메모리 정책 — track resident / search on-demand + prefetch 중첩
6. 모드 계약 — 컨트롤러가 보는 상태·전이·라우팅
7. **검증 계약** — 각 블록의 골든 출처와 비교 방식 (§4)

---

## 2. 디렉토리 사용 — 구조 유지

```
hardware/
├── config/design/     파라미터 헤더 (기존 9개는 레거시 → 새 1벌로 대체)
├── module/
│   ├── include/       ★ 새 HLS 구현
│   ├── src/           ★ 새 top + 블록
│   ├── tb/            ★ 새 테스트벤치
│   └── golden/        algorithm 이 생성한 정수 골든
├── build/hls/         새 tcl (기존 20개는 구 top 대상 → 정리)
├── deploy/            기존 PYNQ 유지 (인터페이스 바뀌면 그때 수정)
└── workspace/         산출물
```

**`archive/hardware/`는 그대로 둡니다** — 참조용이고, 조각을 가져올 원본입니다.

---

## 3. 재작성 전략 — 무엇을 새로 쓰고 무엇을 가져오나

### 새로 씁니다 (전부)

블록 구조·스테이지 체인·인터페이스·버퍼·top·테스트벤치.
**논문 §IV가 스펙이고, 기존 코드는 참고 자료입니다.**

### 조각으로 가져옵니다 — 파일이 아니라 기법

| 무엇 | 어디서 | 왜 |
|---|---|---|
| `#pragma HLS bind_op ... impl=dsp` 곱셈 헬퍼 | `src/rmu_smu.cpp` | DSP 강제 배치 관용구. **RMU/SMU의 핵심** |
| `template<int UNIT_ID>` + `INLINE off` | `include/hgtxr_e2e_vit.hpp:3040` | HLS가 코어 2벌을 만들게 하는 방법. **4코어의 구현 수단** |
| 정수 LUT 인덱싱 (shift + clamp) | `include/hgtxr_cyclic_math.hpp` | **테이블 값 자체는 `algorithm/quantization` export가 정본** |
| 타일/그룹 순회 패턴 | `include/hgtxr_cyclic_*.hpp` | `Gt`·`Gi`·`Go` 그룹 분할이 논문 표기와 대응 |
| 참조모델 비교 테스트벤치 골격 | `tb/tb_cyclic_head_attention.cpp` | **지금 돌아가는 것을 확인한 유일한 tb** (PASS) |

**가져올 때마다** 해당 파일 상단에 `// 출처: archive/hardware/... 의 <무엇>, <왜>`를 답니다.

---

## 4. 검증 — 이게 계획의 중심입니다

기존 코드의 가장 큰 문제는 **검증이 없었다**는 것입니다: 테스트벤치 12개 중 6개만 빌드에
걸렸고, 걸린 것 중 하나는 어서션이 0이었으며, **csim이 기본적으로 `float`로 돌아** 양자화를
전혀 검증하지 않았습니다. 같은 실수를 반복하지 않습니다.

### 3층 검증

| 층 | 무엇 | 도구 | 지금 가능? |
|---|---|---|---|
| **V1 단위** | 블록별 정수 출력 == 골든 | **g++ + ap_fixed** | ✅ WSL 확인됨 |
| **V2 csim** | top 전체 == 골든 | Vitis HLS csim | ✅ WSL `/tools/Xilinx/Vitis_HLS/2023.2` |
| **V3 csynth** | 합성 가능·자원·타이밍 | Vitis HLS csynth | ✅ 같음 |

### 골든의 출처 — `algorithm/quantization`

알고리즘 쪽에 **순수 파이썬 임의정밀 정수 골든**(`i_ops.py`)이 있고, 모든 커널을
**원소별 `==`**로 비교합니다(허용오차 아님). `replay_block_int`·`replay_model_int`가
블록·모델 단위 오라클입니다.

**이걸 HLS 골든 생성기로 씁니다.** 하드웨어와 알고리즘이 **같은 정수 정의**를 공유하게 되고,
지금 교차 참조가 0건인 문제도 여기서 풀립니다.

### 첫 커밋부터 지키는 규칙

- **`HGTXR_HLS_FIXED_CSIM`을 기본 켬.** float 폴백은 없앱니다.
- 새 테스트벤치는 **반드시 골든과 비교**하고 **음성 대조**(일부러 틀린 입력이 실패하는지)를 포함합니다.
- 블록 하나 = 테스트벤치 하나 = `build/hls/`의 tcl 하나. **연결되지 않은 tb를 만들지 않습니다.**

---

## 5. 단계

| | 내용 | 산출물 | 검증 |
|---|---|---|---|
| **S0** | **Spec 작성** (§1) | `module/SPEC.md` 재작성 | 파라미터가 논문·algorithm과 일치 |
| **S1** | 골든 생성기 | `tools/export_hls_golden.py` | algorithm 정수 오라클로 블록별 벡터 생성 |
| **S2** | **RMU · SMU** | `include/hbtxr_rmu.hpp`·`hbtxr_smu.hpp` + tb 2 | **V1** — 정수 원소별 일치 |
| **S3** | 비선형 LUT (RSQRT/EXP/RECIP/GeLU, 논문 크기) | `include/hbtxr_nonlinear.hpp` + tb | **V1** |
| **S4** | **MHA Core** 9단계 체인 | `src/hbtxr_mha_core.cpp` + tb | **V1** — 블록 골든 |
| **S5** | **MLP Core** 6단계 체인 | `src/hbtxr_mlp_core.cpp` + tb | **V1** |
| **S6** | Patch Embedding (Conv-F/Conv-E + shuffler) | `src/hbtxr_patch_embed.cpp` + tb | **V1** |
| **S7** | Global Buffer · interconnect · Weight Prefetcher · controller | `src/hbtxr_{gb,noc,prefetch,ctrl}.cpp` | **V1** |
| **S8** | **4코어 cyclic top** + mode별 terminal decode | `src/hbtxr_top.cpp` | **V2 csim** — 모델 골든 |
| **S9** | csynth · 자원 리포트 | `build/hls/` tcl | **V3** |

**S2~S7은 서로 독립**이라 순서를 바꿔도 됩니다. **S8이 처음으로 전체를 묶습니다.**

### 각 단계의 완료 조건

**"컴파일된다"가 아니라 "골든과 원소별로 같다"입니다.** 그리고 그 검증이
`build/hls/`의 tcl과 `run_cyclic_tb.sh` 같은 러너에 **연결돼 있어야** 합니다.

---

## 6. 기존 코드는 어떻게 되나

| | |
|---|---|
| `module/include`·`src`·`tb` 현재 96개 | **S8까지 그대로 둡니다.** 조각을 가져올 원본이고, 새 구현이 골든을 통과하기 전에 지우면 되돌릴 수 없습니다 |
| S8 통과 후 | 새 구현이 정본. 구 파일은 **삭제**(archive에 원본 있음) |
| `archive/hardware/` | 영구 보존. 손대지 않습니다 |

**두 구현이 공존하는 기간이 생깁니다.** 그건 의도된 것이고, `SPEC.md`와 각 README가
어느 쪽이 정본인지 매 단계 명시합니다.

---

## 7. 위험

| | |
|---|---|
| **Table III 재현 실패** | 목표가 아니라고 §0에 못 박았습니다. 구조가 서면 그때 별도 과제 |
| 4-bit W/A 양자화 정확도 | algorithm 쪽 PTQ가 이미 검증돼 있고(원소별 일치), 골든이 그걸 씁니다 |
| 300 MHz 미달 | S9에서 측정만 하고 기록합니다. 타이밍 최적화는 별도 |
| **보드 검증** | 여전히 막혀 있습니다. S9까지는 보드 없이 갑니다 |
| 작업량 | S2~S8이 실질 구현입니다. **단계마다 골든 통과가 있어 중간에 멈춰도 그때까지는 검증된 상태**입니다 |
