> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active — 논문 구현의 기준 문서
> **소유** hardware

# 논문 ↔ 하드웨어 대응표

출처: `PAPERS/JETCAS_REVISION1_FINAL_MAIN.pdf` §IV (Overall Architecture · Microarchitecture ·
Runtime Scheduling), Table III.

**이 문서가 "무엇을 구현해야 하는가"의 단일 출처입니다.** 지금까지 없었고, 없어서
두 구현 중 어느 것이 논문인지 아무도 몰랐습니다.

---

## 1. 논문이 정의한 것

### 1-1. 최상위 (Fig. 4)

DRAM · CPU · AXI in/out DMA · PS–PL BUS · **shared Global Buffer** · **on-chip interconnect** ·
**Patch Embedding Core** · **4개 cyclic compute core** · **mode별 terminal decode/output routing** ·
**FPGA-side controller**

토큰 타일이 **MHA Core0 → MLP Core0 → MHA Core1 → MLP Core1**을 순회하며
mode-asymmetric cyclic-streaming 루프를 이룹니다.

### 1-2. 두 개의 프리미티브 행렬 엔진 — 논문의 핵심

| | 쓰이는 곳 | 성격 |
|---|---|---|
| **RMU** (Resident Matmul Unit) | Q/K/V 생성 · output projection · FC1 · FC2 | **resident weight** |
| **SMU** (Stream Matmul Unit) | Q×Kᵀ · S×V | **런타임 생성 피연산자** |

> *"HBTXR does not process one transformer block monolithically. Instead, each core is
> organized as a stream-oriented stage chain."*

### 1-3. 스테이지 체인

```
MHA Core : residual split → LayerNorm → Q/K/V(RMU) → head-wise reorder
           → Q×Kᵀ(SMU) → Softmax → S×V(SMU) → output proj(RMU) → residual merge
MLP Core : residual split → LayerNorm → FC1(RMU) → GeLU → FC2(RMU) → residual merge
```

MLP는 **SMU·score buffer·head reorder가 없습니다.** 그게 두 코어의 정의적 차이입니다.

### 1-4. PE 조직 (구현 시 그대로 나와야 하는 수)

| 블록 | PE 수 | PE당 곱셈기 |
|---|---|---|
| Patch Embedding projection | `Do` | `K×K×Ui` + 누산기 1 |
| MHA Q/K/V (RMU) | `Nt × Do` | `Di` + 누산기 1 |
| MHA score (SMU, head-local) | score PE 집합 | `di` (= `d/Gh`) + 누산기 1 |
| MLP FC1 (RMU) | `Nt × Fo` | `Di` + 누산기 1 |

버퍼: 지연 residual FIFO 깊이 `Br` · head-local FIFO · reorder buffer · score buffer ·
line buffer `K−1` 행.

### 1-5. 모드 비대칭 (Fig. 5)

| | 입력 | 깊이 | 헤드 | 가중치 |
|---|---|---|---|---|
| **search** | frame, **Conv-F** | TRB0~**TRB7** (8) | Pupil **Box** | on-demand fetch + prefetch 중첩 |
| **track** | event, **Conv-E** | TRB0~**TRB3** (4) | Pupil **Ellipse** | on-chip resident |

---

## 2. 코드 대응 — **두 구현이 논문의 서로 다른 절반을 갖고 있습니다**

| 논문 블록 | 정본 `hgtxr_e2e_vit.hpp` (6 비트스트림) | `src/*.cpp` = `hgtxr_top` (1 비트스트림) |
|---|---|---|
| **RMU / SMU** | ❌ **0회** | ✅ `rmu_smu.cpp` (109줄, `bind_op impl=dsp`) |
| 4 cyclic core | ✅ `attn_unit<0/1>`·`mlp_unit<0/1>` | ❌ `attention.cpp`·`mlp.cpp` 각 1개 |
| Patch Embedding Core | ✅ | ✅ `frame_patch_embed`·`event_patch_embed` (Conv-F/Conv-E) |
| shared Global Buffer | ✅ `HgtxrGlobalBuffer` | ✅ `global_buffer.cpp` |
| **on-chip interconnect** | ❌ **0회** | ✅ `noc.cpp` |
| **Weight Prefetcher** | ❌ **0회** | ✅ `weight_prefetcher.cpp` |
| FPGA controller | 1회 | ✅ `controller.cpp` |
| 모드 비대칭 traversal | ✅ 7곳 | ✅ `runtime_fsm.cpp` |
| mode별 terminal decode | `mlp_head` **1개** | ✅ `search_head`·`track_head` **분리** |

### 4개 코어는 어떻게 구현돼 있나

```cpp
template <int UNIT_ID>
void hgtxr_e2e_attn_unit(...) {
#pragma HLS INLINE off
  (void)UNIT_ID;          // ← 값은 안 씁니다. 하드웨어 2벌을 만들기 위한 템플릿입니다
```

`INLINE off` + 별개 템플릿 인스턴스 → HLS가 `attn_unit_0_s`/`attn_unit_1_s`를 각각 합성합니다.
자원 표에서 둘 다 **34,434 LUT / 207 DSP**로 동일한 것이 증거입니다. **정당한 HLS 관용구입니다.**

### 판정

**어느 쪽도 논문 구조가 아닙니다.**

- 정본에는 **RMU/SMU·NoC·Weight Prefetcher가 없습니다.** 그런데 논문의 메모리 정책
  (*"search 가중치 on-demand fetch, prefetch가 초기 실행과 중첩"*)은 Prefetcher가 있어야
  성립하고, 트랜스포머 본체는 **RMU/SMU로 정의**돼 있습니다.
- `hgtxr_top`에는 그 셋이 있지만 **4개 코어가 없고**, 비트스트림 1개에
  **테스트벤치가 아무것도 단언하지 않습니다**.

---

## 3. Table III ↔ 저장소 실측 — **일치하지 않습니다**

| | 논문 Table III | 저장소 최고 빌드 (C3b `par16_c3b_mem16`) |
|---|---|---|
| 클럭 | **300 MHz** | **200 MHz** (5 ns) |
| search latency | **1.342 ms** | 3.861 ms (`search_profile_top`) |
| track latency | **0.785 ms** | 0.497 ms (`track_profile_top`) |
| LUT | **145 K (62.9%)** | 126,506 (54.9%) |
| DSP | **814 (47.1%)** | 604 (35.0%) |
| BRAM36K | **184 (59.0%)** | 332 BRAM_18K ≈ 166 |
| URAM | **12 (12.5%)** | **64 (66.7%)** |
| Power | **4.648 W** | 6.777 W (AQ2 track 실측) |
| Throughput | **326 GOPS** | 33.81 GOPS (AQ2 track) |

**URAM이 5배, 클럭이 1.5배, GOPS가 10배 다릅니다.**

그리고 Table III는 **블록별 분해**(DMA/AXI/Controller/Global Buffer/MHA×2/MLP×2/Patch Emb./Head)를
제시합니다. 그 표를 만들려면 **블록이 각각 별개 HLS 모듈**이어야 합니다 — 즉
`hgtxr_top` 구조입니다. 정본의 csynth 리포트는 `attn_unit_*`·`project_qkv`·`controller_run`
같은 다른 분해를 냅니다.

> **저장소의 어떤 빌드도 Table III를 산출하지 않습니다.** 논문 수치가 이 저장소에 없는
> 다른 빌드에서 나왔거나, 추정치이거나 — 어느 쪽인지 **저는 판단할 수 없습니다.**
> 확인은 사용자만 할 수 있습니다.

---

## 4. 구현 계획

**전제: 목표는 Table III를 재현하는 설계입니다.** 그러면 필요한 것은 디렉토리 정리가
아니라 **두 구현의 합병**입니다.

| | 작업 | 근거 |
|---|---|---|
| **H1** | 이 문서 — 대응표 확정 | ✅ 완료 |
| **H2** | **Table III 출처 확인** | 논문 수치가 어느 빌드에서 나왔는지. **사용자만 답할 수 있고, 이후 전부가 여기 걸립니다** |
| **H3** | 정본에 **RMU/SMU 도입** | 논문이 본체를 이 둘로 정의합니다. `src/rmu_smu.cpp`가 참고 구현 |
| **H4** | 정본에 **NoC · Weight Prefetcher** 추가 | Table III에 별도 행으로 있고, prefetch 중첩 주장의 근거 |
| **H5** | **mode별 terminal decode** 통합 | 논문은 한 설계 안의 라우팅, 현재는 `search_profile_top`/`track_profile_top` **별도 비트스트림 2개** |
| **H6** | 블록 경계를 Table III에 맞춤 | 블록별 자원 표를 내려면 HLS 모듈 경계가 그 표와 같아야 합니다 |
| **H7** | 300 MHz 재타이밍 | 현재 200 MHz. WNS 여유 확인 필요 |

### 순서에 대한 판단

**H2가 먼저입니다.** Table III가 이 저장소 밖의 빌드에서 왔다면 H3~H7은 **이미 존재하는
설계를 복원하는 일**이고, 추정치라면 **처음부터 만드는 일**입니다. 비용이 완전히 다릅니다.

### 그 전에는 하지 말아야 할 것

- **`src/`의 얇은 파일 4개(`noc`·`controller`·`global_buffer`·`weight_prefetcher`, 37줄)를
  합치지 마십시오.** ponytail 감사가 이걸 권고했는데 **철회합니다** — 이 파일들은 파편이
  아니라 **논문 블록 다이어그램을 한 블록 = 한 파일로 옮긴 것**이고, H4·H6의 출발점입니다.
- 정본 4,503줄 분해(P7)도 H3와 겹칩니다. RMU/SMU를 도입하면 그 과정에서 분해됩니다.
  **따로 하면 두 번 합니다.**

---

## 5. 부수 확인 — 알고리즘 쪽은 일치합니다

| 논문 | 코드 |
|---|---|
| search 8 TRB · track 4 TRB | `HGTXR_SEARCH_DEPTH 8` · `HGTXR_TRACK_CUT_DEPTH 4` (`cyclic_config.h`) ✅ |
| search=Pupil Box · track=Pupil Ellipse | `search_head.cpp` · `track_head.cpp` ✅ |
| patch/head 8-bit · MHA/MLP 4-bit · 비선형 16-bit | `Q4W8A` 명명과 일치 ✅ |
| 비선형 LUT (RSQRT/EXP/RECIP/GeLU), K∈{8,16,32,64,128} | `algorithm/quantization` 분할 rsqrt·reciprocal ✅ |
