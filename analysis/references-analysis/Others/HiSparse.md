# HiSparse 정밀 분석

## 1. 개요

- **목적**: HBM(High-Bandwidth Memory)을 탑재한 멀티-다이(multi-SLR) FPGA에서 HLS로 구현한 고성능 희소 행렬-벡터 곱(SpMV) 가속기. CPU/GPU/기존 FPGA 대비 대역폭 효율과 throughput 향상을 목표.
- **한줄요약**: CPSR(Cyclic Packed Streams of Rows)로 인코딩된 희소 행렬을 16개 HBM 채널에서 병렬 스트리밍하여, 8-lane shuffle network + URAM 기반 PE 클러스터로 `y = A·x`를 237 MHz에서 계산하는 multi-SLR Vitis 가속기.
- **원논문**: Du, Hu, Zhou, Zhang, "High-Performance Sparse Linear Algebra on HBM-Equipped FPGAs Using HLS: A Case Study on SpMV", **ACM/SIGDA Int'l Symp. on FPGA (FPGA 2022)** (`Readme.md` L14-19). Cornell Zhang Group. DOI 10.5281/zenodo.5819246 (L3). PDF: `fpgafp193a-du.pdf`(repo 내 바이너리, 분석 제외).
- **타깃 디바이스**: Xilinx Alveo **U280** (HBM-equipped, 3-SLR). 툴체인 **Vitis 2020.2**, shell `xilinx_u280_xdma_201920_3` (`Readme.md` L5-6, L23-24). 달성 주파수 237 MHz (L9).

## 2. 디렉토리 구조

### 자체 소스 트리 (분석 대상)
```
HiSparse/
├── Readme.md
├── spmv/                              # ★ HLS 커널 본체
│   ├── libfpga/                       # 재사용 HLS 모듈(헤더 라이브러리)
│   │   ├── common.h                   # 데이터 타입·payload·overlay 상수
│   │   ├── pe.h                       # Processing Element(URAM 누산 + in-flight forwarding)
│   │   ├── shuffle.h                  # arbiter + crossbar + shuffler(셔플 네트워크)
│   │   ├── vecbuf_access_unit.h       # VAU: 벡터버퍼 read/write(double buffering)
│   │   ├── spmv_cluster.h            # 1개 연산 클러스터(ML→SF1→VAU→SF2→PE→PK)
│   │   └── stream_utils.h             # axis_duplicate / axis_merge
│   ├── spmv_sk0.cpp                   # SLR0 sub-kernel: 4 클러스터(HBM0-3)
│   ├── spmv_sk1.cpp                   # SLR1 sub-kernel: 6 클러스터(HBM4-9)
│   ├── spmv_sk2.cpp                   # SLR2 sub-kernel: 6 클러스터(HBM10-15)
│   ├── spmv_vector_loader.cpp        # 입력 벡터 로드 + 3-SLR 복제
│   ├── spmv_result_drain.cpp         # 3-SLR 결과 수집 → HBM write-back
│   ├── k2k_relay.cpp                 # kernel-to-kernel AXIS relay(SLR간 중계)
│   ├── spmv.ini                       # Vitis connectivity(nk/slr/sp/sc 매핑)
│   └── makefile
├── sw/                                # 호스트(XRT/OpenCL) + 데이터 IO
│   ├── host.cpp                       # 호스트 프로그램(검증·벤치)
│   ├── benchmark.cpp                  # 벤치마크 드라이버
│   ├── data_loader.h                  # CSR/.npy 로더
│   ├── data_formatter.h              # CSR 패딩/정규화/CPSR 인코딩 유틸
│   ├── Makefile / bm.sh
├── performance_model/                 # 사이클 정확 성능 모델 + DSE
│   ├── performance_model.cpp
│   ├── design_space_exp.cpp
│   └── include/*.h                    # 커널 헤더 복제본(모델용)
├── unit_tests/                        # 모듈별 단위 테스트(test_pe/shuffle/vau/...)
├── unit_test_wrapper/                # 단위 테스트 wrapper + .ini
└── LICENSE
```

### 제외 목록 (vendor/생성물/바이너리)
- `xrt/includes/` — Xilinx 제공 호스트 헬퍼(`xcl2`, `oclHelper`, `cmdparser`, `logger`). **vendor 코드, 분석 제외**(이름만 언급).
- `demo_spmv.xclbin` (사전 합성 비트스트림, 바이너리)
- `fpgafp193a-du.pdf` (논문 PDF, 바이너리)
- `datasets/` (download.sh로 받는 graph/pruned_nn .npy, 대용량 데이터 — 제외)
- `.git/`, `.gitattributes`, `.gitignore`
- `performance_model/include/*.h`는 `spmv/libfpga/*` 헤더의 모델용 복제본이므로 중복 분석 생략(원본 libfpga만 정밀 분석).

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `spmv/libfpga/common.h` — 타입·구성 정의 (전 모듈 기반)

#### Overlay 구성 (L30, L162-179)
- `PACK_SIZE = 8` (L30): 한 HBM 워드에 묶이는 nnz/벡터 원소 수 = **8-lane 병렬**. 모든 셔플/PE/VAU의 폭.
- `FIFO_DEPTH = 64` (L162).
- `OB_BANK_SIZE = 1024*8`(출력버퍼 뱅크, L164), `VB_BANK_SIZE = 1024*4`(벡터버퍼 뱅크, L165).
- **클러스터 분배**: `SK0_CLUSTER=4`, `SK1_CLUSTER=6`, `SK2_CLUSTER=6` → `NUM_HBM_CHANNELS = 16` (L173-176). 즉 **16개 HBM 채널 = 16개 연산 클러스터**.
- `LOGICAL_OB_SIZE = 16 * OB_PER_CLUSTER`, `LOGICAL_VB_SIZE = VB_PER_CLUSTER` (L178-179).

#### 데이터 타입 / 양자화 (L35-54)
```c
const unsigned IBITS = 8;                          // 정수부 8비트 (L35)
const unsigned FBITS = 32 - IBITS;                 // 소수부 24비트 (L36)
typedef unsigned IDX_T;                            // 인덱스 = 32b unsigned (L37)
typedef ap_ufixed<32, IBITS, AP_RND, AP_SAT> VAL_T;// ★ 고정소수점 (L38)
#define IDX_MARKER 0xffffffff                       // 행 끝 마커 (L8)
```
- **핵심 양자화**: 값 타입 `VAL_T = ap_ufixed<32, 8, AP_RND, AP_SAT>` — **32비트 부호없는 고정소수점**(정수 8b + 소수 24b), 반올림(AP_RND)·포화(AP_SAT). float 대비 DSP/면적 효율. (float/float_pob/float_stall 변형은 별도 IMPL 옵션, `Readme.md` L74-78.)
- `PACK_SIZE=8` 패킹 타입: `PACKED_IDX_T`(8 idx), `PACKED_VAL_T`(8 val), `SPMV_MAT_PKT_T`(indices+vals, L44-50) — HBM 한 워드가 8 nnz의 (idx,val) 쌍.

#### Payload 타입 (intra-kernel dataflow, L59-96)
- 2비트 명령 `INST_T`: `SOD`(start-of-data), `EOD`(end-of-data), `EOS`(end-of-stream) (L60-63). 스트림 동기화의 핵심 토큰.
- `EDGE_PLD_T`(COO 엣지: mat_val,row_idx,col_idx,inst) — matrix loader↔shuffle1 (L66-71).
- `UPDATE_PLD_T`(mat_val,vec_val,row_idx,inst) — VAU 이후 모든 PE (L77-82).
- `VEC_PLD_T`(val,idx,inst) — 벡터 unpacker↔reader, PE 출력 (L89-93).

#### Kernel-to-kernel AXIS 타입 (L140-146)
```c
typedef struct {
    ap_uint<32 * (PACK_SIZE + 1)> data;  // 288비트: pkt_idx(32) + 8×val(32)
    ap_uint<2> user;                     // INST_T
} VEC_AXIS_T;
#define VEC_AXIS_PKT_IDX(p) (p.data(31,0))           // L145
#define VEC_AXIS_VAL(p, i)  (p.data(63+32*i, 32+32*i))// L146
```
- SLR 간/loader↔kernel↔drain 통신은 이 288-bit AXIS 스트림 패킷 단위.

### 3.2 `spmv/libfpga/pe.h` — Processing Element (URAM 누산 코어)

PE는 SpMV의 **누산(scatter-accumulate)** 담당. URAM 출력버퍼에 행별 부분합을 누적.

#### `ufixed_pe_process<id,bank_size,pack_size>()` (L22-90)
- **입력**: `hls::stream<UPDATE_PLD_T> &input`(mat_val,vec_val,row_idx) / **출력**: `VAL_T output_buffer[bank_size]`(URAM).
- **알고리즘**: `pe_process_loop`(L39, II=1)에서 비차단 read(`read_nb`, L47). EOD면 종료(L51). 각 payload에 대해:
  - `bank_addr = row_idx / pack_size` (L63): 8-인터리브된 행을 뱅크 주소로 변환.
  - `incr = mat_val * vec_val` (L64): MAC.
  - `q = output_buffer[bank_addr]` (L65): URAM read.
- **★ in-flight write 포워딩(L31-37, L66-87)** — 핵심 기법:
  - URAM은 RDL=3, WRL=2의 read/write latency를 가져, 같은 주소를 연속 갱신 시 stale read 발생(RAW hazard). 이를 막기 위해 **5-deep in-flight write queue `ifwq[5]`**(L31, `array_partition complete`)를 두고, 현재 `bank_addr`가 직전 5개 write 중 하나와 같으면 그 값(`ifwq[i].value`)을 포워딩(L66-71).
  - `new_q = q_fwd + incr` (L72), `#pragma HLS bind_op impl=dsp latency=0`(L73)으로 DSP 가산기 강제. `reg()`로 가산 후 레지스터 삽입(L74).
  - `output_buffer[bank_addr] = new_q_reg`(L80), ifwq shift(L83-87).
  - **결과**: URAM latency=3에도 **II=1 누산** 달성(`#pragma HLS dependence variable=output_buffer inter false`, L42).

#### `ufixed_pe_output<...>()` (L95-116)
- 누산 완료 후 `output_buffer[0..used_buf_len-1]`를 순회(II=1, L104)하며 `VEC_PLD_T`로 emit. `out_pld.idx = dump_count*pack_size + id`(L109): 뱅크 주소를 절대 행 인덱스로 복원(id = PE lane).

#### `pe<id,bank_size,pack_size>()` (L121-178) — PE 최상위
1. **출력버퍼 reset**(L131-135): `output_buffer` URAM 0 초기화. `bind_storage type=RAM_2P impl=URAM latency=3`(L128).
2. **첫 SOD 대기**(L139-144).
3. **메인 루프**(L148-171): `pipeline off`(L150)로 task 분리. `ufixed_pe_process`(EOD까지 누산, L152) → 다음 토큰이 SOD면 계속(다음 col-partition), EOS면 종료(L156-170). 즉 **여러 column partition을 같은 출력버퍼에 누적**.
4. **결과 dump**(L174-177): SOD → `ufixed_pe_output` → EOD → EOS emit.

### 3.3 `spmv/libfpga/shuffle.h` — 셔플 네트워크 (8-lane 라우팅)

SpMV의 불규칙 인덱싱을 **lane 간 동적 라우팅**으로 해결. `col_idx % num_lanes`(또는 `row_idx % num_lanes`)에 따라 데이터를 올바른 lane으로 보냄.

#### `arbiter_1p<num_lanes>()` (2 오버로드, L24-99 / L102-177)
- EDGE_PLD_T용(col_idx 기준, L24)과 UPDATE_PLD_T용(row_idx 기준, L102) 2개.
- `#pragma HLS latency min=max=ARBITER_LATENCY(=7)`(L19, L35): 7-stage 파이프라인 arbiter.
- **알고리즘**: `rotate_priority`로 회전 우선순위 부여(라운드로빈 공정성). 각 출력 lane(OLid)에 대해, 해당 출력을 요청하는(`addr % num_lanes == OLid`) 입력 lane을 검색하여 grant(`loop_A_arbsearch`, L56-77). grant 못 받은 입력은 `in_resend[ILid]=1`로 재전송 표시(L81-90).

#### `crossbar<PayloadT,num_lanes>()` (L182-201)
- `#pragma HLS inline`(L192). arbiter가 정한 `select[OLid]`에 따라 입력→출력 lane 라우팅. `out_valid[OLid]`면 `output_lanes[OLid].write(in[select[OLid]])`.

#### `shuffler_core<PayloadT,num_lanes>()` (L211-377)
- **3-stage 파이프라인**(F-fetch, A-arbiter, C-crossbar), `#pragma HLS pipeline II=1`(L255).
- **Fetch(F, L275-308)**: 각 입력 lane에서 `read_nb`. resend 표시된 lane은 재전송 payload 우선(L277-279), fetch 완료(EOD 수신)된 lane은 skip(L281). EOD 받으면 `fetch_complete[ILid]=1`.
- **종료 처리(L310-322)**: 모든 lane fetch_complete → SF_ENDING 상태로, `(ARBITER_LATENCY+1)*num_lanes`(L217) extra iteration 동안 arbiter 내부 잔여 패킷 flush 후 종료.
- **Arbiter(A, L326-336)** + **Crossbar(C, L340-346)** 호출. `next_rotate_priority` 매 사이클 회전(L336).
- resend 의존성: `#pragma HLS dependence variable=resend inter RAW true distance=9`(L256) — arbiter latency를 고려한 거리 9.

#### `shuffler<PayloadT,num_lanes>()` (L380-468) — 셔플러 최상위
- SOD 동기화(첫 launch, L391-418) → 출력에 SOD broadcast(L420-423) → `shuffler_core`(L425) → SOD/EOS 동기화(L433-459). EOS 모두 받으면 종료, 출력에 EOS broadcast(L464-467). **column partition마다 반복** 처리.

### 3.4 `spmv/libfpga/vecbuf_access_unit.h` — VAU (벡터 버퍼)

입력 벡터 `x`를 URAM에 캐싱하고, 행렬 엣지의 `col_idx`로 `x[col]`를 gather하여 `vec_val`을 채움. **double buffering**으로 로드/사용 중첩.

#### `vecbuf_reader<id,bank_size,pack_size>()` (L18-84)
- FSM(VR_IDLE/VR_WORK, L15-16). 입력 `EDGE_PLD_T`(col_idx 포함) → 출력 `UPDATE_PLD_T`.
- WORK 상태(L57-77): `abs_addr = col_idx`, `vec_val = vector_buffer[(abs_addr/pack_size) % bank_size]`(L71) — **벡터 gather**. mat_val/row_idx 전달, vec_val 채워 emit. II=1(L31).
- SOD→WORK 진입, EOD→IDLE 복귀, EOS→종료(L42-61).

#### `vecbuf_writer<id,bank_size,pack_size>()` (L92-136)
- 입력 `VEC_PLD_T`(loader가 보낸 벡터 조각) → `vector_buffer[(idx/pack_size) % bank_size] = val`(L128). **다음 column partition용 벡터를 미리 적재**(double buffering의 write 측).

#### `vecbuf_access_unit<...>()` (L146-163) — VAU 최상위
```c
VAL_T vector_buffer[bank_size];
#pragma HLS bind_storage variable=vector_buffer type=RAM_2P impl=URAM  // L155
for (unsigned i = 0; i < num_partitions + 1; i++) {
    #pragma HLS dataflow                                                // L159
    vecbuf_writer<...>(vec_input, vector_buffer);  // 다음 파티션 적재
    vecbuf_reader<...>(input, output, vector_buffer);  // 현재 파티션 gather
}
```
- **double buffering 핵심**: writer(다음 벡터 적재)와 reader(현재 벡터 사용)를 `#pragma HLS dataflow`(L159)로 같은 partition 루프 내 병렬. URAM RAM_2P(L155)로 read/write 동시. `num_partitions+1`은 마지막 EOS 소비용(L157-158).

### 3.5 `spmv/libfpga/spmv_cluster.h` — 1개 연산 클러스터 (dataflow 파이프)

8-lane 1개 HBM 채널을 담당하는 완결 SpMV 파이프. 6단 dataflow: **Vector Unpacker → Matrix Loader → Shuffle1 → VAU(×8) → Shuffle2 → PE(×8) → Result Packer**.

#### `CPSR_matrix_loader()` (L34-107) — CPSR 디코더
- **CPSR(Cyclic Packed Streams of Rows) 포맷 디코딩**. `matrix_hbm`에서 partition별 메타데이터 읽음:
  - `partition_start`(L48), `part_len_pkt`(8개 stream 길이, L49-55).
  - `num_reads = array_max(stream_length)`(L58): 8 lane 중 최장 stream 길이만큼 read.
- **행 마커 처리(L80-82)**: `mat_pkt.indices.data[k] == IDX_MARKER(0xffffffff)`면 데이터가 아니라 **행 전진 마커** → `row_idx[k] += PACK_SIZE * vals.data[k]`(빈 행 skip / row interleaving). 그 외는 `EDGE_PLD_T`(mat_val,col_idx,row_idx) emit(L84-88).
- SOD/EOD/EOS 토큰 부착(L67-69, L95-106). column partition 루프(L42).

#### `spmv_vector_unpacker()` (L109-127)
- `VEC_AXIS_T` 입력 → 8개 `VEC_PLD_T` lane으로 분해(L117-123). `VEC_AXIS_VAL(pkt,k)` bitcast. EOS까지 반복(L125).

#### `spmv_result_packer()` (L133-193)
- 8개 PE 출력 lane → 1개 `VEC_AXIS_T`로 재패킹(L145-171). 8 lane이 모두 SOD/EOD/EOS면 그 토큰을 AXIS user로 설정(`and_reduce`, L173-187). 일반 데이터는 `pkt_idx` 증가시키며 emit.

#### `spmv_cluster<cluster_id>()` (L196-373) — 클러스터 최상위
- `#pragma HLS dataflow`(L227). 6종 FIFO(`ML2SF, SF2VAU, VAU2SF, SF2PE, PE2PK, UPK2VAU`) depth=FIFO_DEPTH, impl=SRL(L213-225).
- **연결(dataflow)**:
  1. `spmv_vector_unpacker(vec_in, UPK2VAU)` (L229)
  2. `CPSR_matrix_loader(matrix_hbm, ..., ML2SF)` (L237)
  3. `shuffler<EDGE_PLD_T,8>(ML2SF, SF2VAU)` — Shuffle1, col_idx 기준 (L249)
  4. `vecbuf_access_unit<0..7>(SF2VAU[k], UPK2VAU[k], VAU2SF[k], ...)` — 8개 VAU (L258-305)
  5. `shuffler<UPDATE_PLD_T,8>(VAU2SF, SF2PE)` — Shuffle2, row_idx 기준 (L311)
  6. `pe<0..7>(SF2PE[k], PE2PK[k], rows_in_partition/PACK_SIZE)` — 8개 PE (L320-359)
  7. `spmv_result_packer(PE2PK, res_out)` (L365)
- **2단 셔플의 의미**: Shuffle1은 **열 인덱스 기준**으로 데이터를 올바른 VAU(벡터 뱅크)로, Shuffle2는 **행 인덱스 기준**으로 올바른 PE(출력 뱅크)로 라우팅. 이것이 8-lane 병렬 SpMV의 핵심.

> 주의: `spmv_cluster.h`는 자기 자신을 include(L16)하고 `#pragma HLS stream variable=FS2PE`(L216)에서 미선언 변수 `FS2PE`를 참조하는 등 사소한 코드 결함이 있음(실제 stream은 `SF2PE`). HLS가 무시하거나 경고 처리할 것으로 "추정" — 실제 합성 영향은 **확인 불가**.

### 3.6 SLR 서브커널 — `spmv_sk0/1/2.cpp`

3개 SLR(super logic region)에 클러스터를 분산. `extern "C"` Vitis 커널.

#### `spmv_sk1()` (L10-121) — 대표 (SLR1, 6 클러스터)
- 인터페이스: 6개 HBM 포트 `matrix_hbm_4..9` 각각 `m_axi ... bundle=spmv_mat4..9`(L25-30), 스칼라는 `s_axilite ... bundle=control`(L31-42), `vec_in/res_out`은 `axis register both`(L44-45).
- `#pragma HLS dataflow`(L47). `axis_duplicate<6>(vec_in, vec_dup)`(L56)로 입력 벡터를 6 클러스터에 복제 → `spmv_cluster<4..9>(matrix_hbm_k, vec_dup[k], res[k], ...)`(L58-116) → `axis_merge<6>(res, res_out)`(L118)로 6 결과 병합.
- `spmv_sk0.cpp`(L13-): 동일 구조, 4 클러스터(HBM0-3, cluster 0-3), `axis_duplicate<4>`/`axis_merge<4>`. `spmv_sk2.cpp`: 6 클러스터(HBM10-15, cluster 10-15)로 "추정"(sk1과 동형).
- **클러스터 ID 매핑**: SK0=cluster 0-3, SK1=cluster 4-9, SK2=cluster 10-15 → 총 16. `common.h` 클러스터 수와 일치.

### 3.7 `spmv/spmv_vector_loader.cpp` — 입력 벡터 로더

#### `load_duplicate()` (L7-79)
- HBM의 `packed_dense_vector`를 column partition(`LOGICAL_VB_SIZE` 단위)으로 분할(L13-19) 로드. 각 partition마다 SOD(L26-32) → 8-packed 벡터 패킷을 `VEC_AXIS_T`로 변환하여 **3개 SLR로 동시 복제**(L41-57) → EOD(L60-66). 마지막 EOS(L71-77).
- `assert(part_len % PACK_SIZE == 0)`(L39).

#### `spmv_vector_loader()` (L96-121) — Vitis 커널
- `packed_dense_vector` m_axi(L103), `to_SLR0/1/2` AXIS 출력(L108-110). `#pragma HLS dataflow`(L112). `load_duplicate`(L115) → `write_k2ks`로 3 SLR FIFO를 AXIS로 펌프(L116-118).

### 3.8 `spmv/spmv_result_drain.cpp` — 결과 수집기

#### `spmv_result_drain()` (L11-126)
- 3 SLR(`from_SLR0/1/2` AXIS)에서 결과를 **라운드로빈**으로 수집해 `packed_dense_result` HBM에 write-back.
- FSM `current_input`(0/1/2, L42-101): SLR0는 `SK0_CLUSTER(4)`개, SLR1/2는 6개씩 패킷을 순서대로 읽음. EOS 받으면 해당 `finished[i]=true`(L49). SOD/EOD는 write 안 함(L52). 데이터 패킷만 `do_write=true`.
- write(L104-113): `abs_pkt_idx = write_counter + row_part_id*LOGICAL_OB_SIZE/PACK_SIZE`(L36,104), 8-packed `PACKED_VAL_T`로 HBM 저장. 모든 SLR `finished`면 종료(L102).

### 3.9 `spmv/k2k_relay.cpp` — SLR 간 AXIS 중계

#### `k2k_relay()` (L7-32)
- `#pragma HLS interface ap_ctrl_none`(L11): 제어 핸드셰이크 없는 free-running 커널. 단순히 `in.read()` → `out.write()` 패스스루(L24-28, II=1). **SLR2는 물리적으로 멀어** loader↔SK2, SK2↔drain 경로에 relay 2개(`relay_SK2_vin`, `relay_SK2_rout`)를 삽입(timing closure용). 근거: `spmv.ini` L7,13-14,35-36,39-40.

### 3.10 `spmv/libfpga/stream_utils.h` — AXIS 유틸

- `axis_duplicate<N>()` (L8-26): 1 AXIS → N 복제. `reg(reg(pkt))` 다중 레지스터 삽입(L17)으로 fanout timing 완화. EOS까지(L24).
- `axis_merge<N>()` (L36-75): N AXIS → 1 cyclic 병합(L40-64). 각 입력 라운드로빈, EOS 받은 입력 skip, 데이터만 `pkt_idx(c)` 부여. 모두 EOS면 종료 후 EOS emit(L67-74).

### 3.11 `spmv/spmv.ini` — Vitis 시스템 연결 (물리 매핑)

- **커널 인스턴스**(L2-7): sk0/sk1/sk2 각 1개, vector_loader(VL), result_drain(RD), k2k_relay 2개.
- **SLR 배치**(L8-14): SK0→SLR0, SK1→SLR1, SK2→SLR2, VL/RD→SLR0, relay 2개→SLR1.
- **HBM 매핑**(L15-32): SK0→HBM[0-3], SK1→HBM[4-9], SK2→HBM[10-15], VL→HBM[20], RD→HBM[21]. **16채널 행렬 + 입력/출력 벡터 각 1채널**.
- **스트림 연결**(L33-40): VL→{SK0,SK1,relay→SK2}, {SK0,SK1,relay←SK2}→RD. 폭 32. SLR2만 relay 경유.

### 3.12 SW & 성능 모델 (요약)

- `sw/host.cpp`: `compute_ref()`(CSR float 참조 SpMV, L33-48), `verify()`(epsilon 1e-4 비교, L50-74), `unpack_vector()`(L76-86). XRT/OpenCL(`xcl2.hpp`, L10)로 xclbin 로드·실행·검증. HBM[0-21] 매핑 상수(L15-24).
- `sw/data_formatter.h`: CSR 전처리 유틸 — `util_round_csr_matrix_dim`(차원 패딩, L14-29), `util_normalize_csr_matrix_by_outdegree`(out-degree 정규화, L32-47), `util_pad_marker_end_of_row_*`(IDX_MARKER 삽입으로 CPSR 행 경계 인코딩, L50-83+). **CSR→CPSR 변환의 호스트측 핵심**.
- `performance_model/`: `performance_model.cpp` + `design_space_exp.cpp` + `include/*`(libfpga 헤더 복제). 합성 없이 사이클 정확 throughput/대역폭을 모델링하고 DSE 수행(클러스터 수·뱅크 크기 등 설계변수 탐색). 상세 분석은 본 과제 범위에서 요약만.
- `unit_tests/`, `unit_test_wrapper/`: PE/shuffle/VAU/cluster/module 단위 csim 테스트. 모듈 단위 검증 인프라.

## 4. 데이터플로우 / 실행 흐름

### 전체 시스템 dataflow
```
HBM(vec)→ [VL] →복제3→ SLR0:[SK0(4클러스터)] ─┐
                      → SLR1:[SK1(6클러스터)] ─┼→ [RD] →HBM(result)
              relay → SLR2:[SK2(6클러스터)] ──┘
HBM(mat0-15)→ 각 클러스터로 직접 스트리밍
```

### 클러스터 내부 dataflow (8-lane)
```
vec_in(AXIS) → Vector Unpacker ─8→ ┐
matrix_hbm → CPSR Matrix Loader ─8→ Shuffle1(col%8) ─8→ VAU×8(x gather) ─8→ Shuffle2(row%8) ─8→ PE×8(URAM 누산) ─8→ Result Packer → res_out(AXIS)
```

### 메모리 계층
| 계층 | 저장소 | 용도 |
|------|--------|------|
| 외부 | HBM 16ch(mat) + 2ch(vec/res) | 행렬(CPSR) 스트리밍, 입출력 벡터 |
| on-chip | VAU URAM `vector_buffer`(VB_BANK_SIZE=4K×8) | 입력 벡터 x 캐싱(double buffer) |
| on-chip | PE URAM `output_buffer`(OB_BANK_SIZE=8K×8) | 행별 부분합 누적 |
| 파이프 | hls::stream FIFO(SRL, depth64) | 스테이지 간 |

### 병렬화 차원
1. **16 클러스터**(16 HBM 채널) — 채널 병렬.
2. **8 lane/클러스터**(PACK_SIZE) — lane 병렬.
3. **dataflow** — 클러스터 내 6단 task 병렬.
4. **double buffering**(VAU) — column partition 간 로드/사용 중첩.
→ 이론 병렬도 16×8 = **128-way MAC**.

### 양자화/데이터타입
- 기본: `ap_ufixed<32,8>`(고정소수점). 변형 IMPL: `fixed`(고정), `float_pob`(float + partial output buffer), `float_stall`(float + stall + row interleaving). float는 누산 latency 때문에 PE에서 stall 또는 partial buffer 필요 — 고정소수점이 가장 효율적임을 시사.

## 5. HW/SW 매핑

| 계층 | 구성요소 | 역할 |
|------|----------|------|
| SW (host) | `sw/host.cpp`, `benchmark.cpp` | xclbin 로드, HBM 버퍼 할당, 실행, 검증, GBPS/GOPS 측정 |
| SW (전처리) | `data_loader.h`, `data_formatter.h` | .npy/CSR 로드, 차원 패딩, out-degree 정규화, **CSR→CPSR 인코딩**(IDX_MARKER) |
| SW (모델) | `performance_model/` | 합성 전 사이클정확 성능 예측 + DSE |
| HW (top kernels) | `spmv_sk0/1/2`, `vector_loader`, `result_drain`, `k2k_relay` | 6종 Vitis 커널 |
| HW (libfpga) | `pe.h, shuffle.h, vecbuf_access_unit.h, spmv_cluster.h, stream_utils.h, common.h` | 재사용 HLS 모듈 |
| HW 물리 매핑 | `spmv.ini` | SLR 배치, HBM 채널 바인딩, AXIS 스트림 연결 |

## 6. 빌드·실행

근거: `Readme.md` L26-78.
1. **데이터셋**: `cd datasets && source download.sh` → graph/pruned_nn .npy.
2. **cnpy 설치**: .npy 로드용 C++ 라이브러리. `CNPY_INCLUDE/CNPY_LIB/LD_LIBRARY_PATH` 환경변수 설정(L39-44).
3. **Vitis 2020.2 설정**: `printenv VITIS` 확인(L46-52).
4. **사전 합성 데모**: `cd sw && make demo` → `demo_spmv.xclbin` 실행. 출력 예: `{Preprocessing: 0.64566 s | SpMV: 0.77102 ms | 49.4087 GBPS | 12.9698 GOPS}`(L62-67). 대역폭 = 연산처리량/2*8(L67).
5. **빌드**: `cd sw && make benchmark IMPL=<fixed/float_pob/float_stall>`(L70-78).
- 커널 빌드(`spmv/makefile`)는 `spmv.ini` connectivity로 v++ 링크. 상세 미열람(요약).

## 7. 의존성

- **HW**: Xilinx Vitis 2020.2, Alveo U280 shell `xilinx_u280_xdma_201920_3`. HLS 라이브러리 `ap_fixed.h, ap_int.h, ap_axi_sdata.h, hls_stream.h`(common.h L4-6). `utils/x_hls_utils.h`의 `reg()`(common.h L16).
- **SW**: cnpy(.npy 로드), Xilinx XRT/OpenCL(`xcl2.hpp`), C++17.
- **vendor(제외)**: `xrt/includes/{xcl2,oclHelper,cmdparser,logger}` — Xilinx 호스트 헬퍼.
- **데이터(제외)**: SuiteSparse graph + pruned NN 행렬 .npy.

## 8. 강점 / 한계 / 리스크

### 강점
- **HBM 대역폭 활용 극대화**: 16채널 동시 스트리밍 + 8-lane 패킹으로 128-way 병렬. 237 MHz 달성(SLR간 relay로 timing closure).
- **셔플 네트워크**(2단 arbiter+crossbar)로 SpMV 불규칙 인덱싱을 동적 라우팅으로 해결 — hls-spmv의 단일포트 BRAM gather 한계를 근본적으로 극복.
- **PE의 in-flight forwarding**(ifwq[5])으로 URAM latency=3에도 II=1 누산 — 고정소수점 가속의 정교한 핵심 기법.
- **CPSR 포맷 + IDX_MARKER 행 인코딩**으로 빈 행 skip / row interleaving을 HW 친화적으로 표현.
- **재사용 가능한 libfpga 모듈 라이브러리** + 성능 모델 + 단위 테스트 → 연구급 완성도.
- SOD/EOD/EOS 토큰 기반 스트림 동기화로 데이터 의존 길이를 우아하게 처리.

### 한계
- **하드코딩된 구성**: SK0=4/SK1=6/SK2=6, PACK_SIZE=8 등 U280 전용 튜닝. 다른 디바이스 이식 시 재설계.
- **고정소수점 정밀도**: `ap_ufixed<32,8>`는 부호없음(unsigned) — 음수 가중치 미지원(out-degree 정규화 그래프 SpMV 가정). 일반 부호 행렬은 float 변형 필요.
- 셔플 네트워크의 resend/재시도 메커니즘은 worst-case throughput을 저하시킬 수 있음(충돌 빈도 의존).
- `spmv_cluster.h`의 self-include(L16) 및 미선언 `FS2PE` 참조(L216) 등 코드 결함 존재(합성 영향 확인 불가).

### 리스크
- Vitis/shell 버전 강결합(2020.2, 특정 U280 shell). 신버전은 별도 `2021+` 브랜치 필요(Readme L5-6).
- cnpy/datasets 외부 의존으로 재현 장벽.
- 부호없는 고정소수점 가정이 응용 도메인을 제약.

## 9. 우리 프로젝트 관점 시사점 (고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적)

> ViT/Transformer 가속기 + XR 시선추적 추정 하에, HiSparse는 **재사용 가치가 매우 높은 레퍼런스**:

1. **셔플 네트워크 → token/head 라우팅 재사용 (최우선)**: 2단 arbiter+crossbar 셔플은 sparse attention(pruned QK^T)이나 MoE(expert dispatch)에서 token을 동적 라우팅할 때 직접 차용 가능. `arbiter_1p`의 rotate_priority 라운드로빈은 head/expert 간 공정 분배에 유용. HG-PIPE 파이프라인에서 토큰 셔플 단계로 모듈화 가능.

2. **PE in-flight forwarding → 누산 II=1 기법**: GEMM/attention 누산 시 on-chip RAM(URAM/BRAM) latency를 ifwq 포워딩으로 숨겨 II=1을 달성하는 기법은, systolic array의 accumulator 뱅킹이나 LayerNorm/softmax 누산 datapath에 그대로 적용 가능. `bind_op impl=dsp latency=0` + `reg()` 패턴도 차용.

3. **고정소수점 양자화 인프라**: `ap_ufixed<32,8,AP_RND,AP_SAT>` + PACK_SIZE 패킹은 우리 INT8/INT4·Mamba 양자화 가속기의 데이터 패킹/포화 처리 baseline. 부호 지원(`ap_fixed`)으로 확장하면 Transformer 가중치에 적합.

4. **HBM 멀티채널 + multi-SLR dataflow 아키텍처**: VL→복제→SLR별 sub-kernel→merge→RD 구조는, 대형 ViT의 weight stationary를 여러 HBM 채널/SLR에 분산하는 데이터 분배 패턴으로 재사용. `k2k_relay`로 SLR간 timing closure하는 기법은 대형 디자인 필수 노하우.

5. **double buffering VAU → 가중치/벡터 프리페치**: VAU의 writer/reader dataflow double buffering은 attention의 K/V 캐시나 FFN 가중치 프리페치에 적용 가능 — 메모리 지연 은닉.

6. **성능 모델 + DSE 인프라**: `performance_model/`의 사이클정확 모델 + design_space_exp는 우리 가속기의 클러스터 수/뱅크 크기/병렬도 DSE에 그대로 방법론 차용 가능(합성 없이 빠른 탐색).

7. **SOD/EOD/EOS 스트림 프로토콜**: 가변 길이(토큰 수 가변, sequence length 가변) dataflow를 토큰 기반으로 동기화하는 패턴은 dynamic shape Transformer 파이프에 유용.

8. **시선추적 직접 연관 낮음**: gaze estimation 자체와는 무관(SpMV 인프라). 단, gaze 모델이 sparse/pruned되거나 graph 기반(예: landmark graph)이면 SpMV 엔진 직접 재사용 가능.

## 10. 근거 표기

- **확인(코드 라인 직접)**: §3 전체(common.h/pe.h/shuffle.h/vecbuf_access_unit.h/spmv_cluster.h/sk1/vector_loader/result_drain/k2k_relay/stream_utils 라인 인용), §3.11 spmv.ini, §3.12 host/data_formatter, §6 빌드(Readme 인용), §1 논문/디바이스(Readme 인용).
- **추정**: `spmv_sk2.cpp` 내부 구조(sk1 동형으로 추정, 직접 미열람), `spmv_cluster.h`의 self-include/FS2PE 결함의 합성 영향, `_auto`류 변형 없음, performance_model 세부.
- **확인 불가**: `fpgafp193a-du.pdf` 논문 본문 수치(바이너리), `demo_spmv.xclbin` 내부, datasets 실제 행렬 특성, makefile 세부 v++ 옵션, FS2PE 미선언이 합성 시 실제로 무시되는지 여부.
