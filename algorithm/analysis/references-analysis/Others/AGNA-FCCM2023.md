# AGNA-FCCM2023 정밀 분석

> 분석 대상 경로: `REF/Others/AGNA-FCCM2023`
> 분석 방식: 자체 핵심 소스(Python DSE/코드생성, HLS 커널, SystemVerilog RTL)를 Read로 직접 읽고 라인 근거 기반 작성. third-party/IP/비트스트림/생성물 제외.

---

## 1. 개요

- **목적**: 타깃 DNN 모델 사양 + 타깃 FPGA 플랫폼 사양을 입력받아, 그 조합에 최적화된 FPGA 가속기(아키텍처 파라미터 + 레이어별 스케줄/인스트럭션)를 **자동 생성**하는 하드웨어 제너레이터.
- **한줄요약**: "**Mixed-Integer Geometric Programming(MIGP) 기반 설계공간탐색(DSE)으로 PE 어레이 아키텍처를 풀고, 그 결과로 RTL/HLS 파라미터 헤더와 레이어 인스트럭션을 코드생성하는, 파라미터화된 cascaded-DSP MAC 어레이 CNN 가속기 생성 프레임워크**".
- **원논문**: Y. Ding, J. Wu, Y. Gao, M. Wang, H. K.-H. So, "Model-Platform Optimized Deep Neural Network Accelerator Generation through Mixed-Integer Geometric Programming", FCCM 2023, pp.83-93, DOI 10.1109/FCCM57271.2023.00018 (README.md L3-L18 근거).
- **타깃 디바이스**: 검증 환경은 Vivado 2021.2, zcu102 (README.md L26, L31). 플랫폼 spec JSON으로 다수 보드 지원 — `zcu102_128_8/16`, `zcu102_256_8/16`, `ultra96_128_8/16`, `ku115_128_16`, `u200_512_8/16` (software/spec/platforms/*.json 파일명 근거). 파일명 규칙은 `<board>_<dbus_width>_<data_width>`로 추정(코드상 dbus_width/data_width를 plltfm spec에서 읽음, 확인 가능).
- **저자 그룹**: Hayden K.-H. So (HKU) 그룹 — REMOT-FPGA-22와 동일 그룹(공저자 Yizhao Gao, Hayden So 중복), 동일 코드 컨벤션/디렉토리 구조(software+hardware 분리) 공유(추정).

---

## 2. 디렉토리 구조 (자체 소스 트리 + 제외 목록)

### 자체 핵심 소스
```
AGNA-FCCM2023/
├── README.md, Dockerfile, LICENSE
├── software/                         # Python: DSE(MIGP) + 스케줄링 + 코드생성
│   ├── agna.py                       # 메인 오케스트레이터 (AGNA 클래스)
│   ├── run_schedule.py               # 엔트리포인트
│   ├── Makefile, run_all.mk          # make all PLATFORM=.. MODEL=..
│   ├── generate_param_header.py      # arch_search.json -> param.h / param.sv 코드생성
│   ├── generate_model_spec.py, generate_platform_spec.py
│   ├── extract.py, environment.yml
│   ├── solver/                       # 최적화 엔진
│   │   ├── base_rnr.py               # Relaxation & Rounding 베이스 (GP완화→정수)
│   │   ├── arch_search.py            # ArchSearch: 아키텍처 MIGP
│   │   ├── op_schedule.py            # OperationSchedule: 레이어별 스케줄 MIGP
│   │   ├── gpkit_solver.py           # GPKit(기하계획 완화) 솔버 래퍼
│   │   ├── scip_solver.py            # SCIP(정수계획) 솔버 래퍼
│   │   └── common.py                 # SolverVar / SolverConstr
│   ├── parser/  (layer_parser.py, model_parser.py)
│   ├── utils/                        # arch_spec, model_spec, node_param,
│   │                                 #  schedule_param, platform_spec,
│   │                                 #  instr_gen, hw_sim, perf_eval, common
│   └── spec/platforms/*.json         # 보드별 자원/대역폭 사양
├── hardware/
│   ├── Makefile, prj/ (generate.tcl, project.tcl, update_prj.tcl)
│   ├── hls/hls_src/                  # 보조 HLS 커널 (제어/IO)
│   │   ├── core_controller.cpp/.h    # 인스트럭션 디코드 → 서브유닛 인스트럭션 생성
│   │   ├── agna_aux.cpp/.h           # m_axi에서 인스트럭션을 읽어 코어로 스트림
│   │   ├── layout_convert.cpp/.h     # NHWC/타일 레이아웃 변환
│   │   ├── param.h / param.h.tpl     # ★코드생성 대상 C 헤더(템플릿)
│   │   ├── common.h, *_tb.cpp        # 공통 헤더, 테스트벤치
│   │   └── *.tcl (Vitis HLS 스크립트)
│   └── rtl/                          # ★메인 데이터패스 (SystemVerilog)
│       ├── param.sv / param.sv.tpl   # ★코드생성 대상 RTL 파라미터 헤더(템플릿)
│       ├── def.sv                    # 파생 파라미터/매크로 정의
│       ├── agna_core.sv, top.sv, top_wrapper.v  # 코어/탑
│       ├── pe_array.sv, pe.sv        # PE 어레이 / 단일 PE
│       ├── cu.sv                     # Compute Unit: DSP48E2 cascade MAC + 누산
│       ├── adder_tree.sv            # 재귀 가산기 트리 (parallel 모드)
│       ├── pe_ctrl_{updt,exec,wb}.sv # PE 3단계 컨트롤(업데이트/실행/라이트백)
│       ├── abuf_exec_addr.sv, prefetch_buf.sv
│       ├── new_buf_{sdp,tdp}.sv, rf.sv, buf_bn.sv, bn.sv, bn_unit.sv
│       ├── acti_unit.sv, res_adder.sv, layout_converter.sv
│       ├── mm2s_*.sv, s2mm_*.sv      # AXI DataMover 명령 생성/태그매칭
│       ├── fifo_axis.sv, axis_interconnect.sv, delay_chain*.sv
│       └── core_mem_itf.sv, ps_ctrl.sv, cu.sv
```

### 분석 제외 (이름만 언급, 분석 안 함)
- `.git/`, `hardware/ip/*.xci` (Xilinx IP: axis_interconnect, axi_datamover, agna_aux IP wrapper), `hardware/hls/hls_ip/`(.gitignore), 생성물 `hardware/prj/`(비트스트림), `software/results/`(런타임 출력), `chip_bd_wrapper.v`(Vivado block-design 래퍼, 생성물 성격).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 software/agna.py — `AGNA` 클래스 (메인 오케스트레이터)
- 역할: PlatformSpec 1개 + ModelSpec 리스트를 받아 (1) 아키텍처 탐색 → (2) 레이어별 스케줄 → (3) JSON/CSV 산출 → (4) 시뮬레이션의 전 과정을 지휘.
- `__init__` (L18-29): `pltfm_spec`, `model_spec_list`, `output_dir` 보관.
- `run()` (L59-89): `arch_search.json`이 있으면 로드, 없으면 `run_arch_search()`로 신규 탐색(L63-72). 그 다음 모델의 **unique_nodes**(같은 레이어 중복 제거)에 대해서만 `op_schedule_<node>.json`을 로드/생성(L74-89). → **레이어 단위 독립 스케줄링** 구조.
- `run_arch_search()` (L165-195): `arch_search_config`를 구성하는데 핵심은 `'pack_dsp': self.pltfm_spec.data_width==8`(L173) — **INT8일 때 DSP packing 활성화**, `'timelimits':300`(L178). `ArchSearch(...).optimize()` 호출 후 best_solution을 `arch_search.json`으로 저장(L185-192).
- `run_op_schedule()` (L197-241): `OperationSchedule(...).optimize()`. 1차 실패 시 `bound_range`를 2배로 늘려 재시도(L225-237) — **rounding 탐색범위 확장 fallback**.
- `update_csv()` (L122-163): 노드별 schedule을 모아 이론 사이클 `theo_cycle`과 스케줄 사이클 `schd_cycle`을 누적, 효율비 `schd_cycle/theo_cycle`와 `cycle/200e3`(=200MHz 가정 시 시간(ms), L160) 출력.
- `simulate()` (L243-264): `HardwareSimulator(arch_spec, model)`로 updt/exec/wrbk 이벤트 시뮬레이션(드로잉/플롯은 주석처리).

### 3.2 software/solver/ — MIGP DSE 엔진 (★ 본 repo의 핵심 기여)
이 repo의 학술적 핵심은 **"Geometric Programming(GP) 완화 → 정수 반올림"**이라는 2단계 최적화다.

#### base_rnr.py — `BaseRnR` (Relaxation & Rounding)
- `optimize()` (L25-46): 동일한 `var_list`/`constr_list`로 (1) **GPKit 솔버로 완화(연속) GP를 풀고**(L30-32), (2) `append_strict_bound()`로 완화 해 주변에 정수변수 경계를 좁힌 뒤(L35), (3) **SCIP 솔버로 정수 비선형 문제를 푼다**(L36-40). 이것이 MIGP를 실제로 푸는 방식.
- `append_strict_bound()` (L48-78): 완화 해 `center_tmp`를 중심으로 `bound_range`(기본 `np.e`, L139)만큼 곱/합 방식으로 lb/ub를 좁힘. `is_strict` 변수에만 적용 — 정수 탐색공간을 GP해 근방으로 축소해 SCIP을 가속.
- `default_config` (L137-145): `bound_range=e, use_close=True, maxnthreads=6, timelimits=300`.

#### arch_search.py — `ArchSearch(BaseRnR)` (아키텍처 변수 탐색)
- 결정변수 `build_var_list()` (L31-58): PE 어레이 차원 `A_{K,C,I,J,H,W}`(6축, `agna_dim_tuple`), `packed_dsp_num`, `pe_num`, 각 (모델×unique노드)별 `_cycle_{dim}`, `_S_{dim}`(공간 병렬 분배), `_cycle`(노드 사이클), `obj`.
- 아키텍처 제약 `append_arch_constr()` (L85-105): `A_I<=A_J`(L88), `A_H<=A_W`(L92), **`A_H*A_W<=32`**(L96), **`8<=pe_num<=128`**(L99-102). → PE 어레이는 6차원(K=출력채널, C=입력채널, I·J=커널, H·W=출력공간) 언롤 팩터의 곱이며, 공간차원 곱을 32로 제한.
- 자원 제약 `append_resource_constr()` (L107-143): `get_packed_dsp_num_expr`/`get_total_dsp_num_expr`로 총 DSP가 **`max_dsp*0.9`** 이하(L129), 총 BRAM이 **`max_bram*0.9`** 이하(L141)가 되도록. 즉 자원의 90%를 상한으로 둠.
- 노드 제약 `append_node_constr()` (L145-269): 각 차원에서 `cycle_d * S_d * A_d >= loop_bound`(L184-195)로 모든 출력원소 커버를 보장하고, SCIP 전용으로 상계도 추가(L196-208). `is_dpws`(depthwise) 노드는 K·C 차원을 특수 처리(L159-181). 통신 사이클(`PlatformPerf(...).evaluate()`)을 노드 사이클 하한으로 추가(L253-269) — **연산-통신 동시 모델링**.
- 목적함수 `append_obj_constr()` (L271-309): `obj >= prod_models( sum_nodes( node_cycle * unique_cnt ) )` — 모델별 총 사이클의 곱(여러 모델 동시 최적화 시 곱 형태)으로 GP에 적합한 posynomial 형태. GPKit은 `>=`, SCIP은 `==`로 scope 분기(L289, L307).
- `build_best_solution()` (L73-83): 해에서 `pe_arch=(A_K..A_W)`, `pe_num`, `buf_arch`, `rf_arch`, `dbus_width`, `data_width`를 모아 `ArchSpec`을 생성.

#### op_schedule.py — `OperationSchedule(BaseRnR)` (레이어별 루프 스케줄)
- 스케줄 변수 `build_var_list()` (L29-59): `agna_level_tuple`(추정: T,S,P,Q,F의 5계층 — 코드에서 `schd_T/S/P/Q/F_*` 사용 확인) × 6차원의 `schd_{level}_{dim}`. 즉 **6차원 루프를 5계층(타일/공간/시간 등)으로 분해**하는 타일링 팩터를 정수 결정변수로 둠. 추가로 `rf_cycle/abuf_cycle/wbuf_cycle/pe_cycle/pbuf_cycle/w_ratio/in_bound_cycle/out_bound_cycle/obj`.
- 아키텍처 고정 제약 `append_arch_constr()` (L106-150): **입력 재사용을 위해 `S_I,T_I,S_J,T_J=1` 고정**(L110-114), **rf 재사용을 위해 `P_K=1`**(L116-118). 각 차원 곱이 1023 이하(인덱스 비트폭 제약, L122-131). depthwise(`is_dpws`)는 K 전 계층=1·`F_C=1`(L133-140), maxpool은 `Q_I/Q_J`를 커널크기로 고정(L142-150).
- 자원 제약 `append_resource_constr()` (L152-216): `prod(S_*)<=pe_num`(공간병렬은 PE수 이내, L156-159), `F_{dim}<=pe_arch[dim]`(PE 내부 언롤 ≤ 아키텍처 언롤, L161-165), rf/abuf/wbuf/pbuf 깊이 사용량 제약(GP는 1.5배 여유, SCIP은 엄밀; L167-216) — **메모리 계층별 버퍼 깊이 위반 방지**.
- 사이클 모델 `append_node_cycle_constr()` (L265-334): abuf/wbuf 사이클은 `... / dbus_width * data_width`로 **대역폭 환산**(L277, L289, L300), pe_cycle=`prod(P)*rf_cycle`(L302-306), pbuf 사이클은 출력 shape×psum 깊이 기반(L308-334).
- 목적함수 `append_obj_constr()` (L336-389): `w_ratio`(weight 재로딩 비율, L340-345), `in_bound_cycle>=abuf_cycle*w_ratio+wbuf_cycle` 및 `>=pe_cycle`(L356-364), `out_bound_cycle>=pbuf_cycle*w_ratio` 및 `>=in_bound_cycle`(L366-373). 최종 `obj`는 `prod(T)*in_bound_cycle + prod(T)*out_bound_cycle`(depthwise는 단순화; L374-389) — **타일 반복수×병목 사이클의 합**.

#### scip_solver.py — `MySCIPSolver`
- `pyscipopt.Model`로 정수 NLP 빌드(L61-72). `scip_vars`는 `use_in_scip` 변수만, `addCons`로 제약 추가, `setObjective(obj)`(L71). 외부 `scip` 실행파일 명령으로 풀이(`solve_scip_model_cmd`, L42)하여 멀티스레드/타임리밋 적용.

### 3.3 software/utils/arch_spec.py — `ArchSpec` (자원 산정 모델)
- `rf_depth = rf_arch//2//data_width`(L39), `abuf_depth = buf_arch[0]*16384//2//data_width`(L43), `wbuf_depth = buf_arch[1]*...`, `pbuf_depth = buf_arch[2]*16384//2//(data_width*2)`(L51). → **BRAM 1개=16384bit(18Kb의 근사) 기준으로 버퍼 깊이 환산**, psum은 2배폭.
- `get_packed_dsp_num_expr()` (L105-116): `pack_dsp`(INT8) 시 `A_I*A_J/2`(DSP48E2 1개에 2 MAC packing), 아니면 `A_I*A_J`. → **INT8 DSP packing의 정량 모델**.

### 3.4 software/utils/instr_gen.py — `InstrGen` (★ 레이어→128bit 인스트럭션 코드생성)
- `replace_bit/replace_instr` (L6-34): 128bit(2×uint64) 인스트럭션 워드의 임의 비트필드를 안전하게 채우는 비트조작 유틸(오버플로 assert 포함).
- `verify_single_node()` (L68-161): 노드 스케줄 검증 후 **하드웨어가 필요로 하는 파생 파라미터**를 계산 — `l_ih/l_iw`(패딩·스트라이드 반영 입력 타일 크기, L79-83), `schd_pq/schd_pqf/schd_ts`(계층 팩터 곱, L85-95), `wpb`(words per block, L96), `opt_*`(주소계산용 사전곱; L97-103) 등.
- `build_single_node()` (L167-303): 한 노드를 **7개의 128bit 인스트럭션**으로 인코딩(`INSTR_PER_NODE=7`). type 0(레이어 형상/스트라이드/패딩, L177-201), type1(P loop factors), type2(PQF), type3(TS/T/S), type4(주소: act_out/weight/act_in base, L282-285), type5(wpb/pe_num/bn/res 주소), type6(opt_* + end_of_model, L292-302). → **컴파일러처럼 모델을 ISA 워드로 변환**.
- `build()` (L56-66): 모든 노드를 순회, 마지막 노드에 `opt_eom`(end-of-model) 세팅(L60).

### 3.5 software/generate_param_header.py — 파라미터 헤더 코드생성
- `generate_param_header()` (L7-20): `arch_search.json`을 읽어 `.tpl` 템플릿의 `$PARAM_PE_NUM$`, `$PARAM_A{K,C,I,J,H,W}$`, `$PARAM_DATA_WIDTH$`, `$PARAM_DBUS_WIDTH$` 등을 정규식 치환.
- `main()` (L23-38): C 헤더 `hls/hls_src/param.h`와 RTL 헤더 `rtl/param.sv`를 동시 생성(L34-37). → **하나의 DSE 결과가 HLS와 RTL 양쪽 파라미터를 동기 생성** (param.sv.tpl L5-12: `HW_CONFIG_PE_NUM`, `HW_CONFIG_A_*`, `HW_CONFIG_MACC_WIDTH`).

### 3.6 hardware/hls/hls_src/ — 보조 HLS 커널 (제어/IO 파이프라인)
#### agna_aux.cpp — 인스트럭션 페처
- `agna_aux()` (L3-15): `m_axi` 버스트(`max_read_burst_length=7`, L7)로 외부 메모리에서 `node_num*INSTR_PER_NODE`개의 128bit 인스트럭션을 읽어 `core_instr` FIFO로 스트림. → DDR의 인스트럭션 영역을 코어로 공급하는 DMA 성격.

#### core_controller.cpp — ★인스트럭션 디코더 & 서브유닛 명령 생성기
- `decode_core_instr()` (L3-226): 노드당 7워드를 읽어 `node_param_t`로 복원(`#pragma HLS pipeline II=1`, L16). type별 switch로 모든 필드 디코드(L19-105). 이후 `s_*_last`(마지막 타일의 잔여 S, L108-127), `is_dpws`/`use_w`/`l_k_dpws` 등 파생값과 `lc_tile_num`, `pqf_ih/iw`(L150-155)를 계산. 디코드된 `node_param`을 **7개 서브유닛 FIFO로 복제 브로드캐스트**(res/mm2s/s2mm/pe_updt/pe_exec/pe_wb/lc; L158-164).
- `gen_mm2s_instr()` (L348-514): 입력 활성/가중치/잔차(res)를 타일 순서(tk→th→tw→tc, sc/sh/sw)로 순회하며 각 타일에 대한 **메모리 타일 인스트럭션**(d_type=act/weight/res, eos/eol/sel/tsk..sw)을 생성. AXI DataMover가 사용할 read 명령의 좌표/순서를 결정 — **루프 타일링을 메모리 접근 순서로 구체화**.
- `gen_s2mm_instr()` (L516-599): 출력 활성을 (tk_dpws,th,tw) 타일로 write-back하는 명령 생성.
- `gen_pe_idx_instr()` (L601-635): PE별 공간 인덱스(sk/sc/sh/sw)를 순차 발급(`pe_idle` 처리 포함) — **각 PE에 어느 공간 좌표를 맡길지 결정**.
- `gen_pe_exec_instr()`/`gen_pe_wb_instr()` (L713-831): 실행/라이트백 단계의 타일 루프 인스트럭션(eol/eos, tc==0 reset 등).
- `gen_lc_instr()` (L832-919): layout converter 인스트럭션(타일 좌표 + f_k/f_h/f_w).
- `core_controller()` (L935-958): `#pragma HLS dataflow`(L944)로 위 함수들을 **병렬 파이프라인**으로 묶고, `node_sync()`(L921-933)로 노드 완료 핸드셰이크.

### 3.7 hardware/rtl/ — 메인 데이터패스 (SystemVerilog)
#### cu.sv — ★Compute Unit (DSP48E2 cascade MAC)
- 두 모드: **`HW_DSP_CASCADED`**(기본; def에서 정의)와 parallel(`else`). cascade 모드는 DSP48E2를 **PCIN/PCOUT 체인으로 직렬 연결**해 누산기를 DSP 내부에 흡수.
- `dsp_a_in/b_in/d_in` 구성(L104-128): a=activation을 18bit 시프트(`<<18`), b=weight(부호확장 18bit), d=activation(27bit). → **DSP48E2의 pre-adder/AD 경로로 INT8 2-MAC packing**을 노린 배선(파라미터 `AMULTSEL("AD")`, `PREADDINSEL("A")`, L294, `INMODE=10101`, L369). 두 결과를 `dsp_p_out[33:18]`(상위, 반올림 보정 L412-418)과 `[15:0]`(하위, L419-423)로 분리 — **1 DSP = 2 MAC**.
- DSP 인스턴스화(L288-667): `USE_SIMD("ONE48")`, `MREG=1/PREG=1` 등 풀 파이프라인. cascade 체인의 첫 DSP는 `PCIN=0`, 중간은 이전 DSP의 `PCOUT`을 받음(L224-228), 마지막에서 `cascade_out`을 뽑음(L229-235).
- 누산/잔차 (L240-273): `acc_out`은 cascade_out 누적(`adder_rst_en`으로 리셋, L243-249), `fout_exec_adder`는 누적값+psum 읽기값(잔차/부분합 합산, L262-268)을 psum 버퍼로 write.
- parallel 모드(L676~): cascade 없이 각 DSP 출력을 `adder_tree`로 합산(L748-757).

#### adder_tree.sv — 재귀 파이프라인 가산기 트리
- `adder_tree` (L5-135): `INPUT_NUM`=`PE_NUM`을 절반씩 재귀 분할(`PART_A_NUM/PART_B_NUM`, L17-18)하여 `$clog2(INPUT_NUM)` 스테이지의 파이프라인 가산기 트리 구성. leaf(`INPUT_NUM==1`)는 지연정렬용 `delay_rf`로 스테이지 수를 맞춤(L22-40) — **모든 경로 지연 균형**.

#### pe.sv — 단일 PE (PARAM_A_* 만큼의 버퍼/RF/CU)
- updt/exec/wb 3-phase 설정 레지스터(L108-179), 태그 디코드로 abuf/wbuf 선택(L196-220), **더블버퍼**(`abuf_updt_sel`/`buf_exec_sel=~updt_sel`, L427-437) — **업데이트와 실행을 핑퐁**. activation buffer는 `new_buf_sdp`를 `A_C`개(L564-589), weight buffer는 `A_K*A_C*A_I*A_J`개(L605-647) 생성. psum buffer는 `new_buf_tdp`(true dual-port)로 `A_K*A_H*A_W` 또는 `A_H*A_W`(`PARAM_PBUF_A_K_ONE`)개(L688-785). 패딩은 `if_padding`으로 0 주입(L552-560). `w_fixed_r`(pooling) 시 weight를 고정상수 `8'b00001000`로 대체(L640) — **maxpool/avgpool을 MAC 경로로 통일**. 끝에서 `cu`를 1개 인스턴스화(L788-804).

#### pe_array.sv — PE 어레이 + 명령 디스패치/데이터 라우팅
- 3개 instr FIFO(updt/exec/wb, 각 `fifo_axis depth=16`, L101-396)로 코어 인스트럭션을 받아 디코드. exec 인스트럭션은 param_type(000~111)별로 F/Q/P/PQ/PQF loop factor, stride/pad, layer size, comp_type/w_fixed/depth_wise를 추출(L246-336). **4개 updt 데이터 FIFO(`depth=1024`, L777-795)**로 들어오는 activation/weight를 `axis_dwidth_converter`로 폭변환(L797-810) 후 PE로 분배. `pe_updt_sel`은 `almost_empty` 기반 라운드로빈 선택(L750-772) — **다중 PE 공급 밸런싱**. tag의 eos/eol/eom 상태기계(L854-941)로 슬롯 핑퐁.

#### agna_core.sv — 코어 탑 (AXI4 master + AXIS)
- 포트(L19-67): `ap_start/done/idle/ready`, AXI4 master(core 메모리 접근), `s_axis_instr`(인스트럭션 입력), `instr_num`. 내부에서 mm2s/s2mm/updt/exec/wb/lc/res 인스트럭션 스트림(L74-107)으로 분배 — core_controller(HLS) 출력과 RTL 데이터패스를 잇는 연결 허브.

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 소프트웨어 파이프라인 (오프라인)
```
model.json + platform.json
  → ModelParser/LayerParser → ModelSpec(노드 6축 loop_bounds)
  → ArchSearch(MIGP: GPKit완화 → SCIP정수) → arch_search.json (pe_arch, pe_num, buf_arch ...)
  → OperationSchedule(노드별 MIGP) → op_schedule_<node>.json (T/S/P/Q/F 타일팩터)
  → InstrGen → 노드당 7×128bit 인스트럭션 (DDR에 적재)
  → generate_param_header → param.h / param.sv (HLS·RTL 동시)
```
- **양자화/데이터타입**: data_width 8 또는 16(plltfm spec). INT8 시 `pack_dsp`로 DSP48E2 1개에 2 MAC(arch_spec.get_packed_dsp_num_expr L112-115, cu.sv의 AD/pre-adder 경로). 누산은 16bit signed(`dsp_p_out_sep[15:0]`, cu.sv L84).

### 4.2 하드웨어 실행 (온라인, 노드 단위 반복)
```
agna_aux(m_axi) → core_instr FIFO
  → core_controller(decode → mm2s/s2mm/pe_updt/pe_exec/pe_wb/lc/res 명령, HLS dataflow)
  → [mm2s_ctrl/cmd_gen/tag_match → axi_datamover] DDR read (act/weight/res)
  → pe_array (4-way updt FIFO → dwidth_conv → PE별 abuf/wbuf 더블버퍼 업데이트)
  → pe.cu (DSP48E2 cascade MAC: act×weight 누산 → acc_out → +psum)
  → bn_unit/acti_unit/res_adder (배치정규화·활성·잔차)
  → layout_converter → s2mm → DDR write (act_out)
  → node_sync 핸드셰이크 → 다음 노드
```
- **메모리 계층**: DDR ↔ (abuf=activation, wbuf=weight, pbuf=psum) BRAM 버퍼 ↔ RF(register file) ↔ DSP. 각 버퍼는 updt/exec 더블버퍼.
- **병렬화**: 6축 공간 언롤(`A_K,A_C,A_I,A_J,A_H,A_W`)이 DSP 어레이 크기, `pe_num`이 PE 복제수. `A_H*A_W<=32`로 공간 언롤 제한(arch_search L96).
- **dataflow**: core_controller·HLS 커널은 `#pragma HLS dataflow`(L944), RTL은 updt→exec→wb 3단 핑퐁 파이프.

---

## 5. HW/SW 매핑

| 소프트웨어(Python) | → | 하드웨어(RTL/HLS) |
|---|---|---|
| `ArchSpec.pe_arch (A_K..A_W)` | → | `param.sv` `HW_CONFIG_A_*` → cu.sv DSP 어레이 차원, pe.sv 버퍼 개수 |
| `ArchSpec.pe_num` | → | `HW_CONFIG_PE_NUM` → pe_array의 PE 복제수, adder_tree 입력수 |
| `ArchSpec.data_width` | → | `HW_CONFIG_MACC_WIDTH` → DSP MAC 폭, INT8 packing 여부 |
| `ScheduleParam (T/S/P/Q/F)` | → | InstrGen → 128bit 인스트럭션 → pe_array loop factor 레지스터 |
| `InstrGen.build_single_node()` | → | core_controller `decode_core_instr` (7워드 → node_param_t) |
| `PlatformSpec.dbus_width` | → | AXI DataMover 폭, abuf/wbuf 대역폭 환산 |

→ **DSE 결과(연속) = 컴파일타임 RTL 파라미터, 스케줄(이산) = 런타임 인스트럭션**으로 명확히 분리. 동일 비트스트림으로 다른 레이어를 인스트럭션만 바꿔 처리(추정: agna_aux가 노드별 인스트럭션을 순차 페치).

---

## 6. 빌드·실행
- **소프트웨어**: `cd software; export SCIPOPTDIR=...; conda env create -f environment.yml; make all PLATFORM=<board> MODEL=<model>` (README L52-63, Makefile L8-12 = `run_schedule.py` + `generate_param_header.py`). 산출물: `software/results/<plat>-<model>/`.
- **하드웨어**: `cd hardware; make all` → `hardware/prj`에 프로젝트/비트스트림(README L65-71). Vivado 2021.2.
- **의존 빌드(scratch)**: Ipopt 3.14.10 + SCIPOpt 8.0.3(`-DTPI=omp`) 직접 빌드 안내(README L73-111). Docker 이미지(`yuhaoding/agna:latest`) 제공(Dockerfile).

## 7. 의존성
- Python 3.9, SCIP 8.0.3(+Ipopt, OMP), GPKit, pyscipopt, numpy (environment.yml/README). HLS: Vitis HLS(.tcl), RTL: Vivado 2021.2 + Xilinx IP(axi_datamover, axis_interconnect — 제외). DSP48E2 프리미티브 직접 인스턴스화(cu.sv) → **Xilinx UltraScale+ 종속**.

## 8. 강점 / 한계 / 리스크
- **강점**: (1) MIGP(GP완화→정수)로 아키텍처+스케줄을 **수학적 최적화**, GP완화로 정수탐색 가속(base_rnr `append_strict_bound`). (2) 하나의 DSE 결과가 RTL·HLS 파라미터와 레이어 인스트럭션을 **자동 동기 생성**. (3) DSP48E2 cascade + INT8 2-MAC packing으로 DSP 효율 극대화. (4) 다중 모델 동시 최적화 지원(obj가 모델별 곱, arch_search L278-291).
- **한계/리스크**: (1) **CNN 전용**(loop bound 6축 K/C/I/J/H/W = conv 의미론; depthwise/maxpool 특수처리만, Transformer/attention 직접 지원 없음 — 확인됨). (2) SCIP 외부 실행파일·Ipopt 빌드 의존으로 환경 구축 난도 높음. (3) DSP48E2 하드코딩 → 다른 벤더 이식 불가. (4) `*0.9`, `*1.5` 등 휴리스틱 여유율이 보드별 튜닝 필요(추정). (5) Transformer의 LayerNorm/Softmax/GELU 비선형 연산 데이터패스 없음(확인 불가 — RTL에 해당 유닛 부재).

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA + XR 시선추적) 관점 시사점
- **자동 코드생성/DSE 재사용(최우선)**: `generate_param_header.py`의 `.tpl` 치환 + `InstrGen`의 노드→ISA 인코딩 패턴은 **HG-PIPE류 ViT 가속기의 파라미터/스케줄 자동화 백본**으로 직접 차용 가능. 6축 loop를 attention(B/Head/N/N/D) 축으로 재정의하면 GEMM/MHSA 타일링 DSE로 확장 가능(추정).
- **MIGP 2단계 최적화**: `BaseRnR`(GP완화→정수반올림) 구조는 ViT의 패치임베딩·QKV·FFN 각 GEMM에 대한 타일/언롤 팩터를 자원제약(DSP·BRAM 90%) 하에서 최적화하는 데 그대로 적용 가능. 우리 양자화(INT8/Mamba) 시 `pack_dsp` 모델을 비트폭별로 일반화할 수 있음.
- **systolic/GEMM 매핑**: cu.sv의 **DSP48E2 cascade(PCOUT→PCIN) 누산 체인**은 systolic MAC 컬럼의 정석 구현 — ViT GEMM의 output-stationary 어레이로 재사용. INT8 2-MAC packing(AD/pre-adder)은 처리량 2배 확보 포인트.
- **인스트럭션 기반 재구성**: "같은 비트스트림 + 노드별 인스트럭션"은 ViT의 가변 레이어(다양한 N, D)를 단일 비트스트림으로 처리하려는 우리 목표와 정확히 부합. core_controller의 `dataflow` 서브유닛 명령 생성 패턴 채용 권장.
- **한계 인지**: AGNA에는 Transformer 비선형(Softmax/LayerNorm/GELU) 유닛이 없으므로, **이 부분은 별도 설계 필요**(Transformer-Accel/ViT-Accelerator repo에서 보강).

## 10. 근거 표기
- 라인 근거는 본문 (파일 Lxx) 형식으로 표기. README/Dockerfile/소스 직접 인용.
- **"추정"**: 파일명 규칙(`board_dbus_data`), `agna_level_tuple`의 정확한 5계층 의미(T/S/P/Q/F는 코드 사용으로 확인되나 각 계층의 물리적 의미는 일부 추정), 비트스트림 1개로 다중 노드 처리(agna_aux 동작상 합리적 추정), 보드별 휴리스틱 튜닝.
- **"확인 불가"**: Transformer 비선형 연산 데이터패스 부재(RTL 파일 목록상 미발견 → 부재로 판단하나, 일부 유닛이 다른 이름일 가능성 배제 못함), `software/results`·`hardware/prj` 산출물 내용(생성물·제외).
- third-party(Xilinx IP, SCIP/Ipopt/GPKit), `.git`, 비트스트림은 이름만 언급하고 분석 제외.
