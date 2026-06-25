# REF 코드베이스·논문 정밀 분석 — 통합 INDEX

> 분석 대상: `REF/` 디렉토리의 모든 **자체 코드베이스**와 **논문 PDF**
> 분석 결과 위치: `REF/Analysis/` (카테고리별 폴더 + 항목별 `.md`)
> 작성: 2026-06 · 언어: 한국어 · 형식: Markdown

---

## 0. 분석 개요 / 방법론

- **목적**: `REF/`에 모아둔 ViT/Transformer/LLM FPGA·ASIC 가속기, ViT 양자화, CNN 가속기, 희소연산 가속기, XR 이벤트 시선추적 코드베이스와 관련 논문을 라인 단위로 정밀 분석하여, 본 프로젝트(추정: **고처리량 ViT/Transformer FPGA 가속기 "HG-PIPE 계열" + XR 이벤트 시선추적**)에서 재사용 가능한 설계·기법·코드를 식별.
- **분석 방식**: 파일 도구(Glob/Grep/Read) 기반 정적 분석. 각 repo의 자체(core) 소스를 함수/클래스/HLS 커널/RTL·SpinalHDL 모듈 단위로 읽고, 파일·함수·라인 근거를 들어 기술.
- **제외 정책(third-party)**: `third_party`, `3rdparty`, `vendor`, `.ip_user_files`, `.Xil`, `node_modules`, `googletest`, `fpnew`, `ramulator`, `softfloat`, `openc910/906`, 외부 프레임워크 원본(TVM/mmdet/mmcv/timm/DeiT 원본 등), 빌드 산출물(`.bit/.xclbin/.xo/.pdi`), 대용량 가중치·데이터는 분석에서 제외하고 이름만 언급.
- **표기 규칙**: 근거가 불확실하면 "추정", 소스/문헌 부재로 확인 못 한 부분은 "확인 불가"로 명시. 각 `.md` 말미에 한계 표기.
- **제약 메모**: 이번 세션에서 sandbox shell(bash)은 UNC 경로 문제로 사용 불가하여, 모든 분석은 파일 도구로 수행됨(합성/시뮬레이션 실행은 미수행 → 정량 PPA는 동봉 리포트가 있을 때만 인용).

---

## 1. 전체 통계

| 카테고리 | 분석 파일 수 | 폴더 |
|---|---|---|
| ViT-Accelerator | 21 | `Analysis/ViT-Accelerator/` |
| ViT-Quantization | 48 | `Analysis/ViT-Quantization/` |
| Transformer-Accel | 32 (고유 24 + 중복 cross-ref 8) | `Analysis/Transformer-Accel/` |
| CNN-Accel | 18 | `Analysis/CNN-Accel/` |
| Others | 11 | `Analysis/Others/` |
| XR-Eye-Tracking (Codebase) | 17 | `Analysis/XR-Eye-Tracking/Codebase/` |
| Papers (논문) | 22 | `Analysis/Papers/` |
| **합계** | **약 169** | |

> 참고: ViT-Accelerator와 Transformer-Accel은 동일 repo가 양쪽에 물리적으로 존재하는 경우가 많아, Transformer-Accel 쪽은 8건을 짧은 **교차참조(cross-ref)**로 처리(상세 분석은 ViT-Accelerator 쪽 문서).

---

## 2. 카테고리별 카탈로그

### 2.1 ViT-Accelerator (`Analysis/ViT-Accelerator/`)

| 파일 | 한 줄 요약 |
|---|---|
| hls-fpga-accelerators.md | GEMM/elementwise/unary/RMSNorm/softmax를 데이터타입·버스폭 전면 파라미터화한 재사용 Vitis HLS 연산자 라이브러리(LUT exp 근사). U250/Kria K26 |
| HLS-Acceleration-of-LLaMA2.md | llama2.c(stories15M) forward 전체를 단일 HLS 커널로, ZCU106 PS 대비 ~5× (수치안정 softmax, partial 누산 분할) |
| TinyTransformer.md | RISC-V SoC AI 가속기용 PyTorch 골든모델(멀티모달 ViT). QKV 융합·MMU 재사용·LUT softmax·InvSqrt LayerNorm 주석화 |
| Transformer-Accelerator-Based-on-FPGA.md | PYNQ-Z1 INT8 weight-stationary systolic GEMM(완성) + 미완 Softmax/GELU/LayerNorm RTL. A_SIZE 불일치 리스크 |
| AURA-FlashAttention-AISC-Accelerator.md | FlashAttention SystemVerilog RTL ASIC — INT8 + 곱셈기 없는 exp 근사(ExpMul) + 1-pass online softmax + 트리리덕션, Synopsys DC |
| transformer-hls-thesis.md | Opus-MT(MarianMT) 인코더 레이어를 ap_fixed 양자화로 ZCU104 HLS + COMET/BLEU 평가까지 닫은 석사 프로젝트 |
| HLSTransformation.md | int8(GS=64) 양자화 TinyLlama2 forward 단일 m_axi HLS 커널, AWS F1(VU9P) + CPU/GPU 전력·지연 벤치 (matmul 미최적화) |
| llama-fpga.md | SpinalHDL FP16 데이터패스 + AWQ W4A16/KV INT8 + RMSNorm/RoPE/Safe-Softmax 생성, 코어 복제·링연결 EdgeLLM (KV260/ZCU104/U250) |
| LLM_FPGA.md | HeteroCL/Allo 생성 Vitis HLS W8A8 Transformer(BERT/GPT/LLaMA) — int8 2-pack GEMM + int32 누산 + per-token requant, 3-SLR, U280 HBM |
| lut-gemm.md | sub-4bit BCQ/RTN 양자 LLM 가중치를 디퀀트 없이 부분합 LUT 조회로 곱하는 NVIDIA GPU W4A16 GEMV CUDA 라이브러리 (NAVER) |
| flightllm_test_demo.md | 6종 ISA 오버레이 + 세마포어 + 의존성 사이클 시뮬레이터로 LLaMA2-7B를 VHK158/U280에 매핑(RTL 비공개, Python 프로파일러·ISA만) |
| ternaryLLM.md | 삼진({-1,0,+1}) 가중치 LLM을 CPU(TCSC+AVX)/GPU(CUDA SpMM)/FPGA(SpinalHDL, Coyote U55C)에 사상. 곱셈→인덱싱+가감산 |
| efficient-transformer-accelerator.md | INT8 32×16 systolic + 시분할 공유 양자화기 SystemVerilog(FPGA/ASIC 듀얼), Taylor degree-2 linear attention(ViTALiTy 계열) |
| FlexLLM.md | SpinQuant(INT4 linear/INT8 MHA + Hadamard) LLaMA-3.2-1B를 composable HLS(TAPA)+RapidStream으로 U280. prefill 2D/decode 1D |
| Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280.md | Vortex(RISC-V GPGPU) OpenCL 커널. 실체는 2-layer MLP toy(어텐션·LN·양자화 없음) — "GPT"는 명목상 |
| TATAA.md | DSP48E2 4×16 PE를 mode_sel로 INT8 GEMM ↔ bfloat16 비선형으로 변형(transformable arithmetic). 직접작성 SystemVerilog, U280 |
| ViT-FPGA-TPU.md | VC707 + PCIe/XDMA 16×16 FP16 output-stationary systolic GEMM + C++/Python 호스트 (ViT 전체 그래프 통합은 미확인) |
| TeraFly.md | LoopLynx(DATE'25) 확장. codegen.py가 toml→HLS 자동생성, weight_packer INT8 패킹, pyxrt 호스트 + 웹 데모 (RTL 없음) |
| Trio-ViT.md | Softmax-free EfficientViT(ReLU linear attention)에 BRECQ/AdaRound PTQ. 순수 PyTorch(RTL/HLS 없음) |
| ViTCoD.md | ViT 어텐션맵 denser/sparser 분할 + 이중엔진 + 학습형 Q/K auto-encoder 압축(HPCA'23). HW는 Python 사이클 시뮬레이터 |
| LUT-LLM.md | Qwen3 1.7B를 V80/VPK180에. 선형층 곱셈을 활성 VQ→2D-LUT 조회→정수덧셈으로 치환(런타임 곱셈 0). TAPA+RapidStream |

### 2.2 ViT-Quantization (`Analysis/ViT-Quantization/`)

> ViT/LLM/VLM/Diffusion 양자화 알고리즘 모음(대부분 PyTorch). 아래 요약은 압축본 — 상세 수식·코드 근거는 각 `.md` 참조. 일부 방법명 해석은 "추정".

| 파일 | 한 줄 요약 |
|---|---|
| FQ-ViT.md | 완전양자화 ViT: LayerNorm Power-of-Two Factor + Log-Int-Softmax (정수 전용) |
| PTQ4ViT.md | ViT PTQ: twin uniform quantization + Hessian-guided scale 탐색 |
| RepQ-ViT.md | scale reparameterization PTQ(채널별 LayerNorm/log2 softmax → 레이어별 등가 변환) |
| I-ViT.md | integer-only ViT: ShiftMax, ShiftGELU, dyadic requant (정수·시프트 전용) |
| Q-ViT-DeiT.md | Q-ViT: 미분가능 양자화 + head-wise 비트폭 (DeiT 백본) |
| psaq-vit.md | data-free PTQ: patch similarity aware 합성샘플 생성 |
| integer-only-transformer.md | I-BERT 계열 정수전용 추론(GELU/Softmax/LayerNorm 정수근사) |
| Bi-ViT.md | 이진(binary) ViT |
| NoisyQuant.md | PTQ에 고정 noisy bias 추가로 활성 분포 정합 |
| outlier-free-transformers.md | clipped softmax + gated attention으로 outlier 억제(양자화 친화) |
| AdaLog.md | adaptive logarithm base 양자화(ViT PTQ) |
| FIMA-Q.md | Fisher Information Matrix 근사 기반 PTQ |
| OFQ.md | oscillation-free 저비트 양자화 학습 |
| transformer-quantization.md | Qualcomm 트랜스포머 양자화(outlier suppression 등) 베이스라인 |
| APHQ-ViT.md | average perturbation Hessian 기반 ViT PTQ |
| UQ-ViT.md | ViT 양자화(통합/불확실성 기반 — 상세는 .md, 해석 일부 추정) |
| Q-DETR.md | 양자화 DETR 검출기(정보병목 기반 distillation) |
| INT8-Flash-Attention-FMHA-Quantization.md | INT8 fused multi-head attention 양자화 |
| int-flashattention.md | INT8 FlashAttention 구현 |
| AdaTSQ.md | adaptive 스케일/2-scale 양자화(상세는 .md, 일부 추정) |
| NoisyQuant/AdaLog/… | (위 항목 참조) |
| Castling-ViT.md | 학습은 linear-angular attention, 추론에 sparse softmax 보강한 효율 ViT |
| RepViT.md | MobileNet식 설계 재검토로 만든 경량 CNN-스타일 ViT 백본 |
| Next-ViT.md | 배포지향 CNN-Transformer 하이브리드 백본 |
| EdgeVisionTransformer.md | 에지 ViT 배포/프루닝(deit_pruning 등) |
| ShiftAddViT.md | 곱셈을 shift&add로 reparameterize한 곱셈기-경감 ViT |
| P2-ViT.md | power-of-two PTQ + 전용 가속기 지향 ViT |
| AMD_QTViT.md | AMD(Brevitas 계열) 양자화 ViT |
| q-hyvit.md | Q-HyViT: 하이브리드 ViT PTQ(Hessian 기반 hybrid reconstruction, bridge zero-point overflow 대응) |
| qflash.md | 양자화 FlashAttention(INT) |
| mimiq.md | mixed-precision 양자화(상세는 .md, 일부 추정) |
| M3ViT.md | MoE ViT + 하드웨어 공동설계(태스크별 expert 게이팅) |
| MBQ.md | modality-balanced 양자화(VLM 대상) |
| qvlm.md | 비전-언어모델 양자화(custom_bitsandbytes 활용) |
| QuantVLA.md | 비전-언어-행동(VLA) 모델 양자화(gr00t 기반) |
| Q-DiT.md | diffusion transformer(DiT) 양자화 |
| Q-VDiT.md | 비디오 diffusion transformer 양자화 |
| ViDiT-Q.md | 비디오/이미지 DiT 양자화 (smooth/dynamic quant) |
| ptq4sam.md | Segment Anything Model PTQ |
| SAQ-SAM.md | sharpness-aware SAM 양자화(추정) |
| PQV-Mobile.md | 모바일 ViT 대상 PTQ 도구 |
| Mix-Quant.md | 혼합정밀 양자화 비트폭 탐색 |
| CLAMP-ViT.md | contrastive-learning 보조 ViT PTQ |
| FPQVAR.md | 비주얼 autoregressive 부동소수 양자화(추정) |
| Quantformer.md | 양자화 인지 트랜스포머(추정) |
| AHCPTQ.md | 하드웨어 친화 PTQ(추정) |
| mixed-non-linear-quantization.md | softmax/GELU/LayerNorm 등 비선형 함수 혼합 양자화 |
| postcalibration4quantization.md | 사후 보정(post-calibration) 양자화 기법(추정) |
| FIMA-Q.md / OFQ.md / … | (상기) |

### 2.3 Transformer-Accel (`Analysis/Transformer-Accel/`)

**고유(전체 심층):**

| 파일 | 한 줄 요약 |
|---|---|
| HG-PIPE.md | ★ 고처리량 ViT FPGA 가속기(본 프로젝트 기반 추정). 전계층 파이프라인/데이지체인 구조 |
| trans-fat.md | I-BERT식 INT8 BERT/RoBERTa 인코더 1레이어를 U200 2장 4-stage 분할 Vitis HLS + 비트정확 골든 |
| ViT-Accelerator.md | ViT-B 단일-head attention을 U50용 16×16 타일 HLS 커널로(교육용; 호스트 본체·멀티헤드·×V 미구현) |
| MobileVit-AI-Hardware-Accelerator.md | MobileViT용 재구성형 weight-stationary 16×16 systolic 4타일 + AGU(im2col) + LayerNorm/Swish RTL (top 통합 결함) |
| Lightening-Transformer-AE.md | HPCA'24 DOTA 광학(ONN) Transformer 가속기 AE — 양자화+광학노이즈 SW 모델 + MZI/MRR 시뮬레이터 |
| FPGA_Friendly_SpinQuant.md | LLaMA-3.2-1B를 SpinQuant(Hadamard R4+W4A4) 후 TAPA HLS로 U280에 prefill+decode+on-FPGA Top-K end-to-end |
| vit-tiny-accelerator.md | TinyViT-5M Zynq-7000 INT8, 8×8 systolic GEMM 공유 + 단일 scheduler FSM (통합 미완, 문법오류 발견) |
| ViT-Accelerator-on-FPGA-with-INT8-quantization.md | DeiT-Tiny급 ViT INT8 비대칭 + streaming softmax Vitis HLS 단일 데이터패스(레이어 시분할) |
| Edge-MoE.md | 멀티태스크 ViT용 ICCAD'23 가속기 — 단일 HLS에 dense-MLP/Top-2 MoE, 선택 expert만 ping/pong prefetch |
| Diff-DiT.md | Vitis HLS DiT 가속기, INT8/INT3 차분경로 13-mode, DSP 2/6-pack, 16×16 OS systolic×32 |
| Transformer_dataflow.md | Vitis HLS C++ Transformer 인코더 템플릿, QKᵀ·softmax 행단위 융합 커널(double 골든) |
| Transformer-Accelerator-Based-on-FPGA.md | (ViT-Accel과 동일 계열) PYNQ-Z1 INT8 systolic GEMM + 미완 비선형 RTL |
| transformer-hls-thesis.md | (동일 계열) opus-mt 인코더 ap_fixed HLS + COMET/BLEU |
| HLS-Acceleration-of-LLaMA2.md | (동일 계열) llama2 FP32 단일 HLS 베이스라인 |
| LLM_FPGA.md | W8A8 Transformer 5종 U280 3-SLR(HLS 소스 부재 → 리포트 역추정) |
| LUT-LLM.md | LUT/VQ INT4 Qwen 디코더 V80(HLS 소스 부재 → 빌드/리포트 역공학) |
| TeraFly.md | HeteroCL/Allo OPT-1.3B 2-SLR 스트림 링(HLS 소스 부재 → 토폴로지 역추론) |
| ternaryLLM.md | 1.58bit 삼진 LLM GEMM 곱셈기 0개(부분 체크아웃) |
| TinyTransformer.md | 사실상 빈 repo(.gitignore만) → 분석 불가, 추정만 |
| Transformer_dataflow.md | (상기) |
| flightllm_test_demo.md | 자체 소스 0건, 사전컴파일 번들 → 디렉토리/바이너리 규약으로 3-SLR/8채널 HBM 역도출 |
| submission.md | Karpathy llama2.c(int8 GS=64) 단일 HLS forward, AWS F1(VU9P) — HLSTransformation 형제 |
| ViTCoD.md | (ViT-Accel과 동일) Python 사이클 시뮬레이터 |
| ViTALiTy.md | HPCA'23 linear Taylor attention — 본 repo는 알고리즘(DeiT 포크)만, HW 부재 |

**중복 cross-ref(상세는 `ViT-Accelerator/` 동명 문서):** AURA-FlashAttention-AISC-Accelerator, hls-fpga-accelerators, HLSTransformation, TATAA, Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280, FlexLLM, efficient-transformer-accelerator, ViT-FPGA-TPU (대부분 byte-동일/스냅샷 차이; ViT-FPGA-TPU는 본 사본에 컨트롤러 RTL 추가본).

### 2.4 CNN-Accel (`Analysis/CNN-Accel/`)

| 파일 | 한 줄 요약 |
|---|---|
| XJTU-Tripler-master.md | HPU 코어(자체 RTL/HLS) CNN 가속기, DAC'19 |
| tataa_tvm_dev-main.md | TATAA용 TVM 포크 — 자체 추가 pass/apps만 분석(TVM 코어는 외부) |
| FlexCNN.md | 자동 컴파일 기반 CNN HLS 가속기 생성(auto_compile) |
| Kria-YOLOv4-Tiny-FPGA-Accelerator.md | 단일 3×3 conv 오프로딩 학부형 INT8(비-systolic, `>>8` 고정시프트) — 안티패턴 대조군 |
| yolo-fpga-accelerator-main.md | 모델 비종속 타일 엔진(Tm32×Tn4=128 MAC, OS) + 호스트 레이어 스케줄러, per-layer Q 정렬 |
| Uint-Packing-master.md | UltraNet 4w4a DSP 패킹(DSP2/DSPopt3로 1 DSP 다중 저비트 MAC) + 전계층 DATAFLOW |
| ESDA.md | 이벤트 sparse CNN(MobileNetV2) 토큰+마스크 dataflow(ZCU102), SCIP ILP DSE + codegen |
| SEE.md | ESDA 포함 사례연구 래퍼 — board/ ZCU102 PYNQ 실측(지연+INA226 전력) 하네스가 고유가치 |
| AnyPackingNet-main.md | 실체는 DeepBurning-MixQ — DSP packing 효율을 미분가능 NAS loss에 결합한 mixed-precision 탐색 |
| yolov2_xilinx_fpga-flex.md | YOLOv2 Zynq(INT16/FT32 단일 HLS IP, Tm×Tn MAC, II=1, ping-pong) |
| dac_sdc_2020_designs-master.md | DAC-SDC'20 3팀 비교(BJUT UltraNet, SkrSkr SkyNet, iSmart 16-MAC 트리), xczu3eg |
| dac_sdc_2021_designs-main.md | DAC-SDC'21 3팀(SJTU 2D PE array, SkrSkr im2col DW/PW, iSmart DSPopt 패킹) |
| dac_sdc_2022_champion-master.md | DAC-SDC'22 우승 UltraNet, INT-Packing(1 DSP=4×4bit) + 전계층 dataflow |
| dac_sdc_2022_designs-main.md | DAC-SDC'22 설계 모음(SEUer, InvolutionNet 등) |
| dac_sdc_2023_champion-main.md | DAC-SDC'23 우승 UltraSpeed(KV260), UINT-Packing |
| SJTU_microe-main.md | SJTU UltraNet+Bypass, 명시적 2D PE array + 4MUL/DSP + im2col형 — ViT GEMM 매핑성 우수 |
| SkrSkr-master.md | ShanghaiTech SkyNet 재구현(W6A8), DW+PW tiled 직접 컨볼루션, fully-integer |
| SkyNet-ZCU104-master.md | SkyNet depthwise-separable 검출망 ZCU104, conv1x1 dot-tree 병렬 + 오프라인 weight reorder |

### 2.5 Others (`Analysis/Others/`)

| 파일 | 한 줄 요약 |
|---|---|
| acap-gemm-sa.md | Versal VCK5000(8×50 AIE) FP32 GEMM — OS systolic AIE 커널 + PL HLS 데이터무버 + AMPL/Gurobi DSE |
| TMMA.md | KV260 INT8 타일드 GEMM(32×32 OS MAC, DSP 1024) — DistilBERT QKV 프로젝션 가속 HLS |
| HiSpMV.md | TAPA/HBM FP32 SpMV(+dense GeMV) 불균형행 흡수(Pre-Accumulator·RDN), automation_tool DSE+codegen |
| HiSpMM.md | HiSpMV를 N0=8 SIMD로 확장한 Sparse-Dense MatMul, balanced/imbalanced 자동선택 (TRETS'25) |
| hls-spmv.md | NTU 교육용 CSR SpMV, naive→fast-stream(II=1) 5단계 최적화 csynth 비교 |
| HiSparse.md | FPGA'22(Cornell) HBM SpMV, ap_ufixed, 128-way, 2단 셔플 + URAM forwarding PE + CPSR |
| DPACS.md | ASPLOS'23 입력적응형 spatial+channel 동적 프루닝 공동설계, coordinate-stream sparse dataflow |
| MSD-FCCM23.md | FCCM'23 부호자리(EB/CSD) 제한 양자화 + LUT bit-serial & DSP 어레이 이종자원 동시활용 |
| AGNA-FCCM2023.md | MIGP(GPKit→SCIP) DSE + RTL/HLS 파라미터·128bit 인스트럭션 자동 codegen, DSP48E2 cascade INT8 2-MAC |
| REMOT-FPGA-22.md | FPGA'22 병렬 Attention Unit이 DVS 이벤트 스트림을 AMAP query/update로 처리하는 이벤트구동 MOT HLS |
| ViM-Q-FCCM-2026.md | ★ FCCM'26 Vision Mamba APoT INT4/5 가중치+INT8 활성 PTQ, selective-scan/conv1d를 곱셈기-free shift/LUT HLS + SpinalHDL 통합 |

### 2.6 XR-Eye-Tracking — Codebase (`Analysis/XR-Eye-Tracking/Codebase/`)

| 파일 | 한 줄 요약 |
|---|---|
| EllSeg.md | 동공/홍채/공막 분할 + 타원 파라미터 회귀(RITnet 계열 다수 버전) |
| RITnet.md | DenseNet식 인코더-디코더 실시간 눈 세그멘테이션(프레임 기반) |
| eyegraph.md | 이벤트 그래프 클러스터링 기반 연속 시선추적 |
| gg_ssms.md | 그래프/SSM 기반 시공간 모델(이벤트) |
| EventMamba.md | Mamba(SSM) 기반 이벤트 시공간 모델 |
| cb-convlstm-eyetracking.md | 3ET용 Change-Based ConvLSTM(시간 sparsity 유도), 경량 |
| Swift-Eye.md | timelens 고FPS 이벤트프레임에서 동공 회전bbox 검출+추적, anti-blink FSM (Swin-T+MMRotate) |
| retina.md | config-driven 경량 CNN을 ANN/SNN/이진/양자화/배포로 일원화, SynSense Speck 뉴로모픽 |
| EV-Eye.md | DAVIS346 하이브리드 벤치마크(NeurIPS'23), U-Net 앵커 + 이벤트 ICP 보간(최대 38.4kHz) |
| FACET.md | EV-Eye 기반 통합(TennSt 인과 Conv3d 좌표회귀 + EPNet CenterNet 타원/GWD loss), AIS2024 |
| E-Track.md | 누적 2극성 프레임→U-Net 세그→타원피팅, U-Net↔경량 이벤트-RoI 추적 FSM 절전 (AICAS'23) |
| EX-Gaze.md | 느린 프레임검출 + 빠른 이벤트추적(FusedMBConv+MobileViTv2 선형어텐션) 하이브리드, Jetson TensorRT |
| event_based_gaze_tracking.md | Angelopoulos >10kHz 공식 repo(실제론 데이터셋 배포+시각화; 9B BHHI 이벤트 포맷이 자산) |
| ESDA.md | (CNN-Accel ESDA와 연계) 이벤트 sparse MobileNetV2 HAWQ INT8 ZCU102 layer-pipeline — 본 프로젝트 최직접 연관 |
| EyeLoRiN.md | 모델 없는 추론시점 후처리(median 필터 + ROI 옵티컬플로우 정렬) (CVPR'25 챌린지 2위) |
| ais2024.md | AIS2024 챌린지: 베이스라인 CNN+GRU / ERVT / TENNs-Eye(인과 Conv3d 스트리밍) 비교 |
| ais2025.md | CVPR'25 챌린지: 베이스라인 GRU / 1위 Transformer(ALiBi) / 3위 TDTracker(3D-CNN+BiGRU+Mamba+SimDR) |

### 2.7 Papers (`Analysis/Papers/`)

| 파일 | 한 줄 요약 |
|---|---|
| 3ET.md | Change-Based ConvLSTM으로 연산 4.7× 절감(BioCAS'23), 결론에 Spartus FPGA 가속 언급 |
| EX-Gaze.md | 하이브리드 event-frame 고주파·저지연 시선추적(on-device XR) |
| MambaPupil.md | 양방향 selective recurrent(Mamba) 이벤트 시선추적 |
| TDTracker.md | ITD(3D CNN) + ETD(FFT→GRU→Mamba) 캐스케이드, SEET MSE 1.30px (CVPRW'25 3위) |
| Swift-Eye-Paper.md | anti-blink 동공추적(Timelens 보간 + Swin/FPN + rotated ellipse), offline 지향 |
| BRAT-CVPRW25.md | Bidirectional Relative-position Attention Transformer 이벤트 시선추적(CVPRW'25) |
| Pei-Lightweight-CVPRW24.md | 경량 시공간 네트워크 온라인 이벤트 시선추적(CVPRW'24) |
| Zhang-Submanifold-CVPRW24.md | Submanifold sparse conv 서브밀리초 지연 시선추적 시스템 공동설계(CVPRW'24) |
| FACET.md | 타원 모델링 기반 빠르고 정확한 이벤트 시선추적(XR) |
| FAPNet.md | 이벤트 시선추적 경량 네트워크(FAPNet) |
| GazeRefine-MicroExpr.md | 추론시점 gaze refinement(미세표정 인식, motion-aware 후처리) |
| EV-Eye.md | 고주파 이벤트 시선추적 데이터셋·방법 재고(NeurIPS'23) |
| EyeGraph.md | modularity-aware 시공간 그래프 클러스터링 연속 시선추적 |
| DualPath-EyeTrack.md | 이중경로 강건성+적응형 시간모델링 이벤트 시선추적 |
| E-Track.md | 이벤트 카메라 XR 시선추적(E-Track) |
| RITnet-Paper.md | 실시간 눈 세그멘테이션(RITnet) 원논문 |
| SynUnlabeled-PupilTrack.md | 합성+unlabeled 데이터로 데이터부족 극복(이벤트 동공추적) |
| AdaptiveSSM-EyeFeature.md | Bayesian SSM(동적 α-가중) + dynamic confidence 눈특징 추정 |
| Retina.md | IAF SNN + temporal weighted-sum, Speck 뉴로모픽 실측 2.89–4.8mW (CVPRW'24) |
| NearEye-10000Hz.md | >10kHz 근안 시선추적(하이브리드 frame+event, parametric pupil; 원문 대용량 미판독→교차인용 기반) |
| Q-HyViT.md | 하이브리드 ViT PTQ(integer-only; FPGA off-chip 데이터이동 절감과 직접 부합) |

---

## 3. 교차 주제 분석

### 3.1 양자화 기법 지도
- **정수 전용(integer-only) 비선형**: I-ViT(ShiftMax/ShiftGELU), FQ-ViT(Log-Int-Softmax/PoT-LayerNorm), integer-only-transformer, Q-HyViT, MSD-FCCM23. → FPGA에서 softmax/GELU/LayerNorm을 곱셈기·부동소수 없이 구현하는 직접 레퍼런스.
- **저비트 가중치/곱셈기 제거**: lut-gemm·LUT-LLM(sub-4bit + LUT 곱), ShiftAddViT(shift&add), ternaryLLM(삼진), ViM-Q(APoT shift/LUT). → DSP 절감/곱셈기-free 데이터패스.
- **PTQ 정밀화**: PTQ4ViT, RepQ-ViT, NoisyQuant, AdaLog, APHQ-ViT, FIMA-Q, CLAMP-ViT, Mix-Quant, P2-ViT. → 보정/스케일 재파라미터화.
- **회전/Hadamard 기반(outlier)**: FlexLLM·FPGA_Friendly_SpinQuant(SpinQuant), outlier-free-transformers, transformer-quantization. → 활성 outlier를 HW 친화적으로 흡수.
- **확장 도메인**: VLM(MBQ/qvlm), VLA(QuantVLA), Diffusion/DiT(Q-DiT/Q-VDiT/ViDiT-Q), SAM(ptq4sam/SAQ-SAM), FlashAttention(qflash/int-flashattention/INT8-FMHA).

### 3.2 하드웨어 데이터플로우 패턴
- **Systolic GEMM**: output-stationary(ViT-FPGA-TPU, MobileVit, acap-gemm-sa, SJTU_microe), weight-stationary(Transformer-Accelerator-Based-on-FPGA, MobileVit). 타일 16×16/32×32 다수.
- **DSP packing**: Uint-Packing/AnyPackingNet/AGNA/Diff-DiT/TATAA(1 DSP에 다중 저비트 MAC, INT8 2-pack). → 본 프로젝트 DSP 효율 1순위 차용.
- **Dataflow 파이프라인(전계층)**: HG-PIPE, ESDA/SEE, UltraNet 계열(DAC-SDC), FlightLLM. → 레이어간 스트리밍·온칩 버퍼링.
- **Composable HLS + floorplan**: FlexLLM/FPGA_Friendly_SpinQuant/LUT-LLM/TeraFly(TAPA + RapidStream, 다중 SLR/HBM).
- **변형 연산기(reconfigurable PE)**: TATAA(INT8↔bf16), MobileVit(LEGO), Edge-MoE(dense/MoE), Diff-DiT(13-mode). 

### 3.3 Attention/Softmax/LayerNorm/GELU 하드웨어화
- FlashAttention/online-softmax: AURA(곱셈기 없는 exp + 1-pass), ViT-Accelerator-on-FPGA-INT8(streaming softmax), qflash/int-flashattention, trans-fat.
- 비선형 LUT/근사: hls-fpga-accelerators(LUT exp), MSD/ViM-Q(shift/LUT), I-ViT/FQ-ViT(정수).
- Linear attention(softmax 회피): ViTALiTy/efficient-transformer-accelerator/Trio-ViT(Taylor/ReLU linear attention).

### 3.4 이벤트 기반 시선추적 모델 계열
- ConvLSTM: cb-convlstm/3ET. SSM(Mamba): EventMamba/MambaPupil/TDTracker(부분)/AdaptiveSSM/gg_ssms. Transformer: BRAT/ais2025-1위/EX-Gaze(linear attn). SNN/뉴로모픽: retina. 세그+타원: EllSeg/RITnet/E-Track/FACET. 후처리: EyeLoRiN/GazeRefine.
- **저지연·on-device 후보**: TENNs-Eye(ais2024, 인과 Conv3d 스트리밍), TDTracker-Mamba(ais2025), 3ET(Spartus FPGA), ESDA(이벤트 sparse FPGA), retina(뉴로모픽 mW급).

### 3.5 타깃 디바이스 분포
- AMD/Xilinx FPGA: U280(다수), U250/U200/U55C/U50, V80/VPK180, ZCU102/104/106, KV260, Zynq-7000/PYNQ-Z1/Z2, VC707, Versal VCK5000. ASIC: AURA(Synopsys DC), Lightening-Transformer(광학). 에지/뉴로모픽: Jetson(TensorRT), MAX78000, SynSense Speck.

---

## 4. 코드베이스 ↔ 논문 매핑

| 코드베이스 | 관련 논문(Analysis/Papers) | 비고 |
|---|---|---|
| ViT-Quantization/q-hyvit | Q-HyViT.md | 동봉 논문 = 코드 |
| XR/cb-convlstm-eyetracking, Papers/3ET | 3ET.md | 3ET = CB-ConvLSTM 구현 |
| XR/ais2025, Papers/TDTracker | TDTracker.md | ais2025 3위 솔루션 |
| XR/Swift-Eye, Papers/Swift-Eye-Paper | Swift-Eye-Paper.md | 코드=논문 |
| XR/EV-Eye, Papers/EV-Eye | EV-Eye.md | 데이터셋/방법 |
| XR/FACET, Papers/FACET | FACET.md | |
| XR/E-Track, Papers/E-Track | E-Track.md | |
| XR/EX-Gaze, Papers/EX-Gaze | EX-Gaze.md | |
| XR/retina, Papers/Retina | Retina.md | SNN 뉴로모픽 |
| XR/eyegraph, Papers/EyeGraph | EyeGraph.md | |
| XR/EventMamba·MambaPupil, Papers/MambaPupil | MambaPupil.md | SSM 계열 |
| Transformer-Accel/Lightening-Transformer-AE | (HPCA24 poster, repo 동봉) | 광학 |
| Transformer-Accel/efficient-transformer-accelerator, ViTALiTy | (HPCA'23 ViTALiTy) | linear attention |
| Others/ViM-Q-FCCM-2026 | (FCCM'26, repo 동봉) | Vision Mamba 양자화 |

---

## 5. 본 프로젝트 관점 핵심 추천 (재사용 Top 10)

> 본 프로젝트 추정: **HG-PIPE 계열 고처리량 ViT/Transformer FPGA 가속기 + XR 이벤트 시선추적**. (HGPIPE/HGTXR 마운트 기준 추정 — 확정 아님)

1. **HG-PIPE** — 전계층 파이프라인 ViT 가속기 골격(본 프로젝트 기반). 1순위 정독.
2. **TATAA** — 단일 PE를 INT8 GEMM ↔ bf16 비선형으로 변형: ViT의 선형/비선형 혼합 워크로드에 자원 재사용.
3. **DSP packing 일가(Uint-Packing / AGNA / AnyPackingNet / Diff-DiT)** — 저비트 MAC을 DSP에 압축해 처리량/면적 효율 극대화.
4. **AURA FlashAttention RTL** — 곱셈기 없는 exp + 1-pass online softmax: attention HW화의 정수/근사 레퍼런스.
5. **I-ViT / FQ-ViT / Q-HyViT** — softmax/GELU/LayerNorm 정수전용화 + 하이브리드 ViT PTQ: off-chip 이동·부동소수 제거.
6. **FlexLLM / FPGA_Friendly_SpinQuant** — TAPA 컴포저블 HLS + RapidStream 다중 SLR/HBM 플로어플랜 + Hadamard outlier 처리(스케일업 방법론).
7. **ViM-Q-FCCM-2026** — Vision Mamba APoT 양자화 + 곱셈기-free selective-scan HLS + SpinalHDL 통합(시선추적에 SSM 도입 시 직접 연계).
8. **ESDA/SEE** — 이벤트 sparse CNN의 zero-skip dataflow + ZCU102 실측(지연+전력) 하네스: XR 시선추적 FPGA 이식의 가장 직접적 출발점.
9. **저지연 이벤트 모델: TENNs-Eye(ais2024) / TDTracker(ais2025) / 3ET** — 스트리밍·경량 시선추적 알고리즘 후보(HW 매핑 용이).
10. **SJTU_microe / yolo-fpga-accelerator** — 명시적 2D PE array + per-layer 양자화 도메인 정렬: GEMM 매핑·스케줄링 패턴 참고.

---

## 6. 알려진 한계 / 확인 불가 항목

- **소스 부재(빌드 산출물/리포트만)**: LLM_FPGA, LUT-LLM, TeraFly, flightllm_test_demo → 라인단위 알고리즘은 "확인 불가", 리포트·바이너리·디렉토리 규약 역추정으로 작성.
- **부분 체크아웃/빈 repo**: ternaryLLM(설계 본체 일부 누락), TinyTransformer(사실상 빈 repo), ViTALiTy(Transformer-Accel 사본은 HW 부재).
- **명칭 vs 실체 불일치**: Tiny-GPT-on-Vortex(실제는 2-layer MLP), AnyPackingNet(실체 DeepBurning-MixQ).
- **정량 PPA 미확보**: 다수 repo에 합성 리포트(LUT/DSP/BRAM/주파수/fps) 미동봉 → 해당 수치 "확인 불가".
- **대용량 PDF 미판독**: NearEye-10000Hz.pdf(>20MB) → 교차인용 기반 작성, 본문 정밀내용 "확인 불가".
- **설계 결함 발견(각 .md §한계에 기록)**: Transformer-Accelerator(A_SIZE 불일치), vit-tiny-accelerator(scheduler 문법오류), MobileVit(top elaborate 불가), ViT-Accelerator(host 본체 부재) 등.

---

## 7. 디렉토리 구조

```
REF/Analysis/
├─ INDEX.md                     ← (본 문서)
├─ _VERIFICATION.md             ← 커버리지·품질 검증 리포트
├─ ViT-Accelerator/   (21)
├─ ViT-Quantization/  (48)
├─ Transformer-Accel/ (32)
├─ CNN-Accel/         (18)
├─ Others/            (11)
├─ XR-Eye-Tracking/Codebase/ (17)
└─ Papers/            (22)
```

---

## 8. 심층 MODULE_GUIDE 현황 (SRC_CASE_MODULE_GUIDE 수준)

> HG-PIPE `SRC_CASE_MODULE_GUIDE.md`와 동형의 모듈 단위 심층 가이드. 경로: `Analysis/<cat>/<repo>/MODULE_GUIDE.md`. 계획: [`_DEEP_ANALYSIS_PLAN.md`](_DEEP_ANALYSIS_PLAN.md).
> 각 가이드: 모듈별 6요소(역할/상하위·Mermaid·call stack·코드위치·실제 코드블록·마이크로아키텍처+정량) + 전 모듈 정량화(MAC lanes/scalar MACs/loop trips/memory, 모두 정적 도출) + 한눈표·읽기순서·병목노브.

**완료 (19편)**

| repo | 경로 | 모듈 수 | 핵심 정량(예) |
|---|---|---|---|
| TATAA | `ViT-Accelerator/TATAA/MODULE_GUIDE.md` | 10 | 512 DSP(32×16), bf16 16cyc, isqrt HW/SW 0x5f37 정합 |
| ESDA | `CNN-Accel/ESDA/MODULE_GUIDE.md` | 11 | DSP 2-MAC packing, 333MHz, block_0 MAC·유효MAC×sparsity |
| AURA-FlashAttention | `ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator/MODULE_GUIDE.md` | 10 | 256 MAC(4PE×64), ExpMul 곱셈기0, 64→1 가산트리 3stage |
| Edge-MoE | `Transformer-Accel/Edge-MoE/MODULE_GUIDE.md` | 12 | 256 MAC/cyc, MoE FFN 38M(16× 절감), GELU 184-LUT (타깃 ZCU102 정정) |
| hls-fpga-accelerators | `ViT-Accelerator/hls-fpga-accelerators/MODULE_GUIDE.md` | 6 | kPackets=BUS/DATA lanes, matmul DRAM 재전송 병목, unary exp 32-LUT |
| FlexLLM | `ViT-Accelerator/FlexLLM/MODULE_GUIDE.md` | 7 | prefill FFN 2048 MAC/cyc, i4 2×2=4MAC/DSP, GQA 32/8 |
| FPGA_Friendly_SpinQuant | `Transformer-Accel/FPGA_Friendly_SpinQuant/MODULE_GUIDE.md` | 6+3 | i4 2×2 4MAC/DSP, FHT R4 곱셈0·13stage, on-FPGA Top-K=5 |
| Diff-DiT | `Transformer-Accel/Diff-DiT/MODULE_GUIDE.md` | 8 | 8192 PE, INT3 6-pack 49152 MAC/cyc, diff softmax 0회 |
| efficient-transformer-accelerator | `ViT-Accelerator/efficient-transformer-accelerator/MODULE_GUIDE.md` | 9 | 512 MAC(32×16), 양자화 8배치 시분할, RTL 버그 2+2건 |
| ViT-FPGA-TPU | `ViT-Accelerator/ViT-FPGA-TPU/MODULE_GUIDE.md` | 6 | 256 MAC(16×16 FP16), 컨트롤러 RTL 확보, 순수 GEMM 판정 |
| MobileVit | `Transformer-Accel/MobileVit-AI-Hardware-Accelerator/MODULE_GUIDE.md` | 9 | 1024 MAC(4타일×16×16), 63cyc/matmul, top elaborate 불가 결함 |
| trans-fat | `Transformer-Accel/trans-fat/MODULE_GUIDE.md` | 7 | 128 MAC/cyc, routed DSP 831, WNS −0.278ns(타이밍 미달), PPA 정정 |
| Transformer-Accelerator-Based-on-FPGA | `ViT-Accelerator/Transformer-Accelerator-Based-on-FPGA/MODULE_GUIDE.md` | 9 | A_size² systolic, 내부 override 16→24·OUT_MEM 21/32 불일치 |
| Transformer_dataflow | `Transformer-Accel/Transformer_dataflow/MODULE_GUIDE.md` | 8 | MAC lanes=1(미최적), attention 행융합 커널, double 골든 |
| ViT-Accelerator(subdir) | `Transformer-Accel/ViT-Accelerator/MODULE_GUIDE.md` | 6 | 256 MAC(16×16), 호스트 본체 실재(정정), 785 vs 197 토큰 불일치 |
| ViT-Accel-INT8 | `Transformer-Accel/ViT-Accelerator-on-FPGA-with-INT8-quantization/MODULE_GUIDE.md` | 6 | 256 MAC(16×16), streaming softmax, GELU 177-LUT, 가중치 재로딩 병목 |
| llama-fpga | `ViT-Accelerator/llama-fpga/MODULE_GUIDE.md` | 12 | SpinalHDL FP16 128 lane/core, W4A16+KV INT8, ring AllReduce 텐서병렬 |
| HLS-Acceleration-of-LLaMA2 | `ViT-Accelerator/HLS-Acceleration-of-LLaMA2/MODULE_GUIDE.md` | 8 | MAC lanes=1(II=1), classifier GEMM 9.22M(58%), KV/가중치 DRAM 메모리바운드 |
| vit-tiny-accelerator | `Transformer-Accel/vit-tiny-accelerator/MODULE_GUIDE.md` | 15 | 8×8=64 DSP, 512KB ping/pong, scheduler_tiler.v 문법오류 2건(합성불가) |

**Phase 2 완료 (이벤트/희소/systolic/CNN, 12편)**

| repo | 경로 | 모듈 수 | 핵심 정량(예) |
|---|---|---|---|
| ViM-Q-FCCM-2026 | `Others/ViM-Q-FCCM-2026/MODULE_GUIDE.md` | 17 | SSM scan 151K trips, linear 256 shift-LUT/cyc(곱셈기-free), conv 64 shift-add |
| HiSparse | `Others/HiSparse/MODULE_GUIDE.md` | 12 | 128-way(16ch×8), PE in-flight forwarding II=1, 2단 셔플, CPSR 비트정합 |
| DPACS | `Others/DPACS/MODULE_GUIDE.md` | 10 | DSP 2-MAC, key 21bit, elastic linebuffer 393Kb, 타깃 part 불일치 정정 |
| HiSpMV | `Others/HiSpMV/MODULE_GUIDE.md` | 10 | 128 PE, II_DIST=5 Pre-Accum, 64b nnz 패킹 대칭, DSE 채널배분식 |
| HiSpMM | `Others/HiSpMM/MODULE_GUIDE.md` | 12 | N0=8 SIMD(640/512 MAC/cyc), delta 25% 변형 자동선택, SpMV 대비 차이 |
| REMOT-FPGA-22 | `Others/REMOT-FPGA-22/MODULE_GUIDE.md` | 10 | 이벤트구동 AMAP, 곱셈기0, FULL/HASH/FIFO 메모리 트레이드, 오버플로 잠재버그 |
| acap-gemm-sa | `Others/acap-gemm-sa/MODULE_GUIDE.md` | 10 | Versal AIE 8×50 3200 MAC/cyc(FP32), 3단 타일, AMPL/Gurobi DSE 정합 |
| TMMA | `Others/TMMA/MODULE_GUIDE.md` | 9 | 32×32=1024 DSP(83%), 합성 PPA 실측, 통신-계산 미중첩 병목 |
| SJTU_microe | `CNN-Accel/SJTU_microe-main/MODULE_GUIDE.md` | 11 | 4MUL/DSP 패킹, conv2 128 INT4-MAC/cyc(MAC 검산일치), bypass 320ch |
| Uint-Packing | `CNN-Accel/Uint-Packing-master/MODULE_GUIDE.md` | 8 | DSP2/DSPopt3 비트배치 해부, 4w4a, 타깃 Ultra96-v2 정정 |
| AnyPackingNet(=DeepBurning-MixQ) | `CNN-Accel/AnyPackingNet-main/MODULE_GUIDE.md` | 10 | DSP packing factor표, NAS 49^L 탐색, dsp_factors 명명 버그 발견 |
| XJTU-Tripler | `CNN-Accel/XJTU-Tripler-master/MODULE_GUIDE.md` | 8 | HPU MPU 8×8=64 MAC, 행렬 512b/누산 2048b, conv 5중루프 |

**Phase 3 완료 (CNN/DAC-SDC/YOLO/Tier2-3, 16편)**

| repo | 경로 | 핵심 정량/발견 |
|---|---|---|
| dac_sdc_2022_champion | `CNN-Accel/dac_sdc_2022_champion-master/MODULE_GUIDE.md` | UltraNet DSPopt3 4세그먼트 비트배치, conv1 256 픽셀-MAC/cyc |
| dac_sdc_2023_champion | `CNN-Accel/dac_sdc_2023_champion-main/MODULE_GUIDE.md` | UINT-Packing(unsigned+subdata 차감), 2022 대비 12행 차이표 |
| dac_sdc_2020_designs | `CNN-Accel/dac_sdc_2020_designs-master/MODULE_GUIDE.md` | 3팀 비교(BJUT/SkrSkr/iSmart), bn_res 폭 미확장 병목 |
| dac_sdc_2021_designs | `CNN-Accel/dac_sdc_2021_designs-main/MODULE_GUIDE.md` | 3팀(SJTU 4MUL/DSP·SkrSkr 2MUL·iSmart FIR4tap), 22champion 계보 |
| dac_sdc_2022_designs | `CNN-Accel/dac_sdc_2022_designs-main/MODULE_GUIDE.md` | InvolutionNet(PL bilinear resize), SEUer=champion cross-ref, ultrateam 부재 |
| SkrSkr | `CNN-Accel/SkrSkr-master/MODULE_GUIDE.md` | SkyNet W6A8 PW 512 MAC/cyc, DRAM 160Mb 경유, im2col 가설 정정 |
| SkyNet-ZCU104 | `CNN-Accel/SkyNet-ZCU104-master/MODULE_GUIDE.md` | conv1x1 16-input dot-tree×32(II=2→256), 512b weight reorder |
| yolo-fpga-accelerator | `CNN-Accel/yolo-fpga-accelerator-main/MODULE_GUIDE.md` | Tm32×Tn4=128 MAC OS, per-layer Q 도메인 정렬, 3중 ping-pong |
| yolov2_xilinx_fpga | `CNN-Accel/yolov2_xilinx_fpga-flex/MODULE_GUIDE.md` | Tm60×Tn2=120 MAC, INT16 128b 62.8 GOP/s(실측표), II=1 |
| SJTU_microe | (Phase2 표 참조) | — |
| Kria-YOLOv4-Tiny | `CNN-Accel/Kria-YOLOv4-Tiny-FPGA-Accelerator/MODULE_GUIDE.md` | ★안티패턴: 단일 conv(전체 ~2.5%)·8 MAC·>>8 고정시프트·입력 32× 재독 |
| FlexCNN | `CNN-Accel/FlexCNN/MODULE_GUIDE.md` | auto_compile codegen, SA 16×14×8=1792 MAC(정정), engine 자동배선 |
| tataa_tvm_dev | `CNN-Accel/tataa_tvm_dev-main/MODULE_GUIDE.md` | 순수 vanilla TVM 0.23(TATAA 자체 추가분 0 — 부재 입증) |
| SEE | `CNN-Accel/SEE/MODULE_GUIDE.md` | ESDA wrapper, ZCU102 PYNQ 지연+INA226 18레일 전력 동시측정 하네스 |
| transformer-hls-thesis | `ViT-Accelerator/transformer-hls-thesis/MODULE_GUIDE.md` | Opus-MT 인코더 50.9M MAC, README↔코드 불일치 6건(GELU→ReLU 등) |
| HLSTransformation(+submission) | `ViT-Accelerator/HLSTransformation/MODULE_GUIDE.md` | TinyLlama2 int8 MAC lanes=1(UNROLL 주석), submission byte-동일 |
| ternaryLLM | `ViT-Accelerator/ternaryLLM/MODULE_GUIDE.md` | 삼진 곱셈0(CPU AVX/GPU SpMM/FPGA SpinalHDL), SSR 소스 부재 |

**심층 MODULE_GUIDE 총계: HW 가속기 47편 완료** (Phase1 19 + Phase2 12 + Phase3 16). 모든 자체 HW repo(Tier1/2/3) 커버. third-party·코드 0건 제외.

---

## 9. S-PyTorch MODULE_GUIDE 현황 (알고리즘: 양자화 + XR 모델)

> HW 가이드와 동형 6요소 + **PyTorch 지표(FLOPs/params/activation memory/비트폭·observer)**로 치환. 경로: `Analysis/<cat>/<repo>/MODULE_GUIDE.md`. 계획: [`_DEEP_ANALYSIS_PLAN_SPYTORCH.md`](_DEEP_ANALYSIS_PLAN_SPYTORCH.md).
> **완료 65편** (S1 Tier A 16 + S2 XR 17 + S3 Tier B 18 + S4 Tier C 14). **ViT-Quantization 48종 + XR 시선추적 모델 17종 전수 완료.**

### 9.1 Tier A — FPGA 직결 양자화 (16, `ViT-Quantization/<repo>/`)
I-ViT(정수전용 ShiftGELU/Shiftmax), FQ-ViT(PTF+Log-Int-Softmax), RepQ-ViT(scale reparam 대수증명), integer-only-transformer(I-ViT+I-BERT 번들), Q-HyViT(hybrid recon), P2-ViT(PoT 시프트), ShiftAddViT(shift&add ~49×↓), PTQ4ViT(twin uniform), NoisyQuant(bias 흡수→MAC 불변), outlier-free-transformers(clipped softmax+gated attn), AdaLog(적응 로그밑→시프트+LUT), APHQ-ViT(APH+MLP recon), mixed-non-linear-quantization(I-ViT/I-BERT/FQ 비선형 함수별 비교), int-flashattention·INT8-FMHA·qflash(GEMM만 INT8·softmax FP32).

### 9.2 S2 — XR 시선추적 모델 (17, `XR-Eye-Tracking/Codebase/<repo>/`)
cb-convlstm(3ET, 0.417M·delta sparsity), EventMamba(포인트 SSM), gg_ssms(그래프+MST SSM), ais2024(TENNs-Eye 등 3종), ais2025(ALiBi Tf·TDTracker), EX-Gaze(선형어텐션 하이브리드), retina(IAF SNN 4모드), E-Track(TF·RoI 96%↓), FACET(EPNet 타원+GWD), EV-Eye(U-Net 31M+ICP), EllSeg, RITnet(0.25M), Swift-Eye(SiamRPN++), eyegraph(데이터셋 repo·알고리즘 부재), event_based_gaze_tracking(데이터셋·9B BHHI), EyeLoRiN(모델 없는 후처리), ESDA(XR cross-ref).

### 9.3 Tier B — PTQ 정밀·백본 (18, `ViT-Quantization/<repo>/`)
Castling-ViT(linear-angular ~6.1×↓), RepViT(reparam 단일 conv), Next-ViT(NCB/NTB), EdgeVisionTransformer(프루닝+외부런타임 PTQ), psaq-vit(data-free 합성), Q-ViT-DeiT(LSQ+head-wise bit), Bi-ViT(1-bit), FIMA-Q(FIM-DPLR), OFQ(oscillation 억제), transformer-quantization(Qualcomm 인프라), UQ-ViT(=Uniform Quant), AdaTSQ(DiT temporal-sensitivity, 코드 미공개), Mix-Quant(실제 Qwen3 분리서빙·오분류), CLAMP-ViT(평가전용), Quantformer(그룹 soft-mixture), AHCPTQ(SAM PTQ·log2 shift), postcalibration4quantization(ONNX 영샷 재보정), M3ViT(MoE active/total).

### 9.3b Tier C — 확장 도메인 양자화 (14, `ViT-Quantization/<repo>/`)
VLM/VLA: MBQ(AWQ 확장 모달리티 균형), qvlm(Q-VLM cross-layer dependency), QuantVLA(GR00T DuQuant W4A8). Diffusion: Q-DiT(group quant+evolutionary), Q-VDiT(LoRA codebook+temporal 보정), ViDiT-Q(smooth+dynamic W4A8). SAM: ptq4sam(BIG+AGQ), SAQ-SAM(PCC/PAR 의미정렬). Detector: Q-DETR(LSQ+W*A8, DRD 코드 부재). 기타: PQV-Mobile(dynamic quant+프루닝), FPQVAR(VAR FP4/FP6 비균일 grid+Hadamard), mimiq(data-free W4A4+head 증류), ViTALiTy(Taylor linear attn, quant 경로 대부분 dead-code), AMD_QTViT(**"QT"=Quadratic Taylor, 양자화 아님** — EfficientViT FP).

### 9.4 교차분석 — FPGA 비선형/곱셈 HW화 의사결정
- **softmax/GELU/LayerNorm 정수화 직접 구현체**: I-ViT(시프트), FQ-ViT(Log-Int-Softmax+PTF), integer-only-transformer(I-BERT 다항). `mixed-non-linear-quantization`이 이 3계열을 함수별로 직접 비교 → **시프트 근사(barrel shifter) vs 다항식(제곱기) 자원·정확도 트레이드오프** 의사결정표.
- **곱셈기 제거**: ShiftAddViT(shift&add), P2-ViT/AdaLog(PoT/로그→시프트), Bi-ViT(1-bit XNOR).
- **attention HW화**: int-flashattention·qflash·INT8-FMHA 모두 **GEMM만 INT8·softmax는 FP32** → **I-ViT IntSoftmax와 결합 시 완전 정수 attention**(권고).
- **outlier 대응**: outlier-free-transformers(clipped softmax+gated), transformer-quantization(PEG), RepQ-ViT(reparam per-tensor화), NoisyQuant(bias 흡수).
- **저지연 XR 모델 후보(HW 이식성↑)**: 3ET(cb-convlstm, delta sparsity)·TENNs-Eye(인과 Conv3d 스트리밍)·retina(IAF SNN mW급)·E-Track(RoI 96%↓). SSM 계열(EventMamba/TDTracker)은 selective scan은 가속 적합하나 FPS/그래프 연산은 비친화.
- **주의(명칭-실체 불일치 발견)**: Mix-Quant(실제 LLM 서빙), UQ-ViT(Uniform), eyegraph/event_based_gaze_tracking(데이터셋 repo), AdaTSQ/CLAMP-ViT(핵심 코드 미공개).

**전체 MODULE_GUIDE 총계: 112편** (HW 가속기 47 + S-PyTorch 65). 모든 자체 HW 가속기 repo + ViT-Quantization 48종 + XR 시선추적 모델 17종 전수 심층화 완료.

