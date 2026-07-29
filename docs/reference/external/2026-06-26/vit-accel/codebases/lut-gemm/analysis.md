---
source_type: codebase
source_name: lut-gemm
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/lut-gemm/analysis.md -->

# lut-gemm Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm`
- repo_remote: `origin	https://github.com/naver-aics/lut-gemm (fetch)`
- category: `LUT quantized GEMM`
- HGTXR relevance: `low`
- matched_paper: `LUT-GEMM`
- paper_title: LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/LUT-GEMM.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `276` |
| 주요 언어 | `C++:104, C/C++ header:54, Python:36, Markdown:31, no_ext:9, C++ header:9, .txt:8, .in:6, .cu:5, YAML:4` |
| LOC 추정 | `C++:57083, C/C++ header:27365, Markdown:13526, Python:7088, .txt:2173, C++ header:1003, Shell:199, YAML:85` |

### Directory Map
- `docs/` (1 entries)
- `examples/` (7 entries)
- `lutGEMM/` (3 entries)
- `lutGEMM/include/` (3 entries)
- `lutGEMM/src/` (3 entries)
- `tests/` (5 entries)
- `tests/include/` (4 entries)
- `tests/opt/` (3 entries)
- `tests/src/` (1 entries)
- `thirdparty/` (1 entries)
- `thirdparty/googletest/` (14 entries)

### Metadata / Config Refs
- `README.md`
- `examples/README.md`
- `thirdparty/googletest/README.md`
- `thirdparty/googletest/googletest/README.md`
- `thirdparty/googletest/googletest/docs/README.md`
- `thirdparty/googletest/googletest/include/gtest/internal/custom/README.md`
- `thirdparty/googletest/googlemock/README.md`
- `thirdparty/googletest/googlemock/docs/README.md`
- `thirdparty/googletest/googlemock/include/gmock/internal/custom/README.md`

### Core Source Refs
- `thirdparty/googletest/googlemock/test/gmock-matchers_test.cc`: 8691 lines
- `thirdparty/googletest/googletest/test/gtest_unittest.cc`: 7801 lines
- `thirdparty/googletest/googletest/src/gtest.cc`: 6816 lines
- `thirdparty/googletest/googlemock/include/gmock/gmock-matchers.h`: 5588 lines
- `thirdparty/googletest/docs/gmock_cook_book.md`: 4299 lines
- `thirdparty/googletest/googlemock/test/gmock-spec-builders_test.cc`: 2774 lines
- `thirdparty/googletest/googletest/include/gtest/gtest.h`: 2502 lines
- `thirdparty/googletest/googletest/test/gtest_pred_impl_unittest.cc`: 2422 lines
- `thirdparty/googletest/docs/advanced.md`: 2379 lines
- `thirdparty/googletest/googletest/include/gtest/internal/gtest-port.h`: 2378 lines

### Hardware-Oriented Source Refs
- `lutGEMM/src/cuda/tmpWeight.hpp`
- `lutGEMM/src/cuda/kernels/mm_t.hpp`
- `lutGEMM/src/cuda/kernels/cublas.h`
- `lutGEMM/src/cuda/kernels/dequant.hpp`
- `lutGEMM/src/cuda/kernels/mv_fp16_bias.hpp`
- `lutGEMM/src/cuda/kernels/gptq_faster_fp16_bias.hpp`
- `lutGEMM/src/cuda/kernels/mv_fp16.hpp`
- `lutGEMM/src/cuda/kernels/mv.hpp`
- `lutGEMM/src/cuda/kernels/gptq_fp16_bias.hpp`
- `lutGEMM/src/cuda/kernels/dequant_fp16.hpp`
- `lutGEMM/include/kernels.h`
- `lutGEMM/include/nQWeight_fp16.h`
- `tests/main.cc`
- `tests/opt/_cublas.cc`
- `tests/src/custom_random.cpp`
- `tests/include/tests.h`
- `tests/include/_cublas.h`
- `tests/include/custom_random.h`
- `tests/include/timer.h`
- `thirdparty/googletest/googletest/samples/prime_tables.h`
- `thirdparty/googletest/googletest/samples/sample3_unittest.cc`
- `thirdparty/googletest/googletest/samples/sample3-inl.h`
- `thirdparty/googletest/googletest/samples/sample2.cc`
- `thirdparty/googletest/googletest/samples/sample1.h`
- `thirdparty/googletest/googletest/samples/sample2_unittest.cc`
- `thirdparty/googletest/googletest/samples/sample10_unittest.cc`
- `thirdparty/googletest/googletest/samples/sample1.cc`
- `thirdparty/googletest/googletest/samples/sample1_unittest.cc`
- `thirdparty/googletest/googletest/samples/sample4.h`
- `thirdparty/googletest/googletest/samples/sample8_unittest.cc`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/README.md`
- # LUT-GEMM
- This repository provides the official implementation of LUT-GEMM from the following paper.
- **LUT-GEMM: Qantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models**
- _Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, and Dongsoo Lee_
- Paper: https://arxiv.org/pdf/2206.09557.pdf
- <p align="center"><img width="500" alt="image" src="docs/overview.png">  </p>
- ## Quick Start
- Run the following commands to get **`Kernel Evaluation`** results in Table 1.
- ``` sh
- mkdir build
- cd build
- cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
- make -j8
- ./tests/tests
- ```
- ## Citation
- ```
- @misc{park2023lutgemm,
- title={LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models},
- author={Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee and Dongsoo Lee},

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: LUT 기반 quantized GEMM으로 dequantization을 제거하고 계산량을 줄인다.
- 알고리즘 축: group-wise quantization과 lookup table product accumulation으로 low-bit GEMM을 처리한다.
- 하드웨어 축: GPU kernel 중심이나 LUT computation idea가 FPGA에도 이전 가능하다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/examples/rtn_parameter.py:83:        scale  = torch.matmul(scale, upack)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/CMakeLists.txt:2:project(nQmatmul CXX C)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/include/kernels.h:26:void matmul(void* output, nQWeight_fp16 &nqW, void* input, int n, int algo=0);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/include/kernels.h:27:void matmul(void* output, void* input, nQWeight_fp16 &nqW, int m, int algo=0);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/include/kernels.h:28:void matmul_gptq(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/include/kernels.h:31:void matmul_gptq_faster(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/CMakeLists.txt:23:    opt/fp16/int3_col_wise_matmul_fp16.cu`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:30:void matmul(void* output, nQWeight_fp16 &nqW, void* input, int n, int algo);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:31:void matmul(void* output, void* input, nQWeight_fp16 &nqW, int m, int algo);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:36:inline void matmul_useCublas(__half* output, nQWeight_fp16 &nqW, __half* input, int n);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:37:inline void matmul_useCublas(__half* output, __half* input, nQWeight_fp16 &nqW, int m);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:40:void matmul_gptq(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:48:void matmul_gptq_faster(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:56:void matmul(void* output, nQWeight_fp16 &nqW, void* input, int n, int algo){`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:62:    else     matmul_useCublas((__half*)output, nqW, (__half*)input, n);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:64:void matmul(void* output, void* input, nQWeight_fp16 &nqW, int m, int algo){`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:70:    else     matmul_useCublas((__half*)output, (__half*)input, nqW, m);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:73:inline void matmul_useCublas(__half* output, nQWeight_fp16 &nqW, __half* input, int n) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/kernels.cu:77:inline void matmul_useCublas(__half* output, __half* input, nQWeight_fp16 &nqW, int m) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/lutGEMM/src/cuda/kernels/gptq_fp16_bias.hpp:35:__global__ void VecQuant3MatMulKernel(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/README.md:11:Abstract: Our proposed kernel, LUT-GEMM, accelerates quantized matrix multiplication by leveraging both uniform and non-uniform quantization techniques. Utilizing sub-4-bit quantized weights, it offers flexibility and achieves high compression ratios, allowing a balance between accuracy and efficiency. Through the use of low-bit quantization and efficient LUT-based operations, it effectively reduces memory usage and computational costs, thereby significantly enhancing the inference speed of large-scale language models.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/README.md:32:      title={LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models}, `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:30:void floatToInt8(int8_t *out, float *inp, int size, float scale=1.0){`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:36:int8_t float2int8(float f, float scale) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:37:    int8_t i = int8_t(f * scale);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:73:    } else if (std::is_same<T, int8_t>::value) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:172:    int8_t *iA, *iB; int32_t *iC;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:185:    floatToInt8(iA, fA, m*k);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:191:    floatToInt8(iB, fB, k*n);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:200:    printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/tests/opt/_cublas.cu:226:        printf("int8 mean error : %lf\n", ierr/m/n);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/examples/README.md:3:## Model Quantization Examples`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/examples/README.md:8:python quant_model_bcq.py \`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/examples/README.md:15:python quant_model_rtn.py \`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/lut-gemm/examples/quant_model_bcq.py:54:        help="Quantization Bits.",`

## 5. HGTXR 적용 해석
- 현재 resource 정책과 충돌하므로 P3 negative-control로만 둔다.

### 적용 가능 모듈
- Quantization / scale calibration / weight generation

## 6. 실험 옵션으로 변환할 때의 규칙
- 먼저 HGTXR-SW 또는 C++ reference에서 accuracy/bit-exact behavior를 검증한다.
- HLS에 반영할 때는 C3b signoff path를 덮어쓰지 말고 새로운 suffix variant로 추가한다.
- 완료된 `csynth.xml`, routed timing/power, PYNQ smoke JSON 없이는 resource matrix의 완료 variant로 승격하지 않는다.
- HGTXR 논문 범위를 벗어나는 구조 변경은 `paper_scope_review_required`로 둔다.

## 7. 리스크와 비적용 조건
- 동적 MoE routing, softmax-free attention, ternary/LUT-heavy compute는 정확도 또는 resource 정책과 충돌할 수 있다.
- 현재 사용자의 resource 방향은 DSP/URAM 활용 증가와 LUT 과사용 억제이므로 LUT-LLM/LUT-GEMM류는 기본값이 아니라 negative-control이다.
- codebase가 LLM 중심이면 HGTXR eye-tracking 적용은 kernel/system-flow 수준으로 제한한다.

## 8. 다음 분석/구현 액션
- 핵심 source file을 1개 선택해 line-by-line 분석을 수행한다.
- 해당 방법을 HGTXR `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml`의 planned experiment와 연결한다.
- SW accuracy metric과 HW resource metric을 같은 manifest에 기록한다.
