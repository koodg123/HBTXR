# EdgeVisionTransformer 정밀 분석 (엣지 디바이스 ViT 양자화·벤치마킹)

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\EdgeVisionTransformer`
> 작성일: 2026-06-20 / 실제 소스 코드 기반. 라인 근거(파일:라인) 표기.
> **제외**: `deit_pruning/vendor/nn_pruning_v1/` (vendor), `.git/`, `__pycache__/`.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **목적**: Vision Transformer(ViT/DeiT/T2T-ViT/Swin)와 CNN을 **모바일·엣지 디바이스에서 추론 성능(지연·메모리·전력)** 측정하고, **다중 런타임(TFLite / TensorRT / ONNX Runtime / OpenVINO / TVM)의 양자화 op별 비용**을 비교하는 도구 모음. (README.md:1-6)
- **원논문/출처**: 독립 논문이라기보다 **벤치마킹 툴킷**. 모델 구현은 외부(kamalkraj Vision-Transformer, yitu T2T-ViT, rishigami Swin-TF)에서 차용, pruning은 are-16-heads / nn_pruning에서 차용. (README.md:2, 6)
- **핵심 아이디어**:
  1. ViT/CNN 모델을 TFLite로 변환(float16 / dynamic-range / INT8 PTQ)하고 adb로 안드로이드 폰에서 지연·메모리 측정. (README.md:18-66, utils.py:242-294)
  2. **op-level 양자화 비용 스윕**: Dense/Conv/DWConv/ReLU 등 단위 연산을 입·출력 차원·seq_len을 바꿔가며 양자화 모델로 만들어, 런타임별 양자화 오버헤드/속도를 측정(experiments/D11xx*). (D1118_tflite_quant_op_test.py 등)
  3. DeiT 구조적 pruning(헤드/FFN) + KD 파이프라인 제공(are_16_heads, deit_pruning). (README.md:68-101)

---

## 2. 디렉토리 구조 (자체 소스, vendor 제외)

```
EdgeVisionTransformer/
├── utils.py                       # ★ tf2tflite(양자화), export_onnx, 모델 로더 모음
├── flops_calculation.py
├── experiments.py
├── modeling/
│   ├── torch_layers/              # ★ PyTorch 참조 레이어 (attention/ffn/norm/activation/residual)
│   ├── layers/                    # TF1 계열 레이어 (attention/ffn/norm/embedding/transformer_encoder)
│   ├── models/                    # vit.py, t2t_vit.py, + CNN zoo(mnasnet/shufflenet/squeezenet 등)
│   └── save_model.py
├── benchmark/
│   ├── tensorrt/
│   │   ├── calibrator.py          # ★ TRT INT8 EntropyCalibrator2
│   │   ├── onnx_trt_test.py, common.py
│   ├── openvino/vino_cli.py       # ★ OpenVINO model_optimizer + benchmark_app CLI 래퍼
│   ├── ADBConnect.py, run_on_device.py, bench_utils.py  # 모바일(adb) 벤치
├── experiments/                   # ★ 런타임별 양자화 op 테스트 (날짜 코드 D11xx/D12xx/D0104)
│   ├── D1118_tflite_quant_op_test.py     # TFLite INT8 op 스윕
│   ├── D1122_onnx_quant_op_test.py       # ONNX Runtime dynamic quant op 스윕
│   ├── D1124_trt_quant_op_test.py        # TensorRT int8_mode op 스윕(torch2trt)
│   ├── D1201_vino_quant_op_test.py       # OpenVINO 양자화 op
│   ├── D1130/D1207_vino/tflite_quant_cnn_test.py   # CNN 양자화 테스트
│   ├── D1130/D1202_tflite_gpu_*           # TFLite GPU 프로파일링
│   ├── D1230_tflite_transformer_power_test.py       # 트랜스포머 전력 측정
│   └── D0104_tvm_fusion_test.py           # TVM fusion 테스트
├── are_16_heads/                  # 헤드 pruning(분석 보조)
└── deit_pruning/src/              # DeiT 구조 pruning + ONNX export (vendor/ 제외)
```

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `utils.py` — 변환·양자화의 중심 (가장 중요)

#### `tf2tflite(...)` (utils.py:242-294) — TFLite PTQ 양자화 구현
- `quantization='float16'` (utils.py:256-259): `optimizations=[Optimize.DEFAULT]`, `supported_types=[tf.float16]` → **FP16 PTQ**.
- `quantization='dynamic'` (utils.py:260-262): `optimizations=[Optimize.DEFAULT]`만 → **dynamic-range 양자화**(가중치 INT8, 활성 런타임 동적).
- `quantization='int8'` (utils.py:263-277): **완전 INT8 PTQ**
  - `representative_data_gen()` (utils.py:265-269): `tf.random.normal(input_shape)`를 100회 yield하는 **합성(난수) calibration 데이터셋** → 활성 스케일 추정. (실데이터 아님, 주석으로 실데이터 대체 가능 표기)
  - `optimizations=[Optimize.DEFAULT]`, `representative_dataset=representative_data_gen`.
  - `target_spec.supported_ops=[TFLITE_BUILTINS_INT8]` (utils.py:273) — 양자화 불가 op 발견 시 에러.
  - `inference_input_type/output_type = tf.int8` (utils.py:276-277) — 입출력까지 INT8.
- `use_flex=True` (utils.py:279-284): `SELECT_TF_OPS`를 추가해 트랜스포머 전용 op(Einsum, ExtractImagePatches, Erf, Roll 등; README.md:34) 지원.
- 함수 끝: convert 후 .tflite 저장. `tf2tflite_dir`(utils.py:297-)는 디렉토리 일괄 변환.
- 모델 로더: `get_torch_deit`(utils.py:52-62, torch.hub deit), `get_huggingface_vit_model`(92-96), `get_swin`(14-47), CNN(mobilenetv2/v3, efficientnet 등 65-90). `export_onnx_fix_batch`(176-177).

### 3.2 `benchmark/tensorrt/calibrator.py` — TRT INT8 캘리브레이터
- `DummyCalibrator(trt.IInt8EntropyCalibrator2)` (calibrator.py:25-71):
  - `IInt8EntropyCalibrator2` 상속 → **TensorRT의 엔트로피(KL-divergence) 기반 INT8 PTQ** 캘리브레이터. (calibrator.py:25, 29)
  - `__init__`: 학습데이터 텐서를 받아 디바이스 메모리 `cuda.mem_alloc(data[0].nbytes * batch_size)` 할당(default batch=64). (calibrator.py:26-39)
  - `get_batch(names)` (calibrator.py:47-58): 현재 인덱스에서 batch_size 만큼 `ravel()`하여 `memcpy_htod`로 GPU 복사 후 디바이스 포인터 반환. 데이터 소진 시 None.
  - `read/write_calibration_cache` (calibrator.py:61-71): **현재 캐시 비활성화(pass)** — 매 실행 재캘리브레이션(주석 처리된 캐시 IO 존재). → 한계 항목.
- 즉 TRT INT8 빌드 시 활성 텐서의 스케일/zp를 엔트로피 방식으로 산출하는 표준 NVIDIA 패턴(NVIDIA 2021 라이선스 헤더 calibrator.py:1-15).

### 3.3 `benchmark/openvino/vino_cli.py` — OpenVINO 변환·벤치 CLI
- `model_optimize(...)` (vino_cli.py:31-39): OpenVINO `mo.py`(model optimizer)를 호출, `--data_type`로 FP32/FP16 등 지정(양자화는 데이터타입 변환 수준). (vino_cli.py:32, 48)
- `openvino_benchmark(...)` (vino_cli.py:73-101): `benchmark_app.py`를 `-api=sync --report_type=detailed_counters`로 실행, **layer별 latency를 CSV에서 정규식 파싱**해 합산(layername_pattern 기본 `Total`). (vino_cli.py:73-101)
- CLI 진입: `mo`/`model_optimize`, `benchmark` 두 서브커맨드(vino_cli.py:148-158).

### 3.4 `experiments/D11xx_*quant_op_test.py` — 런타임별 양자화 op 스윕 (정밀 비교의 핵심)

#### `D1118_tflite_quant_op_test.py` (TFLite INT8 op)
- `make_model(input_shape, op)`로 Keras 단일 op 모델 생성 → `quant_model`이 FP32 + **INT8(`quantization='int8'`)** 두 버전 변환. (D1118:14-23)
- 스윕 대상: Dense(out 160~224, 1~64; in 다양; seq_len 1~197; 2^x 채널), Conv2D(kernel 1/3/5/7, cin/cout 1~128), DepthwiseConv2D, ReLU. (D1118:33-195) → **op·차원별 INT8 양자화 비용 곡선** 수집용.

#### `D1122_onnx_quant_op_test.py` (ONNX Runtime)
- `onnxruntime.quantization.quantize_dynamic(activation_type=QUInt8)`로 **dynamic 양자화**(D1122:35). static(`quantize_static` + `FooDataReader` 난수 캘리브레이션)은 주석 처리(D1122:17-23, 37-40).
- Dense/Conv/DWConv를 동일 차원 스윕으로 ONNX FP32→dynamic-INT8 변환. (D1122:45-149)

#### `D1124_trt_quant_op_test.py` (TensorRT)
- `torch2trt(model, [dummy], int8_mode=True)`로 **TRT INT8 엔진** 생성, FP32 엔진과 함께 저장. (D1124:29-33)
- Dense/Conv/DWConv 스윕(차원 그리드). 입력 shape를 state_dict에 동봉. (D1124:18-114)

> 즉 동일한 op 그리드(Dense/Conv/DWConv)를 **TFLite INT8 / ONNX dynamic / TRT int8 / OpenVINO**로 각각 변환해 **런타임별 양자화 지원·속도·정확도 차이를 횡단 비교**하는 설계. (D1130/D1207은 CNN 모델 전체, D1230은 전력, D0104는 TVM fusion)

### 3.5 `modeling/torch_layers/attention.py` — 참조 ViT 어텐션
- `Attention(nn.Module)` (attention.py:4-48): HuggingFace ViTSelfAttention 참조. `to_query/to_key/to_value`(각 Linear), `to_out`(Linear). (attention.py:19-22)
- `forward` (attention.py:29-48): `transpose_for_scores`로 (B,heads,N,head_size) 변형 → `scores = Q·K^T * scale`(scale=head_size^-0.5) → `Softmax(dim=-1)` → `probs·V` → reshape → to_out. **표준 FP 어텐션**(양자화는 변환 단계에서 외부 런타임이 수행, 여기 코드 자체는 비양자화 참조 구현). (attention.py:36-47)
- `ffn.py`/`norm.py`/`activation.py`(torch_layers): FFN(Linear-act-Linear), LayerNorm, GELU 등 표준 빌딩블록(런타임 양자화 대상 op 정의용). `modeling/models/vit.py`·`t2t_vit.py`는 TF 구현 모델(README.md:6).

### 3.6 pruning (보조)
- `are_16_heads/pruning.py`: 어텐션 헤드 중요도 기반 pruning. `deit_pruning/src/`: nn_pruning 기반 DeiT 구조 pruning + ONNX export(onnx_export.py)·latency 측정(get_latency.py). KD(`--do_distil --teacher_model deit-base`)로 정확도 회복(README.md:97-101). **양자화와 직교하는 압축 축**.

---

## 4. 알고리즘 / 수식 — INT8 PTQ Calibration

- **TFLite INT8 PTQ** (utils.py:263-277): per-tensor affine 양자화. 활성 스케일은 representative_dataset(여기선 `tf.random.normal` 100배치)을 흘려 **min/max 관측 → scale/zero-point** 산출. 양자화식: `q = round(x/scale) + zp`, `scale = (max-min)/(qmax-qmin)`, `inference_input_type=int8`.
- **TensorRT INT8 PTQ** (calibrator.py): `IInt8EntropyCalibrator2` = **엔트로피(KL) 캘리브레이션**. FP32 히스토그램과 INT8 양자화 분포 간 KL divergence 최소화 임계값으로 per-tensor 스케일 결정(NVIDIA 표준). 캐시 IO는 비활성.
- **ONNX dynamic** (D1122:35): 가중치만 사전 INT8, 활성은 런타임 per-batch min/max로 동적 스케일(calibration 불필요).
- **OpenVINO** (vino_cli.py): model optimizer는 data_type 변환 위주; 본 CLI에는 POT(INT8) 캘리브레이션 직접 코드는 없음(데이터타입/벤치 중심). → 확인 가능 범위 내 OpenVINO INT8 캘리브레이션은 별도 experiments(D1201_vino_quant_op)에 분산.

---

## 5. 학습/평가 파이프라인

- **데이터셋**: ImageNet-2012(pruning train/val; README.md:82), op 테스트는 합성 난수.
- **모델 변환·벤치 명령어** (README.md:18-66):
  - `python tools.py tf2tflite --input <saved_model> --output <tflite> [--quantization=float16|dynamic]`
  - 모바일 벤치: adb push 후 `benchmark_model_plus_flex --graph=model.tflite --num_runs=50 ...` 또는 `python tools.py mobile_benchmark ...`.
- **op 스윕**: `python experiments/D1118_tflite_quant_op_test.py --model_zoo_dir <dir>` (런타임별 D11xx 스크립트).
- **DeiT pruning** (README.md:74-101): `torch.distributed.launch src/train_main.py --sparse_preset topk-hybrid-struct-layerwise-tiny --layerwise_thresholds ... --nn_pruning` → finetune(`--final_finetune`) → KD 옵션.
- **배포 대상 런타임**: TFLite(안드로이드/adb), TensorRT(GPU), ONNX Runtime, OpenVINO(CPU), TVM. (다중 런타임 횡단)

---

## 6. 의존성
- TensorFlow(>=2.3, tf.lite; utils.py:243, calibrator는 r2.3+ int8 input API), TensorRT + pycuda(calibrator.py:17-21), torch2trt(D1124:6), onnxruntime.quantization(D1122:2), OpenVINO(mo.py/benchmark_app; vino_cli), timm/torch.hub(utils.py:52-62), transformers(ViT/BERT 로더), TVM(D0104). adb(모바일). `requirements.txt`/`deit_pruning/requirements.txt`(README.md:11-15). 정확한 핀버전은 **확인 불가**(requirements 미열람).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **다중 런타임 횡단 양자화 비교**(TFLite/TRT/ONNX/OpenVINO/TVM)를 동일 op 그리드로 — 배포 타깃별 양자화 지원·속도를 한눈에.
- op-level 세밀 스윕(차원·seq_len·채널)으로 양자화 비용을 차원 함수로 측정 → 형상 의존 병목 식별.
- 실제 모바일(adb) 지연·메모리·전력(D1230) 측정 + pruning/KD 압축 축까지 포함.

**한계 / 리스크**
- **calibration이 합성 난수**(utils.py:265-269 `random.normal`, D1122 FooDataReader) → 실데이터 분포와 괴리, INT8 정확도 측정엔 부적합(지연 측정용).
- TRT 캘리브레이션 캐시 IO 비활성(calibrator.py:61-71) → 재현·반복 빌드 비효율.
- 자체 모델 구현은 참조용 표준 FP 레이어(torch_layers/attention.py 등) — **양자화 알고리즘 자체는 외부 런타임에 위임**(자체 PTQ/QAT 커널 없음).
- 날짜 코드 실험 스크립트(D11xx)는 일회성·하드코딩 경로(`/data/v-xudongwang/...` utils.py:72) 다수 → 재사용성 낮음.
- 코드 노후(TF1 layers, OpenVINO `deployment_tools` 구조는 구버전) → 최신 런타임과 불일치 가능.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적 — 추정)

- **배포 비교 벤치로서 유용**: FPGA 가속기를 양산 SW 런타임(TFLite INT8/TRT INT8/ONNX/OpenVINO)과 **동일 op 그리드로 횡단 비교**하는 방법론을 그대로 차용 가능 — 가속기 대비 baseline 수립에 적합. (experiments/D11xx 패턴)
- **op-level 양자화 비용 곡선**(Dense/Conv 차원 스윕)은 FPGA HLS의 **타일 크기·차원별 latency 모델링**과 직접 대응 — 어떤 차원에서 양자화 이득이 큰지 사전 분석.
- **INT8 PTQ calibration 방식 레퍼런스**: TFLite representative_dataset(min/max), TRT 엔트로피(KL) — FPGA용 per-tensor 스케일/zp 산출 시 어느 통계가 정확한지 비교 기준.
- XR 시선추적 직접 연관은 낮음(범용 ViT/CNN 엣지 벤치). 다만 **모바일/엣지 전력·지연 측정 인프라**(adb, D1230 전력)는 XR 헤드셋 온디바이스 추론 평가 방법론으로 참고적.

---

## 9. 근거 표기 규칙
- 모든 기술 주장은 (파일:라인) 근거. **"추정"**: §8 FPGA/XR 적용 해석.
- **확인 불가**: requirements 정확한 버전 핀, OpenVINO POT INT8 캘리브레이션 세부(본 CLI엔 미존재), tools.py 본체(Glob 미노출 시 추정 — README 인용). vendor(`nn_pruning_v1`)는 제약에 따라 미분석.
