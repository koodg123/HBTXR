# postcalibration4quantization (Quantization Post Calibration, QPC) 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\postcalibration4quantization`
> 분석 방식: README.md, `main.py`, `utils.py`, `requirements.txt` 전체 라인 정독.
> **정체**: ONNX QDQ(Quantize-Dequantize) 모델에 대한 **사후 재보정(post-calibration)** 도구. 저자 = Tairen Piao(Nota AI, README:55). 커널 언어 = **순수 Python(ONNX graph 조작)**, CUDA/Triton/PyTorch 커널 없음.

---

## 1. 개요 (목적/원논문/핵심 아이디어)

- **목적**: 이미 PTQ로 양자화된 ONNX QDQ 모델의 정확도를 **재학습/재calibration 데이터 없이** 회복. scale을 미세 조정하여 양자화 오차(SQNR)를 개선.
- **핵심 아이디어(QPC)**(README:7):
  1. QDQ 모델의 각 레이어를 **수치 범위(output range)** 기준으로 랭킹.
  2. 상위 N개 레이어 선택.
  3. **reduction ratio(<1)**를 곱해 FP32 min/max를 좁힘.
  4. 좁아진 범위로 scale/zero-point 재계산 → 새 QDQ 모델 생성.
- **원논문**: 별도 논문 미명시. Nota AI 사내 실용 도구로 **추정**(저자 이메일 nota.ai). 정식 논문 ID **확인 불가**.
- **분류**: 앞 3개 repo(어텐션/ViT 양자화)와 달리, **양자화 후처리(scale 보정) 일반 도구**. CNN 계열 ONNX(MobileNetV2, ResNet18, MixNet)로 검증.

---

## 2. 디렉토리 구조 (자체 + 제외)

```
postcalibration4quantization/
├── README.md                  # QPC 방법 설명(자체)
├── main.py                    # ★ QPC 본체 — 랭킹/scale 재계산/QDQ 수정/SQNR 비교
├── utils.py                   # ★ inference / sqnr / find_next_nodes
├── requirements.txt           # onnx, onnxruntime, onnxoptimizer 등
├── .gitlab-ci.yml             # CI(QPC 통과 여부 자동 판정)
└── test_models/               # 검증용 ONNX(이름만, 대용량 제외)
    ├── mobilenetv2-7/{*.onnx, *_qdq.onnx}
    ├── resnet18-classification/{*.onnx, *_qdq.onnx}
    └── mixnet-s-classification/{*.onnx, *_qdq.onnx}
─ 제외: .git/, test_models/*.onnx(체크포인트, 이름만 참조)
```

- 커스텀 HW 커널 **없음**. ONNX 그래프 API(`onnx`, `onnx.helper`, `onnx.numpy_helper`)와 `onnxruntime` 추론, `onnxoptimizer`만 사용.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 레이어 범위 랭킹 — `main.py::compute_layer_range_df` (59-87)

```python
for node in model.graph.node:
    if node.op_type in ("QuantizeLinear","DequantizeLinear"): continue   # (62-63) Q/DQ 노드 자체는 스킵
    next_node      = find_next_nodes(model, node.name)        # → 뒤따르는 QuantizeLinear (65)
    next_next_node = find_next_nodes(model, next_node[0].name)# → 그 뒤 DequantizeLinear (66)
    output_scale = to_array(initializer[next_node[0].input[1]]).item()  # Q노드의 scale (68-71)
    layer_range = output_scale * 255                          # (74) 8bit → range = scale·255
    layer_info.append({name, op_type, output_range=layer_range, output_scale})
df = df.sort_values("output_range", ascending=False)         # (85) 범위 큰 순 정렬
```
- 각 연산 노드의 **출력 Q scale × 255**를 "output range"로 보고 내림차순 정렬. 범위가 큰(=양자화 손실 위험 큰) 레이어를 우선순위로.

### 3.2 scale/zero-point 재계산 — `main.py::modify_node` (13-57)

```python
output_scale = ...; output_zp = ...                  # 기존 Q/DQ 노드의 scale/zp (17-18)
fp32_min = (-128 - output_zp) * output_scale          # 현재 양자화가 커버하는 FP 범위 복원 (20)
fp32_max = ( 127 - output_zp) * output_scale          # (21)
new_fp32_min = qpc_config[2] * fp32_min               # reduction ratio(예: 0.95) 곱 (23)
new_fp32_max = qpc_config[3] * fp32_max               # (24)
new_layer_range = new_fp32_max - new_fp32_min         # (26)
new_scale = new_layer_range / 255                     # 8bit 재calibration (27)
new_zp    = round(-128 - new_fp32_min/new_scale).int8 # 새 zero-point (28)
# 새 scale/zp initializer를 그래프에 추가, 노드 input[1]/input[2] 교체 (33-57, 112-116)
```
- **핵심 수식**: 기존 [-128,127] INT8 범위가 표현하던 FP32 동적범위를 복원 → ratio(0.95~0.99)로 **양끝을 잘라 좁힌 뒤** scale을 다시 작게 만듦. clipping range를 축소해 inlier 해상도를 높이는 전략.
- Q노드와 DQ노드 둘 다 동일하게 수정(`post_calibration_onnx`:108-116) — QDQ 쌍 일관성 유지.

### 3.3 선택·적용 파이프라인 — `main.py::post_calibration_onnx` (89-125)
- `threshold_scale = layer_info_df.iloc[qpc_config[0]]["output_scale"]`(100): 상위 N(=`qpc_config[0]`)번째 레이어의 scale을 임계값으로.
- `filtered_names = {scale > threshold인 레이어}`(101): 상위 레이어만 보정 대상.
- 대상 노드의 다음 Q, 그 다음 DQ를 `modify_node`로 재calibration(104-116).
- `onnxoptimizer.optimize(["eliminate_unused_initializer"])`로 정리 후 저장(122-124). 출력 파일명 `{N}_{ratio_min}_{ratio_max}.onnx`(119).

### 3.4 성능 비교(SQNR) — `main.py::compare_performance` (127-144) + `utils.py`
- `inference`(utils.py:5-21): onnxruntime로 더미 가우시안 입력 추론.
- `sqnr`(utils.py:23-29): `SQNR = 10·log10(var(original)/var(error))` — FP32 출력 대비 양자화 출력의 신호대잡음비.
- 비교(130-143): `FP32 vs INT8 qdq`(원본) SQNR을 기준으로, 각 QPC 모델의 SQNR이 **원본보다 높으면 pass**(141).
- `find_next_nodes`(utils.py:31-50): 그래프에서 target 노드 출력을 입력으로 받는 후속 노드 탐색(QDQ 체인 추적용).

### 3.5 main & CI — `main.py::main` (146-202)
- `qpc_configs`(156-181): `[top_N, "min_max", ratio_min, ratio_max]` 24개 조합 그리드(예: `[10,"min_max",0.95,0.95]` ~ `[25,...,0.97]`). N과 ratio를 sweep.
- 각 config로 QPC 모델 생성 → `compare_performance` → **1개라도 SQNR 개선 시 exit(0), 전부 실패 시 exit(1)**(197-202). → CI(.gitlab-ci.yml)에서 회귀 테스트로 활용.
- 사용법(README:48-53):
  ```bash
  python main.py --fp32_model_path .../mobilenetv2-7.onnx \
                 --quantized_model_path .../mobilenetv2-7_qdq.onnx \
                 --output_path ./output
  ```

---

## 4. 알고리즘 / 수식 (post-calibration 보정)

**QPC 보정 절차**:
1. 레이어 i의 현재 양자화 동적범위 복원:
   `fp32_min_i = (-128 - zp_i)·s_i`, `fp32_max_i = (127 - zp_i)·s_i`.
2. 범위가 큰 상위 N개 레이어 선택(`output_range = s_i·255` 내림차순, threshold = N번째 scale).
3. clipping 범위 축소(ratio r∈[0.95,0.99]):
   `min' = r·fp32_min_i`, `max' = r·fp32_max_i`.
4. scale/zp 재계산:
   `s_i' = (max' - min')/255`, `zp_i' = round(-128 - min'/s_i')`.
5. 해당 레이어의 Q·DQ 노드 scale/zp 교체.
6. SQNR로 검증, 개선 시 채택.

**직관**: 큰 동적범위 레이어는 outlier 때문에 scale이 커서 대다수 inlier의 양자화 해상도가 낮다. 양끝을 r배로 잘라 outlier를 saturate시키고 scale을 줄이면 inlier 해상도↑ → 전체 SQNR↑. **추가 데이터·학습 없이 그래프 상수만 수정**하는 zero-shot post-calibration.

---

## 5. 학습/평가 파이프라인 (데이터셋/벤치/명령어)

- **학습 없음**. 입력 데이터 = 랜덤 가우시안 더미(`utils.py`:14) — 정확도가 아니라 **FP32 vs INT8 출력 SQNR**로 보정 효과 측정.
- 검증 모델: MobileNetV2-7, ResNet18, MixNet-S(ImageNet 분류 ONNX, `test_models/`). 각 `.onnx`(FP32)와 `_qdq.onnx`(QDQ INT8) 쌍.
- 실행: `python main.py --fp32_model_path ... --quantized_model_path ..._qdq.onnx --output_path ./output`.
- CI: `.gitlab-ci.yml` — 24개 config 중 1개라도 SQNR 개선하면 통과.

---

## 6. 의존성

- `onnx==1.19.1`, `onnxruntime==1.19.0`, `onnxoptimizer==0.3.13`, `numpy==2.3.5`, `pandas==2.2.2`, `protobuf` 등(requirements.txt). Python ≥ 3.11(README:13).
- **PyTorch/CUDA/Triton 불필요** — 순수 ONNX 그래프 조작 + CPU 추론.

---

## 7. 강점 / 한계 / 리스크

**강점**
- 재학습·calibration 데이터 없이 그래프 상수(scale/zp)만 수정 → 매우 가벼운 zero-shot 정확도 회복. 어떤 QDQ ONNX에도 적용 가능(모델 비종속).
- SQNR 기반 자동 검증 + CI 통합 → 회귀 안전.
- 레이어 범위 랭킹으로 영향 큰 레이어만 선택적 보정(최소 변경).

**한계 / 리스크**
- **랜덤 더미 입력 기반 SQNR**(utils.py:14) — 실데이터 분포 미반영. 실제 top-1 정확도 개선과 SQNR 개선이 항상 일치한다는 보장 없음.
- ratio·N을 그리드 sweep(24조합)으로 brute-force — 자동 최적 탐색 아님. 8bit([-128,127])·QDQ 전제 하드코딩(modify_node:20-28).
- per-tensor scale만 다룸(per-channel weight scale 처리 미확인). 비선형/어텐션 특화 로직 없음(CNN 분류 위주).
- SQNR 개선이 없으면 그대로 fail — 개선 폭 보장 없음.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적)

- **저비트 정확도 회복의 직접 도구**: HG-PIPE류 FPGA 가속기는 저비트(INT8/이하) 양자화가 필수인데, 본 QPC는 **재학습 없이 scale만 미세조정**해 정확도를 회복 → FPGA 배포 직전 후처리 단계로 바로 활용 가능. clipping 범위 축소(ratio)는 FPGA의 고정 비트폭에서 inlier 해상도를 극대화하는 전형적 기법.
- **레이어별 민감도 랭킹 → mixed precision 연계**: output range 랭킹은 어느 레이어가 양자화에 민감한지 식별 — 이는 (3번 repo의) mixed precision 비트 할당과 결합해 "민감 레이어는 고비트 + QPC 보정"이라는 통합 전략으로 확장 가능.
- **scale/zp 재계산식의 HW 의미**: `s' = range'/255`, `zp' = round(-128 - min'/s')`는 FPGA requantization 유닛의 상수(scale·zero-point)를 오프라인에서 재튜닝하는 것과 동일 — RTL 변경 없이 calibration LUT/레지스터 값만 갱신하면 됨.
- **XR 시선추적**: XR eye-tracking 모델을 FPGA에 올릴 때, calibration 데이터 확보가 어려운 상황(개인화·온디바이스)에서 **데이터 없는 zero-shot 보정**은 실용적 가치가 큼. 단 실데이터 SQNR이 아닌 더미 기반이므로, 시선추적 실데이터로 검증 절차를 보강해야 함.

---

## 9. 근거 표기 / 불명확 사항

- **정체**: ONNX QDQ post-calibration(scale 재보정) 도구, 저자 Tairen Piao(Nota AI, README:55) — **확인**. CUDA/Triton/PyTorch 커널 **없음(확인)**, 순수 ONNX graph 조작.
- 정식 논문 ID **확인 불가**(README에 논문 인용 없음). Nota AI 사내 도구로 **추정**.
- 4개 repo 중 유일하게 **어텐션/ViT 비특화** — CNN ONNX(MobileNet/ResNet/MixNet) 검증. ViT-Quantization 폴더에 속하나 적용 대상은 일반 QDQ 모델.
- 정확도(top-1) 영향은 미측정(SQNR만) — 실모델 정확도 효과는 **확인 불가**.
- per-channel/weight scale 처리 여부는 코드상 per-tensor activation Q/DQ만 확인됨(나머지 **확인 불가**).
