# Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280 (Transformer-Accel) — Cross-Ref

> **본 항목은 기존 정밀 분석을 그대로 참조한다.**
> 정밀 분석 원본: [`REF/Analysis/ViT-Accelerator/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280.md`](../ViT-Accelerator/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280.md)
> 대상 repo: `REF/Transformer-Accel/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280`
> 작성 기준: 실제 소스 라인 대조. third_party(softfloat/ramulator/fpnew 등)는 vendor이므로 분석 제외(이름만).

---

## 동일 여부: **동일 (identical, 같은 repo의 중복 배치)**

`REF/ViT-Accelerator/`와 `REF/Transformer-Accel/`에 동일한 Tiny-GPT-on-Vortex repo가 중복 배치되어 있다. 자체 핵심 코드(Vortex GPGPU 위의 tiny-GPT OpenCL 커널/호스트/학습 스크립트)가 라인 단위로 일치함을 확인했다.

### 대조 근거 (자체 핵심 = `tests/opencl/tinygptv1`·`tinygptv2`)

| 항목 | 기존 분석(ViT-Accelerator) | 대상 repo(Transformer-Accel) | 판정 |
|---|---|---|---|
| 디렉토리 구조 | `tinygptv1`/`tinygptv2` (각 kernel.cl, main.cc, Makefile, scripts/train_model.py, .npy 5개) | 동일 | 동일 |
| `tinygptv1/kernel.cl` | 37줄, 단일 `matvec2layer`, phase 0/1, tanh 수동 `(1-e^{-2a})/(1+e^{-2a})` | 37줄, `matvec2layer` L1-12, phase 0/1, tanh L24-25 동일 | 동일 |
| `tinygptv2/kernel.cl` | 172줄, `tinygpt_persist_fused`(persistent) + `ffn_to_logits_slice`(멀티WG), `dot_row_vec4` float4, `D_MAX 64`/`VOCAB_MAX 128` | 172줄, persist L46-115 / slice L118-171, float4 L10-25, 매크로 L4-5 동일 | 동일 |
| 모델 정의 | 2-layer MLP(embed→tanh hidden→logits), 어텐션/양자화 없음, FP32 | 동일(scripts/train_model.py 동봉) | 동일 |

### Vortex 베이스 (vendor, 미수정으로 확인)
- `hw/rtl/{core,fpu,cache,mem}`, `runtime/*`, `sim/*` 모두 upstream Vortex 그대로(기존 분석과 동일 결론). third_party(`fpnew`, `ramulator`, `softfloat`, `openc910`/`opene906`)는 vendor — 분석 제외, 이름만 명시.

---

## 차이점

- **내용·코드 차이 없음.** 두 경로의 자체 핵심 소스는 동일하다(중복 배치). 위치(폴더명)만 `ViT-Accelerator` → `Transformer-Accel`로 다르다.
- 분류상으로는 "GPT/Transformer"라는 명칭에 맞춰 `Transformer-Accel`에 배치된 것이나, 실제 모델은 어텐션·layernorm·KV-cache가 없는 **2-layer MLP**이며 양자화도 없는 FP32다(기존 분석 §3.0·§4.5 근거 유지). 따라서 명칭과 달리 진짜 Transformer 가속기는 아니다.

---

## 결론

대상 repo는 기존 분석 대상과 **완전히 동일한 중복본**이므로 신규 정밀 분석을 생성하지 않고 기존 .md를 참조한다. 재사용 시사점(persistent-kernel 루프, 출력차원 슬라이싱, 온칩 float4 MAC, end-to-end 골든 검증)도 기존 분석 §9를 그대로 따른다.
