# AdaTSQ 코드베이스 정밀 분석

> 분석 대상 repo: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\AdaTSQ`
> 작성일: 2026-06-20
> 근거 표기 규칙: **[README 근거]** = repo README.md/abstract에서 직접 확인 / **[추정]** = 분석자 추론 / **[확인 불가(코드 미공개)]** = 소스코드 부재로 검증 불가

---

## 0. 핵심 결론 (먼저 읽을 것)

- **이 repo에는 실제 양자화 소스코드가 존재하지 않는다.** repo에 들어 있는 것은 `README.md`, `LICENSE`, `.gitignore`, `figs/` (PNG 5개), `.git/` 뿐이다. **`.py` 파일은 단 한 개도 없다** (Glob `**\*.py` 결과 = "No files found"). **[README 근거 + Glob 확인]**
- README 마지막 문장에 **"Our code will be released soon."** (코드 곧 공개 예정)이 명시되어 있다. 즉 **현 시점(repo released 2026-02-10)에서는 코드 미공개 상태**이며, 본 문서의 알고리즘 분석은 전적으로 **README/논문 abstract 텍스트 기반**이다. **[README 근거]**
- 따라서 "구현 디테일", "함수/클래스 구조", "의존성", "하이퍼파라미터" 등은 모두 **확인 불가(코드 미공개)** 로 표기한다.

---

## 1. 개요

### 1.1 정체
- **이름:** AdaTSQ — "AdaTSQ: Pushing the Pareto Frontier of Diffusion Transformers via Temporal-Sensitivity Quantization" **[README 근거]**
- **저자:** Shaoqiu Zhang, Zizhong Ding, Kaicheng Yang, Junyi Wu, Xianglong Yan, Xi Li, Bingnan Duan, Jianping Fang, Yulun Zhang **[README 근거]**
- **출처:** arXiv 2026, eprint **2602.09883**, primaryClass `cs.CV`, URL `https://arxiv.org/abs/2602.09883` **[README 근거]**
- **약어:** TSQ = **T**emporal-**S**ensitivity **Q**uantization (시간(타임스텝) 민감도 기반 양자화). Ada = Adaptive(적응적) **[추정 — 명시적 약어 풀이는 README에 없음]**

### 1.2 목적
- **문제:** Diffusion Transformer(DiT)는 고품질 이미지/비디오 생성의 SOTA 백본이지만, 막대한 연산량·메모리 footprint 때문에 엣지 디바이스 배포가 어렵다. **[README 근거]**
- **기존 한계:** PTQ(Post-Training Quantization)는 LLM에서 효과가 입증됐지만, **DiT 고유의 temporal dynamics(확산 과정의 시간적 동역학)를 무시**하고 기존 방법을 그대로 적용하면 성능이 suboptimal하다. **[README 근거]**
- **목표:** DiT의 **temporal sensitivity(타임스텝별 민감도)** 를 활용해 효율-품질의 **Pareto frontier를 전진**시키는 새로운 PTQ 프레임워크. **[README 근거]**

### 1.3 핵심 아이디어 (2개 축)
1. **Pareto-aware timestep-dynamic bit-width allocation** — 양자화 정책 탐색을 **constrained pathfinding(제약 경로탐색) 문제로 모델링**하고, **end-to-end reconstruction error로 가이드되는 beam search**로 timestep별·layer별 비트폭을 동적으로 할당. **[README 근거]**
2. **Fisher-guided temporal calibration** — **temporal Fisher information**으로 민감한 timestep의 calibration data를 우선시하고, **Hessian 기반 weight 최적화**와 매끄럽게 통합. **[README 근거]**

---

## 2. 디렉토리 구조 (현재 repo 실측)

Glob `...\AdaTSQ\*` 및 `...\AdaTSQ\**\*.py` 실행 결과(`.git/`, `__pycache__` 제외):

```
AdaTSQ/
├── README.md          # 논문 제목/저자/abstract/결과 figure 링크/BibTeX
├── LICENSE            # 라이선스 파일
├── .gitignore         # 표준 Python .gitignore (실제 .py는 없음 — 미래 코드 공개 대비)
└── figs/
    ├── method.png     # (이름만) 제안 방법 개요도 — abstract 직후 삽입
    ├── table.png      # (이름만) Table 1·2 정량 비교
    ├── fig1.png       # (이름만) Figure 1 시각 비교
    ├── fig6.png       # (이름만) Figure 6 시각 비교
    └── fig7.png       # (이름만) Figure 7 시각 비교
```

- **실제 양자화 소스코드(.py 등)는 미공개.** `.py` 파일 0개 (Glob `**\*.py` = "No files found"). **[Glob 확인]**
- `.gitignore`는 표준 Python 템플릿(`__pycache__/`, `*.py[codz]`, build/dist, venv, mypy/ruff 캐시 등)을 포함하고 있으나, 정작 추적되는 `.py` 파일이 없으므로 **미래 코드 공개를 대비한 placeholder**로 보인다. **[추정]**
- README의 figure 참조: `figs/method.png`(방법 개요), `figs/table.png`(Table 1·2 정량), `figs/fig1.png`/`fig6.png`/`fig7.png`(시각 비교). **figure 내부 픽셀 내용은 본 분석에서 열지 않았으므로 수치는 확인 불가.** **[README 근거 / 내용 확인 불가]**

---

## 3. 핵심 알고리즘 정밀 분석 (README/abstract 기반)

> 주의: 아래 모든 알고리즘 서술은 **코드가 아닌 abstract 텍스트의 재구성**이다. 실제 구현 디테일(루프 구조, tensor shape, 양자화 granularity, per-channel/per-tensor 여부 등)은 **확인 불가(코드 미공개)**.

### 3.1 Pareto-aware timestep-dynamic bit-width allocation

**목표:** Diffusion sampling은 다수의 timestep(예: t = T, …, 1)을 거치는데, 각 timestep에서 DiT의 layer별 양자화 민감도가 다르다. 모든 timestep·모든 layer에 균일한 비트폭을 쓰는 대신, **(timestep × layer) 격자 위에서 비트폭을 동적으로 배분**하여 동일 평균 비트 예산에서 품질을 최대화한다. **[README 근거 + 추정]**

**모델링 — constrained pathfinding:** **[README 근거]**
- 정책 탐색(어느 (timestep, layer)에 몇 bit를 줄지)을 **제약 조건이 있는 경로탐색 문제**로 정식화.
- "경로(path)" = timestep 축을 따라가며 각 단계에서 layer별 비트폭 결정을 누적하는 시퀀스. **[추정]**
- "제약(constraint)" = 전체 비트 예산(또는 모델 크기/연산량 상한) — 즉 평균 비트폭/메모리/BitOPs 예산 안에서만 탐색. **[추정]**

**탐색 알고리즘 — beam search:** **[README 근거]**
- 탐색 공간(timestep×layer×후보비트폭)이 지수적으로 크므로 완전탐색이 불가 → **beam search**로 폭 B만큼 유망 후보만 유지하며 전진.
- **가이드 신호 = end-to-end reconstruction error** (최종 출력단의 재구성 오차). 즉 layer-local 오차가 아니라 **전체 파이프라인 출력 품질**로 비트 배분을 평가. **[README 근거]**
- 결과: timestep마다 layer-wise 비트폭이 **다르게** 할당되는 동적 정책. **[README 근거]**

**Pareto-aware의 의미:** 단일 동작점이 아니라 **(효율 vs 품질) Pareto frontier 전체를 밀어올리는** 방향으로 탐색 — 다양한 비트 예산에서 일관되게 우월한 동작점 집합을 찾는다. **[README 근거 + 추정]**

### 3.2 Fisher-guided temporal calibration

**목표:** PTQ는 소량의 calibration data로 양자화 파라미터(scale/zero-point)와 weight rounding을 보정한다. AdaTSQ는 **어느 timestep의 calibration 샘플이 더 중요한지**를 판단해 우선 사용한다. **[README 근거]**

**메커니즘:** **[README 근거]**
1. **Temporal Fisher information** 계산 — 각 timestep이 출력 품질(또는 손실)에 미치는 민감도를 Fisher information으로 측정. Fisher가 큰 timestep = 더 민감 → calibration에서 우선.
2. **민감 timestep 우선 calibration** — 민감한 timestep의 데이터/통계를 더 비중 있게 반영하여 양자화 보정.
3. **Hessian 기반 weight 최적화와 통합** — Hessian 정보를 이용한 weight rounding/보정(GPTQ/BRECQ류 계열)과 **seamless하게 결합**하여 weight를 양자화 친화적으로 조정. **[README 근거 — "Hessian-based weight optimization" 명시]**

**왜 Fisher인가:** Fisher information matrix는 loss curvature의 근사(대각 또는 empirical Fisher)로, **2차 민감도**를 비교적 싸게 추정 가능 → timestep별 중요도 가중에 적합. **[추정]**

### 3.3 두 축의 결합 (전체 파이프라인 추정)

**[추정 — abstract에 명시적 순서/통합 알고리즘 의사코드는 없음]**
1. 캘리브레이션 입력 준비 → timestep별 temporal Fisher information 산출.
2. Fisher 가중 calibration + Hessian 기반 weight 최적화로 각 (timestep, layer, 후보비트폭) 조합의 양자화 결과 준비.
3. constrained pathfinding으로 정식화한 비트 배분 문제를 beam search로 풀어, end-to-end reconstruction error를 최소화하는 timestep-dynamic 비트폭 정책 확정.
4. 확정된 정책으로 최종 양자화 모델 산출 → Pareto frontier 상의 동작점 생성.

---

## 4. 알고리즘 / 수식 (재구성)

> 아래 수식은 abstract의 키워드(beam search, Fisher information, Hessian, reconstruction error)를 표준 정의로 재구성한 것이며, **논문 본문/코드의 실제 수식과 다를 수 있다.** **[추정]**

### 4.1 Timestep-dynamic 비트 배분 (constrained pathfinding + beam search)

비트폭 정책을 $\mathbf{b} = \{ b_{t,\ell} \}$ ($t$ = timestep, $\ell$ = layer, $b_{t,\ell}$ = 할당 비트)로 두면, 최적화 문제는 대략:

$$
\min_{\mathbf{b}} \; \mathcal{E}_{\text{recon}}\big(\hat{x}_0(\mathbf{b}),\, x_0\big)
\quad \text{s.t.} \quad \frac{1}{|T||L|}\sum_{t,\ell} b_{t,\ell} \;\le\; B_{\text{budget}}
$$

- $\mathcal{E}_{\text{recon}}$ : **end-to-end reconstruction error** (최종 생성 출력 $\hat{x}_0$ vs FP 기준 출력 $x_0$). **[README 근거 — guidance 신호]**
- 제약 $B_{\text{budget}}$ : 평균 비트 예산(또는 메모리/BitOPs 상한). **[추정]**
- **constrained pathfinding:** timestep 축을 노드열로 보고, 각 단계의 layer-wise 비트 결정을 간선으로 하여 예산 제약 하 최소비용 경로를 탐색. **[README 근거(정식화) + 추정(그래프 구조)]**

**Beam search 절차 (의사코드, 추정):**
```
beam = [empty_policy]                         # 폭 B 유지
for t in timesteps (T..1):                    # path를 따라 전진
    cand = []
    for partial in beam:
        for bit_assignment of layers at t:    # layer-wise 비트 후보 확장
            if budget_ok(partial + assignment):
                e = end_to_end_recon_error(partial + assignment)
                cand.append((partial+assignment, e))
    beam = top_B(cand, key=recon_error)       # 오차 낮은 B개만 유지
return best(beam)
```
**[추정 — abstract에 의사코드 없음]**

### 4.2 Temporal Fisher information (timestep 민감도)

각 timestep $t$의 민감도를 (대각/empirical) Fisher로 측정:

$$
F_t \;=\; \mathbb{E}\!\left[ \left( \frac{\partial \mathcal{L}}{\partial \theta}\Big|_t \right)^{\!2} \right]
\quad\Rightarrow\quad
w_t \;\propto\; F_t
$$

- $F_t$ 큰 timestep = 손실에 민감 → calibration 가중 $w_t$ 상향. **[README 근거(개념) + 추정(정의식)]**

### 4.3 Hessian-aware weight rounding (GPTQ류 통합)

layer 출력 오차의 2차 근사:

$$
\Delta \mathcal{L} \;\approx\; \tfrac{1}{2}\,\Delta w^{\top} H \,\Delta w,
\qquad H \approx \mathbb{E}[\, x x^{\top} \,]
$$

- Hessian $H$(또는 입력 공분산 근사)를 이용해 양자화 오차 $\Delta w$의 영향을 최소화하는 방향으로 weight를 보정/rounding. **[README 근거 — "Hessian-based weight optimization"]**
- temporal Fisher 가중과 결합되어 **민감 timestep의 통계가 $H$ 추정에 더 크게 반영**되도록 통합. **[추정]**

---

## 5. 평가

> figure/table 픽셀 내용은 열지 않았으므로 구체 수치는 **확인 불가**. 아래는 README 텍스트로 확인되는 범위. **[README 근거]**

### 5.1 대상 모델 (4개 advanced DiT)
- **Flux-Dev**
- **Flux-Schnell**
- **Z-Image**
- **Wan2.1** (비디오 생성 DiT 계열) **[추정 — Wan2.1은 영상 생성 모델]**

(모두 README abstract에 명시) **[README 근거]**

### 5.2 비교 대상 (SOTA baselines)
- **SVDQuant**
- **ViDiT-Q**

AdaTSQ가 이들을 "significantly outperforms" 한다고 주장. **[README 근거]**

### 5.3 결과 자료 (figure 파일)
- `figs/table.png` — Table 1·2 정량 비교(main paper). 수치 내용 **확인 불가**.
- `figs/fig1.png`, `figs/fig6.png`, `figs/fig7.png` — 시각(qualitative) 비교(main paper).
- `figs/method.png` — 방법 개요도.
- 정량 지표 종류(FID, IS, CLIP score, LPIPS, 비트폭, 속도/메모리 등)는 **확인 불가(figure 미열람 + 코드 미공개)**. **[확인 불가]**

---

## 6. 의존성

- **확인 불가 (코드 미공개).** `requirements.txt`/`environment.yml`/`setup.py`/`pyproject.toml` 등 의존성 정의 파일이 repo에 **존재하지 않음**.
- DiT PTQ 프레임워크이고 대상이 Flux/Z-Image/Wan2.1인 점으로 보아 **PyTorch + diffusers + 각 모델별 추론 코드** 의존이 예상되나 **모두 추정**이며 검증 불가. **[추정 / 확인 불가]**

---

## 7. 강점 / 한계 / 리스크

### 7.1 강점 (주장 기준)
- **Temporal 축을 양자화 정책에 1급 시민으로 도입** — LLM PTQ를 DiT에 그대로 옮길 때 놓치던 timestep 동역학을 직접 모델링. **[README 근거]**
- **end-to-end reconstruction error로 가이드** — layer-local proxy가 아닌 최종 출력 품질로 비트 배분 → mixed-precision 결정의 정합성이 높을 가능성. **[README 근거 + 추정]**
- **Pareto-aware** — 단일 동작점이 아니라 frontier 전체를 밀어올린다는 설계로 다양한 예산에 대응. **[README 근거]**
- **Fisher + Hessian 결합** — calibration 데이터 선택(Fisher)과 weight 최적화(Hessian)를 분리하지 않고 통합. **[README 근거]**

### 7.2 한계 / 리스크
- **(최대 리스크) 코드 미공개.** "released soon" 상태로, 현재 재현·검증·통합이 **전부 불가능**. 본 분석은 abstract 텍스트 신뢰에 의존. **[README 근거 — 최우선 리스크]**
- **비용:** beam search × (timestep×layer×비트후보) 탐색은 PTQ치고 무거울 수 있음. beam 폭/후보 수에 따른 탐색 시간·메모리 비용 **확인 불가**. **[추정]**
- **end-to-end recon error 평가 비용:** 각 후보마다 전체 sampling을 돌려 출력 오차를 봐야 한다면 매우 비쌀 수 있음(부분 평가/대리지표 사용 여부 **확인 불가**). **[추정]**
- **검증 공백:** Table 수치, ablation, 하드웨어 실측 속도/메모리 이득 모두 **확인 불가**. arXiv ID(2602.09883)도 미래 날짜 형식이라 외부 교차검증 필요. **[추정]**
- **granularity/형식 미상:** weight/activation 비트, per-channel/tensor, 정수/부동소수 여부 등 핵심 양자화 스펙 **확인 불가**. **[확인 불가]**

---

## 8. 우리 프로젝트(PRJXR-HBTXR) 관점 시사점

> 우리 프로젝트는 **ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적(eye-tracking)** 방향으로 추정됨(상위 디렉토리 구조 기반). **[추정]** 아래는 AdaTSQ 아이디어를 우리 맥락으로 매핑한 시사점.

1. **Timestep-dynamic 비트 배분 → FPGA 레이어/단계별 비트 배분 탐색에 직결.**
   - AdaTSQ의 "(timestep × layer) 격자 위 비트 배분"은, 우리 가속기에서 "(파이프라인 stage × layer) 비트 배분" 또는 "(프레임/시점 × layer) 비트 배분" 문제로 자연스럽게 환산된다. constrained pathfinding 정식화는 **하드웨어 자원 예산(DSP/BRAM/LUT) 제약 하 mixed-precision 배분**과 동형. **[추정]**

2. **Beam search 비트폭 탐색 → mixed-precision 가속기 설계공간 탐색(DSE).**
   - 지수적 비트 조합 공간을 beam search로 줄이는 방식은 HLS/RTL **DSE(pragma/bitwidth/parallelism 조합 탐색)** 와 직접 호환. end-to-end 품질 가이드를 **Pareto(latency vs accuracy, area vs accuracy)** 탐색의 목적함수로 차용 가능. **[추정]**

3. **Fisher 민감도 → 어느 레이어를 고비트로 둘지 결정하는 HW 가이드.**
   - temporal Fisher처럼 **레이어/헤드별 민감도**를 측정하면, FPGA에서 **민감 레이어만 고비트(또는 별도 PE)**, 둔감 레이어는 저비트로 두는 비균일 PE 설계 근거가 된다. 시선추적은 latency 예산이 빡빡하므로 "민감한 곳에만 비트 투자"는 특히 유효. **[추정]**

4. **XR 시선추적 특화 주의점:**
   - AdaTSQ의 "temporal"은 **확산 sampling 타임스텝**이지 우리 의미의 "프레임 시계열"이 아니다. 차용 시 **개념 매핑(diffusion step ↔ video frame/시점)을 명확히** 해야 오용을 피한다. **[추정 — 주의]**
   - 그리고 AdaTSQ는 **DiT 생성 모델**용 PTQ로, **discriminative ViT(시선추적 백본)**와 손실/평가지표가 다르다. reconstruction error 대신 시선좌표 회귀오차/검출 정확도를 가이드 신호로 치환 필요. **[추정 — 주의]**

5. **즉시 활용 vs 보류:**
   - **개념(아이디어):** 단계별 mixed-precision + 민감도 가중 calibration은 즉시 설계 원리로 차용 가능. **[추정]**
   - **구현(코드):** **코드 미공개로 직접 재사용 불가.** 공개 시점까지 SVDQuant/ViDiT-Q 등 **공개된 baseline 코드로 대체 검토**가 현실적. **[README 근거 + 추정]**

---

## 9. 근거 표기 요약

| 항목 | 근거 수준 |
|---|---|
| repo에 `.py` 코드 없음 (README/LICENSE/figs/.git만) | **사실 — Glob 실측** |
| "Our code will be released soon" (코드 미공개) | **사실 — README 근거** |
| 2축(비트 배분 + Fisher calibration) 존재, beam search, end-to-end recon error, temporal Fisher, Hessian 최적화 | **사실 — README abstract 근거** |
| 대상 모델(Flux-Dev/Schnell, Z-Image, Wan2.1), 비교(SVDQuant, ViDiT-Q) | **사실 — README abstract 근거** |
| 수식, 의사코드, 파이프라인 순서, 그래프 구조, 예산 정의 | **추정 — abstract 키워드 재구성** |
| 구현 디테일, 의존성, granularity, 하이퍼파라미터, Table 수치 | **확인 불가 — 코드/figure 미열람·미공개** |
| 우리 프로젝트 정체 및 매핑 시사점 | **추정** |
