# HBTXR_v3_0 Optimizer Pool 참조

최종 갱신: 2026-03-29 KST

Role: `최적화 시스템 아키텍트`, `학습 벤치마크 엔지니어`, `런타임 통합 리뷰어`

## 요약

이 문서는 `HBTXR_v3_0`에 추가된 현재 optimizer-pool 시스템을 설명한다.

다루는 범위:

- `src/hbtxr/optim` 아래의 flat-file optimizer 구조
- `training.optimizer`, `training.optimizer_modifiers`, `training.optimizer_pool` 설정 표면
- 실제 구현 완료된 optimizer 목록
- modifier 동작 방식
- 단일 optimizer 학습과 pool sweep 학습에서 생성되는 산출물

주 코드 기준:

- [registry.py](../../src/hbtxr/optim/registry.py)
- [modifiers.py](../../src/hbtxr/optim/modifiers.py)
- [pool.py](../../src/hbtxr/optim/pool.py)
- [common.py](../../src/hbtxr/optim/common.py)
- [trainer.py](../../src/hbtxr/training/trainer.py)
- [train_hbtxr.py](../../scripts/train_hbtxr.py)

## 1. 패키지 레이아웃

현재 optimizer 시스템은 의도적으로 평평한 구조를 가진다.

- [__init__.py](../../src/hbtxr/optim/__init__.py)
- [registry.py](../../src/hbtxr/optim/registry.py)
- [modifiers.py](../../src/hbtxr/optim/modifiers.py)
- [pool.py](../../src/hbtxr/optim/pool.py)
- [common.py](../../src/hbtxr/optim/common.py)

optimizer별 단일 파일:

- [adamw.py](../../src/hbtxr/optim/adamw.py)
- [adam_mini.py](../../src/hbtxr/optim/adam_mini.py)
- [adema_mix.py](../../src/hbtxr/optim/adema_mix.py)
- [ivon.py](../../src/hbtxr/optim/ivon.py)
- [lion.py](../../src/hbtxr/optim/lion.py)
- [mars.py](../../src/hbtxr/optim/mars.py)
- [prodigy.py](../../src/hbtxr/optim/prodigy.py)
- [adopt.py](../../src/hbtxr/optim/adopt.py)
- [sophia_g.py](../../src/hbtxr/optim/sophia_g.py)
- [soap.py](../../src/hbtxr/optim/soap.py)
- [musgd.py](../../src/hbtxr/optim/musgd.py)

## 2. 설정 표면

기본 설정은 이제 아래 세 블록을 노출한다.

```yaml
training:
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.00005
    betas: [0.9, 0.999]
    eps: 1.0e-8
    kwargs: {}

  optimizer_modifiers:
    cautious: false
    schedule_free: false

  optimizer_pool:
    enabled: false
    compare_mode: sweep
    implemented_only: true
    report_filename: optimizer_pool_report.json
    candidates:
      - name: adamw
      - name: lion
      - name: prodigy
      - name: musgd
```

참고 설정:

- [base.yaml](../../configs/base.yaml)
- [mode2_stage1_optimizer_pool.yaml](../../exps/configs/mode2_stage1_optimizer_pool.yaml)
- [mode2_stage2_optimizer_pool.yaml](../../exps/configs/mode2_stage2_optimizer_pool.yaml)

메모:

- optimizer-pool preset은 main surface가 아니라 `exps/configs/` experimental surface에 있다.

### 2.1 우선순위 규칙

학습률과 weight decay 우선순위:

1. `training.optimizer.lr`
2. `training.optimizer.weight_decay`
3. 없으면 legacy fallback
   - `training.lr`
   - `training.weight_decay`

즉 구형 preset도 동작하지만, 현재 authoritative surface는 새 optimizer block이다.

## 3. 구현 완료 optimizer

| 이름 | 상태 | 메모 |
|---|---|---|
| `adamw` | implemented | `torch.optim.AdamW`를 감싼 얇은 registry wrapper |
| `adam_mini` | implemented | 공식 Adam-mini의 blockwise/headwise update를 HBTXR에 맞게 적응 |
| `adema_mix` | implemented | 공식 PyTorch AdEMAMix 경로, `beta3`는 `kwargs`로 노출 |
| `lion` | implemented | Google Lion PyTorch reference 기반 |
| `prodigy` | implemented | 공식 Prodigy PyTorch reference 기반 |
| `adopt` | implemented | 공개 `timm` adaptation 경로 기반 |
| `ivon` | implemented | sampled-parameter training path와 `ess` 자동 추론 fallback 포함 |
| `soap` | implemented | Shampoo eigenbasis preconditioning이 포함된 공식 SOAP 경로 |
| `musgd` | implemented | 로컬 Ultralytics reference snapshot 기준 |
| `sophia_g` | implemented | `training.optimizer.kwargs.hess_interval` 기반 Hessian refresh cadence 포함 |
| `mars` | implemented | `mars_type`, `gamma` 노출. `kwargs.is_approx=false`일 때 exact multi-pass 활성화 |

## 4. Modifier

### 4.1 `cautious`

구현 위치:

- [modifiers.py](../../src/hbtxr/optim/modifiers.py)

의미:

- base optimizer를 감싼다.
- 제안된 parameter delta를 계산한다.
- 현재 gradient와 부호가 맞지 않는 update를 mask한다.

현재 상태:

- generic wrapper
- 표준 PyTorch `Optimizer` semantics를 따르는 optimizer라면 적용 가능

### 4.2 `schedule_free`

현재 상태:

- `adamw`에 대해서만 구현됨

의미:

- 외부 scheduler를 대체한다.
- optimizer 수준의 `train()` / `eval()` phase switching이 필요하다.
- 그래서 trainer는 train epoch 전에 optimizer `train()`, validation 전에 optimizer `eval()`을 명시적으로 호출한다.

결과:

- `schedule_free=true`이면 외부 scheduler 생성은 비활성화된다.

## 5. MuSGD 통합

source of truth:

- [muon.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/references/ultralytics-main/ultralytics/optim/muon.py)

HBTXR 구현 파일:

- [musgd.py](../../src/hbtxr/optim/musgd.py)

지원 group policy:

| Policy | 의미 |
|---|---|
| `auto_2d_plus` | `2D+` tensor는 Muon path, `1D/scalar/bias/norm`은 SGD path 사용 |
| `all_sgd` | 전체를 SGD path로 처리 |
| `all_muon` | 전체를 Muon path로 처리 |
| `explicit_from_param_groups` | caller가 `kwargs.param_groups`로 미리 만든 param group 사용 |

권장 기본값:

- `auto_2d_plus`

예시:

```yaml
training:
  optimizer:
    name: musgd
    lr: 0.001
    weight_decay: 0.00005
    kwargs:
      muon: 0.5
      sgd: 0.5
      momentum: 0.95
      nesterov: true
      musgd_group_policy: auto_2d_plus
```

## 6. Trainer 통합

trainer는 더 이상 `AdamW`를 하드코딩하지 않는다.

실제 흐름:

1. `training.optimizer`를 읽는다.
2. modifier를 resolve한다.
3. registry를 통해 optimizer를 생성한다.
4. 외부 scheduler 허용 여부를 결정한다.
5. optimizer metadata를 run `hypers/` 디렉터리에 기록한다.

trainer 특수 처리 경로:

- `ivon`
  - train epoch 중 `optimizer.sampled_params(train=True)`를 사용
- `sophia_g`
  - 주기적으로 `optimizer.update_hessian()` refresh 수행
- `mars`
  - 다음 조건이면 exact multi-pass future-group/current-group loop 사용
    - `training.optimizer.name=mars`
    - `training.optimizer.kwargs.is_approx=false`

Exact MARS 메모:

- HBTXR는 공식 exact 순서를 그대로 따른다.
  - future-group gradient pass
  - `update_previous_grad()`
  - current-group gradient pass
  - optimizer step
  - `update_last_grad()`
- upstream는 무한 `get_batch(...)` 루프를 쓰기 때문에, finite loader edge handling은 HBTXR 전용 보정이 들어간다.
- 현재 regression은 CPU execution 기준으로 검증했고, 전용 CUDA exact-MARS benchmark는 아직 없다.

Sophia-G AMP 메모:

- HBTXR는 이제 `GradScaler` 기반 AMP 아래에서도 Hessian-refresh pass를 허용한다.
- refresh loss를 scale한 뒤 `optimizer.update_hessian()` 전에 gradient를 manual unscale한다.
- 이때 upstream의 LM sampled-token objective가 아니라 HBTXR의 supervised `loss_total`을 유지한다.

run마다 기록되는 주요 산출물:

- `hypers/optimizer_resolved.json`
- `hypers/optimizer_diff_summary.json`

## 7. Pool sweep 통합

entry script는 이제 아래를 통해 pool sweep 실행을 지원한다.

- [train_hbtxr.py](../../scripts/train_hbtxr.py)

`training.optimizer_pool.enabled=true`일 때:

- shared config에서 candidate config를 확장한다.
- 각 candidate는 아래 경로에 학습된다.
  - `runs/<exp>/pool/<candidate>/train`
- 요약 산출물은 아래에 기록된다.
  - `optimizer_pool_report.json`
  - `optimizer_pool_summary.csv`
  - `optimizer_pool_rankings.json`

현재 동작:

- `implemented_only=true`에서도 registry에 선언된 전 optimizer가 모두 실제 구현 상태이므로 전체 세트를 유지한다.
- `best_metric_name`을 ranking metric으로 재사용한다.
- 현재 production-wired pool coverage는 다음과 같다.
  - `adamw`
  - `adam_mini`
  - `adema_mix`
  - `ivon`
  - `lion`
  - `mars`
  - `prodigy`
  - `adopt`
  - `sophia_g`
  - `soap`
  - `musgd`

Mode2 ready benchmark preset:

- [mode2_stage1_optimizer_pool.yaml](../../exps/configs/mode2_stage1_optimizer_pool.yaml)
- [mode2_stage2_optimizer_pool.yaml](../../exps/configs/mode2_stage2_optimizer_pool.yaml)

이 preset들은 optimizer-pool 표면은 그대로 두고, 데이터 / stage 계약만 다음처럼 바꾼다.

- interpolated-frame stage1 search benchmark
- materialized mode2 stage2 tracking benchmark

현재 실행 상태:

- 두 preset 모두 아래 테스트로 script-level synthetic smoke coverage를 가진다.
  - [test_optimizer_pool_v3.py](../../tests/test_optimizer_pool_v3.py)
- smoke path는 synthetic target-FPS canonical workspace를 만들고 실제 `train_hbtxr.py` pool sweep entrypoint를 실행한 뒤 다음을 검증한다.
  - `optimizer_pool_report.json` 존재
  - candidate label 정상 출력
- `exps/configs/mode2_stage2_optimizer_pool.yaml`은 여전히 일반 stage2 bootstrap을 요구한다.
  - `--init-checkpoint`로 stage1 checkpoint를 전달해야 한다.

## 8. Stage-aware 통계 필터링

trainer는 이제 stage와 무관한 통계를 아래 출력 경로에 넘기기 전에 필터링한다.

- console step summary
- `history.json`
- `history.jsonl`
- `train_log.txt`

현재 필터 규칙:

- `stage1`
  - 숨김 대상
    - `metric_event_*`
    - `metric_track_*`
    - `loss_event_*`
    - `loss_track_*`
    - `loss_consistency`
- `stage2`
  - 숨김 대상
    - `loss_eye`
    - `loss_mask`

이것은 reporting-layer filter다. supervised objective 자체를 바꾸지 않고, 터미널과 history에 stage-irrelevant noise가 섞이지 않게 한다.

## 9. Diff 문서

구현 완료 또는 계획 중인 각 optimizer는 아래에 전용 diff note를 가진다.

- [optimizers](../analysis/optimizers)

이 문서들은 명시적으로 다음을 분리한다.

- upstream에서 가져온 부분
- HBTXR 통합 과정에서 바뀐 부분
- algorithmic diff 존재 여부

## 10. 검증 상태

이번 패스에서 직접 검증한 테스트:

- [test_optimizer_pool_v3.py](../../tests/test_optimizer_pool_v3.py)
- [test_train_pipeline_v3.py](../../tests/test_train_pipeline_v3.py)
- [test_config_v3.py](../../tests/test_config_v3.py)

문서 작성 시점의 최신 regression 상태:

- focused optimizer tests: `30 passed`
- mode2 pool smoke sync 이후 focused config/console/train-pipeline regression: `11 passed`
- focused train-pipeline tests: `3 passed`
- full repository regression: `129 passed`

실제 pilot benchmark 참고:

- [12_v3_mode2_optimizer_pool_cpu_pilot.md](../exps/20260330_170419_12_v3_mode2_optimizer_pool_cpu_pilot.md)
- 이 파일럿은 synthetic CPU 데이터에서 full optimizer set에 대해 실제 script-level `mode2` pool 실행이 가능함을 확인한다.
- 해석 경계:
  - execution sanity
  - relative pilot behavior
  - 최종 real-data 또는 CUDA ranking은 아님

실제 bounded real-data pilot 참고:

- [13_v3_real_ev_eye_mode2_optimizer_pool_cpu_pilot.md](../exps/20260330_170419_13_v3_real_ev_eye_mode2_optimizer_pool_cpu_pilot.md)
- 이 파일럿은 bounded real EV-Eye subset에 대해서도 `mode2` pool surface를 실제 실행할 수 있음을 확인한다.
- 현재 해석 경계:
  - real-data plumbing 검증
  - 초기 optimizer / stability screen
  - full-dataset benchmark는 아님
  - CUDA throughput / memory benchmark도 아님

CUDA benchmark 상태:

- GPU hardware는 존재한다.
- 현재 프로젝트 `.venv`의 PyTorch는 CPU-only 상태다.
- 따라서 CUDA timing / peak-memory 수집은 하드웨어가 아니라 환경 설정에 의해 막혀 있다.

## 11. 실무 권장 읽기 순서

권장 순서:

1. [base.yaml](../../configs/base.yaml)
2. [registry.py](../../src/hbtxr/optim/registry.py)
3. [trainer.py](../../src/hbtxr/training/trainer.py)
4. [train_hbtxr.py](../../scripts/train_hbtxr.py)
5. [optimizers](../analysis/optimizers) 아래 per-optimizer diff note
