> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active
> **소유** hardware

# tools — 골든 생성

```bash
sh hardware/build/make_golden.sh          # 8벌 전부 (23 초, 20 MB)
sh hardware/build/make_golden.sh tiny-4   # 블록 하나
sh hardware/build/make_golden.sh model    # 모델 전체, 논문 비트폭
```

`export_hls_golden.py` 가 `algorithm/quantization` 정수 오라클로 골든 벡터를
`hardware/workspace/golden/` 에 냅니다. **생성물이라 git 에 없습니다** — seed 로 재현됩니다.
파일 목록과 계약은 [SPEC §9](../docs/SPEC.md).

## 두 가지 스코프

| | 무엇 | 쓰는 곳 |
|---|---|---|
| `--scope block` | 블록 1개 + 스테이지 벡터(rmu·smu·ln·softmax·gelu) | **S2~S5** |
| `--scope model` | **두 스템 · 공유 블록 스택 · 두 헤드** 전체 | **S8** |

모델 스코프는 논문 비트폭을 그대로 씁니다 — `--bits 4`(MHA·MLP) ·
`--seam-bits 4`(스템 출력·잔차 스트림) · `--head-bits 8`(final norm·pooling·헤드).
**백본은 공유**입니다: search 가 `B₁:₈`, track 이 `B₁:₄` 를 **같은 가중치로** 돕니다
([SPEC §7](../docs/SPEC.md)).

## 알아둘 것

- **stdlib 만 씁니다.** 이 머신에도 WSL 에도 numpy·torch 가 없고, `i_ops`/`i_block` 은
  바로 그래서 의존성 없이 쓰였습니다. `lut_calibrate`·`int_calibrate*`·`export*` 는
  전부 numpy 라 못 씁니다.
- **`quantization/__init__.py` 를 우회합니다.** 거기서 torch 백엔드 캘리브레이터를 즉시
  import 하기 때문에, 패키지를 거치면 의존성 없는 모듈 두 개를 쓰려고 torch 를 끌어옵니다.
- **페이로드는 캘리브레이션이 아니라 seed 데이터의 관측 범위에 맞춘 것**입니다 —
  이 레포에 체크포인트가 없습니다. V1 이 재는 것은 **정확도가 아니라 데이터패스 동일성**
  (HLS 정수 == 파이썬 정수)이라 이걸로 충분합니다. 체크포인트와 export 덤프가 생기면
  `build_block` 을 manifest 리더로 바꿉니다 — **파일 레이아웃이 계약이고 그건 안 바뀝니다.**

## 프리셋

| 블록 스코프 | 토큰 | `D` | `F` | 용도 |
|---|---:|---:|---:|---|
| `search-4` `search-8` | 64 | 192 | 768 | search 경로 크기의 블록 |
| `track-4` `track-8` | 16 | 192 | 768 | track 경로 크기의 블록 |
| `tiny-4` `tiny-8` | 4 | 24 | 48 | 개발용 — 같은 그래프, 초 단위 |

`-4` / `-8` 은 MHA·MLP matmul 폭입니다. 블록 스코프에서 **두 벌이 필요한 이유**는
[SPEC §3 혼합정밀](../docs/SPEC.md) — 블록 안에서는 오라클이 엣지별 dtype 을 표현하지 못해
GeLU LUT 입력이 4비트에 눌립니다.

| 모델 스코프 | 깊이 | cut | `D` | |
|---|---:|---:|---:|---|
| `model` | 8 | 4 | 192 | 논문 그대로. 13 초 · 9.8 MB |
| `model-tiny` | 4 | 2 | 24 | `replay_model_int` 대조가 여기서 돕니다 |

감사 스크립트는 여기 두지 않습니다.
계획: [../docs/plans/active/2026-07-31-hls-rewrite-plan.md](../docs/plans/active/2026-07-31-hls-rewrite-plan.md)
