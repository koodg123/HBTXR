> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active
> **소유** hardware

# tools — 골든 생성

```bash
sh hardware/build/make_golden.sh          # 프리셋 6벌 전부
sh hardware/build/make_golden.sh tiny-4   # 하나만
```

`export_hls_golden.py` 가 `algorithm/quantization` 정수 오라클로 스테이지별 골든 벡터를
`hardware/workspace/golden/<mode>-a<bits>/` 에 냅니다. **생성물이라 git 에 없습니다** —
seed 로 재현됩니다. 파일 목록과 계약은 [SPEC §9](../docs/SPEC.md).

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

| | 토큰 | `D` | `F` | 용도 |
|---|---:|---:|---:|---|
| `search` | 64 | 192 | 768 | 논문 search 경로 |
| `track` | 16 | 192 | 768 | 논문 track 경로 |
| `tiny` | 4 | 24 | 48 | 개발용 — 같은 그래프, 초 단위 |

`-4` / `-8` 은 MHA·MLP matmul 폭입니다. **두 벌이 필요한 이유**는
[SPEC §3 혼합정밀](../docs/SPEC.md) — 현재 오라클이 엣지별 dtype 을 표현하지 못합니다.

감사 스크립트는 여기 두지 않습니다.
계획: [../docs/plans/active/2026-07-31-hls-rewrite-plan.md](../docs/plans/active/2026-07-31-hls-rewrite-plan.md)
