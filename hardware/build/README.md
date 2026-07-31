> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active
> **소유** hardware

# build — 골든 생성 · 검증

```bash
sh hardware/build/make_golden.sh      # 정수 골든 (생성물, git 에 없음)
sh hardware/build/run_tb.sh           # V1 — 모든 테스트벤치
```

`run_tb.sh` 는 골든이 없으면 **알아서 만듭니다.**

## V1 은 `vitis_hls` 가 필요 없습니다

`hls::stream` 과 `hls::vector` 는 **plain g++ 에서 컴파일·실행됩니다** — Vitis 헤더만
있으면 됩니다(`XILINX_INCLUDE`, 기본값 `/tools/Xilinx/Vitis_HLS/2023.2/include`, WSL).
라이선스도 프로젝트도 필요 없습니다. csim·csynth(V2·V3)는 S8·S9 에서 `hls/` tcl 로 옵니다.

경고 설정에 이유가 있습니다:

| | |
|---|---|
| `-isystem` (not `-I`) | Vitis 헤더가 자기 `-Wall` 잡음을 냅니다. `-I` 로 두면 **우리 코드의 경고가 파묻힙니다** |
| `-Wno-unknown-pragmas` | g++ 는 `#pragma HLS` 를 모릅니다. 그게 정상입니다 |
| `-Wno-unused-label` | HLS 루프 라벨은 지시자·스케줄 리포트가 읽습니다. g++ 에게는 죽은 코드입니다 |

**나머지 경고는 끄지 않습니다.** `-Wall -Wextra` 가 requant 에서 `54×54 → 108비트` 곱을
잡아냈습니다 — 좁은 타입끼리 곱하지 않고 먼저 넓히면 `ap_int::operator*` 가 피연산자 폭의
합으로 결과를 만듭니다.

## 규칙

**블록 하나 = 테스트벤치 하나 = 러너 한 줄.** 연결 안 된 tb 는 만들지 않습니다 —
구 구현은 tb 12개 중 6개만 빌드에 걸렸고, **가장 잘 만든 3개가 아무 빌드에도 없었습니다.**

`vivado/` · `no_board/` 는 아직 비어 있습니다.
