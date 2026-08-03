> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active
> **소유** hardware

# build — 골든 생성 · 검증

```bash
sh hardware/build/make_golden.sh      # 정수 골든 (생성물, git 에 없음)
sh hardware/build/run_tb.sh           # V1 — 모든 테스트벤치
sh hardware/build/run_csynth.sh       # V3 — 유닛 16개 합성 (vitis_hls 필요)
python3 hardware/tools/report_csynth.py   # 자원·타이밍·II 표
```

`run_tb.sh` 는 골든이 없으면 **알아서 만듭니다.**

## V3 — `run_csynth.sh`

`hls/csynth.tcl` **하나**를 `HBTXR_TOP` 으로 몰아서 씁니다. tcl 을 유닛마다 두면 열여섯 개가
서로 어긋납니다. 파트는 `xczu7ev-ffvc1156-2-e`(ZCU104), 주기 3.333 ns.

| | |
|---|---|
| **2023.2 고정** | 2024.1 도 깔려 있지만, V1 이 비트 일치를 증명한 헤더가 2023.2 입니다. 다른 헤더로 합성하면 같은 설계가 아닙니다 |
| **`-std=c++17` 없음** | 2023.2 프런트엔드에 C++17 을 강제하면 **자기 libstdc++ 8.3 헤더가 파싱에 실패**합니다. 기본 C++14 로 충분합니다 (`std::decay_t` 가 제일 새것) |
| **`settings64.sh` 안 씁니다** | 그 스크립트는 `source` 를 쓰는 bash 전용입니다. `bin/vitis_hls` 런처가 스스로 환경을 잡습니다 |
| **`xargs -P`** | vitis_hls 하나가 1~2 GB 를 뭅니다. `HBTXR_JOBS` 기본 4 |

**종료 코드로 판정하지 않습니다** — vitis_hls 는 합성 실패에도 0 을 내는 경우가 있어서,
`_csynth.rpt` 가 생겼는지로 봅니다.

`report_csynth.py` 는 **II 를 두 곳에서** 읽습니다. 리포트의 루프 표는 `achieved != target`
만 잡고, **파이프라인이 아예 실패한 루프는 `-` 로 나와서 언롤 안 한 외곽 타일과 구별되지
않습니다.** 그건 로그의 스케줄러 경고에만 있습니다.

## V1 은 `vitis_hls` 가 필요 없습니다

`hls::stream` 과 `hls::vector` 는 **plain g++ 에서 컴파일·실행됩니다** — Vitis 헤더만
있으면 됩니다(`XILINX_INCLUDE`, 기본값 `/tools/Xilinx/Vitis_HLS/2023.2/include`, WSL).
라이선스도 프로젝트도 필요 없습니다.

**그래서 V1 이 못 보는 것이 있습니다.** g++ 는 `#pragma HLS` 를 버리므로 **적용된 지시자와
무시된 지시자를 구별하지 못합니다.** S9 의 첫 csynth 가 `array_reshape` 두 개가 여덟 단계 내내
툴에 닿은 적 없었다는 것을 즉시 냈습니다. V1 은 값이 맞는지를 보고, V3 는 그 값이 어떤
하드웨어에서 나오는지를 봅니다 — 둘 다 필요합니다.

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
