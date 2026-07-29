> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** frozen — 스냅샷, 갱신하지 않음
> **소유** hardware

# hardware/ semantic census — 원자료 (2026-07-29)

**불변 스냅샷입니다.** 다시 찍으면 새 날짜 디렉토리를 만듭니다
([DOC-CONVENTIONS](../../../../docs/governance/DOC-CONVENTIONS.md) 규칙 H).

해석과 결론은 [reports/2026-07-29-hardware-census.md](../../reports/2026-07-29-hardware-census.md)에
있습니다. 이 디렉토리는 숫자만 담습니다.

## 무엇의 스냅샷인가

| | |
|---|---|
| 대상 | `hardware/**` (전수) |
| 커밋 | `8a9093e5e93ee66b31fd9ed16a68b6e4b686c454` |
| 소스 뷰 | **committed blobs** — 작업 트리 내용은 읽지 않음 |
| 범위 | 저장소 전체 스캔 후 `hardware/` 로 필터 |

## 재현

```bash
python docs/snapshots/2026-07-15-semantic-census/semantic_scan.py --root . --out <tmpdir>
# 그 다음 path 가 hardware/ 로 시작하는 행만 필터
```

스캐너는 **프로젝트 모듈을 import 하지 않습니다.** Python은 stdlib AST로 파싱하고,
그 외 언어는 보수적인 선언 정규식을 씁니다. 2026-07-15 census(algorithm 트리 대상)와
같은 도구·같은 스키마이므로 두 census는 비교 가능합니다.

## 파일

| 파일 | 행 | 내용 |
|---|---:|---|
| `file-inventory.csv` | 642 | 추적 파일 전수 — 경로·zone·언어·라인·바이트 |
| `python-symbols.csv` | 1,873 | 클래스/함수 — qualname·line·loc |
| `nonpython-symbols.csv` | 545 | C++/Tcl 선언 후보 |
| `import-edges.csv` | 1,247 | import 간선 |
| `line-findings.csv` | 732 | 라인 단위 소견 |
| `duplicate-functions.csv` | 128 | 의미 해시 동일 함수 |
| `duplicate-blobs.csv` | 3 | 동일 내용 파일 |
| `dead-code-candidates.csv` | 248 | **저신뢰 휴리스틱** — 이름 참조 횟수 기반 |
| `semantic-summary.json` | — | 집계 + 방법·한계 |

## 이 자료가 말하지 못하는 것

스캐너 자신의 한계 기록(`semantic-summary.json`의 `limitations`)에 더해:

- **동적 디스패치와 런타임 플러그인 등록은 해석되지 않습니다.** `dead-code-candidates`는
  이름 참조 휴리스틱이며 삭제 근거가 아닙니다.
- **비-Python 심볼은 정규식 후보**이지 컴파일러가 확인한 선언이 아닙니다. HLS 템플릿
  함수는 특히 과소/과대 계상될 수 있습니다.
- **Tcl·비트스트림·리포트의 의미**는 정적 census 밖입니다.
