> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — governance
> **소유** repo

# 문서 규약

이 저장소의 문서를 **어디에 쓰고, 어떻게 표시하고, 언제 옮기는가.**

이 규약은 새로 만든 것이 아닙니다. `docs/track/`과 `hardware/docs/track/`이 이미 거의
같은 규약을 쓰고 있었고 — 같은 파일명, 같은 역할 — 그것이 적힌 곳이 없어서 세 번째 트리
(`algorithm/docs/`)에는 적용되지 않았습니다. **그 결과 33개 커밋이 어떤 추적 문서에도
기록되지 못했습니다.** 이 문서는 이미 작동하던 관행을 성문화하고 빠진 곳을 채웁니다.

---

## 1. 문서 트리는 셋

| 트리 | 담는 것 |
|---|---|
| `docs/` | 횡단·거버넌스·외부 조사. 어느 한 서브시스템에 속하지 않는 것 |
| `algorithm/docs/` | `algorithm/**` 코드가 설명 대상인 것 |
| `hardware/docs/` | `hardware/**` 코드가 설명 대상인 것 |

> **아카이브는 트리로 세지 않습니다.** 2026-07-29에 구 hardware 트리가
> `archive/hardware/`로 이동했고 그 안의 `docs/`도 함께 갔습니다. 아카이브는 **읽기 전용
> 기준**이므로 이 규약을 소급 적용하지 않습니다 — 스냅샷·핸드오프와 같은 취급입니다
> (규칙 G·H). 색인에는 별도 트리로 나오지만 규약의 대상은 아닙니다.

### 배치 규칙 — 한 문장

> **문서는 그것을 거짓으로 만들 수 있는 코드가 있는 트리에 산다.**

| 판정 | 위치 |
|---|---|
| `algorithm/**` 변경이 이 문서를 틀리게 만들 수 있다 | `algorithm/docs/` |
| `hardware/**` 변경이 틀리게 만들 수 있다 | `hardware/docs/` |
| 코드로는 틀려지지 않고 **결정**으로만 바뀐다 | `docs/` |
| **두 서브시스템에 걸친다** | `docs/` |

이 규칙이 필요한 이유: 문서가 자기가 설명하는 코드 옆에 있으면 한 커밋·한 diff·한
리뷰로 묶이고, 코드를 고치면서 문서를 안 고친 것이 **리뷰에서 보입니다.**

**판정 예시**

- `reports/2026-07-27-quantization-part-c-d.md` — 양자화 코드가 바뀌면 수치가 틀려짐 → `algorithm/docs/`
- `ARTIFACT-POLICY` — 어떤 코드로도 안 틀려짐, 결정으로만 → `docs/governance/`
- P&R 리포트 — HLS/제약이 바뀌면 틀려짐 → `hardware/docs/`
- `COMPARISON-TSR-FPGA.md` — 외부 저장소가 대상이고 양쪽 서브시스템에 걸침 → `docs/reference/`

---

## 2. 모든 문서의 첫 3줄

```markdown
> **작성** YYYY-MM-DD · **갱신** YYYY-MM-DD
> **상태** active | frozen | superseded-by:<경로> | consumed | archived
> **소유** repo | algorithm | algorithm/quantization | hardware
```

| 상태 | 뜻 |
|---|---|
| `active` | 갱신되고 있고, 지금 참인 내용 |
| `frozen` | 완료되어 더 갱신하지 않음. 당시에 참이었고 지금도 참 |
| `superseded-by:<경로>` | 대체됨. 어디로 대체됐는지 반드시 명시 |
| `consumed` | 일회성 문서가 소비됨 (주로 handoff) |
| `archived` | 소진됨. 역사 기록으로만 보존 |

**헤더가 없어도 됩니다.** `docs/INDEX.md` 생성기가 git 이력으로 대체하고 *(헤더 없음 —
git 기준)*으로 표시합니다. 헤더는 **git 날짜가 논리적 날짜와 다를 때** 가치가 있습니다 —
2026-04-10 실험을 2026-07-15에 커밋한 리포트 같은 경우입니다.

---

## 3. 파일 이름

**두 종류뿐입니다.**

| 종류 | 형식 | 예 |
|---|---|---|
| **역할 이름** — 고정 어휘. 트리마다 하나씩이고 날짜가 없다 | `UPPER-KEBAB.md` | `STATUS.md` `ARCHITECTURE.md` `MODULE-GUIDE.md` |
| **기록** — 어느 시점의 산출물 | `YYYY-MM-DD-lower-kebab.md` | `2026-07-29-hardware-census.md` |

디렉토리는 항상 `lower-kebab`, 실험 캠페인은 `YYYY-MM-DD-<slug>/`.

### 날짜는 접두사로, 하이픈으로, 한 번만

`_2026_06_10` (언더바·접미사) · `20260410` (붙여쓰기) · `2026-06-` (달까지만) — 안 씁니다.
날짜는 헤더의 `작성`에서 가져옵니다.

**캠페인 디렉토리가 이미 날짜를 말하면 안의 파일은 반복하지 않습니다.**

```
experiments/2026-06-10-final-signoff/signoff/final-signoff-audit.md      ← 날짜 없음
experiments/2026-06-30-aq2-search-track/
└── 2026-07-01-aq2-search-track-metric-checklist.md   ← 캠페인과 다른 날이라 접두사를 단다
```

### `data/`의 산출물은 이름을 바꾸지 않습니다

`.json`·`.csv`는 **도구가 그 이름으로 쓴 것**이고 manifest와 sha256이 그 이름을 참조합니다.
규약은 문서에만 적용합니다. 그래서 `final-signoff-audit.md` 옆에
`data/final_signoff_audit_2026_06_10.json`이 있는 것이 **정상**입니다.

### 예외 둘

- **spec-kit 계획 세트** — `Master-Plan.md` `Sub-Plan.md` `Execution.md` `Validation.md`
  `CHOICE.md`는 고정 어휘이고 루트 `docs/`가 같은 이름을 씁니다.
  `write_spec_plan_conformance_audit.py`가 루트 사본 경로를 하드코딩하고 있습니다.
- **`references/`** — 남의 프로젝트 문서 세트는 그쪽 이름 그대로 둡니다. 우리 규약을
  적용하면 원본과 대조할 수 없게 됩니다.

> **왜 이 절이 생겼나**: `hardware/docs/`만 규약 밖에 있었습니다. 한 트리 안에
> `THIRD_GOAL_REQUIREMENTS_2026_06_16.md` · `req1_environment_audit_2026_06_16.md` ·
> `summary-p0-mode-profile.md` · `2026-06-HBTXR-arch-freeze.md`가 공존했고, 어느 것이
> 규칙인지 알 수 없으니 새 문서마다 네 번째 방식이 생겼습니다. 2026-07-29에 87개를
> 개명해 맞췄습니다 (`archive/`의 원본 이름은 그대로입니다).

---

## 4. 색인은 생성합니다

```bash
python scripts/build_docs_index.py          # docs/INDEX.md 생성
python scripts/build_docs_index.py --check  # 최신이 아니면 exit 1
```

**손으로 유지하지 않습니다.** 이전에 손으로 유지하던 `docs/aegis/INDEX.md`는 331개 중
8개만 담고 있었고 그중 2행은 이미 옮겨간 트리를 가리키고 있었습니다. **움직이는 대상의
수기 색인은 기본값이 "틀림"입니다.**

---

## 5. `track/` — 트리마다 하나씩

각 트리는 `track/` 아래 여섯 파일을 가집니다. 역할이 겹치지 않습니다.

| 파일 | 답하는 질문 | 갱신 시점 | 순서 |
|---|---|---|---|
| `PROGRESS.md` | 계획 대비 어디까지 왔나 | **매 작업 종료** | 최신 위 |
| `CHANGELOG.md` | 무엇이 바뀌었나 | 매 커밋 묶음 | 최신 위 |
| `TODO.md` | 아직 안 한 것 | 발견 시 | 우선순위 |
| `ADR.md` | 왜 그렇게 정했나 | 결정 시 | **추가만** |
| `log.md` | 다른 세션이 이어받을 상태 | 세션 종료 | 최신 위 |
| `CONVERSATION.md` | 사용자가 뭘 요구했나 | 요구가 **바뀔 때만** | 시간순 |

### 규칙 A — 항목 상태의 단일 출처는 `TODO.md`

`PROGRESS.md`는 그것을 **요약**할 뿐입니다. 둘이 어긋나면 `TODO.md`가 맞습니다.

> **이 규칙이 없어서 생긴 일**: 2026-07-21의 AM 작업에 대해 `TODO.md`는 "미착수",
> `PROGRESS.md`는 "not started", `HANDOVER.md`는 "AM-030 완료"라고 각각 말했고,
> **셋 다 틀렸습니다.** 실제로는 AM-900까지 실행된 뒤 flat-functional 재작성이 대체한
> 상태였습니다. 6일간 그대로였습니다.

### 규칙 B — `ADR.md`는 추가만 합니다

결정이 뒤집히면 **새 항목**을 쓰고 옛 항목에 `superseded by ADR-NNN`만 붙입니다.
결정을 지운 기록은 왜 그렇게 했는지도 지웁니다.

### 규칙 C — 서브시스템 작업은 서브시스템 track에 기록합니다

`algorithm/**`을 고쳤으면 `algorithm/docs/track/`을 갱신합니다. `docs/track/`은 저장소
수준(거버넌스·구조·다중 서브시스템)만 담습니다.

---

## 6. `STATUS.md` — 트리마다 하나, "지금"의 단일 출처

네 섹션만 가집니다: **Active · Blocked · Next · Done(최근)**.

### 규칙 D — Blocked 항목은 "누가 풀 수 있나"를 반드시 적습니다

```markdown
| 항목 | 무엇이 막고 있나 | 누가 풀 수 있나 |
|---|---|---|
| A2 학습 체크포인트 | manifest 빌드 + 학습 실행 | **사용자 결정** |
```

**이것이 없으면 blocked와 forgotten이 구분되지 않습니다.** 이 저장소의 백로그 41건이
정확히 그 상태였습니다 — 무엇이 막고 있는지, 누가 풀 수 있는지 적혀 있지 않아서
"차단됨"인지 "잊힘"인지 알 수 없었습니다.

---

## 7. 실험 — 분석과 결과를 분리합니다

수명이 다르기 때문입니다.

| | 실험 **결과** | 실험 **분석** |
|---|---|---|
| 내용 | 숫자, 로그, 리포트 | 왜 했나, 무슨 뜻인가, 다음은 |
| 수명 | **불변 — 한 번 쓰면 수정 안 함** | 개정 가능 |
| 위치 | `<tree>/experiments/YYYY-MM-DD-<slug>/results.md` | `<tree>/reports/` |

```
<tree>/experiments/2026-07-29-hybrid-shared-scalars/
├── plan.md      가설·방법·성공기준        (실행 전 작성)
├── results.md   숫자만. 커밋 SHA + 재현 명령. 불변
└── data/        raw 로그·csv
```

### 규칙 E — 결과를 고쳐야 하면 새 날짜로 새 실험을 만듭니다

> **왜**: 2026-07-29의 B2 측정에서 첫 결과(max-abs, 프로브 1개)는 "공유 스칼라가 track에
> 해롭다"였고, RMS 12프로브×3시드에서 **노이즈로 판명**되어 뒤집혔습니다. 결과가
> 가변이면 그 뒤집힘이 사라지고, 사라지면 왜 최종 결론을 믿을 수 있는지도 사라집니다.

`results.md`는 **커밋 SHA와 재현 명령을 반드시 포함**합니다. 재현할 수 없는 숫자는
결과가 아니라 주장입니다.

---

## 8. 계획 — 디렉토리가 상태를 말합니다

```
<tree>/plans/active/2026-07-25-quantization-part-b-to-d.md
<tree>/plans/done/2026-07-29-quantization-part-b-to-d.md
```

완료되면 **이동**합니다. 삭제하지 않고, 파일 안의 상태 문구에 의존하지도 않습니다 —
파일을 열지 않아도 알 수 있어야 합니다.

`done/`으로 옮길 때 헤더에 결과 링크(리포트·실험)를 답니다.

**별도의 history 문서를 만들지 않습니다.** `CHANGELOG.md` + git이면 충분하고, 세 번째
진실 소스는 반드시 어긋납니다.

---

## 9. Spec · Scope · HANDOVER · NEXT

| | 무엇 | 위치 | 수명 |
|---|---|---|---|
| **Spec** | 서브시스템이 무엇을 하는가 | `<tree>/ARCHITECTURE.md` | 코드와 함께 상시 |
| **Scope** | 무엇을 **안** 하는가 | **Spec 안의 필수 섹션** | 상시 |
| **HANDOVER** | 다음 세션이 이어받을 상태 | `docs/handoff/` | **일회성** |
| **NEXT** | 다음에 할 것 | `STATUS.md`의 Next 섹션 | 별도 파일 없음 |

### 규칙 F — Scope는 독립 문서가 아니라 Spec의 섹션입니다

분리하면 반드시 어긋납니다. `reports/2026-07-27-quantization-part-c-d.md`의 "What it does NOT do"가
본문 바로 옆에 있었기 때문에 2026-07-29에 거짓 주장 4건이 잡혔습니다.

### 규칙 G — HANDOVER는 소비되면 만료됩니다

소비 즉시 상태를 `consumed`로 바꾸고 **만료 고지를 문서 맨 위에 답니다.**
본문은 편집하지 않습니다 — 핸드오프는 어느 시점의 기록이고, 지금 참이 되도록 고쳐 쓰면
그것이 존재한 이유가 사라집니다.

> **왜**: `docs/HANDOVER.md`가 2026-07-21 상태를 권위처럼 들고 있어서 `TODO.md`와
> 6일간 충돌했습니다.

---

## 10. Snapshot — 불변, 날짜, 커밋 SHA

```
<tree>/snapshots/2026-07-15-follow-up-baseline/
├── README.md      무엇의 스냅샷 · 커밋 SHA · 왜 찍었나
└── ...            원본 그대로
```

### 규칙 H — 스냅샷은 갱신하지 않습니다

다시 찍으면 **새 날짜 디렉토리**입니다. 스냅샷을 편집하는 순간 baseline이 아니게 됩니다.

---

## 11. 남의 것은 `references/`

남의 코드베이스·논문·저장소를 조사한 것은 **우리 코드로 틀려지지 않으므로** 서브시스템
트리에 두지 않습니다.

**디렉토리 이름은 복수 `references/`로 통일합니다.** 두 종류를 담지만 이름은 하나입니다 —
단수/복수는 그 차이를 알리기엔 나쁜 신호였습니다.

| 어디 | 무엇 | 편집 |
|---|---|---|
| `docs/references/external/` | 외부 코드베이스·논문 **서베이** — 읽고 요약한 것 | 우리가 씀 |
| `docs/references/COMPARISON-*.md` | 외부 프로젝트와의 비교 분석 | 우리가 씀 |
| `<tree>/docs/references/<project>/` | **선행·자매 프로젝트의 문서 세트 통째로** | **내용 무편집** |

마지막 줄의 경계는 **이름과 내용 사이**입니다:

| | 어떻게 |
|---|---|
| 파일·디렉토리 이름 | **§3을 적용합니다.** 우리 트리 안에서 검색·탐색되는 대상이므로 |
| 링크의 대상과 표시명 | 개명에 맞춰 재작성합니다 |
| 문서 내용 | **편집하지 않습니다** |
| 본문 산문에 적힌 그쪽 저장소 경로 | 그대로 둡니다 — 우리 트리 경로가 아니라 그쪽 기준 서술입니다 |
| 그쪽 소스 트리를 가리키는 깨진 링크 | 그대로 둡니다 |

**대조 기준은 `archive/`입니다.** 이름을 바꾸면 원본과 파일명으로 대조할 수 없으므로,
개명 전 사본이 `archive/`에 남아 있어야 합니다. 없으면 개명하지 않습니다. 그리고 개명한
트리의 README에 **대조표를 답니다** — 예:
[hardware/docs/references/README.md](../../hardware/docs/references/README.md).

> **디렉토리 계층은 재배치하지 않습니다.** 이름은 우리 것으로 바꾸되, 그쪽이 문서를 어떻게
> 조직했는지는 그 자체가 참조 정보입니다. 예를 들어 남의 `experiments/`를 우리 캠페인
> 형식(`YYYY-MM-DD-slug/`)으로 옮기면 그건 개명이 아니라 재구성입니다.

> **미적용**: `docs/reference/`(단수)가 아직 남아 있습니다. `docs/` 정리 때
> `docs/references/`로 개명합니다 — [track/TODO.md](../track/TODO.md).
> `hardware/docs/references/`는 이미 복수입니다.

---

## 12. 적용 현황 (2026-07-29, 5단계 완료)

파일 188개를 `git mv`로 이동했습니다 (이력 보존). `.agents/` 트리는 해소되어 **문서 트리가
넷에서 셋으로** 줄었습니다.

### 옮긴 것

| 이동 | 대상 |
|---|---|
| `docs/Artifact-Policy.md` · `track/CONTRIBUTING.md` · `track/TASK-CARD-PROTOCOL.md` · `aegis/{README,BASELINE-GOVERNANCE}.md` · `provenance/**` | → `docs/governance/` |
| `docs/aegis/plans/*` (4) · `.agents/plans/*` | → `docs/plans/done/` |
| `docs/aegis/baseline/*` · `aegis/work/*/90-evidence.md` · `.agents/recon/**` | → `docs/snapshots/` |
| `docs/HANDOVER*.md` · `.agents/handoff/*` | → `docs/handoff/` |
| `docs/aegis/INDEX.md` · `aegis/work/*/{10-intent,20-checkpoint,99-reflection}.md` | → `docs/archive/2026-07/` |
| `docs/analysis/HANDOVER-*.md` (6) | → `docs/reference/handover-audit-2026-07-15/` |
| `hardware/docs/analysis/integrated-2026-06-26/**` (~140) | → `docs/reference/external/2026-06-26/` |
| `algorithm/docs/COMPARISON-TSR-FPGA.md` | → `docs/reference/` |
| `docs/analysis/{ALGORITHM-IMPORT-CONSUMERS,HBTXR-SEMANTIC-CENSUS,*-MATRIX}.md` · `track/algorithm-modular-baseline.md` | → `algorithm/docs/reports/` (날짜 접두사) |
| `algorithm/docs/QUANTIZATION-*.md` (4) | → `algorithm/docs/{plans/active,plans/done,reports}/` (날짜 접두사) |

### 옮기지 않은 것 — 이유 있음

**`hardware/docs/**` 전체와 루트 `docs/{Master-Plan,Sub-Plan,Spec,Execution,Validation}.md`
는 그대로 둡니다.** 정리가 덜 된 게 아니라 **하드웨어 툴체인이 경로로 고정**하고 있기
때문입니다:

```python
# archive/hardware/tools/write_spec_plan_conformance_audit.py
DOCS = {"master_plan": "docs/Master-Plan.md", "spec": "docs/Spec.md", ...}
docs = {name: root / rel for name, rel in DOCS.items()}     # root = HBTXR/  → 루트 docs/
current_doc_base = root/"hardware" if (root/"hardware"/"docs"/"Spec.md").exists() else root
```

- 루트 `docs/{5종}`은 `DOCS` dict가 **존재 여부와 내용을 검사**합니다.
- `hardware/docs/{5종}`은 `current_doc_base` **분기 조건 자체**입니다.
- `hardware/docs/track/{PROGRESS,HANDOVER,log}.md`와 `hardware/docs/resources/**`도
  같은 도구와 **10개 이상의 테스트**가 읽습니다.

`hardware/tests`는 **손대기 전부터 475개 중 49개가 실패** 중입니다. 이미 취약한 감사
툴체인을, 그것도 휴지 중이라 검증할 수 없는 서브시스템에서, 정돈을 위해 더 망가뜨리는 것은
나쁜 교환입니다. **이동은 하드웨어 소유 작업으로 남깁니다** — 도구와 테스트를 같이 고쳐야
합니다.

### 얼어붙은 문서 안의 경로는 고치지 않았습니다

`docs/{archive,snapshots,handoff}/**` 안의 링크는 **당시 경로 그대로**입니다. 규칙 G·H의
직접적 귀결입니다 — 어느 시점의 기록을 지금 참이 되도록 고쳐 쓰면 그것이 존재한 이유가
사라집니다. 각 디렉토리의 `README.md`가 이 사실을 알립니다.

**색인이 먼저 있었던 이유가 이것입니다** — 이동 후 `--check`와 링크 검사로 깨진 참조를
바로 찾았습니다.
