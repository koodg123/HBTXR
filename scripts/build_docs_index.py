#!/usr/bin/env python3
"""Generate ``docs/INDEX.md`` — one table for every document in the repository.

Run from the repository root::

    python scripts/build_docs_index.py          # write docs/INDEX.md
    python scripts/build_docs_index.py --check  # exit 1 if out of date (for CI)

Why this is generated rather than maintained by hand: the repository already had a
hand-maintained index (``docs/aegis/INDEX.md``). It covered 8 of 331 documents and two of
its rows pointed into a tree that had since moved. A hand-written index of a moving
corpus is a document that is wrong by default.

Where each field comes from:

* **date / status / owner** — the three-line header at the top of the file, when present::

      > **작성** 2026-07-29 · **갱신** 2026-07-29
      > **상태** active
      > **소유** algorithm

* **fallback** — git history. First-commit date for 작성, last-commit for 갱신. A document
  with no header is reported as ``(git)`` so the reader knows the status column is a guess
  rather than a statement. That fallback is deliberate: bulk-inserting a header into ~300
  frozen documents would be a large diff asserting metadata that git already knows.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "INDEX.md"

# (표시명, 경로, 설명) — 문서 트리. 순서가 INDEX의 순서다.
TREES = [
    ("repo", "docs", "횡단·거버넌스·외부 조사"),
    ("algorithm", "algorithm/docs", "학습·평가·양자화·정수 그래프"),
    ("hardware", "hardware/docs", "HLS·RTL·Vivado·PYNQ 가속기"),
    ("agents", ".agents", "handoff·recon (docs/로 흡수 예정)"),
]

# 개별 나열이 가치를 더하지 않는 대량 군집: 요약 한 줄로 접는다.
COLLAPSE = {
    "hardware/docs/analysis/integrated-2026-06-26":
        "외부 코드베이스·논문 서베이 (우리 코드로는 틀려지지 않음 → docs/reference/external/ 이동 대상)",
    "hardware/docs/resources/hgtxr_handover_additional":
        "HGTXR 인계 증거 — 불변 (→ hardware/docs/evidence/ 이동 대상)",
    "hardware/docs/resources/hgtxr_final_evidence":
        "HGTXR 최종 증거 — 불변 (→ hardware/docs/evidence/ 이동 대상)",
}

HEADER_RE = re.compile(
    r"^>\s*\*\*작성\*\*\s*(?P<created>[\d-]+)"
    r"(?:\s*·\s*\*\*갱신\*\*\s*(?P<updated>[\d-]+))?", re.M)
STATUS_RE = re.compile(r"^>\s*\*\*상태\*\*\s*(?P<status>.+?)\s*$", re.M)
OWNER_RE = re.compile(r"^>\s*\*\*소유\*\*\s*(?P<owner>.+?)\s*$", re.M)


@dataclass
class Doc:
    path: str
    created: str
    updated: str
    status: str
    owner: str
    from_header: bool

    @property
    def sort_key(self):
        return (self.updated, self.path)


def git_dates(paths: list[str]) -> dict[str, tuple[str, str]]:
    """First and last commit date per path, in one pass over the log."""
    out: dict[str, tuple[str, str]] = {}
    for path in paths:
        res = subprocess.run(
            ["git", "log", "--format=%ad", "--date=short", "--", path],
            cwd=ROOT, capture_output=True, text=True)
        dates = res.stdout.split()
        out[path] = (dates[-1], dates[0]) if dates else ("?", "?")
    return out


def read_doc(rel: str, text: str, dates: tuple[str, str]) -> Doc:
    head = text[:600]
    m = HEADER_RE.search(head)
    if m:
        created = m.group("created")
        updated = m.group("updated") or created
        status = (STATUS_RE.search(head) or {}).group("status") if STATUS_RE.search(head) else "?"
        owner = (OWNER_RE.search(head) or {}).group("owner") if OWNER_RE.search(head) else "?"
        return Doc(rel, created, updated, status, owner, True)
    return Doc(rel, dates[0], dates[1], "*(헤더 없음 — git 기준)*", "?", False)


def collect() -> list[tuple[str, str, str, list[Doc]]]:
    """[(tree_label, tree_path, tree_desc, docs)] — collapsed groups excluded."""
    result = []
    for label, treepath, desc in TREES:
        base = ROOT / treepath
        if not base.exists():
            continue
        rels = []
        for f in sorted(base.rglob("*.md")):
            rel = f.relative_to(ROOT).as_posix()
            if any(rel.startswith(c + "/") for c in COLLAPSE):
                continue
            rels.append(rel)
        dates = git_dates(rels)
        docs = [read_doc(r, (ROOT / r).read_text(encoding="utf-8", errors="replace"), dates[r])
                for r in rels]
        result.append((label, treepath, desc, docs))
    return result


def collapsed_rows() -> list[tuple[str, int, str, str]]:
    rows = []
    for prefix, why in COLLAPSE.items():
        base = ROOT / prefix
        if not base.exists():
            continue
        files = list(base.rglob("*.md"))
        dates = git_dates([prefix])[prefix]
        rows.append((prefix, len(files), dates[1], why))
    return rows


def render() -> str:
    trees = collect()
    collapsed = collapsed_rows()
    total = sum(len(d) for _, _, _, d in trees) + sum(r[1] for r in collapsed)
    today = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip() or "?"

    L: list[str] = []
    L.append(f"> **작성** 2026-07-29 · **갱신** {today}")
    L.append("> **상태** active — 자동 생성")
    L.append("> **소유** repo")
    L.append("")
    L.append("# HBTXR 문서 색인")
    L.append("")
    L.append("**이 파일은 자동 생성됩니다. 직접 편집하지 마십시오.**")
    L.append("")
    L.append("```bash")
    L.append("python scripts/build_docs_index.py")
    L.append("```")
    L.append("")
    L.append(f"총 **{total}개** 문서. 날짜·상태·소유는 각 문서의 3줄 헤더에서 읽고, "
             "헤더가 없으면 git 이력으로 대체합니다 — 그 경우 *(헤더 없음 — git 기준)*으로 "
             "표시되므로 상태 칸이 추정임을 알 수 있습니다.")
    L.append("")
    L.append("규약: [governance/DOC-CONVENTIONS.md](governance/DOC-CONVENTIONS.md)")
    L.append("")

    # --- 요약 ---
    L.append("## 트리 요약")
    L.append("")
    L.append("| 트리 | 경로 | 문서 | 최종 활동 | 내용 |")
    L.append("|---|---|---:|---|---|")
    for label, treepath, desc, docs in trees:
        extra = sum(r[1] for r in collapsed if r[0].startswith(treepath))
        last = max((d.updated for d in docs), default="?")
        L.append(f"| **{label}** | `{treepath}/` | {len(docs) + extra} | {last} | {desc} |")
    L.append("")

    # --- 지금 봐야 할 것 ---
    L.append("## 먼저 볼 것")
    L.append("")
    L.append("| 알고 싶은 것 | 문서 |")
    L.append("|---|---|")
    L.append("| 저장소 전체 Active/Blocked/Next | [STATUS.md](STATUS.md) |")
    L.append("| 양자화·정수 그래프 현황 | [../algorithm/docs/STATUS.md](../algorithm/docs/STATUS.md) |")
    L.append("| 하드웨어 가속기 현황 | [../hardware/docs/STATUS.md](../hardware/docs/STATUS.md) |")
    L.append("| 문서를 어디에 써야 하나 | [governance/DOC-CONVENTIONS.md](governance/DOC-CONVENTIONS.md) |")
    L.append("")

    # --- 트리별 전체 ---
    for label, treepath, desc, docs in trees:
        L.append(f"## {label} — `{treepath}/`")
        L.append("")
        if not docs:
            L.append("*(문서 없음)*")
            L.append("")
            continue
        L.append("| 문서 | 작성 | 갱신 | 상태 | 소유 |")
        L.append("|---|---|---|---|---|")
        for d in sorted(docs, key=lambda x: x.sort_key, reverse=True):
            name = d.path[len(treepath) + 1:] if d.path.startswith(treepath + "/") else d.path
            # INDEX.md는 docs/ 안에 있으므로 다른 트리로 가려면 정확히 한 단계 위다.
            href = name if treepath == "docs" else f"../{d.path}"
            L.append(f"| [{name}]({href}) | {d.created} | {d.updated} | {d.status} | {d.owner} |")
        L.append("")

    # --- 접힌 군집 ---
    if collapsed:
        L.append("## 접힌 군집")
        L.append("")
        L.append("개별 나열이 가치를 더하지 않는 대량 문서군입니다. 경로로 직접 찾으십시오.")
        L.append("")
        L.append("| 경로 | 문서 | 최종 | 내용 |")
        L.append("|---|---:|---|---|")
        for prefix, n, last, why in collapsed:
            L.append(f"| `{prefix}/` | {n} | {last} | {why} |")
        L.append("")

    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="생성 결과가 현재 파일과 다르면 1로 종료 (CI용)")
    args = ap.parse_args()

    content = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != content:
            print("docs/INDEX.md가 최신이 아닙니다. `python scripts/build_docs_index.py` 실행",
                  file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} 생성 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
