"""코드 이관 대장(P0) ↔ 새 hardware/ 트리를 양방향 대조합니다.

    python scripts/check_migration_manifest.py            # 보고
    python scripts/check_migration_manifest.py --check    # 어긋나면 exit 1

두 방향 모두 0이어야 합니다:
  1. `migrate`인데 새 트리에 없는 것          = 빠뜨림
  2. 새 트리에 있는데 대장이 설명 못 하는 것  = 출처 불명

내용 해시(CRLF 정규화)로 대조합니다. 파일명으로 하면 이름이 바뀐 이관을 놓치고,
문서 이관 때 실제로 파일명 충돌로 문서 1개가 소실됐다가 해시 대조로 복구됐습니다.
"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive" / "hardware"
NEW = ROOT / "hardware"
MANIFEST = NEW / "docs" / "plans" / "active" / "2026-07-29-code-migration-manifest.csv"

# 새 트리에만 있는 것이 정상 — 이관한 게 아니라 새로 쓴 것
NATIVE_PREFIX = ("docs/", "workspace/")
NATIVE_NAME = {"README.md", ".gitignore"}
NATIVE_EXACT = {"deploy/RUNTIME-GUIDE.md", "build/run_cyclic_tb.sh", "tools/tests/conftest.py"}


def is_native(rel: str) -> bool:
    return (rel in NATIVE_EXACT
            or rel.split("/")[-1] in NATIVE_NAME
            or any(rel.startswith(p) for p in NATIVE_PREFIX))


def digest(p: Path) -> str:
    return hashlib.md5(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"대장이 없습니다: {MANIFEST}")
        return 1

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    by_decision = Counter(r["decision"] for r in rows)

    new_files = [p for p in NEW.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts]
    new_hashes = {}
    for p in new_files:
        new_hashes.setdefault(digest(p), []).append(p.relative_to(ROOT).as_posix())

    # [repointed] = 이관 후 의도적으로 편집됨(경로 재지정). 해시가 아니라 파일명으로 대조합니다.
    new_names = {p.name for p in new_files}

    # 방향 1 — migrate 인데 새 트리에 없는 것
    missing = []
    for r in rows:
        if r["decision"] != "migrate":
            continue
        src = ARCHIVE / r["archive_path"]
        if not src.exists():
            missing.append((r["archive_path"], "archive에 원본이 없습니다"))
        elif "[repointed]" in r["reason"]:
            if src.name not in new_names:
                missing.append((r["archive_path"], f"→ {r['dest']} 예정, 아직 없음 (repointed)"))
        elif digest(src) not in new_hashes:
            missing.append((r["archive_path"], f"→ {r['dest']} 예정, 아직 없음"))

    # 방향 2 — 새 트리에 있는데 대장이 설명 못 하는 것
    accounted = set()
    accounted_names = set()
    for r in rows:
        src = ARCHIVE / r["archive_path"]
        if src.exists():
            accounted.add(digest(src))
            if "[repointed]" in r["reason"]:
                accounted_names.add(src.name)
    orphan = []
    for p in new_files:
        rel = p.relative_to(NEW).as_posix()
        if is_native(rel) or p.name in accounted_names:
            continue
        if digest(p) not in accounted:
            orphan.append(rel)

    print(f"대장 {len(rows)}행 — " + " · ".join(f"{k} {v}" for k, v in by_decision.most_common()))
    print(f"새 트리 파일 {len(new_files)}")
    print()
    print(f"[1] migrate 미완료 : {len(missing)}")
    for a, why in missing[:20]:
        print(f"      {a}  ({why})")
    if len(missing) > 20:
        print(f"      … 외 {len(missing) - 20}건")
    print(f"[2] 출처 불명      : {len(orphan)}")
    for o in orphan[:20]:
        print(f"      {o}")
    if len(orphan) > 20:
        print(f"      … 외 {len(orphan) - 20}건")

    if by_decision.get("undecided"):
        print(f"\n미결 {by_decision['undecided']}건 — 대장의 undecided 행을 보십시오")

    return 1 if (missing or orphan) else 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc if "--check" in sys.argv else 0)
