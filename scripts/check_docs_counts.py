"""hardware/docs/experiments 의 손으로 쓴 숫자를 파일시스템과 대조합니다.

왜 있는가: 이 표의 합계는 두 번 틀렸습니다 — `experiments/README.md`가 75,
`STATUS.md`가 78이라고 말했고 실제는 77이었습니다. 규약 §4가 말하는
"움직이는 대상의 수기 색인은 기본값이 틀림"의 사례입니다.

    python scripts/check_docs_counts.py            # 보고
    python scripts/check_docs_counts.py --check    # 어긋나면 exit 1
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "hardware" / "docs" / "experiments"
INDEX = EXP / "README.md"
STATUS = ROOT / "hardware" / "docs" / "STATUS.md"

ROW = re.compile(r"^\|\s*\*{0,2}[\d-]+\*{0,2}\s*\|\s*\[\*{0,2}([^\]]+?)\*{0,2}\]\(([^)]+)/\)\s*\|"
                 r"\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}(\d+|—)\*{0,2}\s*\|", re.M)
TOTAL = re.compile(r"\*\*총 (\d+)캠페인 · 문서 (\d+) · 데이터 (\d+)\.\*\*")
ST_EXP = re.compile(r"\*\*(\d+)캠페인 · 문서 (\d+) · 데이터 (\d+)\*\*")


def actual():
    out = {}
    for c in sorted(EXP.iterdir()):
        if not c.is_dir():
            continue
        md = [p for p in c.rglob("*.md") if p.name != "README.md"]
        data = [p for p in c.rglob("*") if p.is_file() and p.suffix != ".md"]
        out[c.name] = (len(md), len(data))
    return out


def main():
    a = actual()
    text = INDEX.read_text(encoding="utf-8")
    bad = []

    for m in ROW.finditer(text):
        slug, claim_md = m.group(2), int(m.group(3))
        claim_data = 0 if m.group(4) == "—" else int(m.group(4))
        if slug not in a:
            bad.append(f"{INDEX.name}: '{slug}' 캠페인이 존재하지 않습니다")
            continue
        real_md, real_data = a[slug]
        if claim_md != real_md:
            bad.append(f"{slug}: 문서 주장 {claim_md} / 실제 {real_md}")
        if claim_data != real_data:
            bad.append(f"{slug}: 데이터 주장 {claim_data} / 실제 {real_data}")

    tm = TOTAL.search(text)
    tot = (len(a), sum(v[0] for v in a.values()), sum(v[1] for v in a.values()))
    if not tm:
        bad.append(f"{INDEX.name}: 합계 줄을 찾지 못했습니다")
    elif tuple(int(x) for x in tm.groups()) != tot:
        bad.append(f"합계 주장 {tm.groups()} / 실제 {tot}")

    sm = ST_EXP.search(STATUS.read_text(encoding="utf-8"))
    if sm and tuple(int(x) for x in sm.groups()) != tot:
        bad.append(f"STATUS.md 주장 {sm.groups()} / 실제 {tot}")

    if bad:
        print("어긋남:")
        for b in bad:
            print("  -", b)
        return 1
    print(f"일치 — 캠페인 {tot[0]} · 문서 {tot[1]} · 데이터 {tot[2]}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc if "--check" in sys.argv else 0)
