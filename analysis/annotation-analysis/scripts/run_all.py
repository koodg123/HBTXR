"""
run_all.py — run the full EV-Eye annotation/label analysis pipeline (01-05).

Usage:
  python run_all.py --root E:/DATASET/eveye --out ../results
  python run_all.py --root E:/DATASET/eveye --out ../results --exact --predict
  python run_all.py --root E:/DATASET/eveye --out ../results --max-users 3   # quick smoke
"""
import argparse, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("01_analyze_dataset.py",        ["--root", "--out", "--max-users", "--exact"]),
    ("02_analyze_labels.py",         ["--root", "--out"]),
    ("03_analyze_tobii_gaze.py",     ["--root", "--out", "--max-users"]),
    ("04_analyze_tracking_results.py", ["--root", "--out", "--predict"]),
    ("05_eval_annotation_quality.py", ["--out"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--max-users", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for script, accepts in STEPS:
        cmd = [sys.executable, os.path.join(HERE, script)]
        if "--root" in accepts:
            cmd += ["--root", args.root]
        if "--out" in accepts:
            cmd += ["--out", args.out]
        if "--max-users" in accepts and args.max_users:
            cmd += ["--max-users", str(args.max_users)]
        if "--exact" in accepts and args.exact:
            cmd += ["--exact"]
        if "--predict" in accepts and args.predict:
            cmd += ["--predict"]
        print("\n" + "=" * 70 + f"\n[RUN] {' '.join(cmd)}\n" + "=" * 70)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[!] {script} exited with {r.returncode} — continuing.")
    print(f"\n[done] results in {os.path.abspath(args.out)}")
    print("      open annotation_quality_report.md for the synthesis.")


if __name__ == "__main__":
    main()
