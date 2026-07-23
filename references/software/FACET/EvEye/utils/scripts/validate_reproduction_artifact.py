import argparse
import json
import sys
from pathlib import Path


FACET_ROOT = Path(__file__).resolve().parents[3]
if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))

from EvEye.utils.scripts.check_reproduction_status import (
    validate_comparison_json,
    validate_eval_result_json,
    validate_summary_json,
)


VALIDATORS = {
    "eval": validate_eval_result_json,
    "comparison": validate_comparison_json,
    "summary": validate_summary_json,
}


def main():
    parser = argparse.ArgumentParser(
        description="Validate one FACET reproduction JSON artifact."
    )
    parser.add_argument("--type", choices=sorted(VALIDATORS), required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()

    ok, issues, evidence = VALIDATORS[args.type](args.path)
    result = {
        "ok": ok,
        "type": args.type,
        "path": str(args.path),
        "issues": issues,
        "evidence": evidence,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
