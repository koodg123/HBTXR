from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hbtxr.preprocess.target_fps_canonical import build_argparser, run


if __name__ == "__main__":
    summary = run(build_argparser().parse_args())
    print(summary)
