from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_src_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return project_root
