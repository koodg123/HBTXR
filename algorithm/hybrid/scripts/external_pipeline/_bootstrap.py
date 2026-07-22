from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_src_on_path() -> Path:
    # This file lives at hybrid/scripts/external_pipeline/_bootstrap.py, so
    # the hybrid project root is three levels up. Putting the project root
    # (not project_root/src) on sys.path is what makes `import src.*` resolve.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root
