from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_src_on_path() -> Path:
    # hybrid/scripts/external_pipeline/_bootstrap.py -> hybrid is parents[2].
    # hybrid/src is placed on sys.path so `import hybrid.*` resolves without
    # requiring an editable install; the project root (hybrid) is returned so
    # run artifacts land under hybrid/, not under scripts/.
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return project_root
