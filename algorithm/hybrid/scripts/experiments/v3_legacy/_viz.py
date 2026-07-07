from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_root_module(filename: str, module_name: str):
    module_path = Path(__file__).resolve().parents[2] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load root helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROOT_VIZ = _load_root_module("_viz.py", "hbtxr_root_script_viz")

for _name, _value in vars(_ROOT_VIZ).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
