"""Every CLI entrypoint module must import, and import *lightly* (D1).

``scripts/hbtxr.py`` dispatches each subcommand with ``importlib.import_module(target)``
followed by ``module.main(rest)``. Every entrypoint used to do

    from engine.data.factory import build_dataloader

at module scope, which drags the whole HBTXR data pipeline (PIL / cv2 / h5py / tonic) in
at *import* time. So ``hbtxr train|eval|infer|distill|quantize`` each died on an
ImportError before parsing a single argument -- five shipped commands dead at once, and
nothing in the suite noticed. This file is the regression guard for that class of bug.

Two assertions per module, and the second is the one that actually pins the fix:

1. the module imports, and exposes both its ``run_*`` API and the ``main`` the dispatcher
   calls;
2. importing it leaves the data-pipeline modules out of ``sys.modules``.

(2) has to run in a fresh interpreter. In-process, ``sys.modules`` already carries
whatever earlier tests imported, so observing an absence would prove nothing about this
import. Each module is therefore probed in its own subprocess -- cached, because a cold
probe pays for a torch import.

Scope note, stated plainly: in a venv without PIL an eager pipeline import cannot succeed,
so assertion (2) degenerates into assertion (1) there. It earns its keep on a machine that
*has* the data-pipeline packages installed, which is exactly where the laziness would
otherwise rot back to eager unnoticed.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_ALGORITHM_ROOT = Path(__file__).resolve().parents[1]

# Presence of any of these after importing an entrypoint means the data pipeline came
# along. ``engine.data.factory`` is the direct offender; ``dataset.hbtxr.transform`` is
# where the chain bottoms out on ``from PIL import Image``.
_PIPELINE_MODULES = ("engine.data.factory", "dataset.hbtxr", "dataset.hbtxr.transform")

_ENTRYPOINTS = [
    ("engine.train.entrypoint", "run_train"),
    ("engine.eval.entrypoint", "run_eval"),
    ("engine.infer.entrypoint", "run_infer"),
    ("engine.distill.entrypoint", "run_distill"),
    ("quantization.entrypoint", "run_quantize"),
]

_PROBE = """
import importlib, json, sys

module_name, attrs, watched = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
module = importlib.import_module(module_name)
json.dump(
    {
        "missing": [name for name in attrs if not callable(getattr(module, name, None))],
        "leaked": [name for name in watched if name in sys.modules],
    },
    sys.stdout,
)
"""


@functools.lru_cache(maxsize=None)
def _probe(module_name: str, attrs: tuple[str, ...]) -> dict[str, Any]:
    """Import ``module_name`` in a clean interpreter; report attrs + pipeline leakage."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            module_name,
            json.dumps(list(attrs)),
            json.dumps(list(_PIPELINE_MODULES)),
        ],
        cwd=_ALGORITHM_ROOT,  # flat, non-installed layout: algorithm/ is the import root
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, f"importing {module_name} failed:\n{completed.stderr}"
    return json.loads(completed.stdout)


def _cli_command_modules() -> dict[str, str]:
    """Read the subcommand -> module table straight out of the CLI dispatcher."""
    path = _ALGORITHM_ROOT / "scripts" / "hbtxr.py"
    spec = importlib.util.spec_from_file_location("_hbtxr_cli_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module._COMMANDS)


@pytest.mark.parametrize(("module_name", "run_attr"), _ENTRYPOINTS)
def test_entrypoint_exposes_cli_callables(module_name: str, run_attr: str) -> None:
    report = _probe(module_name, (run_attr, "main"))
    assert report["missing"] == []


@pytest.mark.parametrize(("module_name", "run_attr"), _ENTRYPOINTS)
def test_entrypoint_import_does_not_pull_data_pipeline(module_name: str, run_attr: str) -> None:
    report = _probe(module_name, (run_attr, "main"))
    assert report["leaked"] == []


def test_every_cli_subcommand_target_is_importable() -> None:
    """The dispatch table is the contract; a subcommand must not go dead unnoticed.

    Driven off ``_COMMANDS`` rather than a hardcoded list so a subcommand added later is
    covered without anyone remembering to extend this file.
    """
    commands = _cli_command_modules()
    assert set(commands) >= {"train", "eval", "infer", "distill", "quantize"}
    dead = [
        f"{command} -> {module_name}"
        for command, module_name in sorted(commands.items())
        if _probe(module_name, ("main",))["missing"]
    ]
    assert dead == []


# --- the proxies must forward faithfully, not merely exist ------------------
#
# A lazy proxy that drops a kwarg or reorders an argument passes every "does it
# import" test in this file. These call each proxy against a stub factory injected
# into sys.modules and assert the call arrives verbatim.

import importlib
import sys
import types

import pytest

ENGINE_PROXIES = [
    ("engine.train.entrypoint", {"shuffle": True, "modality": "frame"}),
    ("engine.eval.entrypoint", {"shuffle": False, "modality": "frame"}),
    ("engine.infer.entrypoint", {"shuffle": False, "modality": "frame"}),
    ("engine.distill.entrypoint", {"shuffle": True, "modality": "frame"}),
    ("quantization.entrypoint", {"shuffle": False, "modality": "hybrid"}),
]


@pytest.mark.parametrize("module_name,kwargs", ENGINE_PROXIES)
def test_build_dataloader_proxy_forwards_verbatim(module_name, kwargs, monkeypatch):
    module = importlib.import_module(module_name)
    proxy = getattr(module, "_build_dataloader")

    seen: dict[str, object] = {}
    sentinel = object()

    def fake_build_dataloader(*args, **kw):
        seen["args"] = args
        seen["kwargs"] = kw
        return sentinel

    stub = types.ModuleType("engine.data.factory")
    stub.build_dataloader = fake_build_dataloader
    stub.AdaptedLoader = list
    monkeypatch.setitem(sys.modules, "engine.data.factory", stub)

    cfg = {"marker": module_name}
    result = proxy("MANIFEST.jsonl", cfg, **kwargs)

    assert result is sentinel, f"{module_name}._build_dataloader dropped the return value"
    assert seen["args"] == ("MANIFEST.jsonl", cfg), f"positional args mangled: {seen['args']}"
    assert seen["kwargs"] == kwargs, f"keyword args mangled: {seen['kwargs']}"


# --- every relative import in a package __init__ must resolve ----------------
#
# dataset/preprocess/__init__.py imported a module that did not exist, which made the
# whole data pipeline unimportable on EVERY machine. Nothing caught it because the
# package needs PIL to import at all. This check is static, so it works anyway.

_ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


def _package_inits() -> list[Path]:
    skip = {"references", "backup", "archive", "third", "__pycache__", ".git"}
    return [p for p in _ALGORITHM_ROOT.rglob("__init__.py")
            if not (skip & set(p.relative_to(_ALGORITHM_ROOT).parts))]


def test_every_relative_import_in_a_package_init_resolves():
    """Static: a `from .x import ...` whose `.x` does not exist is a broken package."""
    import ast

    broken: list[str] = []
    for init in _package_inits():
        tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is None:
                continue
            target = node.module.split(".")[0]
            pkg_dir = init.parent
            if (pkg_dir / f"{target}.py").exists() or (pkg_dir / target / "__init__.py").exists():
                continue
            rel = init.relative_to(_ALGORITHM_ROOT).as_posix()
            broken.append(f"{rel}:{node.lineno} `from .{node.module} import ...` -> no such module")
    assert not broken, "package __init__ imports a module that does not exist:\n  " + "\n  ".join(broken)
