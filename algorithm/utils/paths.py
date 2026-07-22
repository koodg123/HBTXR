"""Filesystem roots and user-path normalization for HBTXR (leaf module).

Single source of truth for logical roots (data/output/cache/checkpoint/external,
bound via environment variables, see ``algorithm/.env.example``) plus helpers
that normalize user-supplied paths across Windows/WSL and resolve canonical
dataset roots. Merged from the former ``common.paths`` and ``hybrid.utils.paths``.

Example
-------
    from utils.paths import data_root, resolve

    events = resolve("data", "DavisEyeCenterDataset", "user1_events.txt")
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# ---- logical filesystem roots (was common.paths) ---------------------------

ENV_VARS: dict[str, str] = {
    "data": "EVEYE_DATA_ROOT",
    "output": "EVEYE_OUTPUT_ROOT",
    "cache": "EVEYE_CACHE_ROOT",
    "checkpoint": "EVEYE_CHECKPOINT_ROOT",
    "external": "EVEYE_EXTERNAL_ROOT",
}


class PathConfigError(RuntimeError):
    """A required logical root is not configured in the environment."""


def _env_var(kind: str) -> str:
    try:
        return ENV_VARS[kind]
    except KeyError as exc:
        known = ", ".join(sorted(ENV_VARS))
        raise PathConfigError(f"unknown logical root {kind!r}; known roots: {known}") from exc


def root(kind: str) -> Path:
    """Return the configured absolute root for a logical name.

    Raises PathConfigError if the backing environment variable is unset, so a
    missing binding fails loudly instead of silently writing to the wrong place.
    """
    var = _env_var(kind)
    value = os.environ.get(var)
    if not value:
        raise PathConfigError(
            f"logical root {kind!r} requires the {var} environment variable; "
            f"copy algorithm/.env.example and set it"
        )
    return Path(value).expanduser()


def resolve(kind: str, *parts: str | os.PathLike[str]) -> Path:
    """Join path parts under a logical root, rejecting parent-escape."""
    base = root(kind).resolve()
    target = base.joinpath(*[os.fspath(p) for p in parts]).resolve()
    if base != target and base not in target.parents:
        raise PathConfigError(f"resolved path {target} escapes the {kind!r} root {base}")
    return target


def data_root() -> Path:
    return root("data")


def output_root() -> Path:
    return root("output")


def cache_root() -> Path:
    return root("cache")


def checkpoint_root() -> Path:
    return root("checkpoint")


def external_root() -> Path:
    return root("external")


def is_configured(kind: str) -> bool:
    """True if the logical root is bound in the environment (no exception)."""
    return bool(os.environ.get(_env_var(kind)))


# ---- user-path normalization (was hybrid.utils.paths) ----------------------

_WINDOWS_ABS_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
_WSL_ABS_RE = re.compile(r"^/mnt/(?P<drive>[A-Za-z])/(?P<rest>.*)$")


def _normalize_path_string(raw: str) -> str:
    text = os.path.expandvars(os.path.expanduser(str(raw)))
    if os.name != "nt":
        match = _WINDOWS_ABS_RE.match(text)
        if match:
            rest = match.group("rest").replace("\\", "/")
            return f"/mnt/{match.group('drive').lower()}/{rest}"
        return text
    match = _WSL_ABS_RE.match(text)
    if match:
        rest = match.group("rest").replace("/", "\\")
        return f"{match.group('drive').upper()}:\\{rest}"
    return text


def normalize_user_path(path_str: str | Path, base: str | Path | None = None) -> Path:
    path = Path(_normalize_path_string(str(path_str)))
    if not path.is_absolute() and base is not None:
        path = normalize_user_path(base) / path
    return Path(os.path.abspath(str(path))) if path.is_absolute() else path


def resolve_stored_path(canonical_root: str | Path, path_str: str | Path) -> Path:
    p = normalize_user_path(path_str)
    if p.is_absolute():
        return p
    return normalize_user_path(p, base=canonical_root)


def resolve_canonical_dataset_root(
    canonical_root: str | Path,
    canonical_name: str | None = None,
    *,
    prefer_nested: bool = False,
) -> Path:
    root = normalize_user_path(canonical_root)
    name = str(canonical_name or "").strip()
    if not name:
        return root
    if root.name == name:
        return root
    nested = root / name
    root_has_dataset = any((root / marker).exists() for marker in ("sessions", "indexes"))
    nested_has_dataset = any((nested / marker).exists() for marker in ("sessions", "indexes"))
    if prefer_nested:
        if nested_has_dataset or not root_has_dataset:
            return nested
        return root
    if nested_has_dataset:
        return nested
    return root


def resolve_canonical_indexes_root(
    canonical_root: str | Path,
    *,
    canonical_name: str | None = None,
    indexes_root: str | Path | None = None,
    prefer_nested: bool = False,
) -> Path:
    if indexes_root is not None:
        return normalize_user_path(indexes_root)
    dataset_root = resolve_canonical_dataset_root(canonical_root, canonical_name, prefer_nested=prefer_nested)
    return dataset_root / "indexes"
