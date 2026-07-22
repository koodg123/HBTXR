from __future__ import annotations

import os
import re
from pathlib import Path


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
