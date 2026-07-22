"""Filesystem root resolution for HBTXR.

This is the single source of truth for where data, outputs, caches, checkpoints
and vendored external assets live. Code must not hard-code absolute paths such
as ``/mnt/data2T/...``; it asks for a logical root here instead, and the machine
binding comes from an environment variable (see ``algorithm/.env.example``).

Example
-------
    from common.paths import data_root, resolve

    events = resolve("data", "DavisEyeCenterDataset", "user1_events.txt")
    out = resolve("output", "runs", experiment_name)
"""
from __future__ import annotations

import os
from pathlib import Path

# logical root -> environment variable that binds it to a machine
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
