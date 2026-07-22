from __future__ import annotations

import pytest

from common.paths import (
    ENV_VARS,
    PathConfigError,
    is_configured,
    resolve,
    root,
)


def test_every_logical_root_has_an_env_var() -> None:
    assert set(ENV_VARS) == {"data", "output", "cache", "checkpoint", "external"}


def test_unset_root_raises(monkeypatch) -> None:
    monkeypatch.delenv("EVEYE_DATA_ROOT", raising=False)
    assert not is_configured("data")
    with pytest.raises(PathConfigError):
        root("data")


def test_configured_root_resolves(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EVEYE_DATA_ROOT", str(tmp_path))
    assert is_configured("data")
    assert root("data") == tmp_path
    assert resolve("data", "a", "b.txt") == (tmp_path / "a" / "b.txt").resolve()


def test_unknown_root_name_raises() -> None:
    with pytest.raises(PathConfigError):
        root("nope")


def test_parent_escape_is_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EVEYE_DATA_ROOT", str(tmp_path))
    with pytest.raises(PathConfigError):
        resolve("data", "..", "..", "etc")
