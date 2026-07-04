from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.utils.paths import normalize_user_path


def read_json(path: str | Path):
    return json.loads(normalize_user_path(path).read_text(encoding="utf-8"))


def write_json(obj, path: str | Path) -> None:
    p = normalize_user_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    p = normalize_user_path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    p = normalize_user_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
