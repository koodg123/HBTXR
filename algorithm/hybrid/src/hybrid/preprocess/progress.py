from __future__ import annotations

import time


def format_duration(seconds: float) -> str:
    total = max(int(round(float(seconds))), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(
    *,
    name: str,
    status: str,
    started_at: float,
    completed: int = 0,
    total: int | None = None,
    message: str | None = None,
    **extra,
) -> None:
    elapsed = time.perf_counter() - started_at
    parts = [
        f"[{name}]",
        f"status={status}",
        f"elapsed={format_duration(elapsed)}",
    ]
    if total is not None:
        pct = 100.0 * float(completed) / max(float(total), 1.0)
        parts.append(f"progress={completed}/{total}")
        parts.append(f"pct={pct:.1f}%")
        if completed > 0:
            eta = (float(total - completed) * elapsed) / float(completed)
            parts.append(f"eta={format_duration(eta)}")
    elif completed > 0:
        parts.append(f"completed={completed}")
    if message:
        parts.append(f"message={message}")
    for key, value in extra.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)
