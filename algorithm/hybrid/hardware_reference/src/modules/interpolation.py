from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimeLensConfig:
    timelens_root: str = "../../references/timelens"
    checkpoint_file: str = "../../references/timelens/refined_model/attention.bin"
    python_bin: str = "python"


class TimeLensRunner:
    def __init__(self, config: TimeLensConfig) -> None:
        self.config = config

    def command(self, *, image_root: str | Path, event_root: str | Path, output_root: str | Path) -> list[str]:
        return [
            self.config.python_bin,
            str(Path(self.config.timelens_root) / "run_timelens.py"),
            "--image-root",
            str(image_root),
            "--event-root",
            str(event_root),
            "--output-root",
            str(output_root),
            "--checkpoint",
            self.config.checkpoint_file,
        ]

