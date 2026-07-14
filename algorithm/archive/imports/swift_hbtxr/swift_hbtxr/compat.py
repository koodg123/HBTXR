from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .antiblink import AntiBlinkDetector
from .io import write_json


SWIFT_EYE_DEFAULT_THRESHOLDS = {
    "tracking_threshold": 0.0,
    "detection_threshold": 0.75,
    "template_update_threshold": 0.95,
}


SWIFT_EYE_UNET_PREFIX_MAP = [
    ("unet.inc.double_conv.", "inc.net."),
    ("unet.down1.maxpool_conv.1.double_conv.", "down1.net.1.net."),
    ("unet.down2.maxpool_conv.1.double_conv.", "down2.net.1.net."),
    ("unet.down3.maxpool_conv.1.double_conv.", "down3.net.1.net."),
    ("unet.down4.maxpool_conv.1.double_conv.", "down4.net.1.net."),
    ("unet.up1.conv.double_conv.", "up1.conv.net."),
    ("unet.up2.conv.double_conv.", "up2.conv.net."),
    ("unet.up3.conv.double_conv.", "up3.conv.net."),
    ("unet.up4.conv.double_conv.", "up4.conv.net."),
    ("unet.up1.up.", "up1.up."),
    ("unet.up2.up.", "up2.up."),
    ("unet.up3.up.", "up3.up."),
    ("unet.up4.up.", "up4.up."),
    ("unet.outc.conv.", "outc.conv."),
]


@dataclass
class ImportReport:
    imported_keys: list[str]
    skipped_keys: list[str]
    missing_keys: list[str]
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "imported_keys": self.imported_keys,
            "skipped_keys": self.skipped_keys,
            "missing_keys": self.missing_keys,
            "thresholds": self.thresholds,
        }


def _unwrap_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    return payload


def _map_swift_eye_unet_key(key: str) -> str | None:
    for source_prefix, target_prefix in SWIFT_EYE_UNET_PREFIX_MAP:
        if key.startswith(source_prefix):
            return key.replace(source_prefix, target_prefix, 1)
    if key.startswith("unet."):
        return key.split("unet.", 1)[1]
    return None


def import_swift_eye_antiblink_weights(
    *,
    checkpoint_path: str | Path,
    detector: AntiBlinkDetector,
    report_path: str | Path | None = None,
) -> ImportReport:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    detector_state = detector.model.state_dict()
    imported: dict[str, torch.Tensor] = {}
    imported_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        if not key.startswith("unet."):
            skipped_keys.append(str(key))
            continue
        stripped = _map_swift_eye_unet_key(str(key))
        if stripped is None:
            skipped_keys.append(str(key))
            continue
        if stripped in detector_state and detector_state[stripped].shape == value.shape:
            imported[stripped] = value
            imported_keys.append(str(key))
        else:
            skipped_keys.append(str(key))
    detector_state.update(imported)
    detector.model.load_state_dict(detector_state)
    report = ImportReport(
        imported_keys=sorted(imported_keys),
        skipped_keys=sorted(skipped_keys),
        missing_keys=sorted(set(detector_state.keys()) - set(imported.keys())),
        thresholds=dict(SWIFT_EYE_DEFAULT_THRESHOLDS),
    )
    if report_path is not None:
        write_json(report.to_dict(), report_path)
    return report
