from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import nn

from hybrid.optim.adam_mini import build_adam_mini_optimizer
from hybrid.optim.adema_mix import build_adema_mix_optimizer
from hybrid.optim.adamw import build_adamw_optimizer
from hybrid.optim.adopt import build_adopt_optimizer
from hybrid.optim.common import diff_doc_path, optimizer_summary_payload, resolve_optimizer_cfg, resolve_optimizer_modifiers
from hybrid.optim.ivon import build_ivon_optimizer
from hybrid.optim.lion import build_lion_optimizer
from hybrid.optim.mars import build_mars_optimizer
from hybrid.optim.soap import build_soap_optimizer
from hybrid.optim.modifiers import CautiousOptimizer, build_schedule_free_adamw
from hybrid.optim.musgd import build_musgd_optimizer
from hybrid.optim.prodigy import build_prodigy_optimizer
from hybrid.optim.sophia_g import build_sophia_g_optimizer


def _unimplemented_builder(name: str):
    def _builder(model: nn.Module, resolved_cfg: dict[str, Any]):
        raise NotImplementedError(f"Optimizer '{name}' is registered but not implemented in HBTXR yet.")

    return _builder


_REGISTRY: dict[str, dict[str, Any]] = {
    "adamw": {
        "builder": build_adamw_optimizer,
        "implemented": True,
        "source": "torch.optim.AdamW",
        "upstream": "torch.optim.AdamW",
        "algorithmic_diff": False,
    },
    "lion": {
        "builder": build_lion_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/google/automl/master/lion/lion_pytorch.py",
        "upstream": "google/automl lion_pytorch.py",
        "algorithmic_diff": False,
    },
    "prodigy": {
        "builder": build_prodigy_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/konstmish/prodigy/main/prodigyopt/prodigy.py",
        "upstream": "konstmish/prodigy prodigyopt/prodigy.py",
        "algorithmic_diff": False,
    },
    "adopt": {
        "builder": build_adopt_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/optim/adopt.py",
        "upstream": "timm/optim/adopt.py",
        "algorithmic_diff": False,
    },
    "musgd": {
        "builder": build_musgd_optimizer,
        "implemented": True,
        "source": "E:/WSL/Shared/ETRI_SYNC/HBTXR/references/ultralytics-main/ultralytics/optim/muon.py",
        "upstream": "ultralytics/optim/muon.py",
        "algorithmic_diff": False,
    },
    "sophia_g": {
        "builder": build_sophia_g_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/Liuhong99/Sophia/main/sophia.py",
        "upstream": "Liuhong99/Sophia sophia.py",
        "algorithmic_diff": True,
    },
    "ivon": {
        "builder": build_ivon_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/team-approx-bayes/ivon/refs/heads/main/ivon/_ivon.py",
        "upstream": "team-approx-bayes/ivon ivon/_ivon.py",
        "algorithmic_diff": False,
    },
    "adam_mini": {
        "builder": build_adam_mini_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/zyushun/Adam-mini/main/adam_mini/adam_mini.py",
        "upstream": "zyushun/Adam-mini adam_mini/adam_mini.py",
        "algorithmic_diff": True,
    },
    "adema_mix": {
        "builder": build_adema_mix_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/apple/ml-ademamix/main/pytorch/ademamix.py",
        "upstream": "apple/ml-ademamix pytorch/ademamix.py",
        "algorithmic_diff": False,
    },
    "mars": {
        "builder": build_mars_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/AGI-Arena/MARS/main/MARS/optimizers/mars.py",
        "upstream": "AGI-Arena/MARS MARS/optimizers/mars.py",
        "algorithmic_diff": True,
    },
    "soap": {
        "builder": build_soap_optimizer,
        "implemented": True,
        "source": "https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py",
        "upstream": "nikhilvyas/SOAP soap.py",
        "algorithmic_diff": False,
    },
}


def list_optimizer_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_optimizer_metadata(name: str) -> dict[str, Any]:
    normalized = str(name).strip().lower()
    if normalized not in _REGISTRY:
        raise KeyError(f"Unknown optimizer: {normalized}")
    entry = deepcopy(_REGISTRY[normalized])
    entry["name"] = normalized
    entry["diff_doc"] = str(diff_doc_path(normalized))
    return entry


def is_optimizer_implemented(name: str) -> bool:
    return bool(get_optimizer_metadata(name).get("implemented", False))


def build_optimizer(model: nn.Module, cfg: dict[str, Any]):
    training_cfg = cfg.get("training") or {}
    resolved = resolve_optimizer_cfg(training_cfg)
    modifiers = resolve_optimizer_modifiers(training_cfg)
    metadata = get_optimizer_metadata(resolved["name"])

    if modifiers["schedule_free"]:
        if resolved["name"] != "adamw":
            raise NotImplementedError("schedule_free modifier is currently implemented only for adamw.")
        optimizer = build_schedule_free_adamw(model.parameters(), resolved)
        metadata["source"] = "https://raw.githubusercontent.com/facebookresearch/schedule_free/main/schedulefree/adamw_schedulefree.py"
        metadata["upstream"] = "facebookresearch/schedule_free adamw_schedulefree.py"
        metadata["external_scheduler_allowed"] = False
    else:
        optimizer = metadata["builder"](model, resolved)
        metadata["external_scheduler_allowed"] = True

    if modifiers["cautious"]:
        optimizer = CautiousOptimizer(optimizer)
        metadata["algorithmic_diff"] = True

    metadata["modifiers"] = modifiers
    summary = optimizer_summary_payload(
        resolved_cfg=resolved,
        modifiers=modifiers,
        metadata=metadata,
        optimizer=optimizer,
    )
    return optimizer, resolved, metadata, summary
