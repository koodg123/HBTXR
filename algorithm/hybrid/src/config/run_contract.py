from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import yaml

from src.reproduction import write_reproduction_manifest

from .runtime_config import resolve_mode_contract


_TIMESTAMP_SUFFIX = re.compile(r".*_\d{8}_\d{6}$")


def resolve_project_path(path_value: str | Path | None, *, project_root: str | Path) -> Path | None:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (Path(project_root) / path).resolve()


def resolve_experiment_name(cfg: dict, *, config_path: str | Path) -> str:
    experiment_cfg = cfg.get("experiment") or {}
    raw = str(experiment_cfg.get("name", "")).strip()
    if raw:
        return raw
    return Path(config_path).stem


def materialize_experiment_name(experiment_name: str, *, timestamp: str | None = None, resume: bool = False) -> str:
    base = str(experiment_name).strip() or "hbtxr"
    if resume or _TIMESTAMP_SUFFIX.match(base):
        return base
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{stamp}"


def latest_run_root(project_root: str | Path, experiment_name: str) -> Path | None:
    runs_root = Path(project_root) / "runs"
    raw = str(experiment_name).strip()
    if not raw:
        return None
    if _TIMESTAMP_SUFFIX.match(raw):
        for pattern in (raw, f"XR-*/{raw}", f"NON_XR/*/{raw}"):
            roots = sorted(path for path in runs_root.glob(pattern) if path.is_dir())
            if roots:
                return roots[-1]
        return None
    roots = sorted(
        path
        for pattern in (f"{raw}_*", f"XR-*/*{raw}_*", f"NON_XR/*/{raw}_*")
        for path in runs_root.glob(pattern)
        if path.is_dir()
    )
    return roots[-1] if roots else None


def resolve_resume_root(project_root: str | Path, experiment_name: str, resume_arg: str | None) -> Path | None:
    if resume_arg is None:
        return None
    raw = str(resume_arg).strip()
    if raw in {"", "auto", "latest"}:
        return latest_run_root(project_root, experiment_name)
    path = resolve_project_path(raw, project_root=project_root)
    return None if path is None else path


def default_checkpoint_for_stage(run_root: str | Path, stage: str) -> Path:
    filename = "best_track_p10.pt" if str(stage).strip().lower() == "stage2" else "best_search_p10.pt"
    return Path(run_root) / "train" / filename


def _default_manifest_path(*, project_root: str | Path, split: str, manifest_name: str) -> Path:
    return Path(project_root) / "manifests" / manifest_name / f"{split}_manifest.jsonl"


def resolve_manifest_path(
    cfg: dict,
    *,
    project_root: str | Path,
    split: str,
    manifest_override: str | Path | None = None,
) -> Path:
    experiment_cfg = cfg.get("experiment") or {}
    split_name = str(split).strip().lower()
    mapping = {
        "train": experiment_cfg.get("train_manifest"),
        "val": experiment_cfg.get("val_manifest"),
        "test": experiment_cfg.get("test_manifest"),
    }
    manifest = resolve_project_path(
        manifest_override if manifest_override is not None else mapping.get(split_name),
        project_root=project_root,
    )
    if manifest is None:
        manifest = _default_manifest_path(
            project_root=project_root,
            split=split_name,
            manifest_name=resolve_mode_contract(cfg.get("data") or {})["manifest_name"],
        )
    return manifest


def _reference_only_payload(section: Any, *, feature_name: str) -> dict[str, Any] | None:
    if not isinstance(section, dict):
        return None
    payload = deepcopy(section)
    enabled = bool(payload.get("enabled", False))
    payload.setdefault("enabled", enabled)
    payload.setdefault("mode", "available")
    payload["feature_name"] = feature_name
    payload["implemented"] = True
    payload["active_wiring"] = enabled
    payload["status"] = "enabled" if enabled else "available_inactive"
    return payload


def collect_reference_only_reports(cfg: dict) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for feature_name in ("ssl", "pruning"):
        payload = _reference_only_payload(cfg.get(feature_name), feature_name=feature_name)
        if payload is not None:
            reports[feature_name] = payload
    return reports


class RunContractResolver:
    def __init__(self, cfg: dict, *, config_path: str | Path, project_root: str | Path) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.project_root = project_root

    def resolve_run_contract(self, *, action: str, resume_root: str | Path | None = None) -> dict[str, str]:
        experiment_cfg = self.cfg.get("experiment") or {}
        run_cfg = self.cfg.get("run") or {}
        base_name = str(
            run_cfg.get("materialized_experiment_name")
            or experiment_cfg.get("materialized_name")
            or resolve_experiment_name(self.cfg, config_path=self.config_path)
        ).strip()
        if resume_root is not None:
            run_root = resolve_project_path(resume_root, project_root=self.project_root)
            if run_root is None:
                raise ValueError("resume_root could not be resolved")
            materialized_name = run_root.name
        else:
            materialized_name = materialize_experiment_name(base_name)
            run_root = Path(self.project_root) / "runs" / materialized_name
        return {
            "action": str(action),
            "root": str(run_root),
            "materialized_experiment_name": materialized_name,
            "train_dir": str(run_root / "train"),
            "eval_dir": str(run_root / "eval"),
            "infer_dir": str(run_root / "infer"),
            "vis_dir": str(run_root / "vis"),
            "hypers_dir": str(run_root / "hypers"),
        }

    def write_run_artifacts(
        self,
        *,
        run_contract: dict[str, str],
        cli_args: list[str],
        overrides: list[str],
        device: str | None,
        pretrained_report_name: str | None = None,
    ) -> None:
        hypers_dir = Path(run_contract["hypers_dir"])
        hypers_dir.mkdir(parents=True, exist_ok=True)
        resolved_yaml = hypers_dir / "resolved_config.yaml"
        resolved_json = hypers_dir / "resolved_config.json"
        resolved_yaml.write_text(yaml.safe_dump(self.cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        resolved_json.write_text(json.dumps(self.cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        (hypers_dir / "cli_args.txt").write_text(" ".join(cli_args).strip(), encoding="utf-8")
        (hypers_dir / "overrides.txt").write_text("\n".join(overrides) + ("\n" if overrides else ""), encoding="utf-8")
        (hypers_dir / "device.txt").write_text((device or "").strip(), encoding="utf-8")
        run_contract_payload = {
            **run_contract,
            "config_path": str(self.config_path),
            "mode_contract": resolve_mode_contract(self.cfg.get("data") or {}),
        }
        (hypers_dir / "run_contract.json").write_text(json.dumps(run_contract_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        for feature_name, payload in collect_reference_only_reports(self.cfg).items():
            (hypers_dir / f"{feature_name}_reference.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if self.cfg.get("paper"):
            write_reproduction_manifest(
                self.cfg,
                project_root=self.project_root,
                output_dir=run_contract["root"],
                artifacts={"run_contract": run_contract_payload},
            )
        if pretrained_report_name:
            target = hypers_dir / pretrained_report_name
            if not target.exists():
                target.write_text(json.dumps({"status": "pending"}, indent=2, ensure_ascii=False), encoding="utf-8")

    def resolve_training_entry(
        self,
        *,
        train_manifest_override: str | Path | None = None,
        val_manifest_override: str | Path | None = None,
        output_override: str | Path | None = None,
        init_checkpoint_override: str | Path | None = None,
    ) -> dict[str, Any]:
        training_cfg = self.cfg.get("training") or {}
        experiment_cfg = self.cfg.get("experiment") or {}
        stage = str(training_cfg.get("stage", "stage1")).strip().lower()
        experiment_name = resolve_experiment_name(self.cfg, config_path=self.config_path)
        mode_contract = resolve_mode_contract(self.cfg.get("data") or {})

        train_manifest = resolve_manifest_path(
            self.cfg,
            project_root=self.project_root,
            split="train",
            manifest_override=train_manifest_override,
        )
        val_manifest = resolve_manifest_path(
            self.cfg,
            project_root=self.project_root,
            split="val",
            manifest_override=val_manifest_override,
        )
        if not val_manifest.exists() and val_manifest_override is None and experiment_cfg.get("val_manifest") is None:
            val_manifest = None

        output_dir = resolve_project_path(
            output_override if output_override is not None else experiment_cfg.get("output_dir"),
            project_root=self.project_root,
        )
        if output_dir is None:
            run_contract = self.resolve_run_contract(action="train")
            output_dir = Path(run_contract["train_dir"])

        init_checkpoint = resolve_project_path(
            init_checkpoint_override if init_checkpoint_override is not None else experiment_cfg.get("init_checkpoint"),
            project_root=self.project_root,
        )

        return {
            "stage": stage,
            "mode": mode_contract["mode"],
            "canonical_name": mode_contract["canonical_name"],
            "manifest_name": mode_contract["manifest_name"],
            "experiment_name": experiment_name,
            "train_manifest": str(train_manifest),
            "val_manifest": None if val_manifest is None else str(val_manifest),
            "output_dir": str(output_dir),
            "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
        }


def resolve_run_contract(
    cfg: dict,
    *,
    config_path: str | Path,
    project_root: str | Path,
    action: str,
    resume_root: str | Path | None = None,
) -> dict[str, str]:
    resolver = RunContractResolver(cfg, config_path=config_path, project_root=project_root)
    return resolver.resolve_run_contract(action=action, resume_root=resume_root)


def write_run_artifacts(
    *,
    run_contract: dict[str, str],
    cfg: dict,
    config_path: str | Path,
    cli_args: list[str],
    overrides: list[str],
    device: str | None,
    pretrained_report_name: str | None = None,
) -> None:
    resolver = RunContractResolver(cfg, config_path=config_path, project_root=Path(run_contract["root"]).parents[0].parents[0])
    resolver.write_run_artifacts(
        run_contract=run_contract,
        cli_args=cli_args,
        overrides=overrides,
        device=device,
        pretrained_report_name=pretrained_report_name,
    )


def resolve_training_entry(
    cfg: dict,
    *,
    config_path: str | Path,
    project_root: str | Path,
    train_manifest_override: str | Path | None = None,
    val_manifest_override: str | Path | None = None,
    output_override: str | Path | None = None,
    init_checkpoint_override: str | Path | None = None,
) -> dict[str, Any]:
    resolver = RunContractResolver(cfg, config_path=config_path, project_root=project_root)
    return resolver.resolve_training_entry(
        train_manifest_override=train_manifest_override,
        val_manifest_override=val_manifest_override,
        output_override=output_override,
        init_checkpoint_override=init_checkpoint_override,
    )


__all__ = [
    "RunContractResolver",
    "collect_reference_only_reports",
    "default_checkpoint_for_stage",
    "latest_run_root",
    "materialize_experiment_name",
    "resolve_experiment_name",
    "resolve_manifest_path",
    "resolve_project_path",
    "resolve_resume_root",
    "resolve_run_contract",
    "resolve_training_entry",
    "write_run_artifacts",
]
