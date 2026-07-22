import os

import click
import lightning
import numpy as np
import torch

from common.paths import is_configured, resolve
from dataset.dataset_factory import make_dataloader
from engine.callback.callback_factory import make_callbacks
from engine.logger.logger_factory import make_logger
from engine.model_factory import make_model
from engine.tools.load_config import load_config


@click.command()
@click.option("--config", "-c", type=str, default="MemmapDavisEyeCenter_TennSt.yaml")
@click.option("--repeat", "-n", type=int, default=1, help="number of validation passes to average")
def main(config: str, repeat: int) -> None:
    """Validate a model. With --repeat > 1 the metrics are averaged.

    This replaces the former validate.py and validate10times.py; a single pass
    (repeat=1) reproduces validate.py, and repeat=N reproduces validate10times.
    """
    torch.set_float32_matmul_precision("medium")
    config = load_config(config)
    runtime_cfg = config.get("runtime", {})
    disable_cudnn = os.environ.get("FACET_DISABLE_CUDNN", "").lower() in {
        "1",
        "true",
        "yes",
    } or runtime_cfg.get("disable_cudnn", False)
    if disable_cudnn:
        torch.backends.cudnn.enabled = False

    val_dataloader = make_dataloader(config["dataloader"]["val"])
    model = make_model(config["model"])

    trainer_cfg = config.get("trainer", {})
    devices = os.environ.get("FACET_DEVICES", trainer_cfg.get("devices", "auto"))
    if isinstance(devices, str) and "," in devices:
        devices = [int(device.strip()) for device in devices.split(",")]
    elif isinstance(devices, str) and devices.isdigit():
        devices = [int(devices)]

    trainer = lightning.Trainer(
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=devices,
        max_epochs=config["train"].get("max_epochs", 50),
        check_val_every_n_epoch=1,
        logger=make_logger(config["logger"]),
        callbacks=make_callbacks(config["callback"]),
    )

    ckpt_path = config["val"].get("ckpt_path")
    metrics_list = []
    for _ in range(max(1, repeat)):
        metrics = trainer.validate(model=model, dataloaders=val_dataloader, ckpt_path=ckpt_path)
        metrics_list.append(metrics)

    if repeat <= 1:
        return

    avg_metrics = {
        key: np.mean([m[0][key] for m in metrics_list])
        for key in metrics_list[0][0].keys()
    }
    print("Average Metrics over {} validations:".format(repeat))
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")

    # Results are written under the configured output root instead of a
    # hard-coded absolute path (see common.paths / algorithm/.env.example).
    if ckpt_path and is_configured("output"):
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
        result_file = resolve("output", "results", f"{dir_name}.txt")
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            f.write("Average Metrics over {} validations:\n".format(repeat))
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
