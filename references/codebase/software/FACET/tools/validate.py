import lightning
import torch
import click
import os

from EvEye.utils.scripts.load_config import load_config
from EvEye.logger.logger_factory import make_logger
from EvEye.callback.callback_factory import make_callbacks
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model


@click.command()
@click.option("--config", "-c", type=str, default="MemmapDavisEyeCenter_TennSt.yaml")
def main(config: str) -> None:
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

    model_cfg = config["model"]
    model = make_model(model_cfg)

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

    trainer.validate(
        model=model,
        dataloaders=val_dataloader,
        ckpt_path=config["val"].get("ckpt_path"),
    )


if __name__ == "__main__":
    main()
