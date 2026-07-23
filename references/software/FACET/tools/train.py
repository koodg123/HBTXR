import click
import pdb

import torch
import lightning
import os


from EvEye.utils.scripts.load_config import load_config
from EvEye.logger.logger_factory import make_logger
from EvEye.callback.callback_factory import make_callbacks
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model


@click.command()
@click.option("-c", "--config", type=str, default="DavisEyeEllipse_EPNet.yaml")
def main(config: str) -> None:
    lightning.seed_everything(42)
    torch.multiprocessing.set_start_method("spawn", force=True)
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

    if os.environ.get("SM_CHANNEL_ROOT"):
        config["dataloader"]["train"]["dataset"]["root_path"] = os.environ[
            "SM_CHANNEL_ROOT"
        ]
        config["dataloader"]["val"]["dataset"]["root_path"] = os.environ[
            "SM_CHANNEL_ROOT"
        ]
    train_dataloader = make_dataloader(config["dataloader"]["train"])
    val_dataloader = make_dataloader(config["dataloader"]["val"])

    torch.multiprocessing.current_process()._children = None

    model_cfg = config["model"]
    model = make_model(model_cfg)
    if "optimizer" in config["train"].keys():
        model.set_optimizer_config(**config["train"]["optimizer"])

    trainer_cfg = config.get("trainer", {})
    devices = os.environ.get("FACET_DEVICES", trainer_cfg.get("devices", "auto"))
    if isinstance(devices, str) and "," in devices:
        devices = [int(device.strip()) for device in devices.split(",")]
    elif isinstance(devices, str) and devices.isdigit():
        devices = [int(devices)]

    optional_trainer_keys = [
        "accumulate_grad_batches",
        "fast_dev_run",
        "limit_train_batches",
        "limit_val_batches",
        "num_sanity_val_steps",
        "default_root_dir",
        "precision",
    ]
    optional_trainer_kwargs = {
        key: trainer_cfg[key] for key in optional_trainer_keys if key in trainer_cfg
    }

    trainer = lightning.Trainer(
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=devices,
        max_epochs=config["train"].get("max_epochs", 50),
        check_val_every_n_epoch=config["train"].get("check_val_every_n_epoch", 1),
        logger=make_logger(config["logger"]),
        callbacks=make_callbacks(config["callback"]),
        **optional_trainer_kwargs,
    )

    ckpt_path = os.environ.get("FACET_CKPT_PATH") or config["train"].get("ckpt_path")
    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}")

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
