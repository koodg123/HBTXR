import lightning
import torch
import click
import numpy as np
import os

from EvEye.utils.scripts.load_config import load_config
from EvEye.logger.logger_factory import make_logger
from EvEye.callback.callback_factory import make_callbacks
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model


@click.command()
@click.option("--config", "-c", type=str, default="MemmapDavisEyeCenter_TennSt.yaml")
@click.option("--num_validations", "-n", type=int, default=10)
def main(config: str, num_validations: int) -> None:
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

    metrics_list = []

    for _ in range(num_validations):
        metrics = trainer.validate(
            model=model,
            dataloaders=val_dataloader,
            ckpt_path=config["val"].get("ckpt_path"),
        )
        metrics_list.append(metrics)

    # Compute averages
    avg_metrics = {
        key: np.mean([m[0][key] for m in metrics_list])
        for key in metrics_list[0][0].keys()
    }

    # Print averages
    print("Average Metrics over {} validations:".format(num_validations))
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")

    # Get the checkpoint file path and extract the file name
    ckpt_path = config["val"].get("ckpt_path")
    if ckpt_path:
        parent_dir = os.path.dirname(os.path.dirname(ckpt_path))
        dir_name = os.path.basename(parent_dir)
        output_path = "/mnt/data2T/junyuan/eye-tracking/Results"
        os.makedirs(output_path, exist_ok=True)
        result_file_name = f"{output_path}/{dir_name}.txt"

        # Write the results to a file
        with open(result_file_name, 'w') as f:
            f.write("Average Metrics over {} validations:\n".format(num_validations))
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
