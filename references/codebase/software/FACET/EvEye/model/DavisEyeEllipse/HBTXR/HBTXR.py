from typing import Any

import lightning
import torch
import torch.nn as nn
import torch.nn.init as init
from lightning.pytorch.utilities.types import (
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from timm.scheduler.step_lr import StepLRScheduler

from EvEye.model.DavisEyeEllipse.HBTXR.Backbone.DeiT import DeiTConfig, build_deit
from EvEye.model.DavisEyeEllipse.HBTXR.Head.HBTXRHead import HBTXRHead
from EvEye.model.DavisEyeEllipse.HBTXR.Loss import HBTXRCtdetLoss
from EvEye.model.DavisEyeEllipse.HBTXR.Metric import (
    cal_batch_ap,
    cal_batch_iou,
    cal_mean_distance,
    p_acc,
)
from EvEye.model.DavisEyeEllipse.HBTXR.Predict import post_process


LOSS_WEIGHT = {
    "hm_weight": 1,
    "ab_weight": 0.1,
    "ang_weight": 0,
    "trig_weight": 1,
    "reg_weight": 0.1,
    "iou_weight": 15,
    "mask_weight": 1,
}

HEAD_DICT = {"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1}


class HBTXR(lightning.LightningModule):
    def __init__(
        self,
        input_channels=2,
        img_size=256,
        patch_size=4,
        embed_dim=192,
        depth=8,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_eps=1e-6,
        pretrained=False,
        init_seed=0,
        projection_channels=64,
        projection_kernel_size=1,
        head_conv=256,
        head_dict=HEAD_DICT,
        loss_weight=LOSS_WEIGHT,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")

        self.criterion = HBTXRCtdetLoss(loss_weight)
        self.validation_outputs = []
        self.img_size = img_size
        self.patch_size = patch_size
        self.output_size = img_size // patch_size

        config = DeiTConfig(
            model_name="hbtxr_deit",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            num_classes=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_eps=norm_eps,
        )
        self.backbone = build_deit(config, init_seed=init_seed, pretrained=pretrained)
        pad = (projection_kernel_size - 1) // 2
        self.projection = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                projection_channels,
                kernel_size=projection_kernel_size,
                padding=pad,
                bias=False,
            ),
            nn.BatchNorm2d(projection_channels),
            nn.ReLU(inplace=True),
        )
        self.head = HBTXRHead(
            in_channels=projection_channels,
            head_conv=head_conv,
            head_dict=head_dict,
        )
        self._initialize_detection_layers()

    def _initialize_detection_layers(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone.forward_patch_map(x)
        features = self.projection(features)
        return self.head(features)

    def _log(self, name, metric):
        self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True)

    def set_optimizer_config(self, learning_rate: float, weight_decay: float):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None
    ) -> None:
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler = StepLRScheduler(
            optimizer, decay_t=10, decay_rate=0.7, warmup_lr_init=1e-5, warmup_t=5
        )
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def training_step(self, batch, batch_idx):
        input_tensor = batch["input"]
        pred = self(input_tensor)
        loss, loss_show = self.criterion(pred, batch)

        self._log("train_loss", loss)
        for key, value in loss_show.items():
            self._log(f"train_{key}", value)

        if getattr(self, "logger", None) is not None and hasattr(
            self.logger.experiment, "add_image"
        ):
            self.logger.experiment.add_image(
                "train_mask", pred["mask"][0], self.global_step
            )

        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor = batch["input"]
        center = batch["center"]
        close = batch["close"]
        ellipse = batch["ellipse"]
        pred = self(input_tensor)
        dets = post_process(pred)
        loss, _ = self.criterion(pred, batch)
        iou = cal_batch_iou(dets, ellipse, close)
        ap = cal_batch_ap(dets, ellipse, close, iou_thres=0.5, score_threshold=0.5)

        metrics = {
            "val_loss": loss,
            "val_p10_acc": p_acc(dets, center, close, 10),
            "val_p5_acc": p_acc(dets, center, close, 5),
            "val_p3_acc": p_acc(dets, center, close, 3),
            "val_p1_acc": p_acc(dets, center, close, 1),
            "val_mean_distance": cal_mean_distance(dets, center, close),
            "val_IoU": iou,
            "val_AP": ap,
        }

        if any(torch.isnan(tensor) for tensor in metrics.values()):
            return None

        self.validation_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        results = self.validation_outputs
        if not results:
            print("No validation results to process.")
            return

        metrics = {
            key: torch.stack([x[key] for x in results]).mean() for key in results[0]
        }
        for key, value in metrics.items():
            self._log(key, value)
        self.validation_outputs.clear()
