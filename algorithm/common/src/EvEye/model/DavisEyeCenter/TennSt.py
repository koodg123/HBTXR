import debugpy

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from typing import Any
import cv2
import time

import lightning
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from timm.scheduler.step_lr import StepLRScheduler

from EvEye.dataset.DavisEyeCenter.losses import *
from functools import partial
import warnings
from EvEye.dataset.DavisEyeCenter.losses import process_detector_prediction

warnings.formatwarning = (
    lambda message, category, filename, lineno, line=None: f"{category.__name__}: {message}\n"
)


class ActivateLayer(nn.Module):
    """A simple activation layer that uses ReLU as the activation function."""

    def __init__(self):
        super().__init__()
        self.act_layer = nn.ReLU()

    def forward(self, x):
        return self.act_layer(x)


class BatchNormBlock(nn.Module):
    """A simple batch normalization block that uses BatchNorm3d as the normalization layer."""

    def __init__(self, features):
        super().__init__()
        self.bn_block = nn.Sequential(nn.BatchNorm3d(features), ActivateLayer())

    def forward(self, x):
        return self.bn_block(x)


class CausalGroupNormBlock(nn.GroupNorm):
    """A GroupNorm that does not use temporal statistics, to ensure causality"""

    def __init__(self, num_groups, num_channels, **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)

    def forward(self, input):
        x = input.moveaxis(1, 2)  # (B, T, C, H, W)
        x_shape = x.shape
        x = x.flatten(0, 1)  # (B * T, C, H, W)
        x = super().forward(x).reshape(x_shape)
        return x.moveaxis(1, 2)  # (B, C, T, H, W)


class GroupNormBlock(nn.Module):
    """A simple group normalization block that uses GroupNorm as the normalization layer."""

    def __init__(self, features):
        super().__init__()
        self.gn_block = nn.Sequential(
            CausalGroupNormBlock(4, features), ActivateLayer()
        )

    def forward(self, x):
        return self.gn_block(x)


class PointWiseConv(nn.Module):
    """A simple pointwise convolution block that uses Conv3d as the convolution layer."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_block = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pw_block(x)


class SpatialBlock(nn.Module):  # Define the spatial processing block class inherited from nn.Module
    def __init__(
        self,
        in_channels,  # Number of input channels
        out_channels,  # Number of output channels
        depthwise=False,  # Whether to use depthwise separable convolution
        kernel_size=1,  # Convolution kernel size
        full_conv3d=False,  # Whether to use full 3D convolution
        norms="mixed",  # Normalization type: mixed or all group norm
    ):
        super().__init__()  # Initialize the parent class
        kernel = (kernel_size, 3, 3)  # Define the convolution kernel size
        self.kernel_size = kernel_size  # Store the convolution kernel size
        self.full_conv3d = full_conv3d  # Store whether full 3D convolution is used
        self.norms = norms  # Store the normalization type
        self.streaming_mode = False  # Streaming mode, disabled by default
        self.fifo = None  # FIFO buffer used for streaming

        if self.norms == "all_gn":  # If the normalization type is all group norm
            norm_block = GroupNormBlock  # Use a group-normalization block
        else:
            norm_block = BatchNormBlock  # Otherwise use a batch-normalization block

        if depthwise:  # If depthwise separable convolution is used
            self.block = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel,
                    stride=(1, 2, 2),
                    padding=(0, 1, 1),
                    groups=in_channels,
                    bias=False,
                ),  # Depthwise separable convolution layer
                norm_block(in_channels),  # Normalization layer
                PointWiseConv(in_channels, out_channels),  # Pointwise convolution that changes the channel count
                norm_block(out_channels),  # Normalize again
            )

        else:  # If depthwise separable convolution is not used
            self.block = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel, (1, 2, 2), (0, 1, 1), bias=False
                ),  # Standard convolution layer
                norm_block(out_channels),  # Normalization layer
            )

    def streaming(self, enabled=True):  # Set whether streaming mode is enabled
        if enabled:
            assert (
                not self.training
            ), "Can only use streaming mode during evaluation."  # Assert that streaming mode is only enabled during evaluation
        self.streaming_mode = enabled  # Store the streaming mode state

    def reset_memory(self):  # Reset memory or buffers
        self.fifo = None  # Clear the FIFO buffer

    def forward(self, input):  # Define the forward pass
        if self.full_conv3d:  # If full 3D convolution is used
            if self.streaming_mode:  # If streaming is enabled
                return self._streaming_forward(input)  # Run streaming forward propagation
            input = F.pad(
                input, (0, 0, 0, 0, self.kernel_size - 1, 0)
            )  # Pad the input data
            return self.block(input)  # Process data through the convolution block
        else:
            return self.block(input)  # Process data directly through the convolution block

    def _streaming_forward(self, input):  # Define the streaming forward pass
        if self.fifo is None:  # If the FIFO buffer is not initialized
            self.fifo = torch.zeros(
                *input.shape[:2], self.kernel_size, *input.shape[3:]
            ).type_as(
                input
            )  # Initialize the FIFO buffer
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)  # Update the FIFO buffer
        return self.block(self.fifo)  # Process the FIFO data through the convolution block


class TemporalBlock(nn.Module):  # Define the temporal processing block class inherited from nn.Module
    def __init__(
        self,
        in_channels,  # Number of input channels
        out_channels,  # Number of output channels
        kernel_size=3,  # Convolution kernel size, defaulting to 3
        depthwise=False,  # Whether to use depthwise separable convolution
        full_conv3d=False,  # Whether to use full 3D convolution
        norms="mixed",  # Normalization type: mixed, all batch norm, or all group norm
    ):
        super().__init__()  # Initialize the parent class
        assert out_channels % 4 == 0  # Ensure the output channel count is a multiple of 4, which group norm requires
        self.kernel_size = kernel_size  # Store the convolution kernel size
        self.depthwise = depthwise  # Store whether depthwise separable convolution is used
        self.norms = norms  # Store the normalization type
        kernel = (
            (kernel_size, 3, 3) if full_conv3d else (kernel_size, 1, 1)
        )  # Choose the kernel shape based on whether full 3D convolution is used

        self.streaming_mode = False  # Streaming mode, disabled by default
        self.fifo = None  # FIFO buffer used for streaming

        if self.norms == "mixed":  # If the normalization type is mixed
            norm1_block = BatchNormBlock  # Use batch normalization for the first normalization layer
            norm2_block = GroupNormBlock  # Use group normalization for the second normalization layer
        elif self.norms == "all_bn":  # If the normalization type is all batch norm
            norm1_block = BatchNormBlock
            norm2_block = BatchNormBlock
        elif self.norms == "all_gn":  # If the normalization type is all group norm
            norm1_block = GroupNormBlock
            norm2_block = GroupNormBlock

        if depthwise:  # If depthwise separable convolution is used
            self.block = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel,
                    groups=in_channels,
                    bias=False,
                ),  # Depthwise separable convolution layer
                norm1_block(in_channels),  # First normalization layer
                PointWiseConv(in_channels, out_channels),  # Pointwise convolution that changes the channel count
                norm2_block(out_channels),  # Second normalization layer
            )

        else:  # If depthwise separable convolution is not used
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, bias=False),  # Standard convolution layer
                norm2_block(out_channels),  # Normalization layer
            )

    def streaming(self, enabled=True):  # Set whether streaming mode is enabled
        if enabled:
            assert (
                not self.training
            ), "Can only use streaming mode during evaluation."  # Assert that streaming mode is only enabled during evaluation
        self.streaming_mode = enabled  # Store the streaming mode state

    def reset_memory(self):  # Reset memory or buffers
        self.fifo = None  # Clear the FIFO buffer

    def forward(self, input):  # Define the forward pass
        if self.streaming_mode:  # Check whether streaming mode is enabled
            return self._streaming_forward(
                input
            )  # If streaming is enabled, call the streaming forward method

        # If streaming is disabled, prepare the input for convolution
        input = F.pad(
            input, (0, 0, 0, 0, self.kernel_size - 1, 0)
        )  # Pad the input, mainly along the temporal dimension, to avoid losing data during convolution
        # Parameter explanation:
        # (0, 0, 0, 0, self.kernel_size - 1, 0) the fifth parameter, self.kernel_size - 1, is the key padding value; it adds self.kernel_size - 1 zeros at the start of the temporal dimension,
        # which ensures the convolution kernel covers enough temporal context and preserves temporal continuity and completeness.

        return self.block(
            input
        )  # Apply the defined convolution block to the processed input data and return the result
        # self.block is a module containing a series of convolution layers and possibly normalization layers, such as an nn.Sequential container,
        # and it performs the actual data processing tasks such as feature extraction.

    def _streaming_forward(self, input):  # Define the streaming forward pass
        if self.fifo is None:  # If the FIFO buffer is not initialized
            self.fifo = torch.zeros(
                *input.shape[:2], self.kernel_size, *input.shape[3:]
            ).type_as(
                input
            )  # Initialize the FIFO buffer
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)  # Update the FIFO buffer
        return self.block(self.fifo)  # Process the FIFO data through the convolution block


class TennSt(lightning.LightningModule):
    # The TennSt class is a network module inherited from PyTorch nn.Module for spatiotemporal data processing.
    def __init__(
        self,
        channels,
        t_kernel_size,
        n_depthwise_layers,
        detector_head,
        detector_depthwise,
        full_conv3d=False,
        norms="mixed",
        activity_regularization=0,
    ):
        super().__init__()

        self.loss_fn = Losses(detector_head, activity_regularization, self)
        self.metric_p1fn = partial(p_acc, tolerance=1, detector_head=detector_head)
        self.metric_p3fn = partial(p_acc, tolerance=3, detector_head=detector_head)
        self.metric_p5fn = partial(p_acc, tolerance=5, detector_head=detector_head)
        self.metric_p10fn = partial(p_acc, tolerance=10, detector_head=detector_head)

        self.detector = detector_head
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        temporals = [True, False] * 5

        self.backbone = nn.Sequential()
        for i in range(len(depthwises)):
            in_channels, out_channels = (
                channels[i],
                channels[i + 1],
            )
            depthwise = depthwises[i]
            temporal = temporals[i]

            if temporal:
                self.backbone.append(
                    TemporalBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=t_kernel_size,
                        depthwise=depthwise,
                        full_conv3d=full_conv3d,
                        norms=norms,
                    )
                )
            else:
                self.backbone.append(
                    SpatialBlock(
                        in_channels,
                        out_channels,
                        depthwise=depthwise,
                        full_conv3d=full_conv3d,
                        kernel_size=t_kernel_size if full_conv3d else 1,
                        norms=norms,
                    )
                )

        if detector_head:
            self.head = nn.Sequential(
                TemporalBlock(
                    channels[-1],
                    channels[-1],
                    t_kernel_size,
                    depthwise=detector_depthwise,
                ),
                nn.Conv3d(channels[-1], channels[-1], (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                ActivateLayer(),
                nn.Conv3d(channels[-1], 3, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv1d(channels[-1], channels[-1], 1),
                ActivateLayer(),
                nn.Conv1d(channels[-1], 2, 1),
            )

    @staticmethod
    def streaming_inference(model, frames):
        model.eval()
        model.streaming()
        model.reset_memory()

        predictions = []
        inference_times = []
        with torch.inference_mode():
            for frame_id in range(frames.shape[2]):  # stream the frames to the model
                start_time = time.time()
                prediction = model(frames[:, :, [frame_id]])
                end_time = time.time()
                inference_times.append(end_time - start_time)
                predictions.append(prediction)

        predictions = torch.cat(predictions, dim=2)
        return predictions, inference_times

    def streaming(self, enabled=True):
        if enabled:
            warnings.warn(
                "You have enabled the streaming mode of the network. It is expected, but not checked, that the input will be of shape (batch, 1, H, W)."
            )
        for name, module in self.named_modules():
            if name and hasattr(module, "streaming"):
                module.streaming(enabled)

    def reset_memory(self):
        for name, module in self.named_modules():
            if name and hasattr(module, "reset_memory"):
                module.reset_memory()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.detector:
            return self.head((self.backbone(input)))
        else:
            return self.head(self.backbone(input).mean((-2, -1)))

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
        super().configure_optimizers()
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def training_step(self, batch, batch_idx):
        event, center, openness = batch
        height, width = event.shape[-2], event.shape[-1]
        pred = self(event)
        loss = self.loss_fn(pred, center, openness)
        p1_acc, _, _ = self.metric_p1fn(pred, center, openness, height, width)
        p3_acc, _, _ = self.metric_p3fn(pred, center, openness, height, width)
        p5_acc, _, _ = self.metric_p5fn(pred, center, openness, height, width)
        p10_acc, metric_noblinks, distance = self.metric_p10fn(
            pred, center, openness, height, width
        )
        self._log("train_loss", loss)
        self._log("train_p1_acc", p1_acc)
        self._log("train_p3_acc", p3_acc)
        self._log("train_p5_acc", p5_acc)
        self._log("train_p10_acc", p10_acc)
        self._log("train_metric_noblinks", metric_noblinks)
        self._log("train_distance", distance)
        return loss

    def validation_step(self, batch, batch_idx):
        event, center, openness = batch
        height, width = event.shape[-2], event.shape[-1]
        pred = self(event)
        loss = self.loss_fn(pred, center, openness)
        p1_acc, _, _ = self.metric_p1fn(pred, center, openness, 64, 64)
        p3_acc, _, _ = self.metric_p3fn(pred, center, openness, 64, 64)
        p5_acc, _, _ = self.metric_p5fn(pred, center, openness, 64, 64)
        p10_acc, metric_noblinks, distance = self.metric_p10fn(
            pred, center, openness, 64, 64
        )
        pred_frame_list = self.visualize(event, center, pred)
        # self.logger.experiment.add_image()
        self._log("val_loss", loss)
        self._log("val_p1_acc", p1_acc)
        self._log("val_p3_acc", p3_acc)
        self._log("val_p5_acc", p5_acc)
        self._log("val_p10_acc", p10_acc)
        self._log("val_metric_noblinks", metric_noblinks)
        self._log("val_distance", distance)
        for index in range(len(pred_frame_list)):
            self.logger.experiment.add_image(
                f"pred_frames_val/pred_frame_{index:03}",
                pred_frame_list[index],
                dataformats="HWC",
                global_step=self.global_step,
            )

    def on_train_epoch_start(self) -> None:
        return super().on_train_epoch_start()

    def visualize(self, event, center, pred):
        event, center, pred = event.clone(), center.clone(), pred.clone()
        event = event[0]
        event = event.permute(1, 0, 2, 3)
        event = event.detach().cpu().numpy().astype(np.int32)
        t, n, h, w = event.shape

        center = center[0]
        center = center.permute(1, 0)
        center[:, 1] *= h
        center[:, 0] *= w
        center = center.detach().cpu().numpy().astype(np.int32)

        pred = process_detector_prediction(pred)
        pred = pred[0]
        pred = pred.permute(1, 0)
        pred[:, 1] *= h
        pred[:, 0] *= w
        pred = pred.detach().cpu().numpy().astype(np.int32)

        frame_list = []
        for i in range(t):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            off_frame_stack = event[i, 0]
            on_frame_stack = event[i, 1]
            center_x = center[i, 0]
            center_y = center[i, 1]
            pred_x = pred[i, 0]
            pred_y = pred[i, 1]
            canvas[off_frame_stack > 20] = [0, 0, 255]
            canvas[on_frame_stack > 20] = [255, 0, 0]
            canvas = cv2.circle(canvas, (center_x, center_y), 3, (255, 255, 255), -1)
            canvas = cv2.circle(canvas, (pred_x, pred_y), 3, (0, 255, 0), -1)
            frame_list.append(canvas)
        return frame_list


def main():
    input = torch.randn(32, 2, 50, 96, 128)
    # weights = torch.load()
    model = TennSt(
        channels=[2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256],
        t_kernel_size=5,
        n_depthwise_layers=4,
        detector_head=True,
        detector_depthwise=True,
        full_conv3d=False,
        norms="mixed",
    )
    output = model(input)
    print(output.shape)
    # print(model)


if __name__ == "__main__":
    main()
