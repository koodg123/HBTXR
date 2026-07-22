"""Inspect Davis eye-center datasets without burdening their data owner."""

from torch.utils.data import DataLoader

from dataset.DavisEyeCenter.DavisEyeCenterDataset import (
    DavisEyeCenterDataset,
)
from dataset.DavisEyeCenter.MemmapDavisEyeCenterDataset import (
    MemmapDavisEyeCenterDataset,
)


def inspect_davis_eye_center() -> None:
    dataset = DavisEyeCenterDataset(
        root_path="/mnt/data2T/junyuan/eye-tracking/testDataset",
        split="train",
        time_window=40000,
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="causal_linear",
        spatial_affine=True,
        temporal_flip=True,
        temporal_scale=True,
        temporal_shift=True,
        cache=False,
    )
    print(len(dataset))
    dataset[101]
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(len(dataset))
    for index, (event, center, close) in enumerate(dataloader):
        print(f"Batch {index + 1}:")
        print(f"Data shape: {event.shape}")
        print(f"Data dtype: {event.dtype}")
        print(f"Label shape: {center.shape}")
        print(f"Label dtype: {center.dtype}")
        print(f"Close shape: {close.shape}")
        print(f"Close dtype: {close.dtype}")
        print()


def inspect_memmap_davis_eye_center() -> None:
    from event.models.TennSt import TennSt

    dataset = MemmapDavisEyeCenterDataset(
        root_path="/mnt/data2T/junyuan/eye-tracking/datasets/MemmapDavisEyeCenterDataset",
        split="train",
        time_window=40000,
        frames_per_segment=50,
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear",
        spatial_downsaple=False,
        saptial_transform=True,
        temporal_transform=True,
    )

    _model = TennSt(
        channels=[2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256],
        t_kernel_size=5,
        n_depthwise_layers=4,
        detector_head=True,
        detector_depthwise=True,
        full_conv3d=False,
        norms="mixed",
    )

    dataset[120]
    print(f"Total segments: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for index, (event, center, close) in enumerate(dataloader):
        print(f"Batch {index + 1}:")
        print(f"Data shape: {event.shape}")
        print(f"Data dtype: {event.dtype}")
        print(f"Label shape: {center.shape}")
        print(f"Label dtype: {center.dtype}")
        print(f"Close shape: {close.shape}")
        print(f"Close dtype: {close.dtype}")
        print()


def main() -> None:
    inspect_davis_eye_center()


if __name__ == "__main__":
    main()
