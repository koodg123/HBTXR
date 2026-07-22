from typing import Any, Type
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataset.DavisWithMask.DavisWithMaskDataset import DavisWithMaskDataset
from dataset.Test.TestDataset import TestDataset
from dataset.DavisEyeCenter.DavisEyeCenterDataset import DavisEyeCenterDataset
from dataset.DavisEyeCenter.NpyDavisEyeCenterDataset import (
    NpyDavisEyeCenterDataset,
)
from dataset.DavisEyeCenter.DatDavisEyeCenterDataset import (
    DatDavisEyeCenterDataset,
)
from dataset.DavisEyeCenter.MemmapDavisEyeCenterDataset import (
    MemmapDavisEyeCenterDataset,
)
from dataset.DavisEyeCenter.TestTextDavisEyeDataset import TestTextDavisEyeDataset
from dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset
from dataset.DavisEyeEllipse.DavisEyeEllipseFrameDataset import (
    DavisEyeEllipseFrameDataset,
)
from dataset.DavisEyeEllipse.DavisEyeEllipseCenterSequenceDataset import (
    DavisEyeEllipseCenterSequenceDataset,
)

DATASET_CLASSES: dict[str, Type[Dataset]] = dict(
    DavisWithMaskDataset=DavisWithMaskDataset,
    TestDataset=TestDataset,
    DavisEyeCenterDataset=DavisEyeCenterDataset,
    NpyDavisEyeCenterDataset=NpyDavisEyeCenterDataset,
    DatDavisEyeCenterDataset=DatDavisEyeCenterDataset,
    MemmapDavisEyeCenterDataset=MemmapDavisEyeCenterDataset,
    TestTextDavisEyeDataset=TestTextDavisEyeDataset,
    DavisEyeEllipseDataset=DavisEyeEllipseDataset,
    DavisEyeEllipseFrameDataset=DavisEyeEllipseFrameDataset,
    DavisEyeEllipseCenterSequenceDataset=DavisEyeEllipseCenterSequenceDataset,
)


def worker_init_fn(worker_id):
    print(f"Worker {worker_id} initializing")
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Each worker gets a new reference to the memory-mapped file
    # dataset.data = np.memmap(
    #     dataset.data.filename,
    #     dtype=dataset.data.dtype,
    #     mode="r",
    #     shape=dataset.data.shape,
    # )
    print(f"Worker {worker_info.id} using memmap dataset")


def make_dataloader(dataloader_cfg: dict[str, Any]) -> DataLoader:
    """
    make dataloader out of configs

    Args:
        dataloader_cfg (dict[str, Any]): config dict
        split (str): "train", "val", or "test"
    """
    dataset_cfg: dict = copy.deepcopy(dataloader_cfg["dataset"])
    assert dataset_cfg["type"] in DATASET_CLASSES.keys()
    dataset = DATASET_CLASSES[dataset_cfg.pop("type")](**dataset_cfg)
    return DataLoader(
        dataset=dataset,
        batch_size=dataloader_cfg["batch_size"],
        shuffle=dataloader_cfg.get("shuffle", True),
        pin_memory=dataloader_cfg.get("pin_memory", True),
        drop_last=dataloader_cfg.get("drop_last", True),
        num_workers=dataloader_cfg.get("num_workers"),
        prefetch_factor=dataloader_cfg.get("prefetch_factor"),
        persistent_workers=dataloader_cfg.get("persistent_workers"),
        # worker_init_fn=worker_init_fn,
    )


def make_dataset(dataset_cfg: dict[str, Any]) -> Dataset:
    """
    make dataset out of configs

    Args:
        dataset_cfg (dict[str, Any]): config dict
    """
    assert dataset_cfg["type"] in DATASET_CLASSES.keys()
    return DATASET_CLASSES[dataset_cfg.pop("type")](**dataset_cfg)
