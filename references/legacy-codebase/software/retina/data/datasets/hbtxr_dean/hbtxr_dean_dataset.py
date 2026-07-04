import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


class HBTXRDeanDataset(Dataset):
    """Retina dataset wrapper over FACET's HBTXR subject-independent cache."""

    def __init__(self, split, training_params, dataset_params):
        self.split = split
        self.training_params = training_params
        self.dataset_params = dataset_params
        self.image_size = int(dataset_params["img_width"]), int(dataset_params["img_height"])
        self.bbox_w = float(training_params["bbox_w"])

        facet_root = Path(dataset_params["facet_root"]).expanduser().resolve()
        sys.path.insert(0, str(facet_root))
        from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import (  # noqa: PLC0415
            DavisEyeEllipseDataset,
        )

        self.dataset = DavisEyeEllipseDataset(
            root_path=dataset_params["root_path"],
            split=split,
            accumulate_mode="fixed_count",
            sensor_size=(346, 260, 2),
            events_interpolation=dataset_params.get(
                "events_interpolation", "causal_linear_ori"
            ),
            pupil_area=dataset_params.get("pupil_area", 200),
            num_classes=1,
            default_resolution=[self.image_size[1], self.image_size[0]],
        )

    def __len__(self):
        return len(self.dataset)

    def _bbox_from_center(self, center_xy):
        width, height = self.image_size
        x = torch.clamp(center_xy[0] / width, 0.0, 1.0)
        y = torch.clamp(center_xy[1] / height, 0.0, 1.0)
        dx = self.bbox_w / width
        dy = self.bbox_w / height
        return torch.stack(
            [
                torch.clamp(x - dx, 0.0, 1.0),
                torch.clamp(y - dy, 0.0, 1.0),
                torch.clamp(x + dx, 0.0, 1.0),
                torch.clamp(y + dy, 0.0, 1.0),
            ]
        ).float()

    def __getitem__(self, index):
        sample = self.dataset[index]
        events = torch.as_tensor(sample["input"], dtype=torch.float32)
        events = events.unsqueeze(0)

        center64 = torch.as_tensor(sample["center"], dtype=torch.float32) * 4.0
        if int(sample["close"]) != 0:
            center64 = torch.zeros_like(center64)

        label = self._bbox_from_center(center64)
        avg_dt = torch.tensor(0.0, dtype=torch.float32)
        exp_id = torch.tensor(index, dtype=torch.int64)
        return events, label, avg_dt, exp_id
