from typing import Any, Type
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class DvEyeDataset(Dataset, ABC):
    # Here split indicates the dataset split, such as train, val, or test; **kargs contains additional parameters
    def __init__(self, split: str, **kargs) -> None:
        super().__init__()
        self._split = split
        self._init_dataset(**kargs)

    @abstractmethod
    def _init_dataset(self, **kargs):
        pass
