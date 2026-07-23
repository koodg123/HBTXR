import os
from typing import Any, Callable, Optional, Tuple
import h5py
import numpy as np
import pdb
from tonic.dataset import Dataset
import tonic
import tonic.transforms as tonic_transforms
from dataset.custom_transforms import ScaleLabel, NormalizeLabel,TemporalSubsample
class ThreeETplus_Eyetracking(Dataset):
    """3ET DVS eye tracking `3ET <https://github.com/qinche106/cb-convlstm-eyetracking>`_
    ::

        @article{chen20233et,
            title={3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network},
            author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
            journal={arXiv preprint arXiv:2308.11771},
            year={2023}
        }

        authors: Qinyu Chen^{1,2}, Zuowen Wang^{1}
        affiliations: 1. Institute of Neuroinformatics, University of Zurich and ETH Zurich, Switzerland
                      2. Univeristy of Leiden, Netherlands

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        split (string, optional): The dataset split to use, ``train`` or ``val``.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.

    Returns:
         A dataset object that can be indexed or iterated over.
         One sample returns a tuple of (events, targets).
    """

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        split: str = "train",
        data_list_dir: str = './dataset',
        temp_subsample_factor: float = 1.,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        dataset: Optional[Any] = None,
        ind: Optional[int] = None,
        
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.transform_sub = tonic.transforms.Downsample(spatial_factor=1/3)
        self.label_transform_sub = tonic_transforms.Compose([ScaleLabel(1/3),
                                                       TemporalSubsample(temp_subsample_factor),
                                                       NormalizeLabel(pseudo_width=80, pseudo_height=60)
                                                      ])
        data_dir = save_to
        # Load filenames from the provided lists
        if dataset == "t":
            if split == "train":
                filenames = self.load_filenames(os.path.join(data_list_dir, "train_files.txt"))
            elif split == "val":
                filenames = self.load_filenames(os.path.join(data_list_dir, "val_files.txt"))
            elif split == "test":
                filenames = self.load_filenames(os.path.join(data_list_dir, "test_files.txt"))
            else:
                raise ValueError("Invalid split name")
        else:
            if split == "train":
                filenames = self.load_filenames(os.path.join(data_list_dir, f"{dataset}.txt"))
            elif split == "val":
                filenames = self.load_filenames(os.path.join(data_list_dir, "val_files.txt"))
            elif split == "test":
                filenames = self.load_filenames(os.path.join(data_list_dir, "test_files.txt"))
            else:
                raise ValueError("Invalid split name")            
        # Get the data file paths and target file paths
        if split != "test":
            self.data = [os.path.join(data_dir, "train", f, f + ".h5") for f in filenames]
            self.targets = [os.path.join(data_dir, "train", f, "label.txt") for f in filenames]
        else:
            self.data = [os.path.join(data_dir, "test", f, f + ".h5") for f in filenames]
            self.targets = [os.path.join(data_dir, "test", f, "label_zeros.txt") for f in filenames]
        self.split = split
        self.ind = ind
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # index = index + 1
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        # if index == 1022:
        #     pdb.set_trace()
        # get events from .h5 file
        
        with h5py.File(self.data[index], "r") as f:
            # original events.dtype is dtype([('t', '<u8'), ('x', '<u8'), ('y', '<u8'), ('p', '<u8')])
            # t is in us
            events = f["events"][:]
            # events['p'] = events['p']*2 -1  # convert polarity to -1 and 1
        # load the sparse labels
        with open(self.targets[index], "r") as f:
            # target is at the frequency of 100 Hz. It will be downsampled to 20 Hz in the target transformation
            target = np.array(
                [list(map(float, line.strip('()\n').split(', '))) for line in f.readlines()], np.float32)
        original_shape = target.shape[0]

        if self.transform is not None:
            if 'sub' in self.data[index]:
                events = self.transform_sub(events)
            else:
                events = self.transform(events)
        # if 'sub' in self.data[index]:
        #     pdb.set_trace()
        if self.target_transform is not None:
            if 'sub' in self.data[index]:
                target = self.label_transform_sub(target)
            else:
                target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        if self.split == "test":
            target[-1][1] = original_shape
            return events, target
        else:
            return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present()

    def load_filenames(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]