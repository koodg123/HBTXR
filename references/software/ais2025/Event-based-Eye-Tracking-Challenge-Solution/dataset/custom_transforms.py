import numpy as np
import torch
from tonic.slicers import (
    slice_events_by_time,
)
import tonic.functional as tof
from typing import Any, List, Tuple
import random
import pdb
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
class SliceByTimeEventsTargets:
    """
    Modified from tonic.slicers.SliceByTimeEventsTargets in the Tonic Library

    Slices an event array along fixed time window and overlap size. The number of bins depends
    on the length of the recording. Targets are copied.

    >        <overlap>
    >|    window1     |
    >        |   window2     |

    Parameters:
        time_window (int): time for window length (same unit as event timestamps)
        overlap (int): overlap (same unit as event timestamps)
        include_incomplete (bool): include the last incomplete slice that has shorter time
    """

    def __init__(self,time_window, overlap=0.0, seq_length=30, seq_stride=15, include_incomplete=False) -> None:
        self.time_window= time_window
        self.overlap= overlap
        self.seq_length=seq_length
        self.seq_stride=seq_stride
        self.include_incomplete=include_incomplete

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        t = data["t"]
        stride = self.time_window - self.overlap
        assert stride > 0

        if self.include_incomplete:
            n_slices = int(np.ceil(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        else:
            n_slices = int(np.floor(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        n_slices = max(n_slices, 1)  # for strides larger than recording time

        window_start_times = np.arange(n_slices) * stride + t[0]
        if self.include_incomplete and n_slices > 1:
            window_start_times[-1] = t[-1] - self.time_window
        window_end_times = window_start_times + self.time_window

        indices_start = np.searchsorted(t, window_start_times)[:n_slices]
        indices_end = np.searchsorted(t, window_end_times)[:n_slices]
        
        if not self.include_incomplete:
            # get the strided indices for loading labels
            label_indices_start = np.arange(0, targets.shape[0]-self.seq_length, self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
        else:
            label_indices_start = np.arange(0, targets.shape[0], self.seq_stride)
            label_indices_start = label_indices_start[:len(window_start_times)]
            
            if len(label_indices_start) != 1:
                label_indices_start[-1] = targets.shape[0] - self.seq_length

            for i in range(1, len(label_indices_start)):
                if label_indices_start[i] + self.seq_length > targets.shape[0]:
                    indices_start = indices_start[:i+1]
                    indices_end = indices_end[:i+1]
                    label_indices_start = label_indices_start[:i]
                    label_indices_start = np.append(label_indices_start, targets.shape[0] - self.seq_length)
                    n_slices = i
                    break
            
            # if label_indices_start[-2] + self.seq_length > targets.shape[0]:
            #     # 数据集有问题，事件记录时间长，但是target少给了一个
            #     # window_end_times = np.delete(window_end_times, -2)
            #     # window_start_times = np.delete(window_start_times, -2)
            #     pdb.set_trace()
            #     indices_start = np.delete(indices_start, -2)
            #     indices_end = np.delete(indices_end, -2)
            #     label_indices_start = np.delete(label_indices_start, -2)
            #     n_slices -= 1

            label_indices_end = label_indices_start + self.seq_length
            # the last label indices end should be the last label
            label_indices_end[-1] = targets.shape[0]
            label_indices_start[-1] = label_indices_end[-1] - self.seq_length

        assert targets.shape[0] >= label_indices_end[-1]
        try:
            for i in range(n_slices):
                assert indices_start[i] < indices_end[i]-1
                assert label_indices_start[i] < label_indices_end[i]-1
                assert len(indices_start) == len(label_indices_start)
            for i in range(0, len(label_indices_start)):
                assert label_indices_end[i] - label_indices_start[i] == self.seq_length

        except:
            pdb.set_trace()
        
        return list(zip(zip(indices_start, indices_end), zip(label_indices_start, label_indices_end)))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ):
        return_data = []
        return_target = []
        targets = np.pad(targets, ((0, 0), (0, 1)), mode='constant')
        targets[:, -1] = range(targets.shape[0])

        for tuple1, tuple2 in metadata:
            return_data.append(data[tuple1[0]:tuple1[1]])
            return_target.append(targets[tuple2[0]:tuple2[1]])
            # if return_target[-1].shape[0] != 45:
            #     print(return_target[-1].shape)
            #     pdb.set_trace()
        # print(f"metadata: {metadata}")

        return return_data, return_target

class SliceLongEventsToShort:
    def __init__(self, time_window, overlap, include_incomplete):
        """
        Initialize the transformation.

        Args:
        - time_window (int): The length of each sub-sequence.
        """
        self.time_window = time_window
        self.overlap = overlap
        self.include_incomplete = include_incomplete

    def __call__(self, events):
        events = slice_events_by_time(events, self.time_window, self.overlap, self.include_incomplete)
        # pdb.set_trace()
        return events


class Jitter:
    def __init__(self):
        """
        Initialize the transformation.

        Args:
        - time_window (int): The length of each sub-sequence.
        """
        # with h5py.File(f"/gdata1/hanh/event_data/cached_dataset/val_vl_30_vs30_ch4_tbinary/23_0.hdf5", "r") as f:
        #     # original events.dtype is dtype([('t', '<u8'), ('x', '<u8'), ('y', '<u8'), ('p', '<u8')])
        #     # t is in us
        #     self.e_23 = f["data"]['0'][:]
        #     self.t_23 = f['target']['0'][:]

    def __call__(self, data, label):
        if label.shape[0] != 45:
            print(label.shape)
        # x shift
        prob = 0.5
        mix_flag = None
        T, C, H, W = data.shape
        try:
            assert label.shape[0] == T
        except:
            pdb.set_trace()
        # x shift
        if random.random() > prob:
            x = max(1, int(random.random() * 10))
            if random.random() > 0.5:
                data = np.concatenate([data[..., x:], data[..., :x]], axis=-1)
                label[..., 0] = label[..., 0] - x / data.shape[-1]
                label[label[:, 0] < 0, 0] = label[label[:, 0] < 0, 0] + 1
            else:
                data = np.concatenate([data[..., -x:], data[..., :-x]], axis=-1)
                label[..., 0] = label[..., 0] + x / data.shape[-1]
                label[label[:, 0] > 1, 0] = label[label[:, 0] > 1, 0] - 1
                
        # y shift        
        if random.random() > prob:
            y = max(1, int(random.random() * 10))
            if random.random() > 0.5:
                data = np.concatenate([data[..., y:, :], data[..., :y, :]], axis=-2)
                label[..., 1] = label[..., 1] - y / data.shape[-2]
                label[label[:, 1] < 0, 1] = label[label[:, 1] < 0, 1] + 1
            else:
                data = np.concatenate([data[..., -y:, :], data[..., :-y, :]], axis=-2)
                label[..., 1] = label[..., 1] + y / data.shape[-2]
                label[label[:, 1] > 1, 1] = label[label[:, 1] > 1, 1] - 1
        # x flip
        if random.random() > prob:
            data = np.flip(data, axis=-1)
            label[..., 0] = 1 - label[..., 0]

        # y flip
        if random.random() > prob:
            data = np.flip(data, axis=-2)
            label[..., 1] = 1 - label[..., 1]

        # t shift
        if random.random() > prob:
            t = max(1, int(random.random() * 3))
            if random.random() > 0.5:
                data = np.concatenate([data[t:], data[:t]], axis=0)
                label = np.concatenate([label[t:], label[:t]], axis=0)
            else:
                data = np.concatenate([data[-t:], data[:-t]], axis=0)
                label = np.concatenate([label[-t:], label[:-t]], axis=0)
        # vis_event(data, label)
        # pdb.set_trace()
        # random cutout 
        if random.random() > prob:
            h, w = (np.random.randint(6, high=12), np.random.randint(8, high=16))
            top = np.random.randint(4, H - h + 1)
            left = np.random.randint(5, W - w + 1)
            data[:, :, top:top+h, left:left+w] = 0
        # vis_event(data, label)
        # pdb.set_trace()
        # random noise for binary representation
        # if random.random() > 0.8:
        #     total_elements = T * C * W * H
        #     ratio_of_ones = np.random.uniform(0.04, 0.04)
        #     num_ones = int(total_elements * ratio_of_ones)
        #     indices = np.random.choice(total_elements, num_ones, replace=False)
        #     data = data.reshape(-1)
        #     data[indices] = np.random.choice(np.unique(data).tolist(), size=num_ones)
        #     data = data.reshape((T, C, H, W))

        ## random noise for frame representation
        # if random.random() > prob:
        #     mean = 0
        #     sigma = np.random.randint(5, 25)
        #     gauss = np.random.normal(mean,sigma,(T, C, H, W))
        #     data = data + gauss
        # vis_event(data)
        # pdb.set_trace()
        if mix_flag:
            
            gt = (label_[:, :2] * (W, H)).astype(int)
            orig_top = np.maximum(np.zeros(T), gt[:, 1]-neighberhood).astype(int)
            orig_left = np.maximum(np.zeros(T), gt[:, 0]-neighberhood).astype(int)
            orig_bottom = np.minimum(np.ones(T) * H, gt[:, 1]+neighberhood).astype(int)
            orig_right = np.minimum(np.ones(T) * W, gt[:, 0]+neighberhood).astype(int)

            gt = (label[:, :2] * (W, H)).astype(int)
            e23_top = np.maximum(np.zeros(T), gt[:, 1]-neighberhood).astype(int)
            e23_left = np.maximum(np.zeros(T), gt[:, 0]-neighberhood).astype(int)
            e23_bottom = np.minimum(np.ones(T) * H, gt[:, 1]+neighberhood).astype(int)
            e23_right = np.minimum(np.ones(T) * W, gt[:, 0]+neighberhood).astype(int)

            bias = ((orig_bottom - orig_top) - (e23_bottom - e23_top))
            indices1 = np.where(bias > 0)
            orig_bottom[indices1] = orig_bottom[indices1] - np.floor(bias[indices1]/2)
            orig_top[indices1] = orig_top[indices1] + np.ceil(bias[indices1]/2)
            indices2 = np.where(bias < 0)
            e23_bottom[indices2] = e23_bottom[indices2] - np.floor(np.abs(bias[indices2]/2))
            e23_top[indices2] = e23_top[indices2] + np.ceil(np.abs(bias[indices2]/2))

            bias = ((orig_right - orig_left) - (e23_right - e23_left))
            indices1 = np.where(bias > 0)
            orig_right[indices1] = orig_right[indices1] - np.floor(bias[indices1]/2)
            orig_left[indices1] = orig_left[indices1] + np.ceil(bias[indices1]/2)
            indices2 = np.where(bias < 0)
            e23_right[indices2] = e23_right[indices2] - np.floor(np.abs(bias[indices2]/2))
            e23_left[indices2] = e23_left[indices2] + np.ceil(np.abs(bias[indices2]/2))
            
            for t in range(T):
                data[t, :, e23_top[t]:e23_bottom[t], e23_left[t]:e23_right[t]] = data_[t, :, orig_top[t]:orig_bottom[t], orig_left[t]:orig_right[t]]
            # vis_event(data, label)
            # label = label_
        
        # 
        return data.copy(), label

class EventSlicesToMap:
    def __init__(self, sensor_size, n_time_bins, per_channel_normalize, map_type='voxel'):
        """
        Initialize the transformation.

        Args:
        - sensor_size (tuple): The size of the sensor.
        - n_time_bins (int): The number of time bins.
        """
        self.sensor_size = sensor_size
        self.n_time_bins = n_time_bins
        self.per_channel_normalize = per_channel_normalize
        self.map_type = map_type

    def __call__(self, event_slices):
        """
        Apply the transformation to the given event slices.

        Args:
        - event_slices (Tensor): The input event slices.

        Returns:
        - Tensor: A batched tensor of voxel grids.
        """
        ev_maps = []
        for ijk, event_slice in enumerate(event_slices):
            if self.map_type == 'voxel':
                ev_map = tof.to_voxel_grid_numpy(event_slice, self.sensor_size, self.n_time_bins)
            elif self.map_type == 'binary':
                # assert event_slice['t'].shape[0] > 0  
                # pdb.set_trace()
                try:
                    ev_map = tof.to_frame_numpy(event_slice, self.sensor_size, n_time_bins=self.n_time_bins)
                    ev_map = tof.to_bina_rep_numpy(ev_map, n_frames=1, n_bits=self.n_time_bins)
                except:
                    if ijk != 0:
                        ev_maps.append(ev_maps[-1])
                        continue
                    else:
                        ev_map = np.zeros((2, self.sensor_size[1], self.sensor_size[0]), dtype=np.float32)
                        print("event slice is empty")
            
            ev_map = ev_map.reshape(-1, ev_map.shape[-2], ev_map.shape[-1])
            if self.per_channel_normalize:
                # Calculate mean and standard deviation only at non-zero values
                non_zero_entries = (ev_map != 0)
                for c in range(ev_map.shape[0]):
                    mean_c = ev_map[c][non_zero_entries[c]].mean()
                    std_c = ev_map[c][non_zero_entries[c]].std()

                    ev_map[c][non_zero_entries[c]] = (ev_map[c][non_zero_entries[c]] - mean_c) / (std_c + 1e-10)
            ev_maps.append(ev_map)
        return np.array(ev_maps).astype(np.float32)


class SplitSequence:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride

    def __call__(self, sequence, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - sequence (Tensor): The input sequence of frames.
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of sub-sequences.
        - Tensor: A batched tensor of corresponding labels.
        """

        sub_sequences = []
        sub_labels = []

        for i in range(0, len(sequence) - self.sub_seq_length + 1, self.stride):
            sub_seq = sequence[i:i + self.sub_seq_length]
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_sequences.append(sub_seq)
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_sequences), np.stack(sub_labels)
    

class SplitLabels:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride
        # print(f"stride is {self.stride}")

    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        sub_labels = []
        
        for i in range(0, len(labels) - self.sub_seq_length + 1, self.stride):
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_labels)

class ScaleLabel:
    def __init__(self, scaling_factor):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.scaling_factor = scaling_factor


    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:,:2] =  labels[:,:2] * self.scaling_factor
        return labels
    
class TemporalSubsample:
    def __init__(self, temporal_subsample_factor):
        self.temp_subsample_factor = temporal_subsample_factor

    def __call__(self, labels):
        """
        temorally subsample the labels
        """
        interval = int(1/self.temp_subsample_factor)
        return labels[::interval]
    

class NormalizeLabel:
    def __init__(self, pseudo_width, pseudo_height):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.pseudo_width = pseudo_width
        self.pseudo_height = pseudo_height
    
    def __call__(self, labels):
        """
        Apply normalization on label, with pseudo width and height

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:, 0] = labels[:, 0] / self.pseudo_width
        labels[:, 1] = labels[:, 1] / self.pseudo_height
        return labels

