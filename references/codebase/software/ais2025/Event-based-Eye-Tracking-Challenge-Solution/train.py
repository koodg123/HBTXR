import argparse, json, yaml, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from dataset.ThreeET_plus import ThreeETplus_Eyetracking
from dataset.custom_transforms import ScaleLabel, NormalizeLabel,TemporalSubsample, \
    SliceLongEventsToShort, EventSlicesToMap, SliceByTimeEventsTargets, Jitter
    

import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
import pdb
import importlib
import logging
import time
import shutil
import numpy as np

def train(model, train_loader, val_loader, criterion, optimizer, args):
    best_val_loss = float("inf")
    best_val_metric = float("inf")
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.num_epochs
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.5, -1)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6, last_epoch=-1, verbose=False)
    # lr_scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.3, verbose=False)
    # Training loop
    for epoch in range(args.num_epochs):

        model, train_loss, metrics = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args)
        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, args)
            if val_metrics['val_p_error_all']['val_p_error_all'] < best_val_metric:
                best_val_metric = val_metrics['val_p_error_all']['val_p_error_all']
                # save the new best model to MLflow artifact with 3 decimal places of validation loss in the file name

                best_model_path = os.path.join(args.log_dir, f"perror_{val_metrics['val_p_error_all']['val_p_error_all']:.4f}_tsf{args.temporal_subsample_factor}.pth")
                torch.save(model.state_dict(), best_model_path)
                # top_k_checkpoints(args)
                
            print(f"[Validation] at Epoch {epoch+1}/{args.num_epochs}: Val Loss: {val_loss:.4f}, p_error_all: {val_metrics['val_p_error_all']['val_p_error_all']}")
        print(f"Epoch {epoch+1}/{args.num_epochs}: Train Loss: {train_loss:.4f}")
    return model, best_model_path


def main(args):
    num_workers = 24
    with open(os.path.join('./configs', args.config_file), 'r') as f:
        # print(f"Loading config from {args.config_file}")
        config = json.load(f)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    log_dir = os.path.join(config['log_dir'], config['run_name'])
    config['log_dir'] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    args = argparse.Namespace(**config)

    # Define your model, optimizer, and criterion
    model = importlib.import_module(f"model.{args.model}").Model(args).to("cuda")
    model = nn.DataParallel(model)
    # if args.spatial_factor > 0.125:
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "weighted_mse":
        criterion = weighted_MSELoss(weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to("cuda"), \
                                        reduction='mean')
    else:
        raise ValueError("Invalid loss name")

    factor = args.spatial_factor # spatial downsample factor
    temp_subsample_factor = args.temporal_subsample_factor # downsampling original 100Hz label to 20Hz

    # First we define the label transformations
    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    # Then we define the raw event recording and label dataset, the raw events spatial coordinates are also downsampled
    train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", \
                    transform=transforms.Downsample(spatial_factor=factor), 
                    target_transform=label_transform, dataset=args.dataset,
                    temp_subsample_factor=temp_subsample_factor)
    val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform, dataset=args.dataset,
                    temp_subsample_factor=temp_subsample_factor)

    slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds
    train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds
    valid_stride_time = int(10000/temp_subsample_factor*args.val_stride) #microseconds
    train_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                    seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=True)
    val_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-valid_stride_time, \
                    seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=True)

    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToMap(sensor_size=(int(640*factor), int(480*factor), 2), \
                                n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization,
                                map_type=args.map_type)
    ])
    train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_train_tl_{args.train_length}_ts{args.train_stride}_{args.dataset}_dt{temp_subsample_factor}_sf{args.spatial_factor}")
    val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_val_vl_{args.val_length}_vs{args.val_stride}_{args.dataset}_dt{temp_subsample_factor}_sf{args.spatial_factor}")
    train_data = DiskCachedDataset(train_data, 
                                cache_path=f"{args.cache_dir}/train_tl_{args.train_length}_ts{args.train_stride}_{args.dataset}_dt{temp_subsample_factor}_sf{args.spatial_factor}",
                                transforms=Jitter())
    val_data = DiskCachedDataset(val_data, cache_path=f"{args.cache_dir}/val_vl_{args.val_length}_vs{args.val_stride}_{args.dataset}_dt{temp_subsample_factor}_sf{args.spatial_factor}",
                                transforms=None)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, \
                                num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, \
                            num_workers=num_workers)

    # Train your model
    model, best_model_path = train(model, train_loader, val_loader, criterion, optimizer, args)

    # Save your model for the last epoch
    torch.save(model.state_dict(), os.path.join(args.log_dir, f"epoch{args.num_epochs}_{args.temporal_subsample_factor}.pth"))

    from test import test
    model.eval()
    args.checkpoint = best_model_path
    args.test_length = args.val_length
    args.test_stride = args.val_stride
    test(args, model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # training management arguments     
    
    # a config file 
    parser.add_argument("--config_file", 
                        default="sliced_baseline.json", 
                        help="path to JSON configuration file")
    parser.add_argument("--run_name", type=str, help="name of the run")
    # training hyperparameters
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--num_epochs", type=int, help="number of epochs")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_length", type=int)
    parser.add_argument("--val_length", type=int)
    parser.add_argument("--test_length", type=int)
    parser.add_argument("--train_stride", type=int)
    parser.add_argument("--val_stride", type=int)
    parser.add_argument("--test_stride", type=int)
    parser.add_argument("--n_time_bins", type=int)
    parser.add_argument("--map_type", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--spatial_factor", type=float)
    parser.add_argument("--temporal_subsample_factor", type=float)
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.device])
    # def force_cudnn_initialization():
    #     s = 32
    #     dev = torch.device('cuda')
    #     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    # force_cudnn_initialization()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    main(args)

