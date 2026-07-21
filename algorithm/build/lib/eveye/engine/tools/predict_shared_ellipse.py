"""Executable prediction demo for the shared ellipse model.

AM-040 extraction: moved out of the shared model owner so that
eveye.common.models.DavisEyeEllipse.HBTXR.Predict keeps no execution payload.
`os` is imported explicitly here because the source module received it
implicitly through a wildcard import of eveye.utils.visualization.
"""

import os


def main():
    from eveye.common.models.DavisEyeEllipse.HBTXR.HBTXR import HBTXR
    from eveye.event.models.ElNet.ElNet import Creat_MyNet
    from eveye.common.models.DavisEyeEllipse.HBTXR.Predict import predict_txt

    index = 0
    model = HBTXR(
        input_channels=2,
        head_dict={"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1},
        mode="fpn_2d",
    )
    txt_path = "/mnt/data2T/junyuan/Datasets/datasets/DavisEyeCenterDataset/data/user1_left_session_1_0_1_events.txt"
    output_path = "/mnt/data2T/junyuan/eye-tracking/video/output/EventSample"
    model_path = "/mnt/data2T/junyuan/eye-tracking/logs/EPNet_FixedCount5000_TrigERAugFPN2d_AllAug/version_0/checkpoints/epoch=67-val_mean_distance=0.2403.ckpt"
    os.makedirs(output_path, exist_ok=True)
    predict_txt(model, model_path, txt_path, output_path)
    # model = Creat_MyNet(
    #     base_name="dla34",
    #     heads={"hm": 1, "ab": 2, "ang": 1, "trig": 2, "reg": 2, "mask": 1},
    #     pretrained=True,
    #     down_ratio=4,
    #     final_kernel=1,
    #     last_level=5,
    #     head_conv=256,
    #     out_channel=0,
    # )
    # data_path = Path(
    #     "/mnt/data2T/junyuan/Datasets/FixedCount5000Dataset/val/cached_data"
    # )
    # ellipse_path = Path(
    #     "/mnt/data2T/junyuan/Datasets/FixedCount5000Dataset/val/cached_ellipse"
    # )
    # model_path = Path(
    #     "/mnt/data2T/junyuan/eye-tracking/weights/Others/EPNet_FixedCount5000_TrigERAugFPNdw_without_DE&DEA/version_0/checkpoints/epoch=68-val_mean_distance=0.2031.ckpt"
    # )
    # model_path = Path(
    #     "/mnt/data2T/junyuan/eye-tracking/logs/ElNet_FixedCount10000/version_0/checkpoints/epoch=69-val_mean_distance=0.3242.ckpt"
    # )
    # output_path = Path(
    #     "/mnt/data2T/junyuan/eye-tracking/predictions/EPNet_FixedCount5000_TrigERAugFPNdw_without_DE&DEA"
    # )
    # os.makedirs(output_path, exist_ok=True)

    # device = "cuda:0"

    # predict(model, model_path, data_path, ellipse_path, output_path, 500, device)

    # test_inference_time(model, model_path, device)

    # input = torch.rand((1, 2, 256, 256), dtype=torch.float32)
    # flops, params = profile(model, inputs=(input,))
    # print(f"FLOPs: {flops}")
    # print(f"Total parameters: {params}")


if __name__ == "__main__":
    main()
