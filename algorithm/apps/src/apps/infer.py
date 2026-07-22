import click
import numpy as np
import pandas as pd
import torch

from dataset.dataset_factory import make_dataset
from dataset.DavisEyeCenter.losses import process_detector_prediction
from engine.model_factory import make_model
from engine.tools.load_config import load_config


@click.command()
@click.option("--config", "-c", type=str, required=True, help="inference config path")
@click.option("--output", "-o", type=str, default="submission.csv", help="prediction csv output path")
def main(config: str, output: str) -> None:
    """Run streaming inference for one sample and write a predictions csv."""
    cfg = load_config(config)
    test_dataset = make_dataset(cfg["dataset"])
    model = make_model(cfg["model"])
    model.load_state_dict(torch.load(cfg["test"]["ckpt_path"])["state_dict"])
    device = cfg["test"]["map_location"]
    model.to(device)

    event_frames = test_dataset[0].unsqueeze(0).to(device)

    pred = model.streaming_inference(model, event_frames)
    pred = process_detector_prediction(pred)
    pred = pred.squeeze(0)
    pred[0] *= 346
    pred[1] *= 260
    predictions_numpy = pred.detach().cpu().numpy().T.astype(np.int32)

    arange = np.arange(predictions_numpy.shape[0])
    predictions_numpy = np.concatenate([arange[:, None], predictions_numpy], axis=1)

    df = pd.DataFrame(predictions_numpy, columns=["row_id", "x", "y"])
    df.to_csv(output, index=False)

    print("Inference completed.")


if __name__ == "__main__":
    main()
