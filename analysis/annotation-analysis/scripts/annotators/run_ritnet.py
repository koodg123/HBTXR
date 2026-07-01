"""RITnet (DenseNet2D 4-class bg/sclera/iris/pupil) pupil center on EV-Eye anchors.
env: .venv-gsam2 (torch 2.5.1). Recipe: third/Pupil-Labs-Core-RITnet-Plugins/ritnet.
Preproc: gamma(0.8 LUT) + CLAHE(clip 1.5, 8x8) + Normalize([0.5],[0.5]), native res.
Forward: model(x)->[1,4,H,W]; argmax; class 3 = pupil. Uniform ellipse-fit in annlib.

  cd scripts/annotators && ../../.venv-gsam2/bin/python run_ritnet.py --device cpu
  weight via env RITNET_W (default best_model.pkl; alts ritnet_pupil.pkl).
"""
import sys, os
import numpy as np
import cv2
import torch

RIT = "/home/user/project/PRJXR-HBTXR/HBTXR/third/Pupil-Labs-Core-RITnet-Plugins"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RIT)
import annlib

WEI = os.environ.get("RITNET_W", f"{annlib.AA}/weights/RITnet/best_model.pkl")
_TABLE = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)   # gamma 0.8
PUPIL_CLASS = int(os.environ.get("RITNET_PUPIL_CLASS", "3"))


def load_fn(device, a=None):
    from ritnet.densenet import DenseNet2D
    model = DenseNet2D(out_channels=4)                                # bg/sclera/iris/pupil
    sd = torch.load(WEI, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    return model.to(device).eval()


def _prep(img):
    # DenseNet2D has 4 downsamples -> input dims must be /16. Native 346x260 is not,
    # causing skip-concat mismatch. Resize to 640x480 (4:3, /16, ~RITnet train scale).
    g = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    g = cv2.LUT(g, _TABLE)
    g = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(g).astype(np.float32) / 255.0
    g = (g - 0.5) / 0.5
    return torch.from_numpy(g).unsqueeze(0).unsqueeze(0).float()


def detect_fn(model, img, device):
    if img.max() < 20:
        return None, None
    t = _prep(img).to(device)
    with torch.no_grad():
        out = model(t)
    out = out[0] if isinstance(out, (list, tuple)) else out
    pred = out[0].argmax(0).cpu().numpy()
    mask = (pred == PUPIL_CLASS).astype(np.uint8)
    return (mask if mask.sum() >= 5 else None), None


if __name__ == "__main__":
    annlib.run_tool("ritnet", load_fn, detect_fn)
