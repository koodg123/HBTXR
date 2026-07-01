"""EllSeg (DenseElNet / ritnet_v3) pupil center on EV-Eye anchors.
env: .venv-gsam2 (torch 2.5.1). Recipe: third/EllSeg/evaluate_ellseg.py.
Preproc: width->320 (LANCZOS4) + vertical pad/crop to 240 + per-frame z-score.
Forward: enc->dec, seg argmax; class 2 = pupil. Uniform ellipse-fit in annlib.

  cd scripts/annotators && ../../.venv-gsam2/bin/python run_ellseg.py --device cpu
  (smoke) ... run_ellseg.py --keys <key1>,<key2> --device cpu --save-masks
"""
import sys, os
import numpy as np
import cv2
import torch

ELL = "/home/user/project/PRJXR-HBTXR/HBTXR/third/EllSeg"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ELL)
import annlib

WEI = os.environ.get("ELLSEG_W", f"{annlib.AA}/weights/EllSeg/all.git_ok")


def load_fn(device, a=None):
    os.chdir(ELL)                                  # RITnet_v3 import may use rel paths
    from models.RITnet_v3 import DenseNet2D
    model = DenseNet2D()                           # defaults == model_dict['ritnet_v3']
    ck = torch.load(WEI, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _prep(img):
    H0, W0 = img.shape
    sc = 320.0 / W0
    newH = int(round(H0 * sc))
    r = cv2.resize(img, (320, newH), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    pad = 240 - newH
    if pad > 0:
        t = pad // 2
        r = np.pad(r, ((t, pad - t), (0, 0)))
    elif pad < 0:
        c = -pad
        t = c // 2
        r = r[t:t + 240, :]
    s = r.std()
    r = (r - r.mean()) / (s if s > 1e-6 else 1.0)
    return torch.from_numpy(r).unsqueeze(0).unsqueeze(0).float()


def detect_fn(model, img, device):
    if img.max() < 20:                             # dark/blink guard (evaluate_ellseg.py)
        return None, None
    t = _prep(img).to(device)
    with torch.no_grad():
        x4, x3, x2, x1, x = model.enc(t)
        seg = model.dec(x4, x3, x2, x1, x)         # [1,3,240,320]
    pred = seg[0].argmax(0).cpu().numpy()          # 0=bg 1=iris 2=pupil
    mask = (pred == 2).astype(np.uint8)
    return (mask if mask.sum() >= 5 else None), None


if __name__ == "__main__":
    annlib.run_tool("ellseg", load_fn, detect_fn)
