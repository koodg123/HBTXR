"""Edge-Guided near-eye (BDCN edge + DenseNet2D RITnet_v2) pupil center on EV-Eye.
env: .venv-gsam2 (torch 2.5.1). Recipe: third/Edge-Guided-.../evaluate.py.
Two models: BDCN edge (gen_00000016.pt, key 'a') + seg (baseline_edge_16.pkl,
key 'state_dict'). Preproc: width->320 + pad/crop 240 + z-score. Edge branch gets
3x-replicated gray. Seg forward takes many dummy loss args; class 2 = pupil.
NOTE: cross-dataset (trained on TEyeD/OpenEDS/RITEyes, not EV-Eye).

  cd scripts/annotators && ../../.venv-gsam2/bin/python run_edge_guided.py --device cpu
"""
import sys, os
import numpy as np
import cv2
import torch
import yaml

EG = "/home/user/project/PRJXR-HBTXR/HBTXR/third/Edge-Guided-Near-Eye-Image-Analysis-for-Head-Mounted-Displays"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EG)
import annlib

EDGE_W = os.environ.get("EG_EDGE_W", f"{annlib.AA}/weights/Edge-Guided/gen_00000016.pt")
SEG_W = os.environ.get("EG_SEG_W", f"{annlib.AA}/weights/Edge-Guided/baseline_edge_16.pkl")


def load_fn(device, a=None):
    os.chdir(EG)
    from models.RITnet_v2 import DenseNet2D
    from bdcn_new import BDCN
    cfgp = os.path.join(EG, "configs", "baseline_edge.yaml")
    setting = yaml.load(open(cfgp), Loader=yaml.FullLoader)
    seg = DenseNet2D(setting)
    sc = torch.load(SEG_W, map_location="cpu", weights_only=False)
    seg.load_state_dict(sc["state_dict"] if isinstance(sc, dict) and "state_dict" in sc else sc)
    edge = BDCN()
    ec = torch.load(EDGE_W, map_location="cpu", weights_only=False)
    edge.load_state_dict(ec["a"] if isinstance(ec, dict) and "a" in ec else ec)
    return (seg.to(device).eval(), edge.to(device).eval())


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


def detect_fn(models, img, device):
    seg, edge = models
    if img.max() < 20:
        return None, None
    t = _prep(img).to(device)
    with torch.no_grad():
        # Replicate RITnet_v2 forward lines 281-288 (image enc + edge enc bottleneck
        # concat + dec) and STOP at seg logits, skipping the elReg/center-of-mass/loss
        # tail (309-354) that needs GT dummies and breaks on numpy-2.
        e = edge(torch.cat([t, t, t], 1))[-1]                # BDCN fused edge [1,1,240,320]
        x4, x3, x2, x1, x = seg.enc(t)                       # image encoder
        _, _, _, _, x_add = seg.enc(e)                       # edge encoder (bottleneck)
        x = torch.cat((x, x_add), 1)                         # add_edge concat
        op = seg.dec(x4, x3, x2, x1, x)                      # seg logits [1,3,240,320]
    pred = op[0].argmax(0).cpu().numpy()                     # 0=bg 1=iris 2=pupil
    mask = (pred == 2).astype(np.uint8)
    return (mask if mask.sum() >= 5 else None), None


if __name__ == "__main__":
    annlib.run_tool("edge_guided", load_fn, detect_fn)
