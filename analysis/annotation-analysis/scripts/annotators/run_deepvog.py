"""DeepVOG (Keras/TF U-Net) pupil center on EV-Eye anchors.
env: SEPARATE TF env (.venv-deepvog) — NOT .venv-gsam2. TF1/Keras2.2.4 orig; we run
on TF2/tf-keras via a shim (see setup_deepvog_env.sh). GPLv3 (license caution).
Preproc: resize 320x240 (W,H) + /255 + replicate 3ch. Output softmax; channel 1 =
pupil prob; threshold 0.5. Uniform ellipse-fit in annlib.

  cd scripts/annotators && ../../.venv-deepvog/bin/python run_deepvog.py --device cpu
"""
import sys, os
import numpy as np
import cv2

DV = "/home/user/project/PRJXR-HBTXR/HBTXR/third/DeepVOG"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DV)
import annlib


WEI = os.environ.get("DEEPVOG_W", f"{annlib.AA}/weights/DeepVOG/DeepVOG_weights.h5")


def load_fn(device, a=None):
    # Use the vendored+patched builder (_deepvog_model) so we neither edit third/ nor
    # import the DeepVOG package __init__ (which pulls skvideo). Load our explicit .h5.
    from _deepvog_model import DeepVOG_net
    model = DeepVOG_net(input_shape=(240, 320, 3), filter_size=(10, 10))
    model.load_weights(WEI)
    return model


def detect_fn(model, img, device):
    r = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.repeat(r[None, :, :, None], 3, axis=3)   # (1,240,320,3)
    Y = model.predict(x, verbose=0)
    hm = Y[0, :, :, 1]                              # pupil probability (240,320)
    mask = (hm > 0.5).astype(np.uint8)
    return (mask if mask.sum() >= 5 else None), float(hm.max())


if __name__ == "__main__":
    annlib.run_tool("deepvog", load_fn, detect_fn)
