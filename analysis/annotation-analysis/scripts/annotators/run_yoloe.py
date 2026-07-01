"""YOLOE (open-vocab YOLO, as used by X-AnyLabeling) pupil center on EV-Eye anchors.
env: .venv-gsam2 + `pip install ultralytics` (torch 2.5.1). We call ultralytics.YOLOE
DIRECTLY (headless), not the Qt GUI. Text prompt -> boxes/masks -> pupil center.
Weights auto-download on first use (yoloe-11s-seg.pt + mobileclip_blt); needs network.

  cd scripts/annotators && ../../.venv-gsam2/bin/python run_yoloe.py --device cpu \
      --prompt "black pupil" --conf 0.05
"""
import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annlib

MODEL_NAME = os.environ.get("YOLOE_W", "yoloe-11s-seg.pt")


def _extra(ap):
    ap.add_argument("--prompt", default=os.environ.get("YOLOE_PROMPT", "black pupil"))
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=640)


def load_fn(device, a=None):
    from ultralytics import YOLOE
    m = YOLOE(MODEL_NAME)
    prompt = getattr(a, "prompt", "black pupil")
    names = [s.strip() for s in prompt.replace(".", ",").split(",") if s.strip()]
    m.set_classes(names, m.get_text_pe(names))
    m._pupil_conf = getattr(a, "conf", 0.05)
    m._pupil_imgsz = getattr(a, "imgsz", 640)
    return m


def detect_fn(model, img, device):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    dev = 0 if device == "cuda" else "cpu"
    res = model.predict(rgb, imgsz=model._pupil_imgsz, conf=model._pupil_conf,
                        iou=0.7, verbose=False, device=dev)
    r = res[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return None, None
    confs = boxes.conf.cpu().numpy()
    idx = int(confs.argmax())
    score = float(confs[idx])
    if r.masks is not None and idx < len(r.masks.xy):          # seg variant -> polygon (orig coords)
        poly = r.masks.xy[idx]
        if poly is not None and len(poly) >= 3:
            mask = np.zeros(img.shape, np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            if mask.sum() >= 5:
                return mask, score
    x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[idx]             # fallback: box -> filled ellipse
    mask = np.zeros(img.shape, np.uint8)
    cv2.ellipse(mask, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                (max(int((x2 - x1) / 2), 1), max(int((y2 - y1) / 2), 1)), 0, 0, 360, 1, -1)
    return mask, score


if __name__ == "__main__":
    annlib.run_tool("yoloe", load_fn, detect_fn, extra_args=_extra)
