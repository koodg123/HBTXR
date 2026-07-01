"""
08_run_gsam2.py — Grounded-SAM-2 audit annotator over collected samples.

For every collected frame it runs Grounding DINO (text="pupil.") -> best box ->
SAM 2 -> pupil mask -> ellipse-fit center = y_gsam2. For the annotation-precision
proxy it repeats SAM 2 with box jitter (+optional h-flip TTA) and stores the spread.

Writes:  samples/label/{key}/gsam2.json  = {"gsam2_centers": [ per-frame {...} ]}
mirroring unet_dense.json so 10_eval can load them the same way.

RUN THIS IN THE GROUNDED-SAM-2 GPU ENV (needs torch + sam2 + groundingdino + cv2).
Weights: analysis/annotation-analysis/weights/  (see weights/README.md).

Example:
  python 08_run_gsam2.py --out ../samples \
    --sam2-ckpt ../weights/sam2.1_hiera_large.pt --sam2-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --gdino-ckpt ../weights/groundingdino_swint_ogc.pth \
    --gdino-cfg grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --prompt "pupil." --repeats 4 --tta --anchors-only     # fast anchor-only first
"""
import os, csv, glob, json, argparse, sys
import numpy as np
from PIL import Image
import evlib as ev

# --- model imports (repo layout differs across installs) ---------------------
try:
    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
    import grounding_dino.groundingdino.datasets.transforms as GDT
except Exception:
    from groundingdino.util.inference import load_model, load_image, predict
    import groundingdino.datasets.transforms as GDT
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from torchvision.ops import box_convert

_GDTF = None


def _gdino_tensor(rgb):
    """GroundingDINO input tensor for an RGB HWC uint8 array (full frame or ROI crop)."""
    global _GDTF
    if _GDTF is None:
        _GDTF = GDT.Compose([GDT.RandomResize([800], max_size=1333), GDT.ToTensor(),
                             GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t, _ = _GDTF(Image.fromarray(rgb), None)
    return t


def load_models(a):
    dev = a.device
    sam2 = build_sam2(a.sam2_cfg, a.sam2_ckpt, device=dev)
    sam2_predictor = SAM2ImagePredictor(sam2)
    gdino = load_model(model_config_path=a.gdino_cfg,
                       model_checkpoint_path=a.gdino_ckpt, device=dev)
    if dev == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return gdino, sam2_predictor, dev


def gdino_box(gdino, image, w, h, a, roi_rgb=None):
    """Return best pupil-plausible box (xyxy px in the given image) + score, or (None, None).

    Filters: drop boxes wider/taller than `a.max_box_frac` (whole-image / eye-spanning)
    and smaller than `a.min_box` px (marker blobs). Selection: if `a.geom_select` and
    `roi_rgb` given, pick by geometry score (conf + darkness + roundness) — the pupil is
    the dark, round region — else argmax confidence."""
    boxes, conf, labels = predict(model=gdino, image=image, caption=a.prompt,
                                  box_threshold=a.box_thr, text_threshold=a.text_thr, device=a.device)
    if boxes is None or len(boxes) == 0:
        return None, None
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    conf = conf.numpy()
    bw, bh = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    minb = getattr(a, "min_box", 0) or 0
    keep = np.where((bw <= a.max_box_frac * w) & (bh <= a.max_box_frac * h)
                    & (bw >= minb) & (bh >= minb))[0]
    if keep.size == 0:
        return None, None
    if getattr(a, "geom_select", False) and roi_rgb is not None:
        gray = np.asarray(Image.fromarray(roi_rgb).convert("L"))
        best, bs = None, -1e9
        for i in keep:
            X0, Y0, X1, Y1 = xyxy[i]
            reg = gray[int(max(0, Y0)):int(Y1), int(max(0, X0)):int(X1)]
            dark = 1.0 - reg.mean() / 255.0 if reg.size else 0.0
            rnd = min(bw[i], bh[i]) / max(bw[i], bh[i])
            s = float(conf[i]) + 0.5 * dark + 0.5 * rnd
            if s > bs:
                bs, best = s, int(i)
        bi = best
    else:
        bi = int(keep[np.argmax(conf[keep])])
    return xyxy[bi], float(conf[bi])


def sam2_center(predictor, box_xyxy, pupil_max=None, min_area=120):
    """box_xyxy: (4,) -> (ellipse-fit center dict, score, binary mask uint8 0/1).

    SAM2 multimask_output returns ~3 nested masks {pupil, pupil+iris, eye}. Taking the
    top SAM score can select the IRIS on low-contrast frames -- and since pupil/iris are
    concentric the center still matches, a blind spot for center metrics. If `pupil_max`
    is set, restrict candidates to masks with area in [min_area, pupil_max] (drops the
    larger iris/eye masks by size), then take the highest-score among them; else argmax."""
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None,
                                          box=box_xyxy[None, :], multimask_output=True)
    masks = np.asarray(masks); scores = np.asarray(scores)
    m = masks[0] if masks.ndim == 4 else masks
    sc = scores[0] if scores.ndim == 2 else scores
    if pupil_max is not None:
        areas = m.reshape(m.shape[0], -1).sum(axis=1)
        cand = np.where((areas >= min_area) & (areas <= pupil_max))[0]
        bi = int(cand[np.argmax(sc[cand])]) if cand.size else int(np.argmax(sc))
    else:
        bi = int(np.argmax(sc))
    mask = m[bi].astype("uint8")
    st = ev.mask_stats(mask)
    return st, float(sc[bi]), mask


def center_of(st):
    if not st:
        return None
    e = st.get("ellipse")
    if e:
        return [e["cx"], e["cy"]]
    return st.get("centroid")


def _save_gsam2_mask(frame_path, out, mask):
    """Save SAM2 binary mask as 0/255 PNG, mirroring unet_masks/ naming:
    label/{key}/gsam2_masks/{stem}_mask.png (key/stem inferred from frame_path)."""
    key = os.path.basename(os.path.dirname(frame_path))
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    mdir = ev.p(out, "label", key, "gsam2_masks")
    os.makedirs(mdir, exist_ok=True)
    Image.fromarray((mask.astype("uint8") * 255)).save(ev.p(mdir, stem + "_mask.png"))


def process_frame(path, models, a):
    gdino, sam2_predictor, dev = models
    image_source, image = load_image(path)  # RGB HWC + full-frame transform
    H, W = image_source.shape[:2]
    roi = getattr(a, "roi", None)
    if roi:  # crop to eye ROI (removes bottom calibration markers from view)
        x0, y0 = max(0, roi[0]), max(0, roi[1])
        x1, y1 = min(W, roi[2]), min(H, roi[3])
        work = np.ascontiguousarray(image_source[y0:y1, x0:x1])
        img_t = _gdino_tensor(work)
    else:
        x0, y0, work, img_t = 0, 0, image_source, image
    wh, ww = work.shape[:2]
    pmax = a.pupil_area_max if getattr(a, "pupil_select", False) else None
    sam2_predictor.set_image(work)
    box, score = gdino_box(gdino, img_t, ww, wh, a, roi_rgb=work)  # coords in `work`
    if box is None:
        return {"valid": False, "reason": "no_detection"}
    st, msc, mask = sam2_center(sam2_predictor, box, pmax)
    c = center_of(st)
    if c is None:
        return {"valid": False, "reason": "empty_mask"}
    reps = []  # precision proxy: jittered boxes (+ optional hflip TTA); restored to full-frame
    rng = np.random.default_rng(0)
    for _ in range(max(0, a.repeats)):
        jb = box + rng.uniform(-a.jitter, a.jitter, size=4)
        jb = np.clip(jb, [0, 0, 0, 0], [ww, wh, ww, wh])
        st_j, _, _ = sam2_center(sam2_predictor, jb, pmax)
        cj = center_of(st_j)
        if cj:
            reps.append([cj[0] + x0, cj[1] + y0])
    if a.tta:
        flip = work[:, ::-1, :].copy()
        sam2_predictor.set_image(flip)
        fb = np.array([ww - box[2], box[1], ww - box[0], box[3]])
        st_f, _, _ = sam2_center(sam2_predictor, fb, pmax)
        cf = center_of(st_f)
        if cf:
            reps.append([(ww - cf[0]) + x0, cf[1] + y0])
        sam2_predictor.set_image(work)  # restore
    cx, cy = c[0] + x0, c[1] + y0                       # -> full-frame coords
    box_full = [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0]
    if getattr(a, "save_masks", False):
        full_mask = np.zeros((H, W), dtype="uint8")     # paste ROI mask into full canvas
        full_mask[y0:y0 + wh, x0:x0 + ww] = mask
        _save_gsam2_mask(path, a.out, full_mask)
    return {"valid": True, "cx": cx, "cy": cy,
            "area": st.get("area"), "det_score": round(score, 4),
            "sam_score": round(msc, 4), "box": [round(float(v), 1) for v in box_full],
            "repeats": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../samples")
    ap.add_argument("--sam2-ckpt", default="../weights/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--gdino-ckpt", default="../weights/groundingdino_swint_ogc.pth")
    ap.add_argument("--gdino-cfg",
                    default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--prompt", default="pupil.", help="lowercase, end with '.'")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.20)
    ap.add_argument("--max-box-frac", type=float, default=0.55,
                    help="reject boxes wider/taller than this fraction of the image "
                         "(drops whole-image/eye-spanning GDINO hits; pupil is tiny)")
    ap.add_argument("--roi", default=None,
                    help="crop to eye ROI 'x0,y0,x1,y1' before detection (removes bottom "
                         "markers; coords restored to full 346x260). e.g. 25,10,325,195")
    ap.add_argument("--min-box", type=float, default=0.0,
                    help="reject boxes with width or height < this (px); drops tiny marker blobs")
    ap.add_argument("--geom-select", action="store_true",
                    help="pick box by geometry (conf+darkness+roundness) instead of argmax conf")
    ap.add_argument("--pupil-select", action="store_true",
                    help="among SAM2 multimask outputs pick the pupil-sized mask (avoids iris; "
                         "pupil/iris are concentric so score alone can pick the larger iris)")
    ap.add_argument("--pupil-area-max", type=float, default=3000.0,
                    help="max mask area (px) counted as pupil for --pupil-select (iris >~3200)")
    ap.add_argument("--repeats", type=int, default=4, help="jittered SAM2 repeats (precision proxy)")
    ap.add_argument("--jitter", type=float, default=3.0, help="box jitter px")
    ap.add_argument("--tta", action="store_true", help="add h-flip TTA repeat")
    ap.add_argument("--anchors-only", action="store_true", help="process only anchor frames (fast)")
    ap.add_argument("--save-masks", action="store_true",
                    help="save SAM2 binary mask PNG to label/{key}/gsam2_masks/{stem}_mask.png")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.roi:
        args.roi = tuple(int(v) for v in args.roi.split(","))

    # role map from manifest_frames.csv
    roles = {}
    mf = ev.p(args.out, "manifest_frames.csv")
    if os.path.isfile(mf):
        with open(mf, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                roles[(r["key"], int(r["frame_index"]))] = r["role"]

    keys = sorted(os.listdir(ev.p(args.out, "label")))
    print(f"[i] {len(keys)} windows; device={args.device}; loading models ...")
    models = load_models(args)

    done = nvalid = ntotal = 0
    for key in keys:
        ldir = ev.p(args.out, "label", key)
        outp = ev.p(ldir, "gsam2.json")
        if os.path.isfile(outp) and not args.overwrite:
            continue
        fdir = ev.p(args.out, "frame", key)
        frames = sorted(glob.glob(ev.p(fdir, "*.png")))
        rows = []
        for fp in frames:
            idx = None
            base = os.path.basename(fp)
            try:
                idx = int(base.split("_")[0])
            except ValueError:
                pass
            if args.anchors_only and roles.get((key, idx)) != "anchor":
                continue
            res = process_frame(fp, models, args)
            res["idx"] = idx
            rows.append(res)
            ntotal += 1
            nvalid += int(res.get("valid", False))
        ev.dump_json({"prompt": args.prompt, "gsam2_centers": rows}, outp)
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(keys)} windows  (valid {nvalid}/{ntotal})")
    print(f"\n[done] windows={done}  frames={ntotal}  valid={nvalid} "
          f"({100*nvalid/max(1,ntotal):.1f}%)")
    print("  -> samples/label/*/gsam2.json ready for 10_eval")
    if ntotal and nvalid / ntotal < 0.5:
        print("[!] low valid rate: lower --box-thr/--text-thr or refine --prompt "
              "(e.g. 'pupil. black pupil.').")


if __name__ == "__main__":
    main()
