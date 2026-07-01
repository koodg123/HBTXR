"""Validate the pupil/iris fix (--pupil-select) BEFORE committing to a full re-run.
Runs the updated 08 harness on the 4 known iris-confusion anchors + a normal sample,
with pupil-select OFF vs ON, comparing size ratio (GSAM2 radius / GT pupil radius) and
center error. Expect: iris cases ratio 1.9-2.7 -> ~1.0 (ON); normal cases ~1.0 unchanged.

Run after the background harness finishes (needs GPU):
  cd scripts && PYTHONPATH=<repo>/third/Grounded-SAM-2 ../.venv-gsam2/bin/python iris_fix_validate.py
"""
import sys, os, glob, json, importlib.util, types, math, random
import numpy as np
GS = "/home/user/project/PRJXR-HBTXR/HBTXR/third/Grounded-SAM-2"
AA = "/home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis"
os.chdir(GS); sys.path.insert(0, GS); sys.path.insert(0, os.path.join(AA, "scripts"))
spec = importlib.util.spec_from_file_location("m08", os.path.join(AA, "scripts", "08_run_gsam2.py"))
m08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m08)

IRIS = [  # known iris-confusion anchors (ratio 1.9-2.7 pre-fix)
    "smooth_pursuit_user3_left_session_2_0_1_w003_a000010",
    "saccade_user3_right_session_1_0_2_w093_a000910",
    "smooth_pursuit_user5_left_session_2_0_1_w135_a001310",
    "saccade_user6_left_session_1_0_2_w016_a000510",
]
LAB = f"{AA}/samples/label"
allk = [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(f"{LAB}/*/gt.json"))]
random.seed(0)
normal = [k for k in allk if k not in IRIS]
sample = random.sample(normal, 30)


def base_args(pupil):
    return types.SimpleNamespace(
        sam2_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", sam2_ckpt=f"{AA}/weights/gsam2/sam2.1_hiera_large.pt",
        gdino_cfg=f"{GS}/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        gdino_ckpt=f"{AA}/weights/gsam2/groundingdino_swint_ogc.pth",
        prompt="black pupil.", box_thr=0.20, text_thr=0.15, max_box_frac=0.55,
        roi=(25, 10, 325, 195), min_box=18, geom_select=True,
        pupil_select=pupil, pupil_area_max=3000.0,
        repeats=0, jitter=3.0, tta=False, device="cuda", out=f"{AA}/samples", save_masks=False)


def run(models, a, keys):
    out = {}
    for k in keys:
        gt = json.load(open(f"{LAB}/{k}/gt.json"))
        _, _, rx, ry, _ = gt["ellipse_cx_cy_rx_ry_theta"]; gx, gy = gt["ellipse_cx_cy_rx_ry_theta"][:2]
        ai = gt["anchor_idx"]
        fr = glob.glob(f"{AA}/samples/frame/{k}/{ai:06d}_*.png")
        if not fr:
            continue
        r = m08.process_frame(fr[0], models, a)
        if r.get("valid"):
            ratio = (r["area"] / math.pi) ** 0.5 / (rx * ry) ** 0.5
            err = ((r["cx"] - gx) ** 2 + (r["cy"] - gy) ** 2) ** 0.5
            out[k] = (ratio, err)
        else:
            out[k] = (None, None)
    return out

for pupil in (False, True):
    a = base_args(pupil); models = m08.load_models(a)
    ri = run(models, a, IRIS); rn = run(models, a, sample)
    tag = "pupil-select ON " if pupil else "pupil-select OFF"
    print(f"\n=== {tag} ===")
    print("  iris cases (ratio→~1.0 기대 when ON):")
    for k in IRIS:
        ratio, err = ri.get(k, (None, None))
        print(f"    {k[:44]:44s} ratio={ratio if ratio is None else round(ratio,2)}  err={err if err is None else round(err,2)}")
    nr = np.array([v[0] for v in rn.values() if v[0] is not None])
    ne = np.array([v[1] for v in rn.values() if v[1] is not None])
    print(f"  normal(n={len(nr)}): ratio median={np.median(nr):.2f} p95={np.percentile(nr,95):.2f} | center err median={np.median(ne):.2f}")
