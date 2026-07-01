"""SAM3-I (instruction-tuned SAM3, text-prompted) pupil center on EV-Eye anchors.
env: .venv-gsam2 (torch 2.5.1) + SAM3-I deps (timm/einops/ftfy/regex/iopath).
STATUS: UNVALIDATED / GPU-DEFERRED. checkpoint-002.pt is 9.8GB and inference needs
~14-18GB GPU; will NOT fit while the FACET training holds the GPU. RGB-trained ->
near-IR domain gap (replicate gray to 3ch). Run ONLY when a >=20GB GPU is free.
Recipe: third/SAM3-I/scripts/inference.py + sam3/model_builder.py.

  cd scripts/annotators && ../../.venv-gsam2/bin/python run_sam3i.py --device cuda \
      --prompt "pupil" --good-only
"""
import sys, os
import numpy as np
import cv2

SAM3 = "/home/user/project/PRJXR-HBTXR/HBTXR/third/SAM3-I"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SAM3)
sys.path.insert(0, os.path.join(SAM3, "sam3"))
import annlib

CKPT = os.environ.get("SAM3I_W", f"{annlib.AA}/weights/SAM3-I/checkpoint-002.pt")


def _extra(ap):
    ap.add_argument("--prompt", default=os.environ.get("SAM3I_PROMPT", "pupil"))
    ap.add_argument("--force", action="store_true", help="skip GPU-memory preflight")


def _gpu_free_gb():
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return 0.0


def load_fn(device, a=None):
    if device != "cuda":
        raise SystemExit("[sam3i] CPU inference infeasible (ViT-L, no quantization). Use --device cuda.")
    free = _gpu_free_gb()
    if free < 14 and not getattr(a, "force", False):
        raise SystemExit(f"[sam3i] DEFERRED: only {free:.1f}GB GPU free; need ~14-18GB. "
                         f"Free the GPU (FACET training) then rerun, or pass --force to try anyway.")
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        checkpoint_path=CKPT,
        bpe_path=os.path.join(SAM3, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
        device=device, eval_mode=True, enable_segmentation=True,
        inst_stage="1_1", load_from_HF=False)
    model.eval()
    model._prompt = getattr(a, "prompt", "pupil")
    return model


def detect_fn(model, img, device):
    # Build a single-image Datapoint per scripts/inference.py, run forward, take top mask.
    from PIL import Image
    from sam3.train.data.sam3_image_dataset import (
        Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata)
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI)
    from sam3.train.data.collator import collate_fn_api
    import torch

    pil = Image.fromarray(img).convert("RGB")
    tf = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    dp = Datapoint(find_queries=[], images=[SAMImage(data=pil, objects=[], size=[img.shape[0], img.shape[1]])])
    dp.find_queries.append(FindQueryLoaded(
        query_text=model._prompt, image_id=0, object_ids_output=[], is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0, original_image_id=0, original_category_id=1,
            original_size=[img.shape[0], img.shape[1]], object_id=0, frame_index=0)))
    dp = tf(dp)
    batch = collate_fn_api([dp], dict_key="d")["d"]
    with torch.inference_mode():
        out = model(batch, inst_stage="1_1")
    pm = out["pred_masks"]
    if pm is None or pm.shape[1] == 0:
        return None, None
    scores = out.get("pred_logits")
    idx = int(scores[0].argmax()) if scores is not None and scores.numel() else 0
    m = (pm[0, idx] > 0.0).float().cpu().numpy().astype(np.uint8)   # 1008x1008 logits->mask
    score = float(scores[0, idx]) if scores is not None and scores.numel() else None
    return (m if m.sum() >= 5 else None), score


if __name__ == "__main__":
    annlib.run_tool("sam3i", load_fn, detect_fn, extra_args=_extra)
