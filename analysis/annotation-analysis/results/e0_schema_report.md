# E0 — Schema discovery report (data + code cross-verified)

## Assets / label types
| asset | format / frame | nature | evidence |
|---|---|---|---|
| Data_davis (ellipse) | VIA CSV region_shape_attributes(cx,cy,rx,ry,theta), 346x260 | **PRIMARY human annotation** | evaluate_hbtxr_val_motion.py:379 |
| labelled_with_mask | HDF5 data/label (346x260xN, binary) | **DERIVED from ellipse (cv2.ellipse rasterize)** | ev_eye_dataset_utils.py:103; data: axis-corrected IoU~0.95, constant (-1,-1)px offset |
| Data_davis_predict (U-Net) | mask gif | trained on DERIVED masks -> NOT independent of human | EV-Eye/train.py:92 |
| gsam2.repeats | 5 pts/anchor, 346x260 | box-jitter/TTA repeatability | present |

## Key decisions
- **Frame = 346x260** (anisotropic to 64x64: x*5.406, y*4.062). Reported **0.1812 = 64x64 vs U-Net dense**; converted only for the budget figure.
- **Samples = users 1-10 = HBTXR TRAIN subjects** (test=37-48) -> STEP5 E_orig is optimistic (caveated), per user decision.
- **La.3 = rasterization floor** (mask derived); human_mask NOT used as an independent source.
- **La.2-human SKIP** (human labels sparse) -> improved GSAM2 as repeatability proxy.
- **3CH {human, gsam2, unet}** (independence risk accepted); U-Net~derived-human so expect possible SHARED-BIAS -> bracket fallback. Optional independent triple {human, gsam2, ellseg}.
- y_pred used ONLY in STEP5.
