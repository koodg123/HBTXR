"""align.py (E0.2) — unified per-anchor center table in the 346x260 frame.

Long format rows: [key, subject, motion, source, cx, cy] for
source in {human_ellipse, unet, gsam2, pred, ellseg, ritnet, edge_guided,
deepvog, yoloe, sam2_repeat_0..k}.

human_mask is intentionally OMITTED as a per-anchor source: it is DERIVED from the
ellipse (rasterization, verified) so it ~duplicates human_ellipse; its floor is
quantified separately in repeatability.La.3. y_pred is included but consumers must
only use it in corrected_error (STEP5).
Writes tables/align_centers.csv and returns the row list + a validation dict.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S


def build():
    ac = S.load_annotator_centers()
    rows = []
    for a in S.anchors():
        key = a["key"]

        def add(src, xy):
            if xy is not None:
                rows.append(dict(key=key, subject=a["subject"], motion=a["motion"],
                                 source=src, cx=round(float(xy[0]), 4), cy=round(float(xy[1]), 4)))
        add("human_ellipse", a["human"])
        add("unet", a["unet"])
        add("gsam2", a["gsam2"])
        add("pred", a["pred"])
        for t in S.ANNOT:
            add(t, ac[t].get(key))
        if a["repeats"]:
            for k, p in enumerate(a["repeats"]):
                add(f"sam2_repeat_{k}", (p[0], p[1]))
    return rows


def validate(rows):
    from collections import Counter
    cnt = Counter(r["source"] for r in rows)
    bad = [r for r in rows if not (0 <= r["cx"] <= S.W and 0 <= r["cy"] <= S.H)]
    return dict(n_rows=len(rows), by_source=dict(cnt), out_of_frame=len(bad),
                n_anchors=len(set(r["key"] for r in rows)))


def main():
    rows = build()
    v = validate(rows)
    os.makedirs(S.TAB, exist_ok=True)
    with open(f"{S.TAB}/align_centers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "subject", "motion", "source", "cx", "cy"])
        w.writeheader()
        w.writerows(rows)
    print(f"[align] anchors={v['n_anchors']} rows={v['n_rows']} out_of_frame={v['out_of_frame']}")
    print("[align] by_source=" + ", ".join(f"{k}:{val}" for k, val in sorted(v["by_source"].items())))
    print(f"[align] -> {S.TAB}/align_centers.csv  (frame=346x260, all sources unified)")
    return v


if __name__ == "__main__":
    main()
