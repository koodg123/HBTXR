"""tables_out.py — write the spec'd tables/*.csv and a human-readable precision_report.md
from the aggregated precision_summary. Called by run_all (also standalone on the json)."""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S


def _round(x, n=3):
    return "" if x is None else round(float(x), n)


def write(summary):
    S5 = summary["S5"]["STEP5"]
    Lb = summary["Lb"]
    Lc = summary["Lc"]["Lc1_Pn"]
    La = summary["La"]
    os.makedirs(S.TAB, exist_ok=True)

    with open(f"{S.TAB}/per_subject.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "E_orig_median", "mean", "p95", "p99", "max", "n"])
        for k, v in S5["per_subject"].items():
            w.writerow([k, _round(v["median"]), _round(v["mean"]), _round(v["p95"], 2), _round(v["p99"], 2), _round(v["max"], 2), v["n"]])

    with open(f"{S.TAB}/per_motion.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["motion", "E_orig_median", "E_orig_p95", "n"])
        for m, v in S5["per_motion"].items():
            w.writerow([m, _round(v["median"]), _round(v["p95"], 2), v["n"]])

    with open(f"{S.TAB}/Pn_contrast.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair", "valid", "abstain", "Pn_all_10px", "Pn_all_5px", "Pn_all_1px", "median_px"])
        for name, x in Lc.items():
            w.writerow([name, x["n_valid"], x["abstain"], x["Pn_over_all"]["10"], x["Pn_over_all"]["5"], x["Pn_over_all"]["1"], _round(x["median"])])

    with open(f"{S.TAB}/repeats_crosscheck.csv", "w", newline="") as f:
        w = csv.writer(f); lb3 = Lb["Lb3_crosscheck"]
        w.writerow(["sigma_gsam2_3CH", "sigma_gsam2_repeat_La1", "ratio"])
        w.writerow([_round(lb3["sigma_s_3ch"]), _round(lb3["sigma_s_rep_La1"]), _round(lb3["ratio"], 2)])

    with open(f"{S.TAB}/pairwise_agreement.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair", "radial_rms", "var_x", "var_y", "ba_bias_x", "ba_bias_y", "IoU_vs_gt"])
        iou = Lb["Lb1_pairwise"]["IoU"]
        iou_map = {"human-gsam2": iou["gsam2_vs_gt"]["median"], "human-unet": iou["unet_vs_gt"]["median"], "gsam2-unet": iou["gsam2_vs_unet"]["median"]}
        for name, x in Lb["Lb1_pairwise"].items():
            if name == "IoU":
                continue
            w.writerow([name, _round(x["radial_rms"]), _round(x["var_x"]), _round(x["var_y"]),
                        _round(x["ba_bias_x"]), _round(x["ba_bias_y"]), _round(iou_map.get(name))])

    # ---- synthesis report ----
    br = Lb["sigma_human_bracket"]
    conv = S5["reported_0181_converted"]
    e = S5["E_orig"]
    la1 = La["La1_gsam2_repeats"]["sigma_radial"]["median"]
    la2g = La["La2_f2f_jitter"]["gsam2"]["median"]
    la3 = La["La3_rasterization_floor"]
    lb2 = Lb["Lb2_3ch"]; lb2i = Lb["Lb2_3ch_independent"]
    hg = Lb["Lb1_pairwise"]["human-gsam2"]
    R = [
        "# Annotation Precision / Label Noise — synthesis (346×260 px)\n",
        "Samples = users 1-10 (⚠ HBTXR **TRAIN** subjects; test=37-48). Layers La/Lb/Lc measure "
        "label quality (subject-split-agnostic); only STEP5 E_orig is affected by the train-subject caveat.\n",
        "## Label-noise / precision budget",
        "| quantity | px (346×260) | note |",
        "|---|---|---|",
        f"| σ_human (3CH {{h,gsam2,unet}}) | **{_round(lb2['sigma_h_radial'])}** | shared_bias={lb2['shared_bias']} |",
        f"| σ_human (3CH indep {{h,gsam2,ellseg}}) | {_round(lb2i['sigma_h_radial'])} | corroborates; shared_bias={lb2i['shared_bias']} |",
        f"| σ_human bracket | [{_round(br['lower_3ch'])}, {_round(br['upper_human_gsam2_rms'])}] | [3CH, Human↔GSAM2 RMS] |",
        f"| inter-method RMS (human–gsam2) | {_round(hg['radial_rms'])} | independent audit vs GT |",
        f"| GSAM2 perturbation σ (La.1) | {_round(la1)} | box-jitter/TTA repeatability |",
        f"| fixation F2F jitter — GSAM2 (La.2) | {_round(la2g)} | method stability (human-repeat proxy) |",
        f"| rasterization floor (La.3) | {_round(la3['dist']['median'])} | mask DERIVED (IoU {_round(la3['iou']['median'])}); ~1px constant gen-offset, not precision |",
        f"| **reported 0.1812 (64×64 vs dense) → 346×260** | **{conv['iso_approx_346']}** | range {conv['range_346']}; sits AT the σ_human floor |",
        f"| corrected E_orig (pred vs human GT) | **{_round(e['median'])}** | median; ⚠ TRAIN subj, optimistic |",
        "",
        "## Reading",
        f"- Human-GT annotation noise is **σ_human ≈ {_round(lb2['sigma_h_radial'])} px** (bracket [{_round(br['lower_3ch'])}, {_round(br['upper_human_gsam2_rms'])}]), "
        f"independently corroborated by the {{h,gsam2,ellseg}} triple ({_round(lb2i['sigma_h_radial'])}) and by inter-method RMS ({_round(hg['radial_rms'])}).",
        f"- The reported **0.1812 px is a 64×64, dense-label figure ≈ {conv['iso_approx_346']} px in 346×260** — i.e. **at the label-noise floor**. "
        "A metric measured against dense pseudo-labels near the annotation-noise floor cannot separate model error from label noise; that is why it looks 'too good'.",
        f"- Against **human GT**, the honest error is **{_round(e['median'])} px median** (mean {_round(e['mean'],2)}, p95 {_round(e['p95'],1)}, {int(round(100*S5['frac_gross_gt20']))}% gross >20px) — "
        "well ABOVE the σ_human floor → genuine model error, NOT label leakage (a leaking model would sit at the floor with no heavy tail). "
        "⚠ This E_orig is on TRAIN subjects (1-10); the subject-independent value (test 37-48) will be ≥ this.",
        f"- Bland-Altman shows U-Net carries a **systematic (+{_round(hg['ba_bias_x'],2) if False else Lb['Lb1_pairwise']['human-unet']['ba_bias_x']:.2f}, {Lb['Lb1_pairwise']['human-unet']['ba_bias_y']:.2f}) px offset** from human GT (so U-Net noise is mostly bias, not scatter).",
        "\n*Figures: fig/fig_3ch.png (σ decomposition), fig/fig_budget.png (budget). Tables: tables/*.csv. Scalars: results/precision/precision_summary.json.*",
    ]
    open(f"{S.RES}/precision_report.md", "w").write("\n".join(R) + "\n")
    return [f"{S.TAB}/per_subject.csv", f"{S.TAB}/per_motion.csv", f"{S.TAB}/Pn_contrast.csv",
            f"{S.TAB}/repeats_crosscheck.csv", f"{S.TAB}/pairwise_agreement.csv", f"{S.RES}/precision_report.md"]


if __name__ == "__main__":
    outs = write(json.load(open(f"{S.OUT}/precision_summary.json")))
    print("[tables_out] wrote:\n  " + "\n  ".join(outs))
