"""figures.py — fig_3ch (three-cornered-hat sigma decomposition) and fig_budget
(label-uncertainty budget vs corrected model error), all in 346x260 px."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S


def _v(x):
    return float(x) if x is not None else 0.0


def fig_3ch(summary):
    lb2 = summary["Lb"]["Lb2_3ch"]
    lb2i = summary["Lb"]["Lb2_3ch_independent"]
    la1med = summary["La"]["La1_gsam2_repeats"]["sigma_radial"]["median"]
    dep = [_v(lb2["sigma_h_radial"]), _v(lb2["sigma_s_radial"]), _v(lb2["sigma_u_radial"])]
    indep = [_v(lb2i["sigma_h_radial"]), _v(lb2i["sigma_s_radial"]), _v(lb2i["sigma_u_radial"])]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.bar(x - w / 2, dep, w, label="{h,gsam2,unet}" + (" [SHARED-BIAS]" if lb2["shared_bias"] else ""), color="#4477aa")
    ax.bar(x + w / 2, indep, w, label="{h,gsam2,ellseg} indep" + (" [SHARED-BIAS]" if lb2i["shared_bias"] else ""), color="#66bb99")
    ax.axhline(la1med, ls="--", c="gray", lw=1, label=f"La.1 GSAM2 repeat σ={la1med:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(["σ_human", "σ_gsam2", "σ_3rd(unet/ellseg)"])
    ax.set_ylabel("radial σ (px, 346×260)")
    ax.set_title("Three-cornered-hat noise decomposition")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{S.FIG}/fig_3ch.png", dpi=130); plt.close(fig)


def fig_budget(summary):
    la3 = summary["La"]["La3_rasterization_floor"]["dist"]["median"]
    br = summary["Lb"]["sigma_human_bracket"]
    rms = summary["Lb"]["Lb1_pairwise"]["human-gsam2"]["radial_rms"]
    e = summary["S5"]["STEP5"]["E_orig"]["median"]
    o = summary["S5"]["STEP5"]["reported_0181_converted"]["iso_approx_346"]
    items = [("reported 0.181 (64×64→346)", o),
             ("raster floor (La.3, derived)", la3),
             ("σ_human lo (3CH indep)", _v(br["lower_3ch_independent"])),
             ("inter-method RMS (human–gsam2)", rms),
             ("corrected E_orig (median, TRAIN subj)", e)]
    labels = [a for a, _ in items]; vals = [b for _, b in items]
    colors = ["#888888", "#66aa99", "#4488aa", "#ee8866", "#cc3333"]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.barh(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("px (346×260)")
    ax.set_title("Label-uncertainty budget vs corrected model error")
    ax.grid(alpha=.3, axis="x"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(f"{S.FIG}/fig_budget.png", dpi=130); plt.close(fig)


def main(summary):
    fig_3ch(summary)
    fig_budget(summary)
    print(f"[figures] -> {S.FIG}/fig_3ch.png, {S.FIG}/fig_budget.png")


if __name__ == "__main__":
    import json
    main(json.load(open(f"{S.OUT}/precision_summary.json")))
