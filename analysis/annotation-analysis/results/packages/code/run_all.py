"""run_all.py — DAG runner: E0 -> align -> La -> Lb -> Lc -> STEP5 -> figures.
Aggregates all scalars to results/precision/precision_summary.json, writes a gate log
with [DONE/SKIP/FALLBACK/OPTIMISTIC] per step. Run in .venv-gsam2 (needs h5py+scipy+cv2).

  ../.venv-gsam2/bin/python src/run_all.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_schema as S
import align, repeatability, reproducibility, accuracy_view, corrected_error, figures, tables_out


def main():
    log = []

    def gate(step, status, note):
        line = f"[{status}] {step}: {note}"
        log.append(line)
        print("  >> " + line)

    print("=" * 72)
    print("Annotation Precision / Label Noise — DAG (346x260, samples=users 1-10)")
    print("=" * 72)
    S.write_e0_report()
    gate("E0 schema", "DONE", "data+code verified -> results/e0_schema_report.md")

    summary = {}
    print("\n--- E0.2 align ---")
    summary["align"] = align.main()
    gate("E0.2 align", "DONE", f"{summary['align']['n_rows']} rows / {summary['align']['n_anchors']} anchors, out_of_frame={summary['align']['out_of_frame']}")

    print("\n--- Layer A: repeatability ---")
    summary["La"] = repeatability.main()
    gate("La.1 GSAM2 repeats", "DONE", "perturbation sigma (precision floor proxy)")
    gate("La.2-human", "SKIP", "human labels sparse -> improved GSAM2 proxy (decision 4)")
    gate("La.3 raster floor", "DONE", "mask DERIVED from ellipse (not human precision)")

    print("\n--- Layer B: reproducibility ---")
    summary["Lb"] = reproducibility.main()
    sb = summary["Lb"]["Lb2_3ch"]["shared_bias"]
    gate("Lb.2 3CH{h,gsam2,unet}", "FALLBACK-BRACKET" if sb else "DONE",
         "U-Net~derived-human; sigma_human=bracket[3CH, Human-GSAM2 RMS]")

    print("\n--- Layer C: accuracy_view ---")
    summary["Lc"] = accuracy_view.main()
    gate("Lc.1 P_n", "DONE", "inter-method + model, abstain separated")

    print("\n--- STEP 5: corrected error ---")
    summary["S5"] = corrected_error.main()
    gate("STEP5 corrected error", "OPTIMISTIC", "users 1-10 = TRAIN subj (test=37-48) -> NOT subject-independent")

    with open(f"{S.OUT}/precision_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)

    print("\n--- tables + report ---")
    outs = tables_out.write(summary)
    gate("tables + report", "DONE", f"{len(outs)} files (tables/*.csv + precision_report.md)")

    print("\n--- figures ---")
    figures.main(summary)
    gate("figures", "DONE", "fig_3ch.png, fig_budget.png")

    open(f"{S.OUT}/run_gate_log.txt", "w").write("\n".join(log) + "\n")
    print("\n" + "=" * 72)
    print("GATE LOG")
    print("=" * 72)
    for l in log:
        print("  " + l)
    print(f"\nOutputs:\n  {S.OUT}/precision_summary.json\n  {S.OUT}/run_gate_log.txt\n"
          f"  {S.RES}/e0_schema_report.md\n  {S.FIG}/fig_3ch.png, fig_budget.png\n  {S.TAB}/align_centers.csv")


if __name__ == "__main__":
    main()
