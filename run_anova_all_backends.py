"""
run_anova_all_backends.py
Runs analyse_encoding_vs_ansatz.py separately on each backend's results
folder (3 independent ANOVA runs, NOT pooled), then builds a small
side-by-side comparison summary of eta-squared / p-values across backends.

Does NOT modify analyse_encoding_vs_ansatz.py at all — calls it as a
subprocess per folder, exactly as it already works standalone.

Run:
  python run_anova_all_backends.py \
      --folders /path/results_vqc_UNSW_NB15_5k,/path/results_vqc_UNSW_NB15_v8_default_mixed,/path/results_vqc_UNSW_NB15_v8_default_qubit \
      --analyse_script /path/to/analyse_encoding_vs_ansatz.py \
      --dataset_name UNSW_NB15 \
      --out_root /home/nvidia/21PHD1192/qml_id2 \
      --metric f1
"""
import argparse
import re
import subprocess
import pandas as pd
from pathlib import Path


def label_for_folder(folder_path: Path, dataset_name: str) -> str:
    name = folder_path.name
    lname = name.lower()
    if "mixed" in lname:
        return "default_mixed"
    if "qubit" in lname:
        return "default_qubit"
    suffix = re.sub(r"^results_vqc_", "", name, flags=re.IGNORECASE)
    suffix = re.sub(re.escape(dataset_name), "", suffix, flags=re.IGNORECASE)
    suffix = suffix.strip("_")
    suffix = re.sub(r"^v\d+_", "", suffix)
    suffix = suffix or "default"
    return f"lightning_gpu_{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", required=True)
    ap.add_argument("--analyse_script", required=True,
                     help="Path to analyse_encoding_vs_ansatz.py")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--out_root", default="/home/nvidia/21PHD1192/qml_id2")
    ap.add_argument("--metric", default="f1",
                     choices=["f1", "auc", "accuracy", "precision", "recall"])
    args = ap.parse_args()

    folder_paths = [Path(p.strip()) for p in args.folders.split(",") if p.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for p in folder_paths:
        label = label_for_folder(p, args.dataset_name)
        out_dir = out_root / f"anova_{args.dataset_name}_{label}"
        print(f"\n{'='*70}")
        print(f"  Running ANOVA for backend: {label}  ({p.name})")
        print(f"{'='*70}")

        cmd = ["python", args.analyse_script,
               "--results_root", str(p),
               "--metric", args.metric,
               "--out_dir", str(out_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout[-2000:])
        if result.returncode != 0:
            print(f"✗ FAILED for {label}:\n{result.stderr[-2000:]}")
            continue

        stats_csv = out_dir / "stats_table.csv"
        if stats_csv.exists():
            stats_df = pd.read_csv(stats_csv)
            stats_df["backend"] = label
            summary_rows.append(stats_df)
        else:
            print(f"⚠️  {stats_csv} not found after run — check output above.")

    if not summary_rows:
        print("\nNo successful ANOVA runs to summarize.")
        return

    combined = pd.concat(summary_rows, ignore_index=True)
    combined = combined[["backend", "factor", "f_stat", "p_val", "eta2", "metric"]]
    combined = combined.sort_values(["backend", "factor"])

    summary_path = out_root / f"anova_summary_{args.dataset_name}.csv"
    combined.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print(f"  CROSS-BACKEND ANOVA SUMMARY")
    print(f"{'='*70}")
    print(combined.to_string(index=False))
    print(f"\n✓ Saved -> {summary_path}")

    # Quick narrative helper
    print(f"\n{'─'*70}")
    print("  Quick comparison — which factor wins per backend:")
    print(f"{'─'*70}")
    for backend in combined["backend"].unique():
        sub = combined[combined["backend"] == backend]
        enc_eta = sub[sub["factor"] == "encoding"]["eta2"].values
        ans_eta = sub[sub["factor"] == "ansatz"]["eta2"].values
        if len(enc_eta) and len(ans_eta):
            winner = "Encoding" if enc_eta[0] > ans_eta[0] else "Ansatz"
            print(f"  {backend:<20} Encoding η²={enc_eta[0]:.4f}  "
                  f"Ansatz η²={ans_eta[0]:.4f}  -> {winner} dominates")


if __name__ == "__main__":
    main()
