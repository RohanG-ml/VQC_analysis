"""
rename_display_names.py
Creates RENAMED COPIES of the two ORIGINAL files (combined_summary_metrics.csv
and new_format_combined_result.csv) inside the SAME folder — originals are
NEVER modified. Renaming applies only to encoding/ansatz values that are
"style"-inspired by IBM/Qiskit standard circuits, distinguishing our
hand-built implementations (different gate order/structure) from IBM's
literal circuits:

  iqp_embedding       -> IQP_style_encoding
  zz_feature_map      -> ZZ_Feature_Map_style
  efficient_su2_like  -> EfficientSU2_like
  real_amplitudes     -> Real_Amplitude_like

All other encoding/ansatz names (rx_embedding, ry_embedding, rz_embedding,
amplitude_embedding, custom_h_ry_rz, custom_ansatz_1, strongly_entangling)
are left unchanged — they were never IBM-circuit-derived.

Run (once per folder):
  python rename_display_names.py --results_dir /path/to/results_vqc_UNSW_NB15_5k
"""
import argparse
import pandas as pd
from pathlib import Path

RENAME_MAP = {
    "iqp_embedding":      "IQP_style_encoding",
    "zz_feature_map":     "ZZ_Feature_Map_style",
    "efficient_su2_like": "EfficientSU2_like",
    "real_amplitudes":    "Real_Amplitude_like",
}


def apply_rename(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["encoding", "ansatz"]:
        if col in df.columns:
            df[col] = df[col].replace(RENAME_MAP)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    for fname in ["combined_summary_metrics.csv", "new_format_combined_result.csv"]:
        src = rdir / fname
        if not src.exists():
            print(f"  [SKIP] {src} not found")
            continue
        df = pd.read_csv(src)
        df_renamed = apply_rename(df)
        out_path = rdir / f"{src.stem}_renamed.csv"
        df_renamed.to_csv(out_path, index=False)
        print(f"  ✓ {src.name} -> {out_path.name}  (original untouched)")


if __name__ == "__main__":
    main()
