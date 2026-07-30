"""
rename_bootstrap_ci_B.py
Renames encoding/ansatz columns in bootstrap_ci_results_B.csv
to exact paper display names and saves as bootstrap_ci_results_renamed_B.csv

Run:
  python rename_bootstrap_ci_B.py \
      --input  /path/to/bootstrap_ci_results_B.csv \
      --output /path/to/bootstrap_ci_results_renamed_B.csv
"""
import argparse
import pandas as pd
from pathlib import Path

ENCODING_RENAME = {
    "rx_embedding":       "RX_Angle_Encoding",
    "ry_embedding":       "RY_Angle_Encoding",
    "rz_embedding":       "RZ_Angle_Encoding",
    "iqp_embedding":      "IQP_style_encoding",
    "zz_feature_map":     "ZZ_Feature_Map_style",
    "amplitude_embedding":"Amplitude_Embedding",
    "custom_h_ry_rz":     "Custom_H_RY_RZ",
}
ANSATZ_RENAME = {
    "efficient_su2_like": "EfficientSU2_like",
    "real_amplitudes":    "Real_Amplitude_like",
    "custom_ansatz_1":    "custom_ansatz_1",
    "strongly_entangling":"strongly_entangling",
}
DRM_RENAME = {
    "pca4":        "PCA4",
    "ica4":        "ICA4",
    "xgb_pca4":   "XGB_PCA4",
    "autoencoder": "Autoencoder",
}

def rename_row(enc, ans, fs):
    enc_d = ENCODING_RENAME.get(str(enc).lower().strip(), enc)
    ans_d = ANSATZ_RENAME.get(str(ans).lower().strip(), ans)
    drm_d = DRM_RENAME.get(str(fs).lower().strip(), str(fs).upper())
    drm_vqc = f"{drm_d}+VQC({enc_d}+{ans_d})"
    return enc_d, ans_d, drm_d, drm_vqc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded: {len(df)} rows from {Path(args.input).name}")
    print(f"Columns: {list(df.columns)}")

    # Find encoding, ansatz, feature_set columns
    enc_col = next((c for c in df.columns if c.lower() == "encoding"), None)
    ans_col = next((c for c in df.columns if c.lower() == "ansatz"),   None)
    fs_col  = next((c for c in df.columns if "feature_set" in c.lower()), None)

    if not all([enc_col, ans_col, fs_col]):
        print(f"  ERROR: Missing columns. Found: enc={enc_col} ans={ans_col} fs={fs_col}")
        return

    # Apply rename
    renamed = df.apply(
        lambda r: rename_row(r[enc_col], r[ans_col], r[fs_col]),
        axis=1, result_type="expand"
    )
    df["encoding"]      = renamed[0]
    df["ansatz"]        = renamed[1]
    df["DRM"]           = renamed[2]
    df["DRM+VQC(Encoding+Ansatz)"] = renamed[3]

    # Reorder: put DRM+VQC as first column
    cols = ["DRM+VQC(Encoding+Ansatz)", "DRM"] + \
           [c for c in df.columns
            if c not in ["DRM+VQC(Encoding+Ansatz)", "DRM"]]
    df = df[cols]
    df.to_csv(args.output, index=False)
    print(f"✓ Saved: {args.output}")
    print(f"\n  Sample DRM+VQC names:")
    for name in df["DRM+VQC(Encoding+Ansatz)"].head(5):
        print(f"    {name}")
    print(f"\n  f1_pm_B range: {df['f1_pm_B'].head(5).tolist()}")

if __name__ == "__main__":
    main()
