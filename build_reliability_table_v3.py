"""
build_reliability_table_v3.py
Fixes the rank mismatch issue by:
  - Using combined_summary_metrics.csv (internal names) as rank source
  - Merging ALL files on raw (feature_set, ansatz, encoding) keys
  - NEVER building DRM+VQC strings for merging — only for final display
  - Applying display names ONLY at the very last output step

Inputs (all use internal encoding/ansatz names):
  bootstrap_ci_results_B.csv   — true bootstrap (replace=True)
  uncertainty_results.csv      — σ_epi, H, ECE, C*
  combined_summary_metrics.csv — rank source (row order = F1 rank)
  stage1_screened_models.csv   — count only

Run:
  python build_reliability_table_v3.py \
      --results_dir /home/nvidia/.../results_vqc_UNSW_NB15_v8_default_mixed \
      --dataset_name UNSW_NB15 \
      --top_n 10 \
      --out_root /home/nvidia/21PHD1192/qml_id2
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Display name maps (applied ONLY at output stage) ──────────────────────
ENC_RENAME = {
    "rx_embedding":       "RX_Angle_Encoding",
    "ry_embedding":       "RY_Angle_Encoding",
    "rz_embedding":       "RZ_Angle_Encoding",
    "iqp_embedding":      "IQP_style_encoding",
    "zz_feature_map":     "ZZ_Feature_Map_style",
    "amplitude_embedding":"Amplitude_Embedding",
    "custom_h_ry_rz":     "Custom_H_RY_RZ",
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

KEY_COLS = ["feature_set", "ansatz", "encoding"]

B_REMAP = {
    "f1_pm_B":        "f1_pm",
    "f1_ci_str_B":    "f1_ci_str",
    "f1_ci_low_B":    "f1_ci_low",
    "f1_ci_high_B":   "f1_ci_high",
    "f1_ci_width_B":  "f1_ci_width",
    "f1_mean_B":      "f1_mean",
    "f1_std_B":       "f1_std",
    "auc_pm_B":       "auc_pm",
    "auc_ci_str_B":   "auc_ci_str",
    "auc_ci_low_B":   "auc_ci_low",
    "auc_ci_high_B":  "auc_ci_high",
    "auc_ci_width_B": "auc_ci_width",
    "auc_mean_B":     "auc_mean",
    "auc_std_B":      "auc_std",
    "tpr_pm_B":       "tpr_pm",
    "tpr_ci_str_B":   "tpr_ci_str",
    "fpr_pm_B":       "fpr_pm",
    "fpr_ci_str_B":   "fpr_ci_str",
}

def extract_mean(val):
    if isinstance(val, str) and "±" in val:
        try: return float(val.split("±")[0].strip())
        except: return 0.0
    try: return float(val)
    except: return 0.0

def normalise_keys(df):
    """Lowercase and strip key columns so merges are case-insensitive."""
    for col in KEY_COLS:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()
    return df

def make_drm_vqc(row):
    """Build paper-style DRM+VQC display string from internal names."""
    fs  = DRM_RENAME.get(str(row.get("feature_set","")).lower(),
                          str(row.get("feature_set","")).upper())
    enc = ENC_RENAME.get(str(row.get("encoding","")).lower(),
                          str(row.get("encoding","")))
    ans = ENC_RENAME.get(str(row.get("ansatz","")).lower(),
                          str(row.get("ansatz","")))
    return f"{fs}+VQC({enc}+{ans})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",  required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--top_n",        type=int, default=10)
    ap.add_argument("--out_root",     default="/home/nvidia/21PHD1192/qml_id2")
    args = ap.parse_args()

    rdir = Path(args.results_dir)

    # ── File paths ─────────────────────────────────────────────────────────
    boot_path   = rdir / "bootstrap_ci_results_B.csv"
    unc_path    = rdir / "uncertainty_results.csv"
    rank_path   = rdir / "combined_summary_metrics.csv"  # rank source
    stage1_path = rdir / "stage1_screened_models.csv"

    for p, name in [(boot_path,"bootstrap_ci_results_B.csv"),
                    (unc_path, "uncertainty_results.csv"),
                    (rank_path,"combined_summary_metrics.csv")]:
        if not p.exists():
            print(f"✗ {name} not found at {p}"); return

    # ── Load rank from combined_summary_metrics.csv (internal names) ──────
    # Row order IS the rank — sorted by F1 descending by the training pipeline
    rank_df = pd.read_csv(rank_path)
    rank_df = normalise_keys(rank_df)
    rank_df["rank_in_112"] = range(1, len(rank_df)+1)
    n_total = len(rank_df)
    print(f"Rank source: {rank_path.name}  ({n_total} models, "
          f"row order = F1 rank from training pipeline)")

    # ── Stage 1 count ─────────────────────────────────────────────────────
    if stage1_path.exists():
        n_screened = len(pd.read_csv(stage1_path))
    else:
        n_screened = None

    # ── Load bootstrap (_B columns) and remap to standard names ──────────
    boot = pd.read_csv(boot_path)
    boot = normalise_keys(boot)
    for old, new in B_REMAP.items():
        if old in boot.columns and new not in boot.columns:
            boot.rename(columns={old: new}, inplace=True)

    # ── Load uncertainty ───────────────────────────────────────────────────
    unc  = pd.read_csv(unc_path)
    unc  = normalise_keys(unc)

    print(f"\n{'='*65}")
    print(f"  MODEL SCREENING SUMMARY  —  {args.dataset_name}")
    print(f"{'='*65}")
    print(f"  Full factorial grid:                {n_total} models")
    if n_screened:
        print(f"  Stage 1 screened:                   {n_screened} models "
              f"({100*n_screened/n_total:.1f}%)")
        print(f"  Excluded by Stage 1:                {n_total - n_screened} models")
    print(f"  Bootstrap file rows:                {len(boot)}")
    print(f"  Uncertainty file rows:              {len(unc)}")
    print(f"{'='*65}\n")

    # ── Merge 1: bootstrap + uncertainty (on raw internal keys) ──────────
    merged = boot.merge(unc, on=KEY_COLS, how="inner", suffixes=("","_unc"))
    print(f"After boot+unc merge: {len(merged)} models matched")
    if len(merged) < len(boot):
        miss = len(boot) - len(merged)
        print(f"  ⚠ {miss} bootstrap models not matched in uncertainty file")
        # Show which ones
        unmatched = boot.merge(unc, on=KEY_COLS, how="left", indicator=True)
        unmatched = unmatched[unmatched["_merge"]=="left_only"][KEY_COLS]
        print(f"  Unmatched: {unmatched.to_dict('records')[:5]}")

    # ── Merge 2: attach rank from combined_summary_metrics.csv ────────────
    rank_keys = rank_df[KEY_COLS + ["rank_in_112"]].copy()
    merged = merged.merge(rank_keys, on=KEY_COLS, how="left")
    n_missing_rank = merged["rank_in_112"].isna().sum()
    if n_missing_rank > 0:
        print(f"\n  ⚠ {n_missing_rank} models could not be matched for rank "
              f"(name mismatch between bootstrap and combined_summary_metrics):")
        bad = merged[merged["rank_in_112"].isna()][KEY_COLS]
        for _, r in bad.iterrows():
            print(f"    {dict(r)}")
        # Fallback: assign rank by f1_pm descending
        print(f"  Fallback: assigning rank by f1_pm for unmatched models")
        merged["_f1_num"] = merged["f1_pm"].apply(extract_mean)
        merged["rank_in_112"] = merged["rank_in_112"].fillna(
            merged["_f1_num"].rank(method="dense", ascending=False))
        merged = merged.drop(columns=["_f1_num"], errors="ignore")
    else:
        print(f"  ✓ All {len(merged)} models successfully matched for rank")

    # ── Build DRM+VQC display string (AFTER all merges) ───────────────────
    merged["DRM+VQC"] = merged.apply(make_drm_vqc, axis=1)

    # ── Sort by reliability ───────────────────────────────────────────────
    merged["_f1_num"] = merged["f1_pm"].apply(extract_mean)
    n_collapsed = (merged["_f1_num"] < 0.01).sum()
    if n_collapsed > 0:
        print(f"\n  ⚠ {n_collapsed} model(s) show f1_pm ≈ 0 (threshold fragility)."
              f" Sorted to bottom, excluded from top-{args.top_n}.")

    merged = merged.sort_values(
        ["_f1_num", "deploy_coverage", "sigma_epi_mean"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    merged = merged.drop(columns=["_f1_num"], errors="ignore")

    # ── Select display columns ────────────────────────────────────────────
    display_cols = ["DRM+VQC", "rank_in_112",
                    "f1_pm",  "f1_ci_str", "f1_ci_width",
                    "auc_pm", "auc_ci_str","auc_ci_width",
                    "tpr_pm", "fpr_pm",
                    "sigma_epi_mean", "entropy_mean", "ece",
                    "deploy_coverage"]
    display_cols = [c for c in display_cols if c in merged.columns]
    merged = merged.rename(columns={
        "rank_in_112":   "Perf. Rank (of 112)",
        "f1_ci_str":     "F1_CI_95%",
        "f1_ci_width":   "F1_CI_width",
        "auc_ci_str":    "AUC_CI_95%",
        "auc_ci_width":  "AUC_CI_width",
    })
    display_cols = [
        "Perf. Rank (of 112)" if c == "rank_in_112"  else
        "F1_CI_95%"           if c == "f1_ci_str"    else
        "F1_CI_width"         if c == "f1_ci_width"  else
        "AUC_CI_95%"          if c == "auc_ci_str"   else
        "AUC_CI_width"        if c == "auc_ci_width" else c
        for c in display_cols
    ]

    appendix_table = merged[display_cols].reset_index(drop=True)
    main_table     = appendix_table.iloc[:args.top_n].reset_index(drop=True)

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_path     = out_dir / f"reliability_main_top{args.top_n}_{args.dataset_name}_v3.csv"
    appendix_path = out_dir / f"reliability_appendix_{args.dataset_name}_v3.csv"
    main_table.to_csv(main_path, index=False)
    appendix_table.to_csv(appendix_path, index=False)

    print(f"\n✓ Main table    ({len(main_table)} rows) → {main_path}")
    print(f"✓ Appendix table({len(appendix_table)} rows) → {appendix_path}")
    print(f"\nMain preview:")
    with pd.option_context("display.max_columns",None,"display.width",180):
        cols_show = ["DRM+VQC","Perf. Rank (of 112)","f1_pm",
                     "F1_CI_width","auc_pm","AUC_CI_width",
                     "sigma_epi_mean","deploy_coverage"]
        show = [c for c in cols_show if c in main_table.columns]
        print(main_table[show].to_string(index=False))

if __name__ == "__main__":
    main()
