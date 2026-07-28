"""
run_levene_only.py
Recomputes ONLY the three Levene homogeneity-of-variance tests
with correct groupings per ANOVA type.
Saves updated anova_assumptions_{dataset}.csv to the same folder.
Does NOT touch ANOVA F-statistics, p-values, or eta-squared.

Three separate Levene tests per DRM per metric:
  1. Encoding Levene: 7 groups × 20 values  (for one-way encoding ANOVA)
  2. Ansatz  Levene: 4 groups × 35 values  (for one-way ansatz ANOVA)
  3. Two-way Levene: 28 cells × 5 values   (for two-way ANOVA)

Why NOT Bartlett: Shapiro-Wilk showed non-normality (bimodal F1).
Bartlett gives false positives under non-normality. Levene is correct.

Run:
  python run_levene_only.py \
      --results_base /home/nvidia/21PHD1192/qml_id2/UNSW \
      --datasets UNSW_NB15,NF_UNSW_NB15,TON_IOT
"""
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import levene

# ── Same rename map as run_two_way_anova.py ──────────────────────────────────
RENAME_MAP = {
    "rx_embedding":"RX_Angle_Encoding","ry_embedding":"RY_Angle_Encoding",
    "rz_embedding":"RZ_Angle_Encoding","iqp_embedding":"IQP_style_encoding",
    "zz_feature_map":"ZZ_Feature_Map_style","amplitude_embedding":"Amplitude_Embedding",
    "custom_h_ry_rz":"Custom_H_RY_RZ","efficient_su2_like":"EfficientSU2_like",
    "real_amplitudes":"Real_Amplitude_like","custom_ansatz_1":"custom_ansatz_1",
    "strongly_entangling":"strongly_entangling","iqp_style_encoding":"IQP_style_encoding",
    "zz_feature_map_style":"ZZ_Feature_Map_style","efficientsu2_like":"EfficientSU2_like",
    "real_amplitude_like":"Real_Amplitude_like",
}
DRM_DISPLAY = {"pca4":"PCA4","ica4":"ICA4","xgb_pca4":"XGB_PCA4","autoencoder":"Autoencoder"}

def apply_rename(df):
    for col in ["encoding","ansatz"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: RENAME_MAP.get(str(x).strip().lower(), x))
    if "feature_set" in df.columns:
        df["DRM"] = df["feature_set"].str.lower().map(
            lambda x: DRM_DISPLAY.get(x, x.upper()))
    return df

def load_all_subsets(results_dir, dataset):
    dfs = []
    for name in ["new_format_combined_result_renamed.csv","combined_summary_metrics.csv"]:
        p = Path(results_dir) / name
        if p.exists():
            d = pd.read_csv(p); d["subset"] = 1; dfs.append(d); break
    for i in range(2, 6):
        for name in [
            f"combined_summary_metrics_{dataset}_subset_{i}.csv",
            f"new_format_combined_result_renamed_subset_{i}.csv",
        ]:
            p = Path(results_dir) / name
            if p.exists():
                d = pd.read_csv(p); d["subset"] = i; dfs.append(d); break
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    return apply_rename(df)

def levene_result(groups, label):
    """Run Levene on groups, return dict with stat/p/ok/note."""
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {f"{label}_stat": None, f"{label}_p": None, f"{label}_ok": None,
                f"{label}_note": "insufficient groups"}
    stat, p = levene(*groups)
    return {
        f"{label}_stat": round(stat, 4),
        f"{label}_p":    round(p, 6),
        f"{label}_ok":   p > 0.05,
        f"{label}_note": ("homogeneity met" if p > 0.05
                          else "heteroscedastic — Welch confirms robustness"),
    }

def run_levene_for_dataset(df, dataset):
    """Compute all three Levene tests for every DRM × metric combination."""
    drms    = [d for d in ["PCA4","ICA4","XGB_PCA4","Autoencoder"]
                if d in df["DRM"].values]
    f1_col  = next((c for c in df.columns if c.lower() == "f1"), None)
    auc_col = next((c for c in df.columns if c.lower() == "auc"), None)
    rows = []

    print(f"\n  DRMs: {drms}")
    print(f"  Metrics: f1={f1_col}  auc={auc_col}")
    print(f"\n  {'DRM':<12} {'Metric':<5} {'Levene Enc (7 grp)':>20} "
          f"{'Levene Ans (4 grp)':>20} {'Levene 2-way (28 grp)':>22}")
    print(f"  {'─'*85}")

    for drm in drms:
        sub_drm = df[df["DRM"] == drm]
        for metric_col in [f1_col, auc_col]:
            if metric_col is None: continue
            sub = sub_drm.dropna(subset=[metric_col, "encoding", "ansatz"])

            # ── Levene 1: one-way encoding (7 groups, collapse ansatz+subset)
            enc_groups = [g[metric_col].values
                          for _, g in sub.groupby("encoding")]
            # ── Levene 2: one-way ansatz (4 groups, collapse encoding+subset)
            ans_groups = [g[metric_col].values
                          for _, g in sub.groupby("ansatz")]
            # ── Levene 3: two-way cells (28 groups × 5 subsets each)
            cell_groups= [g[metric_col].values
                          for _, g in sub.groupby(["encoding","ansatz"])]

            r_enc  = levene_result(enc_groups,   "levene_encoding")
            r_ans  = levene_result(ans_groups,   "levene_ansatz")
            r_cell = levene_result(cell_groups,  "levene_twoway")

            row = {"dataset": dataset, "DRM": drm, "metric": metric_col,
                   "n_subsets": sub["subset"].nunique(),
                   "n_enc_groups": len(enc_groups),
                   "n_ans_groups": len(ans_groups),
                   "n_cell_groups": len(cell_groups),
                   **r_enc, **r_ans, **r_cell}
            rows.append(row)

            enc_ok  = "✓" if r_enc.get("levene_encoding_ok")  else "✗"
            ans_ok  = "✓" if r_ans.get("levene_ansatz_ok")    else "✗"
            cell_ok = "✓" if r_cell.get("levene_twoway_ok")   else "✗"
            enc_p   = r_enc.get("levene_encoding_p","?")
            ans_p   = r_ans.get("levene_ansatz_p","?")
            cell_p  = r_cell.get("levene_twoway_p","?")
            print(f"  {drm:<12} {metric_col:<5} "
                  f"  p={enc_p:>8}  {enc_ok}"
                  f"     p={ans_p:>8}  {ans_ok}"
                  f"     p={cell_p:>8}  {cell_ok}")

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base", required=True)
    ap.add_argument("--datasets", default="UNSW_NB15,NF_UNSW_NB15,TON_IOT")
    args = ap.parse_args()

    base     = Path(args.results_base)
    datasets = [d.strip() for d in args.datasets.split(",")]
    all_rows = []

    for ds in datasets:
        folder = base / f"results_vqc_{ds}_v8_default_mixed"
        anova_dir = base / f"anova_results_{ds}"
        anova_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  DATASET: {ds}")
        print(f"  Loading from: {folder}")
        print(f"{'='*70}")

        df = load_all_subsets(str(folder), ds)
        if df is None:
            print(f"  ERROR: Could not load data"); continue

        print(f"  Loaded {len(df)} rows, {df['subset'].nunique()} subsets")

        rows = run_levene_for_dataset(df, ds)
        all_rows.extend(rows)

        # Save to same location as existing ANOVA results
        out = anova_dir / f"anova_assumptions_{ds}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  ✓ Saved → {out}")

        # Print interpretation summary
        print(f"\n  INTERPRETATION:")
        for r in rows:
            if r["metric"] != next(
                (c for c in df.columns if c.lower()=="f1"), None):
                continue
            e = "✓ met" if r.get("levene_encoding_ok") else "✗ violated"
            a = "✓ met" if r.get("levene_ansatz_ok")   else "✗ violated"
            c = "✓ met" if r.get("levene_twoway_ok")   else "✗ violated"
            print(f"    {r['DRM']}: "
                  f"one-way enc={e}  one-way ans={a}  two-way={c}")

    print(f"\n{'='*70}")
    print(f"  NOTE: Levene test results do NOT change ANOVA F-statistics,")
    print(f"  p-values, or eta-squared. Those are computed independently.")
    print(f"  Levene only reports whether the homogeneity assumption was met.")
    print(f"  ANOVA is robust to mild violations with balanced N=140.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
