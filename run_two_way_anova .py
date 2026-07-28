"""
run_two_way_anova.py
Two-way ANOVA (Encoding × Ansatz) per DRM, plus one-way ANOVA for DRM.
Uses F1 values from 5 training subsets (original + 4 new).
Includes Shapiro-Wilk (normality), Levene (homogeneity), Holm correction.

Run:
  python run_two_way_anova.py \
      --results_dir /home/nvidia/.../results_vqc_UNSW_NB15_v8_default_mixed \
      --dataset UNSW_NB15 \
      --out_dir /home/nvidia/21PHD1192/qml_id2/anova_results_UNSW_NB15
"""
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene, f_oneway
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    from statsmodels.sandbox.stats.multicomp import multipletests
from itertools import combinations

warnings.filterwarnings("ignore")

# Comprehensive map — all keys LOWERCASE
# Handles both internal names (subsets 2-5) AND
# partially-renamed names (subset 1 new_format_combined_result_renamed.csv)
RENAME_MAP = {
    # ── Encodings: internal names (subsets 2-5) ──────────────────────────
    "rx_embedding":           "RX_Angle_Encoding",
    "ry_embedding":           "RY_Angle_Encoding",
    "rz_embedding":           "RZ_Angle_Encoding",
    "iqp_embedding":          "IQP_style_encoding",
    "zz_feature_map":         "ZZ_Feature_Map_style",
    "amplitude_embedding":    "Amplitude_Embedding",
    "custom_h_ry_rz":         "Custom_H_RY_RZ",
    # ── Encodings: partially-renamed names (subset 1) ────────────────────
    "iqp_style_encoding":     "IQP_style_encoding",      # already renamed
    "zz_feature_map_style":   "ZZ_Feature_Map_style",    # already renamed
    "amplitude_embedding":    "Amplitude_Embedding",     # same as internal
    # ── Ansatzes: internal names (subsets 2-5) ───────────────────────────
    "efficient_su2_like":     "EfficientSU2_like",
    "real_amplitudes":        "Real_Amplitude_like",
    "custom_ansatz_1":        "custom_ansatz_1",
    "strongly_entangling":    "strongly_entangling",
    # ── Ansatzes: partially-renamed names (subset 1) ─────────────────────
    "efficientsu2_like":      "EfficientSU2_like",       # already renamed
    "real_amplitude_like":    "Real_Amplitude_like",     # already renamed
}
DRM_DISPLAY = {"pca4":"PCA4","ica4":"ICA4",
               "xgb_pca4":"XGB_PCA4","autoencoder":"Autoencoder"}

def apply_rename(df):
    """Normalise encoding and ansatz column values to consistent display names.

    Handles two input formats:
      - Internal names (subsets 2-5): "iqp_embedding" → lookup in RENAME_MAP
      - Already-renamed names (subset 1): "IQP_style_encoding" → kept as-is
    All values are lowercased for consistent grouping in ANOVA.
    """
    # Reverse map: display name (lower) → canonical display name
    # so already-renamed values from subset 1 are also normalised
    display_lower_to_canonical = {v.lower(): v for v in RENAME_MAP.values()}

    for col in ["encoding", "ansatz"]:
        if col not in df.columns:
            continue
        def _normalise(x):
            s = str(x).strip()
            sl = s.lower()
            # Try internal name lookup first
            if sl in RENAME_MAP:
                return RENAME_MAP[sl]
            # Try already-renamed (display) name lookup
            if sl in display_lower_to_canonical:
                return display_lower_to_canonical[sl]
            # Return as-is (unknown value)
            return s
        df[col] = df[col].apply(_normalise)

    if "feature_set" in df.columns:
        df["DRM"] = df["feature_set"].str.lower().map(
            lambda x: DRM_DISPLAY.get(x, x.upper()))
    return df

def load_all_subsets(results_dir, dataset=None):
    """Load F1/AUC from subset 1 (original) + subsets 2-5.
    Accepts both new_format_combined_result_renamed*.csv
    and combined_summary_metrics*.csv (raw output from v8).
    """
    dfs = []
    rdir = Path(results_dir)

    # ── Subset 1: original result ─────────────────────────────────────────
    for name in ["new_format_combined_result_renamed.csv",
                 "combined_summary_metrics.csv"]:
        s1 = rdir / name
        if s1.exists():
            d = pd.read_csv(s1)
            d["subset"] = 1
            dfs.append(d)
            print(f"  Subset 1 ({name}): {len(d)} models")
            break

    # ── Subsets 2-5: look for both naming conventions ─────────────────────
    for i in range(2, 6):
        found = False
        candidates = [
            # From generate_new_format_combined (with rename)
            rdir / f"new_format_combined_result_renamed_subset_{i}.csv",
            # From direct copy of combined_summary_metrics (with dataset tag)
            rdir / f"combined_summary_metrics_{dataset}_subset_{i}.csv"
            if dataset else None,
            # Fallback: plain subset name
            rdir / f"combined_summary_metrics_subset_{i}.csv",
        ]
        for p in candidates:
            if p and p.exists():
                d = pd.read_csv(p)
                d["subset"] = i
                dfs.append(d)
                print(f"  Subset {i} ({p.name}): {len(d)} models")
                found = True
                break
        if not found:
            print(f"  Subset {i}: NOT FOUND (subsets 2-5 needed for two-way ANOVA)")

    if not dfs:
        raise FileNotFoundError(f"No combined result files found in {results_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df = apply_rename(df)
    f1_col  = next((c for c in df.columns if c.lower() == "f1"), None)
    auc_col = next((c for c in df.columns if c.lower() == "auc"), None)
    print(f"  Total rows: {len(df)}  f1_col={f1_col}  subsets={sorted(df['subset'].unique())}")
    return df, f1_col, auc_col

def eta_sq(ss_factor, ss_total):
    return ss_factor / ss_total if ss_total > 0 else 0.0

def two_way_anova(df, drm, metric_col):
    """Two-way ANOVA: metric ~ Encoding + Ansatz + Encoding:Ansatz for one DRM.
    Returns None if insufficient replications (< 2 subsets) for a valid error term."""
    sub = df[df["DRM"] == drm].copy()
    sub = sub.dropna(subset=[metric_col, "encoding", "ansatz"])

    n_subsets   = sub["subset"].nunique()
    encodings   = sorted(sub["encoding"].unique())
    ansatzes    = sorted(sub["ansatz"].unique())
    K           = len(encodings)
    L           = len(ansatzes)
    N           = len(sub)
    grand_mean  = sub[metric_col].mean()
    ss_total    = ((sub[metric_col] - grand_mean) ** 2).sum()

    ss_b  = sum(
        len(g) * (g.mean() - grand_mean)**2
        for _, g in sub.groupby("encoding")[metric_col] if len(g) > 0
    )
    ss_a  = sum(
        len(g) * (g.mean() - grand_mean)**2
        for _, g in sub.groupby("ansatz")[metric_col] if len(g) > 0
    )
    cell_means = sub.groupby(["encoding","ansatz"])[metric_col].mean()
    cell_ns    = sub.groupby(["encoding","ansatz"])[metric_col].count()
    ss_cells   = sum(
        cell_ns.get((e,a), 0) * (cell_means.get((e,a), grand_mean) - grand_mean)**2
        for e in encodings for a in ansatzes
    )
    ss_inter = max(ss_cells - ss_b - ss_a, 0)

    df_b     = K - 1
    df_a     = L - 1
    df_inter = df_b * df_a
    df_err   = max(N - K * L, 1)
    ss_err   = max(ss_total - ss_b - ss_a - ss_inter, 0)
    ms_err   = ss_err / df_err

    def safe_F(ss_fac, df_fac):
        ms_fac = ss_fac / max(df_fac, 1)
        if ms_err < 1e-12:           # degenerate — only 1 replication per cell
            return float("nan"), float("nan")
        F = ms_fac / ms_err
        p = 1 - stats.f.cdf(F, df_fac, df_err)
        return round(F, 4), round(p, 6)

    F_enc,   p_enc   = safe_F(ss_b,     df_b)
    F_ans,   p_ans   = safe_F(ss_a,     df_a)
    F_inter, p_inter = safe_F(ss_inter, df_inter)

    degen = ms_err < 1e-12

    return {
        "DRM":              drm,
        "metric":           metric_col,
        "n_obs":            N,
        "n_encodings":      K,
        "n_ansatzes":       L,
        "n_subsets":        n_subsets,
        "degenerate":       degen,
        "note":             ("Single subset — error term=0; use one-way ANOVA only"
                             if degen else ""),
        "F_encoding":       F_enc,   "df_enc":  df_b,
        "p_encoding":       p_enc,   "eta2_encoding":    round(eta_sq(ss_b,     ss_total), 4),
        "F_ansatz":         F_ans,   "df_ans":  df_a,
        "p_ansatz":         p_ans,   "eta2_ansatz":      round(eta_sq(ss_a,     ss_total), 4),
        "F_interaction":    F_inter, "df_inter":df_inter,
        "p_interaction":    p_inter, "eta2_interaction": round(eta_sq(ss_inter, ss_total), 4),
        "eta2_residual":    round(eta_sq(ss_err, ss_total), 4),
    }

    encodings = sorted(sub["encoding"].unique())
    ansatzes  = sorted(sub["ansatz"].unique())

    # Build groups for interaction (encoding × ansatz)
    grand_mean  = sub[metric_col].mean()
    ss_total    = ((sub[metric_col] - grand_mean) ** 2).sum()
    n_total     = len(sub)

    # Encoding main effect (collapse across ansatz)
    enc_means = sub.groupby("encoding")[metric_col].mean()
    enc_ns    = sub.groupby("encoding")[metric_col].count()
    ss_enc    = sum(enc_ns[e] * (enc_means[e] - grand_mean)**2 for e in encodings)
    df_enc    = len(encodings) - 1

    # Ansatz main effect
    ans_means = sub.groupby("ansatz")[metric_col].mean()
    ans_ns    = sub.groupby("ansatz")[metric_col].count()
    ss_ans    = sum(ans_ns[a] * (ans_means[a] - grand_mean)**2 for a in ansatzes)
    df_ans    = len(ansatzes) - 1

    # Interaction: cell means
    cell_means = sub.groupby(["encoding","ansatz"])[metric_col].mean()
    cell_ns    = sub.groupby(["encoding","ansatz"])[metric_col].count()
    ss_cells   = sum(cell_ns.get((e,a), 0) * (cell_means.get((e,a), grand_mean) - grand_mean)**2
                     for e in encodings for a in ansatzes)
    ss_inter   = ss_cells - ss_enc - ss_ans
    df_inter   = df_enc * df_ans

    # Error (within cells)
    ss_err = ss_total - ss_enc - ss_ans - ss_inter
    df_err = n_total - len(encodings) * len(ansatzes)
    df_err = max(df_err, 1)

    ms_enc   = ss_enc   / df_enc
    ms_ans   = ss_ans   / df_ans
    ms_inter = ss_inter / max(df_inter, 1)
    ms_err   = ss_err   / df_err

    F_enc    = ms_enc   / ms_err
    F_ans    = ms_ans   / ms_err
    F_inter  = ms_inter / ms_err

    p_enc    = 1 - stats.f.cdf(F_enc,   df_enc,   df_err)
    p_ans    = 1 - stats.f.cdf(F_ans,   df_ans,   df_err)
    p_inter  = 1 - stats.f.cdf(F_inter, df_inter, df_err)

    return {
        "DRM":           drm,
        "metric":        metric_col,
        "n_obs":         n_total,
        "n_encodings":   len(encodings),
        "n_ansatzes":    len(ansatzes),
        "n_subsets":     sub["subset"].nunique(),
        # Encoding factor
        "F_encoding":    round(F_enc,   4),
        "df_enc":        df_enc,
        "p_encoding":    round(p_enc,   6),
        "eta2_encoding": round(eta_sq(ss_enc, ss_total), 4),
        # Ansatz factor
        "F_ansatz":      round(F_ans,   4),
        "df_ans":        df_ans,
        "p_ansatz":      round(p_ans,   6),
        "eta2_ansatz":   round(eta_sq(ss_ans, ss_total), 4),
        # Interaction
        "F_interaction": round(F_inter, 4),
        "df_inter":      df_inter,
        "p_interaction": round(p_inter, 6),
        "eta2_interaction": round(eta_sq(ss_inter, ss_total), 4),
        # Residual
        "eta2_residual": round(eta_sq(ss_err, ss_total), 4),
    }

def anova_assumptions(df, drm, metric_col):
    """Test normality (Shapiro-Wilk) and homogeneity (Levene) within each DRM."""
    sub    = df[df["DRM"] == drm].dropna(subset=[metric_col])
    groups = [g[metric_col].values
              for _, g in sub.groupby(["encoding","ansatz"])
              if len(g) >= 3]
    results = {"DRM": drm, "metric": metric_col}

    # Levene test (homogeneity of variance)
    if len(groups) >= 2:
        lev_stat, lev_p = levene(*groups)
        results["levene_stat"] = round(lev_stat, 4)
        results["levene_p"]    = round(lev_p,    6)
        results["homogeneity_ok"] = lev_p > 0.05
    # Shapiro-Wilk on residuals
    all_vals   = sub[metric_col].values
    grand_mean = all_vals.mean()
    residuals  = all_vals - sub.groupby(["encoding","ansatz"])[metric_col].transform("mean").values
    if len(residuals) >= 3:
        sw_stat, sw_p = shapiro(residuals)
        results["shapiro_stat"] = round(sw_stat, 4)
        results["shapiro_p"]    = round(sw_p,    6)
        results["normality_ok"] = sw_p > 0.05
    return results

def one_way_anova_factor(df, drm, metric_col, factor):
    """One-way ANOVA: metric ~ factor (either 'encoding' or 'ansatz')."""
    sub    = df[df["DRM"] == drm].dropna(subset=[metric_col, factor])
    levels = sorted(sub[factor].unique())
    K      = len(levels)
    N      = len(sub)

    groups      = [sub[sub[factor] == lvl][metric_col].values for lvl in levels]
    grand_mean  = sub[metric_col].mean()
    ss_total    = ((sub[metric_col] - grand_mean) ** 2).sum()

    ss_b = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups if len(g) > 0)
    ss_w = sum(((g - g.mean())**2).sum()           for g in groups if len(g) > 0)
    df_b = K - 1
    df_w = N - K

    ms_b = ss_b / df_b if df_b > 0 else 0
    ms_w = ss_w / df_w if df_w > 0 else 1
    F    = ms_b / ms_w if ms_w > 0 else 0
    p    = 1 - stats.f.cdf(F, df_b, df_w) if df_b > 0 else 1.0
    eta2 = ss_b / ss_total if ss_total > 0 else 0

    return {
        "DRM":         drm,
        "factor":      factor,
        "metric":      metric_col,
        "n_levels":    K,
        "n_obs":       N,
        "n_subsets":   sub["subset"].nunique(),
        "SS_between":  round(ss_b, 6),
        "SS_within":   round(ss_w, 6),
        "SS_total":    round(ss_total, 6),
        "df_between":  df_b,
        "df_within":   df_w,
        "MS_between":  round(ms_b, 6),
        "MS_within":   round(ms_w, 6),
        "F_stat":      round(F, 4),
        "p_value":     round(p, 6),
        "eta2":        round(eta2, 4),
        "effect_size": ("large" if eta2 >= 0.14 else
                        "medium" if eta2 >= 0.06 else
                        "small"  if eta2 >= 0.01 else "negligible"),
        "significant": p < 0.05,
    }
    """Pairwise Welch t-tests across encodings (collapse ansatz) with Holm correction."""
    sub      = df[df["DRM"] == drm].dropna(subset=[metric_col])
    encodings = sorted(sub["encoding"].unique())
    pairs     = list(combinations(encodings, 2))
    raw_p    = []
    t_stats  = []
    for e1, e2 in pairs:
        g1 = sub[sub["encoding"]==e1][metric_col].values
        g2 = sub[sub["encoding"]==e2][metric_col].values
        if len(g1) < 2 or len(g2) < 2:
            raw_p.append(1.0); t_stats.append(0.0)
            continue
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        raw_p.append(p); t_stats.append(t)
    # Holm correction
    if raw_p:
        rej, corr_p, _, _ = multipletests(raw_p, method="holm")
    else:
        rej = []; corr_p = []
    rows = []
    for (e1, e2), t, rp, cp, r in zip(pairs, t_stats, raw_p, corr_p, rej):
        rows.append({
            "DRM":       drm,
            "enc_1":     e1,
            "enc_2":     e2,
            "t_stat":    round(t,  4),
            "p_raw":     round(rp, 6),
            "p_holm":    round(cp, 6),
            "significant_holm": r,
        })
    return rows

def welch_pairwise_encoding(df, drm, metric_col):
    """Pairwise Welch t-tests across encodings (collapse ansatz) with Holm correction."""
    sub       = df[df["DRM"] == drm].dropna(subset=[metric_col])
    encodings = sorted(sub["encoding"].unique())
    pairs     = list(combinations(encodings, 2))
    raw_p     = []; t_stats = []
    for e1, e2 in pairs:
        g1 = sub[sub["encoding"] == e1][metric_col].values
        g2 = sub[sub["encoding"] == e2][metric_col].values
        if len(g1) < 2 or len(g2) < 2:
            raw_p.append(1.0); t_stats.append(0.0); continue
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        raw_p.append(p); t_stats.append(t)
    # Holm correction
    if raw_p:
        rej, corr_p, _, _ = multipletests(raw_p, method="holm")
    else:
        rej = []; corr_p = []
    rows = []
    for (e1, e2), t, rp, cp, r in zip(pairs, t_stats, raw_p, corr_p, rej):
        rows.append({
            "DRM":              drm,
            "enc_1":            e1,
            "enc_2":            e2,
            "t_stat":           round(t,  4),
            "p_raw":            round(rp, 6),
            "p_holm":           round(cp, 6),
            "significant_holm": bool(r),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dataset",     required=True)
    ap.add_argument("--out_dir",     default=".")
    ap.add_argument("--metric",      default="f1")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  TWO-WAY ANOVA — {args.dataset}")
    print(f"{'='*65}")

    df, f1_col, auc_col = load_all_subsets(args.results_dir, dataset=args.dataset)
    n_subsets = df["subset"].nunique()
    print(f"  Loaded {n_subsets} subsets, {len(df)} total model evaluations")

    drms = [d for d in ["PCA4","ICA4","XGB_PCA4","Autoencoder"]
             if d in df["DRM"].values]
    print(f"  DRMs: {drms}")

    anova_rows = []; assump_rows = []; welch_rows = []; oneway_rows = []

    for drm in drms:
        for metric_col in [f1_col, auc_col]:
            if metric_col is None: continue
            print(f"\n  [{drm}] Two-way ANOVA on {metric_col}")
            row  = two_way_anova(df, drm, metric_col)
            asmp = anova_assumptions(df, drm, metric_col)
            anova_rows.append(row)
            assump_rows.append(asmp)
            print(f"    Encoding: F={row['F_encoding']}  "
                  f"p={row['p_encoding']}  η²={row['eta2_encoding']}")
            print(f"    Ansatz:   F={row['F_ansatz']}  "
                  f"p={row['p_ansatz']}  η²={row['eta2_ansatz']}")
            print(f"    Interact: F={row['F_interaction']}  "
                  f"p={row['p_interaction']}  η²={row['eta2_interaction']}")
            print(f"    Normality (Shapiro): p={asmp.get('shapiro_p','N/A')}  "
                  f"ok={asmp.get('normality_ok','?')}")
            print(f"    Homogeneity (Levene): p={asmp.get('levene_p','N/A')}  "
                  f"ok={asmp.get('homogeneity_ok','?')}")

            # ── One-way ANOVA for encoding and ansatz separately ──────────
            for factor in ["encoding", "ansatz"]:
                ow = one_way_anova_factor(df, drm, metric_col, factor)
                oneway_rows.append(ow)
                print(f"    One-way [{factor}]: F={ow['F_stat']}  "
                      f"p={ow['p_value']}  η²={ow['eta2']}  "
                      f"({ow['effect_size']})")

        # Welch pairwise (F1 only)
        print(f"\n  [{drm}] Welch pairwise encoding comparison (Holm corrected)")
        rows = welch_pairwise_encoding(df, drm, f1_col)
        welch_rows.extend(rows)
        sig  = [r for r in rows if r["significant_holm"]]
        print(f"    {len(sig)}/{len(rows)} pairs significant after Holm correction")

    # Save tables
    pd.DataFrame(anova_rows).to_csv(
        out / f"two_way_anova_{args.dataset}.csv", index=False)
    pd.DataFrame(oneway_rows).to_csv(
        out / f"one_way_anova_{args.dataset}.csv", index=False)
    pd.DataFrame(assump_rows).to_csv(
        out / f"anova_assumptions_{args.dataset}.csv", index=False)
    pd.DataFrame(welch_rows).to_csv(
        out / f"welch_pairwise_encoding_holm_{args.dataset}.csv", index=False)

    # Print two-way ANOVA summary
    print(f"\n{'═'*65}")
    print(f"  TWO-WAY ANOVA SUMMARY — {args.dataset}")
    print(f"{'═'*65}")
    df_anova = pd.DataFrame(anova_rows)
    f1_rows  = df_anova[df_anova["metric"] == f1_col]
    print(f"\n  {'DRM':<12} {'F_enc':>8} {'p_enc':>10} {'η²_enc':>8} "
          f"{'F_ans':>8} {'p_ans':>10} {'η²_ans':>8} "
          f"{'F_inter':>8} {'η²_inter':>8}")
    print(f"  {'─'*100}")
    for _, r in f1_rows.iterrows():
        print(f"  {r['DRM']:<12} {r['F_encoding']:>8.4f} {r['p_encoding']:>10.6f} "
              f"{r['eta2_encoding']:>8.4f} "
              f"{r['F_ansatz']:>8.4f} {r['p_ansatz']:>10.6f} {r['eta2_ansatz']:>8.4f} "
              f"{r['F_interaction']:>8.4f} {r['eta2_interaction']:>8.4f}")

    # Print one-way ANOVA summary
    print(f"\n{'═'*65}")
    print(f"  ONE-WAY ANOVA SUMMARY — {args.dataset}")
    print(f"{'═'*65}")
    df_ow = pd.DataFrame(oneway_rows)
    f1_ow = df_ow[df_ow["metric"] == f1_col]
    print(f"\n  {'DRM':<12} {'Factor':<10} {'K':>4} {'N':>5} "
          f"{'F_stat':>8} {'p_value':>10} {'η²':>8} {'Effect':>12} {'Sig':>5}")
    print(f"  {'─'*80}")
    for _, r in f1_ow.iterrows():
        print(f"  {r['DRM']:<12} {r['factor']:<10} {r['n_levels']:>4} "
              f"{r['n_obs']:>5} {r['F_stat']:>8.4f} {r['p_value']:>10.6f} "
              f"{r['eta2']:>8.4f} {r['effect_size']:>12} "
              f"{'✓' if r['significant'] else '✗':>5}")

    print(f"\n  ✓ Results saved to {out}")

if __name__ == "__main__":
    main()