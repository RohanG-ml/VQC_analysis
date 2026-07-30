"""
build_jsc_score_table.py
Builds the Joint Selection Criterion (JSC) score table from the
reliability appendix CSV (produced by build_reliability_table.py).

Reads:  reliability_appendix_{dataset}.csv   (already exists)
Writes: jsc_score_table_{dataset}.csv        (new file)

Does NOT modify any existing code or files.

JSC scores each model on three axes:
  Axis 1 — Rank_F1:  rank by f1_pm  DESCENDING  (rank 1 = highest f1_pm)
  Axis 2 — Rank_C:   rank by C*     DESCENDING  (rank 1 = highest C*)
  Axis 3 — Rank_S:   rank by σ_epi  ASCENDING   (rank 1 = lowest σ_epi)

All ranks use DENSE ranking:
  Multiple models with identical values share the SAME rank.
  E.g. five models with C*=1.0 all get Rank_C=1.

JSC_score = Rank_F1 + Rank_C + Rank_S
  Lower score = better model overall.

Tiebreaker for equal JSC_score:
  Use Perf.Rank from the performance appendix (lower rank = better F1
  on the original test split).

Final output includes the JSC_Rank column (1 = best joint selection).

Run:
  python build_jsc_score_table.py \
      --appendix /home/nvidia/21PHD1192/qml_id2/reliability_appendix_TON_IOT.csv \
      --dataset_name TON_IOT \
      --out_dir /home/nvidia/21PHD1192/qml_id2 \
      --top_n 5
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def extract_mean(val):
    """Extract numeric mean from '0.9881 ± 0.0017' format."""
    if isinstance(val, str) and "±" in val:
        try: return float(val.split("±")[0].strip())
        except: return np.nan
    try: return float(val)
    except: return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appendix",    required=True,
                     help="Path to reliability_appendix_{dataset}.csv")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--out_dir",      default="/home/nvidia/21PHD1192/qml_id2")
    ap.add_argument("--top_n",        type=int, default=5,
                     help="How many top JSC models to highlight in the summary")
    ap.add_argument("--f1_floor",     type=float, default=0.50,
                     help="Exclude models with f1_pm below this value before scoring "
                          "(catches bootstrap-collapsed models, default=0.50)")
    ap.add_argument("--sigma_bin_1",  type=float, default=0.01,
                     help="σ_epi below this value → Rank_S=1 (excellent stability). "
                          "Default=0.01")
    ap.add_argument("--sigma_bin_2",  type=float, default=0.03,
                     help="σ_epi below this (but above bin_1) → Rank_S=2 (good). "
                          "σ_epi ≥ this value → Rank_S=3. Default=0.03")
    args = ap.parse_args()

    # ── Load appendix ─────────────────────────────────────────────────────
    df = pd.read_csv(args.appendix)
    print(f"Loaded {len(df)} models from {args.appendix}")

    # ── Identify column names (handle renamed versions) ───────────────────
    col_f1   = next((c for c in df.columns if "f1_pm"         in c.lower()), None)
    col_c    = next((c for c in df.columns if "deploy"        in c.lower()), None)
    col_s    = next((c for c in df.columns if "sigma_epi"     in c.lower()), None)
    col_rank = next((c for c in df.columns if "perf" in c.lower()
                                           and "rank" in c.lower()), None)
    col_vqc  = next((c for c in df.columns if "drm" in c.lower()
                                           or "vqc" in c.lower()), None)

    missing = [n for n, c in [("f1_pm", col_f1), ("deploy_coverage", col_c),
                                ("sigma_epi_mean", col_s)] if c is None]
    if missing:
        print(f"✗ Missing required columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        return

    # ── Extract numeric values ────────────────────────────────────────────
    df["_f1_num"] = df[col_f1].apply(extract_mean)
    df["_c_num"]  = pd.to_numeric(df[col_c], errors="coerce")
    df["_s_num"]  = pd.to_numeric(df[col_s], errors="coerce")

    # ── Apply f1 floor to exclude collapsed models ────────────────────────
    n_before = len(df)
    df_valid = df[df["_f1_num"] >= args.f1_floor].copy()
    n_excluded = n_before - len(df_valid)
    if n_excluded > 0:
        print(f"\nExcluded {n_excluded} model(s) with f1_pm < {args.f1_floor} "
              f"(bootstrap-collapsed). {len(df_valid)} models enter JSC scoring.")
    else:
        print(f"\nAll {len(df_valid)} models pass f1_pm ≥ {args.f1_floor} floor.")

    if df_valid.empty:
        print("No models remaining after floor. Lower --f1_floor or check data.")
        return

    # ── Dense ranking for f1_pm and C* (continuous, meaningful differences) ──
    df_valid["Rank_F1"]  = df_valid["_f1_num"].rank(
                               method="dense", ascending=False).astype(int)
    df_valid["Rank_C"]   = df_valid["_c_num"].rank(
                               method="dense", ascending=False).astype(int)

    # ── THRESHOLD BINS for σ_epi (NOT dense rank) ─────────────────────────
    # Rationale: all σ_epi values are typically in the range [0.001, 0.05],
    # meaning ALL models are "stable" by any practical standard. Dense
    # ranking within this narrow range amplifies meaningless micro-differences
    # (e.g. 0.023 vs 0.025) into large rank penalties that swamp f1_pm and C*.
    # Instead, threshold bins reflect genuine stability zones:
    #   Rank_S=1: σ_epi < 0.01   — essentially zero parameter noise effect
    #   Rank_S=2: σ_epi < 0.03   — very small, operationally negligible variation
    #   Rank_S=3: σ_epi ≥ 0.03   — small but measurable variation
    # Models within the same bin are treated as equally stable.
    s_bins = [args.sigma_bin_1, args.sigma_bin_2]
    def sigma_rank(v):
        if pd.isna(v): return 3
        if v < s_bins[0]: return 1
        if v < s_bins[1]: return 2
        return 3
    df_valid["Rank_S"] = df_valid["_s_num"].apply(sigma_rank)

    # Print bin summary so reader can verify
    for r, (lo, hi) in enumerate([(0, s_bins[0]),
                                   (s_bins[0], s_bins[1]),
                                   (s_bins[1], 999)], start=1):
        n = df_valid["Rank_S"].eq(r).sum()
        print(f"  σ_epi Rank_S={r}: {n} models  "
              f"(range: [{lo:.3f}, {hi:.3f}))")

    # ── JSC score = sum of three axis ranks ───────────────────────────────
    df_valid["JSC_score"] = (df_valid["Rank_F1"] +
                              df_valid["Rank_C"]  +
                              df_valid["Rank_S"])

    # ── Tiebreaker: Perf.Rank (lower = better original F1) ───────────────
    if col_rank:
        df_valid["_perf_rank_num"] = pd.to_numeric(
            df_valid[col_rank], errors="coerce").fillna(999)
    else:
        df_valid["_perf_rank_num"] = range(len(df_valid))  # fallback

    # ── Final JSC ranking ─────────────────────────────────────────────────
    df_valid = df_valid.sort_values(
        ["JSC_score", "_perf_rank_num"],
        ascending=[True, True]
    ).reset_index(drop=True)
    df_valid["JSC_Rank"] = range(1, len(df_valid) + 1)

    # ── Build output table ────────────────────────────────────────────────
    out_cols = []
    if col_vqc:   out_cols.append(col_vqc)
    if col_rank:  out_cols.append(col_rank)
    out_cols += [col_f1, "Rank_F1", col_c, "Rank_C", col_s, "Rank_S",
                 "JSC_score", "JSC_Rank"]
    out_cols = [c for c in out_cols if c in df_valid.columns]

    result = df_valid[out_cols].copy()

    # ── Print full score table ────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  JSC SCORE TABLE  —  {args.dataset_name}")
    print(f"  Rank_F1: rank by f1_pm DESC   "
          f"Rank_C: rank by C* DESC   "
          f"Rank_S: rank by σ_epi ASC")
    print(f"  JSC_score = Rank_F1 + Rank_C + Rank_S   "
          f"(lower = better)")
    print(f"  Tiebreaker: Perf.Rank (lower = better original F1)")
    print(f"{'='*100}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result.to_string(index=False))
    print(f"{'='*100}")

    # ── Print top-N selection ─────────────────────────────────────────────
    top = result.iloc[:args.top_n]
    winner = result.iloc[0]

    print(f"\n{'─'*100}")
    print(f"  TOP {args.top_n} MODELS BY JSC:")
    print(f"{'─'*100}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(top.to_string(index=False))

    print(f"\n{'='*100}")
    print(f"  JSC WINNER:  JSC_Rank=1")
    print(f"{'='*100}")
    if col_vqc:
        print(f"  Model:       {winner[col_vqc]}")
    if col_rank:
        print(f"  Perf.Rank:   {winner[col_rank]} of 112  "
              f"(rank by original F1 in performance appendix)")
    print(f"  f1_pm:       {winner[col_f1]}  →  Rank_F1 = {winner['Rank_F1']}")
    print(f"  C*:          {winner[col_c]}  →  Rank_C  = {winner['Rank_C']}")
    print(f"  σ_epi:       {winner[col_s]}  →  Rank_S  = {winner['Rank_S']}")
    print(f"  JSC_score:   {winner['JSC_score']}  "
          f"(= {winner['Rank_F1']} + {winner['Rank_C']} + {winner['Rank_S']})")
    print(f"{'='*100}")

    # ── Explain any ties ──────────────────────────────────────────────────
    # C* ties
    c_star_one = (df_valid["_c_num"] == 1.0).sum()
    if c_star_one > 1:
        print(f"\n  NOTE: {c_star_one} models share C*=1.0 → all receive Rank_C=1.")
        print(f"  Their JSC_score difference comes from Rank_F1 and Rank_S alone.")
        print(f"  Among these, Perf.Rank breaks any remaining JSC_score ties.")

    # JSC score ties
    ties = df_valid[df_valid["JSC_score"] == winner["JSC_score"]]
    if len(ties) > 1:
        print(f"\n  NOTE: {len(ties)} models share JSC_score = {winner['JSC_score']}.")
        print(f"  Tiebreaker (Perf.Rank) selected JSC_Rank=1.")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"jsc_score_table_{args.dataset_name}.csv"
    result.to_csv(out_path, index=False)
    print(f"\n✓ Saved -> {out_path}")

    # ── Write-to-paper instruction ────────────────────────────────────────
    print(f"\n  FOR YOUR PAPER: Show jsc_score_table_{args.dataset_name}.csv "
          f"(top {args.top_n} rows) in the result section. The JSC_Rank=1 "
          f"model is your selected best model for Section 3.6 (unknown attack).")


if __name__ == "__main__":
    main()