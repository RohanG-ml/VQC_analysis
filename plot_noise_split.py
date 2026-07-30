"""
plot_noise_split.py
Saves F1 and AUC noise curves as SEPARATE images, one row each.
Output:
  {out_dir}/noise_plot/Noise_Study_{dataset}_F1.png
  {out_dir}/noise_plot/Noise_Study_{dataset}_AUC.png

Large fonts throughout so labels are readable inside LaTeX sidewaysfigure.

Run:
  python plot_noise_split.py \
      --summary_csv /home/nvidia/.../results_noise_UNSW_NB15 \
      --dataset UNSW_NB15 \
      --jsc_table /home/nvidia/.../jsc_score_table_UNSW_NB15.csv \
      --results_root /home/nvidia/.../results_vqc_UNSW_NB15_v8_default_mixed \
      --out_dir /home/nvidia/21PHD1192/qml_id2 \
      --top_n 5
"""
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
from pathlib import Path

NOISE_STYLE = {
    "amplitude_damping": dict(color="#2196F3", marker="o",
                               label="Amplitude Damping (T\u2081)"),
    "phase_damping":     dict(color="#4CAF50", marker="s",
                               label="Phase Damping (T\u2082)"),
    "depolarizing":      dict(color="#F44336", marker="^",
                               label="Depolarizing"),
}

# ── font sizes (large enough for LaTeX sidewaysfigure) ────────────────────
FS_TITLE   = 18   # subplot title (M1, M2 ...)
FS_AXIS    = 15   # axis labels
FS_TICK    = 13   # tick labels
FS_LEGEND  = 14   # legend text
FS_BOX     = 13   # model name box text
FS_ANNOT   = 11   # noiseless value annotation


def load_summary(summary_path):
    """Find and merge all noise_study_summary*.csv files recursively."""
    p = Path(summary_path)
    if p.is_dir():
        candidates = sorted(p.rglob("noise_study_summary*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No noise_study_summary*.csv under {p}")
        print(f"  Found {len(candidates)} CSV file(s) — merging:")
        for c in candidates:
            print(f"    {c}")
        df = pd.concat([pd.read_csv(c) for c in candidates],
                        ignore_index=True).drop_duplicates()
    else:
        df = pd.read_csv(p)
    print(f"  Loaded {len(df)} rows total.")
    return df


def build_noiseless_map(df, vqc_col, nl_col, f1_col, auc_col,
                         results_root):
    """
    Priority 1: combined_summary_metrics*.csv from results_root
    Priority 2: noise_level == 0.0 rows in the summary CSV
    """
    RENAME = {
        "rx_embedding":       "RX_Angle_Encoding",
        "ry_embedding":       "RY_Angle_Encoding",
        "rz_embedding":       "RZ_Angle_Encoding",
        "iqp_embedding":      "IQP_style_encoding",
        "zz_feature_map":     "ZZ_Feature_Map_style",
        "amplitude_embedding":"Amplitude_Embedding",
        "custom_h_ry_rz":     "Custom_H_RY_RZ",
        "efficient_su2_like": "EfficientSU2_like",
        "real_amplitudes":    "Real_Amplitude_like",
    }
    DRM = {"pca4":"PCA4","ica4":"ICA4",
           "xgb_pca4":"XGB_PCA4","autoencoder":"Autoencoder"}

    noiseless_map = {}
    if results_root:
        rr = Path(results_root)
        # Priority 1: exact filename (avoids picking up ANOVA subset files)
        exact = rr / "combined_summary_metrics.csv"
        if exact.exists():
            candidates = [exact]
        else:
            # Exclude any file with "subset" in the name
            candidates = [p for p in rr.rglob("combined_summary_metrics*.csv")
                          if "subset" not in p.name.lower()]
        if candidates:
            mf = candidates[0]
            mdf = pd.read_csv(mf)
            fsc  = next((c for c in mdf.columns if "feature_set" in c.lower()), None)
            encc = next((c for c in mdf.columns if c.lower()=="encoding"), None)
            ansc = next((c for c in mdf.columns if c.lower()=="ansatz"), None)
            f1c  = next((c for c in mdf.columns if c.lower()=="f1"), None)
            aucc = next((c for c in mdf.columns if c.lower()=="auc"), None)
            if all([fsc, encc, ansc, f1c, aucc]):
                for _, r in mdf.iterrows():
                    drm = DRM.get(str(r[fsc]).lower(), str(r[fsc]).upper())
                    enc = RENAME.get(str(r[encc]).lower(), str(r[encc]))
                    ans = RENAME.get(str(r[ansc]).lower(), str(r[ansc]))
                    key = f"{drm}+VQC({enc}+{ans})"
                    noiseless_map[key] = (float(r[f1c]), float(r[aucc]))
                print(f"  [Noiseless] Source file: {mf.name}")
                print(f"  [Noiseless] Full path:   {mf}")
                print(f"  [Noiseless] Models loaded: {len(noiseless_map)}")

    if not noiseless_map:
        nl0 = (df[df[nl_col] == 0.0]
               .groupby(vqc_col)[[f1_col, auc_col]]
               .mean().reset_index())
        noiseless_map = {r[vqc_col]: (r[f1_col], r[auc_col])
                          for _, r in nl0.iterrows()}
        print(f"  [Noiseless] Fallback: noise_level=0.0 rows "
              f"({len(noiseless_map)} models)")
    return noiseless_map


def draw_row(models_ordered, df_noise, noiseless_map,
             vqc_col, nm_col, nl_col, metric_col,
             metric_label, noise_levels_sorted,
             noise_types_present, figsize):
    """Draw one row (one metric) across all models. Returns fig, axes."""
    n = len(models_ordered)
    fig, axes = plt.subplots(1, n, figsize=figsize,
                              squeeze=False)
    fig.patch.set_facecolor("#FAFAFA")

    norm = lambda s: str(s).strip().lower().replace(" ", "")

    for col, model in enumerate(models_ordered):
        ax = axes[0][col]

        # noiseless baseline
        bl_pair = noiseless_map.get(model)
        if bl_pair is None:
            for k, v in noiseless_map.items():
                if norm(k) == norm(model):
                    bl_pair = v
                    break
        bl_val = None
        if bl_pair:
            bl_val = bl_pair[0] if "f1" in metric_col.lower() else bl_pair[1]

        if bl_val is not None and not math.isnan(float(bl_val)):
            ax.axhline(float(bl_val), color="black",
                        linewidth=2.2, linestyle=":",
                        zorder=5, label=f"Noiseless ({float(bl_val):.4f})")
            ax.text(noise_levels_sorted[-1] * 1.12,
                     float(bl_val),
                     f"  {float(bl_val):.4f}",
                     va="center", ha="left",
                     fontsize=FS_ANNOT, color="black",
                     fontweight="bold")

        # noise curves
        mask_model = df_noise[vqc_col].apply(lambda x: norm(x)==norm(model))
        df_model   = df_noise[mask_model]

        for nt, style in NOISE_STYLE.items():
            if nt not in noise_types_present:
                continue
            sub = (df_model[df_model[nm_col]==nt]
                   .sort_values(nl_col))
            if sub.empty:
                continue
            ax.plot(sub[nl_col].values,
                     sub[metric_col].values,
                     color=style["color"],
                     marker=style["marker"],
                     markersize=9, linewidth=2.4,
                     label=style["label"],
                     alpha=0.90, zorder=4)

        # axes formatting
        ax.set_xscale("log")
        ax.set_xticks(noise_levels_sorted)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticklabels([str(v) for v in noise_levels_sorted],
                            rotation=40, ha="right",
                            fontsize=FS_TICK)
        ax.set_xlim(noise_levels_sorted[0]*0.7,
                     noise_levels_sorted[-1]*1.6)
        ax.set_ylim(0.0, 1.08)
        ax.set_xlabel("Noise level $p$", fontsize=FS_AXIS)
        if col == 0:
            ax.set_ylabel(metric_label, fontsize=FS_AXIS+1,
                           fontweight="bold")
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.grid(True, which="both", alpha=0.20, linestyle=":")
        ax.set_facecolor("#F9F9F9")
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

        # subplot title — model code only
        mk = f"M{col+1}"
        ax.set_title(mk, fontsize=FS_TITLE, fontweight="bold", pad=8)

    return fig, axes


def add_legend(fig, noise_types_present):
    handles = [mlines.Line2D([], [], color="black", linewidth=2.2,
                              linestyle=":", label="Noiseless baseline")]
    for nt in noise_types_present:
        s = NOISE_STYLE[nt]
        handles.append(mlines.Line2D([], [], color=s["color"],
                                      marker=s["marker"],
                                      markersize=9, linewidth=2.4,
                                      label=s["label"]))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles),
               fontsize=FS_LEGEND, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.01),
               borderpad=0.7, handlelength=2.0,
               columnspacing=1.5)


def add_model_box(fig, models_ordered):
    """Full model name table below the legend."""
    lines = ["Model codes:"]
    for mi, model in enumerate(models_ordered, 1):
        lines.append(f"  M{mi}: {model}")
    box_text = "\n".join(lines)
    fig.text(0.5, -0.14, box_text,
              ha="center", va="top",
              fontsize=FS_BOX, fontfamily="monospace",
              linespacing=1.8,
              bbox=dict(boxstyle="round,pad=0.8",
                        facecolor="#EEF2FF",
                        edgecolor="#3F51B5",
                        linewidth=1.8, alpha=0.97))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv",  required=True)
    ap.add_argument("--dataset",      required=True)
    ap.add_argument("--jsc_table",    default=None)
    ap.add_argument("--results_root", default=None)
    ap.add_argument("--out_dir",      default=".")
    ap.add_argument("--top_n",        type=int, default=5)
    args = ap.parse_args()

    # output folder
    out_folder = Path(args.out_dir) / "noise_plot"
    out_folder.mkdir(parents=True, exist_ok=True)

    # load data
    df = load_summary(args.summary_csv)

    # identify columns
    vqc_col = next((c for c in df.columns
                     if "drm" in c.lower() or "vqc" in c.lower()), None)
    nm_col  = next((c for c in df.columns
                     if "noise_model" in c.lower()), None)
    nl_col  = next((c for c in df.columns
                     if "noise_level" in c.lower()), None)
    f1_col  = next((c for c in df.columns if c.lower()=="f1"), None)
    auc_col = next((c for c in df.columns if c.lower()=="auc"), None)

    if not all([vqc_col, nm_col, nl_col, f1_col, auc_col]):
        print(f"ERROR: Missing columns. Found: {list(df.columns)}"); return

    # drop HIM, convert types
    df = df[~df[nm_col].str.contains("him", case=False, na=False)].copy()
    for c in [nl_col, f1_col, auc_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df_noise = df[df[nl_col] > 0.0].copy()
    noise_levels_sorted = sorted(df_noise[nl_col].unique())
    noise_types_present = [nt for nt in NOISE_STYLE
                            if nt in df_noise[nm_col].values]
    print(f"  Noise types: {noise_types_present}")
    print(f"  Noise levels: {noise_levels_sorted}")

    # model order from JSC table
    models_ordered = []
    if args.jsc_table and Path(args.jsc_table).exists():
        jsc = pd.read_csv(args.jsc_table)
        rc  = next((c for c in jsc.columns if "jsc_rank" in c.lower()), None)
        dc  = next((c for c in jsc.columns
                     if "drm" in c.lower() or "vqc" in c.lower()), None)
        if rc and dc:
            models_ordered = list(jsc.sort_values(rc)
                                   .head(args.top_n)[dc].astype(str))
    if not models_ordered:
        models_ordered = list(df[vqc_col].unique())[:args.top_n]
    models_ordered = models_ordered[:args.top_n]
    n = len(models_ordered)

    print(f"\n  Models ({n}):")
    for i, m in enumerate(models_ordered, 1):
        print(f"    M{i}: {m}")

    # noiseless baseline
    noiseless_map = build_noiseless_map(
        df, vqc_col, nl_col, f1_col, auc_col, args.results_root)

    # figure size: wide enough for 5 models, tall enough for readability
    figsize = (5.5 * n, 6.5)
    rect_tight = [0.06, 0.16, 0.96, 0.93]   # left bottom right top

    for metric_col, metric_label, tag in [
        (f1_col,  "F1 Score", "F1"),
        (auc_col, "AUC",      "AUC"),
    ]:
        fig, axes = draw_row(
            models_ordered, df_noise, noiseless_map,
            vqc_col, nm_col, nl_col, metric_col,
            metric_label, noise_levels_sorted,
            noise_types_present, figsize)

        # super title
        fig.suptitle(
            f"Noise Robustness Study  —  Dataset: {args.dataset}  "
            f"|  Metric: {metric_label}",
            fontsize=FS_TITLE, fontweight="bold", y=0.99)

        add_legend(fig, noise_types_present)
        add_model_box(fig, models_ordered)

        plt.tight_layout(rect=rect_tight, pad=1.0)

        # save PNG and PDF
        for ext in [".png", ".pdf"]:
            out_path = out_folder / f"Noise_Study_{args.dataset}_{tag}{ext}"
            plt.savefig(str(out_path), dpi=160, bbox_inches="tight",
                         facecolor="white")
            print(f"  ✓ Saved → {out_path}")
        plt.close()

    print(f"\nAll images saved in: {out_folder}")


if __name__ == "__main__":
    main()