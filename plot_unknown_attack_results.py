"""
plot_unknown_attack.py — v4
Clean layout: legend at bottom, no overlapping titles, boxes aligned to subplots.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

DATASET_UNKNOWN = {
    "NF_UNSW_NB15": ["Shellcode", "Analysis", "Worms"],
    "UNSW_NB15":    ["Shellcode", "Analysis", "Worms"],
    "TON_IOT":      ["Ransomware", "MITM"],
}
DATASET_DISPLAY = {
    "NF_UNSW_NB15": "NF-UNSW-NB15",
    "UNSW_NB15":    "UNSW-NB15",
    "TON_IOT":      "TON-IOT",
}
ATTACK_COLORS = {
    "Shellcode":  "#E91E63",
    "Analysis":   "#FF9800",
    "Worms":      "#9C27B0",
    "Ransomware": "#F44336",
    "MITM":       "#FF5722",
    "_OrigTPR":   "#78909C",
}


# Internal → display name maps (same as training code)
RENAME_MAP = {
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
DRM_DISPLAY = {
    "pca4":        "PCA4",
    "ica4":        "ICA4",
    "xgb_pca4":   "XGB_PCA4",
    "autoencoder": "Autoencoder",
}

def build_vqc_display(feature_set, encoding, ansatz):
    """Reconstruct display DRM+VQC string from internal column values."""
    drm = DRM_DISPLAY.get(str(feature_set).lower(), str(feature_set).upper())
    enc = RENAME_MAP.get(str(encoding).lower(), str(encoding))
    ans = RENAME_MAP.get(str(ansatz).lower(), str(ansatz))
    return f"{drm}+VQC({enc}+{ans})"


def infer_dataset(csv_path):
    name = Path(csv_path).stem
    for ds in sorted(DATASET_UNKNOWN.keys(), key=len, reverse=True):
        if ds.lower() in name.lower():
            return ds
    return name.replace("unknown_attack_detection_", "")


def load_jsc(jsc_path, top_n):
    if not jsc_path or not Path(jsc_path).exists():
        return []
    jsc  = pd.read_csv(jsc_path)
    rc   = next((c for c in jsc.columns if "jsc_rank" in c.lower()), None)
    dc   = next((c for c in jsc.columns
                  if "drm" in c.lower() or "vqc" in c.lower()), None)
    tprc = next((c for c in jsc.columns if "tpr_pm" in c.lower()), None)
    if not rc or not dc: return []
    def mean(v):
        if v is None: return np.nan
        try:
            s = str(v)
            return float(s.split("±")[0].strip()) if "±" in s else float(s)
        except: return np.nan
    return [{"rank": int(r[rc]), "vqc": str(r[dc]).strip(),
              "tpr": mean(r[tprc] if tprc else np.nan)}
            for _, r in jsc.sort_values(rc).head(top_n).iterrows()]


def draw_chart(ax, det_df, jsc_models, dataset, top_n, perf_csv=None):
    """
    perf_csv = path to combined_summary_metrics.csv (or new_format_combined_result.csv).
    This is the CORRECT source for test-set TPR — computed on test_2000.csv
    with the original threshold τ*.  Falls back to tpr_pm from JSC table if
    perf_csv not provided.
    """
    """Draw the bar chart into ax. Returns list of (handle,label) for legend."""
    unknown_types = DATASET_UNKNOWN.get(dataset, [])
    display_name  = DATASET_DISPLAY.get(dataset, dataset)
    vqc_col  = next((c for c in det_df.columns
                      if "drm" in c.lower() or "vqc" in c.lower()), None)
    rank_col = "JSC_Rank" if "JSC_Rank" in det_df.columns else None
    if rank_col:
        det_df = det_df.sort_values(rank_col).head(top_n).reset_index(drop=True)
    else:
        det_df = det_df.head(top_n).reset_index(drop=True)
    n_models = min(len(det_df), top_n)

    norm = lambda s: str(s).strip().replace(" ", "")

    # ── Build TPR reference map ────────────────────────────────────────────
    # Priority 1: performance CSV (combined_summary_metrics.csv) →
    #             gives ACTUAL test-set TPR from single test_2000.csv eval
    # Priority 2: tpr_pm from JSC/reliability table →
    #             gives BOOTSTRAP MEAN TPR (may differ from single-split TPR)
    # Priority 3: Overall_DR fallback (wrong — detection rate on unknown set)
    tpr_map = {}
    if perf_csv and Path(perf_csv).exists():
        try:
            pf     = pd.read_csv(perf_csv)
            # Find VQC identifier column and TPR column
            # combined_summary_metrics_with_TPR_FPR.csv has internal
            # column names (feature_set, encoding, ansatz, TPR).
            # Reconstruct the display DRM+VQC string for key matching.
            fs_col  = next((c for c in pf.columns if "feature_set" in c.lower() or c.lower()=="fs"), None)
            enc_col = next((c for c in pf.columns if c.lower()=="encoding"), None)
            ans_col = next((c for c in pf.columns if c.lower()=="ansatz"), None)
            tpr_pc  = next((c for c in pf.columns
                             if c.lower() in ("tpr","recall","TPR".lower())), None)
            vqc_pc  = next((c for c in pf.columns
                             if "drm" in c.lower() or "vqc" in c.lower()), None)

            if tpr_pc and (fs_col and enc_col and ans_col):
                # Build key from internal names using RENAME_MAP
                for _, pr in pf.iterrows():
                    key = norm(build_vqc_display(
                        pr[fs_col], pr[enc_col], pr[ans_col]))
                    tpr_map[key] = float(pr[tpr_pc])
                print(f"  [TPR ref] Built from feature_set+encoding+ansatz columns "
                      f"in {Path(perf_csv).name} ({len(tpr_map)} models)")
            elif tpr_pc and vqc_pc:
                # Fallback: direct vqc column if present
                for _, pr in pf.iterrows():
                    tpr_map[norm(str(pr[vqc_pc]))] = float(pr[tpr_pc])
                print(f"  [TPR ref] Using DRM+VQC column in {Path(perf_csv).name}")
            else:
                print(f"  [WARN] perf_csv missing required columns — "
                      f"falling back to JSC tpr_pm. "
                      f"Columns found: {list(pf.columns)[:8]}")
        except Exception as _e:
            print(f"  [WARN] Could not read perf_csv ({_e}) — "
                  f"falling back to JSC tpr_pm")

    if not tpr_map:
        # Fallback to bootstrap tpr_pm from JSC table
        tpr_map = {norm(m["vqc"]): m["tpr"] for m in jsc_models}
        if any(not np.isnan(float(v)) for v in tpr_map.values()):
            print(f"  [TPR ref] Using bootstrap tpr_pm from JSC table "
                  f"(note: may differ from single-split test TPR)")
        else:
            print(f"  [WARN] No TPR reference found — grey bar will not show")

    bar_items = unknown_types + ["_OrigTPR"]
    n_bars    = len(bar_items)
    bar_w     = 0.70 / n_bars
    x_centers = np.arange(n_models, dtype=float)
    handles_labels = []

    for bi, item in enumerate(bar_items):
        offset = -(n_bars * bar_w / 2) + bi * bar_w + bar_w / 2
        x_pos  = x_centers + offset
        vals   = []
        for mi in range(n_models):
            row = det_df.iloc[mi]
            if item == "_OrigTPR":
                v = tpr_map.get(norm(str(row.get(vqc_col, ""))), np.nan)
                if np.isnan(float(v if v is not None else float("nan"))) \
                        and "Overall_DR" in row.index:
                    v = float(row["Overall_DR"])
            else:
                col = f"DR_{item}"
                v   = float(row[col]) if col in row.index else np.nan
            vals.append(float(v) if v is not None else np.nan)

        color = ATTACK_COLORS.get(item, "#999")
        alpha = 0.55 if item == "_OrigTPR" else 0.87
        label = ("TPR — test set\n(known classes)" if item == "_OrigTPR"
                 else f"DR: {item}")
        bars  = ax.bar(x_pos, vals, bar_w * 0.88,
                        color=color, alpha=alpha,
                        edgecolor="white", linewidth=0.6,
                        label=label, zorder=3)
        handles_labels.append((bars[0], label))

        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                import math
                if item == "_OrigTPR":
                    # 4 decimal places TRUNCATED (no rounding)
                    display = math.trunc(v * 10000) / 10000
                    fmt = f"{display:.4f}"
                else:
                    # 2 decimal places for unknown attack DR bars
                    fmt = f"{v:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                         v + 0.013, fmt,
                         ha="center", va="bottom",
                         fontsize=7.5 if item != "_OrigTPR" else 8.5,
                         fontweight="bold",
                         color="#222" if item != "_OrigTPR" else "#37474F")

    # Axes styling
    ax.set_xlim(-0.55, n_models - 0.45)
    ax.set_ylim(0, 1.25)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"M{i+1}" for i in range(n_models)],
                        fontsize=12, fontweight="bold")
    ax.set_ylabel("Detection Rate / TPR", fontsize=10)
    ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", alpha=0.20, linestyle=":", zorder=0)
    ax.set_facecolor("#F8F9FA")
    for sp in ax.spines.values(): sp.set_linewidth(0.8)

    # Dataset title INSIDE the axes — positioned well above bars
    ax.text(0.5, 1.04, f"Dataset: {display_name}",
             transform=ax.transAxes,
             ha="center", va="bottom",
             fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="#E3F2FD",
                       edgecolor="#1565C0",
                       linewidth=1.5, alpha=0.95))

    return handles_labels


def draw_namebox(ax_box, det_df, jsc_models, dataset, top_n):
    """
    Draw the model name box by styling the axes itself as the box.
    The axes always matches the subplot width → perfect alignment.
    Title is centred. Model entries are left-aligned below it.
    """
    display_name = DATASET_DISPLAY.get(dataset, dataset)
    vqc_col  = next((c for c in det_df.columns
                      if "drm" in c.lower() or "vqc" in c.lower()), None)
    rank_col = "JSC_Rank" if "JSC_Rank" in det_df.columns else None
    if rank_col:
        det_df = det_df.sort_values(rank_col).head(top_n).reset_index(drop=True)
    else:
        det_df = det_df.head(top_n).reset_index(drop=True)
    n_models = min(len(det_df), top_n)

    # ── Use the axes itself as the box ──────────────────────────────────────
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.set_facecolor("#EEF2FF")        # box fill colour
    ax_box.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)
    for spine in ax_box.spines.values():   # box border
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_edgecolor("#3F51B5")

    # ── Centred title line ──────────────────────────────────────────────────
    ax_box.text(0.5, 0.96,
                 f"Model codes  —  {display_name}",
                 transform=ax_box.transAxes,
                 ha="center", va="top",
                 fontsize=9, fontweight="bold",
                 fontfamily="monospace", color="#1A237E")

    # ── Thin separator line under title ────────────────────────────────────
    ax_box.axhline(y=0.82, xmin=0.02, xmax=0.98,
                    color="#3F51B5", linewidth=0.8, alpha=0.5)

    # ── Model entries: rank code then full VQC name ─────────────────────────
    n_text_rows = n_models * 2 + (n_models - 1)   # 2 lines + gap between
    row_h       = 0.78 / max(n_text_rows, 1)
    y           = 0.78

    for mi in range(n_models):
        row  = det_df.iloc[mi]
        vqc  = str(row.get(vqc_col, f"Model {mi+1}"))
        rank = int(row[rank_col]) if rank_col else mi + 1

        # Rank code line — bold
        ax_box.text(0.025, y,
                     f"M{mi+1}  (JSC Rank #{rank})",
                     transform=ax_box.transAxes,
                     ha="left", va="top",
                     fontsize=8.5, fontweight="bold",
                     fontfamily="monospace", color="#1A237E")
        y -= row_h

        # Full model name — normal weight, indented
        ax_box.text(0.05, y,
                     vqc,
                     transform=ax_box.transAxes,
                     ha="left", va="top",
                     fontsize=8, fontfamily="monospace", color="#333333")
        y -= row_h

        # Gap between models
        if mi < n_models - 1:
            y -= row_h * 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection_csvs", required=True)
    ap.add_argument("--jsc_tables",     default=None)
    ap.add_argument("--perf_csvs",      default=None,
                     help="Comma-separated combined_summary_metrics.csv paths "
                          "(one per dataset, same order as detection_csvs). "
                          "Used for correct test-set TPR reference bar. "
                          "If omitted, falls back to tpr_pm from JSC table.")
    ap.add_argument("--out_dir",        default=".")
    ap.add_argument("--top_n",          type=int, default=3)
    args = ap.parse_args()

    det_paths = [p.strip() for p in args.detection_csvs.split(",") if p.strip()]
    jsc_raw   = ([p.strip() for p in args.jsc_tables.split(",") if p.strip()]
                  if args.jsc_tables else [])
    jsc_paths = (jsc_raw + [None]*len(det_paths))[:len(det_paths)]
    perf_raw  = ([p.strip() for p in args.perf_csvs.split(",") if p.strip()]
                  if args.perf_csvs else [])
    perf_paths= (perf_raw + [None]*len(det_paths))[:len(det_paths)]

    n_ds = len(det_paths)
    fig  = plt.figure(figsize=(7.5 * n_ds, 11))
    fig.patch.set_facecolor("#FFFFFF")

    # ── Layout: 2 rows × n_ds cols ─────────────────────────────────────────
    # Row 0 = charts (taller), Row 1 = name boxes
    # Generous top margin for dataset title labels (drawn inside axes)
    # Generous bottom margin for legend
    gs = GridSpec(
        2, n_ds,
        height_ratios=[3.6, 1.0],
        hspace=0.12,          # gap between chart and box rows
        wspace=0.20,          # gap between datasets
        left=0.06, right=0.97,
        top=0.91,             # room for per-subplot title badges
        bottom=0.12,          # room for legend at bottom
    )

    all_hl = {}   # deduplicated legend entries

    for col, (det_path, jsc_path) in enumerate(zip(det_paths, jsc_paths)):
        ax     = fig.add_subplot(gs[0, col])
        ax_box = fig.add_subplot(gs[1, col])

        if not Path(det_path).exists():
            ax.text(0.5, 0.5, f"File not found:\n{Path(det_path).name}",
                     ha="center", va="center", transform=ax.transAxes,
                     color="red", fontsize=9)
            ax.axis("off"); ax_box.axis("off"); continue

        det_df  = pd.read_csv(det_path)
        dataset = infer_dataset(det_path)
        jsc_m   = load_jsc(jsc_path, args.top_n)

        perf_path = perf_paths[col]
        hl = draw_chart(ax, det_df, jsc_m, dataset, args.top_n,
                         perf_csv=perf_path)
        draw_namebox(ax_box, det_df, jsc_m, dataset, args.top_n)

        for h, lbl in hl:
            if lbl not in all_hl:
                all_hl[lbl] = h

    # ── Legend at the very bottom, below name boxes ─────────────────────────
    if all_hl:
        fig.legend(
            handles=list(all_hl.values()),
            labels=list(all_hl.keys()),
            loc="lower center",
            ncol=min(7, len(all_hl)),
            fontsize=9.5,
            frameon=True, framealpha=0.96,
            edgecolor="#BDBDBD",
            bbox_to_anchor=(0.5, 0.005),
            borderpad=0.7, handlelength=1.8,
            columnspacing=1.2,
        )

    # ── Single super-title at top ────────────────────────────────────────────
    fig.suptitle(
        "Unknown Attack Detection — Top-3 JSC Models per Dataset",
        fontsize=13, fontweight="bold", y=0.99,
    )
    

    out_path = Path(args.out_dir) / "unknown_attack_detection_all_datasets.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=160, bbox_inches="tight",
                 facecolor="#FFFFFF")
    plt.close()
    print(f"\n✓ Saved -> {out_path}")


if __name__ == "__main__":
    main()