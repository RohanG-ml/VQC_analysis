"""
run_unknown_attack_detection_v3.py

Changes from the original:
  1. STRICT checkpoint matching. The folder must be exactly
     {fs}_{train_size}_{test_split}. No fallback to other folders,
     because loading a model trained on a different train_size
     silently produces wrong weights and a wrong threshold.
  2. RENAME_REV extended with custom_ansatz_1, strongly_entangling
     and the capitalised variants that previously caused silent skips.
  3. SANITY CHECK: recomputes test-set F1 with the loaded weights and
     compares against result.json. A mismatch aborts that model rather
     than reporting invalid numbers.
  4. Probability distribution diagnostic, so a zero detection rate can
     be attributed to threshold transfer rather than a code fault.
  5. Integrity check at the end listing any selected model that did
     not produce a result.

Original description follows.

run_unknown_attack_detection.py
Loads the top-N models from the JSC score table, evaluates each on the
unknown attack dataset, and reports Detection Rate (TPR) per attack subtype.

No FPR is reported (unknown attack set has label=1 only — no benign samples).
Uses the saved best model checkpoint — same weights as evaluated in bootstrap/
uncertainty study. Applies SAME MinMaxScaler (fitted on train_5000.csv) and
SAME τ* (from result.json).

Run:
  python run_unknown_attack_detection.py \
      --jsc_table /home/nvidia/21PHD1192/qml_id2/jsc_score_table_TON_IOT.csv \
      --results_root /home/nvidia/21PHD1192/qml_id2/UNSW/results_vqc_TON_IOT_v8_default_mixed \
      --package_root /lp-dev/21PHD1192/qml_id2/UNSW/gpu_package \
      --unknown_csv /lp-dev/21PHD1192/qml_id2/UNSW/unknown_attacks/TON_IOT_unknown.csv \
      --dataset TON_IOT \
      --top_n 3 \
      --out_dir /home/nvidia/21PHD1192/qml_id2 \
      --vqc_script /home/nvidia/21PHD1192/qml_id2/UNSW_code/vqc_4feature_UNSW_v8.py
"""

# ── NOTE: set CUDA_VISIBLE_DEVICES="" at SHELL level before running ────────
# e.g.:  CUDA_VISIBLE_DEVICES="" python run_unknown_attack_detection.py ...
# Setting it inside Python is too late — torch's C++ layer initialises
# CUDA before Python code runs. The shell-level env var is inherited first.
import os
# Attempt in-Python disable as a secondary measure (helps in some configs)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

print(f"[STARTUP] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','not set')!r} — CPU-only mode", flush=True)

import argparse, json, sys, re
print("[STARTUP] standard libs imported", flush=True)
import numpy as np
print("[STARTUP] numpy imported", flush=True)
import pandas as pd
print("[STARTUP] pandas imported", flush=True)
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
print("[STARTUP] sklearn imported", flush=True)

# ── Attack column config ──────────────────────────────────────────────────
ATTACK_COL = {"UNSW_NB15":"Attack","NF_UNSW_NB15":"Attack","TON_IOT":"Attack"}
LABEL_COL  = "Label"

ATTACK_NAME_FIX = {
    "normal":"Benign","benign":"Benign","ddos":"DDoS","dos":"DoS",
    "backdoor":"Backdoor","backdoors":"Backdoor","injection":"Injection",
    "password":"Password","scanning":"Scanning","xss":"XSS",
    "ransomware":"Ransomware","mitm":"MITM","exploit":"Exploits",
    "exploits":"Exploits","fuzzer":"Fuzzers","fuzzers":"Fuzzers",
    "reconnaissance":"Reconnaissance","generic":"Generic",
    "shellcode":"Shellcode","analysis":"Analysis","worms":"Worms",
}

FEATURE_COLS = {
    "pca4":["PC1","PC2","PC3","PC4"],
    "ica4":["IC1","IC2","IC3","IC4"],
    "xgb_pca4":["XP1","XP2","XP3","XP4"],
    "autoencoder":["z0","z1","z2","z3"],
}

def parse_drm_vqc(drm_vqc_str):
    """Parse 'PCA4+VQC(IQP_style_encoding+EfficientSU2_like)' → (fs, enc, ans)"""
    m = re.match(r"(.+?)\+VQC\((.+?)\+(.+?)\)$", str(drm_vqc_str).strip())
    if not m:
        return None, None, None
    drm = m.group(1).strip()
    enc = m.group(2).strip()
    ans = m.group(3).strip()
    drm_map = {"PCA4":"pca4","ICA4":"ica4","XGB_PCA4":"xgb_pca4",
               "AUTOENCODER":"autoencoder"}
    fs = drm_map.get(drm.upper(), drm.lower())
    # Reverse rename map
    RENAME_REV = {
        "RX_Angle_Encoding":"rx_embedding","RY_Angle_Encoding":"ry_embedding",
        "RZ_Angle_Encoding":"rz_embedding","IQP_style_encoding":"iqp_embedding",
        "ZZ_Feature_Map_style":"zz_feature_map","Amplitude_Embedding":"amplitude_embedding",
        "Custom_H_RY_RZ":"custom_h_ry_rz","EfficientSU2_like":"efficient_su2_like",
        "Real_Amplitude_like":"real_amplitudes",
        # ── keys that were missing and caused silent skips ──
        "custom_ansatz_1":"custom_ansatz_1",
        "strongly_entangling":"strongly_entangling",
        "custom_Ansatz_1":"custom_ansatz_1",
        "IQP_style_Encoding":"iqp_embedding",
        "ZZ_Feature_Map_Style":"zz_feature_map",
    }
    enc_internal = RENAME_REV.get(enc, enc)
    ans_internal = RENAME_REV.get(ans, ans)
    return fs, enc_internal, ans_internal

def find_checkpoint(results_root, fs, train_size, test_split, ansatz, encoding):
    """Find best_model.pt for a given model config.

    STRICT matching only. Never falls back to a folder with a different
    train_size or test_split, because loading a model trained on different
    data silently produces wrong weights and a wrong threshold.
    """
    root     = Path(results_root)
    run_key  = f"{ansatz}__{encoding}"
    exact_sd = root / f"{fs}_{train_size}_{test_split}"

    if exact_sd.is_dir():
        pt = exact_sd / f"{run_key}_best_model.pt"
        if pt.exists():
            print(f"  [CKPT] Folder : {exact_sd.name}")
            print(f"  [CKPT] Run key: {run_key}")
            return pt, exact_sd / f"{run_key}_result.json"

        # Folder exists but this run_key is absent -> show what IS there
        present = sorted(p.name.replace("_best_model.pt", "")
                         for p in exact_sd.glob("*_best_model.pt"))
        print(f"  [CKPT] FAIL: run_key {run_key!r} not in {exact_sd.name}/")
        print(f"  [CKPT] {len(present)} checkpoints present:")
        for k in present:
            near = "   <-- similar" if (ansatz in k or encoding in k) else ""
            print(f"           {k}{near}")
        return None, None

    print(f"  [CKPT] FAIL: folder does not exist: {exact_sd}")
    siblings = sorted(p.name for p in root.glob(f"{fs}_*") if p.is_dir())
    print(f"  [CKPT] Folders starting with {fs!r}: {siblings}")
    print(f"  [CKPT] Expected train_size={train_size}, "
          f"test_split={test_split!r}. Adjust --train_size / --test_split "
          f"if the folder name above differs.")
    return None, None

def predict_proba_cpu(model, X, batch_size=256):
    import torch
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    probs = []
    with torch.no_grad():
        for s in range(0, len(X_t), batch_size):
            logits = model(X_t[s:s+batch_size]).detach().cpu()
            probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs).reshape(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsc_table",    required=True)
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--package_root", required=True)
    ap.add_argument("--unknown_csv",  default=None,
                     help="Explicit path to unknown attack CSV. If omitted, "
                          "auto-resolved to package_root/{feature_set}/unknown.csv "
                          "— present in each DRM package folder.")
    ap.add_argument("--dataset",      required=True)
    ap.add_argument("--top_n",        type=int, default=3)
    ap.add_argument("--train_size",   type=int, default=5000)
    ap.add_argument("--test_split",   default="test_balanced_2000")
    ap.add_argument("--out_dir",      default=".")
    ap.add_argument("--vqc_script",   required=True,
                     help="Path to vqc_4feature_UNSW_v8.py for importing build_model")
    args = ap.parse_args()

    # Import build_model from training script
    print(f"[IMPORT] Adding {Path(args.vqc_script).parent} to sys.path", flush=True)
    sys.path.insert(0, str(Path(args.vqc_script).parent))
    print("[IMPORT] Importing torch...", flush=True)
    import torch
    print(f"[IMPORT] torch {torch.__version__} imported — CUDA available: {torch.cuda.is_available()}", flush=True)
    print("[IMPORT] Importing build_model from vqc_4feature_UNSW_v8...", flush=True)
    from vqc_4feature_UNSW_v8 import build_model, make_pl_device
    print("[IMPORT] ✓ build_model and make_pl_device imported successfully", flush=True)

    # Load JSC table
    jsc = pd.read_csv(args.jsc_table)
    jsc_rank_col = next((c for c in jsc.columns if "jsc_rank" in c.lower()), None)
    if jsc_rank_col is None:
        print("✗ JSC_Rank column not found in jsc_table.")
        return
    jsc = jsc.sort_values(jsc_rank_col).head(args.top_n)
    drm_col = next((c for c in jsc.columns if "drm" in c.lower() or "vqc" in c.lower()), None)
    print(f"Top {args.top_n} models from JSC table:")

    results = []
    for rank_idx, jsc_row in jsc.iterrows():
        jsc_rank = jsc_row[jsc_rank_col]
        drm_vqc  = jsc_row[drm_col] if drm_col else ""
        fs, enc, ans = parse_drm_vqc(drm_vqc)
        if fs is None:
            print(f"\n✗ Could not parse: {drm_vqc}"); continue

        print(f"\n{'='*70}")
        print(f"  JSC_Rank={jsc_rank} | {drm_vqc}")
        print(f"  Parsed → DRM (feature_set) : {fs}")
        print(f"  Parsed → Encoding (internal): {enc}")
        print(f"  Parsed → Ansatz  (internal): {ans}")
        print(f"{'='*70}")

        # ── Auto-resolve unknown CSV from this model's DRM folder ─────────
        if args.unknown_csv:
            unk_csv_path = Path(args.unknown_csv)
        else:
            unk_csv_path = Path(args.package_root) / fs / "unknown.csv"

        print(f"\n  [UNKNOWN] Looking for: {unk_csv_path}")
        if not unk_csv_path.exists():
            print(f"  ✗ unknown.csv not found at {unk_csv_path}")
            pkg_fs_dir = Path(args.package_root) / fs
            if pkg_fs_dir.exists():
                unk_candidates = list(pkg_fs_dir.glob("unknown*.csv"))
                print(f"  Available unknown-related files in {pkg_fs_dir.name}/:")
                for p in unk_candidates:
                    print(f"      {p.name}")
            continue
        print(f"  ✓ Found: {unk_csv_path.name}  ({unk_csv_path.stat().st_size//1024} KB)")

        unk_df  = pd.read_csv(unk_csv_path)
        atk_col = ATTACK_COL.get(args.dataset, "Attack")
        if atk_col not in unk_df.columns:
            for alt in ["attack_cat","Attack","attack","type","category"]:
                if alt in unk_df.columns:
                    atk_col = alt; break
        print(f"  [UNKNOWN] {len(unk_df)} samples  |  attack column: '{atk_col}'")
        print(f"  [UNKNOWN] Attack subtype counts:")
        print(unk_df[atk_col].value_counts().to_string().replace("^","    "))

        # Find checkpoint
        ck_path, res_path = find_checkpoint(
            args.results_root, fs, args.train_size, args.test_split, ans, enc)
        if ck_path is None:
            print(f"  ✗ Checkpoint not found in {args.results_root}"); continue
        print(f"  Checkpoint: {ck_path}")

        # Load τ* from result.json
        tau = 0.5
        if res_path and res_path.exists():
            res = json.load(open(res_path))
            tau = float(res.get("metrics", {}).get("threshold", 0.5))
        print(f"  τ* = {tau:.4f}")

        # Load checkpoint and rebuild model
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        cfg = ck["config"]
        cfg["encoding_name"] = enc
        cfg["ansatz_name"]   = ans
        dev, diff_method, _ = make_pl_device(
            int(cfg["n_qubits"]), backend="default.qubit")  # CPU, no noise for eval
        model = build_model(cfg, dev, diff_method)
        model.load_state_dict(ck["model_state"])
        print(f"  Model loaded. n_qubits={cfg['n_qubits']} enc_reps={cfg.get('enc_reps',1)}")

        # Load MinMaxScaler from train_5000.csv
        train_csv = Path(args.package_root) / fs / f"train_{args.train_size}.csv"
        if not train_csv.exists():
            print(f"  ✗ train CSV not found: {train_csv}"); continue
        train_df  = pd.read_csv(train_csv)
        feat_cols = FEATURE_COLS.get(fs, ["PC1","PC2","PC3","PC4"])
        feat_cols = [c for c in feat_cols if c in train_df.columns]
        scaler    = MinMaxScaler(feature_range=(0.0, np.pi))
        scaler.fit(train_df[feat_cols].values.astype(np.float32))

        # ── SANITY CHECK: reproduce the test-set F1 from result.json ────
        # If the loaded weights are the intended ones, evaluating on the
        # SAME test set with the SAME tau must reproduce the stored F1.
        # A mismatch means the wrong checkpoint was loaded, which would
        # make every downstream number meaningless.
        sanity_ok = None
        try:
            test_csv = (Path(args.package_root) / fs /
                        args.test_split / "test.csv")
            if not test_csv.exists():
                alt = Path(args.package_root) / fs / f"{args.test_split}.csv"
                test_csv = alt if alt.exists() else test_csv

            if test_csv.exists() and res_path and res_path.exists():
                from sklearn.metrics import f1_score
                t_df   = pd.read_csv(test_csv)
                t_cols = [c for c in feat_cols if c in t_df.columns]
                lab_c  = next((c for c in ["Label","label","y"]
                               if c in t_df.columns), None)
                if t_cols and lab_c:
                    X_t     = scaler.transform(
                                t_df[t_cols].values.astype(np.float32))
                    p_t     = predict_proba_cpu(model, X_t)
                    pred_t  = (p_t >= tau).astype(int)
                    f1_now  = f1_score(t_df[lab_c].values, pred_t,
                                       zero_division=0)
                    res_j   = json.load(open(res_path))
                    f1_json = float(res_j.get("metrics", {}).get("f1", -1))
                    delta   = abs(f1_now - f1_json)
                    sanity_ok = delta < 0.01

                    print(f"  [SANITY] test F1 recomputed = {f1_now:.4f}")
                    print(f"  [SANITY] test F1 in json    = {f1_json:.4f}")
                    if sanity_ok:
                        print(f"  [SANITY] MATCH (delta={delta:.4f}) "
                              f"-> correct checkpoint loaded")
                    else:
                        print(f"  [SANITY] *** MISMATCH (delta={delta:.4f}) ***")
                        print(f"  [SANITY] The loaded weights do NOT reproduce "
                              f"the stored result.")
                        print(f"  [SANITY] Checkpoint : {ck_path}")
                        print(f"  [SANITY] result.json: {res_path}")
                        print(f"  [SANITY] Skipping this model — its unknown "
                              f"attack numbers would be invalid.")
                        continue
                else:
                    print(f"  [SANITY] skipped (no feature/label columns "
                          f"in {test_csv.name})")
            else:
                print(f"  [SANITY] skipped (test csv or result.json absent)")
        except Exception as e:
            print(f"  [SANITY] check could not run: {e}")

        # Prepare unknown attack features
        unk_feat_cols = [c for c in feat_cols if c in unk_df.columns]
        if not unk_feat_cols:
            print(f"  ✗ Feature columns {feat_cols} not found in unknown CSV"); continue
        missing_cols = [c for c in feat_cols if c not in unk_df.columns]
        if missing_cols:
            print(f"  *** WARNING: unknown.csv is missing {missing_cols}")
            print(f"      Using only {unk_feat_cols} — feature order will")
            print(f"      not match the scaler and predictions will be wrong.")
            continue
        X_unk = scaler.transform(unk_df[unk_feat_cols].values.astype(np.float32))

        # Predict
        probs = predict_proba_cpu(model, X_unk)
        preds = (probs >= tau).astype(int)

        # ── Diagnostic: probability distribution vs threshold ───────────
        # If detection rate is 0, this shows immediately whether the
        # probabilities are all below tau (threshold too high for the
        # unknown distribution) or the model is outputting a constant.
        print(f"  [PROBS] min={probs.min():.4f}  "
              f"p25={np.percentile(probs,25):.4f}  "
              f"median={np.median(probs):.4f}  "
              f"p75={np.percentile(probs,75):.4f}  "
              f"max={probs.max():.4f}")
        print(f"  [PROBS] tau={tau:.4f}  "
              f"fraction >= tau: {(probs >= tau).mean()*100:.2f}%")
        if probs.max() < tau:
            print(f"  [PROBS] ALL probabilities are below tau. Detection "
                  f"rate is 0 because the threshold optimised on the test")
            print(f"          set does not transfer to the unknown attack")
            print(f"          distribution. This is a genuine finding, not "
                  f"a code error.")
        if probs.std() < 1e-4:
            print(f"  [PROBS] *** Probabilities are nearly CONSTANT "
                  f"(std={probs.std():.2e}) ***")
            print(f"          The model is not discriminating at all. Check "
                  f"that the feature columns and scaler are correct.")

        # Detection rate per attack subtype
        subtypes = (unk_df[atk_col].astype(str).str.strip()
                    .apply(lambda x: ATTACK_NAME_FIX.get(x.lower(), x)))
        overall_dr = preds.mean()

        print(f"  Overall Detection Rate: {overall_dr*100:.2f}%  "
              f"({preds.sum()}/{len(preds)} detected as attack)")
        print(f"  Per-subtype Detection Rate:")
        for subtype in sorted(subtypes.unique()):
            mask = subtypes == subtype
            sub_dr = preds[mask].mean()
            print(f"    {subtype:<20} {preds[mask].sum():>5}/{mask.sum():<5} "
                  f"({sub_dr*100:>6.2f}%)")

        row = {"JSC_Rank": jsc_rank, "DRM+VQC": drm_vqc,
               "tau": tau, "Overall_DR": round(overall_dr, 4),
               "n_detected": int(preds.sum()), "n_total": len(preds)}
        for subtype in sorted(subtypes.unique()):
            mask = subtypes == subtype
            row[f"DR_{subtype}"] = round(float(preds[mask].mean()), 4)
        results.append(row)

    # ── Integrity check: did every selected model produce a result? ──
    sel = set(int(r[jsc_rank_col]) for _, r in jsc.iterrows())
    got = set(int(r["JSC_Rank"]) for r in results)
    missing_ranks = sorted(sel - got)

    print(f"\n{'='*70}")
    print(f"  INTEGRITY CHECK")
    print(f"{'='*70}")
    print(f"  Selected from JSC table : {len(sel)}")
    print(f"  Successfully evaluated  : {len(got)}")
    if missing_ranks:
        print(f"\n  *** {len(missing_ranks)} MODEL(S) SKIPPED ***")
        print(f"  Missing JSC_Rank: {missing_ranks}")
        for _, r in jsc.iterrows():
            if int(r[jsc_rank_col]) in missing_ranks:
                print(f"    Rank {int(r[jsc_rank_col])}: {r[drm_col]}")
        print(f"\n  Scroll up to the [CKPT] or [SANITY] messages for these")
        print(f"  models. The reported top-{args.top_n} is INCOMPLETE.")
    else:
        print(f"  All selected models evaluated. Selection is complete.")
    print(f"{'='*70}")

    # Save
    if results:
        out_dir  = Path(args.out_dir)
        out_path = out_dir / f"unknown_attack_detection_{args.dataset}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\n{'='*70}")
        print(f"✓ Saved -> {out_path}")
        print(pd.DataFrame(results)[["JSC_Rank","DRM+VQC","tau","Overall_DR"]].to_string(index=False))

if __name__ == "__main__":
    main()