"""
verify_backend_equivalence.py
Loads ONE actual trained model and runs it through BOTH lightning.gpu
and default.qubit, comparing the resulting probabilities directly.
This settles empirically whether the backend switch changed any numbers.

Run:
  python verify_backend_equivalence.py \
      --results_dir /path/to/results \
      --package_root /path/to/gpu_package \
      --feature_set xgb_pca4 --ansatz custom_ansatz_1 --encoding iqp_embedding
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

CODE_DIR = "/home/nvidia/21PHD1192/qml_id2/UNSW_code"
sys.path.insert(0, CODE_DIR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--package_root", required=True)
    ap.add_argument("--feature_set", required=True)
    ap.add_argument("--ansatz", required=True)
    ap.add_argument("--encoding", required=True)
    args = ap.parse_args()

    # Import functions from the actual script (everything before discover())
    src = open(f"{CODE_DIR}/vqc_bootstrap_v4.py").read()
    stop = src.find("\ndef discover(")
    ns = {}
    exec(compile(src[:stop], "vqc_bootstrap_v4.py", "exec"), ns)
    feature_cols_for = ns["feature_cols_for"]
    build_model       = ns["build_model"]
    predict_proba     = ns["predict_proba"]

    # Find model file
    run_key = f"{args.ansatz}__{args.encoding}"
    rdir = Path(args.results_dir)
    sd = next((d for d in rdir.iterdir()
               if d.is_dir() and d.name.startswith(args.feature_set)), None)
    best_f = sd / f"{run_key}_best_model.pt"
    print(f"Loading: {best_f}")

    ck  = torch.load(str(best_f), map_location="cpu", weights_only=False)
    cfg = ck["config"]

    # Load data (same way the actual script does)
    rest = sd.name[len(args.feature_set):].lstrip("_").split("_")
    ts = next((int(x) for x in rest if x.isdigit()), 5000)
    base = Path(args.package_root) / args.feature_set
    tr_df = pd.read_csv(base / f"train_{ts}.csv")
    te_df = pd.read_csv(base / "test_2000.csv")
    fc = feature_cols_for(args.feature_set, tr_df)
    sc = MinMaxScaler(feature_range=(0.0, np.pi))
    sc.fit(tr_df[fc].values.astype(np.float32))
    X = sc.transform(te_df[fc].values.astype(np.float32)).astype(np.float32)

    # ── Run on lightning.gpu ──────────────────────────────────────────
    import pennylane as qml
    print("\nBuilding model with lightning.gpu...")
    try:
        dev_gpu = qml.device("lightning.gpu", wires=int(cfg["n_qubits"]), shots=None)
        diff_gpu = "adjoint"
        model_gpu = build_model(cfg, dev_gpu, diff_gpu)
        model_gpu.load_state_dict(ck["model_state"])
        model_gpu.eval()
        print("Running predict_proba on lightning.gpu (first 50 samples)...")
        p_gpu = predict_proba(model_gpu, X[:50], batch_size=256)
        gpu_ok = True
    except Exception as e:
        print(f"lightning.gpu unavailable or failed: {e}")
        gpu_ok = False

    # ── Run on default.qubit ────────────────────────────────────────
    print("\nBuilding model with default.qubit...")
    dev_cpu = qml.device("default.qubit", wires=int(cfg["n_qubits"]), shots=None)
    diff_cpu = "backprop"
    model_cpu = build_model(cfg, dev_cpu, diff_cpu)
    model_cpu.load_state_dict(ck["model_state"])
    model_cpu.eval()
    print("Running predict_proba on default.qubit (first 50 samples)...")
    p_cpu = predict_proba(model_cpu, X[:50], batch_size=256)

    # ── Compare ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    if gpu_ok:
        diff = np.abs(p_gpu - p_cpu)
        print(f"Max absolute difference:  {diff.max():.2e}")
        print(f"Mean absolute difference: {diff.mean():.2e}")
        print(f"\nFirst 10 probabilities side by side:")
        print(f"{'lightning.gpu':<18}{'default.qubit':<18}{'diff'}")
        for i in range(10):
            print(f"{p_gpu[i]:<18.8f}{p_cpu[i]:<18.8f}{diff[i]:.2e}")
        if diff.max() < 1e-4:
            print("\n✓ EQUIVALENT — backends agree to floating-point precision.")
            print("  The high entropy/ECE you observed is a genuine model")
            print("  property, NOT an artifact of the backend switch.")
        else:
            print("\n⚠️  DISCREPANCY FOUND — backends disagree beyond float")
            print("   precision. This needs investigation.")
    else:
        print(f"default.qubit probabilities (first 10): {p_cpu[:10]}")
        print("Could not compare — lightning.gpu unavailable on this run.")
    print("="*60)

if __name__ == "__main__":
    main()
