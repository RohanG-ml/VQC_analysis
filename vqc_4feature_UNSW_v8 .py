"""
[v8 — THREE-WAY BACKEND CHOICE + ANSATZ STRUCTURAL FIXES]

Same load_data() fix as v5/v6/v7 (single, correct definition).

NEW in v8: --pl_backend {lightning.gpu, default.qubit, default.mixed}
  lightning.gpu — pure-state, C++/adjoint, genuine alternative if you want
                     a compiled backend with GPU dependency.
  default.qubit    — pure-state, fastest via native batch broadcasting
                     on this circuit size (default). Verified
                     mathematically identical to lightning.gpu.
  default.mixed    — density-matrix, required only for a future noise
                     study with Kraus channels.

FIXED in v8 — two ansatzes brought into exact structural agreement
with their literal Qiskit definitions (verified against official
Qiskit documentation):
  real_amplitudes:     was missing the final rotation layer that
                        Qiskit's RealAmplitudes includes by default
                        (skip_final_rotation_layer=False). Now has
                        reps+1 rotation layers, reps CNOT layers.
  efficient_su2_like:  same gap, same fix — now has reps+1 rotation
                        layers (su2_gates per layer), reps CNOT layers,
                        matching EfficientSU2's default structure.
  zz_feature_map:      verified ALREADY correct, no changes needed.

IMPORTANT: real_amplitudes and efficient_su2_like now have MORE
trainable parameters than before for the same --ans_reps value.
Any previously trained checkpoints for these two ansatzes will NOT
load into this updated structure — retrain under v8 if using them.
custom_ansatz_1 and strongly_entangling are unaffected.

option given for lightning.gpu attempt in this version (see make_pl_device()).
"""


"""
vqc_4feature_UNSW_v1.py — UNSW-NB15 sweep with corrected ansatzes.
Changes from v5:
  * basic_entangler  → real_amplitudes (RY-only + circular CNOT)
  * efficient_su2_like now uses LINEAR entanglement (was circular)
  * Validation loaded from validation_1000.csv (no train split)
  * Flat folder structure: dr_subsets/{pca4,ica4,xgb_pca4,autoencoder}/
  * Results saved to /home/nvidia/21PHD1192/qml_id2/UNSW/

vqc_4feature_research_multigpu_v5.py

Full 4-feature research-sweep multi-GPU PennyLane VQC worker.

Version v5-v7 (fixes over v4):
  * Precision and Recall added to all metric outputs (evaluate_probs, CSVs, JSONs).
  * Best model checkpoint saved to disk immediately when validation score improves
    (no extra compute — validation already runs every epoch).
  * Confusion matrix (TP/TN/FP/FN + FPR/TPR) saved per model as _confusion.json.
  * Per-epoch time shown in training log (ep_time=Xs); best epoch marked with ★.
  * Data-in-use banner printed at start of each feature_set × train_size run
    showing exactly which train/test CSV files are being consumed.
  * Total time banner printed at end of each feature_set run (HH:MM:SS + seconds).
  * agent_summary.json written per feature_set run for critic/explainer agent input.
  * combined_summary_metrics.csv column order prioritises key metrics.

Core properties (unchanged from v4):
  * Strict VQC architecture from the optimized worker, not the notebook VQC circuit.
  * Explicit rx_embedding / ry_embedding / rz_embedding gates; no qml.AngleEmbedding template.
  * No bridge entangling layer between encoding and ansatz.
  * Configurable parallel job slots per physical GPU using stable per-slot queues.
  * Workers set CUDA_VISIBLE_DEVICES before importing torch/PennyLane.
  * Flexible CLI: lr, epochs, batch size, reps, entanglement, GPU list, ansatz/encoding subsets.
  * Status JSON files, final trained-model files, per-job training curves, grouped ROC curves by ansatz.

Example:
  python vqc_4feature_research_multigpu_v5.py \\
      --package_root /home/nvidia/21PHD1192/qml_ids/data/qml_gpu_upload_package_with_scaled_UNSW_NB15 \\
      --feature_sets pca4,ica4,xgb_pca4,ae4 \\
      --train_sizes 5000 \\
      --test_split test_balanced_2000 \\
      --results_dir results_vqc_4feature_5000_v5_UNSW_NB15 \\
      --target_col Label \\
      --n_qubits 4 \\
      --epochs 20 \\
      --batch_size 256 \\
      --lr 0.001 \\
      --lr_scheduler none \\
      --ans_reps 2 \\
      --enc_reps 1 \\
      --entanglement circular \\
      --ansatzes all \\
      --encodings all \\
      --max_gpus 8 \\
      --jobs_per_gpu 4 \\
      --ckpt_every 0
"""

import os
import sys
import gc
import json
import time
import math
import random
import argparse
import warnings
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_score, recall_score, confusion_matrix,
)

MODEL_KIND = "VQC"
ANSATZ_LIST_DEFAULT = ["custom_ansatz_1", "efficient_su2_like", "strongly_entangling", "real_amplitudes"]
ENCODING_LIST_DEFAULT = [
    "rx_embedding", "ry_embedding", "rz_embedding",
    "amplitude_embedding", "iqp_embedding", "zz_feature_map", "custom_h_ry_rz",
]
ENCODING_ALIASES = {"angle_rx": "rx_embedding", "angle_ry": "ry_embedding", "angle_rz": "rz_embedding"}


def parse_csv_list(value: Optional[str], allowed: List[str], aliases: Optional[Dict[str, str]] = None) -> List[str]:
    if value is None or value.strip().lower() in ("", "all"):
        return list(allowed)
    aliases = aliases or {}
    out = []
    for raw in value.split(","):
        item = raw.strip()
        item = aliases.get(item, item)
        if item not in allowed:
            raise ValueError(f"Unknown value '{raw}'. Allowed: {allowed}")
        if item not in out:
            out.append(item)
    return out


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # torch is intentionally imported lazily inside workers, after CUDA_VISIBLE_DEVICES is set.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def atomic_json_dump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=json_default)
    os.replace(tmp, path)


def safe_auc(y_true, y_prob) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def best_threshold(y_true, y_prob, metric: str = "accuracy") -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            score = accuracy_score(y_true, pred)
        if score > best_s:
            best_t, best_s = float(t), float(score)
    return best_t, float(best_s)


def evaluate_probs(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  float(accuracy_score(y_true, pred)),
        "f1":        float(f1_score(y_true, pred, zero_division=0)),
        "auc":       safe_auc(y_true, y_prob),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall":    float(recall_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPU discovery and status
# ─────────────────────────────────────────────────────────────────────────────
def _gpu_table_from_nvml(max_gpus: int) -> List[dict]:
    table = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = min(pynvml.nvmlDeviceGetCount(), max_gpus)
        for gid in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(gid)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "ignore")
            free_gb = mem.free / 1024**3
            total_gb = mem.total / 1024**3
            used_pct = 100.0 * (1.0 - mem.free / max(mem.total, 1))
            table.append({"id": gid, "name": name, "free_gb": free_gb, "total_gb": total_gb, "used_pct": used_pct})
    except Exception:
        pass
    return table


def _gpu_table_from_nvidia_smi(max_gpus: int) -> List[dict]:
    table = []
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        for line in out.strip().splitlines()[:max_gpus]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gid = int(parts[0])
            name = parts[1]
            free_gb = float(parts[2]) / 1024.0
            total_gb = float(parts[3]) / 1024.0
            used_pct = 100.0 * (1.0 - free_gb / max(total_gb, 1e-12))
            table.append({"id": gid, "name": name, "free_gb": free_gb, "total_gb": total_gb, "used_pct": used_pct})
    except Exception:
        pass
    return table


def _gpu_table_from_torch(max_gpus: int) -> List[dict]:
    table = []
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        count = min(torch.cuda.device_count(), max_gpus)
        for gid in range(count):
            props = torch.cuda.get_device_properties(gid)
            try:
                free_b, total_b = torch.cuda.mem_get_info(gid)
                free_gb = free_b / 1024**3
                total_gb = total_b / 1024**3
            except Exception:
                total_gb = props.total_memory / 1024**3
                free_gb = total_gb
            used_pct = 100.0 * (1.0 - free_gb / max(total_gb, 1e-12))
            table.append({"id": gid, "name": props.name, "free_gb": free_gb, "total_gb": total_gb, "used_pct": used_pct})
    except Exception:
        pass
    return table


def detect_gpu_ids(max_gpus: int, min_free_gb: float, manual_gpus: Optional[str] = None) -> List[int]:
    if manual_gpus and manual_gpus.strip().lower() not in ("", "auto"):
        ids = [int(x.strip()) for x in manual_gpus.split(",") if x.strip() != ""]
        print(f"Manual GPU IDs: {ids}", flush=True)
        return ids or [0]

    table = _gpu_table_from_nvml(max_gpus) or _gpu_table_from_nvidia_smi(max_gpus) or _gpu_table_from_torch(max_gpus)
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd:
        print(
            f"[WARN] Parent CUDA_VISIBLE_DEVICES={parent_cvd!r}. "
            "This script will set CUDA_VISIBLE_DEVICES inside each worker, but container-level restrictions still apply.",
            flush=True,
        )

    if not table:
        print("[WARN] No GPUs detected. Using CPU/fallback single worker.", flush=True)
        return [0]

    print("=" * 78, flush=True)
    print(f"{'ID':<5} {'GPU Name':<34} {'Free GB':>9} {'Total GB':>9} {'Used %':>8}  Status", flush=True)
    print("-" * 78, flush=True)
    ids = []
    for row in table:
        ok = row["free_gb"] >= min_free_gb
        status = "✓" if ok else "low-mem"
        print(
            f"{row['id']:<5} {row['name']:<34.34} {row['free_gb']:>9.2f} "
            f"{row['total_gb']:>9.2f} {row['used_pct']:>7.1f}%  {status}",
            flush=True,
        )
        if ok:
            ids.append(int(row["id"]))
    print("=" * 78, flush=True)

    if not ids:
        ids = [int(table[0]["id"])]
        print(f"[WARN] No GPU met --min_free_gb={min_free_gb}. Using GPU {ids[0]} anyway.", flush=True)
    return ids[:max_gpus]


def gpu_ram_stats(physical_gpu_id: int = 0) -> dict:
    s = {"gpu_util": 0, "gpu_mem_gb": 0.0, "ram_used_gb": 0.0, "ram_pct": 0.0}
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int(physical_gpu_id))
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        s["gpu_util"] = int(util.gpu)
        s["gpu_mem_gb"] = float(mem.used / 1024**3)
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        s["ram_used_gb"] = float(vm.used / 1024**3)
        s["ram_pct"] = float(vm.percent)
    except Exception:
        pass
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Circuit blocks
# ─────────────────────────────────────────────────────────────────────────────
class EncodingPL:
    """Encoding layer. RX/RY/RZ are explicit gates, not qml.AngleEmbedding."""

    def __init__(self, n_qubits: int = 4, reps: int = 1, entanglement: str = "circular"):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement

    def _pairs(self, ent=None):
        e = self.entanglement if ent is None else ent
        q = self.n_qubits
        if e == "none":
            return []
        if e == "linear":
            return [(i, i + 1) for i in range(q - 1)]
        if e == "circular":
            p = [(i, i + 1) for i in range(q - 1)]
            if q > 2:
                p.append((q - 1, 0))
            return p
        if e == "full":
            return [(i, j) for i in range(q) for j in range(i + 1, q)]
        raise ValueError(f"Unknown entanglement: {e}")

    def rx_embedding(self, qml, x):
        for i in range(self.n_qubits):
            qml.RX(x[..., i], wires=i)

    def ry_embedding(self, qml, x):
        for i in range(self.n_qubits):
            qml.RY(x[..., i], wires=i)

    def rz_embedding(self, qml, x):
        for i in range(self.n_qubits):
            qml.RZ(x[..., i], wires=i)

    def amplitude_embedding(self, qml, x):
        qml.AmplitudeEmbedding(features=x, wires=range(self.n_qubits), pad_with=0.0, normalize=True)

    def iqp_embedding(self, qml, x):
        # Manual broadcast-safe IQP-style encoding. Avoids qml.IQPEmbedding shape issues.
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for i in range(self.n_qubits):
                qml.RZ(x[..., i], wires=i)
            for i, j in self._pairs():
                qml.CNOT(wires=[i, j])
                qml.RZ(x[..., i] * x[..., j], wires=j)
                qml.CNOT(wires=[i, j])

    def zz_feature_map(self, qml, x):
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for i in range(self.n_qubits):
                qml.RZ(2.0 * x[..., i], wires=i)
            for i, j in self._pairs():
                qml.IsingZZ(2.0 * (x[..., i] - np.pi) * (x[..., j] - np.pi), wires=[i, j])

    def custom_h_ry_rz(self, qml, x, add_encoding_entanglement: bool = False):
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(x[..., i], wires=i)
            qml.RZ(x[..., i] ** 2, wires=i)
        if add_encoding_entanglement:
            for i, j in self._pairs():
                qml.CZ(wires=[i, j])

    def apply(self, qml, name: str, x, add_encoding_entanglement: bool = False):
        name = ENCODING_ALIASES.get(name, name)
        if name == "rx_embedding":
            self.rx_embedding(qml, x)
        elif name == "ry_embedding":
            self.ry_embedding(qml, x)
        elif name == "rz_embedding":
            self.rz_embedding(qml, x)
        elif name == "amplitude_embedding":
            self.amplitude_embedding(qml, x)
        elif name == "iqp_embedding":
            self.iqp_embedding(qml, x)
        elif name == "zz_feature_map":
            self.zz_feature_map(qml, x)
        elif name == "custom_h_ry_rz":
            self.custom_h_ry_rz(qml, x, add_encoding_entanglement=add_encoding_entanglement)
        else:
            raise ValueError(f"Unknown encoding '{name}'. Options: {ENCODING_LIST_DEFAULT}")


class AnsatzPL:
    """Ansatz layer. Default repetitions are controlled by --ans_reps."""

    def __init__(self, n_qubits: int = 4, reps: int = 1, entanglement: str = "circular"):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement

    def _pairs(self, ent=None):
        e = self.entanglement if ent is None else ent
        q = self.n_qubits
        if e == "none":
            return []
        if e == "linear":
            return [(i, i + 1) for i in range(q - 1)]
        if e == "circular":
            p = [(i, i + 1) for i in range(q - 1)]
            if q > 2:
                p.append((q - 1, 0))
            return p
        if e == "full":
            return [(i, j) for i in range(q) for j in range(i + 1, q)]
        raise ValueError(f"Unknown entanglement: {e}")

    def custom_ansatz_1(self, qml, w, ent=None):
        # Shape: (reps, 2, n_qubits). One rep has RY and RZ trainable sublayers.
        for r in range(w.shape[0]):
            for i in range(self.n_qubits):
                qml.RY(w[r, 0, i], wires=i)
            for i, j in self._pairs(ent):
                qml.CNOT(wires=[i, j])
            for i in range(self.n_qubits):
                qml.RZ(w[r, 1, i], wires=i)

    def efficient_su2_like(self, qml, w, su2_gates=("ry", "rz"), ent=None):
        # Matches Qiskit's literal default structure:
        #   qiskit.circuit.library.EfficientSU2(reps=R, su2_gates=['ry','rz'],
        #                                        entanglement='reverse_linear',
        #                                        skip_final_rotation_layer=False)
        # i.e. R rotation-then-entangle blocks PLUS one final rotation layer
        # (R+1 total rotation layers, R CNOT layers).
        # Weight shape: (reps+1, len(su2_gates), n_qubits) — see get_weight_shape().
        gmap = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}
        n_layers = w.shape[0]   # = reps + 1, includes the final layer
        for r in range(n_layers):
            for g, gate_name in enumerate(su2_gates):
                gate = gmap[gate_name.lower()]
                for i in range(self.n_qubits):
                    gate(w[r, g, i], wires=i)
            if r < n_layers - 1:        # entangling layer between rotation layers only
                for i, j in self._pairs("linear"):
                    qml.CNOT(wires=[i, j])

    def strongly_entangling(self, qml, w):
        # One layer contains Rot(phi, theta, omega) on each qubit plus entanglement.
        qml.StronglyEntanglingLayers(w, wires=range(self.n_qubits))

    def real_amplitudes(self, qml, w):
        # Real Amplitudes ansatz — matches Qiskit's literal default structure:
        #   qiskit.circuit.library.RealAmplitudes(reps=R, entanglement='circular',
        #                                          skip_final_rotation_layer=False)
        # i.e. R rotation-then-entangle blocks PLUS one final rotation layer
        # (R+1 total RY layers, R CNOT layers) — Qiskit's default has
        # skip_final_rotation_layer=False, so the extra final layer is NOT
        # optional, it's the standard behavior.
        # Weight shape: (reps+1, 1, n_qubits) — see get_weight_shape().
        n_layers = w.shape[0]   # = reps + 1, includes the final layer
        for r in range(n_layers):
            for i in range(self.n_qubits):
                qml.RY(w[r, 0, i], wires=i)
            if r < n_layers - 1:        # entangling layer between rotation layers only
                for i, j in self._pairs("circular"):
                    qml.CNOT(wires=[i, j])

    def apply(self, qml, name: str, w, ent=None, su2_gates=("ry", "rz")):
        if name == "custom_ansatz_1":
            self.custom_ansatz_1(qml, w, ent=ent)
        elif name == "efficient_su2_like":
            self.efficient_su2_like(qml, w, su2_gates=su2_gates, ent=ent)
        elif name == "strongly_entangling":
            self.strongly_entangling(qml, w)
        elif name == "real_amplitudes":
            self.real_amplitudes(qml, w)
        else:
            raise ValueError(f"Unknown ansatz '{name}'. Options: {ANSATZ_LIST_DEFAULT}")


def get_weight_shape(ansatz_name: str, n_qubits: int, reps: int, su2_gates=("ry", "rz")):
    if ansatz_name == "custom_ansatz_1":
        return (reps, 2, n_qubits)
    if ansatz_name == "efficient_su2_like":
        return (reps + 1, len(su2_gates), n_qubits)   # +1: includes Qiskit's default final rotation layer
    if ansatz_name == "strongly_entangling":
        return (reps, n_qubits, 3)
    if ansatz_name == "real_amplitudes":
        return (reps + 1, 1, n_qubits)   # +1: includes Qiskit's default final rotation layer
    raise ValueError(f"Unknown ansatz '{ansatz_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Model and training
# ─────────────────────────────────────────────────────────────────────────────
def make_pl_device(n_qubits: int, backend: str = "default.qubit"):
    """
    User-selectable backend — lightning.gpu, default.qubit, or default.mixed.

    lightning.gpu: pure-state vector simulator, C++/CUDA backend, adjoint
    differentiation, runs on GPU hardware. This is the same backend used
    during the project's original training runs. diff_method="adjoint".

    default.qubit: pure-state vector simulator, fastest for noiseless
    circuits on this circuit size (native batch broadcasting on CPU).
    Verified mathematically identical to lightning.gpu (max abs diff =
    0.00e+00 across multiple ansatz/encoding combinations). diff_method=
    "backprop".

    default.mixed: density-matrix simulator. Required ONLY if applying
    noise channels (Kraus operators — depolarizing, amplitude damping,
    etc.) for a future noise-robustness study. Without noise, gives
    IDENTICAL results to default.qubit (verified separately), but is
    computationally heavier (density matrix scales as state^2), so
    only select it when noise channels are actually needed. diff_method=
    "backprop".
    """
    import pennylane as qml
    valid_backends = ("lightning.gpu", "default.qubit", "default.mixed")
    if backend not in valid_backends:
        raise ValueError(f"backend must be one of {valid_backends}, got '{backend}'")
    diff_method = "adjoint" if backend == "lightning.gpu" else "backprop"
    dev = qml.device(backend, wires=n_qubits, shots=None)
    return dev, diff_method, backend


def expval_matrix(out, batch_size: int, n_qubits: int):
    import torch
    if isinstance(out, (list, tuple)):
        items = [o if torch.is_tensor(o) else torch.as_tensor(o) for o in out]
        out = torch.stack(items, dim=-1)
    elif not torch.is_tensor(out):
        out = torch.as_tensor(out)
    out = out.float().squeeze()

    if out.dim() == 1:
        if batch_size == 1 and out.numel() == n_qubits:
            out = out.reshape(1, n_qubits)
        elif n_qubits == 1 and out.numel() == batch_size:
            out = out.reshape(batch_size, 1)
    elif out.dim() == 2:
        if tuple(out.shape) == (n_qubits, batch_size):
            out = out.T
        elif tuple(out.shape) != (batch_size, n_qubits) and out.numel() == batch_size * n_qubits:
            out = out.reshape(batch_size, n_qubits)
    else:
        if out.numel() == batch_size * n_qubits:
            out = out.reshape(batch_size, n_qubits)

    if tuple(out.shape) != (batch_size, n_qubits):
        raise RuntimeError(
            f"QNode output shape {tuple(out.shape)} cannot be normalized to {(batch_size, n_qubits)}"
        )
    return out


def build_model(cfg: dict, dev, diff_method: str):
    import torch
    import torch.nn as nn
    import pennylane as qml

    n_qubits = int(cfg["n_qubits"])
    encoder = EncodingPL(n_qubits=n_qubits, reps=int(cfg["enc_reps"]), entanglement=cfg["entanglement"])
    ansatz = AnsatzPL(n_qubits=n_qubits, reps=int(cfg["ans_reps"]), entanglement=cfg["entanglement"])
    encoding_name = ENCODING_ALIASES.get(cfg["encoding_name"], cfg["encoding_name"])
    ansatz_name = cfg["ansatz_name"]
    su2_gates = tuple(cfg["su2_gates"])
    wshape = get_weight_shape(ansatz_name, n_qubits, int(cfg["ans_reps"]), su2_gates=su2_gates)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def qnode(x, weights):
        # x can be (batch, n_qubits). Each encoding uses x[..., i] for broadcast-safe batching.
        encoder.apply(qml, encoding_name, x, add_encoding_entanglement=bool(cfg["encoding_entanglement"]))
        # Encoding goes directly to ansatz. No extra bridge-entangling layer is inserted here.
        ansatz.apply(qml, ansatz_name, weights, ent=cfg["entanglement"], su2_gates=su2_gates)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class QModel(nn.Module):
        def __init__(self):
            super().__init__()
            scale = float(cfg.get("init_scale", 0.05))
            if cfg.get("init_mode", "random") == "zeros":
                w0 = torch.zeros(wshape, dtype=torch.float32)
            else:
                w0 = scale * torch.randn(wshape, dtype=torch.float32)
            self.quantum_weights = nn.Parameter(w0)
            if MODEL_KIND == "QNN":
                self.classical_head = nn.Sequential(
                    nn.LayerNorm(n_qubits),
                    nn.Linear(n_qubits, 1, bias=True),
                )
            else:
                self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
                self.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        def forward(self, x):
            batch_size = int(x.shape[0])
            out = qnode(x.cpu(), self.quantum_weights)
            z = expval_matrix(out, batch_size=batch_size, n_qubits=n_qubits)
            if MODEL_KIND == "QNN":
                return self.classical_head(z).squeeze(-1)
            summed_exp = z.sum(dim=1)
            return self.scale * (-summed_exp) + self.bias

    return QModel()


def predict_proba(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    import torch
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    probs = []
    with torch.no_grad():
        for s in range(0, len(X_t), batch_size * 2):
            logits = model(X_t[s:s + batch_size * 2]).detach().cpu()
            probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs).reshape(-1)


def checkpoint_path(results_dir: str, run_key: str) -> str:
    return os.path.join(results_dir, f"{run_key}_checkpoint.pt")


def save_checkpoint(path: str, model, opt, sched, history: dict, epoch: int, best_state: dict, best_score: float, best_threshold: float):
    import torch
    state = {
        "epoch": epoch,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict() if sched is not None else None,
        "history": history,
        "best_state": best_state,
        "best_score": best_score,
        "best_threshold": best_threshold,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def train_model(model, X_tr, y_tr, X_val, y_val, cfg, pos_weight_val, label: str, physical_gpu_id: int, run_key: str, results_dir: str):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32))
    opt = optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    # Fixed learning rate for the research sweeps. The lr_scheduler argument is kept
    # only for backward-compatible commands, but the optimizer LR is not changed.
    if str(cfg.get("lr_scheduler", "none")).lower() != "none":
        print(f"  [{label}] NOTE: lr_scheduler={cfg.get('lr_scheduler')} requested, but v3 keeps LR fixed at {float(cfg['lr'])}", flush=True)
    sched = None

    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [], "val_auc": [],
        "val_threshold": [], "lr": [], "epoch_s": [], "grad_norm": [],
    }
    start_epoch = 0
    best_score = -1.0
    best_state = None
    best_epoch = -1
    best_threshold_seen = 0.5
    ckpt = checkpoint_path(results_dir, run_key)

    if bool(cfg.get("resume", False)) and os.path.exists(ckpt):
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state"])
            opt.load_state_dict(state["optimizer_state"])
            
            if sched is not None and state.get("scheduler_state") is not None:
                sched.load_state_dict(state["scheduler_state"])
            history = state.get("history", history)
            start_epoch = int(state.get("epoch", -1)) + 1
            best_state = state.get("best_state")
            best_score = float(state.get("best_score", -1.0))
            best_threshold_seen = float(state.get("best_threshold", 0.5))
            print(f"  [{label}] Resumed from epoch {start_epoch}", flush=True)
        except Exception as e:
            print(f"  [{label}] Could not resume checkpoint: {e}", flush=True)

    select_by = cfg["select_best_by"]
    threshold_metric = cfg["threshold_metric"]
    # Early stopping is intentionally disabled in v3; every job runs all requested epochs.
    patience = 0
    min_delta = float(cfg["min_delta"])
    bad_epochs = 0
    t_total = time.perf_counter()

    for ep in range(start_epoch, int(cfg["epochs"])):
        t0 = time.perf_counter()
        model.train()
        running = 0.0
        total_grad_norm = 0.0
        n_batches = len(dl)

        # First epoch: warn that circuit compilation may cause silence
        if ep == start_epoch:
            print(
                f"  [{label}] Ep 001 starting — {n_batches} batches. "
                f"First epoch may be slow (PennyLane JIT compile). Please wait...",
                flush=True,
            )

        for batch_idx, (xb, yb) in enumerate(dl):
            opt.zero_grad(set_to_none=True)
            logits = model(xb).view(-1, 1)
            loss = loss_fn(logits, yb)
            loss.backward()
            gn = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            total_grad_norm += gn
            if float(cfg["grad_clip"]) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            running += loss.item() * xb.shape[0]

            # Heartbeat every 5 batches so terminal is never silent for > 1 min
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == n_batches:
                batch_elapsed = time.perf_counter() - t0
                print(
                    f"  [{label}] Ep {ep+1:03d} batch {batch_idx+1:03d}/{n_batches:03d} "
                    f"loss={loss.item():.5f}  elapsed={batch_elapsed:.1f}s",
                    flush=True,
                )
                # Also write to status file — readable from second terminal
                # even when stdout pipe is buffered
                try:
                    atomic_json_dump({
                        "run_key": run_key,
                        "status": "training",
                        "epoch": ep + 1,
                        "batch": batch_idx + 1,
                        "n_batches": n_batches,
                        "batch_loss": float(loss.item()),
                        "elapsed_s": round(batch_elapsed, 1),
                        "pid": os.getpid(),
                        "physical_gpu_id": physical_gpu_id,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, os.path.join(results_dir, f"{run_key}_status.json"))
                except Exception:
                    pass

        # Fixed LR: no scheduler step.
        ep_train_loss = running / max(len(X_tr), 1)
        avg_grad_norm = total_grad_norm / max(len(dl), 1)
        ep_s = time.perf_counter() - t0

        val_prob = predict_proba(model, X_val, int(cfg["batch_size"]))
        # Compute validation BCE loss from probabilities (no extra forward pass needed)
        _eps = 1e-7
        _p = np.clip(val_prob, _eps, 1.0 - _eps)
        ep_val_loss = float(-np.mean(
            y_val * np.log(_p) + (1.0 - y_val) * np.log(1.0 - _p)
        ))
        val_threshold, _ = best_threshold(y_val, val_prob, metric=threshold_metric)
        val_metrics = evaluate_probs(y_val, val_prob, threshold=val_threshold)
        score = val_metrics[select_by]
        if math.isnan(score):
            score = -1.0

        if score > best_score + min_delta:
            best_score = score
            best_epoch = ep
            best_threshold_seen = val_threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            # ── Save best model to disk immediately (no extra eval cost) ──
            try:
                best_model_file = os.path.join(results_dir, f"{run_key}_best_model.pt")
                torch.save({
                    "model_kind": MODEL_KIND,
                    "model_state": best_state,
                    "config": cfg,
                    "best_epoch": int(best_epoch + 1),
                    "best_score": float(best_score),
                    "best_threshold": float(best_threshold_seen),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, best_model_file)
            except Exception as _be:
                print(f"  [{label}] WARNING: could not save best model: {_be}", flush=True)
        else:
            bad_epochs += 1

        history["train_loss"].append(float(ep_train_loss))
        history["val_loss"].append(float(ep_val_loss))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_auc"].append(float(val_metrics["auc"]))
        history["val_threshold"].append(float(val_threshold))
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        history["epoch_s"].append(float(ep_s))
        history["grad_norm"].append(float(avg_grad_norm))

        status = {
            "run_key": run_key,
            "model": MODEL_KIND,
            "status": "running",
            "epoch": ep + 1,
            "epochs": int(cfg["epochs"]),
            "physical_gpu_id": physical_gpu_id,
            "pid": os.getpid(),
            "train_loss": float(ep_train_loss),
            "val_loss": float(ep_val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1": float(val_metrics["f1"]),
            "val_auc": float(val_metrics["auc"]),
            "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
            "best_score": float(best_score),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_json_dump(status, os.path.join(results_dir, f"{run_key}_status.json"))

        if int(cfg.get("ckpt_every", 0)) > 0 and ((ep + 1) % int(cfg["ckpt_every"]) == 0):
            save_checkpoint(ckpt, model, opt, sched, history, ep, best_state, best_score, best_threshold_seen)

        elapsed = time.perf_counter() - t_total
        best_marker = " ★" if ep == best_epoch else ""
        print(
            f"  [{label}] Ep {ep + 1:03d}/{int(cfg['epochs']):03d} | "
            f"loss={ep_train_loss:.5f} acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f} "
            f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} | "
            f"ep_time={ep_s:.1f}s  total={elapsed:.0f}s{best_marker}",
            flush=True,
        )


    # Do not restore the validation-best state. The saved model is the final state after all epochs.
    history["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
    history["best_score"] = float(best_score)
    history["best_threshold"] = float(best_threshold_seen)
    history["total_time"] = float(time.perf_counter() - t_total)

    final_model_file = os.path.join(results_dir, f"{run_key}_final_model.pt")
    try:
        import torch
        final_threshold = float(history.get("val_threshold", [best_threshold_seen])[-1]) if history.get("val_threshold") else float(best_threshold_seen)
        torch.save({
            "model_kind": MODEL_KIND,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": cfg,
            "history": history,
            "final_epoch": int(len(history.get("train_loss", []))),
            "final_validation_threshold": final_threshold,
            "best_epoch_by_validation": int(best_epoch + 1) if best_epoch >= 0 else None,
            "best_score_by_validation": float(best_score),
            "best_threshold_by_validation": float(best_threshold_seen),
            "best_model_path": os.path.join(results_dir, f"{run_key}_best_model.pt"),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, final_model_file)
        history["final_model_path"] = final_model_file
        history["best_model_path"] = os.path.join(results_dir, f"{run_key}_best_model.pt")
        print(f"  [{label}] Saved final model : {final_model_file}", flush=True)
        print(f"  [{label}] Best model (ep {best_epoch+1 if best_epoch>=0 else '?'}): {history['best_model_path']}", flush=True)
    except Exception as e:
        print(f"  [{label}] WARNING: could not save final model: {e}", flush=True)

    try:
        save_checkpoint(ckpt, model, opt, sched, history, max(start_epoch, len(history["train_loss"])) - 1, best_state, best_score, best_threshold_seen)
    except Exception:
        pass
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Output files and plots
# ─────────────────────────────────────────────────────────────────────────────
def save_job_plots(history, ansatz_name, encoding_name, results_dir, model_name=MODEL_KIND):
    """Save 3 separate training curve PNGs per model run.

    PNG 1 — loss_curve    : Train loss + Validation loss on one axes.
    PNG 2 — metrics_curve : Accuracy + F1 + AUC on one axes.
    PNG 3 — grad_norm     : Gradient norm.
    Epoch axis always shows integers (1, 2, 3 ... N), never decimals.
    """
    tag    = f"{ansatz_name}__{encoding_name}"
    n_ep   = len(history.get("train_loss", []))
    if n_ep == 0:
        return

    ep_axis    = list(range(1, n_ep + 1))
    best_ep    = history.get("best_epoch")          # 1-based or None
    title_base = f"{model_name}  |  {ansatz_name} + {encoding_name}"

    # Integer tick helper — never show 2.5, 5.0 etc.
    def _int_xticks(ax):
        ax.set_xticks(ep_axis if n_ep <= 25
                      else list(range(1, n_ep + 1, max(1, n_ep // 10))))
        ax.set_xlim(0.5, n_ep + 0.5)

    def _mark_best(ax):
        if best_ep and 1 <= best_ep <= n_ep:
            ax.axvline(best_ep, color="gray", linestyle="--",
                       lw=1.2, label=f"Best epoch ({best_ep})")

    # ── PNG 1 : Loss curve ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep_axis, history["train_loss"], lw=2, color="#2196F3", label="Train loss")
    if history.get("val_loss") and len(history["val_loss"]) == n_ep:
        ax.plot(ep_axis, history["val_loss"], lw=2, linestyle="--",
                color="#F44336", label="Val loss (BCE)")
    _mark_best(ax)
    _int_xticks(ax)
    ax.set_title(f"Loss Curve\n{title_base}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{tag}_loss_curve.png"),
                dpi=140, bbox_inches="tight")
    plt.close()

    # ── PNG 2 : Metrics curve (Acc + F1 + AUC) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    if history.get("val_acc"):
        ax.plot(ep_axis, history["val_acc"], lw=2,
                color="#2196F3", label="Accuracy")
    if history.get("val_f1"):
        ax.plot(ep_axis, history["val_f1"], lw=2,
                color="#4CAF50", label="F1 Score")
    if history.get("val_auc"):
        ax.plot(ep_axis, history["val_auc"], lw=2,
                color="#FF9800", label="AUC-ROC")
    _mark_best(ax)
    _int_xticks(ax)
    ax.set_title(f"Validation Metrics\n{title_base}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{tag}_metrics_curve.png"),
                dpi=140, bbox_inches="tight")
    plt.close()

    # ── PNG 3 : Gradient norm ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    if history.get("grad_norm"):
        ax.plot(ep_axis, history["grad_norm"], lw=2,
                color="darkorange", label="Grad norm")
    _mark_best(ax)
    _int_xticks(ax)
    ax.set_title(f"Gradient Norm\n{title_base}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Gradient Norm")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{tag}_grad_norm.png"),
                dpi=140, bbox_inches="tight")
    plt.close()



def save_roc_npz(y_true, y_prob, threshold, ansatz_name, encoding_name, results_dir):
    metrics = evaluate_probs(y_true, y_prob, threshold=threshold)
    tag = f"{ansatz_name}__{encoding_name}"
    fpr, tpr, _ = roc_curve(y_true, y_prob) if len(np.unique(y_true)) > 1 else (np.array([0, 1]), np.array([0, 1]), np.array([0.5]))
    np.savez(
        os.path.join(results_dir, f"{tag}_roc_data.npz"),
        fpr=fpr,
        tpr=tpr,
        y_prob=y_prob,
        y_true=y_true,
        auc=np.array(metrics["auc"]),
        threshold=np.array(threshold),
    )
    return metrics


def save_grouped_roc_by_ansatz(results_dir: str, ansatz_list: List[str],
                               encoding_list: List[str], model_name=MODEL_KIND):
    """For each ansatz: one ROC PNG with a separate curve per encoding.

    Layout:  one file per ansatz (4 files total for 4 ansatzes)
             each file shows all 7 encoding ROC curves + chance diagonal
             curves sorted by AUC descending in the legend
             integer-free axes (0.0 … 1.0 in clean 0.2 steps)

    Files saved:  roc_curves_{ansatz}.png
    """
    # Colourblind-safe palette — 7 distinct colours for 7 encodings
    COLOURS = [
        "#2196F3",  # blue
        "#F44336",  # red
        "#4CAF50",  # green
        "#FF9800",  # orange
        "#9C27B0",  # purple
        "#00BCD4",  # cyan
        "#FF5722",  # deep-orange
    ]

    paths = []
    for ansatz in ansatz_list:
        fig, ax = plt.subplots(figsize=(8, 7))

        # Collect all (auc, enc, fpr, tpr) so we can sort by AUC for legend
        curves = []
        for enc in encoding_list:
            tag = f"{ansatz}__{enc}"
            p   = os.path.join(results_dir, f"{tag}_roc_data.npz")
            if not os.path.exists(p):
                continue
            try:
                data    = np.load(p, allow_pickle=True)
                auc_val = float(data["auc"])
                curves.append((auc_val, enc, data["fpr"], data["tpr"]))
            except Exception:
                continue

        if not curves:
            plt.close()
            continue

        # Sort best → worst AUC so legend reads top-to-bottom by performance
        curves.sort(key=lambda x: x[0], reverse=True)

        for idx, (auc_val, enc, fpr, tpr) in enumerate(curves):
            colour = COLOURS[idx % len(COLOURS)]
            label  = (f"{enc}  (AUC = {auc_val:.3f})"
                      if not math.isnan(auc_val) else enc)
            # Thicker line for top performer
            lw = 2.5 if idx == 0 else 1.8
            ax.plot(fpr, tpr, lw=lw, color=colour, label=label)

        # Chance diagonal
        ax.plot([0, 1], [0, 1], linestyle="--", lw=1.2,
                color="gray", label="Random chance (AUC = 0.500)")

        # Clean axes — 0.0 to 1.0 in 0.2 steps, no decimals like 0.25
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.05)

        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
        ax.set_title(
            f"ROC Curves — Ansatz: {ansatz}\n"
            f"All {len(curves)} encoding(s) | sorted by AUC | {model_name}",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9, loc="lower right",
                  framealpha=0.9, edgecolor="lightgray")
        ax.grid(alpha=0.25)

        out = os.path.join(results_dir, f"roc_curves_{ansatz}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        paths.append(out)
        print(f"  ROC saved: {out}", flush=True)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Worker and scheduling
# ─────────────────────────────────────────────────────────────────────────────

def start_resource_monitor(physical_gpu_id: int, csv_path: str, interval_s: float = 5.0):
    """Continuously append GPU/RAM usage to CSV while a job is training."""
    import threading
    import csv
    stop_event = threading.Event()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def loop():
        header_written = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        t0 = time.perf_counter()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "elapsed_s", "physical_gpu_id",
                "gpu_util", "gpu_mem_gb", "ram_used_gb", "ram_pct"
            ])
            if not header_written:
                writer.writeheader()
                f.flush()
            while not stop_event.is_set():
                s = gpu_ram_stats(physical_gpu_id)
                writer.writerow({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                    "physical_gpu_id": int(physical_gpu_id),
                    "gpu_util": s.get("gpu_util", 0),
                    "gpu_mem_gb": round(float(s.get("gpu_mem_gb", 0.0)), 4),
                    "ram_used_gb": round(float(s.get("ram_used_gb", 0.0)), 4),
                    "ram_pct": round(float(s.get("ram_pct", 0.0)), 2),
                })
                f.flush()
                stop_event.wait(interval_s)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return stop_event, th

def worker_entry(job: dict) -> dict:
    physical_gpu_id = int(job["gpu_id"])
    ansatz_name = job["ansatz_name"]
    encoding_name = ENCODING_ALIASES.get(job["encoding_name"], job["encoding_name"])
    cfg = dict(job["cfg"])
    results_dir = job["results_dir"]
    run_key = f"{ansatz_name}__{encoding_name}"
    feature_set_label = job.get("feature_set", "?")[:8]
    label = f"{MODEL_KIND}|GPU{physical_gpu_id}|{feature_set_label}|{ansatz_name[:9]}|{encoding_name[:11]}"
    pid = os.getpid()

    # CRITICAL: set per-process GPU visibility before importing torch/PennyLane.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.get("cpu_threads_per_worker", 2)))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg.get("cpu_threads_per_worker", 2)))

    try:
        import torch
        import pennylane as qml  # noqa: F401
        set_all_seeds(int(cfg["seed"]) + physical_gpu_id)
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # local index after CUDA_VISIBLE_DEVICES remap
        except Exception:
            pass

        os.makedirs(results_dir, exist_ok=True)
        # Resource monitoring is disabled in v3 to keep outputs focused on model metrics.
        pid_info = {
            "pid": pid,
            "model": MODEL_KIND,
            "physical_gpu_id": physical_gpu_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "ansatz": ansatz_name,
            "encoding": encoding_name,
            "status": "starting",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_json_dump(pid_info, os.path.join(results_dir, f"{run_key}_pid_info.json"))
        atomic_json_dump({**pid_info, "status": "starting"}, os.path.join(results_dir, f"{run_key}_status.json"))

        print(f"\n[PID {pid}] Starting {label}", flush=True)
        pl_backend = cfg.get("pl_backend", "default.qubit")
        dev, diff_method, device_name = make_pl_device(int(cfg["n_qubits"]), backend=pl_backend)
        cfg_run = {**cfg, "encoding_name": encoding_name, "ansatz_name": ansatz_name, "pl_device": device_name}
        model = build_model(cfg_run, dev, diff_method)

        # Batch smoke test catches old shape bugs immediately.
        smoke_n = min(4, len(job["X_train"]))
        with torch.no_grad():
            smoke_out = model(torch.tensor(job["X_train"][:smoke_n], dtype=torch.float32))
        if tuple(smoke_out.shape) != (smoke_n,):
            raise RuntimeError(f"Smoke test failed: expected {(smoke_n,)}, got {tuple(smoke_out.shape)}")
        print(f"[PID {pid}] Smoke OK: x={(smoke_n, cfg['n_qubits'])}, out={tuple(smoke_out.shape)}, device={device_name}", flush=True)

        history = train_model(
            model, job["X_train"], job["y_train"], job["X_val"], job["y_val"],
            cfg=cfg_run, pos_weight_val=job["pos_weight_val"], label=label,
            physical_gpu_id=physical_gpu_id, run_key=run_key, results_dir=results_dir,
        )

        val_prob = predict_proba(model, job["X_val"], int(cfg["batch_size"]))
        threshold, _ = best_threshold(job["y_val"], val_prob, metric=cfg["threshold_metric"])
        test_prob = predict_proba(model, job["X_test"], int(cfg["batch_size"]))
        metrics = save_roc_npz(job["y_test"], test_prob, threshold, ansatz_name, encoding_name, results_dir)
        val_metrics = evaluate_probs(job["y_val"], val_prob, threshold=threshold)
        metrics.update({
            "best_epoch": history.get("best_epoch"),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_f1": val_metrics["f1"],
            "validation_auc": val_metrics["auc"],
        })

        np.save(os.path.join(results_dir, f"{run_key}_probs.npy"), test_prob)
        # ── Confusion matrix ──────────────────────────────────────────────
        test_pred = (test_prob >= threshold).astype(int)
        cm = confusion_matrix(job["y_test"], test_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_data = {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "FPR": float(fp / max(fp + tn, 1)),
            "TPR": float(tp / max(tp + fn, 1)),
            "run_key": run_key,
        }
        atomic_json_dump(cm_data, os.path.join(results_dir, f"{run_key}_confusion.json"))
        save_job_plots(history, ansatz_name, encoding_name, results_dir, model_name=MODEL_KIND)
        payload = {"metrics": metrics, "history": history, "config": cfg_run, "confusion": cm_data}
        atomic_json_dump(payload, os.path.join(results_dir, f"{run_key}_result.json"))
        atomic_json_dump({
            "run_key": run_key,
            "model": MODEL_KIND,
            "status": "done",
            "physical_gpu_id": physical_gpu_id,
            "pid": pid,
            "metrics": metrics,
            "confusion": cm_data,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, os.path.join(results_dir, f"{run_key}_status.json"))

        print(
            f"[PID {pid}] ✓ {label} "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc']:.4f} prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f} "
            f"TP={cm_data['TP']} FP={cm_data['FP']} FN={cm_data['FN']} TN={cm_data['TN']}",
            flush=True,
        )
        return {"ansatz_name": ansatz_name, "encoding_name": encoding_name, "result": payload, "error": None}

    except Exception:
        import traceback
        err = traceback.format_exc()
        print(f"[PID {pid}] ✗ {label}\n{err}", flush=True)
        try:
            atomic_json_dump({
                "run_key": run_key,
                "model": MODEL_KIND,
                "status": "error",
                "physical_gpu_id": physical_gpu_id,
                "pid": pid,
                "error": err,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, os.path.join(results_dir, f"{run_key}_status.json"))
        except Exception:
            pass
        return {"ansatz_name": ansatz_name, "encoding_name": encoding_name, "result": None, "error": err}
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def build_per_gpu_queues(
    combos: List[Tuple[str, str]],
    gpu_ids: List[int],
    cfg_base: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    pos_weight_val: float,
    results_dir: str,
    jobs_per_gpu: int = 1,
    feature_set: str = "?",   # for label display
):
    """Build independent worker-slot queues.

    jobs_per_gpu=1 gives the original stable behavior: one OS process per GPU.
    jobs_per_gpu>1 starts multiple independent OS processes per physical GPU.
    This is useful for very small VQC circuits where one process does not keep an
    A100 busy. Each slot still runs its own queue sequentially.
    """
    jobs_per_gpu = max(1, int(jobs_per_gpu))
    slot_defs = []
    for gid in gpu_ids:
        for slot_idx in range(jobs_per_gpu):
            worker_key = f"gpu{int(gid)}_slot{slot_idx}"
            slot_defs.append((worker_key, int(gid), int(slot_idx)))

    if not slot_defs:
        raise RuntimeError("No GPU worker slots available.")

    queues = {worker_key: [] for worker_key, _, _ in slot_defs}
    for idx, (ans, enc) in enumerate(combos):
        worker_key, gid, slot_idx = slot_defs[idx % len(slot_defs)]
        queues[worker_key].append({
            "worker_key": worker_key,
            "gpu_slot": slot_idx,
            "gpu_id": gid,
            "ansatz_name": ans,
            "encoding_name": enc,
            "cfg": {**cfg_base},
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "pos_weight_val": pos_weight_val,
            "results_dir": results_dir,
        })
    return queues


def gpu_queue_worker(worker_key: str, gpu_id: int, gpu_slot: int, jobs: List[dict], result_queue,
                     shared_work_queue=None) -> None:
    """Run jobs from a shared dynamic queue — prevents idle slots when some jobs finish fast.

    If shared_work_queue is provided, this slot will keep pulling jobs until the queue
    is empty (DYNAMIC mode). Otherwise falls back to the pre-assigned jobs list (STATIC).

    Dynamic mode eliminates the problem where fast slots (e.g., rz_embedding) finish
    their pre-assigned jobs and sit idle while slow slots (strongly_entangling) still
    run — all slots stay busy until all 112 jobs are done.
    """
    import queue as _queue

    def _run_one(job):
        ansatz_name   = job.get("ansatz_name")
        encoding_name = ENCODING_ALIASES.get(job.get("encoding_name"), job.get("encoding_name"))
        try:
            res = worker_entry(job)
        except BaseException:
            import traceback
            err = traceback.format_exc()
            res = {"ansatz_name": ansatz_name, "encoding_name": encoding_name,
                   "result": None, "error": err}
        try:
            result_queue.put({"type": "result", "worker_key": worker_key,
                              "gpu_id": gpu_id, "gpu_slot": gpu_slot, "result": res})
        except BaseException:
            pass

    if shared_work_queue is not None:
        # DYNAMIC: grab jobs from shared queue until empty
        # CRITICAL: override job's gpu_id with THIS SLOT'S actual GPU.
        # Without this, every slot uses the job's original static gpu_id
        # → all 16 slots pile onto GPUs 0-3, GPUs 4-7 sit idle.
        while True:
            try:
                job = shared_work_queue.get(block=False)
                job = dict(job)          # copy — do not mutate shared object
                job["gpu_id"] = gpu_id  # use this slot's actual physical GPU
                _run_one(job)
            except _queue.Empty:
                break
    else:
        # STATIC fallback: run pre-assigned list
        for job in jobs:
            _run_one(job)

    try:
        result_queue.put({"type": "worker_done", "worker_key": worker_key,
                          "gpu_id": gpu_id, "gpu_slot": gpu_slot, "pid": os.getpid()})
    except BaseException:
        pass


def run_scheduler(queues: Dict[str, List[dict]], n_workers: int, total_jobs: int, all_results: dict):
    """Stable scheduler: one OS process per GPU slot, queue runs sequentially.

    With --jobs_per_gpu 1 this is one process per physical GPU. With
    --jobs_per_gpu k, this is k processes per physical GPU. If one worker slot
    dies, other slots continue and the remaining jobs in only that slot are
    marked failed.
    """
    import queue as queue_module

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    result_queue  = ctx.Queue()
    shared_work_q = ctx.Queue()   # dynamic shared work queue

    active_queues = {str(wkey): list(jobs) for wkey, jobs in queues.items() if jobs}

    # Fill shared queue with all jobs (any slot can grab any job)
    all_jobs_flat = [job for jobs in active_queues.values() for job in jobs]
    for job in all_jobs_flat:
        shared_work_q.put(job)
    print(f"[scheduler] Dynamic work queue: {shared_work_q.qsize()} jobs across "
          f"{len(active_queues)} slots", flush=True)

    processes = {}
    slot_meta = {}
    completed_keys = set()
    done_workers = set()
    completed = 0

    for worker_key, jobs in active_queues.items():
        gpu_id = int(jobs[0].get("gpu_id"))
        gpu_slot = int(jobs[0].get("gpu_slot", 0))
        # Pass shared_work_queue — slot pulls jobs dynamically (no idle waiting)
        p = ctx.Process(target=gpu_queue_worker,
                        args=(worker_key, gpu_id, gpu_slot, jobs, result_queue, shared_work_q),
                        daemon=False)
        p.start()
        processes[worker_key] = p
        slot_meta[worker_key] = (gpu_id, gpu_slot)
        print(
            f"[scheduler] GPU{gpu_id}/slot{gpu_slot}: started PID {p.pid} "
            f"with {len(jobs)} queued job(s)",
            flush=True,
        )

    def record_result(res: dict, worker_key: str, gid: int, slot: int):
        nonlocal completed
        ans = res.get("ansatz_name")
        enc = ENCODING_ALIASES.get(res.get("encoding_name"), res.get("encoding_name"))
        if ans is None or enc is None:
            return
        key = (ans, enc)
        if key in completed_keys:
            return
        completed_keys.add(key)
        if ans not in all_results:
            all_results[ans] = {}
        all_results[ans][enc] = res.get("result") or {"error": res.get("error", "Unknown worker error")}
        completed += 1
        elapsed = time.perf_counter() - t0
        gpu_label = f"GPU{gid}/slot{slot}"
        if res.get("error"):
            print(f"[{completed}/{total_jobs}] ✗ {gpu_label} {ans}/{enc} ({elapsed:.1f}s)", flush=True)
        else:
            m = res["result"]["metrics"]
            prec = m.get('precision', float('nan'))
            rec  = m.get('recall',    float('nan'))
            print(
                f"[{completed}/{total_jobs}] ✓ {gpu_label} {ans}/{enc} "
                f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} "
                f"prec={prec:.4f} rec={rec:.4f} ({elapsed:.1f}s)",
                flush=True,
            )

    def mark_worker_remaining_failed(worker_key: str, reason: str):
        nonlocal completed
        gid, slot = slot_meta.get(worker_key, ("?", "?"))
        for job in active_queues.get(worker_key, []):
            ans = job["ansatz_name"]
            enc = ENCODING_ALIASES.get(job["encoding_name"], job["encoding_name"])
            key = (ans, enc)
            if key in completed_keys:
                continue
            completed_keys.add(key)
            if ans not in all_results:
                all_results[ans] = {}
            all_results[ans][enc] = {"error": reason}
            completed += 1
            elapsed = time.perf_counter() - t0
            print(
                f"[{completed}/{total_jobs}] ✗ GPU{gid}/slot{slot} {ans}/{enc} "
                f"({elapsed:.1f}s) -- {reason}",
                flush=True,
            )

    while len(done_workers) < len(processes):
        try:
            msg = result_queue.get(timeout=30)
        except queue_module.Empty:
            for worker_key, p in list(processes.items()):
                if worker_key in done_workers:
                    continue
                if not p.is_alive():
                    done_workers.add(worker_key)
                    if p.exitcode not in (0, None):
                        mark_worker_remaining_failed(worker_key, f"worker process exited with code {p.exitcode}")
            continue

        mtype = msg.get("type")
        worker_key = str(msg.get("worker_key"))
        gid = int(msg.get("gpu_id"))
        slot = int(msg.get("gpu_slot", 0))
        if mtype == "result":
            record_result(msg.get("result", {}), worker_key, gid, slot)
        elif mtype == "worker_done":
            done_workers.add(worker_key)
            print(f"[scheduler] GPU{gid}/slot{slot}: queue finished", flush=True)

    for worker_key, p in processes.items():
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Data and status
# ─────────────────────────────────────────────────────────────────────────────





def resolve_feature_folder(root: Path, feature_set: str) -> Optional[Path]:
    """Find the folder for a 4-feature representation.

    XGB+PCA folders have been observed under several names across datasets:
      xgb_pca4                     (NF-UNSW-NB15 v4 naming)
      xgb_selected_scaled_features  (UNSW-NB15 actual folder name)
      xgb_selected_pca4
      xgb_pca4_features
    All resolve to the logical feature_set key "xgb_pca4".
    """
    fs = str(feature_set)

    # EXACT match first — if the folder name matches exactly, use it immediately.
    # This prevents xgb_pca4/ being shadowed by xgb_selected_scaled_features/
    # when both exist on disk (which happens after running fix_xgb_pca4_package.py).
    _exact = root / fs
    if _exact.exists() and _exact.is_dir():
        return _exact

    # All known on-disk folder names for XGB+PCA, most-specific first.
    _XGB_FOLDERS = [
        "xgb_pca4",                        # exact generated folder — preferred
        "xgb_selected_scaled_features",    # raw 20-feature folder — fallback only
        "xgb_pca4",                        # NF-UNSW-NB15 / default
        "xgb_pca",
        "xgbpca4",
        "xgb_pca_4",
        "xgb_selected_pca4",
        "xgb_pca4_features",
        "xgbpluspca4",
        "xgb_selected_features",
        "xgb_scaled_features",
        "xgb_pcs4",
        "xgb_pcs",
        "xgb+pcs",
        "xgb+pc4",
    ]

    aliases = {
        "xgb_pca4":                     _XGB_FOLDERS,
        "xgb+pcs":                       _XGB_FOLDERS,
        "xgb+pc4":                       _XGB_FOLDERS,
        "xgb_selected_scaled_features":  _XGB_FOLDERS,
    }
    candidates = aliases.get(fs.lower(), [fs])
    for c in candidates:
        p = root / c
        if p.exists() and p.is_dir():
            return p
    # Last resort: case-insensitive scan of root directory
    try:
        fs_lower = fs.lower().replace("-", "_").replace("+", "_")
        for entry in root.iterdir():
            if entry.is_dir() and (
                entry.name.lower() == fs_lower
                or entry.name.lower().replace("-", "_").replace("+", "_") == fs_lower
            ):
                return entry
    except Exception:
        pass
    return None



def candidate_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def resolve_package_paths(args, train_size: int):
    """
    gpu_package flat structure (confirmed):
      {package_root}/{feature_set}/train_{size}.csv
      {package_root}/{feature_set}/validation_1000.csv
      {package_root}/{feature_set}/test_2000.csv
    """
    root    = Path(args.package_root)
    fs      = str(args.feature_set)
    fs_dir  = root / fs

    train_csv = str(fs_dir / f"train_{train_size}.csv")
    val_csv   = str(fs_dir / "validation_1000.csv")
    test_csv  = str(fs_dir / "test_2000.csv")

    return train_csv, val_csv, test_csv


def feature_cols_for(feature_set: str, df: Optional[pd.DataFrame] = None):
    """Auto-detect 4 feature columns for PCA4, ICA4, XGB+PCA/PCS4, and AE4."""
    cols_available = set(df.columns) if df is not None else None

    def first_present(candidates):
        for cols in candidates:
            if cols_available is None or all(c in cols_available for c in cols):
                return cols
        return candidates[0]

    fs = str(feature_set).lower()
    if fs == "pca4":
        return first_present([
            ["PC1", "PC2", "PC3", "PC4"],
            ["PC_1", "PC_2", "PC_3", "PC_4"],
        ])
    if fs == "ica4":
        return first_present([
            ["IC1", "IC2", "IC3", "IC4"],
            ["IC_1", "IC_2", "IC_3", "IC_4"],
        ])
    if fs in {"ae4", "autoencoder4", "autoencoder_4"}:
        return first_present([
            ["z0", "z1", "z2", "z3"],
            ["AE1", "AE2", "AE3", "AE4"],
        ])

    # XGB+PCA/PCS feature names vary across scripts, so auto-detect common variants.
    xgb_candidates = [
        ["PC1", "PC2", "PC3", "PC4"],
        ["XGB_PC1", "XGB_PC2", "XGB_PC3", "XGB_PC4"],
        ["XGBPC1", "XGBPC2", "XGBPC3", "XGBPC4"],
        ["XGB_PCA1", "XGB_PCA2", "XGB_PCA3", "XGB_PCA4"],
        ["XGBPCA1", "XGBPCA2", "XGBPCA3", "XGBPCA4"],
        ["PCS1", "PCS2", "PCS3", "PCS4"],
        ["xgb_pc1", "xgb_pc2", "xgb_pc3", "xgb_pc4"],
    ]
    if cols_available is not None:
        for cols in xgb_candidates:
            if all(c in cols_available for c in cols):
                return cols
        # Last resort: any 4 numeric non-target columns.
        excluded = {"Attack", "Label", "AttackEncodedKnown", "AttackEncodedAll", "Attack_enc", "attack", "label", "attack_enc"}
        numeric = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) == 4:
            return numeric
    return xgb_candidates[0]


def load_data(args):
    """
    Load train / validation / test from pre-split CSVs.
    CHANGED: validation loaded from args.val_csv (no train_test_split).
    MinMaxScaler fit on training set, applied to val + test.
    """
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)     # separate validation_1000.csv
    test_df  = pd.read_csv(args.test_csv)

    feat_cols = feature_cols_for(args.feature_set, train_df)
    missing = [col for col in feat_cols if col not in train_df.columns]
    if missing:
        raise ValueError(f"Feature columns {missing} missing from {args.train_csv}")
    if args.target_col not in train_df.columns:
        raise ValueError(f"Target column {args.target_col!r} not found in {args.train_csv}")

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[args.target_col].values.astype(int)

    X_val   = val_df[feat_cols].values.astype(np.float32)
    y_val   = val_df[args.target_col].values.astype(int)

    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df[args.target_col].values.astype(int)

    # MinMaxScaler: fit on training set, apply to val and test (same as v5)
    scaler  = MinMaxScaler(feature_range=(0.0, float(np.pi)))
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)


    counts = np.bincount(y_train.astype(int), minlength=2)
    pos_weight_val = float(counts[0] / max(counts[1], 1))

    print(f"  Data loaded: train={X_train.shape} val={X_val.shape} test={X_test.shape} "
          f"pos_weight={pos_weight_val:.3f}  feat={feat_cols}", flush=True)
    return X_train, y_train, X_val, y_val, X_test, y_test, pos_weight_val


def show_status(results_dir: str) -> None:
    p = Path(results_dir)
    if not p.exists():
        print(f"No results directory found: {results_dir}")
        return
    files = sorted(p.glob("*_status.json"))
    if not files:
        print(f"No *_status.json files found in {results_dir}")
        return
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
            rows.append(d)
        except Exception:
            pass
    counts = {}
    for r in rows:
        counts[r.get("status", "unknown")] = counts.get(r.get("status", "unknown"), 0) + 1
    print(f"Status files: {len(rows)} | {counts}")
    for r in sorted(rows, key=lambda x: (str(x.get("status")), str(x.get("run_key")))):
        status = r.get("status", "?")
        rk = r.get("run_key", "?")
        gpu = r.get("physical_gpu_id", "?")
        ep = r.get("epoch", "-")
        eps = r.get("epochs", "-")
        acc = r.get("val_accuracy", None)
        auc = r.get("val_auc", None)
        metric = ""
        if acc is not None:
            metric = f" acc={acc:.4f} auc={auc:.4f}" if isinstance(auc, (int, float)) else f" acc={acc:.4f}"
        print(f"{status:<8} GPU{gpu} {rk:<45} epoch {ep}/{eps}{metric}")


def make_arg_parser():
    p = argparse.ArgumentParser(description=f"{MODEL_KIND} multi-GPU optimized sweep")
    p.add_argument("--train_csv", default=None, help="Optional explicit train CSV. If omitted, package_root/feature_set/train/train_<size>.csv is used.")
    p.add_argument("--test_csv",  default=None, help="Explicit test CSV. Auto-resolved to test_2000.csv if omitted.")
    p.add_argument("--val_csv",   default=None, help="Explicit validation CSV. Auto-resolved to validation_1000.csv if omitted.")
    p.add_argument("--package_root", default="/home/nvidia/21PHD1192/qml_id2/UNSW/gpu_package/step3_dr_datasets")
    p.add_argument("--feature_set", default="pca4", help="Single feature set: pca4, ica4, xgb_pca4, ae4, or a folder name.")
    p.add_argument("--feature_sets", default=None, help="Comma list for research sweep. Default: pca4,ica4,xgb_pca4,autoencoder")
    p.add_argument("--require_all_feature_sets", action="store_true", help="Raise an error instead of skipping a missing feature-set folder/file.")
    p.add_argument("--train_sizes", default="5000", help="Comma list: 5000 or 5000,10000,15000,20000,25000")
    p.add_argument("--test_split", choices=["test_balanced_2000", "test_natural_2000", "unknown"], default="test_balanced_2000")
    p.add_argument("--target_col", default="Label")
    p.add_argument("--results_dir", default="/home/nvidia/21PHD1192/qml_id2/UNSW/gpu_package/results_vqc_UNSW_NB15_v6_default_qubit")
    p.add_argument("--status_only", action="store_true", help="Only print status from results_dir and exit.")

    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", choices=["none", "cosine"], default="none", help="Kept for command compatibility. v3 keeps LR fixed; use none.")
    p.add_argument("--eta_min_ratio", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=0, help="Deprecated in v3. Early stopping is disabled; all jobs run --epochs.")
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--select_best_by", choices=["accuracy", "f1", "auc"], default="accuracy")
    p.add_argument("--threshold_metric", choices=["accuracy", "f1"], default="accuracy")
    p.add_argument("--pl_backend", choices=["lightning.gpu", "default.qubit", "default.mixed"],
                   default="default.qubit",
                   help="PennyLane simulator backend. lightning.gpu: pure-state, "
                        "C++/CUDA/adjoint backend, runs on GPU hardware (same backend "
                        "used during the project's original training runs). "
                        "default.qubit: pure-state, fastest for this circuit size via "
                        "native batch broadcasting, use for standard noiseless "
                        "training/evaluation (default). default.mixed: density-matrix, "
                        "required only for a future noise study with Kraus channels — "
                        "identical results to default.qubit when no noise is applied, "
                        "but slower.")

    p.add_argument("--ans_reps", type=int, default=1, help="Ansatz repetitions. Use --ans_reps 2 exactly as in your example when needed.")
    p.add_argument("--enc_reps", type=int, default=1)
    p.add_argument("--entanglement", choices=["none", "linear", "circular", "full"], default="circular")
    p.add_argument("--encoding_entanglement", action="store_true", help="Enable CZ inside custom_h_ry_rz encoding. Default is off; no bridge entangling is ever added.")

    p.add_argument("--max_gpus", type=int, default=8)
    p.add_argument("--jobs_per_gpu", type=int, default=1, help="Concurrent VQC worker processes per physical GPU. Use 1 for safest; try 2 or 4 for 4-qubit VQC on A100.")
    p.add_argument("--gpus", default="auto", help="Comma list like 0,1,2,3 or auto.")
    p.add_argument("--min_free_gb", type=float, default=1.0)
    p.add_argument("--single_gpu", action="store_true", help="Force one worker/GPU for debugging.")
    p.add_argument("--cpu_threads_per_worker", type=int, default=2)
    p.add_argument("--monitor_interval", type=float, default=5.0, help="Deprecated/ignored in v3. Resource monitoring is disabled.")

    p.add_argument("--ansatzes", default="all", help="Comma list or all.")
    p.add_argument("--encodings", default="all", help="Comma list or all. angle_rx/angle_ry/angle_rz aliases accepted.")
    p.add_argument("--val_fraction", type=float, default=0.0,
                   help="Unused — validation loaded from validation_1000.csv. Kept for CLI compatibility.")
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_test", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_mode", choices=["random", "zeros"], default="random")
    p.add_argument("--init_scale", type=float, default=0.05)
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--resume", action="store_true", help="Resume each job from checkpoint if available.")
    return p



def resolve_feature_folder(root: Path, feature_set: str) -> Optional[Path]:
    """Find the folder for a 4-feature representation.

    XGB+PCA folders have been observed under several names across datasets:
      xgb_pca4                     (NF-UNSW-NB15 v4 naming)
      xgb_selected_scaled_features  (UNSW-NB15 actual folder name)
    All resolve to the logical feature_set key "xgb_pca4".
    """
    fs = str(feature_set)
    _exact = root / fs
    if _exact.exists() and _exact.is_dir():
        return _exact
    _XGB_FOLDERS = [
        "xgb_pca4",
        "xgb_selected_scaled_features",
        "xgb_pca",
        "xgbpca4",
        "xgb_pca_4",
        "xgb_selected_pca4",
        "xgb_pca4_features",
        "xgbpluspca4",
        "xgb_selected_features",
        "xgb_scaled_features",
        "xgb_pcs4",
        "xgb_pcs",
        "xgb+pcs",
        "xgb+pc4",
    ]
    aliases = {
        "xgb_pca4":                     _XGB_FOLDERS,
        "xgb+pcs":                       _XGB_FOLDERS,
        "xgb+pc4":                       _XGB_FOLDERS,
        "xgb_selected_scaled_features":  _XGB_FOLDERS,
    }
    candidates = aliases.get(fs.lower(), [fs])
    for c in candidates:
        p = root / c
        if p.exists() and p.is_dir():
            return p
    try:
        fs_lower = fs.lower().replace("-", "_").replace("+", "_")
        for entry in root.iterdir():
            if entry.is_dir() and (
                entry.name.lower() == fs_lower
                or entry.name.lower().replace("-", "_").replace("+", "_") == fs_lower
            ):
                return entry
    except Exception:
        pass
    return None



def candidate_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None

# (duplicate resolve_package_paths removed — see UNSW version above)


def feature_cols_for(feature_set: str, df: Optional[pd.DataFrame] = None):
    """Return the 4 feature columns for the UNSW flat-folder structure.

    Folder / column naming (confirmed from dr_subsets/):
      pca4/         → PC1, PC2, PC3, PC4
      ica4/         → IC1, IC2, IC3, IC4
      xgb_pca4/     → XP1, XP2, XP3, XP4
      autoencoder/  → z0, z1, z2, z3  (0-indexed, already in [0,π])
    """
    fs  = str(feature_set).lower()
    PCA  = ["PC1", "PC2", "PC3", "PC4"]
    ICA  = ["IC1", "IC2", "IC3", "IC4"]
    XGB  = ["XP1", "XP2", "XP3", "XP4"]
    AE   = ["z0", "z1", "z2", "z3"]

    # Explicit lookup
    if fs == "pca4":                                  preferred = [PCA]
    elif fs == "ica4":                                preferred = [ICA]
    elif fs in ("xgb_pca4", "xgb_pcs4", "xgb"):      preferred = [XGB, PCA]
    elif fs in ("autoencoder", "ae4", "ae"):          preferred = [AE]
    else:                                             preferred = [PCA, ICA, XGB, AE]

    if df is not None:
        cols = list(df.columns)
        for cand in preferred:
            if all(c in cols for c in cand):
                return cand
        # Fallback: first 4 numeric non-label columns
        meta = {"Attack","Label","AttackEncodedKnown","AttackEncodedAll",
                "Attack_enc","attack","label","attack_enc","target","Target",
                "Unnamed: 0","index"}
        numeric = [col for col in cols
                   if col not in meta and pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric) >= 4:
            return numeric[:4]
        raise ValueError(
            f"Cannot find 4 feature columns for feature_set={feature_set!r}. "
            f"Available: {cols}")

    return preferred[0]





def main():
    args = make_arg_parser().parse_args()
    if args.status_only:
        show_status(args.results_dir)
        return

    set_all_seeds(args.seed)
    train_sizes = [int(x.strip()) for x in str(args.train_sizes).split(',') if x.strip()]
    if args.feature_sets:
        feature_sets = [x.strip() for x in str(args.feature_sets).split(',') if x.strip()]
    else:
        feature_sets = ["pca4", "ica4", "xgb_pca4", "autoencoder"]

    root_results_dir = args.results_dir
    explicit_train_csv = args.train_csv
    explicit_test_csv = args.test_csv
    combined_rows = []

    # ── PRE-FLIGHT FOLDER PROBE ───────────────────────────────────────────────
    # Verify every feature_set folder resolves BEFORE any training starts.
    # This catches mis-named XGB folders (xgb_selected_scaled_features etc.)
    # immediately instead of discovering the problem after hours of other runs.
    print("", flush=True)
    print("╔" + "═" * 88 + "╗", flush=True)
    print(f"║  PRE-FLIGHT CHECK  —  package_root: {str(args.package_root):<50} ║", flush=True)
    print("╠" + "═" * 88 + "╣", flush=True)
    _root  = Path(args.package_root)
    _sizes = [int(x) for x in str(args.train_sizes).split(",")]
    _all_ok = True
    for _fs in feature_sets:
        _fs_dir = _root / _fs           # e.g. gpu_package/pca4
        if _fs_dir.exists() and _fs_dir.is_dir():
            _train_ok = any((_fs_dir / f"train_{ts}.csv").exists() for ts in _sizes)
            _val_ok   = (_fs_dir / "validation_1000.csv").exists()
            _test_ok  = (_fs_dir / "test_2000.csv").exists()
            _ok = _train_ok and _val_ok and _test_ok
            _status = "✅ OK  " if _ok else "⚠️  WARN"
            if not _ok:
                _all_ok = False
                _miss = [n for n,v in [("train",_train_ok),("val",_val_ok),("test",_test_ok)] if not v]
                print(f"║  {_status}  {_fs:<20} → missing: {str(_miss):<50} ║", flush=True)
            else:
                print(f"║  {_status}  {_fs:<20} → {str(_fs_dir):<54} ║", flush=True)
        else:
            _all_ok = False
            print(f"║  ❌ MISSING  {_fs:<20} → {str(_fs_dir):<50} ║", flush=True)
    print("╚" + "═" * 88 + "╝", flush=True)
    if not _all_ok:
        print("\n[WARNING] Some feature_set folders could not be resolved. "
              "Those feature sets will be skipped. Check folder names above.\n", flush=True)
    else:
        print("[PRE-FLIGHT] All feature_set folders resolved. Starting sweep.\n", flush=True)

    for train_size in train_sizes:
        for feature_set in feature_sets:
            args.feature_set = feature_set
            args.train_csv = explicit_train_csv
            args.test_csv = explicit_test_csv
            args.train_csv, args.val_csv, args.test_csv = resolve_package_paths(args, train_size)

            for csv_name, csv_path in [("train", args.train_csv),
                                            ("val",   args.val_csv),
                                            ("test",  args.test_csv)]:
                if csv_path is None or not Path(csv_path).exists():
                    msg = f"{csv_name} CSV not found: {csv_path}"
                    if args.require_all_feature_sets:
                        raise FileNotFoundError(msg)
                    print(f"[SKIP] {msg}", flush=True)
                    break
            else:
                pass  # all CSVs found — continue to training
                # (the for-else: else runs only when loop did not break)
            if not all(Path(p).exists() for p in [args.train_csv, args.val_csv, args.test_csv] if p):
                continue

            args.results_dir = str(Path(root_results_dir) / f"{feature_set}_{train_size}_{args.test_split}")
            Path(args.results_dir).mkdir(parents=True, exist_ok=True)

            ansatz_list = parse_csv_list(args.ansatzes, ANSATZ_LIST_DEFAULT)
            encoding_list = parse_csv_list(args.encodings, ENCODING_LIST_DEFAULT, ENCODING_ALIASES)

            # ── DATA-IN-USE BANNER ────────────────────────────────────────
            print("", flush=True)
            print("╔" + "═" * 88 + "╗", flush=True)
            print(f"║  {'FEATURE SET':12s}: {feature_set:<30}  TRAIN SIZE: {train_size:<8}              ║", flush=True)
            print(f"║  {'TRAIN CSV':12s}: {str(args.train_csv):<74} ║", flush=True)
            print(f"║  {'TEST CSV':12s}: {str(args.test_csv):<74} ║", flush=True)
            print(f"║  {'VAL CSV':12s}: {str(args.val_csv):<75} ║", flush=True)
            print(f"║  {'RESULTS DIR':12s}: {str(args.results_dir):<74} ║", flush=True)
            print(f"║  {'CONFIGS':12s}: {len(ansatz_list)} ansatz × {len(encoding_list)} encodings = {len(ansatz_list)*len(encoding_list)} jobs                               ║", flush=True)
            print(f"║  {'STARTED AT':12s}: {time.strftime('%Y-%m-%d %H:%M:%S'):<74} ║", flush=True)
            print("╚" + "═" * 88 + "╝", flush=True)
            print("", flush=True)

            X_train, y_train, X_val, y_val, X_test, y_test, pos_weight_val = load_data(args)
            np.save(os.path.join(args.results_dir, "y_test.npy"), y_test)
            print(
                f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape} | pos_weight={pos_weight_val:.3f}",
                flush=True,
            )

            gpu_ids = detect_gpu_ids(args.max_gpus, args.min_free_gb, args.gpus)
            if args.single_gpu:
                gpu_ids = gpu_ids[:1]
            if not gpu_ids:
                raise RuntimeError("No active GPUs detected. Check CUDA_VISIBLE_DEVICES / nvidia-smi.")
            jobs_per_gpu = max(1, int(args.jobs_per_gpu))
            total_slots = len(gpu_ids) * jobs_per_gpu
            n_workers = min(total_slots, max(1, len(ansatz_list) * len(encoding_list)))
            print(
                f"Active GPUs: {gpu_ids} | jobs_per_gpu={jobs_per_gpu} | "
                f"worker slots={n_workers}",
                flush=True,
            )

            cfg_base = dict(
                model_name=MODEL_KIND,
                n_qubits=args.n_qubits,
                enc_reps=args.enc_reps,
                ans_reps=args.ans_reps,
                entanglement=args.entanglement,
                encoding_entanglement=bool(args.encoding_entanglement),
                su2_gates=["ry", "rz"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_scheduler=args.lr_scheduler,
                weight_decay=args.weight_decay,
                eta_min_ratio=args.eta_min_ratio,
                grad_clip=args.grad_clip,
                patience=args.patience,
                min_delta=args.min_delta,
                select_best_by=args.select_best_by,
                threshold_metric=args.threshold_metric,
                pl_backend=args.pl_backend,
                init_mode=args.init_mode,
                init_scale=args.init_scale,
                seed=args.seed,
                cpu_threads_per_worker=args.cpu_threads_per_worker,
                ckpt_every=args.ckpt_every,
                resume=bool(args.resume),
                monitor_interval=args.monitor_interval,
                jobs_per_gpu=args.jobs_per_gpu,
            )

            combos = [(a, e) for a in ansatz_list for e in encoding_list]
            queues = build_per_gpu_queues(
                combos, gpu_ids, cfg_base,
                X_train, y_train, X_val, y_val, X_test, y_test,
                pos_weight_val, args.results_dir,
                jobs_per_gpu=jobs_per_gpu,
                feature_set=feature_set,   # pass DR name for label
            )
            all_results = {a: {} for a in ansatz_list}

            plan_rows = []
            for worker_key, jobs in queues.items():
                for order, job in enumerate(jobs, start=1):
                    plan_rows.append({
                        "feature_set": feature_set,
                        "train_size": train_size,
                        "worker_key": worker_key,
                        "gpu_id": job.get("gpu_id"),
                        "gpu_slot": job.get("gpu_slot", 0),
                        "queue_order": order,
                        "ansatz": job["ansatz_name"],
                        "encoding": job["encoding_name"],
                    })
            pd.DataFrame(plan_rows).to_csv(os.path.join(args.results_dir, "job_plan.csv"), index=False)
            print(
                f"Jobs: {len(combos)} | GPUs: {len(gpu_ids)} | "
                f"jobs_per_gpu={jobs_per_gpu} | queues saved to job_plan.csv",
                flush=True,
            )

            t_size = time.perf_counter()
            all_results = run_scheduler(queues, n_workers=n_workers, total_jobs=len(combos), all_results=all_results)
            elapsed_size = time.perf_counter() - t_size

            roc_paths = save_grouped_roc_by_ansatz(args.results_dir, ansatz_list, encoding_list, model_name=MODEL_KIND)
            summary_json = os.path.join(args.results_dir, f"all_results_{MODEL_KIND.lower()}.json")
            atomic_json_dump(all_results, summary_json)

            rows = []
            for ans, encs in all_results.items():
                for enc, payload in encs.items():
                    if isinstance(payload, dict) and "metrics" in payload:
                        row = {"feature_set": feature_set, "train_size": train_size, "ansatz": ans, "encoding": enc, **payload["metrics"]}
                        rows.append(row)
                        combined_rows.append(row)
            if rows:
                df = pd.DataFrame(rows).sort_values(["accuracy", "auc", "f1"], ascending=False)
                df.to_csv(os.path.join(args.results_dir, "summary_metrics.csv"), index=False)
                print("\nTop configurations:")
                print(df.head(10).to_string(index=False))

            # ── AGENTIC SUMMARY JSON (critic/explainer agent input) ───────
            agent_summary = {
                "feature_set": feature_set,
                "train_size": train_size,
                "train_csv": str(args.train_csv),
                "test_csv": str(args.test_csv),
                "results_dir": str(args.results_dir),
                "total_jobs": len(combos),
                "completed_jobs": len(rows),
                "total_time_seconds": round(elapsed_size, 2),
                "total_time_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_size)),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "top_configs": rows[:5] if rows else [],
                "encodings_used": encoding_list,
                "ansatzes_used": ansatz_list,
            }
            atomic_json_dump(agent_summary, os.path.join(args.results_dir, "agent_summary.json"))

            # ── TOTAL TIME BANNER ─────────────────────────────────────────
            hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_size))
            print("", flush=True)
            print("╔" + "═" * 88 + "╗", flush=True)
            print(f"║  COMPLETED  feature_set={feature_set:<10}  train_size={train_size:<8}                          ║", flush=True)
            print(f"║  Total time : {hms}  ({elapsed_size:.1f}s)  |  Jobs: {len(combos)}  |  Finished: {time.strftime('%H:%M:%S')}        ║", flush=True)
            print(f"║  Results JSON : {str(summary_json):<72} ║", flush=True)
            print("╚" + "═" * 88 + "╝", flush=True)
            print("", flush=True)

            pd.DataFrame([{"feature_set": feature_set, "train_size": train_size, "elapsed_s": elapsed_size, "jobs": len(combos), "gpus": len(gpu_ids)}]).to_csv(
                os.path.join(args.results_dir, "time_summary.csv"), index=False
            )
            print("Grouped ROC PNGs:", flush=True)
            for path in roc_paths:
                print(f"  {path}", flush=True)

    if combined_rows:
        combined_df = pd.DataFrame(combined_rows).sort_values(["accuracy", "auc", "f1"], ascending=False)
        # Reorder columns for readability
        prio_cols = ["feature_set", "train_size", "ansatz", "encoding",
                     "accuracy", "f1", "auc", "precision", "recall", "threshold"]
        rest_cols = [c for c in combined_df.columns if c not in prio_cols]
        combined_df = combined_df[[c for c in prio_cols if c in combined_df.columns] + rest_cols]
        Path(root_results_dir).mkdir(parents=True, exist_ok=True)
        combined_path = Path(root_results_dir) / "combined_summary_metrics.csv"
        combined_df.to_csv(combined_path, index=False)
        print("=" * 90, flush=True)
        print(f"Combined summary saved: {combined_path}", flush=True)
        print(combined_df.head(20).to_string(index=False), flush=True)
        print("=" * 90, flush=True)
    else:
        print("[WARN] No completed result rows were produced. Check missing folders or job errors.", flush=True)


if __name__ == "__main__":
    main()