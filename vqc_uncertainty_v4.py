"""
[NOTE: Inference uses default.qubit — CPU vectorized simulator,
 NOT lightning.gpu. For 4-qubit circuits, GPU kernel-launch overhead
 exceeds actual computation; CPU broadcasting is faster. --gpu_ids
 below only labels parallel CPU worker processes for logging.]

vqc_uncertainty_final.py
═════════════════════════
Uncertainty analysis using IDENTICAL model loading as vqc_4feature_UNSW_v3.py.

For each model:
  1. Load best_model.pt → rebuild model with build_model() + load_state_dict()
  2. Load final_model.pt → gradient norm history → adaptive σ
  3. Load test_2000.csv → MinMaxScaler(fit on train_5000)
  4. N=30 perturbations: deepcopy model, perturb quantum_weights, predict_proba()
  5. Compute σ_epi, H̄, VR, ECE, C*

Run:
  python vqc_uncertainty_final.py \
      --results_dir  /path/to/results \
      --package_root /path/to/gpu_package \
      --gpu_ids 5,6,7 --n_perturb 30
"""

import os, gc, json, time, copy, warnings, argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# EXACT COPIES FROM vqc_4feature_UNSW_v3.py
# ══════════════════════════════════════════════════════════════════
ANSATZ_LIST_DEFAULT  = ["custom_ansatz_1","efficient_su2_like","strongly_entangling","real_amplitudes"]
ENCODING_LIST_DEFAULT= ["rx_embedding","ry_embedding","rz_embedding","amplitude_embedding",
                         "iqp_embedding","zz_feature_map","custom_h_ry_rz"]
ENCODING_ALIASES     = {"angle_rx":"rx_embedding","angle_ry":"ry_embedding","angle_rz":"rz_embedding"}
MODEL_KIND           = "VQC"


class EncodingPL:
    def __init__(self,n_qubits=4,reps=1,entanglement="circular"):
        self.n_qubits=n_qubits; self.reps=reps; self.entanglement=entanglement
    def _pairs(self,ent=None):
        e=self.entanglement if ent is None else ent; q=self.n_qubits
        if e=="none":    return []
        if e=="linear":  return [(i,i+1) for i in range(q-1)]
        if e=="circular":
            p=[(i,i+1) for i in range(q-1)]
            if q>2: p.append((q-1,0))
            return p
        if e=="full":    return [(i,j) for i in range(q) for j in range(i+1,q)]
        raise ValueError(f"Unknown entanglement: {e}")
    def rx_embedding(self,qml,x):
        for i in range(self.n_qubits): qml.RX(x[...,i],wires=i)
    def ry_embedding(self,qml,x):
        for i in range(self.n_qubits): qml.RY(x[...,i],wires=i)
    def rz_embedding(self,qml,x):
        for i in range(self.n_qubits): qml.RZ(x[...,i],wires=i)
    def amplitude_embedding(self,qml,x):
        qml.AmplitudeEmbedding(features=x,wires=range(self.n_qubits),pad_with=0.0,normalize=True)
    def iqp_embedding(self,qml,x):
        for _ in range(self.reps):
            for i in range(self.n_qubits): qml.Hadamard(wires=i)
            for i in range(self.n_qubits): qml.RZ(x[...,i],wires=i)
            for i,j in self._pairs():
                qml.CNOT(wires=[i,j]); qml.RZ(x[...,i]*x[...,j],wires=j); qml.CNOT(wires=[i,j])
    def zz_feature_map(self,qml,x):
        for _ in range(self.reps):
            for i in range(self.n_qubits): qml.Hadamard(wires=i)
            for i in range(self.n_qubits): qml.RZ(2.0*x[...,i],wires=i)
            for i,j in self._pairs():
                qml.IsingZZ(2.0*(x[...,i]-np.pi)*(x[...,j]-np.pi),wires=[i,j])
    def custom_h_ry_rz(self,qml,x,add_encoding_entanglement=False):
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i); qml.RY(x[...,i],wires=i); qml.RZ(x[...,i]**2,wires=i)
        if add_encoding_entanglement:
            for i,j in self._pairs(): qml.CZ(wires=[i,j])
    def apply(self,qml,name,x,add_encoding_entanglement=False):
        name=ENCODING_ALIASES.get(name,name)
        if   name=="rx_embedding":        self.rx_embedding(qml,x)
        elif name=="ry_embedding":        self.ry_embedding(qml,x)
        elif name=="rz_embedding":        self.rz_embedding(qml,x)
        elif name=="amplitude_embedding": self.amplitude_embedding(qml,x)
        elif name=="iqp_embedding":       self.iqp_embedding(qml,x)
        elif name=="zz_feature_map":      self.zz_feature_map(qml,x)
        elif name=="custom_h_ry_rz":      self.custom_h_ry_rz(qml,x,add_encoding_entanglement)
        else: raise ValueError(f"Unknown encoding '{name}'")


class AnsatzPL:
    def __init__(self,n_qubits=4,reps=1,entanglement="circular"):
        self.n_qubits=n_qubits; self.reps=reps; self.entanglement=entanglement
    def _pairs(self,ent=None):
        e=self.entanglement if ent is None else ent; q=self.n_qubits
        if e=="none":    return []
        if e=="linear":  return [(i,i+1) for i in range(q-1)]
        if e=="circular":
            p=[(i,i+1) for i in range(q-1)]
            if q>2: p.append((q-1,0))
            return p
        if e=="full":    return [(i,j) for i in range(q) for j in range(i+1,q)]
        raise ValueError(f"Unknown entanglement: {e}")
    def custom_ansatz_1(self,qml,w,ent=None):
        for r in range(w.shape[0]):
            for i in range(self.n_qubits): qml.RY(w[r,0,i],wires=i)
            for i,j in self._pairs(ent):   qml.CNOT(wires=[i,j])
            for i in range(self.n_qubits): qml.RZ(w[r,1,i],wires=i)
    def efficient_su2_like(self,qml,w,su2_gates=("ry","rz"),ent=None):
        # Matches Qiskit's literal default structure: reps+1 rotation layers,
        # reps CNOT layers (skip_final_rotation_layer=False). EXACT copy of
        # vqc_4feature_UNSW_v8.py's AnsatzPL.efficient_su2_like — kept in
        # sync so bootstrap/uncertainty load v8-trained checkpoints correctly.
        gmap={"rx":qml.RX,"ry":qml.RY,"rz":qml.RZ}
        n_layers=w.shape[0]   # = reps + 1, includes the final layer
        for r in range(n_layers):
            for g,gn in enumerate(su2_gates): [gmap[gn.lower()](w[r,g,i],wires=i) for i in range(self.n_qubits)]
            if r < n_layers-1:
                for i,j in self._pairs("linear"): qml.CNOT(wires=[i,j])
    def strongly_entangling(self,qml,w):
        qml.StronglyEntanglingLayers(w,wires=range(self.n_qubits))
    def real_amplitudes(self,qml,w):
        # Matches Qiskit's literal default structure: reps+1 rotation layers,
        # reps CNOT layers (skip_final_rotation_layer=False). EXACT copy of
        # vqc_4feature_UNSW_v8.py's AnsatzPL.real_amplitudes.
        n_layers=w.shape[0]   # = reps + 1, includes the final layer
        for r in range(n_layers):
            for i in range(self.n_qubits): qml.RY(w[r,0,i],wires=i)
            if r < n_layers-1:
                for i,j in self._pairs("circular"): qml.CNOT(wires=[i,j])
    def apply(self,qml,name,w,ent=None,su2_gates=("ry","rz")):
        if   name=="custom_ansatz_1":    self.custom_ansatz_1(qml,w,ent=ent)
        elif name=="efficient_su2_like": self.efficient_su2_like(qml,w,su2_gates=su2_gates,ent=ent)
        elif name=="strongly_entangling":self.strongly_entangling(qml,w)
        elif name=="real_amplitudes":    self.real_amplitudes(qml,w)
        else: raise ValueError(f"Unknown ansatz '{name}'")


def get_weight_shape(ansatz_name,n_qubits,reps,su2_gates=("ry","rz")):
    if ansatz_name=="custom_ansatz_1":    return (reps,2,n_qubits)
    if ansatz_name=="efficient_su2_like": return (reps+1,len(su2_gates),n_qubits)   # +1: final rotation layer
    if ansatz_name=="strongly_entangling":return (reps,n_qubits,3)
    if ansatz_name=="real_amplitudes":    return (reps+1,1,n_qubits)   # +1: final rotation layer
    raise ValueError(f"Unknown ansatz '{ansatz_name}'")


def expval_matrix(out,batch_size,n_qubits):
    import torch
    if isinstance(out,(list,tuple)):
        out=torch.stack([o if torch.is_tensor(o) else torch.as_tensor(o) for o in out],dim=-1)
    elif not torch.is_tensor(out): out=torch.as_tensor(out)
    out=out.float().squeeze()
    if out.dim()==1:
        if batch_size==1 and out.numel()==n_qubits: out=out.reshape(1,n_qubits)
        elif n_qubits==1 and out.numel()==batch_size: out=out.reshape(batch_size,1)
    elif out.dim()==2:
        if tuple(out.shape)==(n_qubits,batch_size): out=out.T
        elif tuple(out.shape)!=(batch_size,n_qubits) and out.numel()==batch_size*n_qubits:
            out=out.reshape(batch_size,n_qubits)
    else:
        if out.numel()==batch_size*n_qubits: out=out.reshape(batch_size,n_qubits)
    if tuple(out.shape)!=(batch_size,n_qubits):
        raise RuntimeError(f"QNode output {tuple(out.shape)} → expected {(batch_size,n_qubits)}")
    return out


def make_pl_device(n_qubits, backend="default.qubit"):
    """
    User-selectable backend — lightning.gpu, default.qubit, or default.mixed.
    Matches vqc_4feature_UNSW_v8.py's make_pl_device() exactly, so released
    code can switch backends freely at evaluation time too.

    lightning.gpu: pure-state, C++/CUDA, adjoint diff_method, runs on GPU
    hardware. Empirically verified equivalent to default.qubit (max abs
    diff = 0.00e+00), but adjoint diff_method causes severe per-sample
    loop overhead for this circuit size (measured ~2217s/model vs
    ~2s/model on default.qubit). With K=200 bootstrap subsets, this
    difference is decisive for feasibility — default.qubit/default.mixed
    are strongly recommended over lightning.gpu for this evaluation step.

    default.qubit: pure-state, fastest via native batch broadcasting.

    default.mixed: density-matrix simulator. Required for any future
    noise-channel study (Kraus operators). Identical results to
    default.qubit when no noise is applied, verified separately.
    """
    import pennylane as qml
    valid_backends = ("lightning.gpu", "default.qubit", "default.mixed")
    if backend not in valid_backends:
        raise ValueError(f"backend must be one of {valid_backends}, got '{backend}'")
    diff_method = "adjoint" if backend == "lightning.gpu" else "backprop"
    dev = qml.device(backend, wires=n_qubits, shots=None)
    return dev, diff_method, backend


def build_model(cfg,dev,diff_method):
    import torch, torch.nn as nn, pennylane as qml
    n_qubits=int(cfg["n_qubits"])
    encoder=EncodingPL(n_qubits=n_qubits,reps=int(cfg["enc_reps"]),entanglement=cfg["entanglement"])
    ansatz =AnsatzPL (n_qubits=n_qubits,reps=int(cfg["ans_reps"]),entanglement=cfg["entanglement"])
    enc_name=ENCODING_ALIASES.get(cfg["encoding_name"],cfg["encoding_name"])
    ans_name=cfg["ansatz_name"]
    su2_gates=tuple(cfg["su2_gates"])
    wshape=get_weight_shape(ans_name,n_qubits,int(cfg["ans_reps"]),su2_gates=su2_gates)
    @qml.qnode(dev,interface="torch",diff_method=diff_method)
    def qnode(x,weights):
        encoder.apply(qml,enc_name,x,add_encoding_entanglement=bool(cfg.get("encoding_entanglement",False)))
        ansatz.apply(qml,ans_name,weights,ent=cfg["entanglement"],su2_gates=su2_gates)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    class QModel(nn.Module):
        def __init__(self):
            super().__init__()
            w0=float(cfg.get("init_scale",0.05))*__import__("torch").randn(wshape,dtype=__import__("torch").float32)
            self.quantum_weights=nn.Parameter(w0)
            self.scale=nn.Parameter(__import__("torch").tensor([1.0],dtype=__import__("torch").float32))
            self.bias =nn.Parameter(__import__("torch").tensor([0.0],dtype=__import__("torch").float32))
        def forward(self,x):
            bs=int(x.shape[0])
            out=qnode(x.cpu(),self.quantum_weights)
            z=expval_matrix(out,batch_size=bs,n_qubits=n_qubits)
            return self.scale*(-z.sum(dim=1))+self.bias
    return QModel()


def predict_proba(model,X,batch_size=256):
    """Exact copy from vqc_4feature_UNSW_v3.py."""
    import torch
    model.eval()
    X_t=torch.tensor(X,dtype=torch.float32)
    probs=[]
    with torch.no_grad():
        for s in range(0,len(X_t),batch_size*2):
            logits=model(X_t[s:s+batch_size*2]).detach().cpu()
            probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs).reshape(-1)


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
KNOWN_FS = ["xgb_pca4","autoencoder","pca4","ica4"]
STRATIFY_CANDIDATES = ["AttackEncodedKnown","AttackEncodedAll","Attack","attack",
                        "attack_cat","Label","label"]
ATTACK_MAP_UNSW = {0:"Benign",1:"Exploits",2:"Fuzzers",3:"Reconnaissance",
                   4:"DoS",5:"Generic",6:"Backdoor"}
ATTACK_MAP_TON  = {0:"Benign",1:"DDoS",2:"DoS",3:"Backdoor",
                   4:"Injection",5:"Password",6:"Scanning",7:"XSS"}


def feature_cols_for(feature_set, df=None):
    """Exact copy from vqc_4feature_UNSW_v3.py with autoencoder variant added."""
    import pandas as pd
    cols_available = set(df.columns) if df is not None else None
    def first_present(candidates):
        for cols in candidates:
            if cols_available is None or all(c in cols_available for c in cols):
                return cols
        return candidates[0]
    fs = str(feature_set).lower()
    if fs == "pca4":
        return first_present([["PC1","PC2","PC3","PC4"],["PC_1","PC_2","PC_3","PC_4"]])
    if fs == "ica4":
        return first_present([["IC1","IC2","IC3","IC4"],["IC_1","IC_2","IC_3","IC_4"]])
    if fs in {"ae4","autoencoder4","autoencoder_4","autoencoder"}:
        return first_present([["z0","z1","z2","z3"],["AE1","AE2","AE3","AE4"]])
    # xgb_pca4 — many possible column names
    xgb_candidates = [
        ["PC1","PC2","PC3","PC4"],["XGB_PC1","XGB_PC2","XGB_PC3","XGB_PC4"],
        ["XGBPC1","XGBPC2","XGBPC3","XGBPC4"],["XGB_PCA1","XGB_PCA2","XGB_PCA3","XGB_PCA4"],
        ["XGBPCA1","XGBPCA2","XGBPCA3","XGBPCA4"],["PCS1","PCS2","PCS3","PCS4"],
        ["XP1","XP2","XP3","XP4"],["xgb_pc1","xgb_pc2","xgb_pc3","xgb_pc4"],
    ]
    if cols_available is not None:
        for cols in xgb_candidates:
            if all(c in cols_available for c in cols): return cols
        excluded={"Attack","Label","AttackEncodedKnown","AttackEncodedAll",
                  "Attack_enc","attack","label","attack_enc"}
        numeric=[col for col in df.columns
                 if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric)==4: return numeric
    return xgb_candidates[0]


def load_all_datasets_once(package_root, train_size=5000,
                            data_file="test_2000.csv"):
    """
    Load ALL 4 feature sets ONCE — same pattern as vqc_4feature_UNSW_v3.py.
    MinMaxScaler fit on train_{train_size}.csv, applied to data_file.
    Returns dict: { fs: (X_scaled, y_bin) }
    """
    all_data = {}
    for fs in KNOWN_FS:
        base = Path(package_root) / fs
        train_csv = base / f"train_{train_size}.csv"
        data_csv  = base / data_file
        if not train_csv.exists() or not data_csv.exists():
            print(f"  [SKIP] {fs}: file not found"); continue

        train_df = pd.read_csv(train_csv)
        data_df  = pd.read_csv(data_csv)

        feat_cols = feature_cols_for(fs, train_df)
        missing = [col for col in feat_cols if col not in train_df.columns]
        if missing:
            print(f"  [SKIP] {fs}: columns {missing} not found"); continue

        scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        scaler.fit(train_df[feat_cols].values.astype(np.float32))
        X = scaler.transform(data_df[feat_cols].values.astype(np.float32)).astype(np.float32)

        bin_col = next(c for c in ["Label","label"] if c in data_df.columns)
        y = data_df[bin_col].values.astype(int)

        all_data[fs] = (X, y)
        print(f"  Loaded {fs}: X={X.shape}  feat={feat_cols}")

    return all_data


# ══════════════════════════════════════════════════════════════════
# UNCERTAINTY METRICS
# ══════════════════════════════════════════════════════════════════
def uncertainty_metrics(all_p, y, tau, bins=10):
    """
    all_p: (N_perturb, n_test) array of probabilities for class 1.
    y:     (n_test,) true binary labels (0 or 1).
    Returns σ_epi, H̄, VR, ECE (corrected), C* (corrected).

    ECE fix (Guo et al. 2017, Top-1 confidence calibration):
      Confidence must be the probability assigned to the CHOSEN class,
      not unconditionally the probability of class 1. For a sample
      predicted as class 0 (pm < tau), confidence = 1 - pm, not pm.
      Binning by pm for class 0 predictions causes large artificial
      gaps: a perfectly-calibrated class-0 prediction with pm=0.4
      (confidence=0.6, accuracy=1.0) was previously scored as gap
      |1.0 - 0.4| = 0.6 instead of the correct |1.0 - 0.6| = 0.4.

    C* fix (Geifman & El-Yaniv 2017):
      The original code broke on the first k where cumulative error
      exceeded 5%, making c_star = 0 whenever the single most-certain
      prediction happened to be misclassified. The correct logic is:
      find the MAXIMUM k such that the top-k most certain samples
      maintain cumulative error ≤ 5%, without stopping early.
    """
    N, n = all_p.shape
    pm = all_p.mean(0)
    sg = all_p.std(0)
    eps = 1e-7
    pmc = np.clip(pm, eps, 1 - eps)
    H = -pmc * np.log2(pmc) - (1 - pmc) * np.log2(1 - pmc)
    vr = 1 - np.maximum((all_p >= tau).sum(0), (all_p < tau).sum(0)) / N
    preds = (pm >= tau).astype(int)

    # ── Corrected ECE ─────────────────────────────────────────────
    # confidence = probability assigned to the CHOSEN (predicted) class
    confidences = np.where(preds == 1, pm, 1 - pm)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (confidences >= lo) & (confidences < hi)
        if not m.any():
            continue
        bin_acc  = float((preds[m] == y[m]).mean())
        bin_conf = float(confidences[m].mean())
        ece += (m.sum() / n) * abs(bin_acc - bin_conf)

    # ── Corrected C* ──────────────────────────────────────────────
    # Track the MAXIMUM k where cumulative error ≤ 5% — do NOT break
    # on first violation, since a single wrong early prediction would
    # otherwise zero out the metric for an otherwise excellent model.
    cert  = 1 - H
    order = np.argsort(-cert)
    c_star = 0.0
    for k in range(1, n + 1):
        if float((preds[order[:k]] != y[order[:k]]).mean()) <= 0.05:
            c_star = k / n   # track highest valid coverage found

    return dict(sigma_epi_mean=float(sg.mean()),
                sigma_epi_std=float(sg.std()),
                entropy_mean=float(H.mean()),
                entropy_std=float(H.std()),
                vr_mean=float(vr.mean()),
                ece=float(ece),
                deploy_coverage=float(c_star))


def reorder_by_combined_summary(df, results_dir):
    for fname in ["combined_summary_metrics.csv","new_format_combined_result.csv"]:
        p=Path(results_dir)/fname
        if p.exists():
            ref=pd.read_csv(p); ref["_ord"]=range(len(ref))
            df=df.merge(ref[["feature_set","ansatz","encoding","_ord"]],
                        on=["feature_set","ansatz","encoding"],how="left")
            df=df.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
            print(f"  Row order matched to {fname}")
            return df
    return df


# ══════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════
def worker(gpu_id, jobs, preloaded_data, n_perturb, result_q, pl_backend):
    # CRITICAL: limit BLAS/OpenMP threads to 1 per process (see bootstrap script comment)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    np.random.seed(42)

    for job in jobs:
        t0=time.perf_counter(); rk=job["run_key"]
        try:
            # ── Load best model ───────────────────────────────────────
            ck=torch.load(job["best_f"],map_location="cpu",weights_only=False)
            cfg=ck["config"]
            dev,diff_method,dev_name=make_pl_device(int(cfg["n_qubits"]), backend=pl_backend)
            model=build_model(cfg,dev,diff_method)
            model.load_state_dict(ck["model_state"])
            model.eval()
            print(f"  [GPU{gpu_id}] {rk}: model on {dev_name}",flush=True)

            # ── Adaptive σ from gradient norm (final_model.pt) ────────
            sigma=0.01
            ff=Path(job["subdir"])/f"{rk}_final_model.pt"
            if ff.exists():
                try:
                    fck=torch.load(ff,map_location="cpu",weights_only=False)
                    gn=fck.get("history",{}).get("grad_norm",[])
                    if len(gn)>=3: sigma=max(float(np.mean(gn[-3:])),0.01)
                    elif gn:       sigma=max(float(np.mean(gn)),0.01)
                except: pass

            # ── Use pre-loaded scaled data (loaded ONCE in main) ─────
            if job["fs"] not in preloaded_data:
                raise KeyError(f"No data for feature_set={job['fs']}")
            X_test, y_test = preloaded_data[job["fs"]]

            # ── Training threshold ────────────────────────────────────
            tau=float(ck.get("best_threshold",0.5))
            rf=Path(job["subdir"])/f"{rk}_result.json"
            if rf.exists():
                try: tau=float(json.load(open(rf)).get("metrics",{}).get("threshold",tau))
                except: pass

            # ── N=30 perturbations using deepcopy + predict_proba ─────
            # Same function as training — no manual circuit reconstruction
            all_p=np.zeros((n_perturb,len(y_test)))
            for ni in range(n_perturb):
                m_perturbed=copy.deepcopy(model)
                with torch.no_grad():
                    noise=torch.randn_like(m_perturbed.quantum_weights)*sigma
                    m_perturbed.quantum_weights.copy_(model.quantum_weights+noise)
                all_p[ni]=predict_proba(m_perturbed,X_test,batch_size=256)
                del m_perturbed

            met=uncertainty_metrics(all_p,y_test,tau)
            n_params=model.quantum_weights.numel()
            elapsed=round(time.perf_counter()-t0,1)

            row=dict(feature_set=job["fs"],train_size=job["ts"],
                     ansatz=job["ansatz"],encoding=job["encoding"],
                     sigma_noise=float(sigma),n_params=n_params,
                     elapsed_s=elapsed,gpu_id=gpu_id,**met)
            result_q.put({"ok":True,"rk":rk,"row":row,"gpu":gpu_id})
            print(f"  [GPU{gpu_id}] ✓ {rk}  "
                  f"σ={sigma:.5f}  σ_epi={met['sigma_epi_mean']:.5f}  "
                  f"H̄={met['entropy_mean']:.4f}  ECE={met['ece']:.4f}  "
                  f"C*={met['deploy_coverage']:.4f}  ({elapsed}s)",flush=True)

        except Exception:
            import traceback
            result_q.put({"ok":False,"rk":rk,"gpu":gpu_id,"err":traceback.format_exc()})
            print(f"  [GPU{gpu_id}] ✗ {rk}",flush=True)
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except: pass

    result_q.put({"done":True,"gpu":gpu_id})


def discover(results_dir, top_n_csv=None):
    jobs=[]
    for sd in sorted(Path(results_dir).iterdir()):
        if not sd.is_dir(): continue
        fs=next((f for f in KNOWN_FS if sd.name.startswith(f)),None)
        if not fs: continue
        rest=sd.name[len(fs):].lstrip("_").split("_")
        ts=next((int(x) for x in rest if x.isdigit()),5000)
        for bf in sorted(sd.glob("*_best_model.pt")):
            rk=bf.stem.replace("_best_model",""); rk_p=rk.split("__")
            jobs.append(dict(run_key=rk,fs=fs,ts=ts,subdir=str(sd),best_f=str(bf),
                             ansatz=rk_p[0],encoding=rk_p[1] if len(rk_p)>1 else ""))

    if top_n_csv:
        import pandas as pd
        sel = pd.read_csv(top_n_csv)
        sel_keys = set(zip(sel["feature_set"], sel["ansatz"], sel["encoding"]))
        before = len(jobs)
        jobs = [j for j in jobs if (j["fs"], j["ansatz"], j["encoding"]) in sel_keys]
        print(f"  --top_n_selection applied: {before} -> {len(jobs)} models "
              f"(from {top_n_csv})")
    return jobs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results_dir",  required=True)
    ap.add_argument("--package_root", required=True)
    ap.add_argument("--gpu_ids",      default="5,6,7")
    ap.add_argument("--jobs_per_gpu", type=int, default=1,
                   help="Parallel worker processes per GPU label (CPU-bound, label only)")
    ap.add_argument("--n_perturb",    type=int,default=30)
    ap.add_argument("--pl_backend", choices=["lightning.gpu","default.qubit","default.mixed"],
                   default="default.qubit",
                   help="PennyLane backend for inference. default.qubit recommended for speed; default.mixed for noise-study continuity; lightning.gpu matches training exactly but is very slow here.")
    ap.add_argument("--top_n_selection", default=None,
                   help="Path to a topN_selection.csv (from select_top10_models.py) to restrict the run to only those models instead of all discovered models.")
    args=ap.parse_args()

    base_gpu_ids=[int(g) for g in args.gpu_ids.split(",")]
    # Replicate each GPU id jobs_per_gpu times -> more parallel CPU workers
    # e.g. gpu_ids=[1,2,3,4], jobs_per_gpu=2 -> 8 worker slots: 1,1,2,2,3,3,4,4
    gpu_ids=[g for g in base_gpu_ids for _ in range(args.jobs_per_gpu)]
    print(f"  Worker slots: {len(gpu_ids)} ({len(base_gpu_ids)} GPU labels x {args.jobs_per_gpu} jobs_per_gpu)")
    jobs=discover(args.results_dir, top_n_csv=args.top_n_selection)
    print(f"\nFound {len(jobs)} models | GPUs: {gpu_ids} | N_PERTURB={args.n_perturb}")
    if not jobs: print("No models found."); return

    # Load ALL 4 feature sets ONCE — same pattern as VQC training code
    print("\nLoading all feature set data (once, shared across all models)...")
    preloaded_data = load_all_datasets_once(
        args.package_root, train_size=5000, data_file="test_2000.csv")
    print(f"  Loaded {len(preloaded_data)} feature sets\n")

    splits=[jobs[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    for i,gid in enumerate(gpu_ids): print(f"  GPU{gid}: {len(splits[i])} models")

    ctx=mp.get_context("spawn"); rq=ctx.Queue()
    procs=[ctx.Process(target=worker,
                       args=(gid,splits[i],preloaded_data,args.n_perturb,rq,args.pl_backend),daemon=False)
           for i,gid in enumerate(gpu_ids)]
    for p in procs: p.start()

    done=0; rows=[]; t0=time.perf_counter()
    while done<len(gpu_ids):
        import queue as Q
        try: msg=rq.get(timeout=300)
        except Q.Empty: continue
        if msg.get("done"):  done+=1
        elif msg.get("ok"):  rows.append(msg["row"])
        else: print(f"  ✗ {msg.get('rk')}:\n{msg.get('err','')}")
    for p in procs: p.join()

    if not rows: print("No results."); return

    df=pd.DataFrame(rows)
    df=reorder_by_combined_summary(df,args.results_dir)
    out=Path(args.results_dir)/"uncertainty_results.csv"
    df.to_csv(out,index=False)
    print(f"\n✓ Done in {time.perf_counter()-t0:.1f}s → {out}")
    print(df[["feature_set","ansatz","encoding","sigma_epi_mean",
              "entropy_mean","ece","deploy_coverage"]].head(5).to_string(index=False))

if __name__=="__main__": main()