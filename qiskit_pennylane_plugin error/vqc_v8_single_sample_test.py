"""
vqc_v8_single_sample_test.py
ISOLATION TEST — removes qml.batch_input entirely. Tests ONLY whether
a from_qiskit-converted circuit supports a basic forward+backward pass
on a SINGLE sample. This isolates whether the problem is:
  (a) batch_input's incompatibility with from_qiskit circuits, or
  (b) something more fundamental in the conversion/binding itself.

Run:
  python vqc_v8_single_sample_test.py --n_qubits 4
"""
import time, traceback
import numpy as np
import torch

def test_single_sample(device_name, diff_method, n_qubits=4):
    result = dict(device=device_name, diff_method=diff_method,
                   single_forward_ok=False, single_backward_ok=False,
                   manual_loop_epoch_time_s=None, error=None)
    try:
        import pennylane as qml
        # Use the NEWER FUNCTION-based API (not the deprecated
        # BlueprintCircuit classes) — returns an already-decomposed,
        # plain QuantumCircuit, avoiding lazy-gate wrappers that the
        # PennyLane converter may silently drop.
        try:
            from qiskit.circuit.library import zz_feature_map, real_amplitudes
            qiskit_encoding = zz_feature_map(feature_dimension=n_qubits, reps=1)
            qiskit_ansatz   = real_amplitudes(num_qubits=n_qubits, reps=1)
        except ImportError:
            # Fallback for older qiskit without the function-based API
            from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
            qiskit_encoding = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
            qiskit_ansatz   = RealAmplitudes(num_qubits=n_qubits, reps=1)

        qc = qiskit_encoding.compose(qiskit_ansatz)
        qc = qc.decompose()   # safety net: force full expansion to
                               # elementary gates before conversion
        pl_circuit = qml.from_qiskit(qc)

        n_enc = qiskit_encoding.num_parameters
        n_ans = qiskit_ansatz.num_parameters
        result["n_enc_params"] = n_enc
        result["n_ans_params"] = n_ans

        dev = qml.device(device_name, wires=n_qubits, shots=None)

        # NO batch_input — single sample, direct call
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def quantum_model(x_single, weights):
            pl_circuit(x_single, weights)
            return qml.expval(qml.PauliZ(0))

        x_one = torch.rand(n_enc, dtype=torch.float32) * np.pi
        weights = torch.randn(n_ans, dtype=torch.float32, requires_grad=True)

        t0 = time.perf_counter()
        out = quantum_model(x_one, weights)
        result["single_forward_ok"] = True
        result["forward_time_s"] = round(time.perf_counter()-t0, 4)

        loss = (torch.sigmoid(out) - 1.0)**2
        loss.backward()
        result["single_backward_ok"] = weights.grad is not None
        result["grad_norm"] = float(weights.grad.norm()) if weights.grad is not None else None

        # Manual per-sample loop over 100 "samples" — measures the REAL
        # cost of training this way if batch_input cannot be used.
        X_fake = torch.rand(100, n_enc, dtype=torch.float32) * np.pi
        y_fake = torch.randint(0, 2, (100,)).float()
        opt = torch.optim.Adam([weights], lr=0.01)

        t_ep = time.perf_counter()
        opt.zero_grad()
        total_loss = 0.0
        for i in range(100):
            out_i = quantum_model(X_fake[i], weights)
            total_loss = total_loss + (torch.sigmoid(out_i) - y_fake[i])**2
        total_loss = total_loss / 100
        total_loss.backward()
        opt.step()
        result["manual_loop_epoch_time_s"] = round(time.perf_counter() - t_ep, 3)
        result["manual_loop_loss"] = float(total_loss.item())

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


def main():
    print("="*70)
    print("  SINGLE-SAMPLE ISOLATION TEST (no batch_input)")
    print("="*70)

    combos = [
        ("default.qubit", "backprop"),
        ("lightning.gpu", "adjoint"),
    ]

    for device_name, diff_method in combos:
        print(f"\n{'─'*70}")
        print(f"Testing: device={device_name}  diff_method={diff_method}")
        print(f"{'─'*70}")
        r = test_single_sample(device_name, diff_method)
        if r["error"]:
            print(f"  ✗ FAILED: {r['error']}")
            print(f"\n  Full traceback:\n{r['traceback']}")
        else:
            print(f"  ✓ single_forward_ok={r['single_forward_ok']}  "
                  f"single_backward_ok={r['single_backward_ok']}")
            print(f"  forward_time={r.get('forward_time_s')}s  "
                  f"grad_norm={r.get('grad_norm')}")
            print(f"\n  MANUAL 100-sample loop (1 epoch, Python for-loop,")
            print(f"  the realistic cost if batch_input cannot be used):")
            print(f"    time = {r.get('manual_loop_epoch_time_s')}s  "
                  f"loss = {r.get('manual_loop_loss')}")
            print(f"\n  Projected for 5000 samples: "
                  f"~{r.get('manual_loop_epoch_time_s', 0) * 50:.1f}s/epoch")
            print(f"  Projected for 20 epochs × 112 models: "
                  f"~{r.get('manual_loop_epoch_time_s', 0) * 50 * 20 * 112 / 3600:.1f} hours")

if __name__ == "__main__":
    main()
