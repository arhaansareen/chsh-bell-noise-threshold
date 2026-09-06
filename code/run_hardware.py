"""
run_hardware.py
===============
Runs real CHSH Bell inequality measurements on IBM Quantum hardware.

For each of the 4 Bell states we measure 4 correlator settings (a,b),
(a,b'), (a',b), (a',b') using Ry rotations before measurement.

Results are saved to:
  results/hardware_results.json

Usage:
    export IBM_QUANTUM_TOKEN="..."
    python run_hardware.py
"""

import json
import math
import os
import sys
import time

import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)
from chsh_core import BELL_STATES, OPTIMAL_ANGLES, CLASSICAL_BOUND, TSIRELSON_BOUND

BACKEND_NAME = "ibm_fez"
SHOTS = 4096

# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def _build_bell_circuit(state_name: str, theta_a_deg: float, theta_b_deg: float):
    """
    Build a 2-qubit circuit that:
      1. Prepares the requested Bell state
      2. Rotates qubit 0 by Ry(theta_a) and qubit 1 by Ry(theta_b)
      3. Measures both qubits
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)

    # --- Bell state preparation ---
    if state_name == "Phi+":
        qc.h(0)
        qc.cx(0, 1)
    elif state_name == "Phi-":
        qc.h(0)
        qc.cx(0, 1)
        qc.z(0)
    elif state_name == "Psi+":
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
    elif state_name == "Psi-":
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
        qc.z(0)
    else:
        raise ValueError(f"Unknown Bell state: {state_name}")

    # --- Measurement basis rotations ---
    theta_a_rad = math.radians(theta_a_deg)
    theta_b_rad = math.radians(theta_b_deg)
    qc.ry(-theta_a_rad, 0)
    qc.ry(-theta_b_rad, 1)

    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def _correlator_from_counts(counts: dict) -> float:
    """
    E(a, b) = (N_00 + N_11 - N_01 - N_10) / N_total
    Qiskit bit string order: qubit 1 is leftmost, qubit 0 is rightmost.
    """
    total = sum(counts.values())
    e = 0
    for bitstring, n in counts.items():
        b0 = int(bitstring[-1])   # qubit 0
        b1 = int(bitstring[-2]) if len(bitstring) > 1 else 0  # qubit 1
        parity = (-1) ** (b0 + b1)
        e += parity * n
    return e / total


def chsh_from_correlators(e_ab, e_ab_prime, e_a_prime_b, e_a_prime_b_prime) -> float:
    return e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime


# ---------------------------------------------------------------------------
# Main hardware runner
# ---------------------------------------------------------------------------

def run_hardware_chsh(token: str):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.compiler import transpile

    print("Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(BACKEND_NAME)
    print(f"Backend: {backend.name} | qubits: {backend.num_qubits}")
    print(f"Shots per circuit: {SHOTS}\n")

    results = {}
    all_circuits = []
    circuit_labels = []

    # Build all circuits upfront for batched transpilation
    for state_name, angles in OPTIMAL_ANGLES.items():
        a, a_prime = angles["a"], angles["a_prime"]
        b, b_prime = angles["b"], angles["b_prime"]

        settings = [
            ("ab",       a,       b),
            ("ab_prime", a,       b_prime),
            ("a_prime_b", a_prime, b),
            ("a_prime_b_prime", a_prime, b_prime),
        ]
        for label, ta, tb in settings:
            qc = _build_bell_circuit(state_name, ta, tb)
            qc.name = f"{state_name}_{label}"
            all_circuits.append(qc)
            circuit_labels.append((state_name, label))

    print(f"Transpiling {len(all_circuits)} circuits...")
    t0 = time.time()
    transpiled = transpile(all_circuits, backend=backend, optimization_level=1)
    print(f"Transpilation done in {time.time() - t0:.1f}s\n")

    print("Submitting job to hardware...")
    sampler = Sampler(backend)
    job = sampler.run(transpiled, shots=SHOTS)
    print(f"Job ID: {job.job_id()}")
    print("Waiting for results (this may take a few minutes)...")

    job_result = job.result()
    print("Job complete.\n")

    # Parse results
    for i, (state_name, label) in enumerate(circuit_labels):
        pub_result = job_result[i]
        counts = pub_result.data.c.get_counts()

        if state_name not in results:
            results[state_name] = {}
        results[state_name][label] = {
            "correlator": _correlator_from_counts(counts),
            "counts": counts,
        }

    # Compute CHSH S per Bell state
    summary = {}
    print("=" * 50)
    print("CHSH Hardware Results")
    print(f"  Classical bound : {CLASSICAL_BOUND}")
    print(f"  Tsirelson bound : {TSIRELSON_BOUND:.6f}")
    print("=" * 50)

    for state_name in OPTIMAL_ANGLES:
        r = results[state_name]
        e_ab          = r["ab"]["correlator"]
        e_ab_prime    = r["ab_prime"]["correlator"]
        e_a_prime_b   = r["a_prime_b"]["correlator"]
        e_a_prime_b_prime = r["a_prime_b_prime"]["correlator"]

        S = chsh_from_correlators(e_ab, e_ab_prime, e_a_prime_b, e_a_prime_b_prime)
        violates = abs(S) > CLASSICAL_BOUND

        print(f"\n{state_name}:")
        print(f"  E(a,b)       = {e_ab:.4f}")
        print(f"  E(a,b')      = {e_ab_prime:.4f}")
        print(f"  E(a',b)      = {e_a_prime_b:.4f}")
        print(f"  E(a',b')     = {e_a_prime_b_prime:.4f}")
        print(f"  S            = {S:.4f}  |S|={abs(S):.4f}  {'VIOLATES' if violates else 'does NOT violate'} classical bound")

        summary[state_name] = {
            "S": S,
            "S_abs": abs(S),
            "violates_classical_bound": violates,
            "correlators": {
                "E_ab": e_ab,
                "E_ab_prime": e_ab_prime,
                "E_a_prime_b": e_a_prime_b,
                "E_a_prime_b_prime": e_a_prime_b_prime,
            },
            "raw": results[state_name],
        }

    output = {
        "meta": {
            "backend": BACKEND_NAME,
            "shots": SHOTS,
            "job_id": job.job_id(),
            "classical_bound": CLASSICAL_BOUND,
            "tsirelson_bound": TSIRELSON_BOUND,
        },
        "results": summary,
    }

    out_path = os.path.join(RESULTS_DIR, "hardware_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    return output


if __name__ == "__main__":
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        print("ERROR: IBM_QUANTUM_TOKEN not set.")
        sys.exit(1)
    run_hardware_chsh(token)
