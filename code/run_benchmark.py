"""
run_benchmark.py
================
Main benchmark runner for the CHSH Bell inequality noise threshold analysis.

Sweeps noise rate p in [0, 0.5] for all combinations of:
  - Bell states: Φ+, Φ-, Ψ+, Ψ-
  - Noise types: depolarizing, dephasing, amplitude_damping

Saves results to:
  /Users/arhaansareen/bell-paper/code/results/bell_results.json
  /Users/arhaansareen/bell-paper/code/results/bell_results.csv

Usage:
    python run_benchmark.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure the code directory is on the path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from chsh_core import (
    BELL_STATES,
    NOISE_CHANNELS,
    OPTIMAL_ANGLES,
    CLASSICAL_BOUND,
    TSIRELSON_BOUND,
    chsh_value,
    find_threshold,
    analytical_depolarizing_S,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(CODE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_NOISE_STEPS = 50
P_MAX = 0.5
P_VALUES = np.linspace(0.0, P_MAX, N_NOISE_STEPS)

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep() -> pd.DataFrame:
    """
    Run the full parameter sweep over all Bell states and noise channels.

    Returns:
        DataFrame with columns:
            bell_state, noise_type, p, S, above_classical
    """
    records = []
    total = len(BELL_STATES) * len(NOISE_CHANNELS) * N_NOISE_STEPS
    done = 0
    t0 = time.time()

    for state_name, state_fn in BELL_STATES.items():
        rho0 = state_fn()
        angles = OPTIMAL_ANGLES[state_name]

        for noise_name, noise_fn in NOISE_CHANNELS.items():
            for p in P_VALUES:
                rho_noisy = noise_fn(rho0, p)
                S = chsh_value(
                    rho_noisy,
                    angles["a"], angles["a_prime"],
                    angles["b"], angles["b_prime"],
                )
                records.append({
                    "bell_state": state_name,
                    "noise_type": noise_name,
                    "p": float(p),
                    "S": float(S),
                    "above_classical": bool(S > CLASSICAL_BOUND),
                })
                done += 1

            elapsed = time.time() - t0
            print(
                f"  [{done:4d}/{total}]  {state_name:5s}  {noise_name:20s}"
                f"  elapsed={elapsed:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(records)
    return df


def compute_thresholds(df: pd.DataFrame) -> dict:
    """
    Find p* for each (bell_state, noise_type) combination.

    Returns:
        Nested dict: thresholds[state][noise] = p*
    """
    thresholds = {}
    for state in BELL_STATES:
        thresholds[state] = {}
        for noise in NOISE_CHANNELS:
            mask = (df["bell_state"] == state) & (df["noise_type"] == noise)
            sub = df[mask].sort_values("p")
            p_star = find_threshold(sub["p"].values, sub["S"].values)
            thresholds[state][noise] = p_star
    return thresholds


def main():
    print("=" * 60)
    print("CHSH Bell Inequality Noise Benchmark")
    print(f"  Bell states : {list(BELL_STATES.keys())}")
    print(f"  Noise types : {list(NOISE_CHANNELS.keys())}")
    print(f"  p steps     : {N_NOISE_STEPS}  (0 → {P_MAX})")
    print(f"  Classical bound : {CLASSICAL_BOUND}")
    print(f"  Tsirelson bound : {TSIRELSON_BOUND:.6f}")
    print("=" * 60)

    # Run sweep
    print("\nRunning sweep...")
    df = run_sweep()
    print(f"\nSweep complete. {len(df)} data points computed.")

    # Compute thresholds
    thresholds = compute_thresholds(df)
    analytical_p_star = 1.0 - 1.0 / np.sqrt(2)

    print("\nCritical noise rates p* (S crosses classical bound 2.0):")
    print(f"  Analytical (depolarizing): {analytical_p_star:.6f}")
    for state in BELL_STATES:
        for noise in NOISE_CHANNELS:
            p_star = thresholds[state][noise]
            print(f"  {state:5s}  {noise:20s}  p* = {p_star:.6f}" if not np.isnan(p_star)
                  else f"  {state:5s}  {noise:20s}  p* = N/A (never crosses bound)")

    # Build summary dict for JSON
    summary = {
        "meta": {
            "p_values": P_VALUES.tolist(),
            "n_steps": N_NOISE_STEPS,
            "p_max": P_MAX,
            "classical_bound": CLASSICAL_BOUND,
            "tsirelson_bound": TSIRELSON_BOUND,
            "analytical_p_star_depolarizing": analytical_p_star,
        },
        "thresholds": {
            state: {
                noise: (None if np.isnan(v) else v)
                for noise, v in noise_dict.items()
            }
            for state, noise_dict in thresholds.items()
        },
        "S_vs_p": {},
    }

    # Store S vs p arrays per (state, noise)
    for state in BELL_STATES:
        summary["S_vs_p"][state] = {}
        for noise in NOISE_CHANNELS:
            mask = (df["bell_state"] == state) & (df["noise_type"] == noise)
            sub = df[mask].sort_values("p")
            summary["S_vs_p"][state][noise] = sub["S"].tolist()

    # Also add analytical S for depolarizing
    summary["analytical_depolarizing_S"] = analytical_depolarizing_S(P_VALUES).tolist()

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "bell_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON → {json_path}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "bell_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV  → {csv_path}")

    return df, summary


if __name__ == "__main__":
    main()
