"""
chsh_core.py
============
Core physics routines for the CHSH Bell inequality benchmark.

Provides:
- Bell state preparation (density matrices)
- Noise channel application
- Correlator E(θ_A, θ_B) computation via Tr[(σ_α⊗σ_β) ρ]
- CHSH parameter S computation
- Threshold p* finding via linear interpolation

All density matrices are 4×4 complex numpy arrays in the
computational basis {|00⟩, |01⟩, |10⟩, |11⟩}.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Maximally mixed 4×4 state
I4_normalized = np.eye(4, dtype=complex) / 4.0


# ---------------------------------------------------------------------------
# Bell state density matrices
# ---------------------------------------------------------------------------

def bell_phi_plus() -> np.ndarray:
    """|Φ+⟩ = (|00⟩+|11⟩)/√2  →  density matrix."""
    v = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return np.outer(v, v.conj())


def bell_phi_minus() -> np.ndarray:
    """|Φ-⟩ = (|00⟩-|11⟩)/√2  →  density matrix."""
    v = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    return np.outer(v, v.conj())


def bell_psi_plus() -> np.ndarray:
    """|Ψ+⟩ = (|01⟩+|10⟩)/√2  →  density matrix."""
    v = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    return np.outer(v, v.conj())


def bell_psi_minus() -> np.ndarray:
    """|Ψ-⟩ = (|01⟩-|10⟩)/√2  →  density matrix."""
    v = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    return np.outer(v, v.conj())


BELL_STATES = {
    "Phi+": bell_phi_plus,
    "Phi-": bell_phi_minus,
    "Psi+": bell_psi_plus,
    "Psi-": bell_psi_minus,
}


# ---------------------------------------------------------------------------
# Noise channels
# ---------------------------------------------------------------------------

def apply_depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Global depolarizing channel on two qubits:
        ρ → (1-p)ρ + p·I/4

    The maximally mixed state I/4 has equal weight on all four
    computational-basis states.

    Args:
        rho: 4×4 density matrix.
        p:   Depolarizing rate in [0, 1].

    Returns:
        Noisy 4×4 density matrix.
    """
    return (1.0 - p) * rho + p * I4_normalized


def _apply_single_qubit_kraus(rho: np.ndarray, kraus: list, qubit: int) -> np.ndarray:
    """
    Apply a single-qubit Kraus channel to one qubit of a 2-qubit system.

    Args:
        rho:   4×4 density matrix.
        kraus: List of 2×2 Kraus operators {K_i} satisfying Σ K†K = I.
        qubit: 0 (first/left qubit) or 1 (second/right qubit).

    Returns:
        4×4 density matrix after the channel.
    """
    out = np.zeros((4, 4), dtype=complex)
    for K in kraus:
        if qubit == 0:
            K_full = np.kron(K, I2)
        else:
            K_full = np.kron(I2, K)
        out += K_full @ rho @ K_full.conj().T
    return out


def apply_local_dephasing(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Independent dephasing (Z-channel) on each qubit at rate p.

        Single-qubit dephasing: ρ_q → (1-p)ρ_q + p·Z ρ_q Z

    Kraus operators: K0 = √(1-p)·I,  K1 = √p·Z

    Applied independently to qubit 0 then qubit 1.

    Args:
        rho: 4×4 density matrix.
        p:   Dephasing rate per qubit in [0, 1].

    Returns:
        Noisy 4×4 density matrix.
    """
    K0 = np.sqrt(1.0 - p) * I2
    K1 = np.sqrt(p) * Z
    kraus = [K0, K1]
    rho_noisy = _apply_single_qubit_kraus(rho, kraus, qubit=0)
    rho_noisy = _apply_single_qubit_kraus(rho_noisy, kraus, qubit=1)
    return rho_noisy


def apply_amplitude_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
    """
    Independent amplitude damping on each qubit at rate γ.

    Kraus operators: K0 = [[1,0],[0,√(1-γ)]],  K1 = [[0,√γ],[0,0]]

    Applied independently to qubit 0 then qubit 1.

    Args:
        rho:   4×4 density matrix.
        gamma: Damping rate per qubit in [0, 1].

    Returns:
        Noisy 4×4 density matrix.
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1.0 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    kraus = [K0, K1]
    rho_noisy = _apply_single_qubit_kraus(rho, kraus, qubit=0)
    rho_noisy = _apply_single_qubit_kraus(rho_noisy, kraus, qubit=1)
    return rho_noisy


NOISE_CHANNELS = {
    "depolarizing": apply_depolarizing,
    "dephasing": apply_local_dephasing,
    "amplitude_damping": apply_amplitude_damping,
}


# ---------------------------------------------------------------------------
# Measurement / correlator
# ---------------------------------------------------------------------------

def rotation_y(theta_rad: float) -> np.ndarray:
    """2×2 Ry(θ) rotation matrix: Ry(θ) = exp(-i θ/2 Y)."""
    c = np.cos(theta_rad / 2.0)
    s = np.sin(theta_rad / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def correlator(rho: np.ndarray, theta_A_deg: float, theta_B_deg: float) -> float:
    """
    Compute E(θ_A, θ_B) = Tr[(σ_n_A ⊗ σ_n_B) ρ]

    where σ_n = cos(θ)Z + sin(θ)X  (measurement in the XZ plane).

    Equivalently: rotate each qubit by Ry(-θ) then measure ⟨ZZ⟩.

    ⟨ZZ⟩ after rotation = Tr[(Z⊗Z) · (Ry(-θ_A)⊗Ry(-θ_B)) ρ (Ry(-θ_A)⊗Ry(-θ_B))†]

    Args:
        rho:         4×4 density matrix.
        theta_A_deg: Alice's measurement angle in degrees.
        theta_B_deg: Bob's measurement angle in degrees.

    Returns:
        Real-valued correlator in [-1, 1].
    """
    theta_A = np.deg2rad(theta_A_deg)
    theta_B = np.deg2rad(theta_B_deg)

    # Rotation matrices Ry(-θ) to change basis
    R_A = rotation_y(-theta_A)
    R_B = rotation_y(-theta_B)

    # Full 4×4 rotation
    R = np.kron(R_A, R_B)

    # Rotate the state
    rho_rot = R @ rho @ R.conj().T

    # ZZ observable: eigenvalue +1 for |00⟩,|11⟩ and -1 for |01⟩,|10⟩
    ZZ = np.kron(Z, Z)
    val = np.trace(ZZ @ rho_rot)
    return float(np.real(val))


def chsh_value(rho: np.ndarray,
               a: float, a_prime: float,
               b: float, b_prime: float) -> float:
    """
    Compute the CHSH parameter:
        S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|

    Args:
        rho:     4×4 density matrix.
        a, a':   Alice's angles in degrees.
        b, b':   Bob's angles in degrees.

    Returns:
        S ≥ 0.
    """
    E_ab   = correlator(rho, a, b)
    E_ab_p = correlator(rho, a, b_prime)
    E_a_pb = correlator(rho, a_prime, b)
    E_a_pb_p = correlator(rho, a_prime, b_prime)
    S = abs(E_ab - E_ab_p + E_a_pb + E_a_pb_p)
    return S


# ---------------------------------------------------------------------------
# Optimal CHSH angles for each Bell state
# ---------------------------------------------------------------------------

# Standard optimal angles (in degrees) for maximal CHSH violation.
# For |Φ+⟩: a=0, a'=90, b=45, b'=135  →  S = 2√2
# For |Φ-⟩: same angles work (gives S = 2√2, signs flip but absolute value same)
# For |Ψ+⟩: a=0, a'=90, b=45, b'=135  gives S = 2√2
# For |Ψ-⟩: a=0, a'=90, b=45, b'=135  gives S = 2√2

OPTIMAL_ANGLES = {
    # |Φ+⟩: E(θ_A,θ_B) = +cos(θ_A - θ_B)
    # Standard angles give S = 2√2 with E(0,45)-E(0,135)+E(90,45)+E(90,135)
    "Phi+": dict(a=0.0,   a_prime=90.0,  b=45.0,  b_prime=135.0),
    # |Φ-⟩: E(θ_A,θ_B) = -cos(θ_A - θ_B)  → swap b ↔ b' to compensate sign
    "Phi-": dict(a=0.0,   a_prime=90.0,  b=135.0, b_prime=45.0),
    # |Ψ+⟩: same sign flip as Φ-, swap b ↔ b'
    "Psi+": dict(a=0.0,   a_prime=90.0,  b=135.0, b_prime=45.0),
    # |Ψ-⟩: E(θ_A,θ_B) = +cos(θ_A - θ_B) (same sign as Φ+)
    "Psi-": dict(a=0.0,   a_prime=90.0,  b=45.0,  b_prime=135.0),
}

CLASSICAL_BOUND = 2.0
TSIRELSON_BOUND = 2.0 * np.sqrt(2)


# ---------------------------------------------------------------------------
# Threshold finding
# ---------------------------------------------------------------------------

def find_threshold(p_values: np.ndarray, S_values: np.ndarray,
                   classical_bound: float = CLASSICAL_BOUND) -> float:
    """
    Find the critical noise rate p* where S crosses the classical bound.

    Uses linear interpolation between the last point above and first
    point below the bound.

    Args:
        p_values:       Array of noise rates.
        S_values:       Corresponding CHSH S values.
        classical_bound: Default 2.0.

    Returns:
        p* (float) or np.nan if S never crosses the bound.
    """
    above = S_values > classical_bound
    if not np.any(above):
        return np.nan
    if np.all(above):
        return np.nan

    # Find index of first crossing (False after a True)
    for i in range(len(above) - 1):
        if above[i] and not above[i + 1]:
            # Linear interpolation
            p0, S0 = p_values[i], S_values[i]
            p1, S1 = p_values[i + 1], S_values[i + 1]
            p_star = p0 + (classical_bound - S0) * (p1 - p0) / (S1 - S0)
            return float(p_star)

    return np.nan


def analytical_depolarizing_S(p: np.ndarray) -> np.ndarray:
    """
    Analytical CHSH value under global depolarizing noise:
        S(p) = (1-p) · 2√2

    Args:
        p: Noise rate(s) as scalar or array.

    Returns:
        S value(s).
    """
    return (1.0 - np.asarray(p)) * TSIRELSON_BOUND
