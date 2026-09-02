"""
make_figures.py
===============
Generate all six figures for the CHSH Bell inequality paper.

Reads pre-computed results from:
    /Users/arhaansareen/bell-paper/code/results/bell_results.json

Saves figures to:
    /Users/arhaansareen/bell-paper/figures/fig{1-6}.pdf  (and .png at 300 DPI)

Figures:
  fig1 — Bell circuit schematic (matplotlib manual drawing)
  fig2 — S vs p for depolarizing on |Φ+⟩ (simulated + analytical + bounds)
  fig3 — All three noise types on |Φ+⟩
  fig4 — All four Bell states under depolarizing
  fig5 — Angle heatmap for |Φ+⟩ noiseless
  fig6 — Threshold p* bar chart for each noise model

Usage:
    python make_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import numpy as np
import seaborn as sns

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
FIGURES_DIR = "/Users/arhaansareen/bell-paper/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)
from chsh_core import (
    CLASSICAL_BOUND,
    TSIRELSON_BOUND,
    chsh_value,
    correlator,
    bell_phi_plus,
    find_threshold,
    analytical_depolarizing_S,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

DPI = 300

NOISE_COLORS = {
    "depolarizing":    "#2166AC",
    "dephasing":       "#D6604D",
    "amplitude_damping": "#4DAC26",
}
NOISE_LABELS = {
    "depolarizing":    "Global depolarizing",
    "dephasing":       "Local dephasing",
    "amplitude_damping": "Amplitude damping",
}
BELL_COLORS = {
    "Phi+": "#1B7837",
    "Phi-": "#762A83",
    "Psi+": "#E08214",
    "Psi-": "#2166AC",
}
BELL_LABELS = {
    "Phi+": r"$|\Phi^+\rangle$",
    "Phi-": r"$|\Phi^-\rangle$",
    "Psi+": r"$|\Psi^+\rangle$",
    "Psi-": r"$|\Psi^-\rangle$",
}


def save_fig(fig, name: str):
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved {name}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results():
    path = os.path.join(RESULTS_DIR, "bell_results.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 — Bell circuit schematic
# ---------------------------------------------------------------------------

def fig1_circuit():
    """
    Manual matplotlib drawing of the CHSH Bell circuit.

    Shows:
      - State preparation: H on qubit 0, CNOT
      - Both qubits transmitted
      - Alice measures in basis θ_A (Ry(-θ_A) then Z-measurement)
      - Bob   measures in basis θ_B (Ry(-θ_B) then Z-measurement)
      - Four measurement settings annotated
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")

    # Wire y-positions
    y0 = 2.5   # qubit 0 (Alice, top)
    y1 = 1.0   # qubit 1 (Bob, bottom)

    wire_color = "#333333"
    lw_wire = 2.0

    # ---- Qubit labels ----
    ax.text(-0.05, y0, r"$|0\rangle$", va="center", ha="right", fontsize=13,
            fontweight="bold")
    ax.text(-0.05, y1, r"$|0\rangle$", va="center", ha="right", fontsize=13,
            fontweight="bold")

    # ---- Wires ----
    ax.plot([0.0, 9.8], [y0, y0], color=wire_color, lw=lw_wire, zorder=1)
    ax.plot([0.0, 9.8], [y1, y1], color=wire_color, lw=lw_wire, zorder=1)

    # ---- Helper: draw a box gate ----
    def gate_box(cx, cy, label, color="#AECDE8", width=0.65, height=0.55):
        rect = plt.Rectangle(
            (cx - width / 2, cy - height / 2), width, height,
            facecolor=color, edgecolor="#333333", linewidth=1.5, zorder=3
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=4)

    # ---- H gate on qubit 0 ----
    gate_box(1.0, y0, "H", color="#FDDBC7")

    # ---- CNOT ----
    cnot_x = 2.2
    # Control dot on qubit 0
    ax.plot(cnot_x, y0, "o", color="#333333", markersize=10, zorder=4)
    # Target circle on qubit 1
    r_target = 0.25
    circle = plt.Circle((cnot_x, y1), r_target,
                         facecolor="white", edgecolor="#333333", linewidth=1.8, zorder=3)
    ax.add_patch(circle)
    # Cross inside target
    ax.plot([cnot_x - r_target, cnot_x + r_target], [y1, y1],
            color="#333333", lw=1.8, zorder=4)
    ax.plot([cnot_x, cnot_x], [y1 - r_target, y1 + r_target],
            color="#333333", lw=1.8, zorder=4)
    # Vertical control line
    ax.plot([cnot_x, cnot_x], [y1 + r_target, y0],
            color="#333333", lw=1.8, zorder=2)

    # ---- Label: Bell state preparation ----
    ax.text(1.6, 3.2, "Bell state\npreparation",
            ha="center", va="bottom", fontsize=9, color="#555555",
            style="italic")
    ax.annotate("", xy=(1.6, 3.0), xytext=(1.6, 3.2),
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.0))

    # ---- Noise region ----
    noise_x0, noise_x1 = 3.0, 5.5
    noise_mid = (noise_x0 + noise_x1) / 2
    noise_rect = plt.Rectangle(
        (noise_x0, y1 - 0.5), noise_x1 - noise_x0, y0 - y1 + 1.0,
        facecolor="#FFF9C4", edgecolor="#BBAA00",
        linewidth=1.5, linestyle="--", zorder=1, alpha=0.7
    )
    ax.add_patch(noise_rect)
    ax.text(noise_mid, (y0 + y1) / 2, "Noise\nchannel\n" + r"$\mathcal{N}_p$",
            ha="center", va="center", fontsize=9, color="#666600",
            style="italic", zorder=2)

    # ---- Alice's measurement block ----
    # Ry rotation gate
    gate_box(6.5, y0, r"$R_y(-\theta_A)$", color="#B2DF8A", width=1.1)
    # Measurement symbol
    meas_x = 7.8
    m_w, m_h = 0.7, 0.5
    rect_m = plt.Rectangle(
        (meas_x - m_w / 2, y0 - m_h / 2), m_w, m_h,
        facecolor="#E8F4F8", edgecolor="#333333", linewidth=1.5, zorder=3
    )
    ax.add_patch(rect_m)
    # Arc inside meter symbol
    theta_arc = np.linspace(0, np.pi, 60)
    arc_r = 0.18
    ax.plot(
        meas_x + arc_r * np.cos(theta_arc),
        y0 - 0.07 + arc_r * np.sin(theta_arc),
        color="#333333", lw=1.2, zorder=4
    )
    ax.annotate("", xy=(meas_x + 0.18, y0 + 0.14),
                xytext=(meas_x - 0.02, y0 - 0.07),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0),
                zorder=4)

    # ---- Bob's measurement block ----
    gate_box(6.5, y1, r"$R_y(-\theta_B)$", color="#B2DF8A", width=1.1)
    rect_m2 = plt.Rectangle(
        (meas_x - m_w / 2, y1 - m_h / 2), m_w, m_h,
        facecolor="#E8F4F8", edgecolor="#333333", linewidth=1.5, zorder=3
    )
    ax.add_patch(rect_m2)
    arc_r2 = 0.18
    ax.plot(
        meas_x + arc_r2 * np.cos(theta_arc),
        y1 - 0.07 + arc_r2 * np.sin(theta_arc),
        color="#333333", lw=1.2, zorder=4
    )
    ax.annotate("", xy=(meas_x + 0.18, y1 + 0.14),
                xytext=(meas_x - 0.02, y1 - 0.07),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0),
                zorder=4)

    # ---- Angle annotations ----
    ax.text(8.7, y0 + 0.05,
            r"$\theta_A \in \{0°, 90°\}$",
            ha="left", va="center", fontsize=9, color="#1A6B1A")
    ax.text(8.7, y1 + 0.05,
            r"$\theta_B \in \{45°, 135°\}$",
            ha="left", va="center", fontsize=9, color="#1A6B1A")

    # ---- Qubit labels left ----
    ax.text(-0.05, y0 + 0.55, "Qubit 0 (Alice)",
            ha="right", va="center", fontsize=8, color="#555555")
    ax.text(-0.05, y1 + 0.55, "Qubit 1 (Bob)",
            ha="right", va="center", fontsize=8, color="#555555")

    # ---- Separating dashed vertical line (Alice | Bob) ----
    ax.plot([5.7, 5.7], [0.2, 3.3], color="#999999",
            lw=1.0, linestyle=":", zorder=1)
    ax.text(4.5, 3.35, "Source", ha="center", fontsize=9,
            color="#555555", style="italic")
    ax.text(7.5, 3.35, "Alice", ha="center", fontsize=9,
            color="#1A6B1A", fontweight="bold")
    ax.text(5.85, 3.35, "Bob", ha="left", fontsize=9,
            color="#1A6B1A", fontweight="bold")

    # ---- CHSH formula annotation ----
    ax.text(5.0, -0.4,
            r"$S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|$",
            ha="center", va="center", fontsize=10,
            bbox=dict(facecolor="#F0F0F0", edgecolor="#BBBBBB",
                      boxstyle="round,pad=0.3"))

    fig.suptitle("CHSH Bell Test Circuit", fontsize=14, fontweight="bold", y=1.01)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — S vs p for depolarizing on |Φ+⟩
# ---------------------------------------------------------------------------

def fig2_depolarizing_main(data):
    p_vals = np.array(data["meta"]["p_values"])
    S_sim = np.array(data["S_vs_p"]["Phi+"]["depolarizing"])
    S_ana = np.array(data["analytical_depolarizing_S"])

    p_star_sim = data["thresholds"]["Phi+"]["depolarizing"]
    p_star_ana = 1.0 - 1.0 / np.sqrt(2)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Simulated
    ax.plot(p_vals, S_sim, color="#2166AC", lw=2.2, label="Simulated (exact DM)")

    # Analytical
    ax.plot(p_vals, S_ana, color="black", lw=1.8, linestyle="--",
            label=r"Analytical $S=(1-p)\cdot 2\sqrt{2}$")

    # Classical bound
    ax.axhline(CLASSICAL_BOUND, color="#D6604D", lw=1.5, linestyle="--",
               label=f"Classical bound $S=2$")

    # Tsirelson bound
    ax.axhline(TSIRELSON_BOUND, color="#4DAC26", lw=1.5, linestyle=":",
               label=r"Tsirelson bound $S=2\sqrt{2}$")

    # p* marker
    if p_star_sim is not None:
        ax.axvline(p_star_sim, color="#888888", lw=1.3, linestyle="-.",
                   label=f"$p^*_{{\\rm sim}}={p_star_sim:.4f}$")
        ax.annotate(
            f"$p^* = {p_star_sim:.4f}$\n(analytical: {p_star_ana:.4f})",
            xy=(p_star_sim, CLASSICAL_BOUND),
            xytext=(p_star_sim + 0.05, CLASSICAL_BOUND + 0.3),
            fontsize=9,
            arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.0),
            bbox=dict(facecolor="white", edgecolor="#AAAAAA",
                      boxstyle="round,pad=0.3", alpha=0.9),
        )

    # Shaded quantum region
    ax.fill_between(p_vals, CLASSICAL_BOUND, S_sim,
                    where=(S_sim > CLASSICAL_BOUND),
                    alpha=0.12, color="#2166AC",
                    label="Quantum advantage region")

    ax.set_xlabel("Noise rate $p$", fontsize=12)
    ax.set_ylabel("CHSH parameter $S$", fontsize=12)
    ax.set_title(r"$S$ vs Noise Rate: Depolarizing Noise on $|\Phi^+\rangle$",
                 fontsize=13)
    ax.set_xlim(0, P_MAX)
    ax.set_ylim(0.8, 3.0)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.4)

    # Add secondary annotation for bounds
    ax.text(0.48, TSIRELSON_BOUND + 0.04, r"$2\sqrt{2}$",
            ha="right", va="bottom", fontsize=9, color="#4DAC26")
    ax.text(0.48, CLASSICAL_BOUND + 0.04, "$2$",
            ha="right", va="bottom", fontsize=9, color="#D6604D")

    fig.tight_layout()
    return fig

P_MAX = 0.5


# ---------------------------------------------------------------------------
# Figure 3 — All noise types on |Φ+⟩
# ---------------------------------------------------------------------------

def fig3_noise_comparison(data):
    p_vals = np.array(data["meta"]["p_values"])

    fig, ax = plt.subplots(figsize=(7, 5))

    for noise, label in NOISE_LABELS.items():
        S = np.array(data["S_vs_p"]["Phi+"][noise])
        ax.plot(p_vals, S, color=NOISE_COLORS[noise], lw=2.2, label=label)

    ax.axhline(CLASSICAL_BOUND, color="#AA2211", lw=1.5, linestyle="--",
               label="Classical bound $S=2$")
    ax.axhline(TSIRELSON_BOUND, color="#4DAC26", lw=1.2, linestyle=":",
               label=r"Tsirelson bound $S=2\sqrt{2}$", alpha=0.7)

    # Annotate p* for each noise
    for noise in NOISE_LABELS:
        p_star = data["thresholds"]["Phi+"][noise]
        if p_star is not None:
            ax.axvline(p_star, color=NOISE_COLORS[noise],
                       lw=1.0, linestyle=":", alpha=0.7)

    ax.set_xlabel("Noise rate $p$", fontsize=12)
    ax.set_ylabel("CHSH parameter $S$", fontsize=12)
    ax.set_title(r"Noise Model Comparison on $|\Phi^+\rangle$", fontsize=13)
    ax.set_xlim(0, P_MAX)
    ax.set_ylim(0.8, 3.0)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — All Bell states under depolarizing
# ---------------------------------------------------------------------------

def fig4_bell_states(data):
    p_vals = np.array(data["meta"]["p_values"])

    fig, ax = plt.subplots(figsize=(7, 5))

    # All Bell states have identical S under depolarizing noise (same decay rate).
    # Use distinct linestyles + markers to show they overlap.
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    markevery = 8

    for (state, label), ls, mk in zip(BELL_LABELS.items(), linestyles, markers):
        S = np.array(data["S_vs_p"][state]["depolarizing"])
        ax.plot(p_vals, S, color=BELL_COLORS[state], lw=2.0,
                linestyle=ls, marker=mk, markevery=markevery,
                markersize=5, label=label)

    ax.axhline(CLASSICAL_BOUND, color="#AA2211", lw=1.5, linestyle="--",
               label="Classical bound $S=2$")
    ax.axhline(TSIRELSON_BOUND, color="#4DAC26", lw=1.2, linestyle=":",
               label=r"Tsirelson bound", alpha=0.7)

    ax.set_xlabel("Noise rate $p$", fontsize=12)
    ax.set_ylabel("CHSH parameter $S$", fontsize=12)
    ax.set_title("Bell States Under Global Depolarizing Noise\n"
                 "(all four states overlap — same decay rate)", fontsize=12)
    ax.set_xlim(0, P_MAX)
    ax.set_ylim(0.8, 3.0)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.4)

    # Note about threshold
    p_star = data["thresholds"]["Phi+"]["depolarizing"]
    if p_star is not None:
        ax.axvline(p_star, color="#888888", lw=1.2, linestyle="-.",
                   label=f"$p^*={p_star:.4f}$")
        ax.text(p_star + 0.005, 0.95,
                f"$p^*={p_star:.4f}$\n(all states)",
                fontsize=8, color="#666666", va="bottom")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Angle heatmap (noiseless |Φ+⟩)
# ---------------------------------------------------------------------------

def fig5_angle_heatmap():
    rho = bell_phi_plus()
    n_grid = 60
    thetas = np.linspace(0, 180, n_grid)

    S_grid = np.zeros((n_grid, n_grid))
    for i, tA in enumerate(thetas):
        for j, tB in enumerate(thetas):
            # For the heatmap, we keep Alice's second angle fixed at a'=a+90°
            # and Bob's second angle at b'=b+90°, sweeping the base angles
            S_grid[i, j] = chsh_value(rho, tA, tA + 90, tB, tB + 90)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(
        S_grid,
        origin="lower",
        extent=[0, 180, 0, 180],
        aspect="auto",
        cmap="RdYlGn",
        vmin=0, vmax=TSIRELSON_BOUND + 0.1,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CHSH parameter $S$", fontsize=11)
    cbar.ax.axhline(CLASSICAL_BOUND, color="black", lw=1.5, linestyle="--")
    cbar.ax.axhline(TSIRELSON_BOUND, color="darkgreen", lw=1.5, linestyle=":")

    # Mark optimal point
    ax.plot(45, 0, "w*", markersize=14, label="Optimal $(a=0°,b=45°)$",
            markeredgecolor="black", markeredgewidth=0.8)

    # Classical bound contour
    CS = ax.contour(thetas, thetas, S_grid, levels=[CLASSICAL_BOUND],
                    colors=["black"], linewidths=1.5, linestyles="--")
    ax.clabel(CS, fmt=r"$S=2$", fontsize=8)

    ax.set_xlabel(r"Alice base angle $\theta_A$ (°)", fontsize=12)
    ax.set_ylabel(r"Bob base angle $\theta_B$ (°)", fontsize=12)
    ax.set_title(
        r"CHSH $S$ vs Measurement Angles — Noiseless $|\Phi^+\rangle$" + "\n"
        r"($a=\theta_A,\; a'=\theta_A+90°,\; b=\theta_B,\; b'=\theta_B+90°$)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Threshold p* bar chart
# ---------------------------------------------------------------------------

def fig6_threshold_bars(data):
    analytical_p_star = data["meta"]["analytical_p_star_depolarizing"]

    noises = list(NOISE_LABELS.keys())
    p_stars = [data["thresholds"]["Phi+"][n] for n in noises]
    labels = [NOISE_LABELS[n] for n in noises]
    colors = [NOISE_COLORS[n] for n in noises]

    # Replace None with nan for display
    p_stars_plot = [v if v is not None else float("nan") for v in p_stars]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    x = np.arange(len(noises))
    bars = ax.bar(x, p_stars_plot, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.55, zorder=3)

    # Annotate bars
    for bar, val in zip(bars, p_stars_plot):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    # Analytical line for depolarizing
    ax.axhline(analytical_p_star, color="#333333", lw=1.5, linestyle="--",
               zorder=4, label=f"Analytical $p^*_{{\\rm dep}}={analytical_p_star:.4f}$")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Critical noise rate $p^*$", fontsize=12)
    ax.set_title(
        r"Noise Threshold $p^*$ by Channel" + "\n"
        r"(CHSH $S$ drops below classical bound $S=2$)",
        fontsize=12,
    )
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Add annotation about fastest decay
    p_stars_valid = [(n, v) for n, v in zip(noises, p_stars_plot)
                     if not np.isnan(v)]
    if p_stars_valid:
        fastest = min(p_stars_valid, key=lambda x: x[1])
        ax.text(
            0.98, 0.98,
            f"Fastest decay:\n{NOISE_LABELS[fastest[0]]}\n($p^*={fastest[1]:.4f}$)",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(facecolor="#FFF3CD", edgecolor="#CCAA00",
                      boxstyle="round,pad=0.4", alpha=0.9),
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading results...")
    data = load_results()
    print("  Loaded.")

    print("Generating fig1 — circuit schematic...")
    fig = fig1_circuit()
    save_fig(fig, "fig1_circuit")
    plt.close(fig)

    print("Generating fig2 — depolarizing main result...")
    fig = fig2_depolarizing_main(data)
    save_fig(fig, "fig2_depolarizing")
    plt.close(fig)

    print("Generating fig3 — noise type comparison...")
    fig = fig3_noise_comparison(data)
    save_fig(fig, "fig3_noise_comparison")
    plt.close(fig)

    print("Generating fig4 — Bell state comparison...")
    fig = fig4_bell_states(data)
    save_fig(fig, "fig4_bell_states")
    plt.close(fig)

    print("Generating fig5 — angle heatmap (takes a moment)...")
    fig = fig5_angle_heatmap()
    save_fig(fig, "fig5_angle_heatmap")
    plt.close(fig)

    print("Generating fig6 — threshold bar chart...")
    fig = fig6_threshold_bars(data)
    save_fig(fig, "fig6_thresholds")
    plt.close(fig)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
