# CHSH Bell Inequality Violation Under Noise

**Authors:** Arhaan Sareen, Aditya Saxena  
**Affiliation:** QSYS 2026 Participant, Institute for Quantum Computing, University of Waterloo  
**Correspondence:** arhaan.sareen@gmail.com  
**Target venue:** Journal of Student Research / arXiv quant-ph

---

## Overview

This repository contains the LaTeX source and supporting code for the paper:

> *CHSH Bell Inequality Violation Under Noise: A Threshold Analysis for Near-Term Quantum Devices*

The central contribution is the analytical derivation and numerical verification of the
**decoherence threshold** p* — the minimum depolarizing noise rate at which a two-qubit Bell
state can no longer violate the CHSH inequality:

```
p* = 1 - 1/sqrt(2) ≈ 0.2929
```

---

## Repository Structure

```
bell-paper/
├── README.md
└── paper/
    ├── main.tex          — Main LaTeX source (~4000 words)
    ├── bibliography.bib  — BibTeX references (14 entries)
    └── figures/          — Place generated figure PNGs here
        ├── fig1_bell_circuit.png
        ├── fig2_chsh_vs_noise.png
        ├── fig3_noise_comparison.png
        ├── fig4_bell_states.png
        ├── fig5_angle_heatmap.png
        └── fig6_threshold_comparison.png
```

---

## Building the Paper

Requires a TeX distribution (TeX Live or MiKTeX) with the following packages:
`amsmath`, `amssymb`, `graphicx`, `booktabs`, `hyperref`, `natbib`,
`setspace`, `microtype`, `caption`, `braket`, `physics`, `geometry`,
`float`, `xcolor`, `enumitem`, `authblk`.

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with `latexmk`:

```bash
cd paper/
latexmk -pdf main.tex
```

---

## Generating Figures

The simulation code lives in `../code/` (the qsys-paper working directory).
Run the simulation scripts to produce the figure PNGs, then copy them into `paper/figures/`.

Key results the figures must show:
- **fig2:** S(p) = (1-p)·2√2, threshold at p* ≈ 0.2929, MAE < 0.001 vs simulation
- **fig3:** Depolarizing (p* ≈ 0.293) > Dephasing (p* ≈ 0.25) > Amplitude damping (p* ≈ 0.21)
- **fig4:** All four Bell states overlap identically under depolarizing noise
- **fig5:** Heatmap of S over (θ_A, θ_A') — max at (0, π/2)
- **fig6:** Bar chart of p* for each noise model

---

## Key Analytical Result

Under global depolarizing noise with rate p:

```
S(p) = (1 - p) · 2√2
p*   = 1 - 1/√2 ≈ 0.2929   (exact, not an approximation)
```

Proof relies on tracelessness of Pauli matrices: Tr[σ_i] = 0, so the maximally mixed
component I/4 contributes zero to any Pauli correlator.

---

## Dependencies

- Qiskit 1.x (`qiskit`, `qiskit.quantum_info`)
- NumPy, Matplotlib (for figures)
- Python 3.10+
