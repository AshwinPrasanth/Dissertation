# Combining Static Global and Dynamic Local Features for Fast Learning-to-Branch Heuristics

**MSc Dissertation — Advanced Artificial Intelligence**
Ashwin Prasanth (Student ID: 25234753)
Supervisor: Dr. Deepak Ajwani
UCD School of Computer Science, University College Dublin — August 2026

This repository contains the full implementation, training pipelines, and evaluation
scripts accompanying the dissertation *"Combining Static Global and Dynamic Local
Features for Fast Learning-to-Branch Heuristics."* It supports two combinatorial
optimisation settings:

1. **Maximum Independent Set / Minimum Vertex Cover (MIS/MVC)** — solved with a
   custom SCIP-based Branch-and-Bound framework. See [`README_MIS.md`](README_MIS.md).
2. **Hitting Set (HS)** — solved with a modified version of the UZL solver
   (PACE 2025 Exact Track winner), integrating a learned branching policy into
   the embedded CaDiCaL MaxSAT solver via EvalMaxSAT. See [`README_HittingSet.md`](README_HittingSet.md).

For a map of the repository, per-file responsibilities, and guidance on adapting
the pipeline to new problems/solvers, see [`README_CodeStructure.md`](README_CodeStructure.md).

---

## 1. Motivation

Exact solvers for NP-hard problems depend heavily on the variable selected at
each branching or decision point:

- LP-based Branch-and-Bound solvers (e.g. SCIP) can use anything from cheap
  heuristics (most-fractional, pseudocost) to **Strong Branching**, which is
  informative but too expensive to run at every node.
- Conflict-driven MaxSAT solvers (e.g. CaDiCaL) typically rely on cheap
  activity-based heuristics (EVSIDS), which are inexpensive but uninformative
  early in search, before conflict information has accumulated.

Learning-to-branch approaches can inject additional signal into this decision,
but many recompute expensive features from the *residual* instance at every
node, adding overhead to an already high-frequency part of the solver.

**Core research question:** does a *global* optimisation signal, computed
**once** before search begins, remain useful as a branching heuristic
throughout an evolving search tree — without needing to be repeatedly
recomputed or numerically exact?

## 2. Approach

1. A **Multiplicative Weights Update Algorithm (MWUA)** oracle is run **once**
   on the reduced covering-formulation instance (shared by MVC and Hitting
   Set). This produces:
   - an averaged fractional solution,
   - per-vertex certainty information,
   - normalised constraint weights.
2. This static, one-shot signal is combined with lightweight, **solver-specific
   dynamic features** (computed cheaply at each branching node) into a
   15-dimensional feature vector per candidate variable.
3. **XGBoost Learning-to-Rank** models are trained to imitate an expert
   branching signal:
   - **MIS/MVC**: imitates **Strong Branching** within SCIP.
   - **Hitting Set**: imitates **native CaDiCaL** decisions within the
     PACE 2025-winning UZL/EvalMaxSAT solver.
4. At deployment, the learned policy is used **only for the first *k* decision
   layers** of search (depth-bounded intervention: e.g. D1, D2, D5, D10, or
   unrestricted "Full"), after which control reverts to the solver's native
   branching strategy. Learning is confined strictly to **variable selection**
   — propagation, conflict analysis, bounding, and pruning are left unchanged.

## 3. Key Results

| Setting | Headline result |
|---|---|
| **MIS/MVC** | ML intervention to depth 2 gives a median **25.9%** reduction in explored nodes and **11.9%** reduction in solve time, improving on 83.0% and 72.0% of instances respectively. Depth 5 attains the highest exact-optimality rate. Models trained on small DIMACS graphs generalise to dense BHOSLIB instances and to SNAP/DIMACS10 graphs of up to **1.8M vertices**. |
| **Hitting Set** | Guidance over the first two decision levels solves **76/100** private PACE 2025 test instances vs. **75/100** for the native CaDiCaL baseline, with a **23.1%** median runtime reduction. |
| **General finding** | Extending learned control *further* into the search **degrades** performance in both settings — broader intervention is not necessarily better. Downstream branching quality is **non-monotonic** in MWUA iteration count: it improves sharply up to ~5,000 iterations, then degrades/fluctuates, even as the MWUA fractional solution keeps converging to the LP optimum. The most useful signal is not the most numerically converged one. |

## 4. Repository Layout (top level)

```
dissertation_ashwin/
├── CHSZLabLib/build-kamis/_kamis     KaMIS/ReduMIS reduction interface (shared)
├── training/                         MIS/MVC pipeline           → see README_MIS.md
├── HIT/                              Hitting Set feature generation
└── PACE2025/                         Modified UZL/CaDiCaL solver → see README_HittingSet.md
```

Full per-file documentation is in [`README_CodeStructure.md`](README_CodeStructure.md).

## 5. Software / Hardware Environment

All experiments reported in the dissertation were run on:

- **Hardware:** CPU-only server, AMD EPYC 7281 (16 physical cores / 32 threads), 94 GB RAM.
- **OS:** Ubuntu 18.04.6 LTS.
- **Software:**
  - Python 3.10.20
  - XGBoost 2.1.4
  - PySCIPOpt 6.2.1
  - NumPy 2.2.6
  - SciPy 1.15.2
  - Rust (stable toolchain, via `cargo`) for the PACE2025/UZL solver
  - C++17 toolchain for EvalMaxSAT/CaDiCaL
  - KaMIS/ReduMIS build (`CHSZLabLib/build-kamis/_kamis`) for exact reductions

A conda environment named `dissertation` was used throughout:

```bash
conda activate dissertation
```

> **Reproducibility note:** exact package versions are listed above; a
> `requirements.txt` / `environment.yml` pinning these versions is recommended
> when re-creating the environment (see [`README_CodeStructure.md`](README_CodeStructure.md)
> §"Environment setup").

## 6. Benchmarks Used

| Domain | Benchmarks | Role |
|---|---|---|
| MIS/MVC | DIMACS | Training / model development (small graphs) |
| MIS/MVC | BHOSLIB | Dense out-of-distribution generalisation test |
| MIS/MVC | SNAP, DIMACS10 | Large-scale generalisation test (up to 1.8M vertices) |
| Hitting Set | PACE 2025 Exact Track — 100 public instances | Training / instrumentation (32 instrumented, 24 used for training) |
| Hitting Set | PACE 2025 Exact Track — 100 private test instances | End-to-end held-out evaluation |

## 7. High-Level Reproduction Path

For either domain, the pipeline follows the same conceptual stages:

```
Raw instances (DIMACS graphs / PACE hypergraphs)
        ↓
Exact reduction / kernelisation (KaMIS/ReduMIS)
        ↓
MWUA oracle (once per instance) → static global features
        ↓
Solver-specific dynamic features (Strong Branching stats / CaDiCaL CDCL signals)
        ↓
Branching-decision data collection (expert imitation targets)
        ↓
Grouped Learning-to-Rank dataset construction
        ↓
Instance-level train/validation/test split
        ↓
XGBoost ranking-model training
        ↓
Deployment: bounded-depth or full learned branching inside the exact solver
        ↓
Anytime / end-to-end evaluation vs. native solver baseline
```

See [`README_MIS.md`](README_MIS.md) and [`README_HittingSet.md`](README_HittingSet.md)
for the exact commands, scripts, and configuration used for each domain.

## 8. Citing / Attribution

If reusing this code, please cite the dissertation:

> Ashwin Prasanth, *"Combining Static Global and Dynamic Local Features for
> Fast Learning-to-Branch Heuristics,"* MSc Dissertation, School of Computer
> Science, University College Dublin, August 2026. Supervisor: Dr. Deepak Ajwani.

This work builds on and modifies:
- **SCIP / PySCIPOpt** for the MIS/MVC Branch-and-Bound framework.
- **KaMIS / ReduMIS** for exact kernelisation reductions.
- **UZL** and **EvalMaxSAT / CaDiCaL** (PACE 2025 Exact Track winning Hitting
  Set solver), modified to embed a learned branching policy.
- **XGBoost** (Chen & Guestrin) for the Learning-to-Rank models.

## 9. Use of Generative AI

Per the dissertation's Appendix C ("Use of Generative AI"), generative AI
tools were used in parts of this project's development; refer to the thesis
document itself for the full disclosure statement.
