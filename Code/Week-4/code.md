# branching.py : Certainty-Guided Branching

Lightweight branching framework for exact DFS Branch-and-Bound search using:

* static global structural signals,
* dynamic local graph refinement,
* persistence tracking,
* and lightweight empirical search feedback.

The framework explores the hypothesis that:

> structurally stable variables produce stronger long-range search guidance than classical uncertainty-first branching.

---

# Core Idea

Instead of repeatedly recomputing expensive graph embeddings during search, the solver computes a **global structural snapshot once at the root** using MWUA-style structural certainty, then combines it with lightweight local updates during DFS exploration.

```text id="9t0m8u"
           Root Graph
                │
                ▼
    MWUA Structural Snapshot
                │
                ▼
     Exact DFS Branch-and-Bound
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
 LP Certainty  Residual   Persistence
                Graph       Signals
                 │
                 ▼
      Composite Branch Score
                 │
                 ▼
        Variable Selection
```

---

# Certainty-First Branching

The primary branching strategy combines:

```text id="mlvy6e"
score(v) =
    α · LP_certainty
  + β · MWUA_certainty
  + γ · local_structure
  + δ · pseudo_cost
  + η · persistence
  + ζ · propagation_gain
```

The framework therefore integrates:

| Signal Type             | Scope                         |
| ----------------------- | ----------------------------- |
| LP certainty            | local relaxation confidence   |
| MWUA certainty          | global root-level structure   |
| residual graph features | dynamic local structure       |
| pseudo-cost             | empirical branch quality      |
| persistence             | DFS stabilization behaviour   |
| propagation gain        | expected simplification power |

---

# Main Research Hypothesis

The project investigates whether:

> root-level structural certainty remains informative surprisingly deep into exact search.

The current experiments suggest:

* high-certainty variables stabilize early,
* residual graph density collapses faster,
* MWUA certainty behaves similarly to lightweight backbone estimation,
* and certainty-guided branching scales better than uncertainty-first branching.

---

# Implemented Branching Strategies

| Strategy          | Description                              |
| ----------------- | ---------------------------------------- |
| `certainty_first` | Composite structural certainty branching |
| `mwua_only`       | Root-level MWUA guidance only            |
| `most_fractional` | Classical LP uncertainty-first branching |
| `degree`          | Residual graph degree heuristic          |
| `pseudo_cost`     | Empirical pseudo-cost branching          |
| `random`          | Uniform random baseline                  |

---

# Current Research Direction

The framework is evolving toward:

## Structural Persistence in Exact Search

where:

* root-level structural information,
* local search refinement,
* and stabilization dynamics

are combined to guide exact combinatorial optimization efficiently without expensive repeated global recomputation.

# core.py — Structural Snapshot and Residual Graph Engine

`core.py` implements the structural foundation of the exact Branch-and-Bound framework.

The module combines:

* root-level global structural analysis,
* lightweight MWUA-based certainty estimation,
* incremental residual graph tracking,
* and exact LP relaxation support.

The design intentionally avoids expensive repeated global recomputation during search.

---

# System Architecture

```text id="1j26ik"
          Graph / MILP Problem
                    │
                    ▼
             LP Relaxation
                    │
                    ▼
        Root MWUA Structural Snapshot
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Global Certainty      LP Certainty
         │                     │
         └──────────┬──────────┘
                    ▼
         StructuralFeatureEngine
                    │
                    ▼
             Exact DFS Search
                    │
                    ▼
          Incremental GraphState
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 Residual Degree  Local Density  Propagation Gain
```

---

# Core Components

| Component                 | Purpose                               |
| ------------------------- | ------------------------------------- |
| `MILPProblem`             | Generic binary MILP representation    |
| `solve_lp()`              | LP relaxation solver                  |
| `PrincipledMWUA`          | Root-level global structural snapshot |
| `StructuralFeatureEngine` | Lightweight root feature extraction   |
| `GraphState`              | Incremental residual graph tracking   |
| `lp_forced_assignments()` | Exact LP reductions                   |

---

# 1. LP Relaxation

At each Branch-and-Bound node, the framework solves the LP relaxation:

```math id="p0kr3k"
0 \leq x_i \leq 1
```

instead of enforcing binary assignments immediately.

The LP provides:

* lower bounds,
* fractional uncertainty,
* pruning information,
* and local branching guidance.

The framework augments classical LP reasoning with structural certainty signals rather than replacing exact optimization.

---

# 2. Principled MWUA Snapshot

The central research idea of the framework is computing a lightweight global structural snapshot once at the root node using a principled Multiplicative Weights Update Algorithm (MWUA).

Instead of recomputing expensive graph embeddings during search, the framework computes:

* approximate global fractional assignments,
* structural certainty values,
* and constraint pressure statistics

only once at initialization.

---

# MWUA Intuition

Each constraint maintains a weight:

```text id="tly1jh"
violated constraints → higher weight
stable constraints   → lower influence
```

Variables connected to repeatedly violated constraints accumulate stronger structural importance.

The resulting MWUA certainty behaves as a lightweight approximation of:

* structural stability,
* global search pressure,
* and pseudo-backbone behaviour.

---

# MWUA Output Features

The MWUA process produces:

| Feature              | Meaning                                  |
| -------------------- | ---------------------------------------- |
| `mwua_x_avg`         | approximate global fractional assignment |
| `mwua_certainty`     | structural certainty estimate            |
| `constraint_weights` | learned structural pressure distribution |

The certainty score is defined as:

```math id="x5r8e5"
|x_i - 0.5|
```

where:

* values near `0.5` indicate ambiguity,
* values near `0` or `1` indicate stabilization.

---

# 3. StructuralFeatureEngine

`StructuralFeatureEngine` computes the root-level feature snapshot used throughout exact search.

The implementation intentionally keeps the representation minimal:

| Static Global Features   |
| ------------------------ |
| MWUA certainty           |
| MWUA average assignments |
| root LP certainty        |

This lightweight design is central to the dissertation hypothesis that:

> root-level structural information may remain useful surprisingly deep into exact search.

---

# 4. GraphState — Incremental Residual Graph Tracking

`GraphState` maintains a lightweight dynamic view of the residual graph during DFS exploration.

As variables become fixed:

* vertices are removed,
* neighbourhood structure changes,
* and residual statistics are updated incrementally.

This avoids expensive recomputation of global graph features at every node.

---

# Residual Structural Signals

The residual graph exposes several dynamic local signals:

| Signal                 | Purpose                             |
| ---------------------- | ----------------------------------- |
| residual degree        | remaining graph influence           |
| local density          | neighbourhood structural complexity |
| active neighbour ratio | surviving structural connectivity   |
| propagation gain       | expected simplification potential   |

---

# Propagation Gain

A lightweight local propagation estimate is defined using:

```text id="p2yq3q"
propagation_gain =
    0.5 · degree_term
  + 0.3 · density_term
  + 0.2 · neighbour_ratio
```

This estimates how strongly branching on a variable may simplify the remaining search space.

---

# 5. Residual Graph Collapse

The incremental graph tracking infrastructure enables analysis of:

* residual density evolution,
* structural stabilization,
* propagation behaviour,
* and search-space collapse dynamics.

These measurements form a major part of the dissertation’s structural persistence analysis.

---

# 6. Exact LP Reductions

The framework includes exact LP-forced assignments:

```math id="ov6m4m"
x_i = 0
\quad \text{or} \quad
x_i = 1
```

which can be safely fixed during search.

These reductions are based on classical Nemhauser–Trotter-style LP properties for Vertex Cover.

---

# Research Direction

The module is designed around the dissertation hypothesis that:

> lightweight global structural certainty combined with incremental local refinement can guide exact combinatorial search efficiently without repeated expensive global recomputation.

Current experiments investigate:

* structural persistence,
* pseudo-backbone behaviour,
* residual graph collapse,
* and long-range usefulness of root-level MWUA certainty.

