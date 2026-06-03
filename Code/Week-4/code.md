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
