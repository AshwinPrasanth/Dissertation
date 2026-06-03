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

# solver.py — Structural Persistence Guided Exact Search

`solver.py` implements the exact Branch-and-Bound search engine used throughout the dissertation experiments.

The solver combines:

* root-level MWUA structural guidance,
* lightweight dynamic graph refinement,
* confidence-guided DFS diving,
* branch-and-reduce simplification,
* persistence tracking,
* and exact LP-based optimization.

The framework preserves exact optimality while studying how structural certainty evolves during combinatorial search.

---

# Solver Architecture

```text id="l6e95m"
            Root LP Relaxation
                     │
                     ▼
       MWUA Structural Snapshot
                     │
                     ▼
        Exact DFS Branch-and-Bound
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
 Reductions     Persistence     Pseudo-Costs
      │              │              │
      └──────────────┼──────────────┘
                     ▼
      Certainty-Guided Branching
                     │
                     ▼
        Residual Graph Evolution
                     │
                     ▼
             Exact Optimality
```

---

# Core Solver Components

| Component              | Purpose                             |
| ---------------------- | ----------------------------------- |
| `BranchAndBoundSolver` | Main exact DFS solver               |
| `SolverTrace`          | Full search instrumentation         |
| `DepthStats`           | Per-depth structural analysis       |
| `Reductions`           | Lightweight branch-and-reduce rules |
| `BBNode`               | Search node representation          |
| persistence tracking   | structural stabilization analysis   |

---

# 1. Exact DFS Branch-and-Bound

The solver performs classical exact DFS Branch-and-Bound:

```text id="tqmql6"
solve LP
    ↓
prune / reduce
    ↓
branch
    ↓
explore recursively
```

The framework remains fully exact:

* LP bounds preserve correctness,
* pruning remains safe,
* and optimality guarantees are retained.

Structural guidance influences:

* branch ordering,
* search trajectory,
* and residual simplification,
  not correctness.

---

# 2. Confidence-Guided DFS Diving

The solver biases DFS exploration toward structurally stable trajectories using:

```text id="j6j2di"
priority(node) =
    depth
  + λ · certainty
```

High-certainty branches are explored earlier, encouraging:

* faster stabilization,
* accelerated simplification,
* and earlier incumbent discovery.

This creates a certainty-aware DFS exploration policy instead of purely depth-first traversal.

---

# 3. Branch-and-Reduce Framework

Before branching, lightweight exact reductions are applied:

| Reduction               | Purpose                           |
| ----------------------- | --------------------------------- |
| LP-forced assignments   | Nemhauser–Trotter simplifications |
| isolated vertex removal | eliminate irrelevant variables    |
| pendant reductions      | exact graph simplification        |
| LP-degree agreement     | certainty-guided exact reductions |

The solver therefore combines:

* exact reductions,
* structural certainty,
* and local graph dynamics.

---

# 4. Structural Persistence Tracking

A central research component of the solver is:

## Persistence Tracking

Variables repeatedly stabilizing toward the same assignment accumulate persistence scores during DFS exploration.

The solver maintains:

```text id="h6qlzb"
persistence_zero
persistence_one
persistence_visits
```

which approximate:

* long-range stabilization,
* structural rigidity,
* and pseudo-backbone behaviour.

This mechanism enables analysis of:

> how root-level structural information persists during exact search.

---

# 5. Residual Graph Evolution

The solver continuously tracks residual graph dynamics:

| Signal           | Meaning                         |
| ---------------- | ------------------------------- |
| residual density | remaining structural complexity |
| residual degree  | surviving graph influence       |
| local gain       | expected simplification power   |
| active vertices  | remaining search size           |

These measurements enable study of:

* residual graph collapse,
* search stabilization,
* and structural simplification trajectories.

---

# 6. Pseudo-Cost Learning

The framework includes full pseudo-cost tracking with reliability gating.

Pseudo-costs estimate the historical usefulness of branching decisions:

```text id="8od0if"
large LP improvement
    →
higher future branch preference
```

Pseudo-costs are combined with:

* MWUA certainty,
* persistence,
* and local graph structure

to guide future branching.

---

# 7. Diversified Incumbent Search

The solver performs shallow exploratory dives using perturbed certainty weights to discover early incumbents.

This lightweight diversification strategy improves:

* early upper bounds,
* subtree pruning,
* and trajectory diversity

without expensive strong branching or large-scale parallel search.

---

# 8. Full Search Instrumentation

The solver records complete search dynamics for research analysis.

Tracked signals include:

| Metric                     | Purpose                     |
| -------------------------- | --------------------------- |
| certainty evolution        | stabilization analysis      |
| persistence evolution      | structural consistency      |
| pruning statistics         | subtree effectiveness       |
| incumbent improvements     | solution trajectory         |
| residual density evolution | graph collapse analysis     |
| backbone-style statistics  | variable stability analysis |

This instrumentation forms the basis of the dissertation’s structural persistence experiments.

---

# Main Research Hypothesis

The solver investigates whether:

> root-level structural certainty remains informative surprisingly deep into exact combinatorial search.

Current experiments suggest:

* structurally certain variables stabilize earlier,
* residual graph density collapses faster,
* MWUA certainty behaves similarly to lightweight backbone estimation,
* and certainty-guided branching scales better than uncertainty-first branching.

---

# Research Direction

The framework is evolving toward:

## Structural Persistence Guided Exact Search

where:

* lightweight global structural snapshots,
* dynamic local refinement,
* persistence tracking,
* and exact branch-and-reduce search

are combined to study long-range structural behaviour in combinatorial optimization.
