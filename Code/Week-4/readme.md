# MWUA-Guided Exact Search Dynamics

> Static Global Structure + Lightweight Local Refinement for Exact DFS Search

---

# Overview

This project investigates whether a **single root-level MWUA (Multiplicative Weights Update Algorithm) structural snapshot** remains informative throughout exact branch-and-bound search for combinatorial optimization problems such as:

* Minimum Vertex Cover (MVC)
* Maximum Independent Set (MIS)

The framework studies:

* structural certainty persistence,
* residual graph evolution,
* stabilization dynamics,
* and certainty-guided exact DFS search.

The implementation intentionally avoids repeated expensive global recomputation and instead follows the dissertation philosophy:

[
\boxed{
\text{
Compute global structure once,
then reuse it during exact search using lightweight local updates.
}
}
]

---

# Core Solver Architecture

```text
Graph
 ↓
Root-Level MWUA Structural Snapshot
 ↓
Approximate Backbone-Like Structural Certainty
 ↓
LP Relaxation + Lightweight Local Features
 ↓
Certainty-Guided Variable Selection
 ↓
Exact DFS Branch-and-Bound
 ↓
Residual Graph Evolution Analysis
 ↓
Persistence / Stabilization Measurement
```

---

# Experiment 1 — Variable Stability & Deferred Uncertainty

## Objective

Analyze whether high-certainty MWUA variables stabilize earlier during DFS search.

## Key Findings

### High-certainty variables stabilize rapidly

| Variable | MWUA Certainty | Fractional Frequency | Stability |
| -------- | -------------- | -------------------- | --------- |
| 0        | 1.00           | 0.018                | 0.857     |
| 10       | 1.00           | 0.036                | 0.793     |

These variables exhibited:

* extremely low fractional persistence,
* high assignment consistency,
* near-deterministic stabilization.

---

### Deferred Uncertainty emerges naturally

Medium-certainty variables remained ambiguous substantially longer:

| MWUA Certainty | Fractional Frequency | Stability |
| -------------- | -------------------- | --------- |
| 0.73           | ~0.29                | ~0.37     |
| 0.46           | ~0.42                | ~0.46     |

This led to the emergence of:

[
\boxed{
\text{
Deferred Uncertainty
}
}
]

where structurally uncertain variables remain unresolved deeper into DFS exploration.

---

### Backbone-like behaviour

High-certainty MWUA variables naturally behaved similarly to pseudo-backbones:

* low ambiguity,
* strong stabilization,
* persistent assignment consistency,

despite no explicit backbone computation being used.

---

# Experiment 2 — Residual Graph Evolution

## Objective

Study how branching strategies affect residual graph collapse during DFS search.

Compared strategies:

* `certainty_first`
* `mwua_only`
* `most_fractional`

---

## Accelerated structural collapse

Residual graph density at depth 20:

| Strategy        | Residual Density |
| --------------- | ---------------- |
| certainty_first | 0.248            |
| mwua_only       | 0.295            |
| most_fractional | 0.419            |

Certainty-guided branching simplified residual subproblems substantially faster.

[
\boxed{
\text{
Certainty-guided branching shapes
the evolution of the entire search tree.
}
}
]

---

## MWUA-only already strong

Even without local LP refinement:

* MWUA-only consistently outperformed LP branching,
* reduced residual density faster,
* stabilized assignments earlier.

This strongly supports:

> root-level structural information persists surprisingly deep into exact search.

---

# Experiment 3 — Density-Wise Structural Behaviour

## Objective

Evaluate branching behaviour under varying graph densities.

[
p \in [0.1, 0.6]
]

Compared:

* `certainty_first`
* `most_fractional`
* `degree`

---

## Fewer explored search nodes

| Density | Certainty-First | Most-Fractional |
|---|---|
| 0.20 | 9 | 15 |
| 0.40 | 17 | 29 |
| 0.60 | 31 | 43 |

Importantly:

* prune rates remained nearly identical,
* indicating that gains originate from:

  * improved structural guidance,
  * not aggressive pruning.

---

# Experiment 4 — Scaling Behaviour & Structural Persistence

## Objective

Evaluate whether MWUA structural certainty remains effective as graph size increases.

[
n = 10 \rightarrow 80
]

---

## Scaling advantage increases with problem size

| Graph Size | Certainty-First | Most-Fractional |
|---|---|
| 25 | 25 | 51 |
| 40 | 75 | 171 |
| 60 | 631 | 1557 |
| 80 | 1423 | 3509 |

At (n=80):

[
\boxed{
59% \text{ fewer explored nodes}
}
]

than classical LP branching.

---

## Strong persistence correlations

Observed correlations:

[
corr(MWUA,\ stability)=0.398
]

[
corr(MWUA,\ fractional\ persistence)=-0.826
]

The strong negative persistence correlation suggests:

* high-certainty variables stabilize early,
* remain fractional less often,
* retain structural value deep into search.

---

# Experiment 5 — MWUA Signal Ablation

## Objective

Isolate the contribution of each structural signal.

Compared:

* LP-only
* MWUA-only
* LP + MWUA
* degree-enhanced variants
* adaptive decay variants

---

## LP-only branching performs worst

| Instance | LP-Only | MWUA-Only |
|---|---|
| n=20, trial 0 | 29 | 17 |
| n=20, trial 1 | 29 | 19 |

Local LP ambiguity alone was insufficient for effective guidance.

---

## MWUA dominates the framework

| Instance | Full | MWUA-Only |
|---|---|
| n=20, trial 0 | 17 | 17 |
| n=20, trial 2 | 19 | 19 |

This strongly suggests:

[
\boxed{
\text{
The root MWUA snapshot carries
most of the useful long-range search information.
}
}
]

---

## Adaptive decay schedules unnecessary

Removing:

* certainty decay,
* fallback mixing,
* adaptive coefficient fading,

produced nearly identical behaviour.

This significantly simplified the architecture and strengthened the persistence hypothesis.

---

# Experiment 6 — Reduction Rules vs Structural Guidance

## Objective

Determine whether improvements arise from:

* branching guidance,
* or exact reduction rules.

---

## Branching dominates reductions

| Instance | Reductions ON | Reductions OFF |
|---|---|
| n=20, trial 0 | 13 | 13 |
| n=20, trial 1 | 13 | 13 |

Prune rates remained effectively unchanged.

[
\boxed{
\text{
certainty-guided branching itself,
not aggressive reduction machinery,
is the dominant source of search efficiency.
}
}
]

---

# Experiment 7 — Residual Simplification & Reduction Overhead

## Objective

Analyze whether certainty-guided branching naturally produces cleaner residual graph states.

---

## Reduction overhead drops substantially

At (n=25):

| Strategy        | Reduction Fixes |
| --------------- | --------------- |
| certainty_first | 71              |
| most_fractional | 221             |
| random          | 102             |

Despite exploring fewer nodes, certainty-first required dramatically fewer reduction operations.

This suggests:

* earlier structural stabilization,
* cleaner residual graphs,
* fewer tangled subproblems.

---

# Emerging Hypothesis

The current results strongly support:

[
\boxed{
\text{
MWUA structural certainty behaves like
a lightweight approximation of backbone stability.
}
}
]

This framing:

* explains persistence,
* explains accelerated stabilization,
* explains residual graph collapse,
* explains why MWUA-only performs strongly,
* explains why LP-only underperforms.

---

# Current Dissertation Direction

## Root Stage

Compute once:

* MWUA structural certainty,
* global structural statistics,
* root-level confidence signals.

---

## Search Stage

Use lightweight local refinement:

* LP certainty,
* residual graph information,
* persistence,
* local simplification signals.

---

## Lightweight ML Stage

Replace manual branching formulas with:

* Random Forest,
* XGBoost,
* lightweight tabular ML.

using features such as:

* MWUA certainty,
* LP certainty,
* persistence,
* residual density,
* local reduction gain.

---

# Central Scientific Claim

[
\boxed{
\text{
Root-level structural certainty remains informative
surprisingly deep into exact combinatorial search
and induces accelerated residual graph simplification
without repeated expensive global recomputation.
}
}
]
