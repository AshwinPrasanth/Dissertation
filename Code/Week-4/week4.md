# Implementation Documentation- Week-3

# branching.py: Certainty-Guided Branching Framework

`branching.py` implements the branching component of the exact DFS Branch-and-Bound framework used throughout the dissertation experiments.

The module studies the central hypothesis that:

> structurally stable variables provide stronger long-range search guidance than classical uncertainty-first branching.

Instead of relying solely on local LP ambiguity, the framework combines:

* root-level global structural certainty,
* lightweight dynamic local graph information,
* empirical branching history,
* and persistence signals accumulated during search.

The implementation is intentionally lightweight and avoids expensive per-node global recomputation.

---

# Core Branching Philosophy

Classical exact solvers typically branch on the **most fractional** LP variable:

```math
x_i \approx 0.5
```

under the assumption that maximum uncertainty should be resolved first.

In contrast, this framework explores a **certainty-first** hypothesis:

> variables exhibiting strong structural certainty may stabilize the search tree earlier and induce faster residual graph simplification.

The branching score combines multiple structural signals:

```text
score(v) =
    α · LP_certainty(v)
  + β · MWUA_certainty(v)
  + γ · local_structural_signal(v)
  + δ · pseudo_cost(v)
  + η · persistence(v)
  + ζ · local_reduction_gain(v)
```

where:

| Signal                  | Purpose                             |
| ----------------------- | ----------------------------------- |
| LP certainty            | local relaxation confidence         |
| MWUA certainty          | root-level global structural signal |
| local structural signal | residual graph influence            |
| pseudo-cost             | empirical branching utility         |
| persistence             | stability across DFS exploration    |
| local reduction gain    | expected simplification potential   |

The framework therefore combines:

* static global structure,
* dynamic local refinement,
* temporal stabilization behaviour,
* and empirical search feedback.

---

# Branching Strategies

The module provides several interchangeable branching policies for controlled ablation studies.

## 1. CertaintyFirstBranching

Primary dissertation strategy.

Combines:

* LP certainty,
* MWUA structural certainty,
* residual graph information,
* pseudo-cost reliability,
* persistence tracking,
* and local propagation estimates.

This strategy evaluates whether lightweight structural certainty can guide DFS exploration more effectively than uncertainty-first branching.

---

## 2. MostFractionalBranching

Classical LP baseline.

Selects the variable closest to:

```math
0.5
```

representing maximum LP ambiguity.

Used as the primary comparison baseline against certainty-guided branching.

---

## 3. MWUAOnlyBranching

Ablation strategy using only the root-level MWUA structural snapshot.

No local LP ranking is used.

This directly evaluates the dissertation hypothesis that:

> a single root-level structural snapshot may remain informative deep into exact search.

---

## 4. DegreeBranching

Graph-structural baseline.

Prioritizes high-degree residual variables and local graph influence without using LP or MWUA information.

---

## 5. PseudoCostBranching

Pure empirical branching strategy.

Uses historical branching effectiveness estimates without structural guidance.

This tests whether structural certainty provides additional value beyond classical pseudo-cost reasoning.

---

## 6. RandomBranching

Uniform random baseline.

Used primarily for sanity-check comparisons.

---

# Pseudo-Cost Tracking

The framework includes a lightweight pseudo-cost tracker which estimates the historical usefulness of branching decisions.

For variable `i` and direction `d ∈ {0,1}`:

```text
PC(i,d)
```

stores the average objective improvement obtained from previous branches.

Reliability gating is used so pseudo-costs are trusted only after sufficient observations.

---

# Persistence Signals

A key research component of the framework is the introduction of:

## Certainty Persistence

Variables that repeatedly stabilize toward the same assignment during DFS exploration accumulate higher persistence scores.

This mechanism acts as a lightweight approximation of:

* long-range stabilization,
* structural rigidity,
* and pseudo-backbone behaviour.

The persistence mechanism is central to the dissertation’s emerging hypothesis on:

> structural persistence during exact search.

---

# Research Motivation

The framework is designed to investigate whether:

* root-level structural information remains useful deep into exact search,
* repeated expensive recomputation is unnecessary,
* and lightweight global-local feature fusion can induce accelerated residual graph collapse.

The implementation intentionally prioritizes:

* interpretability,
* lightweight computation,
* exact-search compatibility,
* and controlled experimental analysis.

---

# Current Research Direction

The experiments increasingly suggest that:

> MWUA structural certainty behaves like a lightweight approximation of backbone stability.

This hypothesis is currently being investigated through:

* persistence analysis,
* residual graph evolution,
* density-wise search dynamics,
* and lightweight ML-guided branching experiments.
