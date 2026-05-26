# MWUA Feature Extraction Framework (CPAIOR 2026 Reference Implementation)

This repository contains the reference implementation of the **Multiplicative Weights Update Algorithm (MWUA)** feature generation framework used in the CPAIOR 2026 work:

> *“A Scalable Learning Approach for Efficient Computation of Independent Set and Cover Variants”*
> Ryan O’Connor, Noah Coleman, Darren Strash, Saurabh Ray, and Deepak Ajwani

The implementation focuses on generating lightweight structural signals for combinatorial optimization problems such as:

* Minimum Vertex Cover (MVC)
* Maximum Independent Set (MIS)
* General Hitting Set Problems

The code is designed as a **fast structural feature generator**, not as a full exact optimization solver.

---

# Core Idea

The framework computes a **global structural certainty snapshot** over a graph using a variant of the **Multiplicative Weights Update Algorithm (MWUA)**.

Instead of repeatedly recomputing expensive graph features during optimization, the algorithm:

1. analyzes the graph structure once,
2. iteratively updates constraint weights,
3. generates fractional variable assignments,
4. and produces structural certainty signals that can later guide:

   * branching,
   * pruning,
   * reductions,
   * or learning-to-branch heuristics.

The implementation is lightweight, scalable, and structure-driven.

---

# Repository Structure

## `mwua_feature.cpp`

High-level feature generation pipeline.

Responsibilities:

* reads graph edge-list input,
* constructs the hitting-set formulation,
* converts MVC/MIS constraints into MWUA-compatible structures,
* executes the MWUA solver,
* writes fractional solutions and structural features.

Important concept:

* each edge becomes a constraint/set,
* vertices become selectable elements,
* the graph is transformed into a generic hitting-set problem.

This file mainly handles preprocessing and feature extraction orchestration.

---

## `mwua_impl.cpp`

Core MWUA implementation.

This file contains the actual optimization logic responsible for:

* constraint weighting,
* iterative multiplicative updates,
* greedy fractional assignment,
* violation/slack computation,
* running-average consensus generation.

### Key Components

#### `greedySolveCombined(...)`

Most important routine in the implementation.

Given current constraint weights:

* greedily constructs fractional assignments,
* prioritizes high-pressure variables,
* distributes coverage mass efficiently,
* and generates interim structural solutions.

This method is central to how the framework produces:

* soft certainty estimates,
* structural pressure signals,
* and lightweight global guidance.

---

#### Constraint Weight Updates

The MWUA iteratively:

1. evaluates constraint satisfaction,
2. measures violation/slack,
3. upweights difficult constraints,
4. and recomputes fractional solutions.

This gradually builds:

* global structural awareness,
* variable importance estimates,
* and stable consensus assignments.

---

#### `Xavg`

The final output is not a single instantaneous solution.

Instead, the framework maintains:

* a running average of fractional assignments across iterations.

This averaged solution acts as:

* a global structural certainty map,
* highlighting variables consistently pushed toward:

  * 0 (unlikely),
  * or 1 (structurally important).

---

## `mwua.h`

Header file defining:

* solver interfaces,
* data structures,
* configuration parameters,
* and helper utilities.

---

## `toy1graph`

Small toy graph instance for testing and experimentation.

Can be used to:

* compile the implementation,
* verify outputs,
* and inspect generated MWUA feature values.

---

# Conceptual Interpretation

This MWUA framework should be interpreted as:

```text
Global Structural Signal Generator
```

NOT as:

* a full exact solver,
* a branch-and-bound engine,
* or a neural optimization system.

The generated fractional certainties can later be used inside:

* exact branch-and-bound solvers,
* learning-to-branch systems,
* pruning heuristics,
* or hybrid optimization pipelines.

---

# Important Theoretical Insight

The implementation is fundamentally:

* structure-dependent,
* not size-dependent.

The generated signals depend on:

* graph topology,
* constraint interactions,
* residual structural pressure,
* and hitting-set geometry,

rather than:

* graph IDs,
* fixed graph sizes,
* or learned embeddings.

This makes the framework naturally scalable and transferable across graph distributions.

---

# Relationship to Exact Optimization

This implementation itself does NOT perform:

* exact branch-and-bound,
* pruning,
* cutting planes,
* or combinatorial search.

Instead, it provides:

* lightweight global structural priors
  which can guide exact search systems.

In a larger solver architecture, MWUA can act as:

```text
Global Prior
    +
Dynamic Local Search Corrections
```

where:

* MWUA provides root-level structural certainty,
* and local search mechanisms adapt during optimization.

---

# Why This Matters

Classical exact branching methods such as strong branching are powerful but computationally expensive because they repeatedly solve local LP approximations during search.

This framework explores an alternative philosophy:

```text
Can a single cheap global structural snapshot remain useful deep into optimization?
```

The MWUA-generated certainties attempt to capture:

* persistent structural importance,
* backbone-like variables,
* and globally difficult constraints
  using only lightweight iterative updates.

---
