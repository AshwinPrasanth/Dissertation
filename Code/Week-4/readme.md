# MWUA-Guided Structural Persistence in Exact Search

Lightweight exact Branch-and-Bound framework for studying how root-level structural information persists during combinatorial optimization search.

The project investigates whether a single global structural snapshot, computed once at the root using a principled MWUA formulation, remains informative surprisingly deep into exact DFS exploration without repeated expensive global recomputation.

The framework focuses on:

* structural persistence,
* certainty-guided branching,
* residual graph evolution,
* search-trajectory shaping,
* and lightweight global-local exact search guidance.

---

# Central Research Hypothesis

> Root-level MWUA structural certainty behaves like a lightweight approximation of backbone stability and remains informative surprisingly deep into exact combinatorial search.

The experiments increasingly suggest that:

* structurally certain variables stabilize earlier,
* residual graph density collapses faster,
* ambiguity persists selectively,
* and certainty-guided branching scales substantially better than uncertainty-first branching.

---

# Dissertation Motivation

Modern ML-guided combinatorial solvers often rely on:

* repeated graph embedding recomputation,
* large neural architectures,
* or expensive online feature extraction.

This project explores an alternative direction:

```text
compute expensive global structure once,
then reuse it throughout exact search
using lightweight local refinement.
```

The framework therefore studies:

```text
static global structure
            +
lightweight local refinement
            +
exact DFS search
```

rather than:

* heavy neural recomputation,
* end-to-end black-box optimization,
* or reduction-heavy heuristic systems.

---

# System Architecture

```text
               Graph / MVC Instance
                        │
                        ▼
              Root LP Relaxation
                        │
                        ▼
         MWUA Structural Snapshot
         (computed once at root)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
 Global Structural               LP Certainty
    Certainty                   Local Refinement
         │                             │
         └──────────────┬──────────────┘
                        ▼
          Certainty-Guided Branching
                        │
                        ▼
           Exact DFS Branch-and-Bound
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Persistence     Residual Graph   Pseudo-Costs
     Tracking         Evolution
                        │
                        ▼
           Structural Collapse Analysis
```

---

# Exact Search Preservation

The framework is a fully exact combinatorial solver.

Structural guidance influences:

* branching order,
* search trajectory,
* and residual simplification,

while preserving:

* LP correctness,
* exact pruning,
* and optimality guarantees.

The objective is not to replace exact search, but to study how structural certainty shapes exact search dynamics.

---

# Core Components

| File             | Purpose                                    |
| ---------------- | ------------------------------------------ |
| `core.py`        | MWUA preprocessing + residual graph engine |
| `branching.py`   | certainty-guided branching policies        |
| `solver.py`      | exact DFS Branch-and-Bound solver          |
| `experiments.py` | structural persistence analysis suite      |

---

# Branching Strategies

| Strategy          | Description                        |
| ----------------- | ---------------------------------- |
| `certainty_first` | MWUA + LP certainty                |
| `mwua_only`       | root structural certainty only     |
| `most_fractional` | classical LP uncertainty branching |
| `degree`          | local graph-topology heuristic     |
| `pseudo_cost`     | empirical branching                |
| `random`          | random baseline                    |

---

# Experimental Philosophy

The framework studies exact search as a dynamic evolving process rather than only measuring:

* runtime,
* explored nodes,
* or objective values.

The experiments analyze:

```text
Root Structural Snapshot
            │
            ▼
   Search Trajectory Evolution
            │
            ▼
  Residual Graph Simplification
            │
            ▼
 Structural Stabilization
            │
            ▼
 Exact Search Efficiency
```

The primary focus is therefore:

* search dynamics,
* persistence,
* stabilization,
* and structural collapse.

---

# Experiment 1 — Variable Stability and Deferred Uncertainty

## Objective

To determine whether variables with high root-level MWUA certainty stabilize earlier during DFS search.

Tracked metrics:

* MWUA certainty,
* fractional frequency,
* assignment stability,
* directional fixation frequencies.

---

## Key Findings

### 1. High-certainty variables stabilize rapidly

Variables with MWUA certainty near 1.0 exhibited:

* extremely low fractional persistence,
* high assignment consistency,
* strong directional stabilization.

| Variable | MWUA Certainty | Fractional Frequency | Stability |
| -------- | -------------- | -------------------- | --------- |
| 0        | 1.00           | 0.018                | 0.857     |
| 10       | 1.00           | 0.036                | 0.793     |

These variables almost never remained ambiguous during deeper DFS exploration.

---

### 2. Deferred uncertainty emerges naturally

Mid-certainty variables exhibited:

* significantly higher ambiguity persistence,
* increased oscillatory behaviour,
* and delayed stabilization.

| MWUA Certainty | Fractional Frequency | Stability |
| -------------- | -------------------- | --------- |
| 0.73           | ~0.29                | ~0.37     |
| 0.46           | ~0.42                | ~0.46     |

This suggests that uncertainty is not uniformly distributed throughout search.

Instead:

* high-certainty variables stabilize early,
* medium-certainty variables stabilize gradually,
* low-certainty variables remain persistent ambiguity sources.

This phenomenon is referred to as:

# Deferred Uncertainty

---

### 3. Backbone-like behaviour emerges naturally

Variables with:

* high MWUA certainty,
* low fractional persistence,
* high stability,

behave similarly to pseudo-backbones:
variables whose assignments become effectively fixed throughout large portions of the search tree.

Importantly:

* no explicit backbone computation was used,
* the behaviour emerged directly from root structural certainty.

---

# Experiment 2 — Residual Graph Evolution and Search Dynamics

## Objective

To analyze how branching strategies affect:

* residual graph collapse,
* structural stabilization,
* and downstream search complexity.

Compared strategies:

* `certainty_first`
* `mwua_only`
* `most_fractional`

Tracked:

* residual density,
* persistence,
* LP certainty,
* active vertices,
* local reduction gain.

---

## Key Findings

### 1. Certainty-first accelerates residual collapse

Residual density at depth 20:

| Strategy        | Residual Density |
| --------------- | ---------------- |
| certainty_first | 0.248            |
| mwua_only       | 0.295            |
| most_fractional | 0.419            |

Despite similar active-vertex reductions, certainty-guided branching simplified the residual structure substantially faster.

This suggests:
certainty-guided branching creates structurally easier downstream subproblems.

---

### 2. MWUA-only already outperforms LP branching

Even without local refinement:

* MWUA-only produced lower residual densities,
* faster stabilization,
* and stronger collapse behaviour

than classical LP branching.

This directly supports the hypothesis that:
a single root structural snapshot contains meaningful long-range search information.

---

### 3. Static global + lightweight local performs best

The strongest behaviour consistently emerged from combining:

* root MWUA certainty,
* with local LP refinement.

This directly validates the dissertation proposal philosophy:

```text
static global structure
+
lightweight local updates
```

---

# Experiment 3 — Density-Wise Branching Behaviour

## Objective

To analyze how branching strategies behave across varying graph densities.

Compared:

* `certainty_first`
* `most_fractional`
* `degree`

Density range:

```text
0.1 → 0.6
```

Tracked:

* explored nodes,
* runtime,
* prune rates,
* reduction fixes.

---

## Key Findings

### 1. Certainty-first consistently reduces explored nodes

| Density | certainty_first | most_fractional |
| ------- | --------------- | --------------- |
| 0.20    | 9               | 15              |
| 0.40    | 17              | 29              |
| 0.50    | 21              | 29              |
| 0.60    | 31              | 43              |

This demonstrates that:
root structural certainty provides substantially better search guidance than local LP ambiguity alone.

---

### 2. Improvements are not caused by stronger pruning

Prune rates remained nearly identical:

| Strategy        | Prune Rate |
| --------------- | ---------- |
| certainty_first | 0.516      |
| most_fractional | 0.512      |

This is scientifically important because it suggests:
the advantage originates from improved search trajectories rather than aggressive pruning.

---

### 3. Certainty-first produces cleaner residual states

At density 0.60:

| Strategy        | Reduction Fixes |
| --------------- | --------------- |
| certainty_first | 106             |
| most_fractional | 180             |

Despite exploring fewer nodes, certainty-first required substantially fewer reduction-triggering states.

This suggests that certainty-guided branching naturally steers search toward:

* cleaner residual subproblems,
* earlier stabilization,
* and reduced structural entanglement.

---

### 4. Regime-dependent behaviour appears

At density 0.30:

| Strategy        | Explored Nodes |
| --------------- | -------------- |
| certainty_first | 33             |
| degree          | 13             |

This demonstrates that:

* MWUA-guided certainty is not uniformly dominant,
* graph structure strongly influences guidance quality,
* and different structural regimes may favour different signals.

This result strengthens the scientific credibility of the framework by revealing nuanced behaviour rather than universal dominance.

---

# Experiment 4 — Scaling Behaviour and Structural Persistence

## Objective

To determine whether root structural certainty remains useful as graph size increases.

Graph sizes:

```text
n = 10 → 80
```

Compared:

* certainty_first
* mwua_only
* most_fractional
* pseudo_cost
* degree
* random

---

## Key Findings

### 1. Certainty-first scales substantially better

| Graph Size | certainty_first | most_fractional |
| ---------- | --------------- | --------------- |
| 25         | 25              | 51              |
| 40         | 75              | 171             |
| 50         | 355             | 771             |
| 60         | 631             | 1557            |
| 80         | 1423            | 3509            |

At:

```text
n = 80
```

certainty-first explored approximately:

```text
59% fewer nodes
```

than classical LP branching.

Importantly:
the advantage increases with scale rather than disappearing.

---

### 2. Runtime improvements mirror node reductions

| Graph Size | certainty_first | most_fractional |
| ---------- | --------------- | --------------- |
| 50         | 13.9s           | 17.7s           |
| 60         | 54.5s           | 68.0s           |
| 80         | 259.8s          | 366.8s          |

This indicates that:
certainty-guided branching simplifies downstream subproblems rather than merely shrinking tree size cosmetically.

---

### 3. Similar prune rates imply trajectory improvement

Typical prune rates:

| Strategy        | Typical Range |
| --------------- | ------------- |
| certainty_first | ~50–57%       |
| most_fractional | ~50–52%       |
| mwua_only       | ~50–53%       |

The gains therefore originate primarily from:

* earlier stabilization,
* reduced ambiguity persistence,
* and improved traversal of the search tree.

---

### 4. MWUA-only already provides strong long-range guidance

Example at:

```text
n = 40
```

| Strategy        | Explored Nodes |
| --------------- | -------------- |
| mwua_only       | 123            |
| most_fractional | 203            |
| pseudo_cost     | 263            |
| random          | 277            |

This strongly supports the dissertation hypothesis that:
root structural certainty alone carries meaningful long-range search information.

---

### 5. Structural predictability interpretation

The experiments increasingly suggest that MWUA certainty behaves less like a local importance score and more like a predictor of future search stability.

Variables with high MWUA certainty:

* stabilize earlier,
* remain fractional less often,
* and exhibit more deterministic DFS trajectories.

This interpretation aligns strongly with the observed pseudo-backbone behaviour.

---

# Experiment 5 — MWUA Ablation and Structural Signal Analysis

## Objective

To isolate the contribution of individual structural signals.

Compared:

* `lp_only`
* `mwua_only`
* `lp+mwua`
* `lp+mwua+degree`
* `full`
* `no_decay`

---

## Key Findings

### 1. LP-only consistently performs worst

| Instance      | LP-Only | MWUA-Only |
| ------------- | ------- | --------- |
| n=15, trial 0 | 13      | 7         |
| n=20, trial 0 | 29      | 17        |

This suggests that:
local LP ambiguity alone is insufficient for strong exact-search guidance.

---

### 2. MWUA-only nearly matches the full framework

| Instance      | Full Framework | MWUA-Only |
| ------------- | -------------- | --------- |
| n=20, trial 0 | 17             | 17        |
| n=20, trial 2 | 19             | 19        |

This is one of the strongest findings in the project.

It suggests that:
the root MWUA structural snapshot itself contains most of the useful long-range search information.

---

### 3. Additional heuristic complexity contributes surprisingly little

Degree heuristics and additional handcrafted terms produced minimal measurable improvement.

This significantly simplifies the emerging architecture.

The results increasingly support a cleaner framework centered around:

* root structural certainty,
* lightweight local refinement,
* and exact DFS search.

---

### 4. Adaptive decay schedules appear unnecessary

The `no_decay` configuration behaved almost identically to the full framework.

This suggests that:
root structural certainty naturally persists deep into exact search without requiring engineered decay schedules.

This is a major architectural simplification result.

---

# Experiment 6 — Reduction Rules vs Structural Guidance

## Objective

To isolate whether improvements originate from:

* branching guidance,
* or reduction machinery.

Compared:

* reductions ON
* reductions OFF

---

## Key Findings

### 1. Reductions do not significantly alter search-tree structure

| Instance      | Reductions ON | Reductions OFF |
| ------------- | ------------- | -------------- |
| n=15, trial 0 | 7             | 7              |
| n=20, trial 1 | 13            | 13             |

This strongly suggests that:
the branching trajectory itself is the dominant source of search efficiency.

---

### 2. Prune behaviour remains effectively identical

Prune rates remained nearly unchanged across configurations.

This demonstrates that:
the framework’s improvements are not artifacts of aggressive reduction pipelines.

---

### 3. Reduction overhead can exceed benefits

Although reductions simplify local states, they also introduce measurable runtime overhead.

This suggests that:
certainty-guided branching already induces substantial residual simplification naturally.

---

### 4. Structural guidance dominates reduction effects

The experiments increasingly support the conclusion that:
global structural certainty is the primary source of search guidance,
while reductions act mainly as secondary local simplifiers.

---

# Experiment 7 — Reduction Overhead and Residual Simplification

## Objective

To determine whether certainty-guided branching naturally produces cleaner residual graph states.

Compared:

* certainty_first
* most_fractional
* random

---

## Key Findings

### 1. Certainty-first drastically reduces reduction overhead

Example at:

```text
n = 25
```

| Strategy        | Reduction Fixes |
| --------------- | --------------- |
| certainty_first | 71              |
| most_fractional | 221             |
| random          | 102             |

Despite exploring fewer nodes, certainty-first also required substantially fewer reductions.

This strongly suggests:
certainty-guided branching naturally avoids highly tangled residual states.

---

### 2. Similar prune rates reinforce trajectory interpretation

Because prune rates remain similar, the gains appear to originate primarily from:

* cleaner search trajectories,
* accelerated stabilization,
* and reduced structural entanglement.

---

# Correlation Results

Observed correlations:

```text
corr(MWUA, stability) = 0.398
corr(MWUA, fractional persistence) = -0.826
```

The strong negative correlation with fractional persistence is particularly important.

It indicates that:

* variables with high MWUA certainty stabilize earlier,
* remain fractional less often,
* and exhibit more deterministic assignment behaviour during DFS exploration.

This strongly supports the structural-persistence hypothesis.

---

# Emerging Scientific Interpretation

The experiments increasingly suggest that the framework behaves less like:

* a conventional branching heuristic,

and more like:

# search-trajectory shaping

where:

* root structural certainty,
* stabilization dynamics,
* and residual collapse

jointly influence how DFS exploration evolves over time.

The primary advantage therefore appears to be:

* earlier structural collapse,
* reduced ambiguity persistence,
* and traversal toward structurally easier subproblems earlier during search.

---

# Emerging Dissertation Architecture

## Root Stage

Compute once:

* MWUA structural certainty,
* global structural statistics,
* root-level confidence signals.

---

## Search Stage

Compute lightweight local updates:

* LP certainty,
* residual graph information,
* persistence signals,
* local simplification estimates.

---

## Lightweight ML Stage (Next Phase)

Replace manually weighted branching formulas with lightweight ML models such as:

* XGBoost,
* Random Forests,
* lightweight tabular learners.

The model will learn branching utility from:

* MWUA certainty,
* LP certainty,
* persistence,
* residual density,
* propagation gain,
* local graph structure.

The objective is:

* learned structural fusion,
* without sacrificing scalability,
* interpretability,
* or exact-search compatibility.

---

# Central Conclusion

The current experiments provide strong evidence that:

> a single root-level structural snapshot can remain informative surprisingly deep into exact combinatorial search and can naturally induce accelerated structural collapse without repeated expensive global recomputation.

The framework increasingly supports the broader hypothesis that:

> structural persistence is a fundamental phenomenon governing exact DFS search dynamics.
