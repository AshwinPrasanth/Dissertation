# Branch & Bound Variable Selection — Reading Notes

**Thesis:** Chayanne, Section 4.3  
**Date:** 18 May 2026 — Week 0 → 1  
**Topic:** Balancing the effectiveness of strong branching against cost; slashing reduced-tree benefit

---

## Table of Contents

1. [Overview](#overview)
2. [Branch and Bound — Fundamentals](#1-branch-and-bound--fundamentals)
3. [Strong Branching](#2-strong-branching)
4. [Heuristics — First Fail / Min Remaining Value](#3-heuristics--first-fail--min-remaining-value)
5. [Offline Learning — Behavioural Cloning](#4-offline-learning--behavioural-cloning)
6. [GNN Architecture](#5-gnn-architecture)
7. [Online Proxy Learning — 3-Phase Pipeline](#6-online-proxy-learning--3-phase-pipeline)
8. [SVM Rank — Learning to Branch](#7-svm-rank--learning-to-branch)
9. [Dataset Construction & Bipartite Labelling](#8-dataset-construction--bipartite-labelling)
10. [Related Work — Papers \[13\], \[15\], \[20\]](#9-related-work--papers-13-15-20)
11. [Open Questions & Items to Follow Up](#10-open-questions--items-to-follow-up)

---

## Overview

Section 4.3 covers how to learn a variable selection policy for Branch & Bound (B&B) solvers. The core tension is:

- **Strong Branching (SB)** is the gold-standard variable selector but is computationally expensive (requires 2× LP solves per candidate variable).
- The goal is to **approximate SB cheaply**, either offline (behavioural cloning / GNN) or online (proxy ranking via SVM Rank), and switch to it once it is reliable enough.

Variable selection is framed as an **optimisation problem**; feature selection is framed as an **ML problem**.

---

## 1. Branch and Bound — Fundamentals

### MIP formulation

Mixed Integer Programming minimises a linear cost over integer/binary variables:

```
z* = min { c^T x | Ax ≥ b, x ∈ ℝⁿ, xᵢ ∈ ℤ }
```

### LP relaxation

Deletes the integrality requirement. Gives a lower bound:

```
z̄ ≤ z*
```

If the LP relaxation solution contains fractional variables, B&B splits on one of them.

### Splitting rules

Given a fractional variable xⱼ with value x̂ⱼ:

| Branch | Constraint added |
|--------|-----------------|
| Downward | xⱼ ≤ ⌊x̂ⱼ⌋ |
| Upward   | xⱼ ≥ ⌈x̂ⱼ⌉ |

The **candidate set** contains all fractional variables at a node. A score is computed for each; the best is selected:

```
j* = argmax sⱼ
```

> **Note (§4.31):** Branching at the top of the search tree has the greatest impact. More at top → better overall effect.

---

## 2. Strong Branching

Strong Branching explicitly evaluates the LP bound improvement for each candidate variable.

### Score formula

For each candidate xⱼ, create two child nodes (round up and round down) and measure the lower-bound improvement:

```
Δ⁻ⱼ = z̄⁻ⱼ − z̄        (downward branch improvement)
Δ⁺ⱼ = z̄⁺ⱼ − z̄        (upward branch improvement)

SBⱼ = max(Δ⁻ⱼ, ε) × max(Δ⁺ⱼ, ε)
```

Multiplying (rather than summing) ensures we select the variable with the least worst shrink — a variable that improves both directions.

### Problem

Requires **2 LP solves per candidate variable** → computationally very expensive at scale.

### Pseudocode branching (avg gain per unit change)

```
ψ⁻ⱼ  = avg drop per unit (downward)
ψ⁺ⱼ  = avg gain per unit (upward)

Estimate bound change:  −(xⱼ − ⌊xⱼ⌋) · ψ⁻ⱼ × (xⱼ − ⌊xⱼ⌋) · ψ⁺ⱼ
```

---

## 3. Heuristics — First Fail / Min Remaining Value

*(Section 4.3.3)*

- **Rule:** Pick the variable with the smallest remaining domain.
- **Rationale:** Fail to succeed — detect dead-ends early to avoid wasting compute time.
- **Limitation:** Only looks at local information; ignores global context.

---

## 4. Offline Learning — Behavioural Cloning

*(Section 4.3.2)*

### Motivation

Instead of learning general branching rules, learn **specialised branching rules per instance**. Approximate strong branching *before* B&B begins.

### Behavioural cloning pipeline

```
SB → State, SB action (data) → Classification
```

The SB decisions are treated as expert demonstrations. A classifier learns to imitate them.

### Why not Reinforcement Learning?

B&B is modelled as an MDP (Markov Decision Process):

- **Environment:** solver state
- **Agent:** branching policy
- **State:** rule / solver snapshot
- **Action:** variable to branch on

RL fails here for two reasons:

1. Branch decisions are only known to be good at the **end of the full episode** (when the tree closes). Reward is extremely delayed.
2. The environment is **chaotic** — small decisions cascade unpredictably.

**Imitation learning** is used instead, cloning SB behaviour directly.

### MILP as a bipartite graph

The key insight enabling GNN-based learning:

```
MILP → Bipartite Graph
         ├── Variable nodes
         └── Constraint nodes
              └── Edge: variable xᵢ has non-zero coefficient in constraint cⱼ
```

This representation is:
- **Permutation-invariant** — variable ordering doesn't matter
- **Size-generalising** — the GNN looks at local neighbourhoods; it doesn't care about global problem size
- **Sparse** — exploits MILP sparsity (MILP-fast)

> **Key result:** Train GNN on small, cheap SB instances → massive performance on large instances. This is exactly what is needed: **GNN scaling**.

### Bipartite state encoding

State is encoded as G = (C, E, V):

- **C** — constraint nodes
- **V** — variable nodes
- **E** — edges

No flattening to a fixed-width vector. Handles m constraints and n variables natively.

---

## 5. GNN Architecture

### Policy parameterisation — interleaved half-convolution

Standard GNN: passes messages simultaneously across all neighbours.

Custom convolution layer (required because graph is bipartite — two distinct node types):

**Step 1 — Variable → Constraint pass:**

```
cᵢ ← fc(cᵢ, Σ_{(cᵢ,vⱼ)∈E} gc(cᵢ, vⱼ, eᵢⱼ))
```

**Step 2 — Constraint → Variable pass:**

```
vⱼ* ← fv(vⱼ, Σ gv(cᵢ, vⱼ, eᵢⱼ))
```

Where:
- `gv`, `gc` — message generator functions (generate message vectors)
- `Σ` — aggregates incoming messages (no ordering)
- `fc`, `fv` — fuse two vectors: variable's local info + message bundle info (node update functions)

> **GNN complexity ∝ sparsity.** More efficient with sparse graphs; dense graphs increase complexity significantly.

### Masked softmax output

After message passing:

1. Each variable node holds a fused **64-dimensional vector**.
2. Score variables → logits (good/not).
3. **Masked softmax:** branch fractional variables only. Perfect whole values (e.g. 7.0) are discarded (masked out).

```
e^(logit) / Σ e^(logit)   [only over unmasked variables]
```

### Why no traditional normalisation?

Traditional normalisation would cause **exact structural count dominance**: a variable appearing in 1000 constraints vs 10 constraints would look identical after normalisation, losing critical structural information.

---

## 6. Online Proxy Learning — 3-Phase Pipeline

*(Section 4.31)*

The online pipeline switches from expensive SB to a cheap learned proxy once the proxy is reliable.

### Pipeline overview

```
Start
  │
  ▼
Phase 1: Collection
  Real SB runs → collect (features, SB score) pairs
  Assign best rank label per variable
  │
  ▼
Phase 2: Train
  Train light ranking regressor (SVM Rank)
  │
  ▼
Phase 3: Switch
  Reliability threshold met?
  ├── YES → Freeze model, use proxy for all subsequent branching
  └── NO  → Collect more data, retrain
```

### Feature engineering

Each variable xᵢ has features φ(xᵢ):

| Feature type | Count |
|--------------|-------|
| Static       | 18    |
| Dynamic      | 54    |

Additional: neighbourhood features, stats.

**Sample table structure:** one variable = one data point (per node)

| Data point (Node) | Variable | f₁  | f₂  |
|-------------------|----------|-----|-----|
| 1                 | x₁       | 0.7 | 3.0 | ← Train Proxy
| 2                 | x₂       | 0.0 | 5.0 |

> **Target** is only calculated at early stages of the tree during SB, until the reliability threshold holds.  
> **Train model:** cheap features → expensive SB target.

### SVM feature vector

Variable xᵢ is converted to a feature vector:

```
φ(xᵢ) = [φᵤxᵢ]  ∈ ℝᵈ
          [φᵈxᵢ]
```

Where φᵤ = variable features, φᵈ = domain/dynamic features.

---

## 7. SVM Rank — Learning to Branch

*(Paper [13])*

### Prediction vs ranking proxy

| Approach | Paper | Method |
|----------|-------|--------|
| Predict exact score | [15] | Guess exact SB score — IMP |
| Predict ordering    | [13] | Predict ordering of variables — Learning to Rank [online] |

### Pairwise training

SB computes `score(xᵢ) > score(xⱼ)`, building preference constraints.

SVM Rank finds weight vector **w** satisfying preference pairs `(xᵢ > xⱼ)` via:

```
f(x) = wᵀ φ(x)
```

**Optimisation problem:**

```
min   ½‖w‖² + C Σᵢⱼ ξᵢⱼ
w,ξ

subject to:
  wᵀ(φ(xᵢ) − φ(xⱼ)) ≥ 1 − ξᵢⱼ    ∀ preference pairs
  ξᵢⱼ ≥ 0
```

Training freezes once this optimisation problem is solved. The optimal **w\*** is used to compute proxies for all subsequent branching:

```
Proxy(xₖ) = w*ᵀ φ(xₖ)
```

Calculate vector of proxies → sort → branch on top variable.

### Pairwise ranking loss (at node Nᵢ)

Preference set at node Nᵢ:

```
Pᵢ = {(xⱼ, xₖ) : (j, k ∈ Cᵢ) and yᵢ ≥ yₖ}
```

Where one variable is good (label 1) and the other bad (label 0). Goal: find optimal weight that gives higher score to good than bad.

**Full SVM Rank objective:**

```
w* = argmin  Σ    1    Σ          max(0, 1 − wᵀ(φ(xⱼ,Nᵢ) − φ(xₖ,Nᵢ))) + λ‖w‖²
           N∈N  |Pᵢ|  (j,k)∈Pᵢ
```

> **Open question:** Normalising penalty and how λ‖w‖² interacts with the pairwise ranking objective. ← ASK

### Degree-2 polynomial kernel

SVM cannot directly infer relationships between features. Solution: degree-2 polynomial mapping.

```
K(y, z) = (yᵀz + 1)²    (Degree 2)
```

Final feature vector φ ≈ **2600 dimensions**.

### Query-based normalisation (per node)

Features are not normalised globally — instead, min-max normalisation is applied at each node:

```
x_norm = (x − x_min) / (x_max − x_min)   ∈ [0, 1]
```

This adds **dynamicity to static features**: at each node, a variable's values are normalised relative to the other candidates at that node (immediate competitive context, not global competitiveness).

> Why? A variable with coefficient 5 when others are in range [1, 10] is very different from when others are in range [100, 1000]. Global normalisation would lose this.

---

## 8. Dataset Construction & Bipartite Labelling

*(Paper [13] — Analysis: Learning to Branch, SVM Rank)*

### Dataset construction

1. Run SB for all nodes.
2. Solver identifies candidate variable subset per successful node (10–20 variables, sorted by PC sort).
3. Data consists of three components:

| Component | Description |
|-----------|-------------|
| Nodes | Search tree logs |
| Candidates | Fractional variable evaluations per node |
| Features | φ(xᵢ, Nᵢ) — feature map to variable xᵢ at node Nᵢ |

> **Note:** The same variable can appear as multiple data points at different nodes — this is the **dynamic feature** component.

### Bipartite labelling scheme

Convert continuous, noisy SB scores to binary labels yᵢ ∈ {0, 1}:

```
yᵢ = 1    if SBⱼ ≥ (1 − α) × SB*
yᵢ = 0    otherwise
```

Where:
- `SB*` — max SB score at that node
- `α` — relaxation parameter ∈ [0, 1] (how wide the "top" group is)

**Semantics:** All variables with label `1` inside a node must be ranked higher than those with label `0`. The model can ignore `0`s and focus ranking effort on `1`s.

**Relaxed top — podium selection:** No single winner. A good podium (top group) is selected rather than forcing one best variable. This is more robust to noise in SB scores.

> ↑↑ **To ask on Monday:** How exactly the labelling scheme interacts with the ranking objective — especially the constraint that all `1`-labelled variables must be ranked above all `0`-labelled ones.

### 2 common problems with traditional MIP variable selection

1. **Bias** — rules that apply regardless of problem instance
2. **Same generic rules** irrespective of use case

Variable selection is important; design solver strategy with data.

---

## 9. Related Work — Papers \[13\], \[15\], \[20\]

### Problems with existing approaches

| Paper | Issue |
|-------|-------|
| [13] | Unnecessarily complex — requires perfect pairwise rank for all variables; ≈2600 features per pair |
| [15] | Too volatile — predicting exact score is unstable |

### Paper [20] — Gasse, GTCNN

Both search tree and linear equations **change in structure and size** across problems. A fixed-architecture network cannot generalise across them. GTCNN (Graph Tree CNN) solves both:

```
Soln to both → GNN → GTCNN
```

### Variable-constant bipartite graph

MILP is naturally a bipartite graph:

- Variable nodes ↔ constraint nodes mapping
- A variable has a non-zero coefficient in a constraint → edge exists
- GNN learns automatically through graph edges via message passing

**Size generalisation:** Achieved because GNN looks at local neighbourhood, not global size. Does not care about problem size → generalises from small train to large scale.

> **To ask:** Small train → large scale generalisation was not explicitly discussed in detail. What guarantees this?

### Branching as classification (contrast to [13])

[13] uses **ranking**. The GTCNN approach uses **classification** (behavioural cloning), optimised with low cross-entropy loss. This is simpler and avoids the complexity of perfect pairwise ranking.

### Validation

- Metric: **Spearman's rank correlation**
- Low correlation shows best improvement changes per problem instance
- Test method: hand solver z* (optimal) before search start; tree shape is then determined solely by branching variable choices
- Cuts applied to root only
- 10 random seeds to avoid large variance errors

### Metrics

```
acc @ 1, 5, 10
```

(Accuracy of top-1, top-5, top-10 variable selections)

### Connecting to MWU — Papers \[13\] and \[20\]

Both papers hardcoded a node threshold **σ = 500** for running expert strategies and switching between strategies.

---

## 10. Open Questions & Items to Follow Up

| # | Question | Priority |
|---|----------|----------|
| 1 | How to set the **reliability threshold**? When is the predictive proxy reliable enough? | High |
| 2 | **When to stop training** — decision factor: top > bottom (has less effect). [15] vs [13] comparison. | Medium |
| 3 | **Bipartite labelling** — how exactly the constraint "all `1`s must rank above all `0`s" interacts with the loss | **Monday ↑↑** |
| 4 | **Normalising penalty** — role of λ‖w‖² in SVM Rank; how it prevents weight domination | High |
| 5 | **Small train → large scale** generalisation — not explicitly discussed; what guarantees it? | Medium |
| 6 | **Tabular ML fails** — fixed-width feature vector breaks for variable-size problems. Should training follow the same complexity curriculum? | Medium |
| 7 | **ExtraTrees** — mentioned as a doubt; relationship to proxy model unclear | Low |

---

## Appendix — Key Symbols

| Symbol | Meaning |
|--------|---------|
| z* | Optimal MIP objective value |
| z̄ | LP relaxation lower bound |
| xⱼ | Variable j |
| x̂ⱼ | Fractional LP value of variable j |
| φ(x) | Feature vector mapping |
| w | Weight vector (SVM Rank) |
| SBⱼ | Strong branching score for variable j |
| Δ⁻ⱼ, Δ⁺ⱼ | Downward/upward LP bound improvement |
| ξᵢⱼ | Slack variable (SVM soft margin) |
| α | Relaxation parameter for bipartite labelling |
| SB* | Max strong branching score at a node |
| σ | Node threshold (hardcoded at 500 in [13] and [20]) |

---

*Notes taken from handwritten research journal — 18 May 2026*
