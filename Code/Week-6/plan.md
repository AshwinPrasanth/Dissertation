Based on everything we've discussed over the last few weeks (your proposal, Ryan's paper, the MWUA implementation, your supervisor's feedback, and your experimental goals), I think your dissertation can be organized into **four progressive phases**. This keeps the work focused and ensures that every experiment contributes to the central research question.

---

# Dissertation Research Plan

## Research Question

> **Can a one-time global structural representation obtained using MWUA replace repeatedly recomputed global features in Learning-to-Branch while maintaining search efficiency for exact Maximum Independent Set?**

---

# Phase I — Reduction Engine (Preprocessing)

**Goal**

Reduce the graph as much as possible **before** MWUA and SCIP.

This is **not** your contribution, but it creates a much stronger experimental platform.

Pipeline:

```text
Original Graph
        │
        ▼
Connected Components
        │
Degree-0
        │
Degree-1 (Pendant)
        │
Degree-2 Folding
        │
Twin Reduction
        │
Domination
        │
(Optional LP/Nemhauser-Trotter)
        │
        ▼
Kernel Graph
```

---

## Deliverables

For every graph record

| Metric               | Description |
| -------------------- | ----------- |
| Original vertices    | n           |
| Original edges       | m           |
| Kernel vertices      | n'          |
| Kernel edges         | m'          |
| Vertex reduction (%) |             |
| Edge reduction (%)   |             |
| Reduction runtime    |             |

This becomes one complete experimental section.

---

# Phase II — Global MWUA Representation

Run MWUA **once** on the reduced kernel.

Pipeline

```text
Kernel

↓

MWUA

↓

x_avg

↓

MIS score = 1 - x_avg
```

Freeze this representation.

No recomputation later.

---

## Experiments

Study

* convergence
* runtime
* score distribution
* score stability

Record

* violation curve
* runtime
* score histogram
* top-k vertices

This phase is almost complete.

---

# Phase III — Exact Branch-and-Bound Analysis

Now use SCIP.

Your branching rule uses

```text
Global

↓

MWUA score
```

No GCN yet.

---

## Branching depth study

Exactly what your proposal says.

For

```text
D = 0

...

10
```

Measure

* nodes
* runtime
* objective

Question:

> How long does the static MWUA snapshot remain useful?

---

## Search Space Analysis

This came from your supervisor.

Instead of only

```text
Nodes explored
```

Measure

```text
Search reduction %
```

Example

| Method | Nodes |
| ------ | ----- |
| SCIP   | 24000 |
| MWUA   | 15000 |

↓

Search reduction

37.5%

---

## Pruning Analysis

This is another experiment.

Every node disappears because of

### 1

Infeasible

### 2

Bound

### 3

Integral

Count

| Reason     | Count |
| ---------- | ----: |
| Infeasible |       |
| Bound      |       |
| Integral   |       |

Compare

SCIP

vs

MWUA

Question

> Which pruning mechanism benefits most from better branching?

---

## Anytime Analysis

This was another point your supervisor raised.

Measure

```text
Time

↓

Best incumbent
```

Example

| Time | Incumbent |
| ---- | --------- |
| 0    | 8         |
| 2    | 10        |
| 5    | 12        |
| 30   | 14        |
| 140  | 15        |

Compare

SCIP

vs

MWUA

Question

> Which method reaches good solutions faster?

---

# Phase IV — Learning-to-Branch

Now introduce learning.

Pipeline

```text
Kernel

↓

MWUA

↓

Branch-and-Bound

↓

Residual graph

↓

Dynamic local features

↓

Bipartite graph

↓

GCN

↓

Branch score
```

---

## Static Features

These never change.

Examples

* MWUA score
* Original degree
* Core number
* Centrality

---

## Dynamic Features

Updated every node.

Examples

* LP value
* Pseudo-cost
* Residual degree
* Fractionality
* Branch depth

---

## Bipartite Graph

Exactly like Gasse.

Variable nodes

Constraint nodes

Edges

---

## GCN

Simple

```
2 GraphConv layers

↓

MLP

↓

Ranking
```

No huge network.

The novelty is not the network.

The novelty is

> static MWUA + lightweight local updates

---

# Experimental Dataset

I would divide it into

## Synthetic

BHOSLIB

* frb30-15 (5)

* frb40-19 (5)

---

## Real-world

SNAP

Prefer

* ca-GrQc
* ca-HepTh
* ca-HepPh
* ca-CondMat
* ca-AstroPh

These remain sparse after reduction and fit your motivation well.

---

# Final Evaluation

For every graph

| Category  | Metrics                                      |
| --------- | -------------------------------------------- |
| Reduction | vertices removed, edges removed, kernel size |
| MWUA      | runtime, convergence, score statistics       |
| Branching | nodes, runtime, objective                    |
| Search    | search reduction %                           |
| Pruning   | infeasible, bound, integral                  |
| Anytime   | incumbent vs time                            |
| Learning  | accuracy, node reduction, runtime            |

---

# Contributions

I think your dissertation contributions naturally become:

### Contribution 1

A lightweight preprocessing pipeline that aggressively reduces graph size before exact search.

---

### Contribution 2

A one-time MWUA global representation computed on the kernel graph.

---

### Contribution 3

An empirical study of **how long** this static representation remains useful throughout Branch-and-Bound (the (D=0\ldots10) experiments).

---

### Contribution 4

A Learning-to-Branch framework that combines the frozen global MWUA representation with dynamic local features using a lightweight bipartite GCN.

---

# Timeline

### Phase 1 (Current)

* ✅ MWUA implementation
* ⏳ Reduction engine
* ⏳ Kernel statistics

### Phase 2

* D = 0–10 experiments
* Search-space analysis
* Pruning statistics
* Anytime curves

### Phase 3

* Bipartite graph construction
* Dataset generation
* GCN training

### Phase 4

* Final comparisons
* Ablation studies
* Dissertation writing

---

## Overall

This plan keeps every stage tied to a single research question:

1. **Reduce** the graph so only the hard kernel remains.
2. **Summarize** that kernel once with MWUA.
3. **Measure** how much that static summary helps exact search over time (depth, pruning, anytime behavior).
4. **Learn** a branching policy that combines the frozen global summary with inexpensive dynamic local information.

The progression is coherent, each phase builds on the previous one, and it remains faithful to the dissertation proposal while incorporating your supervisor's recent guidance on reductions, search-space analysis, and anytime performance.
