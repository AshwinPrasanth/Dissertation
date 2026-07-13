Yes. I would describe **our setup** like this in the README:

# Learning to Branch with Static Global and Dynamic Local Features

## Overview

This project studies Learning to Branch for exact Maximum Independent Set and Minimum Vertex Cover solving with SCIP.

The central research question is:

> **Can expensive global information computed once at the root be combined with cheap local information updated during Branch-and-Bound to predict strong branching decisions?**

Strong Branching is used as the expert for generating training supervision. The final learned brancher is intended to replace expensive SB during inference.

Our core hypothesis is that **root-level global information has a finite useful lifetime**. It may be highly predictive near the root, but branching changes the residual problem. As the search progresses, cheap local information from the current state may become increasingly important.

---

## Optimization Problem

We solve Maximum Independent Set through the equivalent Minimum Vertex Cover formulation:

[
\min \sum_{v\in V}x_v
]

subject to

[
x_u+x_v\geq1
\qquad \forall (u,v)\in E
]

and

[
x_v\in{0,1}.
]

A vertex with (x_v=1) belongs to the vertex cover.

The corresponding independent set is:

[
I=V\setminus C.
]

---

## Overall Pipeline

```text
Generate graph instance
        ↓
Build MVC model in SCIP
        ↓
Compute expensive static features once
        ↓
Start Branch-and-Bound
        ↓
At selected B&B nodes:
    obtain fractional LP candidates
        ↓
    construct current residual graph
        ↓
    compute cheap dynamic candidate features
        ↓
    run limited Strong Branching
        ↓
    save candidate features and SB scores
        ↓
Build pairwise ranking supervision
        ↓
Train ranking model
        ↓
Evaluate static / dynamic / hybrid features
        ↓
Study predictive behaviour across B&B depth
        ↓
Design adaptive branching policy
        ↓
Integrate learned scorer into SCIP
        ↓
Evaluate complete solver performance
```

---

## Synthetic Training Instances

The controlled experiment uses four graph families:

* Erdős-Rényi
* Barabási-Albert
* Watts-Strogatz
* Random Regular

The configuration is:

```python
GRAPH_SIZES = [
    50,
    100,
]

TARGET_DEGREES = [
    5,
    10,
    20,
]

SEEDS = [
    42,
    43,
    44,
    45,
    46,
]
```

The total number of graph instances is:

[
4\times2\times3\times5
======================

120.

]

Each graph family contributes 30 instances.

The controlled grid varies:

```text
graph family
graph size
target degree
random realization
```

---

## Graph-Level Data Split

The graph instances are split by random seed:

```text
Training seeds     : 42, 43, 44
Validation seed    : 45
Test seed          : 46
```

This gives:

```text
72 training graphs
24 validation graphs
24 test graphs
```

The split occurs at the **graph level**.

Rows from a test graph never enter the training set.

The current experiment therefore measures generalization to **unseen graph realizations from the same controlled graph distributions**.

Unseen-family and real-graph transfer are separate later experiments.

---

## Strong Branching Data Collection

Strong Branching acts as the expert label generator.

The current configuration is:

```python
MAX_SB_NODES = 20
CANDIDATE_LIMIT = None
STRONGBRANCH_ITLIM = 50
```

### SB node budget

At most 20 successfully labelled branching states are collected from each graph.

A labelled state is a B&B node where candidate features and valid Strong Branching scores are successfully stored.

### Candidate set

```python
CANDIDATE_LIMIT = None
```

means that all available SCIP LP branching candidates at the selected node are evaluated by Strong Branching.

This is deliberate.

Introducing a candidate pre-filter based on pseudocost, degree, LP certainty, or MWUA could bias the expert candidate pool toward one of the signals being studied.

### Limited Strong Branching

Each SB probe is limited to 50 LP iterations:

```python
STRONGBRANCH_ITLIM = 50
```

This bounds expert-evaluation cost.

### Collection termination

The training-data generator is a **label collector**, not a solver benchmark.

The intended behaviour is:

```text
collect SB sample 1
collect SB sample 2
...
collect SB sample 20
        ↓
save sample 20
        ↓
interrupt collection solve
        ↓
move to next graph
```

Continuing to solve the graph after the final stored sample produces no additional training supervision.

Full solves are reserved for final branching-policy evaluation.

---

## Residual Graph

Dynamic features are computed from the current residual graph.

The residual graph represents the part of the original graph that remains relevant under the current B&B variable bounds and fixings.

Conceptually:

```text
Original graph
        ↓
Branch on variable
        ↓
Variable bounds change
        ↓
Residual problem changes
        ↓
Local graph structure changes
        ↓
Dynamic candidate features change
```

The static features remain frozen.

The dynamic features change with the current search state.

This distinction is central to the dissertation.

---

## The 15 Candidate Features

Every branching candidate is represented using 15 features.

They are divided into **six static global features** and **nine dynamic/local features**.

### Static global features — 6

These are computed once and reused throughout search:

```text
1. PageRank
2. MWUA average fractional value
3. MWUA incident-weight minimum
4. MWUA incident-weight maximum
5. MWUA incident-weight average
6. Luby selection frequency
```

The current feature indices are:

```python
STATIC_INDICES = [
    4,
    8,
    9,
    10,
    11,
    14,
]
```

The expensive global computation is therefore amortized across the B&B tree.

### Dynamic/local features — 9

These describe the candidate in the current residual or LP state:

```text
1. Degree rank
2. Minimum neighbour degree rank
3. Maximum neighbour degree rank
4. Average neighbour degree rank
5. Core number
6. Clustering coefficient
7. Degree centrality
8. Current LP value
9. LP certainty
```

The current indices are:

```python
DYNAMIC_INDICES = [
    0,
    1,
    2,
    3,
    5,
    6,
    7,
    12,
    13,
]
```

LP certainty is:

[
|x_v-0.5|.
]

---

## MWUA Root Snapshot

MWUA is used to construct a global root-level representation.

The MWUA computation is performed **once before Branch-and-Bound begins**.

For every vertex, the snapshot provides signals including:

```text
average approximate fractional value
minimum incident edge weight
maximum incident edge weight
average incident edge weight
```

These values are frozen.

We do **not** recompute MWUA at every B&B node.

This is intentional.

One of the main research questions is:

> **How long does the root MWUA snapshot remain useful as the B&B state evolves?**

---

## Raw Branching Samples

At every successfully collected SB node, the collector stores a `BranchSample`.

Conceptually, a sample contains:

```text
graph identity
graph family
B&B node number
B&B depth
candidate variable IDs
15-dimensional candidate feature vectors
Strong Branching scores
selected SB candidate
search-state metadata
```

A graph `.pkl` file can therefore contain multiple sequentially stored B&B states.

---

## Pairwise Ranking Labels

The learning problem is formulated as pairwise ranking.

At one B&B node, let:

[
s^*=\max_j s_j
]

be the maximum Strong Branching score.

We currently use:

[
\alpha=0.2.
]

A candidate is classified as GOOD when:

[
s_j\geq(1-\alpha)s^*.
]

All remaining candidates are BAD.

For example:

```text
Candidate A    12.0
Candidate B    10.5
Candidate C     9.8
Candidate D     4.0
Candidate E     2.5
```

The threshold is:

[
0.8\times12=9.6.
]

Therefore:

```text
GOOD
A
B
C

BAD
D
E
```

Preference pairs are created:

```text
A > D
A > E
B > D
B > E
C > D
C > E
```

The number of pairs is capped per B&B node:

```python
MAX_PAIRS_PER_NODE = 200
```

Flat SB nodes are skipped.

Nodes without a valid GOOD/BAD split are also skipped.

---

## Pairwise Feature Representation

For a preferred candidate (i) and a worse candidate (j), the current ablation uses:

[
\Delta x_{ij}=x_i-x_j.
]

The positive example is:

[
x_i-x_j,\qquad y=1.
]

The reversed example is:

[
x_j-x_i,\qquad y=0.
]

This creates an antisymmetric ranking representation.

The model dimensions are therefore:

```text
Static model      6
Dynamic model     9
Hybrid model      15
```

---

## Current Ranking Model

The current ranking model is deliberately simple:

```text
StandardScaler
        ↓
LogisticRegression
```

The immediate goal is **not to maximize prediction accuracy using a complex learner**.

The goal is to understand the information carried by the static and dynamic feature groups.

A linear model makes the feature ablation easier to interpret.

---

## Static, Dynamic, and Hybrid Ablation

Three models are evaluated.

### Static only

Uses the six frozen global features.

Research question:

> How long does root-level global information remain predictive of SB preference?

### Dynamic only

Uses the nine current-state features.

Research question:

> Can cheap local information capture branching relevance as the residual problem evolves?

### Hybrid

Uses all 15 features.

Research question:

> Does frozen global context combined with cheap local refinement improve branching prediction?

---

## Depth-Wise Evaluation

Predictions are grouped by B&B depth:

```text
depth 0
depth 1
depth 2
depth 3
depth 4
depth 5+
```

Aggregate accuracy alone can hide changes in feature relevance during search.

Our initial four-graph pilot produced:

| Depth | Static | Dynamic | Hybrid |
| ----- | -----: | ------: | -----: |
| 0     | 0.9497 |  0.7757 | 0.8604 |
| 1     | 0.7868 |  0.8583 | 0.8974 |
| 2     | 0.6606 |  0.9132 | 0.9033 |
| 3     | 0.4529 |  0.7649 | 0.7338 |
| 4     | 0.4684 |  0.7161 | 0.6817 |
| 5+    | 0.4767 |  0.5075 | 0.5409 |

Overall pilot accuracy was:

```text
Static only     : 0.5778
Dynamic only    : 0.7343
Hybrid          : 0.7393
```

These results come from only four graph instances and are **pilot observations, not final dissertation conclusions**.

However, the pattern motivates the expanded 120-instance experiment:

```text
Root
→ static information is extremely strong

Early search
→ hybrid and local information become important

Intermediate search
→ dynamic information dominates

Deep search
→ all current linear signals weaken
```

---

## Depth-Adaptive Branching Hypothesis

A fixed hybrid model assumes that static and dynamic information should have the same role throughout the entire search.

Our pilot suggests that this may be incorrect.

A natural adaptive score is:

[
S(v,d)
======

\lambda(d)S_{\text{static}}(v)
+
[1-\lambda(d)]S_{\text{dynamic}}(v).
]

Here:

```text
v                branching candidate
d                B&B depth
S_static         static ranking score
S_dynamic        dynamic ranking score
lambda(d)        static-signal contribution
```

The hypothesis is:

[
\lambda(d)\downarrow
]

as search depth increases.

Conceptually:

```text
depth 0       mostly static
depth 1       hybrid
depth 2-4     mostly dynamic
depth 5+      weak-signal / adaptive fallback regime
```

The adaptive policy must be designed using validation data only.

The seed-46 test set remains untouched until the policy is frozen.

---

## Final SCIP Evaluation

Ranking accuracy is **not the final objective**.

The learned scorer must be integrated into SCIP and evaluated end to end.

The planned branching comparison is:

```text
SCIP default branching
Strong Branching
Most Fractional / LP baseline
MWUA-only branching
Static learned brancher
Dynamic learned brancher
Naive hybrid learned brancher
Depth-adaptive hybrid brancher
```

The primary solver metrics are:

```text
B&B nodes
total solve time
solved instances under a fixed time limit
primal/dual gap for unsolved instances
branching-decision overhead
```

Strong Branching is used as the expensive training expert.

The learned brancher must **not call Strong Branching during inference**.

---

## External Graph Evaluation

The synthetic experiment is a controlled representation study.

Later experiments will use real undirected graphs and irreducible cores.

The intended pipeline is:

```text
Real graph
        ↓
ReduMIS kernelization
        ↓
Irreducible core
        ↓
MVC/MIS SCIP model
        ↓
Learned branching policy
```

Kernelization allows the ML component to focus on the difficult residual problem rather than relearning established reduction rules.

Potential external evaluation sources include DIMACS10-style sparse undirected graphs and suitable real graph instances.

---

## Separation of Experimental Tasks

The project separates three different tasks.

### 1. Training-data generation

Purpose:

```text
collect expert SB supervision
```

The collection solve may stop after the SB sample budget is complete.

### 2. Ranking and representation evaluation

Purpose:

```text
measure what static and dynamic features predict
```

This uses graph-level train, validation, and test separation together with depth-wise analysis.

### 3. Solver evaluation

Purpose:

```text
measure whether the learned branching policy improves exact solving
```

These runs continue to termination or a fixed time limit and report solver-level metrics.

The three tasks should not be mixed.

---

## Current Experimental Sequence

```text
1. Generate 120 synthetic graph instances

2. Collect up to 20 limited-SB states per graph

3. Audit the raw data
   - failed instances
   - SB nodes collected
   - candidate counts
   - flat nodes
   - depth distribution
   - collection time

4. Split graphs by seed
   train      = 42, 43, 44
   validation = 45
   test       = 46

5. Build pairwise ranking supervision

6. Train static-only model

7. Train dynamic-only model

8. Train hybrid model

9. Evaluate all models by B&B depth

10. Design a depth-adaptive policy using validation data

11. Freeze the adaptive policy

12. Evaluate on untouched test graphs

13. Integrate learned scoring into SCIP

14. Run complete solver comparisons

15. Evaluate transfer to reduced real graph cores
```

---

## Core Dissertation Idea

The project is **not simply asking whether machine learning can imitate Strong Branching**.

The central research problem is the **temporal usefulness of optimization features during Branch-and-Bound**.

The dissertation investigates whether:

> **A global graph representation can be computed once, reused while it remains informative, and progressively complemented or replaced by cheap local state information as the search evolves.**

In compact form:

[
\boxed{
\text{Frozen global context}
+
\text{cheap local refinement}
\rightarrow
\text{fast adaptive branching}
}
]

The key question is not only:

> Which feature set is best?

It is:

[
\boxed{
\text{When is each source of information useful during search?}
}
]

This is the README I would use for **our exact current setup**. I tried to generate it as a downloadable `.md`, but the file runtime failed twice, so I don't want to give you a fake sandbox link. The text above is complete and can be saved directly as `ASHWIN/README.md`.
