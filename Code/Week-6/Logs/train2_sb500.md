# Progress: 20 July 

# Learning-to-Branch Dataset Generation Pipeline

## Overview

This directory contains the generated training data for the proposed Learning-to-Branch framework.

The objective is to learn a branching policy for Maximum Independent Set (MIS) using strong branching decisions as supervision.

Instead of directly solving the MILP during inference using expensive strong branching, the proposed approach learns a ranking function that approximates the ordering produced by strong branching.

The pipeline consists of:

```
DIMACS Graph Instances
          |
          |
     Graph Conversion
          |
          |
   MIS Transformation
          |
          |
  KaMIS/ReduMIS Kernelization
          |
          |
  Root Static Feature Extraction
          |
          |
  SCIP Branch-and-Bound
          |
          |
 Strong Branching Data Collection
          |
          |
 Ranking Dataset
          |
          |
 Learning-to-Branch Model
```

---

# 1. Problem Formulation

The dataset is generated for the Maximum Independent Set problem.

Given a graph:

[
G=(V,E)
]

the objective is:

[
\max \sum_{v\in V} x_v
]

subject to:

[
x_u+x_v \leq 1
]

for every edge:

[
(u,v)\in E
]

where:

[
x_v\in{0,1}
]

---

# 2. Source Dataset

The initial instances are taken from the DIMACS maximum clique benchmark.

Since maximum clique and maximum independent set are complementary problems:

[
MIS(G)=MC(\bar{G})
]

the input graphs are transformed:

```
DIMACS Clique graph
          |
          |
       complement
          |
          |
       MIS graph
```

Example:

```
C125-9
```

represents a clique instance.

After complementation:

```
MIS graph
n = 125
m = 787
```

---

# 3. Graph Loading

Input format:

```
.mtx
```

The parser:

* ignores comments
* reads graph dimensions
* creates NetworkX graph
* removes self loops

Example:

```
C125-9.mtx

125 vertices
787 edges
```

---

# 4. Graph Kernelization

Before training data generation, exact preprocessing using KaMIS ReduMIS is applied.

Pipeline:

```
Original MIS graph

       |
       |
 ReduMIS Kernelization

       |
       |

Reduced graph/core
```

The reductions are exact and preserve the optimum solution.

Tracked statistics:

```
original_n
original_m

core_n
core_m

reduction_ratio
```

Example:

```
MANN-a27

Original:
378 nodes

Kernel:
324 nodes

Reduction:
14.3%
```

---

# 5. Static Feature Extraction

Static features are computed once at the root node.

These features remain unchanged during branch-and-bound.

Current static features:

## MWUA Features

Generated using the Maximum Weighted Upper Approximation procedure.

Features:

```
mwua_xavg

mwua_weight_min

mwua_weight_max

mwua_weight_avg
```

These represent global graph information and solution tendency.

---

## Centrality Features

PageRank:

```
pagerank
```

captures graph importance.

---

## Luby Feature

Feature:

```
luby_frequency
```

captures randomized branching relevance.

---

Current static feature dimension:

```
6 features
```

---

# 6. Strong Branching Data Collection

SCIP is used as the data generation engine.

The solver configuration:

```
Presolve: OFF
Heuristics: OFF
Separating: OFF
```

Reason:

The objective is to collect pure branching decisions rather than exploit SCIP solver components.

The branch rule intercepts SCIP branching decisions.

At every node:

1. SCIP identifies candidate variables.

2. Strong branching evaluates each candidate.

3. The resulting scores are stored.

Example:

```
Node 50

Candidates:

x12   SB score = 0.92
x34   SB score = 0.74
x91   SB score = 0.21

Ranking:

x12 > x34 > x91
```

---

# 7. SB Collection Limit

For scalability:

```
MAX_SB_NODES = 500
```

Meaning:

Each graph contributes at most:

```
500 branching states
```

Example:

```
C125-9

SB nodes:
500

Branching states:
499
```

---

# 8. Candidate Data

Each branching state contains:

```
candidate variables
candidate features
strong branching scores
```

The data is naturally a ranking problem.

Example:

```
State 105

Candidate    Score

v1            0.91
v2            0.62
v3            0.44
v4            0.11
```

The learning objective is:

```
rank v1 > v2 > v3 > v4
```

---

# 9. Dataset Statistics

Generated dataset:

```
dimacs_ltb_training500
```

Each completed graph contains:

```
SB nodes:
500

Branch decisions:
499

Candidates:
10k-80k
```

Examples:

## C125-9

```
nodes:
125

edges:
787

SB states:
500

candidates:
32588
```

---

## brock200-1

```
nodes:
200

edges:
5066

SB states:
500

candidates:
22863
```

---

## san400-0-9-1

```
nodes:
400

edges:
7980

SB states:
500

candidates:
79509
```

---

# 10. Dataset Exclusions

Some graphs solve immediately.

Example:

```
hamming6-2

SCIP nodes:
1

SB nodes:
0
```

These graphs do not provide branching supervision.

They are excluded from ranking training.

Condition:

```
collection_complete=True
```

is preferred.

---

# 11. Why Ranking Instead of Classification

Initial approach:

```
candidate -> best/not best
```

Problem:

Only one positive example exists.

Example:

```
Candidate A : 1
Candidate B : 0
Candidate C : 0
Candidate D : 0
```

This loses information.

New approach:

Pairwise ranking:

```
A > B > C > D
```

Advantages:

* preserves SB ordering
* matches Khalil et al.
* avoids class imbalance
* scales across graphs

---

# 12. Planned Learning Model

The next stage uses:

## XGBoost Learning-to-Rank

Objective:

```
rank:pairwise
```

Input:

```
candidate features
```

Group:

```
branching node
```

Target:

```
strong branching score ordering
```

Output:

```
ranking score for each candidate
```

---

# 13. Evaluation

The learned model will be integrated into SCIP.

Comparison:

| Method           | Description             |
| ---------------- | ----------------------- |
| SCIP default     | Built-in branching      |
| Strong branching | Oracle                  |
| Previous MLP     | Classification approach |
| XGBoost Ranker   | Proposed approach       |

Metrics:

Solver:

```
solve time
nodes explored
optimality gap
```

Learning:

```
Top-1 accuracy
Hit@k
MRR
NDCG
Kendall tau
Spearman correlation
```

---

# Current Status

Completed:

✅ DIMACS preprocessing
✅ MIS conversion
✅ ReduMIS kernelization
✅ MWUA feature extraction
✅ SCIP integration
✅ Strong branching collection
✅ SB500 dataset generation

Next:

⬜ Convert SB datasets into ranking format
⬜ Train XGBoost Ranker
⬜ Integrate ranking model into SCIP
⬜ Compare against SCIP default

---

This README captures the current methodology and will also directly help when writing the dissertation methodology/experiments chapter. The next thing after this should be the `ranking_dataset.py` conversion because the SB500 format is already correct.
