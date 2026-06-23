# brock200_4: MWUA-Guided Branching Analysis

## Instance

| Property |      Value |
| -------- | ---------: |
| Graph    | brock200_4 |
| Vertices |        200 |
| Edges    |     13,089 |
| MIS Size |          8 |

Solver configuration:

```text
Presolve     OFF
Heuristics   OFF
Separating   OFF
```

The objective is to isolate the effect of the branching rule on branch-and-bound search.

---

## Results

| Method       |  Nodes | Runtime (s) | MIS | Node Reduction |
| ------------ | -----: | ----------: | --: | -------------: |
| Default SCIP | 16,859 |       19.87 |   8 |           0.0% |
| MWUA D=0     | 17,009 |       22.02 |   8 |          -0.9% |
| MWUA D=1     | 16,405 |       21.74 |   8 |           2.7% |
| MWUA D=2     | 17,614 |       22.60 |   8 |          -4.5% |

All methods reached the optimal MIS value of 8.

---

## Key Observation

The MWUA-guided branching policy has a smaller effect on brock200_4 than on brock200_2.

The best configuration (D=1) reduces the search tree from:

```text
16,859 → 16,405
```

corresponding to approximately:

```text
2.7%
```

fewer branch-and-bound nodes.

Unlike brock200_2, the improvement is modest and the search appears relatively insensitive to the MWUA signal.

---

## Figure 1: Search Tree Size



---

## Figure 2: Relative Improvement



---

## Figure 3: Runtime Comparison



---

## Figure 4: Node Reduction Profile


---

## Conclusion

For brock200_4, MWUA-derived branching produces only a modest improvement. The best configuration (D=1) reduces the search tree by approximately 2.7%, while stronger MWUA influence (D=2) increases the number of explored nodes. Compared with brock200_2, this suggests that the effectiveness of root-level MWUA information depends on graph structure and problem instance characteristics.

Overall, the result is positive but considerably weaker than the improvements observed on brock200_2 and the larger sparse network instances.


# brock200_2: MWUA-Guided Branching Analysis

## Instance

| Property |      Value |
| -------- | ---------: |
| Graph    | brock200_2 |
| Vertices |        200 |
| Edges    |      9,876 |
| MIS Size |         11 |

Solver configuration:

```text
Presolve     OFF
Heuristics   OFF
Separating   OFF
```

The objective is to isolate the effect of the branching rule on branch-and-bound search.

---

## Results

| Method       |   Nodes | Runtime (s) | MIS | Node Reduction |
| ------------ | ------: | ----------: | --: | -------------: |
| Default SCIP | 109,174 |       85.60 |  11 |           0.0% |
| MWUA D=0     |  95,812 |       92.02 |  11 |          12.2% |
| MWUA D=1     |  98,350 |       88.87 |  11 |           9.9% |
| MWUA D=2     | 103,349 |       90.57 |  11 |           5.3% |
| MWUA D=10    | 113,584 |       99.70 |  11 |          -4.0% |

All methods reached the optimal MIS value of 11.

---

## Key Observation

The MWUA-guided branching strategy changes the search tree substantially.

The strongest configuration (D=0) reduces the number of explored branch-and-bound nodes from:

```text
109,174 → 95,812
```

corresponding to a reduction of approximately:

```text
12.2%
```

while preserving optimality.

Interestingly, increasing D beyond a small value degrades performance:

```text
D=0  → best
D=1  → good
D=2  → weaker
D=10 → worse than default
```

This suggests that a moderate amount of MWUA guidance is beneficial, whereas excessive influence may distort branching decisions.

---

## Figure 1: Branch-and-Bound Nodes



---

## Figure 2: Relative Improvement


---

## Figure 3: Runtime Comparison



---

## Conclusion

For brock200_2, MWUA-derived root-level structural information consistently influences branch-and-bound behavior. The best configuration reduces the search tree by over 12% while preserving optimality. However, the results also indicate that stronger MWUA influence is not necessarily beneficial, highlighting the importance of selecting an appropriate scoring formulation.


# C250.9: MWUA-Guided Branching Analysis

## Instance

| Property |  Value |
| -------- | -----: |
| Graph    | C250.9 |
| Vertices |    250 |
| Edges    | 27,984 |
| MIS Size |      5 |

Solver configuration:

```text
Presolve     OFF
Heuristics   OFF
Separating   OFF
```

The objective is to evaluate the impact of MWUA-derived branching priorities on branch-and-bound search.

---

## Results

| Method       | Nodes | Runtime (s) | MIS | Node Reduction |
| ------------ | ----: | ----------: | --: | -------------: |
| Default SCIP | 2,623 |       15.31 |   5 |           0.0% |
| MWUA D=0     | 2,411 |       18.70 |   5 |           8.1% |
| MWUA D=1     | 2,555 |       19.16 |   5 |           2.6% |

All methods reached the optimal MIS value of 5.

---

## Key Observation

The MWUA-guided branching policy consistently improves search tree size on this instance.

The strongest configuration (D=0) reduces the number of explored branch-and-bound nodes from:

```text
2623 → 2411
```

corresponding to approximately:

```text
8.1%
```

fewer search nodes.

Although runtime increases slightly, the reduction in tree size indicates that the MWUA signal is influencing branching decisions in a beneficial way.

---

## Figure 1: Search Tree Size


---

## Figure 2: Relative Improvement



---

## Figure 3: Runtime Comparison



---

## Figure 4: Search Efficiency



---

## Conclusion

C250.9 demonstrates a clear positive effect from MWUA-guided branching. The best configuration (D=0) reduces branch-and-bound node count by approximately 8.1% while preserving optimality. Compared to brock200_4, the effect is stronger, though still smaller than the gains observed on brock200_2.

Together with the previous DIMACS experiments, these results suggest that root-level MWUA structural information can consistently alter search behavior and often reduce the size of the branch-and-bound tree, even when computed only once at the root node.


# CA-GrQc: MWUA-Guided Branching Analysis

## Instance

| Property   |   Value |
| ---------- | ------: |
| Graph      | CA-GrQc |
| Vertices   |   5,242 |
| Edges      |  14,484 |
| Time Limit |   300 s |

Solver configuration:

```text
Presolve     OFF
Heuristics   OFF
Separating   OFF
```

Unlike the DIMACS experiments, this instance was not solved within the time limit. Therefore, search quality is evaluated using:

* Primal Bound (best MIS found)
* Dual Bound
* Optimality Gap
* Search Tree Size

---

## Results

| Method        |  Nodes | Primal |   Dual | Gap (%) |
| ------------- | -----: | -----: | -----: | ------: |
| Default SCIP  | 68,752 |   2412 | 2545.0 |    5.51 |
| D=0 Weighted  | 60,837 |   2409 | 2535.5 |    5.25 |
| D=1 Weighted  | 57,739 |   2415 | 2537.5 |    5.07 |
| D=1 Dist(0.5) | 53,027 |   2407 | 2537.5 |    5.42 |
| D=2 Weighted  | 62,329 |   2420 | 2542.0 |    5.04 |
| D=2 Dist(0.5) | 45,390 |   2410 | 2541.5 |    5.46 |

---

## Key Observation

This graph reveals two distinct behaviors.

### Weighted MWUA

The weighted score:

```text
score = α·xavg + β·weight
```

improves both search quality and tree size.

Best result:

```text
D=2 Weighted
```

achieves:

```text
Primal : 2412 → 2420
Gap    : 5.51% → 5.04%
```

while still reducing explored nodes.

---

### Distance-from-0.5

The certainty score:

```text
score = |xavg − 0.5|
```

produces the smallest search tree.

Best result:

```text
D=2 Dist(0.5)
```

reduces nodes from:

```text
68,752 → 45,390
```

corresponding to approximately:

```text
34.0%
```

fewer branch-and-bound nodes.

However, this reduction does not translate into improved primal bounds or gap quality.

---

## Figure 1: Search Tree Size (Zoomed)


---

## Figure 2: Node Reduction Relative to Default


---

## Figure 3: Optimality Gap (Zoomed)



---

## Figure 4: Best Independent Set Found




---

## Figure 5: Trade-off Between Tree Size and Gap


---

## Conclusion

CA-GrQc demonstrates the strongest MWUA effect observed so far.

Two different branching behaviors emerge:

1. **Weighted MWUA** improves search quality by finding larger independent sets and reducing the optimality gap.
2. **Distance-from-0.5 scoring** aggressively shrinks the search tree, reducing explored nodes by up to 34%, but provides limited improvement in solution quality.

These results suggest that MWUA-derived root-level structural information remains informative deep into the branch-and-bound process and can significantly alter search behavior on large sparse real-world graphs.
