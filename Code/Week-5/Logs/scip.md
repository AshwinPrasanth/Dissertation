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
