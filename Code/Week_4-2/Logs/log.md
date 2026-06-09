# LOG 1: CPAIOR MWUA Reproduction & MIS Extension: 9-6-26

### Motivation

* Initial MWUA implementation behaved similarly to a weighted-degree heuristic.
* Degree branching consistently outperformed MWUA branching.
* Investigated original CPAIOR implementation to reproduce the intended MWUA feature-generation process.

---

### Code Updates

**MWUA Reimplementation**

* Normalized constraint weighting.
* CPAIOR greedy fractional oracle.
* Constraint-violation weight updates.
* Running fractional averages.
* Incident constraint-weight statistics (min/max/avg).

**MIS Integration**

* Added Maximum Independent Set formulation.
* Added MIS LP relaxation support.
* Added MIS-compatible MWUA implementation.
* Added MIS-specific violation updates.

---

### MVC Re-evaluation (n=50, p=0.2)

| Strategy    | Avg Nodes |
| ----------- | --------: |
| Degree      |     238.2 |
| LP          |     583.6 |
| CPAIOR MWUA |    1551.1 |

**Observation**

* CPAIOR MWUA rankings differed significantly from degree rankings.
* Direct MWUA branching became worse than Degree and LP branching.
* Indicates MWUA is not a standalone branching heuristic.

---

### MIS Validation (n=50, p=0.2)

| Metric                | LP | MWUA |
| --------------------- | -: | ---: |
| Unique Feature Values |  1 |   21 |

**LP Relaxation**

```text
Half-integral vertices: 50 / 50
Fraction = 1.0
```

**Observation**

* LP collapses to all 0.5 assignments.
* MWUA produces a rich spectrum of feature values.
* Successfully reproduces the main motivation described in the CPAIOR paper.

---

### Direct Branching on MIS

| Strategy    | Avg Nodes |
| ----------- | --------: |
| Degree      |     211.4 |
| LP          |     953.2 |
| CPAIOR MWUA |    2193.1 |

**Finding**

* MWUA features are informative.
* MWUA certainty is not an effective direct branching signal.
* Consistent with CPAIOR, where MWUA is used as an ML feature rather than a branching heuristic.

---

### Additional Graph Families Added

* Erdős–Rényi (ER)
* Barabási–Albert (BA)
* Watts–Strogatz (WS)

Purpose:

* Evaluate feature behavior across diverse graph structures instead of relying solely on ER random graphs.

---

### Key Takeaway

**Reproduced the central CPAIOR observation:**

> LP features collapse on MIS, while MWUA generates a rich and informative feature spectrum.

**Next Phase:** Feature extraction pipeline (MWUA + structural graph features) and feature-importance analysis using machine learning.
