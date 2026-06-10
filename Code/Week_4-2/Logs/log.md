# LOG 1: CPAIOR MWUA Reproduction & MIS Extension: 8-6-26

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


# LOG 2: Feature Extraction Pipeline: 9-6-26

Implemented a CPAIOR-inspired feature extraction pipeline consisting of:

- Degree-rank features (4)
- Centrality features (4)
- MWUA features (4)
- LP features (2)
- Luby frequency feature (1)

**Total:** 15 vertex-level features

---

## Initial Observations

Experiments were conducted on Erdős–Rényi Maximum Independent Set (MIS) instances with:

- $n = 50$
- $p = 0.2$

### Results

- LP relaxation features collapsed to a single value ($x = 0.5$ for all vertices).
- LP certainty was identically zero.
- MWUA features produced approximately 20–22 distinct values across vertices.
- Luby frequency produced approximately 24 distinct values across vertices.

### Interpretation

These preliminary results indicate that LP-derived features provide little discriminative information for the tested MIS instances, while MWUA and heuristic-based features generate richer vertex rankings that may be more useful for guiding exact search.



# LOG 3: Testing on wider graph families for code reliability [9-6-26]

## Objective

Before generating the training dataset, the feature extraction pipeline was benchmarked on multiple graph families to verify that the extracted features provide meaningful and diverse signals.

The goal was to evaluate feature variability and discriminative power rather than prediction performance.

---

## Graph Families

The benchmark was performed on three synthetic graph families:

| Graph Family | Parameters |
|-------------|------------|
| Erdős–Rényi (ER) | n = 100, p = 0.2 |
| Barabási–Albert (BA) | n = 100, m = 3 |
| Watts–Strogatz (WS) | n = 100, k = 6, β = 0.1 |


---

## Key Findings

### LP Features Frequently Collapse

LP-derived features provided very limited discrimination.

| Graph | LP Unique Values |
|---------|---------:|
| ER | 1 |
| BA | 2 |
| WS | 1 |

- ER and WS LP relaxations assigned every vertex a value of 0.5.
- LP certainty collapsed to zero.
- BA produced only binary LP values (0 or 1).
- LP features contained little structural information overall.

---

### MWUA Features Remain Informative

MWUA-derived features retained substantial variability across all graph families.

| Graph | Unique MWUA x_avg Values |
|---------|---------:|
| ER | 43 |
| BA | 22 |
| WS | 48 |

- MWUA features remained highly discriminative even when LP collapsed.
- Weight-based features captured varying levels of constraint pressure.
- MWUA signals consistently contained more information than LP signals.

---

### Luby Frequency Provides Strong Signal

| Graph | Unique Frequency Values |
|---------|---------:|
| ER | 31 |
| BA | 59 |
| WS | 24 |

- Luby frequency exhibited strong variability across all graph families.
- The feature captures how consistently a vertex appears in maximal independent sets.
- Provides a useful heuristic signal independent of LP and MWUA.

---

### Structural Features

> Degree and centrality features remained informative across most graph families.
> Particularly useful features included: Degree Rank, Neighbor Average Rank, PageRank
> Core Number was informative on ER graphs but collapsed on BA and WS graphs.

---

## Preliminary Conclusion

The benchmark suggests that:

1. LP-derived features frequently collapse and provide limited information.
2. MWUA-derived features consistently retain rich and diverse signals.
3. Luby frequency provides a strong complementary heuristic signal.
4. Degree and PageRank features remain informative across multiple graph structures.

These results provide preliminary evidence that MWUA and structural features may offer more useful learning signals than LP-derived features for branching and optimization tasks.

---
