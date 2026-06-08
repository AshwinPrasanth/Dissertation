# Week 4.2: Dissertation Progress (Phase 1 Prototype)

## Research Question

**Can MWUA-derived structural features replace LP-derived features for guiding exact Branch-and-Bound search?**

---

## Optimization Problem

### Minimum Vertex Cover (MVC)

Objective:

$$
\min \sum_{v \in V} x_v
$$

Subject to:

$$
x_u + x_v \ge 1
\quad \forall (u,v)\in E
$$

where:

- $x_v = 1$ if vertex $v$ is selected
- $x_v = 0$ otherwise

**Goal:** Find the smallest set of vertices that covers every edge.

---

## Implementation Completed

### `problem.py`

- Binary MILP formulation
- MVC conversion

### `lp.py`

- LP relaxation solver
- SciPy `linprog`
- HiGHS backend

### `branching.py`

- Most-Fractional LP branching
- MWUA-guided branching

### `solver.py`

- Exact DFS Branch-and-Bound
- LP-bound pruning
- Optimality guarantees

### `mwua.py`

- Root-level MWUA structural feature extraction

---

## MWUA Prototype

Constraint weights initialized as:

$$
w_i = 1
$$

Oracle:

$$
\text{pressure} = (-A)^T w
$$

Approximate assignment:

$$
x = \frac{\text{pressure}}{\max(\text{pressure})}
$$

Weight update:

$$
w_i \leftarrow w_i e^{\eta \cdot \text{violation}_i}
$$

Final certainty score:

$$
|x_{\text{avg}} - 0.5|
$$

---

## Experimental Setup

### Graph Family

Erdős–Rényi random graphs

$$
G(n,p)
$$

Tested:

- $n \in \{30, 50, 70\}$
- $p = 0.2$

Results averaged across multiple random seeds.

**Note:** No standard benchmark datasets have been evaluated yet.

---

## Preliminary Findings

- LP relaxations frequently produced all-0.5 solutions, providing little branching guidance.
- MWUA produced meaningful variable rankings.
- MWUA certainty was correlated with degree but not identical ($\rho \approx 0.65$).
- MWUA branching reduced search effort while preserving optimality.
- For very sparse graphs ($p = 0.05$), both branching strategies perform almost identically because the search tree is already small.
- As graph density increases, LP relaxations become less informative and the search space grows substantially.
- MWUA consistently provides stronger branching guidance, reducing the number of explored nodes.

### Same Edges (n = 30), Different Graph Density

To evaluate how graph density affects branching effectiveness, we fixed the graph size at $n = 30$ and varied the edge probability $p$ in Erdős–Rényi graphs.

| p | LP Avg Nodes | MWUA Avg Nodes | Improvement |
|---|-------------:|---------------:|------------:|
| 0.05 | 1.4 | 1.4 | 0% |
| 0.10 | 4.5 | 3.6 | ~20% |
| 0.20 | 51.8 | 31.9 | ~38% |
| 0.30 | 78.5 | 51.8 | ~34% |
| 0.40 | 97.7 | 56.2 | ~42% |

### Scaling Up edges with same Graph Density

| n | LP Nodes | MWUA Nodes | Reduction |
|---|---------:|-----------:|----------:|
| 30 | 51.8 | 31.9 | 38% |
| 50 | 583.6 | 267.9 | 54% |
| 70 | 4625.0 | 1983.9 | 57% |

---

## Current Status

**Preliminary evidence suggests that a single root-level MWUA structural snapshot can significantly reduce exact Branch-and-Bound search effort, with benefits increasing as graph size grows.**

---

## Key Takeaway

This prototype clearly demonstrates:

- What was implemented
- What was tested
- What has not yet been tested
- What the current evidence suggests

The results provide initial evidence that MWUA-derived structural features can act as an effective replacement for LP-derived branching guidance in exact Branch-and-Bound search for Minimum Vertex Cover.
