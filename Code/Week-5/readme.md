# materials

1. https://github.com/microsoft/MILP-Evolve
2. https://researchrepository.ucd.ie/server/api/core/bitstreams/527ea111-fa4a-4cdf-bb50-5a1e1586a863/content
3. https://proceedings.mlr.press/v176/gasse22a/gasse22a.pdf
4. https://openreview.net/pdf?id=NdcQQ82mfy

## MWUA Oracle

The MWUA oracle is a lightweight LP surrogate that iteratively reweights constraints and constructs a greedy fractional solution.

Given constraint matrix $A$ and constraint weights $w$:

For iteration $t$:

### 1. Normalize Weights

$$
\hat{w} = \frac{w}{\sum_i w_i}
$$

---

### 2. Compute Variable Scores

$$
D = |A|^T \hat{w}
$$

---

### 3. Greedy Fractional Oracle

$$
x_t = \texttt{GreedySolve}(D)
$$

where variables are processed in descending score order and fractional mass is assigned until the required coverage budget is reached.

---

### 4. Update Running Average

$$
x_{\text{avg}}
\leftarrow
x_{\text{avg}}
+
\frac{1}{t}
\left(
x_t - x_{\text{avg}}
\right)
$$

---

### 5. Compute Constraint Violations

$$
v_i =
\begin{cases}
\max(0,\,1-\text{cover}_i) & \text{MVC} \\
\max(0,\,\text{cover}_i-1) & \text{MIS}
\end{cases}
$$

---

### 6. Multiplicative Weight Update

$$
w_i \leftarrow w_i(1+\epsilon v_i)
$$

---

Repeat until convergence or the iteration budget is exhausted.

## Output Features

- $x_{\text{avg}}$

- Certainty:

$$
|x_{\text{avg}} - 0.5|
$$

- Constraint weight statistics:
  - final weight
  - minimum weight
  - maximum weight
  - average weight

---

## Current Setting

For the current experiments (graphs up to roughly $10^4$ vertices), MWUA is run for **100 iterations** since each iteration takes only a few milliseconds.

For larger-scale experiments (million-node graphs), the plan is to switch to a fixed time budget following Ryan's implementation:

- MIS: 90 seconds  
- VCP3 / VC: 60 seconds  

This keeps feature extraction computationally bounded while allowing the number of MWUA iterations to scale naturally with instance size.
