## 1. **MWU Capturing the Global Snapshot**

The MWU is computed **exactly once at the root** and reused throughout the entire tree:

```python
def solve(self, problem: MILPProblem) -> BBSolution:
    mwu_weights = self.scorer.compute(problem)  # ← Computed ONCE at root
    root = BBNode()
    # ... then mwu_weights is passed to every branching decision
```

The `MWUScorer.compute()` blends two signals 50-50:

1. **Objective Signal**: Normalized absolute coefficients from `c` (cost vector)
2. **Structural Signal**: Column sums of constraint matrices (how much each variable appears across all constraints)

Then it applies multiplicative weight updates over 4 iterations:
```python
weights = np.ones(n_vars, dtype=float)
for _ in range(self.iterations):
    weights *= np.exp(self.learning_rate * score)  # Learning rate = 0.35
    weights /= np.mean(weights)  # Keep normalized
```

This captures **global problem structure once**, avoiding expensive per-node computations.

---

## 2. **Node Selection Heuristic**

**Depth-First Search (DFS) using a LIFO stack.**

```python
while stack:
    node = stack.pop()  # ← LIFO (most recent branch)
    # ... solve LP, check integrality, branch ...
    stack.append(left_child)   # Push less-promising first
    stack.append(right_child)  # Push more-promising second
```

This ensures the solver explores one branch deep before backtracking, which:
- Finds an incumbent solution fast (good for pruning later nodes)
- Minimizes memory (stack only holds ancestors of current node)
- Biases toward feasible solutions early

---

## 3. **Variable Selection & Ordering Heuristic**

**"Farthest from 0.5" strategy with MWU tie-breaking.**

```python
def _select_branch_variable(self, x: np.ndarray, mwu_weights: np.ndarray) -> int | None:
    fractional = self._fractional_indices(x)  # Find all x_i ∈ (0,1)
    
    candidates = sorted(
        fractional.tolist(),
        key=lambda index: (
            -abs(x[index] - 0.5),      # PRIMARY: Maximize distance from 0.5
            -mwu_weights[index],        # SECONDARY: Highest MWU weight
            index,                       # TERTIARY: Lowest index
        ),
    )
    return int(candidates[0])
```

**Ranking order:**
1. **Primary**: Variables closest to 0 or 1 (farthest from 0.5) are chosen first
   - Rationale: High certainty reduces uncertainty immediately
2. **Secondary**: Among equally fractional variables, pick the one with highest MWU weight
   - Rationale: MWU captures structural importance; branch on "important" variables first
3. **Tertiary**: Lowest index (deterministic tie-break)

**Asymmetric branching preference:**
```python
def _create_children(self, node: BBNode, var_index: int, value: float):
    preferred = rounded_up if value >= 0.5 else rounded_down
    alternative = rounded_down if preferred == rounded_up else rounded_up
    
    return alternative_child, preferred_child  # Return (left, right)
    # Then push left first, right second → right popped first (DFS explores preferred first)
```

This means:
- If `x[i] = 0.8`, try fixing to 1 first (preferred), then 0
- If `x[i] = 0.3`, try fixing to 0 first (preferred), then 1
- The preferred branch is explored deeper in DFS, discovering feasible solutions faster

---

## Summary Table

| Aspect | Mechanism | Purpose |
|--------|-----------|---------|
| **Global Snapshot** | MWU score computed once at root, blending objective + structural signal | Avoid expensive per-node scoring; capture problem structure statically |
| **Node Selection** | LIFO stack (DFS) | Fast incumbent discovery + low memory |
| **Variable Selection** | Farthest from 0.5, then highest MWU | Reduce fractional uncertainty; prioritize structurally important variables |
| **Variable Ordering** | Asymmetric branching (prefer natural rounding) | Deeper exploration of more likely branch |

This design avoids expensive heuristics like Strong Branching while still steering the search toward good solutions early.
