## Here, I keep track of all the insights and next step brainstorming ideas for Week_4-2

### Insight 1

* Current MWU oracle is a weighted degree accumulation and hence it doesnt outperform the nodes explored, when tested agains the Degree based approach.
* variable pressure = sum of incident edge weights for MVC. MWU computes the weighted degree and not the global features.
* Hence the current shift happen towards reworking on the oracle to the greedy based strategy followed in the CPAIOR paper.

> What the MWU in CPAIOR looks like:
> The paper's MWUA is fundamentally an LP surrogate. The oracle is not: pressure = A.T @ w.
> The actual oracle is: Given current constraint weights, find a greedy solution that minimizes weighted violation.
> A greedy oracle would produce: MWUA solution that depends on- edge interactions, constraint history, repeated violations, global search dynamics; which is much closer to LP behavior.

**Instead of solving the LP directly:**

$$ constraint weights -> oracle -> update weights -> oracle -> update weights $$

> After many rounds: average primal solution approximates the LP solution.

**For a vertex cover:**
The oracle is typically: Given current constraint weights, find a greedy solution that minimizes weighted violation.

> For Vertex Cover this becomes something like: edge weights -> (pick vertices that cover, high-weight edges)

**Algorithm**

Inside each MWUA iteration:

1. Compute weighted edge importance.
2. Greedily choose vertices covering highest-weight edges.
3. Produce:

   $x_t$

   Update violated constraints.

   Average:

   $x_{\text{avg}}$

   across rounds.
