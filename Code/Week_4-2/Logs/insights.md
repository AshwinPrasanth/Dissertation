## Here, I keep track of all the insights and next step brainstorming ideas for Week_4-2

### Insight 1 (8-6-26)

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


### Insight 2 (9-6-26)

***Feature Extraction Pipeline***

The `features.py` module was developed prior to `datasets.py` and extracts vertex-level features from five complementary perspectives. The goal is to capture local structure, global structure, optimization pressure, relaxation behavior and heuristic signals that may be useful for learning branching decisions.

---

**1. Neighborhood Features (Degree):** Looks at the immediate neighbors. It assumes that the important nodes are those with many connections. 

***Degree Rank:*** uses percentile rank than raw numbers so as to work with graphs of multiple sizes. 

***Neighbor Ranks (min, max, avg)*** are also calculated. Example: If a node has a low degree but its neighbors have a max_rank of 1.0, it means this node is connected to a "super-hub." Neighbor-rank statistics capture the structural context of a vertex. A low-degree vertex connected to high-rank neighbors plays a very different role from a low-degree vertex surrounded by other low-degree vertices.

---

**2. Global Structure Features (Centrality):** Looks at how the node sits in the entire "map". Extracted using networkx algorithm.

***PageRank*** Measures importance recursively: a vertex is considered important if it is connected to other important vertices.

***Core Number*** finds nodes that are part of a highly connected core (K-Core).  Nodes with high core lie deep in the densest part of the graph. locates nodes in the densest, most interconnected center of the graph.

***Clustering Coefficient*** detects if a node is part of a tight-knit group (clique) or a sparse bridge.  In a tight clique, we might only need to pick a few nodes to cover many edges. If clustering is low, you have to work harder to cover those "spread out" connections.

***Degree Centrality*** Provides a normalized measure of connectivity independent of graph size. Unlike raw degree, degree centrality allows comparisons across graphs with different numbers of vertices.

---

**3. MWUA Features (Constraint Pressure):** Looks at which constraints are mathematically hard to solve. Extracted from the MWU algorithm.

>  **Constraint Weights :** In each iteration, constraints that remain highly violated receive increased weights by the current assignment have their weights increased exponentially. This transforms the geometric difficulty of an edge into a numerical weight.

***x_avg*** This represents the running average of the variable's value across all rounds. A high value indicates that the vertex was consistently required to satisfy high-weight constraints.

***Weight Aggregates*** These features measure the Local Constraint Pressure. A vertex connected to incident edges with high final weights is identified as a "load-bearer" for the graph's most difficult regions.

---

**4. LP Features:** These features are extracted from the LP relaxation of the underlying integer optimization problem.

***lp_value*** The continuous value assigned to a vertex when binary constraints are relaxed to the interval [0, 1]. It serves as a baseline indicator of membership likelihood.

***lp_certainty*** calculated as |x_{LP} - 0.5|. This metric identifies "indecisive" variables. A certainty of 0 (where x=0.5) indicates a structural bottleneck where the LP solver cannot determine a clear direction, marking it as a high-priority candidate for branching.

---

**5. Heuristic Features (Luby):** captures the probability of a vertex belonging to a MIS based on a repeated randomized parallel heuristic.

***Luby Frequency:*** Luby’s algorithm is executed for $N$ trials (e.g., $N=100$) using distinct random seeds. The frequency feature is the normalized count of how often a vertex was selected in the independent set.

> Vertices with high frequency are considered robust. This provides a stochastic signal of a vertex's suitability for the solution, independent of the formal optimization or weight-update processes.

---

## Feature Summary

| Category | Features |
|-----------|-----------|
| Degree / Neighborhood | Degree Rank, Neighbor Min Rank, Neighbor Max Rank, Neighbor Avg Rank |
| Centrality | PageRank, Core Number, Clustering Coefficient, Degree Centrality |
| MWUA | Constraint Weight Min, Constraint Weight Max, Constraint Weight Avg, x_avg |
| LP | lp_value, lp_certainty |
| Luby | Luby Frequency |

**Total: 15 vertex-level features**
