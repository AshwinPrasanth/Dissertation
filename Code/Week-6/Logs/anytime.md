# Detailed Experimental Report: From Static MWUA Branching to Spine-Guided Hybrid Search

## 1. Research objective

The central objective of this dissertation experiment is to investigate whether a **static global representation obtained from the Multiplicative Weights Update Algorithm, MWUA, can provide useful guidance to an exact Branch-and-Bound solver without repeatedly solving expensive optimisation relaxations or deploying a computationally expensive graph neural network during search**.

The initial hypothesis was relatively direct:

> If MWUA produces a meaningful fractional representation of the graph at the root node, vertices with highly certain MWUA values may provide strong branching decisions.

The work therefore began by treating MWUA certainty as a **branch-variable score**. However, the experimental results progressively showed that the role of MWUA is more subtle. A variable can be globally stable according to MWUA without necessarily being the best variable for partitioning the current Branch-and-Bound search space.

This observation led to a sequence of increasingly refined experiments:

1. establish a static MWUA representation;
2. use MWUA directly as a branching score;
3. study depth-limited MWUA intervention;
4. replace depth-based intervention with a preferred search spine;
5. separate **variable selection** from **branch direction**;
6. use dynamic residual graph structure to choose the variable;
7. use the static MWUA prediction only to determine which child should be explored first.

The resulting research direction is therefore no longer simply:

[
\text{MWUA score}\rightarrow\text{branch variable}.
]

Instead, the proposed approach is:

[
\boxed{
\text{Dynamic local state}
\rightarrow
\text{choose branch variable}
}
]

combined with

[
\boxed{
\text{Static global MWUA prior}
\rightarrow
\text{choose preferred branch direction}
}
]

This represents the transition from **MWUA as a branching heuristic** to **MWUA as a global search-order prior**.

---

# 2. Problem formulation

The experiments are performed on the Maximum Independent Set problem.

Given an undirected graph

[
G=(V,E),
]

the Maximum Independent Set problem seeks a largest subset

[
S\subseteq V
]

such that no two vertices in (S) are adjacent.

The binary integer programming formulation used by SCIP is

[
\max \sum_{v\in V}x_v
]

subject to

[
x_u+x_v\leq1
\qquad
\forall (u,v)\in E,
]

where

[
x_v\in{0,1}.
]

The interpretation is

[
x_v=1
]

if vertex (v) belongs to the independent set, and

[
x_v=0
]

otherwise.

The current solver constructs one binary SCIP variable for each kernel vertex, one edge constraint for each kernel edge, and maximises the sum of all binary variables. This exact MIS model is visible directly in the current implementation. 

---

# 3. Large graph preprocessing and KaMIS reduction

## 3.1 Motivation for reduction

The original test graph is the SNAP `web-Google` graph.

The graph contains approximately

[
875,713
]

vertices and

[
4,322,051
]

edges.

Attempting to construct and solve the complete binary optimisation problem directly caused the Python process to be killed on the local machine. This was expected because constructing hundreds of thousands of binary SCIP variables and millions of graph constraints creates substantial memory and solver overhead.

The experimental pipeline was therefore changed to include **KaMIS/ReduMIS kernelisation before MWUA feature extraction and exact solving**.

The graph is first loaded as an undirected NetworkX graph. Self-loops are removed during loading and vertex labels are converted to contiguous integer identifiers. 

The graph is then converted into CSR representation using the arrays

[
\text{xadj}
]

and

[
\text{adjncy}.
]

The current implementation explicitly constructs this CSR representation before passing the graph to the KaMIS reduction binding. 

KaMIS reduction is invoked through

```python
_kamis.redumis_kernel(...)
```

and returns the reduced adjacency structure and reverse mapping. 

## 3.2 Reduction result

For `web-Google`, the reduction produced:

| Quantity          |     Value |
| ----------------- | --------: |
| Original vertices |   875,713 |
| Original edges    | 4,322,051 |
| Kernel vertices   |       345 |
| Kernel edges      |     1,414 |
| Core ratio        |  0.000394 |
| Vertices removed  |    99.96% |

Thus, the computational domain for MWUA and SCIP was reduced from

[
875,713
]

vertices to only

[
345.
]

The key point is that **MWUA is not computed on the original 875,713-node graph in the current pipeline**.

The pipeline is:

```text
Original SNAP graph
        ↓
CSR conversion
        ↓
KaMIS / ReduMIS reductions
        ↓
345-node irreducible kernel
        ↓
MWUA feature extraction
        ↓
SCIP exact MIS solve
```

The reduced graph is reconstructed as a NetworkX graph and returned from `reduce_graph`.  The same returned graph then replaces `G` inside `solve_instance`. 

This is important because it guarantees that the graph supplied to the branching rule is the **same reduced graph represented by the SCIP variables**.

---

# 4. MWUA root representation

## 4.1 Feature extraction

After kernelisation, the reduced graph is converted into the optimisation representation required by the existing MWUA feature implementation.

The current pipeline performs:

```text
Reduced graph G
      ↓
build_vertex_cover_problem(G)
      ↓
DatasetBuilder
      ↓
MWUA feature computation
```

The current implementation constructs the vertex-cover problem and builds the feature dataset before extracting the `mwua_xavg` feature. 

This follows the MWUA implementation framework supplied by Ryan, where the underlying solver operates on general hitting-set style constraints and vertex cover is represented as a special case.

The MWUA process maintains multiplicative constraint weights and repeatedly constructs fast interim solutions. The interim solver is particularly important because, unlike simple textbook MWUA examples, vertex-cover constraints have the form

[
x_u+x_v\geq1.
]

The coefficients and right-hand side are generally equal to one. Therefore, the simple assignment strategy often shown in basic MWUA examples cannot directly provide a useful interim solution.

Ryan's implementation instead uses a specialised greedy interim solver. The average behaviour of a variable across MWUA rounds is represented through the `mwua_xavg` feature.

## 4.2 Converting the representation for MIS

The current SCIP problem is Maximum Independent Set, whereas the MWUA problem representation originates from vertex cover.

For a graph,

[
S\text{ is an independent set}
]

if and only if

[
V\setminus S
]

is a vertex cover.

Thus,

[
x_v^{MIS}=1-x_v^{VC}.
]

The extracted feature used in the current code is interpreted as the MIS-side average representation before certainty and prediction are calculated.

The code obtains `mwua_xavg` and then calculates

[
C_{MWUA}(v)
===========

\left|
x_v^{MWUA}-0.5
\right|.
]

This is implemented as

```python
scores = np.abs(
    mis_xavg - 0.5
)
```

in the current solver. 

The predicted binary direction is

[
P_{MWUA}(v)
===========

\begin{cases}
1,&x_v^{MWUA}>0.5\
0,&x_v^{MWUA}\leq0.5.
\end{cases}
]

The current code implements this threshold directly. 

Therefore, MWUA provides two distinct signals.

### MWUA certainty

[
C_{MWUA}(v)=|x_v^{MWUA}-0.5|.
]

This measures how far the MWUA representation lies from the ambiguous midpoint.

For example,

[
x_v^{MWUA}=0.998
]

gives

[
C_{MWUA}(v)=0.498.
]

Similarly,

[
x_v^{MWUA}=0.502
]

gives

[
C_{MWUA}(v)=0.002.
]

The first vertex is highly certain according to the static MWUA representation. The second is highly uncertain.

### MWUA prediction

The second signal is the side of (0.5):

[
x_v^{MWUA}>0.5
\Rightarrow
P_{MWUA}(v)=1.
]

This predicts that the MIS branch

[
x_v=1
]

should be prioritised.

Conversely,

[
x_v^{MWUA}\leq0.5
\Rightarrow
P_{MWUA}(v)=0.
]

This predicts that

[
x_v=0
]

should be prioritised.

The distinction between **certainty** and **prediction** became central to the later experiments.

---

# 5. Initial experiment: MWUA as a direct branching score

The first implementation treated MWUA certainty as a direct branch-variable score.

At every eligible Branch-and-Bound node, SCIP exposes a set of LP branching candidates

[
C.
]

The MWUA branching policy selected

[
v^*
===

\arg\max_{v\in C}
C_{MWUA}(v).
]

In words:

> Among the current fractional branching candidates, select the variable with the highest root MWUA certainty.

The initial intuition was that a highly certain MWUA variable represented a stable structural decision and might therefore provide a strong branch.

This produced encouraging results on smaller Erdős–Rényi graphs.

Earlier experiments showed, for example:

| Graph setting | LP average nodes | MWUA average nodes | Degree average nodes |
| ------------- | ---------------: | -----------------: | -------------------: |
| (n=30,p=0.2)  |             51.8 |               31.9 |                 25.4 |
| (n=50,p=0.2)  |            583.6 |              267.9 |                238.2 |
| (n=70,p=0.2)  |           4625.0 |             1983.9 |               1794.2 |

MWUA significantly outperformed the simple LP-based branching strategy as graph size increased.

The experiments also identified relationships between MWUA certainty and variable behaviour:

[
\operatorname{corr}
(
MWUA,
stability
)
=

0.398
]

and

[
\operatorname{corr}
(
MWUA,
fractional\ persistence
)
=

-0.826.
]

These results suggested that MWUA was capturing meaningful information about variable stability and persistence.

However, the large kernel experiment exposed an important limitation.

---

# 6. Depth-limited MWUA branching

## 6.1 Motivation

Applying a static root representation throughout the complete search tree is potentially problematic.

MWUA is calculated once:

[
G_0
\rightarrow
x^{MWUA}.
]

However, after several Branch-and-Bound decisions, SCIP is solving a residual problem

[
G_t.
]

Generally,

[
G_t\neq G_0.
]

The root MWUA representation is therefore increasingly stale as the solver moves deeper into the tree.

To investigate how long the static representation remained useful, a depth-limited branching policy was implemented.

MWUA controls branching only while

[
depth\leq D.
]

After depth (D), the custom branch rule returns control to SCIP.

The tested depths included

[
D=0,\ 2,\ 5,\ 10
]

and unrestricted MWUA intervention.

## 6.2 Interpretation of depth control

For

[
D=0,
]

MWUA intervenes only at the root.

For

[
D=2,
]

MWUA can control every visited search node in the first three levels of the tree.

Conceptually:

```text
                     root
                     MWUA
                    /    \
                 MWUA    MWUA
                 / \      / \
              MWUA MWUA MWUA MWUA
```

Thus, depth-based control is not a single trajectory.

It applies the static MWUA representation to an entire shallow region of the search tree.

## 6.3 Initial depth experiment

An earlier benchmark produced:

| Strategy     |  Runtime |  Nodes | Search reduction |
| ------------ | -------: | -----: | ---------------: |
| SCIP         | 381.33 s | 12,773 |            0.00% |
| MWUA Root    | 374.34 s | 12,855 |           -0.64% |
| MWUA Depth2  | 364.35 s | 11,175 |           12.51% |
| MWUA Depth5  | 357.03 s | 10,783 |           15.58% |
| MWUA Depth10 | 369.21 s | 13,427 |           -5.12% |
| MWUA Full    | 357.34 s | 21,539 |          -68.63% |

This experiment initially suggested a **useful intervention window**.

Shallow MWUA intervention improved search, but excessive MWUA control degraded performance.

In particular,

[
D=5
]

reduced search by approximately

[
15.58%.
]

However, unrestricted MWUA increased the node count by approximately

[
68.63%.
]

The first interpretation was therefore:

> The root MWUA snapshot contains useful information, but the information becomes stale as search progresses.

This motivated the idea of a **lookback horizon** or **trust horizon** for the static global representation.

---

# 7. Large kernel depth experiment

The same conceptual experiment was then tested on the 345-node `web-Google` kernel.

The default SCIP solve explored approximately

[
670,000
]

nodes in the earlier baseline run.

The root-only MWUA experiment produced:

[
840,350
]

nodes and

[
205.27\text{ s}.
]

The depth-two MWUA experiment produced:

[
1,581,768
]

nodes and

[
371.01\text{ s}.
]

Thus,

[
D=0
\rightarrow
840,350
]

whereas

[
D=2
\rightarrow
1,581,768.
]

Increasing MWUA control from the root to depth two almost doubled the explored search tree.

This result changed the interpretation of the earlier experiments.

The problem was not simply that MWUA became stale at large depths.

Even at very shallow depth, applying the same static certainty ranking **across multiple branches of the tree** could be harmful.

Consider the root decision

[
x_{199}=1.
]

If MWUA predicts (x_{199}=1), then one child is the preferred MWUA region and the other is the deferred region.

Depth-based search behaves conceptually as follows:

```text
                 x199
                /    \
             x=0      x=1
              ↓        ↓
            MWUA      MWUA
```

The problem is that the left child

[
x_{199}=0
]

already contradicts the preferred MWUA direction.

Nevertheless, the depth policy continues applying the same root MWUA certainty ranking inside this contradictory residual region.

This raised a new question:

> Should the static MWUA representation control a region of the search tree, or should it only define one preferred trajectory?

This led to the spine experiment.

---

# 8. Transition from depth-based search to MWUA spine search

## 8.1 Supervisor's anytime intuition

The search strategy was redesigned around the idea:

> Solve the certain decisions first, move toward the uncertain region, and allow the exact solver to backtrack later.

This is fundamentally different from depth-based control.

Depth-based control asks:

> At which tree depths should MWUA be allowed to branch?

Spine-based control asks:

> How long should the solver follow one MWUA-consistent preferred trajectory?

The spine therefore represents a **single path through the Branch-and-Bound tree**.

Conceptually:

```text
                   root
                  /    \
          deferred      preferred
             ↓               ↓
            SCIP            MWUA
                            /  \
                    deferred    preferred
                       ↓             ↓
                      SCIP          MWUA
                                      ↓
                                  uncertain
                                      ↓
                                     SCIP
```

MWUA controls only the preferred child sequence.

When SCIP later backtracks into a deferred child, the custom MWUA spine policy does not attempt to reapply the static root ranking.

This is a major conceptual distinction.

The depth policy is

[
\boxed{
\text{static signal over a shallow tree region}
}
]

whereas the spine policy is

[
\boxed{
\text{static signal along one confidence-consistent trajectory}
}
]

---

# 9. Implementation of the MWUA spine

At the root, the branch rule identifies the preferred MWUA candidate.

The preferred child is assigned a very high node selection priority.

The deferred child receives a very low priority.

Conceptually:

[
priority(preferred)=+10^6
]

and

[
priority(deferred)=-10^6.
]

Both children remain in the Branch-and-Bound tree.

Therefore, the approach **does not prune the non-preferred MWUA branch**.

This is essential for exactness.

The solver remains capable of proving optimality because the deferred child can later be explored through normal backtracking.

The branch rule records the SCIP node number of the preferred child:

```text
preferred_node_number
```

At the next branch callback, MWUA is permitted to intervene only if the current SCIP node number matches this stored preferred node.

Therefore:

```text
current node = preferred node
        ↓
MWUA spine may continue

current node ≠ preferred node
        ↓
return DIDNOTRUN
        ↓
SCIP controls branching
```

This mechanism ensures that MWUA follows exactly one preferred trajectory.

The spine records:

* selected vertices;
* MWUA certainties;
* predicted directions;
* spine length;
* uncertainty stops;
* length stops;
* off-spine callbacks;
* residual degrees in the current hybrid implementation.

The current `solver_runner.py` prints these quantities after solving. 

---

# 10. Certainty-based spine experiment

The first spine implementation retained the original MWUA variable-selection policy.

At every spine node,

[
v^*
===

\arg\max_{v\in C}
C_{MWUA}(v).
]

The MWUA prediction then determined the preferred child.

Thus, MWUA performed two jobs:

```text
MWUA certainty
       ↓
WHICH VARIABLE?

MWUA prediction
       ↓
WHICH CHILD FIRST?
```

The spine length was controlled using

[
K.
]

For example,

[
K=1
]

allows one MWUA decision on the preferred trajectory.

[
K=3
]

allows three consecutive MWUA decisions.

The policy stops when either the maximum spine length is reached or the selected candidate falls below a certainty threshold.

---

# 11. Spine results

The current observed results are:

| Strategy      | MWUA decisions |              Nodes |  Runtime |
| ------------- | -------------: | -----------------: | -------: |
| SCIP baseline |              0 | approximately 670k | baseline |
| Spine K1      |              1 |            788,108 | 185.93 s |
| Spine K2      |              2 |          1,120,583 | 268.61 s |
| Spine K3      |              3 |          1,436,441 | 331.84 s |
| Spine K8      |              8 |          1,242,602 | 288.88 s |

For K1, the selected vertex was

[
199.
]

For K2, the trajectory was

[
199\rightarrow230.
]

For K3,

[
199\rightarrow230\rightarrow106.
]

The corresponding MWUA certainties were

[
0.498047,
]

[
0.497846,
]

and

[
0.497500.
]

For the eight-step spine, the selected vertices were

[
[199,230,106,183,323,138,313,258].
]

Their certainties were approximately

[
[
0.498047,
0.497846,
0.497500,
0.497500,
0.497420,
0.497380,
0.496867,
0.496594
].
]

All eight predictions were

[
x_v=1.
]

After the eighth decision, the next available MWUA candidate had certainty approximately

[
0.000648.
]

This produced a very clear **certainty cliff**:

```text
0.498047
0.497846
0.497500
0.497500
0.497420
0.497380
0.496867
0.496594
     ↓
0.000648
```

Thus, the MWUA representation appears to identify a highly certain prefix followed by a sharply uncertain region.

This strongly supports the supervisor's intuition that there may exist a transition from a certain region to an uncertain region.

However, the search results revealed a second, equally important phenomenon.

---

# 12. MWUA certainty is not direct branching utility

From K1 to K2, the node count increased from

[
788,108
]

to

[
1,120,583.
]

This is an increase of approximately

[
42.2%.
]

The only additional MWUA variable was vertex 230.

Its certainty was

[
0.497846,
]

almost identical to the first vertex's certainty of

[
0.498047.
]

Nevertheless, branching on this second highly certain variable substantially worsened the exact search trajectory.

K3 increased the node count further to

[
1,436,441.
]

The first three certainties were all approximately

[
0.498.
]

Yet search performance progressively degraded.

This provides strong evidence that

[
\boxed{
\text{MWUA certainty}
\neq
\text{branching utility}
}
]

A highly certain variable may be globally stable without producing a useful partition of the current search space.

This distinction is fundamental.

MWUA certainty asks:

> How strongly does the global root representation favour one assignment for this variable?

Branching utility asks:

> If this variable is branched on at the current Branch-and-Bound node, how effectively will the two children decompose the remaining search problem?

These are not equivalent objectives.

For example, consider:

| Vertex | MWUA certainty | Current residual degree |
| ------ | -------------: | ----------------------: |
| (A)    |          0.498 |                       2 |
| (B)    |          0.470 |                      40 |

MWUA certainty chooses (A).

However, fixing (A) may affect only two active neighbours.

Fixing (B) may alter the status of forty active neighbours.

Therefore, (B) may create a much stronger search decomposition even though its global MWUA certainty is lower.

The spine experiment therefore exposed a limitation in the original policy:

[
v^*
===

\arg\max C_{MWUA}(v).
]

The MWUA signal was being asked to choose the branch variable even though its observed strength appeared to be **assignment stability and directional information**.

---

# 13. Revised hypothesis: separate variable selection and branch direction

The experimental evidence motivated a decomposition of the branching decision into two separate questions.

## Question 1: Which variable should be branched on?

This requires information about the **current residual search state**.

## Question 2: Which child should be explored first?

This may benefit from the **static global MWUA prior**.

Therefore, the proposed architecture is:

```text
CURRENT RESIDUAL GRAPH
          ↓
choose branch variable
          ↓
ROOT MWUA SNAPSHOT
          ↓
choose preferred child
```

Mathematically,

[
v^*
===

\arg\max_{v\in C}
D_{\mathrm{res}}(v)
]

and

[
b^*
===

P_{MWUA}(v^*).
]

This is the current hybrid experiment.

---

# 14. Dynamic residual degree

The dynamic local feature currently used is **residual degree**.

For each SCIP variable (x_v), the current local bounds are inspected.

A binary variable is considered active if

[
LB_v<0.5
]

and

[
UB_v>0.5.
]

For an unfixed binary variable,

[
LB_v=0,\qquad UB_v=1.
]

A variable fixed to zero has

[
LB_v=UB_v=0.
]

A variable fixed to one has

[
LB_v=UB_v=1.
]

In the installed PySCIPOpt version, these values are accessed directly from the SCIP variable using:

```python
var.getLbLocal()
var.getUbLocal()
```

The residual degree of vertex (v) is then

[
D_{\mathrm{res}}(v)
===================

\left|
{
u\in N(v):
u\text{ remains unfixed}
}
\right|.
]

Thus, residual degree is not the original graph degree

[
D_G(v).
]

Instead, it is recomputed conceptually with respect to the current Branch-and-Bound state.

Suppose a vertex originally has five neighbours:

```text
        v
     / / \ \ \
    a b  c d  e
```

At the root,

[
D_{\mathrm{res}}(v)=5.
]

Suppose branching and propagation fix (a) and (c).

The active neighbours are now

[
b,d,e.
]

Therefore,

[
D_{\mathrm{res}}(v)=3.
]

This provides a dynamic local signal.

MWUA remains fixed at the root:

[
C_{MWUA}(v)=\text{constant}.
]

Residual degree changes during search:

[
D_{\mathrm{res}}(v,t).
]

This creates the intended static-global/dynamic-local combination.

---

# 15. Current candidate selection policy

At a SCIP Branch-and-Bound node, the branch rule obtains the LP branching candidates.

For every candidate (v), the branch rule constructs information of the form

[
(
v,
C_{MWUA}(v),
D_{\mathrm{res}}(v)
).
]

Two selectors are now supported.

## MWUA selector

The MWUA selector ranks primarily by certainty:

[
v^*
===

\arg\max_v
\left(
C_{MWUA}(v),
D_{\mathrm{res}}(v)
\right).
]

Residual degree is only used as a tie-breaking signal.

Conceptually:

```text
highest MWUA certainty
        ↓
tie?
        ↓
higher residual degree
```

## Residual-degree selector

The dynamic selector ranks primarily by current residual degree:

[
v^*
===

\arg\max_v
\left(
D_{\mathrm{res}}(v),
C_{MWUA}(v)
\right).
]

MWUA certainty is only used as a tie breaker.

Conceptually:

```text
highest current residual degree
        ↓
tie?
        ↓
higher MWUA certainty
```

This separation is important because the proposed hybrid no longer claims that MWUA directly predicts the strongest branch variable.

Instead:

> Dynamic local graph information identifies a structurally influential variable in the current search state.

---

# 16. Branch direction

After selecting the variable, the solver must determine which child should receive preferred search priority.

Suppose the selected variable is

[
x_v.
]

The two children are

[
x_v=0
]

and

[
x_v=1.
]

Both children are created.

The experiment now compares two direction policies.

## 16.1 LP direction

When

```python
use_mwua_direction=False
```

the current LP value of the selected variable is inspected.

If

[
x_v^{LP}\geq0.5,
]

the preferred direction is

[
x_v=1.
]

Otherwise,

[
x_v=0.
]

Thus,

[
P_{LP}(v)
=========

\mathbf{1}
[
x_v^{LP}\geq0.5
].
]

This produces the Degree K1 strategy.

## 16.2 MWUA direction

When

```python
use_mwua_direction=True
```

the root MWUA prediction is used:

[
P_{MWUA}(v)
===========

\mathbf{1}
[
x_v^{MWUA}>0.5
].
]

This produces the Hybrid K1 strategy.

The important point is that MWUA does **not select the variable** in the hybrid experiment.

The current residual graph selects the variable.

MWUA only answers:

> Given that the dynamic local policy has selected this variable, which assignment should be explored first?

---

# 17. Degree K1 versus Hybrid K1

The current ablation is deliberately designed to isolate the value of MWUA direction.

## Degree K1

```text
LP branching candidates
        ↓
calculate current residual degree
        ↓
choose maximum residual-degree vertex
        ↓
inspect current LP value
        ↓
choose LP-preferred child
        ↓
continue SCIP
```

Mathematically,

[
v^*
===

\arg\max_{v\in C}
D_{\mathrm{res}}(v)
]

and

[
b^*
===

\mathbf{1}
[
x_{v^*}^{LP}\geq0.5
].
]

## Hybrid K1

```text
LP branching candidates
        ↓
calculate current residual degree
        ↓
choose maximum residual-degree vertex
        ↓
look up root MWUA prediction
        ↓
choose MWUA-preferred child
        ↓
continue SCIP
```

Mathematically,

[
v^*
===

\arg\max_{v\in C}
D_{\mathrm{res}}(v)
]

and

[
b^*
===

\mathbf{1}
[
x_{v^*}^{MWUA}>0.5
].
]

The variable-selection equation is identical.

Only the preferred direction changes.

Therefore, if both strategies report

```text
vertex = 42
residual_degree = 19
```

but Degree K1 chooses

```text
prediction = 0
```

and Hybrid K1 chooses

```text
prediction = 1
```

then any resulting difference in search trajectory is attributable to the child-ordering policy.

This is a much cleaner ablation than comparing pure MWUA branching against SCIP.

---

# 18. Why K1 is used first

The first hybrid experiment uses a spine length of one:

[
K=1.
]

This means the custom policy intervenes once.

After the preferred root branch is created, normal SCIP search resumes.

This is intentional.

The earlier experiments showed:

[
K1=788,108
]

[
K2=1,120,583
]

[
K3=1,436,441.
]

Repeated intervention introduces complex interactions between variable selection, child direction, and the changing residual graph.

Using K1 first isolates the directional effect.

The current research question is therefore not yet:

> Does repeated hybrid guidance solve the problem faster?

It is:

[
\boxed{
\text{For the same dynamically selected variable, does MWUA provide a better search direction than the current LP value?}
}
]

Only after this is answered should the hybrid policy be extended to

[
K=2,3,\ldots
]

or to an adaptive stopping rule.

---

# 19. The anytime interpretation

The proposed method should be understood as a **search-ordering policy**, not a pruning rule.

Both branches remain alive.

Suppose MWUA predicts

[
x_v=1.
]

The tree still contains:

```text
             x_v
            /   \
         x=0     x=1
       deferred  preferred
```

The (x_v=0) child is not deleted.

Instead, the (x_v=1) child is explored earlier.

The hypothesis is that the MWUA representation may guide the solver toward high-quality feasible solutions earlier.

This motivates an anytime evaluation.

Define the best incumbent objective at time (t) as

[
I(t)
====

\max
{
f(x):
x\text{ has been found by time }t
}.
]

For MIS, higher values are better.

The anytime incumbent curve should therefore plot

[
t
]

on the x-axis and

[
I(t)
]

on the y-axis.

Because an incumbent changes only when a better solution is found, this is naturally a step curve.

The central anytime question is:

> At equal computational budgets, does the MWUA-guided search reach stronger feasible solutions earlier than standard search?

This is different from time to prove optimality.

A strategy may find the optimal independent set after 20 seconds but require 300 seconds to prove that no better solution exists.

Another strategy may require 100 seconds to find the same solution but prove optimality after 200 seconds.

The first strategy has better early anytime behaviour even though its final exact solving time is worse.

---

# 20. Primal-gap curve

For cross-instance comparison, the incumbent objective can be normalised using the known optimal objective

[
z^*.
]

The primal gap is

[
g(t)
====

\frac{
z^*-I(t)
}{
|z^*|
}.
]

For the current kernel,

[
z^*=138.
]

If the current incumbent is

[
I(t)=136,
]

then

[
g(t)
====

\frac{138-136}{138}
\approx0.01449.
]

A lower primal-gap curve is better.

The area under the primal-gap curve can then be calculated:

[
AUC_{gap}
=========

\int_0^Tg(t),dt.
]

The interpretation is

[
\boxed{
\text{lower primal-gap AUC}
===========================

\text{better anytime performance}
}
]

This metric is particularly appropriate because the dissertation is now investigating **search prioritisation**, not merely final node count.

---

# 21. Current experimental interpretation

The experiments so far support four important conclusions.

### First, root MWUA contains meaningful information

The earlier correlation experiments and certainty behaviour show that MWUA is not random.

MWUA certainty is related to stability and strongly related to fractional persistence.

The sharp certainty transition after the first eight spine candidates also indicates that MWUA identifies a structurally distinct high-certainty subset.

### Second, MWUA certainty is not a direct branch-utility score

The K1, K2, and K3 results show that variables with almost identical MWUA certainty can have dramatically different effects on the exact search tree.

Therefore,

[
C_{MWUA}(v)
]

should not automatically be interpreted as

[
\text{branch quality}(v).
]

### Third, static root information should not be uniformly applied across the search tree

The depth-two result showed that applying root MWUA guidance across both preferred and deferred shallow branches can substantially increase the search tree.

This motivates trajectory-specific rather than depth-specific intervention.

### Fourth, the role of MWUA may be directional

MWUA appears naturally suited to answering:

> Which assignment does the global root representation favour?

This corresponds directly to child ordering.

Dynamic residual structure is more naturally suited to answering:

> Which variable currently has the greatest structural influence?

This corresponds to variable selection.

The emerging hypothesis is therefore:

[
\boxed{
\text{MWUA is a global directional prior rather than a direct branching score.}
}
]

---

# 22. Evolution of the proposed method

The research progression can be summarised as follows.

### Stage 1

```text
Highest MWUA certainty
        ↓
branch on that variable
```

Hypothesis:

> Global certainty directly predicts branching quality.

### Stage 2

```text
Highest MWUA certainty
        ↓
branch only until depth D
        ↓
return control to SCIP
```

Hypothesis:

> Static MWUA is useful only near the root.

### Stage 3

```text
Highest MWUA certainty
        ↓
follow preferred MWUA child
        ↓
continue along one spine
        ↓
stop at uncertainty or length K
        ↓
SCIP backtracking
```

Hypothesis:

> Static MWUA should guide one confidence-consistent trajectory rather than an entire tree region.

### Stage 4

```text
Current residual degree
        ↓
choose variable

Root MWUA prediction
        ↓
choose preferred child
```

Current hypothesis:

> Static global and dynamic local information have different roles in exact search.

This final stage is the closest match to the dissertation's central theme:

> **Combining Static Global and Dynamic Local Features for Fast Learning-to-Branch Heuristics.**

However, the experiments are now revealing a more precise formulation than the original weighted-score idea.

Rather than immediately calculating

[
Score(v)
========

\alpha MWUA
+
\beta Degree
+
\gamma LP
+
\delta PseudoCost,
]

the current approach decomposes the branching operation functionally:

[
\boxed{
\text{Dynamic local feature}
\rightarrow
\text{variable selection}
}
]

and

[
\boxed{
\text{Static global feature}
\rightarrow
\text{branch direction}
}
]

This decomposition is experimentally cleaner and substantially easier to interpret.

---

# 23. Current contribution being tested

The current algorithmic contribution can be stated as:

> We investigate whether a root-computed MWUA representation can serve as a static global directional prior for exact combinatorial search. Rather than repeatedly using MWUA certainty as a branch-variable score, the proposed hybrid policy selects structurally influential variables using the current residual graph and uses the root MWUA prediction to prioritise the corresponding child node. The policy operates along a preferred search spine while preserving both branches, thereby retaining exactness and allowing natural Branch-and-Bound backtracking.

The core research question is now:

[
\boxed{
\text{Can a static MWUA global prior improve search ordering when paired with dynamic local variable selection?}
}
]

The immediate Degree K1 versus Hybrid K1 experiment isolates this question.

If the same residual-degree variable is selected in both strategies, but the MWUA-directed strategy produces better incumbent trajectories, lower primal-gap AUC, fewer explored nodes, or faster optimality, then the experiment provides direct evidence that **MWUA contributes useful directional information even when it is not a direct measure of branch-variable utility**.

That is the full experimental story so far, and importantly, it is a much stronger dissertation narrative than simply saying **“we tried MWUA branching at different depths.”** The experiments now form a logical progression where each negative or mixed result identifies a limitation of the previous design and motivates the next algorithmic refinement.
