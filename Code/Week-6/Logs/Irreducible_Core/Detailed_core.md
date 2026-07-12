# Irreducible-Core Preprocessing Pipeline Using KaMIS Kernelization

## Overview

This module implements a root-level graph kernelization pipeline for large-scale Maximum Independent Set problems.

The purpose of the pipeline is to reduce a large input graph to the explicit residual graph that remains after the fixed-point reduction suite implemented by the KaMIS branch-and-reduce framework has been exhausted.

The resulting residual graph is referred to throughout this project as the **irreducible core** or **reduced kernel**.

The pipeline is designed for the dissertation project:

> **Combining Static Global and Dynamic Local Features for Fast Learning-to-Branch Heuristics**

The original branching framework computes optimization signals such as Multiplicative Weights Update Algorithm features and local branching features on the graph being solved. This approach is practical for small and medium-sized instances but becomes computationally expensive on graphs containing hundreds of thousands or millions of vertices.

The irreducible-core pipeline changes the order of computation.

Instead of applying MWUA directly to the raw graph, the graph is first passed through exact, solution-preserving reduction rules.

The resulting pipeline is:

```text
Raw Graph
    |
    v
KaMIS Fixed-Point Kernelization
    |
    v
Irreducible Core
    |
    v
MWUA Root Snapshot
    |
    v
MWUA Certainty Scores
    |
    v
Dynamic Local Features
    |
    v
Certainty-Guided Branching
    |
    v
Core MIS Solution
    |
    v
Solution Lifting
    |
    v
MIS Solution on Original Graph
```

The main principle is simple:

> Mathematical reductions handle graph structure that can be resolved safely without branching. MWUA is applied only to the residual search space where the reduction rules stop.

This separates the responsibilities of kernelization and learning-guided search.

---

# 1. Motivation

The original dissertation experiments focused on branching strategies for exact Maximum Independent Set and Minimum Vertex Cover search.

The central observation from the earlier experiments was that branching trajectory strongly affects search complexity.

The implemented branching signals included:

* LP certainty
* MWUA certainty
* residual degree
* pseudo-cost
* persistence
* local gain

The MWUA root snapshot was particularly promising because it provided a global optimization signal without repeatedly recomputing expensive LP-relaxation features at every search node.

However, applying a feature extractor directly to a massive graph creates a new scalability problem.

For example, the SNAP `web-Google` graph contains:

```text
Vertices : 875,713
Edges    : 4,322,051
```

Its undirected CSR representation contains:

```text
8,644,102 adjacency entries
```

Running the complete Python feature and solver stack directly on such an instance is computationally undesirable. Large Python graph objects, feature arrays, residual graph state, LP structures, and repeated neighborhood operations can create substantial memory pressure.

The goal of this work is therefore not to make MWUA process every vertex in a million-node graph.

Instead, the goal is to determine:

> Can exact root-level kernelization restrict MWUA computation to the unresolved residual search space?

The proposed computational pipeline is:

```text
Large graph
    |
    v
Fast exact reductions
    |
    v
Small residual kernel
    |
    v
MWUA
```

This is referred to as the **math-first reduction pipeline**.

---

# 2. KaMIS and ReduMIS

KaMIS is a framework for solving Maximum Independent Set problems.

The repository contains several MIS algorithms, including evolutionary algorithms, local-search algorithms, and branch-and-reduce implementations.

The ReduMIS execution path combines graph reductions with an evolutionary MIS framework.

The existing CHSZLabLib KaMIS Python binding exposed:

```python
redumis(...)
```

and:

```python
online_mis(...)
```

The existing `redumis` binding runs the broader ReduMIS search framework.

Internally, the binding constructs:

```cpp
reduction_evolution<branch_and_reduce_algorithm>
```

and calls:

```cpp
evo.perform_mis_search(
    mis_config,
    G,
    independent_set,
    best_nodes
);
```

This is not suitable for the dissertation preprocessing stage.

The objective of the new pipeline is not to run the entire ReduMIS solver.

The objective is to expose only the root-level fixed-point reduction engine.

The relevant KaMIS class is:

```cpp
branch_and_reduce_algorithm
```

located at:

```text
external_repositories/KaMIS/
    lib/mis/kernel/
        branch_and_reduce_algorithm.h
        branch_and_reduce_algorithm.cpp
```

The class exposes the following relevant methods:

```cpp
void initial_reduce_graph();
void reduce_graph();

size_t number_of_nodes_remaining() const;

void convert_adj_lists(
    graph_access &G,
    std::vector<NodeID> &reverse_mapping
) const;

void extend_finer_is(
    std::vector<bool> &independent_set
);
```

The key method used by the new pipeline is:

```cpp
reduce_graph()
```

---

# 3. What `reduce_graph()` Does

The KaMIS branch-and-reduce implementation contains a fixed-point reduction loop.

Conceptually, the implementation performs:

```text
repeat
    apply reduction rules
until no reduction changes the graph
```

The reduction engine considers the following rules:

```text
Degree-1 reduction
Dominance reduction
Unconfined reduction
LP reduction
Packing reduction
Degree-2 folding
Twin reduction
Funnel reduction
Desk reduction
```

The exact set of active reductions depends on the internal KaMIS reduction configuration.

The root reduction loop repeatedly applies the enabled rules.

Conceptually:

```text
Input Graph
    |
    v
Degree-1 Reduction
    |
    v
Dominance / Unconfined Reduction
    |
    v
LP Reduction
    |
    v
Packing Reduction
    |
    v
Degree-2 Folding
    |
    v
Twin Reduction
    |
    v
Funnel Reduction
    |
    v
Desk Reduction
    |
    v
Did the graph change?
    |
    +---- YES ----> Repeat
    |
    +---- NO -----> Return residual graph
```

This repeated application is important.

A reduction can expose new structures that were not previously reducible.

For example:

```text
Unconfined reduction
        |
        v
Vertex removed
        |
        v
Neighbour degree decreases
        |
        v
Degree-2 structure appears
        |
        v
Degree-2 folding
        |
        v
Dominance relationship appears
        |
        v
Dominance reduction
```

The process continues until the graph reaches a fixed point.

The residual graph is therefore not the result of a single reduction pass.

It is the graph remaining after the complete enabled reduction suite has been exhausted.

---

# 4. Reduction Rules

## 4.1 Degree-1 Reduction

Consider a vertex (v) with exactly one neighbour (u).

```text
v --- u
```

For Maximum Independent Set, an optimal solution can be transformed so that the leaf vertex can be selected appropriately without decreasing the solution quality.

The reduction removes locally resolvable structures associated with low-degree vertices.

Degree-1 reductions are especially effective on sparse graphs.

The reduction may also trigger cascades.

For example:

```text
a --- b --- c --- d
```

After reducing one endpoint structure, another low-degree structure can appear.

This explains why sparse Erdős-Rényi instances can sometimes be completely reduced.

In the synthetic experiment:

```text
n = 1000
p = 0.002
average degree approximately 2
```

the complete graph was reduced to an empty kernel.

Observed result:

```text
Original vertices : 1000
Original edges    : 999

Core vertices     : 0
Core edges        : 0

Vertices removed  : 100.00%
Reduction time    : 0.002725 s
```

---

## 4.2 Dominance Reduction

Dominance reductions identify vertices whose neighbourhood structure makes another decision redundant.

A typical structural relationship involves neighbourhood containment.

Conceptually:

```text
N[u] contained in N[v]
```

Under the appropriate MIS reduction condition, one vertex can dominate another.

The dominated structure can be removed while preserving the ability to reconstruct an optimal independent set.

Dominance is particularly useful in graphs containing repeated or nested neighbourhood structures.

---

## 4.3 Unconfined Reduction

The unconfined reduction detects vertices that can safely be excluded from consideration in a maximum independent set.

The rule explores a sequence of neighbouring structures.

The objective is to prove that an optimal solution exists without a particular vertex.

Unlike degree-based rules, the unconfined rule can identify reducible vertices using a larger local structural pattern.

This makes the rule effective on real-world networks where graph topology contains overlapping neighbourhood structures.

---

## 4.4 LP Reduction

The branch-and-reduce implementation contains an LP-based reduction.

This is related to the half-integral structure of the linear programming relaxation of Vertex Cover.

For Vertex Cover, the LP relaxation has the form:

[
\min \sum_{v \in V} x_v
]

subject to:

[
x_u + x_v \geq 1
]

for every edge:

[
(u,v) \in E
]

with:

[
0 \leq x_v \leq 1.
]

The relevant relaxation admits half-integral solutions:

```text
0
1/2
1
```

Vertices associated with integral decisions can be removed according to the corresponding reduction theorem.

The unresolved half-integral region forms part of the residual kernel.

The KaMIS implementation computes the required structure internally using matching and flow-related machinery.

This is different from the dissertation's Python LP feature computation.

The KaMIS LP reduction is part of the exact reduction engine.

The dissertation's LP certainty feature is a branching signal.

These two components must not be confused.

```text
KaMIS LP reduction
    =
exact preprocessing reduction

LP certainty
    =
branching feature
```

---

## 4.5 Packing Reduction

Packing constraints encode restrictions on groups of vertices.

A packing constraint can identify assignments that would violate an existing combinatorial condition.

The KaMIS branch-and-reduce implementation maintains internal packing structures and applies packing-based reductions when enabled.

These reductions can remove additional vertices after other structural transformations.

Packing reductions are also important because their effectiveness can increase after folding or vertex assignments modify the residual graph.

---

## 4.6 Degree-2 Folding

Degree-2 folding handles vertices with two neighbours under specific structural conditions.

Consider:

```text
u --- v --- w
```

where:

```text
degree(v) = 2
```

Rather than immediately assigning every original vertex, the reduction can replace a local structure with a smaller equivalent representation.

This is a graph transformation.

Therefore:

> A vertex disappearing from the explicit residual graph does not necessarily mean that its MIS decision has been individually fixed.

Some decisions are represented implicitly through the folding transformation.

This distinction is critical when interpreting kernel size.

For example:

```text
Original graph : 1000 vertices
Core graph     : 747 vertices
```

The correct interpretation is:

> 253 vertices were removed from the explicit residual search graph.

It is not necessarily correct to state:

> 253 independent binary decisions were permanently fixed.

The folding state must be reversed during solution lifting.

---

## 4.7 Twin Reduction

Twin vertices have identical or structurally equivalent neighbourhood patterns.

For example:

```text
N(u) = N(v)
```

under the required reduction conditions.

Such structures create redundant decisions.

The twin reduction replaces or removes equivalent structures while preserving the maximum independent set objective.

Real-world graphs containing repeated neighbourhood patterns can expose many such opportunities.

---

## 4.8 Funnel Reduction

Funnel reductions detect a specific local graph configuration in which the neighbourhood structure around a vertex permits a safe transformation.

The rule is more expressive than simple degree-based reductions.

The transformation is stored internally so that the corresponding original assignments can later be reconstructed.

---

## 4.9 Desk Reduction

The desk reduction detects another structured local configuration that admits a solution-preserving graph transformation.

Like folding and funnel reductions, the transformation may alter the explicit residual graph rather than simply fixing one original vertex.

The branch-and-reduce implementation stores the required modification information.

---

# 5. Why the Result Is Called an Irreducible Core

After the fixed-point reduction loop terminates, none of the enabled reduction rules can make further progress.

The resulting graph is therefore irreducible with respect to the particular KaMIS reduction suite being used.

Formally, if:

[
R
]

represents the enabled reduction suite, the residual graph:

[
G_K
]

satisfies:

[
R(G_K) = G_K.
]

This does not mean that the graph is impossible to reduce using every known kernelization theorem.

It means:

> The enabled KaMIS branch-and-reduce reduction suite has reached a fixed point.

For this reason, the project uses the terms:

```text
irreducible core
residual kernel
reduced core
```

The most precise dissertation wording is:

> **Residual kernel after fixed-point application of the KaMIS branch-and-reduce reduction suite.**

---

# 6. Difference Between Kernelization and Full ReduMIS

This distinction is central to the implementation.

The original ReduMIS binding performs:

```text
Reductions
    |
    v
Evolutionary MIS Search
    |
    v
Local Search
    |
    v
Recursive Reduction
    |
    v
Final MIS Solution
```

The existing binding contains:

```cpp
reduction_evolution<branch_and_reduce_algorithm> evo;

evo.perform_mis_search(
    mis_config,
    G,
    independent_set,
    best_nodes
);
```

The new kernel binding does not call the evolutionary framework.

It does not call:

```text
perform_mis_search()
perform_evolutionary()
solve()
rec()
branching()
decompose()
compute_maximal_is()
```

Instead, the new binding directly constructs the branch-and-reduce reduction engine and calls:

```cpp
reducer.reduce_graph();
```

The new pipeline is therefore:

```text
Graph
    |
    v
KaMIS reductions only
    |
    v
Residual kernel
```

No branching is performed by KaMIS during this preprocessing stage.

No evolutionary search is performed.

No final MIS solution is computed by ReduMIS.

This allows the dissertation's MWUA-guided solver to operate on the residual kernel.

---

# 7. CHSZLabLib Repository Customization

The existing CHSZLabLib repository already contained Python bindings for KaMIS.

The relevant file is:

```text
bindings/kamis_binding.cpp
```

Before modification, the binding exposed:

```python
redumis
online_mis
```

The Python module definition was conceptually:

```cpp
PYBIND11_MODULE(_kamis, m) {
    m.def("redumis", ...);
    m.def("online_mis", ...);
}
```

A new binding was added:

```python
redumis_kernel
```

The resulting Python module exposes:

```text
online_mis
redumis
redumis_kernel
```

This was verified using:

```python
import _kamis

print(dir(_kamis))
```

Observed output:

```text
[
    '__doc__',
    '__file__',
    '__loader__',
    '__name__',
    '__package__',
    '__spec__',
    'online_mis',
    'redumis',
    'redumis_kernel'
]
```

The kernel function was successfully exposed as a pybind11 built-in method.

---

# 8. New `redumis_kernel` Binding

A new C++ function was introduced in:

```text
bindings/kamis_binding.cpp
```

The function is exposed to Python as:

```python
_kamis.redumis_kernel(...)
```

The binding accepts the same CSR-style graph representation used by the existing KaMIS bindings:

```text
xadj
adjncy
vwgt
```

The graph representation follows the standard compressed sparse row structure.

For a graph containing (n) vertices:

```text
len(xadj) = n + 1
```

The neighbours of vertex (v) are stored in:

```python
adjncy[xadj[v]:xadj[v + 1]]
```

For an undirected graph, each edge appears twice in `adjncy`.

Therefore:

[
|E| = \frac{|\text{adjncy}|}{2}.
]

The current experiments use an empty vertex-weight array:

```python
vwgt = np.asarray([], dtype=np.int32)
```

because the dissertation experiments currently consider unweighted Maximum Independent Set.

---

# 9. Internal Kernel Extraction Process

The new binding performs the following operations.

## Step 1: Build a KaMIS `graph_access` Object

The existing helper:

```cpp
build_graph(...)
```

is reused.

The function receives:

```text
xadj
adjncy
vwgt
```

and constructs:

```cpp
graph_access G;
```

Before graph construction, adjacency lists are sorted using:

```cpp
sort_adjacency_lists(...)
```

For the unweighted case, the graph is built using:

```cpp
G.build_from_metis(...)
```

---

## Step 2: Convert `graph_access` to Adjacency Vectors

The KaMIS branch-and-reduce constructor expects:

```cpp
std::vector<std::vector<int>>
```

rather than `graph_access`.

An adjacency-vector representation is therefore constructed.

Conceptually:

```cpp
std::vector<std::vector<int>> adj(
    G.number_of_nodes()
);

forall_nodes(G, node) {
    forall_out_edges(G, edge, node) {
        NodeID neighbor = G.getEdgeTarget(edge);
        adj[node].push_back(neighbor);
    } endfor
} endfor
```

The resulting structure is:

```text
adj[0] = neighbours of vertex 0
adj[1] = neighbours of vertex 1
...
adj[n-1] = neighbours of vertex n-1
```

---

## Step 3: Construct the KaMIS Reducer

The reducer is created directly:

```cpp
branch_and_reduce_algorithm reducer(
    adj,
    adj.size()
);
```

This initializes the internal branch-and-reduce graph state.

The reducer maintains information including:

```text
adj
x
y
rn
modifieds
packing constraints
fold state
reduction snapshots
```

The important variable for kernel extraction is:

```cpp
rn
```

which represents the number of remaining vertices.

---

## Step 4: Run Fixed-Point Reduction

The binding calls:

```cpp
reducer.reduce_graph();
```

This applies the reduction suite until no enabled rule can further reduce the graph.

Importantly, the binding does not call the recursive exact solver.

The distinction is:

```text
reduce_graph()
    =
root-level fixed-point kernelization

rec()
    =
reduction + lower bounds + decomposition + branching

solve()
    =
complete exact solver execution
```

Only `reduce_graph()` is used.

---

## Step 5: Extract the Residual Graph

After reduction, the remaining graph is converted back into a KaMIS `graph_access` object.

The existing method:

```cpp
convert_adj_lists(...)
```

is used.

Conceptually:

```cpp
graph_access reduced;
std::vector<NodeID> reverse_mapping;

reducer.convert_adj_lists(
    reduced,
    reverse_mapping
);
```

The method creates a compact residual graph.

Suppose the surviving original vertices are:

```text
0
2
3
5
8
```

The core graph uses compact identifiers:

```text
0
1
2
3
4
```

The mapping is:

```text
core 0 -> original 0
core 1 -> original 2
core 2 -> original 3
core 3 -> original 5
core 4 -> original 8
```

This information is stored in:

```text
reverse_mapping
```

---

## Step 6: Convert the Core Back to CSR

The residual `graph_access` object is converted to CSR arrays.

The binding returns:

```python
core_xadj
core_adjncy
reverse_mapping
```

The Python interface is therefore:

```python
core_xadj, core_adjncy, reverse_mapping = (
    _kamis.redumis_kernel(
        xadj,
        adjncy,
        vwgt,
    )
)
```

The number of core vertices is:

```python
core_n = len(core_xadj) - 1
```

The number of core edges is:

```python
core_m = len(core_adjncy) // 2
```

---

# 10. Reverse Mapping

The `reverse_mapping` array maps compact core vertex identifiers to surviving original graph identifiers.

For example, the synthetic experiment returned:

```text
[0, 2, 3, 4, 5, 7, 8, 9, 12, 13, ...]
```

This means:

```text
Core vertex 0 -> Original vertex 0
Core vertex 1 -> Original vertex 2
Core vertex 2 -> Original vertex 3
Core vertex 3 -> Original vertex 4
Core vertex 4 -> Original vertex 5
Core vertex 5 -> Original vertex 7
```

For `web-Google`, the first mappings were:

```text
2110
4967
8266
11366
11460
14251
15529
16605
19664
20306
22521
23623
24691
24977
26089
26531
30562
32340
35527
35712
```

This demonstrates that the 345 core vertices correspond to a sparse subset of the original graph identifiers.

The mapping allows core-level scores to be associated with surviving original vertices.

For example:

```python
original_vertex = reverse_mapping[core_vertex]
```

This is sufficient for identifying the original IDs of surviving vertices.

However, `reverse_mapping` alone is not sufficient for full MIS solution lifting.

---

# 11. Solution Lifting

Several KaMIS reductions are graph transformations.

Examples include:

```text
degree-2 folding
twin reduction
funnel reduction
desk reduction
alternative structures
```

These transformations may encode relationships between original vertices.

Therefore, solving the residual core gives an MIS assignment for the compact core graph.

It does not directly provide the complete assignment for the original graph.

The complete execution pipeline must eventually perform:

```text
Core MIS Assignment
        |
        v
Insert core decisions into reducer state
        |
        v
Reverse stored modifications
        |
        v
Recover folded decisions
        |
        v
Recover reduced decisions
        |
        v
Original graph MIS assignment
```

The KaMIS reducer contains:

```cpp
extend_finer_is(...)
```

which is used by the existing ReduMIS framework to extend a residual independent set through the stored reductions.

The existing evolutionary implementation performs:

```cpp
full_reducer->extend_finer_is(
    independent_set
);
```

The current `redumis_kernel` interface returns the residual graph and reverse mapping but does not yet expose the reducer state to Python after the function returns.

The reducer object currently exists only during the C++ binding call.

Therefore:

> The current implementation is complete for kernel extraction and core-level branching experiments but does not yet implement external core-solution lifting.

This is the next required engineering stage for a complete end-to-end exact solver.

A future binding should preserve the reduction state across the core solve.

Possible implementations include:

### Option A: Stateful C++ Kernel Object

Expose a pybind11 class:

```python
kernelizer = _kamis.ReduMISKernalizer(
    xadj,
    adjncy,
    vwgt,
)

core_xadj, core_adjncy = kernelizer.reduce()

core_solution = solve_core(...)

original_solution = kernelizer.lift(
    core_solution
)
```

The C++ object would keep:

```cpp
std::unique_ptr<branch_and_reduce_algorithm>
```

alive between `reduce()` and `lift()`.

This is the preferred architecture.

### Option B: Entire Pipeline Inside C++

The Python code could provide a branching callback or core solution to a single C++ execution path.

This is more complex and creates tighter coupling between the dissertation solver and KaMIS.

The stateful kernel object is therefore preferred.

---

# 12. Build-System Customization

CHSZLabLib uses CMake and pybind11.

The complete Python package could not initially be installed using:

```bash
pip install -e . --no-build-isolation
```

The first encountered error was:

```text
BackendUnavailable:
Cannot import 'scikit_build_core.build'
```

More importantly, the complete CHSZLabLib repository contains multiple external algorithms and build targets that are unrelated to the dissertation kernelization experiment.

Building the complete ecosystem was unnecessary.

The CMake configuration contains the target:

```cmake
add_library(kamis_redumis_static STATIC
    ${KAMIS_ROOT_KAHIP_SOURCES}
    ${KAMIS_ROOT_LIB_SOURCES}
    ${KAMIS_ONLINE_SOURCES}
)
```

The Python `_kamis` module links to:

```text
kamis_redumis_static
```

The relevant dependency chain is:

```text
_kamis
    |
    v
kamis_redumis_static
    |
    v
KaMIS sources
```

The `_kamis` binding does not require unrelated CHSZLabLib targets such as HeiCut.

A dedicated build directory was therefore created:

```text
build-kamis
```

The build was configured using:

```bash
cmake -S . -B build-kamis -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_C_FLAGS="-I/opt/homebrew/opt/libomp/include" \
  -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/libomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/opt/libomp/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/homebrew/opt/libomp/lib"
```

The OpenMP paths correspond to the Homebrew `libomp` installation on macOS.

Only the `_kamis` target was built:

```bash
cmake --build build-kamis --target _kamis -j2
```

The build completed successfully:

```text
[91/91] Linking CXX shared module
_kamis.cpython-313-darwin.so
```

The generated module was located at:

```text
build-kamis/_kamis.cpython-313-darwin.so
```

The module can be loaded without installing the complete CHSZLabLib package:

```bash
PYTHONPATH="$(pwd)/build-kamis:$PYTHONPATH" python
```

or from the dissertation root:

```bash
PYTHONPATH="$(pwd)/CHSZLabLib/build-kamis:$PYTHONPATH" \
python anytime/experiments/reduce_snap_graph.py
```

This isolated build strategy avoids compiling unrelated CHSZLabLib components.

---

# 13. Initial Synthetic Validation

The binding was first validated using Erdős-Rényi graphs.

## Experiment 1: Moderate Average Degree

Configuration:

```text
n = 1000
p = 0.01
```

Observed graph:

```text
Vertices : 1000
Edges    : 4985
```

Kernelization result:

```text
Core vertices     : 992
Core edges        : 4977
Core ratio        : 0.9920
Vertices removed  : 0.80%
Reduction time    : 0.012216 s
```

The graph was largely irreducible under the enabled reduction suite.

---

## Experiment 2: Sparse Graph

Configuration:

```text
n = 1000
p = 0.002
```

Observed graph:

```text
Vertices : 1000
Edges    : 999
```

Kernelization result:

```text
Core vertices     : 0
Core edges        : 0
Core ratio        : 0.0000
Vertices removed  : 100.00%
Reduction time    : 0.002725 s
```

The reduction suite completely eliminated the explicit residual graph.

No core branching was required.

---

## Experiment 3: Intermediate Density

Configuration:

```text
n = 1000
p = 0.005
```

Observed graph:

```text
Vertices : 1000
Edges    : 2495
```

Kernelization result:

```text
Core vertices     : 747
Core edges        : 2117
Core ratio        : 0.7470
Vertices removed  : 25.30%
Reduction time    : 0.007561 s
```

This experiment produced a partial kernel.

The results demonstrate that kernelization effectiveness depends strongly on graph structure.

Summary:

| p     | Approximate Average Degree | Core Vertices | Core Ratio | Removed |
| ----- | -------------------------: | ------------: | ---------: | ------: |
| 0.002 |                          2 |             0 |      0.000 |  100.0% |
| 0.005 |                          5 |           747 |      0.747 |   25.3% |
| 0.010 |                         10 |           992 |      0.992 |    0.8% |

These experiments motivated evaluation on real-world graph structures.

---

# 14. SNAP Graph Processing Pipeline

A Python experiment was implemented at:

```text
anytime/experiments/reduce_snap_graph.py
```

The experiment performs:

```text
SNAP edge-list file
        |
        v
NetworkX graph loading
        |
        v
Self-loop removal
        |
        v
Integer node relabelling
        |
        v
CSR conversion
        |
        v
_kamis.redumis_kernel()
        |
        v
Core statistics
```

The current loader uses:

```python
nx.read_edgelist(
    path,
    comments="#",
    nodetype=int,
    create_using=nx.Graph(),
)
```

Self-loops are removed using:

```python
G.remove_edges_from(
    nx.selfloop_edges(G)
)
```

The graph is converted to contiguous integer identifiers using:

```python
nx.convert_node_labels_to_integers(
    G,
    first_label=0,
    ordering="sorted",
)
```

The graph is then converted to CSR.

The kernel binding is called using:

```python
core_xadj, core_adjncy, reverse_mapping = (
    _kamis.redumis_kernel(
        xadj,
        adjncy,
        vwgt,
    )
)
```

The experiment records:

```text
original vertices
original edges
average degree
CSR conversion time
CSR adjacency size
core vertices
core edges
core ratio
removed ratio
edge core ratio
reduction time
mapping size
```

---

# 15. SNAP Collaboration-Network Results

Three SNAP collaboration graphs were evaluated.

## ca-HepPh

Original graph:

```text
Vertices       : 12,008
Edges          : 118,489
Average degree : 19.7350
```

Kernel result:

```text
Core vertices    : 0
Core edges       : 0
Core ratio       : 0.000000
Vertices removed : 100.00%
Reduction time   : 0.007077 s
```

---

## ca-HepTh

Original graph:

```text
Vertices       : 9,877
Edges          : 25,973
Average degree : 5.2593
```

Kernel result:

```text
Core vertices    : 0
Core edges       : 0
Core ratio       : 0.000000
Vertices removed : 100.00%
Reduction time   : 0.006341 s
```

---

## ca-AstroPh

Original graph:

```text
Vertices       : 18,772
Edges          : 198,050
Average degree : 21.1006
```

Kernel result:

```text
Core vertices    : 0
Core edges       : 0
Core ratio       : 0.000000
Vertices removed : 100.00%
Reduction time   : 0.010796 s
```

All three collaboration graphs reached an empty residual kernel.

The correct interpretation is:

> The fixed-point KaMIS reduction suite eliminated the complete explicit residual search graph for these instances.

This does not mean every vertex was independently assigned by a trivial reduction.

Folding and other transformations can represent original decisions implicitly.

The reduction state is required for final solution reconstruction.

---

# 16. `web-Google` Result

The most important current result was obtained on the SNAP `web-Google` graph.

Original graph:

```text
Vertices       : 875,713
Edges          : 4,322,051
Average degree : 9.8709
```

Loading time:

```text
8.918571 s
```

CSR conversion time:

```text
13.981826 s
```

CSR adjacency entries:

```text
8,644,102
```

KaMIS kernelization produced:

```text
Core vertices    : 345
Core edges       : 1,414
Core ratio       : 0.000394
Vertices removed : 99.96%
Edge core ratio  : 0.000327
Reduction time   : 0.630341 s
```

The explicit vertex domain changed from:

```text
875,713
```

to:

```text
345
```

The vertex-domain reduction factor is approximately:

[
\frac{875713}{345} \approx 2538.
]

The complete structural transformation is:

```text
web-Google

875,713 vertices
4,322,051 edges
        |
        | KaMIS fixed-point reductions
        | 0.630341 seconds
        v
345 vertices
1,414 edges
```

Only approximately:

```text
0.0394%
```

of the original vertices remain explicitly represented in the residual kernel.

This result motivates applying MWUA only after kernelization.

---

# 17. Revised MWUA Pipeline

The previous experimental architecture was:

```text
Raw Graph
    |
    v
MWUA
    |
    v
Feature Extraction
    |
    v
Branch-and-Bound
```

For massive graphs, this requires MWUA and the feature pipeline to operate directly on the complete graph.

The revised architecture is:

```text
Raw Graph
    |
    v
KaMIS Kernelization
    |
    v
Residual Core
    |
    v
MWUA Root Snapshot
    |
    v
Static MWUA Certainty
    |
    v
Dynamic Residual Degree
    |
    v
Pseudo-Cost / Local Features
    |
    v
Certainty-Guided Branching
```

For `web-Google`:

```text
MWUA domain before preprocessing:
875,713 vertices

MWUA domain after preprocessing:
345 vertices
```

The objective is not to compare the runtime of raw MWUA and core MWUA on every massive graph.

Running raw MWUA may itself be computationally impractical and can create unnecessary memory pressure.

Instead, the pipeline treats exact kernelization as the computational gate.

The experimental question becomes:

> Can exact root-level kernelization make MWUA-based optimization features feasible on large graph instances by restricting feature extraction to the residual search space?

---

# 18. Test-Time Processing

Kernelization is performed at test time.

The graph does not need to be preprocessed and saved as a separate core file.

The intended solver execution is:

```text
Receive Graph G
      |
      v
Convert G to CSR
      |
      v
Run KaMIS kernelization
      |
      v
Check core size
      |
      +-----------------------------+
      |                             |
      | core size = 0               | core size > 0
      |                             |
      v                             v
No MWUA required               Run MWUA on core
                                    |
                                    v
                              Branch on core
                                    |
                                    v
                              Solve core MIS
                                    |
                                    v
                              Lift solution
```

The total runtime should eventually be reported as:

[
T_{\text{total}}
================

T_{\text{load}}
+
T_{\text{CSR}}
+
T_{\text{kernel}}
+
T_{\text{MWUA}}
+
T_{\text{solve}}
+
T_{\text{lift}}.
]

For solver-level comparisons, graph loading may also be reported separately.

The algorithmic pipeline time is:

[
T_{\text{pipeline}}
===================

T_{\text{kernel}}
+
T_{\text{MWUA}}
+
T_{\text{solve}}
+
T_{\text{lift}}.
]

This prevents preprocessing cost from being hidden.

---

# 19. Empty-Core Behaviour

If:

```python
core_vertices == 0
```

then the explicit residual graph has been completely eliminated by the reduction suite.

The pipeline should not run MWUA.

Conceptually:

```python
if core_n == 0:
    skip_mwua()
    reconstruct_solution()
```

For the current collaboration-network experiments:

```text
ca-HepPh   -> empty core
ca-HepTh   -> empty core
ca-AstroPh -> empty core
```

MWUA has no residual vertices to score.

This creates a natural computational gate:

```text
Kernelization
    |
    v
Is core empty?
    |
    +-- YES --> Skip feature computation
    |
    +-- NO ---> Run MWUA
```

---

# 20. Why MWUA Is Applied to the Core

The reduction engine and MWUA perform different tasks.

KaMIS reductions answer:

> Can this graph structure be transformed or resolved using a proven reduction rule?

MWUA answers:

> Among the unresolved variables, which vertices exhibit strong optimization pressure or certainty?

The branching heuristic answers:

> Which unresolved decision should be explored next?

Therefore:

```text
KaMIS reductions
    =
mathematical preprocessing

MWUA
    =
global optimization signal

dynamic local features
    =
current residual-state information

branching policy
    =
search navigation
```

The pipeline is deliberately hierarchical.

```text
Mathematical certainty
        |
        v
Optimization certainty
        |
        v
Local search-state information
        |
        v
Branching decision
```

This is consistent with the dissertation's static-global and dynamic-local feature design.

The MWUA root snapshot remains a static global signal.

The only change is that the root is now the root of the **irreducible residual search instance**, rather than necessarily the complete raw graph.

---

# 21. Current Repository Structure

The relevant project structure is:

```text
Dissertation/
|
|-- CHSZLabLib/
|   |
|   |-- bindings/
|   |   |
|   |   `-- kamis_binding.cpp
|   |
|   |-- external_repositories/
|   |   |
|   |   `-- KaMIS/
|   |       |
|   |       `-- lib/
|   |           |
|   |           `-- mis/
|   |               |
|   |               `-- kernel/
|   |                   |
|   |                   |-- branch_and_reduce_algorithm.h
|   |                   `-- branch_and_reduce_algorithm.cpp
|   |
|   |-- build-kamis/
|   |   |
|   |   `-- _kamis.cpython-313-darwin.so
|   |
|   `-- CMakeLists.txt
|
|-- anytime/
|   |
|   `-- experiments/
|       |
|       |-- reduce_snap_graph.py
|       `-- analyze_redumis_kernelization.py
|
`-- datasets/
    |
    `-- snap/
        |
        |-- ca-HepPh.txt
        |-- ca-HepTh.txt
        |-- ca-AstroPh.txt
        `-- web-Google.txt
```

---

# 22. Files Modified

## `bindings/kamis_binding.cpp`

This file was modified to expose the KaMIS root-level reduction engine.

The new functionality:

```text
redumis_kernel
```

was added alongside:

```text
redumis
online_mis
```

The binding:

1. receives CSR arrays from Python,
2. constructs `graph_access`,
3. converts the graph to adjacency vectors,
4. constructs `branch_and_reduce_algorithm`,
5. calls `reduce_graph()`,
6. extracts the reduced graph using `convert_adj_lists()`,
7. converts the residual graph to CSR,
8. returns the residual CSR arrays and reverse mapping.

The pybind11 module definition was extended with:

```cpp
m.def(
    "redumis_kernel",
    &py_redumis_kernel,
    ...
);
```

---

## `CMakeLists.txt`

The existing KaMIS build target was inspected and reused.

The relevant target is:

```text
kamis_redumis_static
```

No separate copy of the KaMIS source tree was created.

The existing static-library dependency structure was retained.

The `_kamis` module was built as an isolated CMake target using:

```bash
cmake --build build-kamis --target _kamis -j2
```

This avoided building unrelated CHSZLabLib components.

---

## `anytime/experiments/reduce_snap_graph.py`

This experiment was added to evaluate kernelization on SNAP graphs.

Responsibilities:

```text
load SNAP graph
remove self-loops
convert node labels
construct CSR
call redumis_kernel
measure kernel time
report core statistics
inspect reverse mapping
```

---

## `anytime/experiments/analyze_redumis_kernelization.py`

This experiment was designed for systematic synthetic graph analysis.

Graph families include:

```text
Erdős-Rényi
Barabási-Albert
Watts-Strogatz
Random Regular
```

The intended graph sizes are:

```text
1,000
5,000
10,000
```

Target degree regimes are:

```text
2
5
10
20
```

For Erdős-Rényi graphs, edge probability is calculated as:

[
p = \frac{d}{n-1}
]

where:

```text
d = target expected average degree
```

This avoids comparing graphs with drastically different average degrees as (n) increases.

The experiment records:

```text
family
n
target_degree
actual_average_degree
seed
original_vertices
original_edges
core_vertices
core_edges
core_ratio
removed_ratio
edge_core_ratio
reduction_time
```

---

# 23. Current Limitations

## 23.1 Solution Lifting Is Not Yet Exposed

The current binding returns:

```text
core_xadj
core_adjncy
reverse_mapping
```

The internal `branch_and_reduce_algorithm` object is destroyed when the C++ binding function returns.

Therefore, the stored fold and reduction state is lost.

This means the current pipeline can:

```text
extract core
run MWUA
study core branching
```

but cannot yet take an externally generated core MIS and reconstruct the complete original MIS using the same reducer instance.

A stateful pybind11 wrapper is required.

---

## 23.2 NetworkX Loading Overhead

The current SNAP loader uses NetworkX.

For `web-Google`:

```text
Graph loading time : 8.918571 s
CSR conversion     : 13.981826 s
Kernelization      : 0.630341 s
```

The actual KaMIS reduction is significantly faster than the Python graph loading and CSR conversion stages.

This shows that the current bottleneck is not kernelization.

It is the Python/NetworkX ingestion path.

For million-node experiments, the next systems optimization should replace:

```text
SNAP -> NetworkX -> CSR
```

with:

```text
SNAP edge list -> direct CSR
```

A direct parser can:

1. read integer edge pairs,
2. remove self-loops,
3. symmetrize directed input when required,
4. remap sparse node identifiers,
5. count degrees,
6. allocate CSR arrays,
7. populate adjacency entries.

This avoids constructing millions of Python edge and adjacency objects.

---

## 23.3 Directed SNAP Graphs

Maximum Independent Set is defined on the undirected graph used by the solver.

Some SNAP datasets are directed.

The current experiment loads graphs using:

```python
create_using=nx.Graph()
```

which creates an undirected representation.

The methodology must explicitly state the graph conversion policy.

For directed datasets, the current interpretation is:

> An undirected edge ({u,v}) is created when the input contains a connection between (u) and (v), irrespective of direction.

This preprocessing policy should be kept consistent across experiments.

---

## 23.4 Reduction-Level Configuration

The active reduction rules depend on KaMIS's internal static reduction configuration.

The current kernel binding uses the existing `branch_and_reduce_algorithm` behaviour.

For rigorous ablation experiments, future work may expose the reduction level to Python.

For example:

```python
redumis_kernel(
    xadj,
    adjncy,
    vwgt,
    reduction_level=3,
)
```

This would permit experiments such as:

```text
Degree reductions only
vs
Degree + LP
vs
Degree + LP + folding
vs
Full reduction suite
```

Such an experiment could quantify which reduction classes are responsible for the dramatic `web-Google` kernel collapse.

---

# 24. Planned End-to-End Pipeline

The final intended API is:

```python
kernelizer = ReduMISKernelizer(G)

core = kernelizer.reduce()

if core.number_of_nodes() == 0:
    solution = kernelizer.lift_empty_core()

else:
    mwua_features = MWUAFeatureExtractor(
        core
    ).extract()

    solver = BranchAndBoundSolver(
        core,
        branching_strategy=MWUABranching(
            mwua_features
        ),
    )

    core_solution = solver.solve()

    solution = kernelizer.lift(
        core_solution
    )
```

The conceptual solver is:

```text
function solve(G):

    K, reduction_state = kernelize(G)

    if |V(K)| = 0:
        return lift(reduction_state, empty_solution)

    mwua = compute_MWUA(K)

    static_features = MWUA_certainty(mwua)

    core_solution = branch_and_bound(
        K,
        static_features,
        dynamic_local_features
    )

    return lift(
        reduction_state,
        core_solution
    )
```

---

# 25. Research Interpretation

The irreducible-core pipeline changes the role of MWUA.

The original hypothesis was:

> A static MWUA-based global graph representation may replace expensive repeatedly computed LP-relaxation features for learning-guided combinatorial optimization.

The kernelization results suggest a refined architecture:

> Exact root-level kernelization can first remove graph regions handled by established mathematical reductions, after which MWUA can provide a static global optimization signal specifically over the unresolved residual search space.

This creates three computational layers:

```text
Layer 1
Exact structural reductions

Layer 2
Static global MWUA signal

Layer 3
Dynamic local branching features
```

The architecture can be described as:

> **Math-first, global-second, local-dynamic branching.**

The KaMIS reduction suite handles reducible graph structure.

MWUA characterizes the remaining optimization landscape.

Dynamic features describe the current branch-and-bound state.

The branching policy combines these signals to navigate the unresolved search space.

---

# 26. Key Current Result

The strongest current scalability result is:

```text
SNAP web-Google

Original graph:
875,713 vertices
4,322,051 edges

KaMIS fixed-point kernelization:
0.630341 seconds

Residual kernel:
345 vertices
1,414 edges

Explicit vertex reduction:
99.96%

Explicit vertex-domain reduction factor:
approximately 2,538x
```

This result demonstrates that applying MWUA directly to the raw graph is not necessarily the appropriate computational architecture.

For this instance, exact preprocessing reduces the explicit MWUA domain from:

```text
875,713 vertices
```

to:

```text
345 vertices
```

The resulting research direction is therefore:

> **Apply MWUA only after exact root-level kernelization and evaluate its value as a branching signal on the residual kernel where the reduction suite can no longer make progress.**

---

# 27. Next Engineering Steps

The immediate next steps are:

1. Replace the stateless `redumis_kernel` function with a stateful pybind11 kernelizer object.

2. Preserve the `branch_and_reduce_algorithm` instance after reduction.

3. Add a `reduce()` method returning the core CSR representation.

4. Add a `lift(core_solution)` method.

5. Validate the lifted MIS on the original graph.

6. Verify that the lifted solution is independent.

7. Compare the lifted solution objective with KaMIS on small instances.

8. Connect the residual core directly to `MWUAFeatureExtractor`.

9. Run MWUA once at the core root.

10. Compare MWUA certainty, LP certainty, and degree branching on the same residual kernels.

11. Record:

```text
kernelization time
core size
MWUA feature time
LP feature time
branch-and-bound nodes
solve time
total pipeline time
```

12. Replace NetworkX-based SNAP ingestion with direct edge-list-to-CSR conversion for large-scale experiments.

---

# Summary

This project extends the existing CHSZLabLib KaMIS Python binding with direct access to the fixed-point reduction engine used by the KaMIS branch-and-reduce implementation.

The new:

```python
_kamis.redumis_kernel(...)
```

binding extracts the residual kernel without invoking the ReduMIS evolutionary algorithm or the KaMIS branching solver.

The implementation directly calls:

```cpp
branch_and_reduce_algorithm::reduce_graph()
```

and extracts the residual graph using:

```cpp
convert_adj_lists(...)
```

The resulting core is returned to Python in CSR form together with a core-to-original reverse mapping.

Synthetic experiments demonstrated that kernelization effectiveness is strongly structure-dependent.

Real SNAP collaboration graphs were completely reduced.

Most significantly, the SNAP `web-Google` graph was reduced from:

```text
875,713 vertices
```

to:

```text
345 vertices
```

in approximately:

```text
0.63 seconds
```

removing:

```text
99.96%
```

of the vertices from the explicit residual search graph.

The resulting dissertation pipeline is:

```text
Raw Graph
    |
    v
KaMIS Fixed-Point Kernelization
    |
    v
Irreducible Core
    |
    v
MWUA Root Snapshot
    |
    v
Static Global Certainty
    |
    v
Dynamic Local Features
    |
    v
Certainty-Guided Branching
    |
    v
Core Solution
    |
    v
KaMIS Solution Lifting
    |
    v
Original-Graph MIS Solution
```

The current kernel-extraction stage is operational.

The next critical implementation task is preserving the KaMIS reduction state and exposing solution lifting so that an MIS produced by the dissertation's MWUA-guided core solver can be reconstructed into a valid MIS for the original graph.
