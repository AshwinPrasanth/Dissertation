# KaMIS Irreducible-Core Pipeline for MWUA-Guided Branching

This repository contains the experimental implementation for the MSc dissertation:

> **Combining Static Global and Dynamic Local Features for Fast Learning-to-Branch Heuristics**

The project investigates whether a static Multiplicative Weights Update Algorithm (MWUA) signal can guide branching in exact combinatorial optimization without repeatedly computing expensive LP-relaxation features.

For large graphs, the current pipeline first applies the exact reduction suite from the KaMIS branch-and-reduce framework. MWUA is then computed only on the residual graph that remains after the reduction rules reach a fixed point.

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
Static MWUA Certainty
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
Original-Graph MIS Solution
```

The central idea is to avoid computing MWUA features for vertices that can already be handled by exact mathematical reductions.

---

## Motivation

Earlier experiments in this project showed that branching trajectory has a significant effect on exact Maximum Independent Set and Minimum Vertex Cover search.

The investigated branching signals include:

* MWUA certainty
* LP certainty
* residual degree
* pseudo-cost
* persistence
* local gain

MWUA was particularly promising as a root-level global signal.

However, applying MWUA directly to graphs containing hundreds of thousands or millions of vertices creates unnecessary memory and computational overhead.

The revised approach is therefore:

```text
Math first -> Global optimization signal -> Dynamic local search
```

KaMIS reductions first remove or transform mathematically resolvable graph structure.

MWUA is then applied only to the unresolved residual search graph.

---

## Irreducible-Core Kernelization

The kernelization pipeline directly uses:

```cpp
branch_and_reduce_algorithm
```

from:

```text
external_repositories/KaMIS/lib/mis/kernel/
```

The relevant implementation files are:

```text
branch_and_reduce_algorithm.h
branch_and_reduce_algorithm.cpp
```

The kernel extractor directly calls:

```cpp
reducer.reduce_graph();
```

The complete ReduMIS evolutionary search is not executed.

The following are not called by the kernel extraction pipeline:

```text
perform_mis_search()
perform_evolutionary()
solve()
rec()
branching()
```

The purpose of the new binding is only to apply the KaMIS reduction suite and extract the residual graph.

---

## Reduction Process

The KaMIS branch-and-reduce implementation contains multiple reduction rules, including:

* Degree-1 reduction
* Dominance reduction
* Unconfined reduction
* LP reduction
* Packing reduction
* Degree-2 folding
* Twin reduction
* Funnel reduction
* Desk reduction

The rules are applied repeatedly.

Conceptually:

```text
repeat
    apply enabled reduction rules
until the graph no longer changes
```

A reduction can expose another reducible structure.

For example:

```text
Vertex reduction
        |
        v
Neighbour degree changes
        |
        v
Degree-2 structure appears
        |
        v
Folding
        |
        v
Dominance structure appears
        |
        v
Dominance reduction
```

The graph returned after this process is the residual kernel.

In this project, the terms **irreducible core**, **residual core**, and **residual kernel** refer to this graph.

More precisely, it is:

> The residual kernel after fixed-point application of the enabled KaMIS branch-and-reduce reduction suite.

---

## CHSZLabLib Customization

CHSZLabLib already contained Python bindings for KaMIS.

The original `_kamis` module exposed:

```python
redumis
online_mis
```

The KaMIS binding was modified at:

```text
bindings/kamis_binding.cpp
```

A new function was added:

```python
redumis_kernel
```

The resulting module exposes:

```text
online_mis
redumis
redumis_kernel
```

The new binding gives Python direct access to the KaMIS reduction engine.

---

## `redumis_kernel`

The kernel binding receives a graph in CSR format:

```python
xadj
adjncy
vwgt
```

For a graph containing `n` vertices:

```python
len(xadj) == n + 1
```

The neighbours of vertex `v` are stored in:

```python
adjncy[xadj[v]:xadj[v + 1]]
```

For the current unweighted MIS experiments:

```python
vwgt = np.asarray([], dtype=np.int32)
```

The binding performs the following steps:

```text
CSR graph
    |
    v
graph_access
    |
    v
vector<vector<int>> adjacency
    |
    v
branch_and_reduce_algorithm
    |
    v
reduce_graph()
    |
    v
convert_adj_lists()
    |
    v
Residual graph
    |
    v
Residual CSR
```

The Python call is:

```python
core_xadj, core_adjncy, reverse_mapping = (
    _kamis.redumis_kernel(
        xadj,
        adjncy,
        vwgt,
    )
)
```

The number of residual vertices is:

```python
core_n = len(core_xadj) - 1
```

The number of residual edges is:

```python
core_m = len(core_adjncy) // 2
```

---

## Core-to-Original Mapping

The kernel binding also returns:

```python
reverse_mapping
```

The mapping relates compact core identifiers to surviving original graph identifiers.

For example:

```text
reverse_mapping = [0, 2, 3, 5, 8]
```

means:

```text
Core vertex 0 -> Original vertex 0
Core vertex 1 -> Original vertex 2
Core vertex 2 -> Original vertex 3
Core vertex 3 -> Original vertex 5
Core vertex 4 -> Original vertex 8
```

This mapping allows core-level MWUA scores and branching decisions to be associated with the corresponding surviving original vertices.

The reverse mapping is not sufficient for complete MIS solution reconstruction after folding and other graph transformations.

Full reconstruction requires the internal KaMIS reduction state.

---

## Building the KaMIS Binding

A dedicated CMake build directory is used:

```bash
mkdir -p build-kamis
```

Configure the build on macOS using:

```bash
cmake -S . -B build-kamis -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_C_FLAGS="-I/opt/homebrew/opt/libomp/include" \
  -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/libomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/opt/libomp/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/homebrew/opt/libomp/lib"
```

Build only the `_kamis` Python module:

```bash
cmake --build build-kamis --target _kamis -j2
```

The generated module is located at:

```text
build-kamis/_kamis.cpython-313-darwin.so
```

The module can be tested using:

```bash
PYTHONPATH="$(pwd)/build-kamis:$PYTHONPATH" python - <<'PY'
import _kamis

print("KaMIS module loaded")
print(dir(_kamis))
print("Kernel binding:", _kamis.redumis_kernel)
PY
```

Expected functions include:

```text
online_mis
redumis
redumis_kernel
```

---

## Running the SNAP Kernelization Experiment

The SNAP reduction experiment is located at:

```text
anytime/experiments/reduce_snap_graph.py
```

From the dissertation root, run:

```bash
PYTHONPATH="$(pwd)/CHSZLabLib/build-kamis:$PYTHONPATH" \
python anytime/experiments/reduce_snap_graph.py
```

The experiment:

1. loads a SNAP edge-list graph,
2. removes self-loops,
3. creates an undirected graph,
4. relabels vertices to contiguous integer identifiers,
5. converts the graph to CSR,
6. runs KaMIS fixed-point kernelization,
7. reports residual-core statistics.

The reported metrics include:

```text
Original vertices
Original edges
Average degree
Graph load time
CSR conversion time
CSR adjacency size
Core vertices
Core edges
Core ratio
Vertices removed
Edge core ratio
Reduction time
Mapping size
```

---

## Synthetic Validation

The binding was initially validated on Erdős-Rényi graphs containing 1,000 vertices.

| Edge Probability | Original Edges | Core Vertices | Core Ratio | Vertices Removed |
| ---------------- | -------------: | ------------: | ---------: | ---------------: |
| 0.002            |            999 |             0 |      0.000 |           100.0% |
| 0.005            |          2,495 |           747 |      0.747 |            25.3% |
| 0.010            |          4,985 |           992 |      0.992 |             0.8% |

The results show that reduction effectiveness is highly dependent on graph structure.

Sparse instances can exhibit cascading reductions, while denser random graphs can remain largely irreducible.

---

## SNAP Results

### `ca-HepPh`

```text
Original vertices : 12,008
Original edges    : 118,489

Core vertices     : 0
Core edges        : 0

Vertices removed  : 100.00%
Reduction time    : 0.007077 s
```

### `ca-HepTh`

```text
Original vertices : 9,877
Original edges    : 25,973

Core vertices     : 0
Core edges        : 0

Vertices removed  : 100.00%
Reduction time    : 0.006341 s
```

### `ca-AstroPh`

```text
Original vertices : 18,772
Original edges    : 198,050

Core vertices     : 0
Core edges        : 0

Vertices removed  : 100.00%
Reduction time    : 0.010796 s
```

These instances were completely eliminated from the explicit residual search graph by the reduction suite.

---

## `web-Google`

The most significant current result was obtained on the SNAP `web-Google` graph.

Original graph:

```text
Vertices       : 875,713
Edges          : 4,322,051
Average degree : 9.8709
```

CSR representation:

```text
Adjacency entries : 8,644,102
```

KaMIS kernelization:

```text
Core vertices      : 345
Core edges         : 1,414
Core ratio         : 0.000394
Vertices removed   : 99.96%
Edge core ratio    : 0.000327
Reduction time     : 0.630341 s
```

The explicit graph domain was reduced from:

```text
875,713 vertices
```

to:

```text
345 vertices
```

The vertex-domain reduction factor is approximately:

```text
2,538x
```

The complete transformation is:

```text
web-Google
875,713 vertices
4,322,051 edges
        |
        | KaMIS reductions
        | 0.630341 s
        v
345 vertices
1,414 edges
```

Only approximately `0.0394%` of the original vertices remain in the explicit residual graph.

---

## MWUA Integration

The previous experimental architecture was:

```text
Raw Graph
    |
    v
MWUA
    |
    v
Branch-and-Bound
```

The revised architecture is:

```text
Raw Graph
    |
    v
KaMIS Kernelization
    |
    v
Irreducible Core
    |
    v
MWUA Root Snapshot
    |
    v
MWUA Certainty
    |
    v
Dynamic Local Features
    |
    v
Branch-and-Bound
```

For `web-Google`, the MWUA vertex domain changes from:

```text
875,713 vertices
```

to:

```text
345 vertices
```

MWUA is not run on the raw `web-Google` graph solely to obtain a runtime comparison.

The large raw graph creates unnecessary memory and computational pressure.

Instead, exact kernelization is treated as the first stage of the solver.

The research question is:

> Can exact root-level kernelization make MWUA-based feature extraction computationally feasible on large graph instances by restricting MWUA to the unresolved residual search space?

---

## Test-Time Pipeline

Kernelization is performed when an input graph is processed.

A separate offline core-generation stage is not required.

The intended execution is:

```text
Input Graph
    |
    v
Convert to CSR
    |
    v
KaMIS Kernelization
    |
    v
Check Core Size
    |
    +-------------------------+
    |                         |
    | Core Empty              | Core Non-Empty
    |                         |
    v                         v
Skip MWUA                 Run MWUA
                              |
                              v
                         Solve Core
                              |
                              v
                         Lift Solution
```

The algorithmic pipeline runtime is:

[
T_{\mathrm{pipeline}}
=====================

T_{\mathrm{kernel}}
+
T_{\mathrm{MWUA}}
+
T_{\mathrm{solve}}
+
T_{\mathrm{lift}}.
]

Graph loading and CSR construction can be reported separately as system-level input-processing costs.

---

## Empty Cores

If:

```python
core_n == 0
```

there are no residual vertices on which to compute MWUA features.

The pipeline should skip MWUA:

```python
if core_n == 0:
    skip_mwua()
    reconstruct_solution()
```

This occurred for:

```text
ca-HepPh
ca-HepTh
ca-AstroPh
```

The empty core does not imply that every original vertex was independently fixed by a trivial rule.

Some reductions, such as folding, store implicit structural transformations.

The reduction state is required to reconstruct the original MIS assignment.

---

## Current Limitation: Solution Lifting

The current `redumis_kernel` function returns:

```text
core_xadj
core_adjncy
reverse_mapping
```

The internal:

```cpp
branch_and_reduce_algorithm
```

object is destroyed when the binding call returns.

This means that internal information associated with:

```text
folds
modified structures
alternative structures
reduction assignments
```

is lost after core extraction.

The current implementation is therefore suitable for:

```text
kernel-size analysis
MWUA feature experiments
core-level branching experiments
```

but external core solutions cannot yet be fully lifted to the original graph.

The next implementation stage is a stateful pybind11 kernelizer.

The intended API is:

```python
kernelizer = _kamis.ReduMISKernelizer(
    xadj,
    adjncy,
    vwgt,
)

core_xadj, core_adjncy, reverse_mapping = (
    kernelizer.reduce()
)

core_solution = solve_core(
    core_xadj,
    core_adjncy,
)

original_solution = kernelizer.lift(
    core_solution
)
```

The C++ object will preserve:

```cpp
std::unique_ptr<branch_and_reduce_algorithm>
```

between `reduce()` and `lift()`.

KaMIS's existing:

```cpp
extend_finer_is(...)
```

mechanism can then be used to reverse the stored reduction transformations.

---

## Repository Structure

```text
Dissertation/
|
|-- CHSZLabLib/
|   |
|   |-- bindings/
|   |   `-- kamis_binding.cpp
|   |
|   |-- external_repositories/
|   |   `-- KaMIS/
|   |       `-- lib/
|   |           `-- mis/
|   |               `-- kernel/
|   |                   |-- branch_and_reduce_algorithm.h
|   |                   `-- branch_and_reduce_algorithm.cpp
|   |
|   |-- build-kamis/
|   |   `-- _kamis.cpython-313-darwin.so
|   |
|   `-- CMakeLists.txt
|
|-- anytime/
|   `-- experiments/
|       |-- reduce_snap_graph.py
|       `-- analyze_redumis_kernelization.py
|
`-- datasets/
    `-- snap/
        |-- ca-HepPh.txt
        |-- ca-HepTh.txt
        |-- ca-AstroPh.txt
        `-- web-Google.txt
```

---

## Current Research Architecture

The current solver architecture contains three layers.

### Layer 1: Exact Structural Reduction

KaMIS handles graph structures that can be safely resolved or transformed using proven reduction rules.

### Layer 2: Static Global Optimization Signal

MWUA is computed once on the residual core.

The resulting root snapshot provides a static certainty signal for the remaining vertices.

### Layer 3: Dynamic Local Search State

Residual degree, pseudo-cost, and related local features describe the current branch-and-bound state.

The complete architecture is:

```text
Exact Mathematical Reduction
            |
            v
Static Global MWUA Signal
            |
            v
Dynamic Local Features
            |
            v
Branching Decision
```

The design can be summarized as:

> **Math-first, global-second, local-dynamic branching.**

---

## Next Steps

The immediate implementation tasks are:

1. Implement a stateful `ReduMISKernelizer` pybind11 class.
2. Preserve the KaMIS reducer after kernel extraction.
3. Add `reduce()`.
4. Add `lift(core_solution)`.
5. Validate lifted independent sets on the original graph.
6. Compare lifted objective values with KaMIS on small graphs.
7. Connect `core_xadj` and `core_adjncy` directly to the MWUA feature extractor.
8. Compute the MWUA root snapshot on the irreducible core.
9. Compare MWUA certainty, LP certainty, and degree branching.
10. Record kernelization time, feature time, nodes explored, solve time, and total pipeline time.
11. Replace the NetworkX SNAP loader with direct edge-list-to-CSR construction for large graphs.

---

## Key Result

The current pipeline reduced SNAP `web-Google` from:

```text
875,713 vertices
4,322,051 edges
```

to:

```text
345 vertices
1,414 edges
```

using the KaMIS fixed-point reduction suite in approximately:

```text
0.630341 seconds
```

This removed:

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
Core MIS Solution
    |
    v
Solution Lifting
    |
    v
Original-Graph MIS Solution
```

The kernel-extraction stage is operational.

The next critical step is to preserve the KaMIS reduction state and expose solution lifting, allowing an MIS generated by the MWUA-guided core solver to be reconstructed into a valid MIS for the original graph.
