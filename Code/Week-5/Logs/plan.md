# Week 5 Plan — Dataset Design and SCIP Exploration for MWUA LTB

## Objective

Establish a rigorous dataset generation and benchmarking pipeline for studying whether a **static global MWUA snapshot**, combined with inexpensive local information, can guide branching decisions in Branch-and-Bound for:

* Maximum Independent Set (MIS)
* Minimum Vertex Cover (MVC)

The primary goal this week is **not model training**. Instead, the focus is identifying graph distributions where branching decisions are non-trivial and where learning-based branching may provide value.

---

# Motivation

A learning-to-branch policy requires supervision from meaningful branch-and-bound search trees.

Many graph instances are unsuitable because:

* They are solved during preprocessing.
* They produce very shallow search trees.
* They contain little useful branching information.

Before constructing a learning dataset, immediate next plan is to determine:

1. Which graph families generate meaningful branching behavior.
2. Which parameter regimes are genuinely difficult.
3. Whether SCIP exposes sufficient information for future teacher-label generation.

---

# Dataset Strategy

The graph collection will be organized into three tiers.

---

## Tier 1 — Synthetic Training Graphs

### Purpose

Primary source of training instances.

Advantages:

* Controlled structure.
* Easy parameter sweeps.
* Allows systematic hardness analysis.

### Graph Families

#### Erdős-Rényi

G(n,p)

Parameters:

* n = 50–300
* p varied across difficulty regimes

Purpose:

* Baseline random graph family.
* Used to identify phase transition regions.

---

#### Barabási-Albert

Parameters:

* n = 50–300
* m = 2–5

Purpose:

* Scale-free topology.
* Hub-dominated structure.

---

#### Random Regular Graphs

Parameters:

* n = 50–300
* d = 3–6

Purpose:

* Degree information becomes nearly constant.
* Tests whether MWUA and global structural features remain informative when local degree heuristics fail.

---

#### Random Geometric Graphs

Parameters:

* n = 50–300
* radius varied

Purpose:

* Strong local clustering.
* Spatial graph structure.

---

#### Watts-Strogatz Small-World Graphs

Parameters:

* n = 50–300
* rewiring probability varied

Purpose:

* Intermediate regime between regular and random graphs.
* Useful for studying interactions between local and global structure.

---

# Tier 2 — Hard Benchmark Evaluation

### Purpose

Evaluate trained policies on established combinatorial optimization benchmarks.

These graphs are not primarily used for training.

### Sources

#### DIMACS Benchmark Suite

Examples:

* brock
* p_hat
* keller

Purpose:

* Standard MIS benchmarking.
* Widely used in exact optimization research.

---

#### BHOSLIB

Examples:

* frb instances

Purpose:

* Specifically designed to generate difficult branch-and-bound behavior.
* Useful for evaluating learned branching policies.

---

# Tier 3 — Real-World Generalization

### Purpose

Evaluate out-of-distribution generalization.

These graphs will not be used for primary training.

### Sources

Examples:

* ca-GrQc
* ca-HepTh
* wiki-Vote

### Methodology

Rather than using full graphs, induced subgraphs or ego-networks will be extracted.

Purpose:

* Assess transfer from synthetic training distributions to real-world networks.

---

# SCIP Exploration

Before generating labels, we must understand where branching occurs.

The focus this week is identifying graph regimes that produce meaningful search trees.

---

## Baseline A — Default SCIP

Purpose:

* Reference solver configuration.
* Measure tree size and runtime under standard settings.

Metrics:

* Runtime
* Nodes explored
* Objective value

---

## Baseline B — Reduced SCIP

Purpose:

* Investigate search behavior with less aggressive preprocessing.

Candidate settings:

* Reduced presolving
* Reduced propagation
* Reduced primal heuristics

Goal:

* Expose branch-and-bound structure when default SCIP solves instances at the root node.

---

## Baseline C — Teacher Investigation

Purpose:

* Determine which supervision source is feasible.

Questions:

* Can SCIP expose branching candidates?
* Can SCIP expose branching decisions?
* Can Full Strong Branching be accessed through PySCIPOpt?
* Are candidate scores available?

This baseline is exploratory only.

No assumptions are made yet regarding the final teacher mechanism.

---

# Hardness Mapping Experiments

The first objective is identifying graph regimes that generate substantial branching activity.

---

## Erdős-Rényi Sweep

Parameters:

* n = 100
* n = 150

Density sweep:

p ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}

Record:

* Runtime
* Nodes explored
* Objective value

Goal:

* Identify phase transition region.

---

## Random Regular Sweep

Parameters:

* n = 100
* d ∈ {3,4,5,6}

Record:

* Runtime
* Nodes explored

Goal:

* Determine whether regular structure produces deeper search trees.

---

## Additional Family Sweeps

Repeat similar experiments for:

* Barabási-Albert
* Random Geometric
* Watts-Strogatz

---

# Graph Statistics Logging

For every generated graph, store:

* Graph family
* Number of vertices
* Number of edges
* Density
* Average degree
* Maximum degree
* Clustering coefficient
* Generation parameters

These statistics will later support analysis of branching difficulty.

---

# Weekly Tasks

## Graph Generation

* [ ] Implement Erdős-Rényi generator
* [ ] Implement Barabási-Albert generator
* [ ] Implement Random Regular generator
* [ ] Implement Random Geometric generator
* [ ] Implement Watts-Strogatz generator

---

## SCIP Benchmarking

* [ ] Build PySCIPOpt benchmarking pipeline
* [ ] Run Erdős-Rényi parameter sweep
* [ ] Run Random Regular sweep
* [ ] Run Barabási-Albert sweep
* [ ] Run Random Geometric sweep
* [ ] Run Watts-Strogatz sweep

---

## Benchmark Preparation

* [ ] Download DIMACS instances
* [ ] Download BHOSLIB instances
* [ ] Build graph loaders
* [ ] Build SNAP subgraph sampling utility

---

## Teacher Investigation

* [ ] Investigate SCIP branching callbacks
* [ ] Investigate candidate-variable access
* [ ] Investigate Full Strong Branching availability
* [ ] Determine feasible teacher-label mechanism

---

# Success Criteria

This week is considered successful if:

1. At least one graph family exhibits substantial variation in search-tree size as parameters change.
2. Non-trivial branch-and-bound trees are observed.
3. Fractional LP states are observed during search.
4. Multiple branching candidates exist at search nodes.
5. Candidate hard graph regimes are identified for future dataset generation.
6. A feasible teacher-label extraction pathway within SCIP is identified.

---

# Deliverables

By the end of the week:

* Graph generation framework
* SCIP benchmarking framework
* Hardness maps for multiple graph families
* Benchmark graph loaders
* Preliminary understanding of SCIP teacher extraction
* Candidate graph distributions for future learning-to-branch dataset generation
