# MIS/MVC Learning-to-Branch Pipeline

This document describes the **Maximum Independent Set (MIS) / Minimum Vertex
Cover (MVC)** pipeline, located in `training/`. It covers the exact
Branch-and-Bound framework, feature generation, XGBoost ranking-model
training, and the anytime/end-to-end evaluation used in the dissertation.

See the [main README](README.md) for the overall project context and
[`README_CodeStructure.md`](README_CodeStructure.md) for a file-by-file
reference.

---

## 1. Problem & Approach

MIS and MVC are complementary: a Maximum Independent Set in graph *G*
corresponds directly to a Minimum Vertex Cover of the complement/kernelized
graph. This pipeline:

1. Reduces the input graph using **KaMIS/ReduMIS** exact kernelization.
2. Formulates the reduced instance as an MVC problem solved with a custom
   **SCIP**-based Branch-and-Bound solver (via `PySCIPOpt`).
3. Computes a **static, one-shot MWUA (Multiplicative Weights Update
   Algorithm)** signal over the reduced covering instance.
4. Combines the MWUA signal with lightweight dynamic (LP/search-state)
   features into a **15-dimensional feature vector** per branching candidate.
5. Trains an **XGBoost Learning-to-Rank** model to imitate **Strong
   Branching** decisions.
6. Deploys the learned ranker as SCIP's branching rule for the first *k*
   Branch-and-Bound levels (depth-bounded), or for the entire search
   ("Full"), after which native SCIP branching resumes.

## 2. Directory Contents (`training/`)

| File | Purpose |
|---|---|
| `solver.py` | Custom Branch-and-Bound solver driver (SCIP integration). |
| `problem.py` | MIS/MVC problem representation. |
| `lp.py` | LP relaxation construction/solving utilities. |
| `branching.py` | Branching-rule interface and candidate selection logic. |
| `features.py` | Per-candidate feature computation (graph, LP, MWUA-derived). |
| `feature_builder.py` | Assembles the 15-dimensional feature vector from raw signals. |
| `mwua.py` | Multiplicative Weights Update Algorithm oracle (run once per instance). |
| `generate_dimacs_training_data.py` | Runs Strong-Branching-instrumented Branch-and-Bound over the DIMACS corpus, dumping raw branch samples. |
| `dataset_ltb.py` | Learning-to-branch dataset assembly utilities. |
| `dataset_branchrule.py` | Interfaces the sample collector with SCIP's branching-rule plugin API. |
| `branch_sample.py` | Data structure / logic for an individual branch sample (candidates, LP info, Strong Branching gains/scores, expert decision). |
| `dataset_writer.py` | Serialises collected samples to disk. |
| `ranking_dataset_xgb.py` | Converts raw samples into grouped Learning-to-Rank format for XGBoost. |
| `graph_split.py` | Instance-level train/validation/test graph splitting (prevents leakage across a single graph's samples). |
| `train_xgboost_ranker.py` | Trains the XGBoost ranking model. |
| `xgb_branching.py` | Learned branching policy, **bounded-depth** intervention. |
| `xgb_branching_full.py` | Learned branching policy, **unrestricted** ("Full") intervention. |
| `solver_xgb.py` | Solver configuration wired to the bounded-depth learned branching rule. |
| `solver_xgb_full.py` | Solver configuration wired to the Full learned branching rule. |
| `solver_scip_default.py` | Native SCIP baseline (default branching, no learned intervention). |
| `run_scip_test.py` | Runs the native SCIP baseline over the evaluation benchmark set. |
| `run_xgb_test.py` | Runs the bounded-depth learned configurations. |
| `run_xgb_test_full.py` | Runs the Full learned configuration. |
| `analyse_depth_performance.py` | Compares performance across ML intervention depths. |
| `analyse_feature_importance.py` | Global feature-importance analysis for the trained ranker. |
| `analyse_feature_importance_depth.py` | Feature importance broken down by search depth. |
| `analyse_feature_decay.py` | Studies how feature usefulness changes/decays over search depth. |
| `analyse_mwua_depth.py` | Studies the effect of MWUA iteration count on downstream branching quality. |
| `plot_anytime_results.py` | Generates anytime (incumbent-quality-vs-time) plots. |
| `plot_feature_importance_heatmap.py` | Generates feature-importance heatmap figures. |

Shared dependency:

```
CHSZLabLib/build-kamis/_kamis      # KaMIS/ReduMIS reduction interface, reduction setting 3
```

## 3. Data / Benchmarks

| Corpus | Role | Notes |
|---|---|---|
| **DIMACS** (2nd DIMACS Implementation Challenge) | Training / model development | 56 source instances across 12 graph families. Maximum-Clique-form instances are converted to MIS via graph complementation, then kernelized. 48 of 56 graphs contribute to the final processed ranking dataset after filtering. |
| **BHOSLIB** | Generalisation evaluation (dense, out-of-distribution) | Never used in training. |
| **SNAP, DIMACS10** | Generalisation evaluation (large scale) | Up to ~1.8M vertices; never used in training. |

Strong-Branching-instrumented samples are collected from the **first 500
Branch-and-Bound nodes at which Strong Branching is executed**, per
benchmark instance, to keep data-generation cost tractable.

## 4. Environment Setup

```bash
conda activate dissertation
cd ~/dissertation_ashwin/training
```

Required software (see main README §5 for exact pinned versions):
- Python 3.10.20
- PySCIPOpt 6.2.1 (requires a matching SCIP Optimization Suite install)
- XGBoost 2.1.4
- NumPy 2.2.6, SciPy 1.15.2
- `CHSZLabLib/build-kamis/_kamis` built and available on `PATH`/importable

## 5. Reproduction: Step by Step

### 5.1 Generate training data (Strong Branching imitation)

```bash
python generate_dimacs_training_data.py
```
Runs the instrumented Branch-and-Bound solver over the 56 DIMACS instances,
computing Strong Branching scores `S(v_i) = min(Δ↓_i, Δ↑_i) + 0.1·max(Δ↓_i, Δ↑_i)`
for each candidate at sampled nodes, and serialising complete per-node
candidate evaluations (not just the selected variable) via `dataset_writer.py`.

### 5.2 Build the ranking dataset

```bash
python ranking_dataset_xgb.py
```
Converts raw branch samples into grouped Learning-to-Rank format (one group
per Branch-and-Bound node, candidates within the group compared against each
other). `graph_split.py` performs the train/validation/test split **at the
graph level** to avoid leaking correlated samples from the same graph across
splits.

### 5.3 Train the XGBoost ranker

```bash
python train_xgboost_ranker.py
```
Trains the Learning-to-Rank model on the 15-dimensional candidate feature
vectors. The `max_depth` hyperparameter was swept during development
(depths 2–10); `max_depth = 6` was selected as the best trade-off between
Top-1 accuracy and Hit@5 (see dissertation §5.2 for the full sweep).

### 5.4 Evaluate

```bash
python run_scip_test.py       # native SCIP baseline
python run_xgb_test.py        # bounded-depth learned branching (e.g. D1, D2, D5, D10)
python run_xgb_test_full.py   # unrestricted ("Full") learned branching
```

Bounded configurations invoke `xgb_branching.py` (via `solver_xgb.py`); the
Full configuration invokes `xgb_branching_full.py` (via `solver_xgb_full.py`).
In bounded mode, once the specified depth is reached, control reverts to
SCIP's native branching rule for the remainder of the search.

### 5.5 Analyse and plot results

```bash
python analyse_depth_performance.py
python analyse_feature_importance.py
python analyse_feature_importance_depth.py
python analyse_feature_decay.py
python analyse_mwua_depth.py
python plot_anytime_results.py
python plot_feature_importance_heatmap.py
```

## 6. Evaluation Protocol & Time Limits

| Experiment | Instances | Time limit |
|---|---|---|
| Small-scale checks | — | 120 s |
| BHOSLIB anytime / depth comparison | 20 BHOSLIB instances, ML depths 1–10 + Full | 1,200 s |
| Large-scale SNAP/DIMACS10 anytime | 10 instances, ML depths 2, 5, Full | 12,000 s |
| Extended-budget optimality proving | 20 instances (SNAP/DIMACS10/BHOSLIB), depths 2, 5, Full | up to 12,000 s |

SCIP configuration: default presolving, primal heuristics, cutting planes, LP
processing, and node selection are **unchanged**. Only the branching-variable
selection is modified; the custom XGBoost branching rule is assigned the
highest branching priority and `maxbounddist = 1.0` (learned rule eligible at
all nodes, subject to the depth bound).

## 7. Headline Results

- ML intervention to **depth 2**: median **25.9%** reduction in explored
  nodes, **11.9%** reduction in solve time; improves on **83.0%** / **72.0%**
  of instances respectively.
- **Depth 5** attains the highest exact-optimality rate among bounded
  configurations.
- Models trained only on small DIMACS graphs **generalise** to dense BHOSLIB
  instances and to SNAP/DIMACS10 graphs up to ~1.8M vertices.
- Extending learned control beyond the optimal depth (toward "Full")
  **degrades** performance — broader intervention is not monotonically
  better.
- Downstream branching quality is **non-monotonic** in MWUA iteration
  count: sharp improvement up to ~5,000 iterations, then degradation, even
  though the MWUA fractional solution keeps converging toward the true LP
  optimum at higher iteration counts.

## 8. Notes for Reuse

- The MWUA oracle (`mwua.py`) and feature builder (`feature_builder.py`) are
  reusable for any problem expressible as a **covering formulation** — this
  is exactly what makes the same static signal usable for both MVC and
  Hitting Set (see [`README_HittingSet.md`](README_HittingSet.md)).
- `branching.py` / `dataset_branchrule.py` show the general pattern for
  hooking a Learning-to-Rank model into a SCIP branching-rule plugin; this
  can be adapted to other SCIP-based exact solvers.
- To retrain on a new graph corpus, replace the instance list consumed by
  `generate_dimacs_training_data.py`, keep the graph-level split logic in
  `graph_split.py`, and re-run steps 5.1–5.3.
