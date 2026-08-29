# Hitting Set Learning-to-Branch Pipeline

This document describes the **Hitting Set (HS)** pipeline, combining the
Python-based feature-generation code in `HIT/` with the modified **PACE
2025-winning UZL solver** (`PACE2025/`), which embeds a learned XGBoost
branching policy inside its **EvalMaxSAT / CaDiCaL** component.

See the [main README](README.md) for overall project context and
[`README_CodeStructure.md`](README_CodeStructure.md) for a file-by-file
reference.

---

## 1. Problem & Approach

Hitting Set is solved via the PACE 2025 Exact Track-winning **UZL** solver,
which routes (reduced) instances through **EvalMaxSAT**, which in turn calls
**CaDiCaL** as its underlying SAT engine. CaDiCaL's native branching heuristic
is activity-based (EVSIDS-style) and is most informative once conflict
information has accumulated — i.e. it is weaker early in search.

This pipeline:

1. Applies the same **once-per-instance MWUA oracle** used for MVC (shared
   covering formulation) to the reduced Hitting Set instance, producing a
   static global signal (fractional solution, certainty, constraint weights).
2. Combines this static signal with **hypergraph-structural features** and
   **live CDCL/CaDiCaL dynamic signals** into a **19-dimensional** feature
   vector per candidate variable (12 static + 7 dynamic).
3. Instruments native CaDiCaL decisions (inside EvalMaxSAT) on the 100 public
   PACE 2025 instances to build imitation-learning supervision.
4. Trains an **XGBoost Learning-to-Rank** model (`rank:ndcg`, NDCG@1 — the
   solver only needs the single top-ranked candidate at each decision).
5. **Exports** the trained model to a native C++ header (`ml_model.hpp`) so
   inference happens inside CaDiCaL's decision loop without any Python/XGBoost
   runtime dependency at solve time.
6. Recompiles UZL with the embedded model and evaluates **native CaDiCaL** vs.
   **ML-guided CaDiCaL** (bounded to the first *k* decision layers: D2, D5,
   D10) on the 100-instance PACE 2025 **private** test set.

## 2. Directory Contents

### `HIT/` — feature generation
| File | Purpose |
|---|---|
| `benchmark_uzl.py` | Benchmarks solver configurations (native vs. ML-guided) over instance sets. |
| `generate_instance_features.py` | Computes instance-level (whole-hypergraph) features. |
| `generate_vertex_features.py` | Computes per-vertex candidate features (static + MWUA-derived), written to CSV for consumption by the solver at runtime. |
| `solver.py`, `problem.py`, `lp.py`, `branching.py` | Hitting Set analogues of the MIS/MVC solver components (problem representation, LP relaxation, branching interface). |
| `features.py` | Per-candidate feature computation. |
| `mwua.py` | MWUA oracle (shared design with `training/mwua.py`), run once per reduced instance. |

### `PACE2025/` — modified solver, dataset construction, ML integration
| File | Purpose |
|---|---|
| `build_ranking_dataset.py` | Builds the grouped Learning-to-Rank dataset from collected CaDiCaL decision samples. |
| `build_eval_ranking_dataset.py` | Builds the held-out/evaluation ranking dataset. |
| `train_xgb_ranker.py` | Trains the main XGBoost ranking model (`rank:ndcg`, NDCG@1). |
| `train_xgb_depth.py` | Trains depth-specific ranking model variants. |
| `train_xgb_ablation.py` | Trains ablation model variants (e.g. feature-subset studies). |
| `feature_importance.py`, `feature_importance_depth2.py` | Feature-importance analysis (overall, and for the D2 configuration). |
| `run_one_config.sh` | Runs the benchmark for a single solver configuration. |
| `run_all_experiments.sh` | Runs the full experiment matrix (native + D1/D2/D5/D10). |
| `run_test_benchmark.sh` | Runs a quick smoke-test benchmark. |
| `src/formula.rs` | Rust: modified UZL formula-handling code. |
| `src/solve.rs` | Rust: modified UZL solve-loop / routing code. |
| `native/EvalMaxSAT/lib/EvalMaxSAT/src/EvalMaxSAT.h` | EvalMaxSAT core header (modified for ML-branch integration). |
| `native/.../Formula.cpp`, `Formula.hpp` | EvalMaxSAT formula representation (modified). |
| `native/.../cadicalinterface.h` | Interface layer between EvalMaxSAT and CaDiCaL (modified). |
| `native/.../cadical/src/decide.cpp` | CaDiCaL's variable-decision entry point (modified to consult the learned model). |
| `native/.../cadical/src/internal.hpp`, `options.hpp` | CaDiCaL internals/options (extended with ML-branch config, e.g. `ML_BRANCH`, `ML_DEPTH`). |
| `native/.../cadical/src/mlbranch.cpp` | Core ML-branching logic: collects native decisions during data generation, and invokes the exported model at inference time. |
| `native/.../cadical/src/ml_model.hpp` | Generated C++ header embedding the trained XGBoost model (tree structure, thresholds) for the primary configuration. |
| `native/.../cadical/src/ml_model_D2.hpp` | Separate exported model preserved for the D2 depth configuration. |

### `/tmp` development scripts (preserved for reproducibility)
Used during the Hitting Set experiments for dataset construction, evaluation,
depth-specific analysis, feature analysis, and model export:
`build_reduced_ranking_dataset.py`, `create_reduced_ranking_split.py`,
`generate_cadical_reduced_features.py`, `generate_private_cadical_reduced_features.py`,
`train_reduced_xgb_ranker.py`, `train_eval_depths.py`, `train_eval_depth2.py`,
`train_eval_depth3.py`, `train_save_d4.py`, `evaluate_reduced_xgb.py`,
`evaluate_d2_by_depth.py`, `d2_metrics_by_depth.py`, `d2_feature_importance.py`,
`d2_feature_importance_by_depth.py`, `compare_all_cadical_features.py`,
`compare_32_cadical_features.py`, `compare_reduced_features.py`,
`analyze_feature_difference.py`, `analyze_mismatch_vs_performance.py`,
`per_instance_ranking.py`, `generate_ml_model.py`.

> These scripts reflect the original `/tmp/` development environment; the
> repository copies function identically at their new paths. If reproducing
> from scratch, either restore them to `/tmp/` or update path references
> inside them to your working directory.

## 3. Feature Representation

19-dimensional candidate representation:

```
f_HS(v, N) = [f_static(v), ℓ, |T|, P, C, log(1 + stab_v), a_v, L_v] ∈ R^19
```

- **12 static features** (`f_static`) — hypergraph-structural and MWUA-derived
  (fractional solution, certainty, normalised weights), computed **once**
  before search.
- **7 dynamic CaDiCaL features** (ℓ, |T|, P, C, log-stability, activity,
  learned-clause participation) — cheap, computed live from the current
  CDCL search state.

## 4. Data / Benchmarks

| Item | Value |
|---|---|
| Source corpus | 100 public PACE 2025 Exact Track Hitting Set instances |
| Instances reaching CaDiCaL (instrumentable) | 32 (only instances routed through `EvalMaxSAT → CaDiCaL` generate branching samples) |
| Sampling limit | up to 25,000 decision groups per instance |
| Candidates per group | 32 |
| Final ranking groups | 764,432 |
| Final candidate observations | 24,461,824 |
| Train / validation / test split | 24 / 4 / 4 **instances** (random seed 42) — split at the instance level, never by row |
| Held-out end-to-end evaluation set | 100 **private** PACE 2025 Exact Track test instances |

**Why instance-level splitting?** Candidate rows from the same instance are
strongly correlated (same hypergraph structure appears repeatedly across
search states). A row-wise split would leak structural information about
test instances into training, producing an overly optimistic generalisation
estimate.

## 5. XGBoost Configuration (Hitting Set)

| Parameter | Value |
|---|---|
| Objective | `rank:ndcg` |
| Evaluation metric | NDCG@1 |
| Tree method | Histogram |
| Maximum tree depth | 2 (swept over 2/3/6/10 during development) |
| Learning rate | 0.1 |
| Estimators | 300 |
| Early stopping | 10 rounds (on the 4 validation instances) |
| Minimum child weight | 10 |
| Subsample | 0.8 |
| Column subsampling | 0.8 |
| L2 regularisation | 1.0 |
| Random seed | 42 |
| Threads | 16 |

`rank:ndcg`/NDCG@1 is used (rather than pairwise ranking as in MIS) because
the deployed solver only needs the single top-ranked candidate at each
decision point, and supervision is a one-positive-per-group target (the
native CaDiCaL decision) rather than a graded relevance signal.

Each group is checked at data-construction time to contain **exactly 32
candidates and exactly one positive label**.

## 6. Environment Setup

```bash
conda activate dissertation
cd ~/dissertation_ashwin/PACE2025
```

Required software (see main README §5 for exact pinned Python versions):
- Python 3.10.20, XGBoost 2.1.4, NumPy 2.2.6, SciPy 1.15.2
- **Rust** stable toolchain (`cargo`) — for the UZL solver
- **C++17** toolchain — for EvalMaxSAT/CaDiCaL
- `CHSZLabLib/build-kamis/_kamis` (KaMIS/ReduMIS) — for reductions applied upstream by `HIT/`

## 7. Reproduction: Step by Step

### 7.1 Build the baseline (native) solver

```bash
cargo build --release
```

### 7.2 Generate features (HIT/)

```bash
cd ~/dissertation_ashwin/HIT
python generate_instance_features.py
python generate_vertex_features.py
```
This applies the MWUA oracle once per reduced instance and writes vertex-level
feature CSVs (e.g. `vertex_features_cadical_private_reduced.csv`) consumed by
the solver at runtime via the `VERTEX_FEATURES` environment variable.

### 7.3 Collect native CaDiCaL decisions & build the ranking dataset

```bash
cd ~/dissertation_ashwin/PACE2025
python build_ranking_dataset.py
```
Native decisions are collected via `mlbranch.cpp` (compiled without
`ML_BRANCH` active, so CaDiCaL runs its normal EVSIDS-based selection while
recording candidate sets and the chosen variable at each decision).

### 7.4 Train the ranking model

```bash
python train_xgb_ranker.py
```
Uses the configuration in §5. Depth-specific variants can be produced with
`train_xgb_depth.py`, and ablations with `train_xgb_ablation.py`. The `/tmp`
scripts `train_eval_depths.py`, `train_eval_depth2.py`, `train_eval_depth3.py`,
`train_save_d4.py` were used for additional per-depth model variants (e.g.
`ml_model_D2.hpp`).

### 7.5 Export the model to C++ and recompile

```bash
python generate_ml_model.py
cargo build --release
```
`generate_ml_model.py` converts the trained XGBoost model from its JSON
representation into a native C++ predictor (`ml_model.hpp` / `ml_model_D2.hpp`),
embedding only the trees retained up to the model's `best_iteration`. This
avoids any Python/XGBoost runtime dependency inside the exact search loop.
Recompiling with `cargo build --release` links the updated model into the
`uzl_hs` binary.

### 7.6 Run the benchmark

A representative single-instance invocation:

```bash
cd ~/dissertation_ashwin/PACE2025

INSTANCE=private_exact_014 \
VERTEX_FEATURES="$HOME/dissertation_ashwin/hit/data/vertex_features_cadical_private_reduced.csv" \
ML_BRANCH=1 \
/usr/bin/time -f "\nTIME=%e\nCPU=%P\nMAXRSS=%M KB" \
timeout --signal=TERM --kill-after=25s 1800s \
./target/release/uzl_hs \
< /tmp/private_exact_014_reduced.hgr \
> /tmp/private_exact_014_d2_ml.log 2>&1
```

- `ML_BRANCH=1` activates the learned branching path in `mlbranch.cpp`
  (set `ML_BRANCH=0` / unset for the native-CaDiCaL baseline).
- `ML_DEPTH` (set inside the runner scripts, e.g. `run_private_d5_ml.sh`)
  controls how many **initial decision layers** are controlled by the
  learned policy before control reverts to native CaDiCaL.
- `timeout ... 1800s` applies a **1,800-second** wall-clock limit per
  instance, with a 25s grace period before `SIGKILL`.
- `/usr/bin/time` records wall-clock time, CPU utilisation, and peak memory.

Run the full 100-instance benchmark via the shell runners:

```bash
bash run_all_experiments.sh    # or run_one_config.sh for a single configuration
bash run_test_benchmark.sh     # quick smoke test
```

New depth configurations are typically derived from an existing runner, e.g.
creating a D10 runner from D5:

```bash
cp /tmp/run_private_d5_ml.sh /tmp/run_private_d10_ml.sh
sed -i 's/ML_DEPTH = "5"/ML_DEPTH = "10"/'                       /tmp/run_private_d10_ml.sh
sed -i 's/private_d5_ml_results/private_d10_ml_results/'         /tmp/run_private_d10_ml.sh
sed -i 's/"D5_ML"/"D10_ML"/'                                      /tmp/run_private_d10_ml.sh
rm -f ~/dissertation_ashwin/private_d10_ml_results.csv
/tmp/run_private_d10_ml.sh
```

### 7.7 Result files

| File | Configuration |
|---|---|
| `private_cadical_results.csv` | Native CaDiCaL baseline |
| `private_d2_ml_results.csv` | ML-guided branching, first 2 decision layers |
| `private_d5_ml_results.csv` | ML-guided branching, first 5 decision layers |
| `private_d10_ml_results.csv` | ML-guided branching, first 10 decision layers |

All configurations use the **same** reduced instances and the **same**
1,800-second time limit, isolating the effect of replacing native CaDiCaL
variable selection with the learned XGBoost ranking policy for a given
initial decision depth. After the configured depth is reached, control
**always** returns to native CaDiCaL branching.

## 8. Full Pipeline Summary

```
PACE instances
      ↓
HIT feature and MWUA generation
      ↓
Native CaDiCaL decision collection (mlbranch.cpp, ML_BRANCH inactive)
      ↓
Grouped ranking dataset construction (build_ranking_dataset.py)
      ↓
Instance-level train/validation/test split (24/4/4, seed 42)
      ↓
XGBoost ranking-model training (rank:ndcg, NDCG@1)
      ↓
C++ model export to ml_model.hpp (generate_ml_model.py)
      ↓
Recompile UZL with embedded model (cargo build --release)
      ↓
Native and ML-guided evaluation on all 100 private instances
```

## 9. Headline Results

- Native CaDiCaL baseline: **75/100** private PACE 2025 instances solved
  within 1,800s.
- **D2** (learned control for first 2 decision layers): **76/100** instances
  solved, **23.1%** median runtime reduction vs. native.
- Extending ML control further (D5, D10) **degrades** overall performance —
  consistent with the MIS/MVC finding that broader intervention is not
  necessarily better.

## 10. Notes for Reuse

- `mwua.py` in `HIT/` mirrors the design of `training/mwua.py` — both operate
  on the shared covering-formulation abstraction, which is what allows the
  *same* static signal design to serve two different exact solvers.
- The `ML_BRANCH` / `ML_DEPTH` environment-variable pattern in `mlbranch.cpp`
  is a convenient template for adding bounded-depth learned intervention to
  any CDCL-style solver without permanently altering its native decision
  path.
- To add a new depth configuration, follow the runner-derivation pattern in
  §7.6 rather than writing a new shell script from scratch.
- If retargeting to a different CaDiCaL-based solver, the integration points
  to replicate are: `decide.cpp` (call site), `options.hpp` (config flags),
  and `mlbranch.cpp` (feature assembly + model inference).
