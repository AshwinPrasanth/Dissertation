# Code Structure & Reuse Guide

This document gives a complete map of the repository, explains how the three
components (`training/`, `HIT/`, `PACE2025/`) relate to one another, and
describes how to adapt the pipeline to new problems, solvers, or benchmarks.

See also: [`README.md`](README.md) (project overview), [`README_MIS.md`](README_MIS.md),
[`README_HittingSet.md`](README_HittingSet.md) (per-domain reproduction guides).

---

## 1. Full Repository Tree

```
dissertation_ashwin/
│
├── CHSZLabLib/
│   └── build-kamis/
│       └── _kamis                         KaMIS/ReduMIS reduction interface (shared by both domains)
│
├── training/                              MIS/MVC experiments
│   │
│   ├── solver.py                          Custom Branch-and-Bound driver (SCIP/PySCIPOpt)
│   ├── problem.py                         MIS/MVC problem representation
│   ├── lp.py                              LP relaxation construction/solving
│   ├── branching.py                       Branching-rule interface
│   ├── features.py                        Per-candidate feature computation
│   ├── feature_builder.py                 Assembles 15-dim feature vector
│   ├── mwua.py                            MWUA oracle (once per instance)
│   │
│   ├── generate_dimacs_training_data.py   Strong-Branching-instrumented sample generation
│   ├── dataset_ltb.py                     Learning-to-branch dataset assembly
│   ├── dataset_branchrule.py              SCIP branching-rule plugin interface
│   ├── branch_sample.py                   Per-sample data structure
│   ├── dataset_writer.py                  Serialisation
│   ├── ranking_dataset_xgb.py             Grouped Learning-to-Rank dataset construction
│   ├── graph_split.py                     Instance(graph)-level train/val/test split
│   │
│   ├── train_xgboost_ranker.py            XGBoost LTR training
│   ├── xgb_branching.py                   Learned policy, bounded-depth
│   ├── xgb_branching_full.py              Learned policy, unrestricted ("Full")
│   ├── solver_xgb.py                      Solver config: bounded-depth learned branching
│   ├── solver_xgb_full.py                 Solver config: Full learned branching
│   ├── solver_scip_default.py             Native SCIP baseline
│   │
│   ├── run_scip_test.py                   Evaluation: native SCIP
│   ├── run_xgb_test.py                    Evaluation: bounded-depth learned
│   ├── run_xgb_test_full.py               Evaluation: Full learned
│   │
│   ├── analyse_depth_performance.py       Depth-comparison analysis
│   ├── analyse_feature_importance.py      Global feature importance
│   ├── analyse_feature_importance_depth.py  Feature importance by depth
│   ├── analyse_feature_decay.py           Feature usefulness decay over depth
│   ├── analyse_mwua_depth.py              MWUA iteration-count sensitivity
│   ├── plot_anytime_results.py            Anytime performance plots
│   └── plot_feature_importance_heatmap.py Feature-importance heatmaps
│
├── HIT/                                   Hitting Set feature generation
│   │
│   ├── benchmark_uzl.py                   Benchmark runner (native vs. ML-guided)
│   ├── generate_instance_features.py      Instance-level (hypergraph) features
│   ├── generate_vertex_features.py        Per-vertex candidate features → CSV
│   ├── solver.py                          HS solver driver
│   ├── problem.py                         HS problem representation
│   ├── lp.py                              LP relaxation utilities (HS analogue)
│   ├── branching.py                       Branching interface (HS analogue)
│   ├── features.py                        Per-candidate feature computation
│   └── mwua.py                            MWUA oracle (HS analogue, same design as training/mwua.py)
│
└── PACE2025/                              Modified UZL solver and ML integration
    │
    ├── build_ranking_dataset.py           Grouped ranking dataset construction (training instances)
    ├── build_eval_ranking_dataset.py      Grouped ranking dataset construction (eval instances)
    ├── train_xgb_ranker.py                Main XGBoost ranking model training
    ├── train_xgb_depth.py                 Depth-specific model training
    ├── train_xgb_ablation.py              Ablation-study model training
    ├── feature_importance.py              Feature importance analysis
    ├── feature_importance_depth2.py       Feature importance, D2 configuration
    │
    ├── run_one_config.sh                  Run one benchmark configuration
    ├── run_all_experiments.sh             Run full experiment matrix
    ├── run_test_benchmark.sh              Quick smoke-test benchmark
    │
    ├── src/
    │   ├── formula.rs                     Modified UZL formula handling (Rust)
    │   └── solve.rs                       Modified UZL solve loop / routing (Rust)
    │
    └── native/
        └── EvalMaxSAT/
            └── lib/
                ├── EvalMaxSAT/src/
                │   ├── EvalMaxSAT.h              EvalMaxSAT core (modified)
                │   ├── Formula.cpp                Formula representation (modified)
                │   ├── Formula.hpp
                │   └── cadicalinterface.h         EvalMaxSAT ↔ CaDiCaL interface (modified)
                │
                └── cadical/src/
                    ├── decide.cpp                 Variable-decision entry point (modified)
                    ├── internal.hpp               CaDiCaL internals (extended)
                    ├── options.hpp                Solver options (ML_BRANCH, ML_DEPTH, etc.)
                    ├── mlbranch.cpp                Core ML-branching logic (collection + inference)
                    ├── ml_model.hpp                Exported XGBoost model (primary configuration)
                    └── ml_model_D2.hpp             Exported XGBoost model (D2 configuration)

/tmp development scripts (preserved in-repo; see README_HittingSet.md §2)
│
├── build_reduced_ranking_dataset.py
├── create_reduced_ranking_split.py
├── generate_cadical_reduced_features.py
├── generate_private_cadical_reduced_features.py
├── train_reduced_xgb_ranker.py
├── train_eval_depths.py
├── train_eval_depth2.py
├── train_eval_depth3.py
├── train_save_d4.py
├── evaluate_reduced_xgb.py
├── evaluate_d2_by_depth.py
├── d2_metrics_by_depth.py
├── d2_feature_importance.py
├── d2_feature_importance_by_depth.py
├── compare_all_cadical_features.py
├── compare_32_cadical_features.py
├── compare_reduced_features.py
├── analyze_feature_difference.py
├── analyze_mismatch_vs_performance.py
├── per_instance_ranking.py
└── generate_ml_model.py
```

Key structural difference between the two domains:

| | MIS/MVC (`training/`) | Hitting Set (`PACE2025/` + `HIT/`) |
|---|---|---|
| Host solver | Python, SCIP/PySCIPOpt | Rust (UZL) → C++ (EvalMaxSAT/CaDiCaL) |
| Model inference | In-process Python (XGBoost) | Compiled C++ (model exported to a header) |
| Expert supervision | Strong Branching (explicit LP evaluation) | Native CaDiCaL decision (implicit, one-positive-per-group) |
| Ranking objective | `rank:pairwise`, NDCG@5 | `rank:ndcg`, NDCG@1 |
| Feature dimensionality | 15 | 19 |
| Data partition unit | Graph | Instance |

Both share the identical **MWUA oracle design** (`mwua.py` in each
directory) — this is the load-bearing piece of the shared covering
formulation, and is the natural place to look first when adapting the
approach to a **third** covering-style problem.

## 3. Environment Setup (both domains)

```bash
conda create -n dissertation python=3.10.20
conda activate dissertation

pip install xgboost==2.1.4 numpy==2.2.6 scipy==1.15.2 PySCIPOpt==6.2.1
```

- **SCIP Optimization Suite** must be installed separately and match the
  `PySCIPOpt` version (see PySCIPOpt's own compatibility table).
- **KaMIS/ReduMIS** must be built to produce `CHSZLabLib/build-kamis/_kamis`
  (used by both `training/` and `HIT/`).
- **Rust** (stable toolchain via `rustup`) is required to build `PACE2025/`
  (`cargo build --release`).
- A **C++17**-capable compiler is required for the EvalMaxSAT/CaDiCaL native
  build (invoked as part of `cargo build --release` via its build script, or
  separately if the native library is built standalone).

Recommended: pin the exact environment with a lockfile once recreated, e.g.:

```bash
conda env export --no-builds > environment.yml
```

## 4. General Pipeline Pattern (applies to both domains)

Both pipelines instantiate the same six-stage template:

1. **Reduce** the raw instance (KaMIS/ReduMIS kernelization).
2. **Compute a static global signal once** (MWUA oracle) over the reduced
   instance.
3. **Instrument an expert** to generate branching supervision:
   - MIS/MVC: explicit Strong Branching evaluation at sampled nodes.
   - Hitting Set: passive recording of native CaDiCaL's own decisions.
4. **Build a grouped Learning-to-Rank dataset**, partitioned at the
   **instance/graph level** (never by row) to prevent leakage.
5. **Train an XGBoost ranker** and select hyperparameters (particularly
   `max_depth`) via a held-out validation set / sweep.
6. **Deploy the ranker as a bounded-depth branching rule**: intervene for the
   first *k* decision layers, then hand control back to the solver's native
   strategy. Evaluate both anytime behaviour and end-to-end (time-to-solve /
   nodes-explored) performance against the native baseline, across a range of
   *k* (including an unrestricted "Full" configuration where applicable).

## 5. Adapting This Pipeline to a New Problem or Solver

To reuse this framework for a different exact-search problem:

1. **Formulate the problem as a covering (or otherwise MWUA-amenable)
   instance**, if you want to reuse the static global signal as-is. If not,
   replace `mwua.py`'s oracle with whatever once-computed global signal is
   appropriate for the new problem (the surrounding feature-building and
   ranking pipeline is agnostic to what the static signal actually is).
2. **Implement a feature builder** analogous to `feature_builder.py` /
   `features.py`, combining the static signal with cheap, solver-native
   dynamic features available at each decision point.
3. **Pick or build an expert supervision source**:
   - If your solver supports an expensive-but-informative oracle (like Strong
     Branching), instrument it the way `generate_dimacs_training_data.py`
     does, storing the *complete* per-candidate evaluation (not just the
     selected action) so it can be reused for different training-target
     definitions later.
   - If not, passively record the solver's own native decisions the way
     `mlbranch.cpp` does for CaDiCaL — this is cheaper but yields a weaker,
     one-positive-per-group signal, which typically pushes you toward an
     `rank:ndcg`/NDCG@1 objective rather than pairwise ranking.
4. **Build the grouped ranking dataset** with an instance-level (not
   row-level) split — see `graph_split.py` (MIS) and the seed-42 instance
   shuffle in `PACE2025/` for the two reference implementations.
5. **Train with XGBoost's ranking objectives**, sweeping `max_depth` and
   using early stopping on a held-out validation partition. Use Table 4.2 in
   the dissertation (reproduced in the per-domain READMEs) as a starting
   hyperparameter grid.
6. **Integrate as a bounded-depth branching rule**:
   - If the host solver has a Python plugin API (e.g. SCIP branching rules),
     wire the ranker in directly, following `dataset_branchrule.py` /
     `xgb_branching.py` as a template.
   - If the host solver is compiled (C/C++/Rust) and needs low-latency
     inference, **export the trained model to native code** rather than
     shelling out to a Python runtime — see `generate_ml_model.py` and
     `ml_model.hpp` for the reference conversion (tree structure + thresholds
     embedded directly, truncated to `best_iteration`).
   - In both cases, gate the learned rule behind a **depth counter** so it
     controls only the first *k* decisions before reverting to the solver's
     native strategy — this was found to be essential: unrestricted learned
     control consistently **underperforms** a well-chosen bounded depth in
     both domains studied here.
7. **Evaluate against the native baseline** using matched instances, matched
   time limits, and the same hardware — isolate the branching-rule change as
   the only experimental variable (as done in both `run_*_test*.py` /
   `run_all_experiments.sh` harnesses).

## 6. Practical Gotchas Worth Preserving

- **Never split ranking data by row.** Always split by graph/instance —
  candidate rows from the same instance are correlated, and row-level splits
  produce optimistic generalisation estimates (see §4.2.3 of the
  dissertation).
- **MWUA iteration count is not "more is better."** Downstream branching
  performance improves sharply up to a moderate iteration budget (~5,000 in
  this work), then degrades and fluctuates even as the MWUA solution keeps
  converging numerically. Treat the iteration count as a hyperparameter to
  sweep, not a knob to maximise.
- **Bounded-depth intervention beats unrestricted intervention.** In both
  domains, the best-performing configuration used the learned policy for
  only a handful of initial decision layers. Always evaluate a depth sweep
  (e.g. D1/D2/D5/D10/Full), don't assume "Full" is the strong baseline.
- **Store the complete expert evaluation, not just the chosen action**, when
  the expert supports it (Strong Branching case) — this lets you rebuild
  ranking targets / try alternative scoring functions without re-running the
  expensive expert computation.
- **Keep dataset consistency checks** (e.g. "each group has exactly 32
  candidates and exactly one positive label" in the Hitting Set pipeline) as
  an automated check in the dataset-construction script, not just as a
  documented invariant — the PACE2025 pipeline enforces this directly in
  `build_ranking_dataset.py`.

## 7. Where to Look First When...

| Task | Start here |
|---|---|
| Understanding the static global signal | `training/mwua.py` or `HIT/mwua.py` |
| Understanding the candidate feature vector | `training/feature_builder.py` (15-dim) or the `f_HS` definition in `README_HittingSet.md` §3 (19-dim) |
| Reproducing MIS/MVC training end-to-end | [`README_MIS.md`](README_MIS.md) §5 |
| Reproducing Hitting Set training end-to-end | [`README_HittingSet.md`](README_HittingSet.md) §7 |
| Adding a new ML-controlled depth configuration | `README_HittingSet.md` §7.6 (runner-derivation pattern) |
| Understanding solver/branching integration for a compiled solver | `PACE2025/native/.../cadical/src/mlbranch.cpp`, `decide.cpp` |
| Understanding solver/branching integration for a Python solver | `training/dataset_branchrule.py`, `xgb_branching.py` |
| Re-exporting a retrained model to C++ | `PACE2025/generate_ml_model.py` |
