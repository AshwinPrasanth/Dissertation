# Week 2 — Deferred-Uncertainty Exact Search

## Abstract

Week 2 extends the exact Branch-and-Bound work from Week 1 into a deferred-uncertainty search framework. The solver still preserves exact optimization semantics, but it now prioritizes easy structural decisions first, tracks how certainty persists across depth, and uses lightweight residual-graph statistics to study when the root-level MWUA snapshot remains informative. The goal is not to learn a policy end-to-end, but to build a structurally interpretable exact search procedure that naturally defers ambiguous variables deeper into DFS.

## Slide-ready summary

We upgraded the exact B&B solver into a deferred-uncertainty framework. The solver now combines LP certainty, MWUA certainty, residual degree ratio, pseudo-cost, certainty persistence, and local reduction gain, while keeping the search exact and lightweight. Week 2 also adds stronger residual graph maintenance, shallow diversified incumbent dives, and experiments that measure certainty decay, backbone stability, and residual graph collapse.

## What was built

This folder contains the Week 2 code, experiment harness, and generated outputs for the deferred-uncertainty version of the exact B&B solver.

### Repository layout

- `core.py` — problem definitions, the root MWUA snapshot, and the incremental residual `GraphState`.
- `branching.py` — branching strategies, including the deferred-uncertainty composite rule and the classic baselines.
- `solver.py` — exact B&B driver, reductions, persistence tracking, shallow incumbent dives, and instrumentation.
- `experiments.py` — experiment suite for strategy comparison, backbone/persistence analysis, certainty evolution, scaling, density sweeps, ablations, reductions, and pruning heatmaps.
- `results/` — main experiment outputs produced by the full suite.
- `results_smoke/` — small smoke-test outputs used to validate the Week 2 changes.

### Core idea

Week 2 keeps the exact search semantics from Week 1 but shifts the interpretation:

- variables that are already near 0 or 1 are treated as structurally easy,
- ambiguous variables are deferred deeper into DFS,
- the solver tracks whether certainty persists across depths,
- residual graph features are updated incrementally instead of recomputed from scratch,
- and exact reductions are triggered when certainty and local structure agree.

## Main outcomes

- The residual active graph is now maintained incrementally with only unresolved vertices active.
- Branching was expanded from certainty-first scoring into deferred-uncertainty scoring.
- The solver now tracks certainty persistence, residual graph evolution, and local reduction gain.
- Exact certainty-guided reductions were added without changing optimization semantics.
- The solver can seed early incumbents with a few shallow diversified dives.
- The experiment suite now includes a dedicated deferred-uncertainty analysis path.

## Early results

The completed full-suite blocks already show the intended behavior:

- On the full strategy comparison, `certainty_first` explores far fewer nodes than the uncertainty-heavy baselines and is close to the `degree` baseline in this graph family.
- From the current full `strategy_comparison.csv`, approximate means are:
	- `degree`: mean explored nodes 45.53, mean time 1.165 s, mean prune rate 0.5276
	- `certainty_first`: mean explored nodes 49.67, mean time 1.166 s, mean prune rate 0.5269
	- `mwua_only`: mean explored nodes 52.20, mean time 1.207 s, mean prune rate 0.5272
	- `most_fractional`: mean explored nodes 89.13, mean time 1.380 s, mean prune rate 0.5197
	- `random`: mean explored nodes 91.80, mean time 1.463 s, mean prune rate 0.5195
	- `pseudo_cost`: mean explored nodes 91.53, mean time 1.413 s, mean prune rate 0.5196
- The backbone analysis currently shows MWUA certainty is positively correlated with backbone stability and strongly negatively correlated with fractional frequency:
	- MWUA vs backbone stability: `r = 0.398`
	- MWUA vs fractional frequency: `r = -0.826`
- The deferred-uncertainty smoke and full analysis both show the expected residual collapse pattern: active vertices and residual edge density decrease with depth while certainty and persistence increase.

## Design choices

- The solver remains exact branch-and-bound rather than heuristic search.
- No DRL, PPO, GraphSAGE, repair loops, or rollback-state machinery were added.
- MWUA still runs once at the root and serves as a static structural snapshot.
- The branching rule remains interpretable: it is a weighted composite of LP certainty, MWUA certainty, residual graph signals, pseudo-cost, persistence, and local gain.
- Graph state updates are incremental and focused on the unresolved subgraph.

## Week 2 analysis questions

The new experiments are designed to answer:

- How long does the root MWUA snapshot remain useful?
- Do high-certainty variables stabilize early and form backbone-like patterns?
- Does resolving easy variables first reduce the active residual graph faster?
- How much does certainty persistence help compared with raw LP fractionality?
- When do simpler heuristics like most-fractional outperform or underperform deferred-uncertainty branching?
- Do exact reductions amplify pruning by exploiting structural certainty?

## Results and artifacts

The full Week 2 suite writes results into `results/`. A smoke-test run is stored in `results_smoke/`.

### Smoke test outputs

- Deferred-uncertainty analysis CSV: [Lit/week-2/results_smoke/deferred_uncertainty_analysis.csv](Lit/week-2/results_smoke/deferred_uncertainty_analysis.csv)
- Deferred-uncertainty analysis figure: [Lit/week-2/results_smoke/deferred_uncertainty_analysis.png](Lit/week-2/results_smoke/deferred_uncertainty_analysis.png)

### Smoke-test observations

The smoke run on `n=8` already shows the expected shape of the deferred-uncertainty diagnostics: LP certainty and certainty persistence increase from root to depth 1, while the residual edge density and active vertex count decrease. The generated figure is a compact sanity check that the new signals are being tracked correctly.

### Full results folder

The full suite currently generates these CSV and PNG outputs:

- Strategy comparison: [Lit/week-2/results/strategy_comparison.csv](Lit/week-2/results/strategy_comparison.csv), [Lit/week-2/results/strategy_comparison.png](Lit/week-2/results/strategy_comparison.png)
- Backbone / certainty persistence: [Lit/week-2/results/backbone_analysis.csv](Lit/week-2/results/backbone_analysis.csv), [Lit/week-2/results/backbone_analysis.png](Lit/week-2/results/backbone_analysis.png)
- Certainty evolution: [Lit/week-2/results/certainty_evolution.png](Lit/week-2/results/certainty_evolution.png)
- Deferred-uncertainty dynamics: [Lit/week-2/results/deferred_uncertainty_analysis.csv](Lit/week-2/results/deferred_uncertainty_analysis.csv), [Lit/week-2/results/deferred_uncertainty_analysis.png](Lit/week-2/results/deferred_uncertainty_analysis.png)

The full experimental suite has finished and all outputs have been written to the `results/` folder.

Run summary:

- Total solver runs (full suite): 338
- Worst-case node cap if every run hit its `max_nodes`: ~45,980,000 nodes

All results are available in [Lit/week-2/results](Lit/week-2/results).

Final CSV / PNG artifacts produced by the suite:

- Strategy comparison: [Lit/week-2/results/strategy_comparison.csv](Lit/week-2/results/strategy_comparison.csv), [Lit/week-2/results/strategy_comparison.png](Lit/week-2/results/strategy_comparison.png)
- Backbone / certainty persistence: [Lit/week-2/results/backbone_analysis.csv](Lit/week-2/results/backbone_analysis.csv), [Lit/week-2/results/backbone_analysis.png](Lit/week-2/results/backbone_analysis.png)
- Certainty evolution: [Lit/week-2/results/certainty_evolution.png](Lit/week-2/results/certainty_evolution.png)
- Deferred-uncertainty dynamics: [Lit/week-2/results/deferred_uncertainty_analysis.csv](Lit/week-2/results/deferred_uncertainty_analysis.csv), [Lit/week-2/results/deferred_uncertainty_analysis.png](Lit/week-2/results/deferred_uncertainty_analysis.png)
- Scaling benchmark: [Lit/week-2/results/scaling_benchmark.csv](Lit/week-2/results/scaling_benchmark.csv), [Lit/week-2/results/scaling_benchmark.png](Lit/week-2/results/scaling_benchmark.png)
- Density sweep: [Lit/week-2/results/density_sweep.csv](Lit/week-2/results/density_sweep.csv), [Lit/week-2/results/density_sweep.png](Lit/week-2/results/density_sweep.png)
- MWUA ablation: [Lit/week-2/results/mwua_ablation.csv](Lit/week-2/results/mwua_ablation.csv), [Lit/week-2/results/mwua_ablation.png](Lit/week-2/results/mwua_ablation.png)
- Reductions impact: [Lit/week-2/results/reductions_impact.csv](Lit/week-2/results/reductions_impact.csv), [Lit/week-2/results/reductions_impact.png](Lit/week-2/results/reductions_impact.png)
- Per-depth pruning heatmap: [Lit/week-2/results/depth_pruning_heatmap.png](Lit/week-2/results/depth_pruning_heatmap.png)

If the run is still in progress, this folder will remain partially populated until the terminal job completes.

## Inline figures

<p align="center">
	<img src="Lit/week-2/results_smoke/deferred_uncertainty_analysis.png" alt="deferred uncertainty analysis" width="900"/>
</p>

<p align="center">
	<img src="Lit/week-2/results/strategy_comparison.png" alt="strategy comparison" width="900"/>
</p>

<p align="center">
	<img src="Lit/week-2/results/backbone_analysis.png" alt="backbone analysis" width="900"/>
</p>

<p align="center">
	<img src="Lit/week-2/results/certainty_evolution.png" alt="certainty evolution" width="900"/>
</p>

Add the full-suite PNG files here after the long run completes. The README will be updated with direct figure links and thumbnails once the outputs exist.

## Reproducibility

### Prerequisites

- Python 3.13 in the workspace virtual environment: `.venv/`
- Recommended: activate the provided venv before running experiments

### Install dependencies

```bash
source /Users/ambikaprasanth/Desktop/Dissertation/.venv/bin/activate
pip install -r requirements.txt
# If there is no requirements.txt, install these:
# pip install networkx numpy scipy matplotlib
```

### Run the smoke test

```bash
python Lit/week-2/experiments.py --quick
```

### Run the full Week 2 suite

```bash
python Lit/week-2/experiments.py
```

### Run a targeted deferred-uncertainty analysis

```bash
python - <<'PY'
from pathlib import Path
import sys

repo = Path('/Users/ambikaprasanth/Desktop/Dissertation')
sys.path.insert(0, str(repo / 'Lit' / 'week-2'))

from experiments import run_deferred_uncertainty_analysis

out = repo / 'Lit' / 'week-2' / 'results_smoke'
out.mkdir(exist_ok=True)
run_deferred_uncertainty_analysis(out_dir=out, n=8, max_nodes=120)
PY
```

## Where to look in the code

- `GraphState` in `core.py` — incremental residual graph maintenance and local structural statistics.
- `CertaintyFirstBranching` in `branching.py` — deferred-uncertainty branching rule.
- `BranchAndBoundSolver.solve()` in `solver.py` — certainty persistence tracking, exact reductions, and diversified incumbent generation.
- `run_deferred_uncertainty_analysis()` in `experiments.py` — persistence and residual-graph dynamics across depth.

## Suggested next steps

- Run the full suite and fill in the `results/` section with the generated CSV and PNG links.
- Compare deferred-uncertainty branching against most-fractional branching using the full outputs.
- Add figure captions and a short table summarizing the Week 2 ablations.
- If needed, create a separate slide-summary README for presentation use.
