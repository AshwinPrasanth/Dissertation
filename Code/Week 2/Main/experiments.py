"""experiments.py — Research-grade experimental benchmark suite.

Implements all experiments described in the dissertation prompt:

1. run_strategy_comparison()
   Compare all 6 branching strategies across graph sizes.
   Metrics: nodes explored, pruning rate, first incumbent depth/time,
   total runtime, optimality gap.

2. run_backbone_analysis()
   Study certainty persistence: how stable are variable assignments
   across depth? Do high-MWUA-certainty variables settle early?

3. run_certainty_evolution()
   Track mean LP certainty and MWUA certainty as depth increases.
   Hypothesis: certainty-first diving increases average certainty faster.

4. run_scaling_benchmark()
   Runtime and node-count scaling vs graph size (n = 10..120).

5. run_density_sweep()
   Runtime and nodes vs graph density for fixed n.

6. run_mwua_vs_lp_ablation()
   Ablation: certainty_first vs mwua_only vs lp_only_certainty.

All results are saved as CSVs and publication-quality plots.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from core import (
    MILPProblem,
    StructuralFeatureEngine,
    build_mvc_problem,
)
from branching import (
    CertaintyFirstBranching,
    CertaintyFirstConfig,
    DegreeBranching,
    MostFractionalBranching,
    MWUAOnlyBranching,
    PseudoCostBranching,
    RandomBranching,
    make_strategy,
)
from solver import BranchAndBoundSolver, BBSolution


# ---------------------------------------------------------------------------
# Colour palette (accessible, publication-friendly)
# ---------------------------------------------------------------------------
PALETTE = {
    "certainty_first": "#2d6a4f",
    "most_fractional": "#e63946",
    "random":          "#8338ec",
    "degree":          "#fb8500",
    "pseudo_cost":     "#118ab2",
    "mwua_only":       "#06d6a0",
}
MARKERS = {
    "certainty_first": "o",
    "most_fractional": "s",
    "random":          "^",
    "degree":          "D",
    "pseudo_cost":     "P",
    "mwua_only":       "X",
}


def _make_solver(strategy_name: str, max_nodes: int = 200_000) -> BranchAndBoundSolver:
    return BranchAndBoundSolver(
        branching=make_strategy(strategy_name),
        tolerance=1e-6,
        max_nodes=max_nodes,
        use_reductions=True,
        certainty_lambda=0.5,
        track_backbone=True,
    )


def _build_gnp(n: int, p: float, seed: int) -> Tuple[nx.Graph, MILPProblem]:
    G = nx.gnp_random_graph(n=n, p=p, seed=seed)
    # Ensure connected (add a spanning path if disconnected)
    if not nx.is_connected(G) and G.number_of_nodes() > 1:
        comps = list(nx.connected_components(G))
        for i in range(len(comps) - 1):
            u = next(iter(comps[i]))
            v = next(iter(comps[i + 1]))
            G.add_edge(u, v)
    return G, build_mvc_problem(G)


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 1. Strategy comparison
# ---------------------------------------------------------------------------

def run_strategy_comparison(
    out_dir: Path,
    sizes: List[int] = (15, 20, 25, 30, 40),
    p: float = 0.4,
    n_trials: int = 3,
    max_nodes: int = 150_000,
) -> None:
    """Compare all branching strategies across graph sizes."""
    strategies = ["certainty_first", "most_fractional", "random", "degree", "pseudo_cost", "mwua_only"]
    rows: List[dict] = []

    for n in sizes:
        for trial in range(n_trials):
            seed = 42 + n * 100 + trial
            G, problem = _build_gnp(n, p, seed)

            for strat in strategies:
                solver = _make_solver(strat, max_nodes=max_nodes)
                sol = solver.solve(problem)
                tr = sol.trace
                rows.append({
                    "n": n,
                    "trial": trial,
                    "strategy": strat,
                    "status": sol.status,
                    "obj": sol.objective_value,
                    "explored": tr.explored_nodes,
                    "pruned": tr.pruned_nodes,
                    "prune_rate": f"{tr.pruning_rate():.4f}",
                    "first_inc_depth": tr.first_incumbent_depth,
                    "first_inc_time": f"{tr.first_incumbent_time:.6f}" if tr.first_incumbent_time else "NA",
                    "total_time": f"{tr.total_time:.6f}",
                    "reduction_fixes": tr.reduction_fixes,
                    "lp_time_frac": f"{tr.lp_solve_time / max(tr.total_time, 1e-9):.4f}",
                })
                print(f"  n={n:3d} trial={trial} {strat:20s} "
                      f"nodes={tr.explored_nodes:7d} prune={tr.pruning_rate():.2%} "
                      f"t={tr.total_time:.3f}s")

    _write_csv(out_dir / "strategy_comparison.csv",
               ["n","trial","strategy","status","obj","explored","pruned",
                "prune_rate","first_inc_depth","first_inc_time","total_time",
                "reduction_fixes","lp_time_frac"],
               rows)

    _plot_strategy_comparison(rows, sizes, strategies, out_dir)
    print(f"[strategy_comparison] saved to {out_dir}")


def _plot_strategy_comparison(rows, sizes, strategies, out_dir):
    # Aggregate over trials
    from collections import defaultdict
    agg: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for r in rows:
        agg[(r["n"], r["strategy"])].append(r)

    def mean_field(key, n, strat):
        vals = [float(r[key]) for r in agg[(n, strat)] if r[key] not in ("NA", None, "")]
        return np.mean(vals) if vals else np.nan

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Branching Strategy Comparison (MVC, p=0.4)", fontsize=14, fontweight="bold")

    metrics = [
        ("explored",       "Nodes Explored",       axes[0, 0]),
        ("prune_rate",     "Pruning Rate",          axes[0, 1]),
        ("total_time",     "Total Runtime (s)",     axes[0, 2]),
        ("first_inc_depth","First Incumbent Depth", axes[1, 0]),
        ("first_inc_time", "Time to First Incumbent (s)", axes[1, 1]),
        ("reduction_fixes","Reduction Fixes",       axes[1, 2]),
    ]

    for field, ylabel, ax in metrics:
        for strat in strategies:
            y = [mean_field(field, n, strat) for n in sizes]
            ax.plot(sizes, y, marker=MARKERS[strat], color=PALETTE[strat],
                    label=strat.replace("_", " "), linewidth=1.8, markersize=6)
        ax.set_xlabel("Graph Size (n)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "strategy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Backbone analysis
# ---------------------------------------------------------------------------

def run_backbone_analysis(
    out_dir: Path,
    n: int = 30,
    p: float = 0.4,
    seed: int = 42,
    max_nodes: int = 100_000,
) -> None:
    """Study variable stability: do high-certainty variables settle earlier?"""
    G, problem = _build_gnp(n, p, seed)

    solver = BranchAndBoundSolver(
        branching=CertaintyFirstBranching(),
        tolerance=1e-6,
        max_nodes=max_nodes,
        track_backbone=True,
    )
    sol = solver.solve(problem)
    tr = sol.trace

    if tr.var_fractional_count is None:
        print("[backbone] no backbone data collected")
        return

    total_visits = tr.explored_nodes
    frac_freq = tr.var_fractional_count / max(total_visits, 1)
    fix1_freq = tr.var_fixed_one_count / max(total_visits, 1)
    fix0_freq = tr.var_fixed_zero_count / max(total_visits, 1)
    # Backbone-like stability: how often is the variable consistently fixed one way
    stability = np.maximum(fix1_freq, fix0_freq) / np.maximum(frac_freq + fix1_freq + fix0_freq, 1e-9)

    # Compute MWUA certainty from feature engine
    feat_engine = StructuralFeatureEngine()
    features = feat_engine.compute(problem)
    mwua_cert = features.mwua_certainty

    # Sort variables by MWUA certainty
    sort_idx = np.argsort(mwua_cert)[::-1]

    rows = [{"var": int(i), "mwua_certainty": float(mwua_cert[i]),
             "frac_freq": float(frac_freq[i]), "stability": float(stability[i]),
             "fix1_freq": float(fix1_freq[i]), "fix0_freq": float(fix0_freq[i])}
            for i in sort_idx]
    _write_csv(out_dir / "backbone_analysis.csv",
               ["var","mwua_certainty","frac_freq","stability","fix1_freq","fix0_freq"],
               rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Backbone / Certainty Persistence Analysis (n={n}, p={p})", fontsize=13)

    vars_sorted = np.arange(n)
    axes[0].bar(vars_sorted, mwua_cert[sort_idx], color=PALETTE["certainty_first"], alpha=0.8)
    axes[0].set_xlabel("Variable (sorted by MWUA certainty)")
    axes[0].set_ylabel("MWUA Certainty")
    axes[0].set_title("MWUA Certainty per Variable")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(mwua_cert, stability, alpha=0.7, color=PALETTE["mwua_only"], s=50)
    # Correlation
    corr = np.corrcoef(mwua_cert, stability)[0, 1]
    axes[1].set_xlabel("MWUA Certainty")
    axes[1].set_ylabel("Backbone Stability")
    axes[1].set_title(f"MWUA Certainty vs Backbone Stability\n(r = {corr:.3f})")
    axes[1].grid(alpha=0.3)

    axes[2].scatter(mwua_cert, frac_freq, alpha=0.7, color=PALETTE["most_fractional"], s=50)
    corr2 = np.corrcoef(mwua_cert, frac_freq)[0, 1]
    axes[2].set_xlabel("MWUA Certainty")
    axes[2].set_ylabel("Fractional Frequency in B&B")
    axes[2].set_title(f"MWUA Certainty vs Fractional Frequency\n(r = {corr2:.3f})")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "backbone_analysis.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[backbone] MWUA–stability correlation: r={corr:.3f}")
    print(f"[backbone] MWUA–frac_freq correlation:  r={corr2:.3f}")


# ---------------------------------------------------------------------------
# 3. Certainty evolution
# ---------------------------------------------------------------------------

def run_certainty_evolution(
    out_dir: Path,
    n: int = 35,
    p: float = 0.4,
    seed: int = 7,
    max_nodes: int = 80_000,
) -> None:
    """Track how LP certainty changes with depth under different strategies."""
    G, problem = _build_gnp(n, p, seed)

    strategies_to_compare = ["certainty_first", "most_fractional", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Certainty Evolution Across B&B Depth (n={n}, p={p})", fontsize=13)

    for strat in strategies_to_compare:
        solver = BranchAndBoundSolver(
            branching=make_strategy(strat),
            tolerance=1e-6,
            max_nodes=max_nodes,
            track_backbone=True,
        )
        sol = solver.solve(problem)
        ev = sol.trace.certainty_evolution

        if not ev:
            continue
        # Bin by depth
        from collections import defaultdict
        depth_cert: Dict[int, List[float]] = defaultdict(list)
        for depth, lp_c, _ in ev:
            depth_cert[depth].append(lp_c)

        depths = sorted(depth_cert.keys())
        mean_cert = [np.mean(depth_cert[d]) for d in depths]
        axes[0].plot(depths, mean_cert, marker=MARKERS[strat],
                     color=PALETTE[strat], label=strat.replace("_", " "),
                     linewidth=1.8, markersize=5, alpha=0.85)

        # Cumulative nodes vs mean certainty
        sorted_ev = sorted(ev, key=lambda x: x[0])
        node_idxs = list(range(1, len(sorted_ev) + 1))
        running_cert = [x[1] for x in sorted_ev]
        axes[1].plot(node_idxs[:5000], running_cert[:5000],
                     color=PALETTE[strat], label=strat.replace("_", " "),
                     linewidth=1.2, alpha=0.75)

    axes[0].set_xlabel("Tree Depth")
    axes[0].set_ylabel("Mean LP Certainty (|x - 0.5|)")
    axes[0].set_title("LP Certainty vs Depth")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("B&B Node Index (first 5000)")
    axes[1].set_ylabel("LP Certainty at Node")
    axes[1].set_title("Certainty Over Search Progress")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "certainty_evolution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[certainty_evolution] saved")


# ---------------------------------------------------------------------------
# 4. Scaling benchmark
# ---------------------------------------------------------------------------

def run_scaling_benchmark(
    out_dir: Path,
    sizes: List[int] = (10, 15, 20, 25, 30, 40, 50, 60, 80, 100),
    p: float = 0.4,
    strategies: List[str] = ("certainty_first", "most_fractional", "random"),
    max_nodes: int = 300_000,
) -> None:
    rows: List[dict] = []

    for n in sizes:
        G, problem = _build_gnp(n, p, seed=42 + n)
        for strat in strategies:
            solver = _make_solver(strat, max_nodes=max_nodes)
            sol = solver.solve(problem)
            tr = sol.trace
            rows.append({
                "n": n,
                "edges": G.number_of_edges(),
                "density": f"{nx.density(G):.4f}",
                "strategy": strat,
                "status": sol.status,
                "obj": sol.objective_value,
                "explored": tr.explored_nodes,
                "pruned": tr.pruned_nodes,
                "prune_rate": f"{tr.pruning_rate():.4f}",
                "time_s": f"{tr.total_time:.6f}",
                "reduction_fixes": tr.reduction_fixes,
                "first_inc_depth": tr.first_incumbent_depth,
            })
            print(f"  scaling n={n:3d} {strat:20s} nodes={tr.explored_nodes:8d} t={tr.total_time:.3f}s")

    _write_csv(out_dir / "scaling_benchmark.csv",
               ["n","edges","density","strategy","status","obj","explored",
                "pruned","prune_rate","time_s","reduction_fixes","first_inc_depth"],
               rows)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Scaling Benchmark (MVC, p=0.4)", fontsize=13, fontweight="bold")

    from collections import defaultdict
    strat_data: Dict[str, Dict[str, List]] = {s: {"n":[], "t":[], "nodes":[]} for s in strategies}
    for r in rows:
        s = r["strategy"]
        strat_data[s]["n"].append(r["n"])
        strat_data[s]["t"].append(float(r["time_s"]))
        strat_data[s]["nodes"].append(r["explored"])

    for strat in strategies:
        d = strat_data[strat]
        axes[0].plot(d["n"], d["t"], marker=MARKERS[strat], color=PALETTE[strat],
                     label=strat.replace("_", " "), linewidth=1.8)
        axes[1].plot(d["n"], d["nodes"], marker=MARKERS[strat], color=PALETTE[strat],
                     label=strat.replace("_", " "), linewidth=1.8)
        axes[2].semilogy(d["n"], d["nodes"], marker=MARKERS[strat], color=PALETTE[strat],
                         label=strat.replace("_", " "), linewidth=1.8)

    for ax, ylabel, title in [
        (axes[0], "Runtime (s)",        "Runtime vs Graph Size"),
        (axes[1], "Nodes Explored",     "Search Tree Size vs Graph Size"),
        (axes[2], "Nodes (log scale)",  "Search Tree Size (log) vs Graph Size"),
    ]:
        ax.set_xlabel("Graph Size (n)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "scaling_benchmark.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[scaling] saved")


# ---------------------------------------------------------------------------
# 5. Density sweep
# ---------------------------------------------------------------------------

def run_density_sweep(
    out_dir: Path,
    n: int = 50,
    densities: List[float] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
    strategies: List[str] = ("certainty_first", "most_fractional", "degree"),
    max_nodes: int = 200_000,
) -> None:
    rows: List[dict] = []

    for p in densities:
        G, problem = _build_gnp(n, p, seed=99 + int(p * 100))
        for strat in strategies:
            solver = _make_solver(strat, max_nodes=max_nodes)
            sol = solver.solve(problem)
            tr = sol.trace
            rows.append({
                "density": f"{p:.2f}",
                "strategy": strat,
                "explored": tr.explored_nodes,
                "time_s": f"{tr.total_time:.6f}",
                "prune_rate": f"{tr.pruning_rate():.4f}",
                "reduction_fixes": tr.reduction_fixes,
            })
            print(f"  density p={p:.2f} {strat:20s} nodes={tr.explored_nodes:8d} t={tr.total_time:.3f}s")

    _write_csv(out_dir / "density_sweep.csv",
               ["density","strategy","explored","time_s","prune_rate","reduction_fixes"],
               rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Difficulty vs Graph Density (n={n})", fontsize=13)

    from collections import defaultdict
    sd: Dict[str, Dict] = {s: {"p":[], "t":[], "nodes":[]} for s in strategies}
    for r in rows:
        s = r["strategy"]
        sd[s]["p"].append(float(r["density"]))
        sd[s]["t"].append(float(r["time_s"]))
        sd[s]["nodes"].append(r["explored"])

    for strat in strategies:
        d = sd[strat]
        axes[0].plot(d["p"], d["t"], marker=MARKERS[strat], color=PALETTE[strat],
                     label=strat.replace("_", " "), linewidth=1.8)
        axes[1].plot(d["p"], d["nodes"], marker=MARKERS[strat], color=PALETTE[strat],
                     label=strat.replace("_", " "), linewidth=1.8)

    for ax, ylabel, title in [
        (axes[0], "Runtime (s)",    "Runtime vs Density"),
        (axes[1], "Nodes Explored", "Search Tree Size vs Density"),
    ]:
        ax.set_xlabel("Graph Density p")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "density_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[density_sweep] saved")


# ---------------------------------------------------------------------------
# 6. MWUA vs LP ablation
# ---------------------------------------------------------------------------

def run_mwua_ablation(
    out_dir: Path,
    sizes: List[int] = (15, 20, 25, 30, 35),
    p: float = 0.4,
    n_trials: int = 5,
    max_nodes: int = 100_000,
) -> None:
    """Ablation: decompose the four-signal certainty-first branching score.

    Answers the key research questions directly:
    - Does MWUA add value beyond the free LP certainty?         (lp_only vs lp+mwua)
    - Does residual degree add beyond LP+MWUA?                  (lp+mwua vs lp+mwua+degree)
    - How important is depth-adaptive decay?                    (full vs no_decay)
    - Is the full combination justified?                        (full vs all ablations)

    The 'full' line should dominate; each ablation row shows what is lost
    by removing one signal.  This is the core experimental evidence for the
    thesis claim.
    """
    from collections import defaultdict
    configs = {
        "full (α+β+γ+δ)":        CertaintyFirstConfig(alpha=0.40, beta=0.35, gamma=0.15, delta=0.10),
        "lp_only (α)": CertaintyFirstConfig(alpha=1.00, beta=0.00, gamma=0.00, delta=0.00),
        "mwua_only (β)": CertaintyFirstConfig(alpha=0.00, beta=1.00, gamma=0.00, delta=0.00),
        "lp+mwua (α+β)": CertaintyFirstConfig(alpha=0.53, beta=0.47, gamma=0.00, delta=0.00),
        "lp+mwua+degree (α+β+γ)": CertaintyFirstConfig(alpha=0.42, beta=0.37, gamma=0.21, delta=0.00),
        "no_decay": CertaintyFirstConfig(alpha=0.40, beta=0.35, gamma=0.15, delta=0.10, depth_decay=0.0),
    }

    rows: List[dict] = []
    for n in sizes:
        for trial in range(n_trials):
            G, problem = _build_gnp(n, p, seed=200 + n * 10 + trial)
            for name, cfg in configs.items():
                solver = BranchAndBoundSolver(
                    branching=CertaintyFirstBranching(config=cfg),
                    tolerance=1e-6, max_nodes=max_nodes, track_backbone=False,
                )
                sol = solver.solve(problem)
                tr = sol.trace
                rows.append({
                    "n": n, "trial": trial, "config": name,
                    "explored": tr.explored_nodes,
                    "time_s": f"{tr.total_time:.6f}",
                    "prune_rate": f"{tr.pruning_rate():.4f}",
                    "first_inc_depth": tr.first_incumbent_depth,
                })

    _write_csv(out_dir / "mwua_ablation.csv",
               ["n","trial","config","explored","time_s","prune_rate","first_inc_depth"],
               rows)

    config_nodes: Dict[str, Dict[int, List]] = {c: defaultdict(list) for c in configs}
    config_times: Dict[str, Dict[int, List]] = {c: defaultdict(list) for c in configs}
    for r in rows:
        config_nodes[r["config"]][r["n"]].append(r["explored"])
        config_times[r["config"]][r["n"]].append(float(r["time_s"]))

    palette = ["#2d6a4f", "#e63946", "#fb8500", "#118ab2", "#8338ec", "#adb5bd"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Signal Ablation: Four-Signal Certainty-First Branching", fontsize=13, fontweight="bold")

    for (name, _), color in zip(configs.items(), palette):
        lw = 2.4 if "full" in name else 1.4
        ls = "-" if "full" in name else "--"
        axes[0].plot(sizes, [np.mean(config_nodes[name][n]) for n in sizes],
                     marker="o", color=color, label=name, linewidth=lw, linestyle=ls, markersize=6)
        axes[1].plot(sizes, [np.mean(config_times[name][n]) for n in sizes],
                     marker="o", color=color, label=name, linewidth=lw, linestyle=ls, markersize=6)

    for ax, ylabel, title in [
        (axes[0], "Mean Nodes Explored", "Search Tree Size by Signal Combination"),
        (axes[1], "Mean Runtime (s)",    "Runtime by Signal Combination"),
    ]:
        ax.set_xlabel("Graph Size (n)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "mwua_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[mwua_ablation] saved")


# ---------------------------------------------------------------------------
# 7. Reductions impact analysis
# ---------------------------------------------------------------------------

def run_reductions_impact(
    out_dir: Path,
    sizes: List[int] = (20, 30, 40, 50),
    p: float = 0.4,
    n_trials: int = 5,
    max_nodes: int = 100_000,
) -> None:
    """Compare certainty-first with and without branch-and-reduce."""
    rows: List[dict] = []

    for n in sizes:
        for trial in range(n_trials):
            G, problem = _build_gnp(n, p, seed=500 + n * 10 + trial)
            for use_red in [True, False]:
                solver = BranchAndBoundSolver(
                    branching=CertaintyFirstBranching(),
                    tolerance=1e-6,
                    max_nodes=max_nodes,
                    use_reductions=use_red,
                    track_backbone=False,
                )
                sol = solver.solve(problem)
                tr = sol.trace
                rows.append({
                    "n": n, "trial": trial,
                    "reductions": "on" if use_red else "off",
                    "explored": tr.explored_nodes,
                    "time_s": f"{tr.total_time:.6f}",
                    "prune_rate": f"{tr.pruning_rate():.4f}",
                    "reduction_fixes": tr.reduction_fixes,
                })

    _write_csv(out_dir / "reductions_impact.csv",
               ["n","trial","reductions","explored","time_s","prune_rate","reduction_fixes"],
               rows)

    from collections import defaultdict
    data = {"on": defaultdict(list), "off": defaultdict(list)}
    for r in rows:
        data[r["reductions"]][r["n"]].append(r["explored"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("on", "#2d6a4f"), ("off", "#e63946")]:
        mean_nodes = [np.mean(data[label][n]) for n in sizes]
        ax.plot(sizes, mean_nodes, marker="o", color=color,
                label=f"reductions={label}", linewidth=1.8, markersize=7)

    ax.set_xlabel("Graph Size (n)")
    ax.set_ylabel("Mean Nodes Explored")
    ax.set_title("Impact of Branch-and-Reduce Reductions\n(certainty-first branching)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reductions_impact.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[reductions_impact] saved")


# ---------------------------------------------------------------------------
# 8. Per-depth pruning heatmap
# ---------------------------------------------------------------------------

def run_depth_pruning_heatmap(
    out_dir: Path,
    n: int = 30,
    p: float = 0.4,
    seed: int = 42,
    strategies: List[str] = ("certainty_first", "most_fractional", "random"),
    max_nodes: int = 100_000,
) -> None:
    """Heatmap: pruning rate per depth level per strategy."""
    G, problem = _build_gnp(n, p, seed)

    all_prune_rates: Dict[str, Dict[int, float]] = {}
    max_depth = 0

    for strat in strategies:
        solver = _make_solver(strat, max_nodes=max_nodes)
        sol = solver.solve(problem)
        ds = sol.trace.depth_stats

        if ds:
            max_depth = max(max_depth, max(ds.keys()))

        rates = {}
        for d, stat in ds.items():
            total_d = stat.explored
            pruned_d = stat.pruned_bound + stat.pruned_infeasible + stat.pruned_reduction
            rates[d] = pruned_d / total_d if total_d > 0 else 0.0
        all_prune_rates[strat] = rates

    if max_depth == 0:
        return

    depths = list(range(max_depth + 1))
    matrix = np.zeros((len(strategies), len(depths)), dtype=float)
    for i, strat in enumerate(strategies):
        for j, d in enumerate(depths):
            matrix[i, j] = all_prune_rates.get(strat, {}).get(d, np.nan)

    fig, ax = plt.subplots(figsize=(min(max_depth + 2, 20), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths], fontsize=7)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([s.replace("_", " ") for s in strategies], fontsize=9)
    ax.set_xlabel("Tree Depth")
    ax.set_title(f"Pruning Rate by Depth and Strategy (n={n}, p={p})")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Pruning Rate")
    fig.tight_layout()
    fig.savefig(out_dir / "depth_pruning_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[depth_pruning_heatmap] saved")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all(out_dir: Optional[Path] = None, quick: bool = False) -> None:
    """Run the full experimental suite."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Research-Grade B&B Experimental Suite")
    print("=" * 60)

    if quick:
        # Reduced parameters for a fast smoke-test
        print("\n[MODE: QUICK — reduced parameters for speed]")
        run_strategy_comparison(out_dir, sizes=[10, 15, 20], n_trials=2, max_nodes=30_000)
        run_backbone_analysis(out_dir, n=20, max_nodes=30_000)
        run_certainty_evolution(out_dir, n=20, max_nodes=30_000)
        run_scaling_benchmark(out_dir, sizes=[10, 15, 20, 25], max_nodes=30_000,
                              strategies=["certainty_first", "most_fractional", "random"])
        run_density_sweep(out_dir, n=20, max_nodes=30_000)
        run_mwua_ablation(out_dir, sizes=[12, 15, 20], n_trials=3, max_nodes=30_000)
        run_reductions_impact(out_dir, sizes=[15, 20], n_trials=3, max_nodes=30_000)
        run_depth_pruning_heatmap(out_dir, n=20, max_nodes=30_000)
    else:
        run_strategy_comparison(out_dir)
        run_backbone_analysis(out_dir)
        run_certainty_evolution(out_dir)
        run_scaling_benchmark(out_dir)
        run_density_sweep(out_dir)
        run_mwua_ablation(out_dir)
        run_reductions_impact(out_dir)
        run_depth_pruning_heatmap(out_dir)

    print("\n" + "=" * 60)
    print(f"  All results saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run B&B experimental suite")
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test run")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    run_all(out_dir=Path(args.out) if args.out else None, quick=args.quick)