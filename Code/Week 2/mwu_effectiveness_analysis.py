"""Measure MWU effectiveness by comparing against random branching baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import linprog

try:
    from Lit.bnb_solver import MILPProblem, build_problem
except ModuleNotFoundError:
    from bnb_solver import MILPProblem, build_problem


@dataclass
class TreeStatistics:
    """Track B&B tree statistics at each depth level."""
    depth: int
    explored_at_depth: int
    pruned_at_depth: int
    first_incumbent_depth: Optional[int] = None
    first_incumbent_time: Optional[float] = None


class InstrumentedBBSolver:
    """Branch-and-bound with detailed tree instrumentation."""

    def __init__(
        self,
        tolerance: float = 1e-7,
        max_nodes: int | None = None,
        use_mwu: bool = True,
        mwu_weights: np.ndarray | None = None,
    ) -> None:
        self.tolerance = float(tolerance)
        self.max_nodes = max_nodes
        self.use_mwu = use_mwu
        self.mwu_weights = mwu_weights
        self.tree_stats: dict[int, TreeStatistics] = {}
        self.first_incumbent_depth: Optional[int] = None
        self.first_incumbent_time: Optional[float] = None
        self.total_time: float = 0.0
        self.explored_nodes = 0
        self.pruned_nodes = 0

    def solve(self, problem: MILPProblem) -> tuple[Optional[np.ndarray], float, dict[int, TreeStatistics]]:
        best_objective = np.inf
        best_x: Optional[np.ndarray] = None
        t_start = perf_counter()

        stack: list[tuple[tuple[tuple[int, float], ...], int]] = [((), 0)]  # (fixings, depth)
        self.explored_nodes = 0
        self.pruned_nodes = 0

        while stack:
            if self.max_nodes is not None and self.explored_nodes >= self.max_nodes:
                break

            fixings, depth = stack.pop()
            self.explored_nodes += 1

            if depth not in self.tree_stats:
                self.tree_stats[depth] = TreeStatistics(depth=depth, explored_at_depth=0, pruned_at_depth=0)
            self.tree_stats[depth].explored_at_depth += 1

            lp_result = self._solve_relaxation(problem, fixings)
            if lp_result is None or lp_result[0] > best_objective - self.tolerance:
                self.pruned_nodes += 1
                self.tree_stats[depth].pruned_at_depth += 1
                continue

            obj_val, x = lp_result

            is_integral = np.all(np.abs(x - np.rint(x)) <= self.tolerance)
            if is_integral:
                best_objective = obj_val
                best_x = np.clip(np.rint(x), 0.0, 1.0)
                if self.first_incumbent_depth is None:
                    self.first_incumbent_depth = depth
                    self.first_incumbent_time = perf_counter() - t_start
                    self.tree_stats[depth].first_incumbent_depth = depth
                    self.tree_stats[depth].first_incumbent_time = self.first_incumbent_time
                continue

            branch_var = self._select_branch_variable(x, depth)
            if branch_var is None:
                self.pruned_nodes += 1
                self.tree_stats[depth].pruned_at_depth += 1
                continue

            # Create children with depth tracking
            rounded_down = 0.0
            rounded_up = 1.0
            preferred = rounded_up if x[branch_var] >= 0.5 else rounded_down
            alternative = rounded_down if preferred == rounded_up else rounded_up

            alt_fixings = fixings + ((branch_var, alternative),)
            pref_fixings = fixings + ((branch_var, preferred),)

            # Push less-promising first so more-promising is popped first (DFS)
            stack.append((alt_fixings, depth + 1))
            stack.append((pref_fixings, depth + 1))

        self.total_time = perf_counter() - t_start
        return best_x, self.total_time, self.tree_stats

    def _solve_relaxation(self, problem: MILPProblem, fixings: tuple[tuple[int, float], ...]) -> Optional[tuple[float, np.ndarray]]:
        bounds = list(problem.bounds)
        for var_index, value in fixings:
            bounds[var_index] = (float(value), float(value))

        result = linprog(
            c=problem.c,
            A_ub=problem.A_ub,
            b_ub=problem.b_ub,
            A_eq=problem.A_eq,
            b_eq=problem.b_eq,
            bounds=bounds,
            method="highs",
        )

        if result.status != 0 or result.x is None:
            return None
        return float(result.fun), np.asarray(result.x, dtype=float)

    def _select_branch_variable(self, x: np.ndarray, depth: int) -> Optional[int]:
        fractional = np.flatnonzero(
            (x > self.tolerance) & (x < 1.0 - self.tolerance) & (np.abs(x - np.rint(x)) > self.tolerance)
        )
        if fractional.size == 0:
            return None

        if self.use_mwu and self.mwu_weights is not None:
            # MWU-guided: sort by distance from 0.5, then by MWU weight
            candidates = sorted(
                fractional.tolist(),
                key=lambda index: (
                    -abs(x[index] - 0.5),
                    -self.mwu_weights[index],
                    index,
                ),
            )
            return int(candidates[0])
        else:
            # Random baseline: just pick random fractional
            return int(np.random.choice(fractional))


def build_mvc_problem(graph: nx.Graph) -> tuple[MILPProblem, np.ndarray]:
    """Build MVC problem and compute MWU weights."""
    try:
        from Lit.bnb_solver import MWUScorer
    except ModuleNotFoundError:
        from bnb_solver import MWUScorer

    n_vars = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    c = np.ones(n_vars, dtype=float)
    A_ub = np.zeros((n_edges, n_vars), dtype=float)
    b_ub = -np.ones(n_edges, dtype=float)

    for edge_idx, (u, v) in enumerate(graph.edges()):
        A_ub[edge_idx, u] = -1.0
        A_ub[edge_idx, v] = -1.0

    problem = build_problem(c=c, A_ub=A_ub, b_ub=b_ub)
    scorer = MWUScorer(learning_rate=0.35, iterations=4)
    mwu_weights = scorer.compute(problem)
    return problem, mwu_weights


def main() -> None:
    root = Path(__file__).resolve().parent
    analysis_dir = root / "analysis_output"
    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    graphml_path = analysis_dir / "mvc_graph.graphml"
    if not graphml_path.exists():
        raise FileNotFoundError(f"Expected graph file at {graphml_path}")

    graph = nx.read_graphml(graphml_path)
    mapping = {node: int(node) if str(node).isdigit() else node for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, mapping)

    problem, mwu_weights = build_mvc_problem(graph)

    # Run MWU-guided solver
    print("Running MWU-guided solver...")
    solver_mwu = InstrumentedBBSolver(tolerance=1e-5, use_mwu=True, mwu_weights=mwu_weights, max_nodes=50000)
    best_x_mwu, time_mwu, stats_mwu = solver_mwu.solve(problem)

    # Run random baseline (multiple runs to average)
    print("Running random branching baseline (10 trials)...")
    times_random = []
    stats_random_list = []
    for trial in range(10):
        solver_random = InstrumentedBBSolver(tolerance=1e-5, use_mwu=False, mwu_weights=None, max_nodes=50000)
        _, time_random, stats_random = solver_random.solve(problem)
        times_random.append(time_random)
        stats_random_list.append(stats_random)

    mean_time_random = np.mean(times_random)
    std_time_random = np.std(times_random)

    print(f"\nMWU-guided: {time_mwu:.4f}s, explored={solver_mwu.explored_nodes}, first_incumbent_depth={solver_mwu.first_incumbent_depth}")
    print(
        f"Random baseline (avg): {mean_time_random:.4f}s ± {std_time_random:.4f}s, explored={np.mean([s.explored_nodes for s in [solver_random]])}  "
    )
    print(f"Speedup: {mean_time_random / time_mwu:.2f}x")

    # Plot depth statistics
    max_depth_mwu = max(stats_mwu.keys()) if stats_mwu else 0
    max_depth_random = max(max(stats.keys()) if stats else 0 for stats in stats_random_list)
    max_depth = max(max_depth_mwu, max_depth_random)

    depths = list(range(max_depth + 1))
    explored_mwu = [stats_mwu.get(d, TreeStatistics(d, 0, 0)).explored_at_depth for d in depths]
    pruned_mwu = [stats_mwu.get(d, TreeStatistics(d, 0, 0)).pruned_at_depth for d in depths]

    explored_random_avg = [np.mean([stats.get(d, TreeStatistics(d, 0, 0)).explored_at_depth for stats in stats_random_list]) for d in depths]
    pruned_random_avg = [np.mean([stats.get(d, TreeStatistics(d, 0, 0)).pruned_at_depth for stats in stats_random_list]) for d in depths]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(depths, explored_mwu, marker="o", label="MWU-guided", color="#06d6a0")
    axes[0, 0].plot(depths, explored_random_avg, marker="s", label="Random (avg)", color="#ef476f", alpha=0.7)
    axes[0, 0].set_xlabel("Tree Depth")
    axes[0, 0].set_ylabel("Nodes Explored at Depth")
    axes[0, 0].set_title("Nodes Explored per Depth Level")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(depths, pruned_mwu, marker="o", label="MWU-guided", color="#06d6a0")
    axes[0, 1].plot(depths, pruned_random_avg, marker="s", label="Random (avg)", color="#ef476f", alpha=0.7)
    axes[0, 1].set_xlabel("Tree Depth")
    axes[0, 1].set_ylabel("Nodes Pruned at Depth")
    axes[0, 1].set_title("Nodes Pruned per Depth Level")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    cumul_explored_mwu = np.cumsum(explored_mwu)
    cumul_explored_random = np.cumsum(explored_random_avg)
    axes[1, 0].plot(depths, cumul_explored_mwu, marker="o", label="MWU-guided", color="#06d6a0")
    axes[1, 0].plot(depths, cumul_explored_random, marker="s", label="Random (avg)", color="#ef476f", alpha=0.7)
    axes[1, 0].set_xlabel("Tree Depth")
    axes[1, 0].set_ylabel("Cumulative Nodes Explored")
    axes[1, 0].set_title("Cumulative Search Progress")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    if solver_mwu.first_incumbent_depth is not None:
        axes[1, 1].axvline(solver_mwu.first_incumbent_depth, color="#06d6a0", linestyle="--", label=f"MWU first incumbent @ depth {solver_mwu.first_incumbent_depth}")
    if stats_random_list and stats_random_list[0]:
        first_incumbent_random_avg = np.mean([max(s.keys()) if s else 0 for s in stats_random_list])
        axes[1, 1].axvline(first_incumbent_random_avg, color="#ef476f", linestyle="--", alpha=0.7, label=f"Random first incumbent @ depth {first_incumbent_random_avg:.1f}")
    axes[1, 1].set_xlabel("Tree Depth")
    axes[1, 1].set_ylabel("Time to First Incumbent (s)")
    axes[1, 1].set_title("First Incumbent Solution Discovery")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("MWU Effectiveness: Comparison vs Random Branching", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "mwu_effectiveness.png", dpi=220)
    plt.close(fig)

    # Report
    report_path = plot_dir / "mwu_effectiveness_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("MWU Effectiveness Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n\n")
        f.write("MWU-Guided Solver:\n")
        f.write(f"  Runtime: {time_mwu:.6f}s\n")
        f.write(f"  Explored nodes: {solver_mwu.explored_nodes}\n")
        f.write(f"  Pruned nodes: {solver_mwu.pruned_nodes}\n")
        f.write(f"  First incumbent depth: {solver_mwu.first_incumbent_depth}\n")
        f.write(f"  Time to first incumbent: {solver_mwu.first_incumbent_time:.6f}s\n\n")
        f.write("Random Branching Baseline (10 trials):\n")
        f.write(f"  Mean runtime: {mean_time_random:.6f}s ± {std_time_random:.6f}s\n")
        f.write(f"  Speedup (MWU / Random): {mean_time_random / time_mwu:.2f}x\n\n")
        f.write("Key Findings:\n")
        f.write(f"  - MWU explores {solver_mwu.explored_nodes} nodes\n")
        f.write(f"  - Random explores ~{np.mean([s.explored_nodes for s in [solver_random]])} nodes on average\n")
        f.write(f"  - MWU finds first incumbent at depth {solver_mwu.first_incumbent_depth}\n")
        f.write(f"  - Max tree depth (MWU): {max_depth_mwu}\n")
        f.write(f"  - Max tree depth (Random avg): {max_depth_random:.1f}\n")

    print(f"\nSaved effectiveness plots to {plot_dir}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
