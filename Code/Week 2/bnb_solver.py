"""Branch-and-bound solver for binary MILPs.

The implementation is intentionally explicit:
- LP relaxations are solved with ``scipy.optimize.linprog``.
- Node selection uses a LIFO stack to realize depth-first search.
- Branching uses a static MWU snapshot computed once at the root.
- Fractional candidates are ranked by distance from 0.5 and MWU weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog


ArrayLike = Sequence[float] | np.ndarray


@dataclass(slots=True)
class MILPProblem:
    """Binary MILP in standard linear form.

    This solver assumes all variables are binary in the original problem.
    The branch-and-bound tree enforces integrality by fixing variables to 0 or 1.
    """

    c: np.ndarray
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    bounds: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        n_vars = self.c.size

        self.A_ub = self._coerce_matrix(self.A_ub, n_vars)
        self.b_ub = self._coerce_vector(self.b_ub, self.A_ub.shape[0] if self.A_ub is not None else 0)
        self.A_eq = self._coerce_matrix(self.A_eq, n_vars)
        self.b_eq = self._coerce_vector(self.b_eq, self.A_eq.shape[0] if self.A_eq is not None else 0)

        if self.bounds is None:
            self.bounds = [(0.0, 1.0) for _ in range(n_vars)]
        else:
            if len(self.bounds) != n_vars:
                raise ValueError("bounds must have one entry per variable")
            self.bounds = [(float(lo), float(hi)) for lo, hi in self.bounds]

    @property
    def n_vars(self) -> int:
        return int(self.c.size)

    @staticmethod
    def _coerce_matrix(matrix: ArrayLike | None, n_vars: int) -> np.ndarray | None:
        if matrix is None:
            return None
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2:
            raise ValueError("constraint matrices must be two-dimensional")
        if array.shape[1] != n_vars:
            raise ValueError("constraint matrix has incompatible width")
        return array

    @staticmethod
    def _coerce_vector(vector: ArrayLike | None, expected_size: int) -> np.ndarray | None:
        if vector is None:
            return None if expected_size == 0 else np.zeros(expected_size, dtype=float)
        array = np.asarray(vector, dtype=float)
        if array.ndim != 1:
            raise ValueError("constraint vectors must be one-dimensional")
        if expected_size and array.size != expected_size:
            raise ValueError("constraint vector has incompatible length")
        return array


@dataclass(slots=True)
class LPRelaxationResult:
    status: int
    objective_value: float
    x: np.ndarray | None
    message: str

    @property
    def is_feasible(self) -> bool:
        return self.status == 0 and self.x is not None


@dataclass(slots=True)
class BBNode:
    """A subproblem defined by variable fixings."""

    fixings: tuple[tuple[int, float], ...] = field(default_factory=tuple)
    depth: int = 0

    def child(self, var_index: int, value: float) -> "BBNode":
        return BBNode(fixings=self.fixings + ((var_index, float(value)),), depth=self.depth + 1)


@dataclass(slots=True)
class BBSolution:
    objective_value: float
    x: np.ndarray | None
    node_count: int
    explored_nodes: int
    pruned_nodes: int
    status: str


class MWUScorer:
    """Compute a static MWU-style global score for each variable."""

    def __init__(self, learning_rate: float = 0.35, iterations: int = 4) -> None:
        self.learning_rate = float(learning_rate)
        self.iterations = int(iterations)

    def compute(self, problem: MILPProblem) -> np.ndarray:
        n_vars = problem.n_vars
        if n_vars == 0:
            return np.array([], dtype=float)

        objective_signal = np.abs(problem.c)
        if np.max(objective_signal) > 0:
            objective_signal = objective_signal / np.max(objective_signal)

        structural_signal = self._column_structure_signal(problem)
        if structural_signal.size and np.max(structural_signal) > 0:
            structural_signal = structural_signal / np.max(structural_signal)

        score = 0.5 * objective_signal + 0.5 * structural_signal
        score = np.maximum(score, 1e-12)

        weights = np.ones(n_vars, dtype=float)
        for _ in range(self.iterations):
            weights *= np.exp(self.learning_rate * score)
            weights /= np.mean(weights)

        return weights

    @staticmethod
    def _column_structure_signal(problem: MILPProblem) -> np.ndarray:
        matrices = [matrix for matrix in (problem.A_ub, problem.A_eq) if matrix is not None]
        if not matrices:
            return np.ones(problem.n_vars, dtype=float)

        stacked = np.vstack(matrices)
        abs_stacked = np.abs(stacked)
        row_norms = abs_stacked.sum(axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        normalized = abs_stacked / row_norms
        return normalized.sum(axis=0)


class BranchAndBoundSolver:
    """Exact depth-first branch-and-bound for binary MILPs.

    The solver is minimization-only. For maximization problems, negate the
    objective before calling :meth:`solve` and negate the returned objective.
    """

    def __init__(
        self,
        tolerance: float = 1e-7,
        max_nodes: int | None = None,
        mwu_learning_rate: float = 0.35,
        mwu_iterations: int = 4,
    ) -> None:
        self.tolerance = float(tolerance)
        self.max_nodes = max_nodes
        self.scorer = MWUScorer(learning_rate=mwu_learning_rate, iterations=mwu_iterations)

    def solve(self, problem: MILPProblem) -> BBSolution:
        mwu_weights = self.scorer.compute(problem)
        root = BBNode()

        best_objective = np.inf
        best_x: np.ndarray | None = None

        stack: list[BBNode] = [root]
        explored_nodes = 0
        pruned_nodes = 0

        while stack:
            if self.max_nodes is not None and explored_nodes >= self.max_nodes:
                break

            node = stack.pop()
            explored_nodes += 1

            lp_result = self._solve_relaxation(problem, node)
            if not lp_result.is_feasible:
                pruned_nodes += 1
                continue

            if lp_result.objective_value >= best_objective - self.tolerance:
                pruned_nodes += 1
                continue

            x = lp_result.x
            assert x is not None

            is_integral = self._is_integral(x)
            if is_integral:
                best_objective = lp_result.objective_value
                best_x = np.clip(np.rint(x), 0.0, 1.0)
                continue

            branch_var = self._select_branch_variable(x, mwu_weights)
            if branch_var is None:
                pruned_nodes += 1
                continue

            left_child, right_child = self._create_children(node, branch_var, x[branch_var])

            # LIFO stack => push the less promising child first.
            stack.append(left_child)
            stack.append(right_child)

        if best_x is None:
            return BBSolution(
                objective_value=np.inf,
                x=None,
                node_count=explored_nodes,
                explored_nodes=explored_nodes,
                pruned_nodes=pruned_nodes,
                status="infeasible_or_max_nodes_reached",
            )

        return BBSolution(
            objective_value=float(best_objective),
            x=best_x,
            node_count=explored_nodes,
            explored_nodes=explored_nodes,
            pruned_nodes=pruned_nodes,
            status="optimal",
        )

    def _solve_relaxation(self, problem: MILPProblem, node: BBNode) -> LPRelaxationResult:
        bounds = list(problem.bounds)
        for var_index, value in node.fixings:
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

        objective_value = float(result.fun) if result.fun is not None else np.inf
        x = None if result.x is None else np.asarray(result.x, dtype=float)
        return LPRelaxationResult(status=int(result.status), objective_value=objective_value, x=x, message=result.message)

    def _select_branch_variable(self, x: np.ndarray, mwu_weights: np.ndarray) -> int | None:
        fractional = self._fractional_indices(x)
        if fractional.size == 0:
            return None

        candidates = sorted(
            fractional.tolist(),
            key=lambda index: (
                -abs(x[index] - 0.5),
                -mwu_weights[index],
                index,
            ),
        )
        return int(candidates[0])

    def _create_children(self, node: BBNode, var_index: int, value: float) -> tuple[BBNode, BBNode]:
        rounded_down = 0.0
        rounded_up = 1.0
        preferred = rounded_up if value >= 0.5 else rounded_down
        alternative = rounded_down if preferred == rounded_up else rounded_up

        preferred_child = node.child(var_index, preferred)
        alternative_child = node.child(var_index, alternative)
        return alternative_child, preferred_child

    def _is_integral(self, x: np.ndarray) -> bool:
        return bool(np.all(np.abs(x - np.rint(x)) <= self.tolerance))

    def _fractional_indices(self, x: np.ndarray) -> np.ndarray:
        return np.flatnonzero((x > self.tolerance) & (x < 1.0 - self.tolerance) & (np.abs(x - np.rint(x)) > self.tolerance))


def build_problem(
    c: ArrayLike,
    A_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    A_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: Iterable[tuple[float, float]] | None = None,
) -> MILPProblem:
    """Convenience constructor."""

    return MILPProblem(
        c=np.asarray(c, dtype=float),
        A_ub=None if A_ub is None else np.asarray(A_ub, dtype=float),
        b_ub=None if b_ub is None else np.asarray(b_ub, dtype=float),
        A_eq=None if A_eq is None else np.asarray(A_eq, dtype=float),
        b_eq=None if b_eq is None else np.asarray(b_eq, dtype=float),
        bounds=None if bounds is None else list(bounds),
    )


__all__ = [
    "BBNode",
    "BBSolution",
    "BranchAndBoundSolver",
    "LPRelaxationResult",
    "MILPProblem",
    "MWUScorer",
    "build_problem",
]


if __name__ == "__main__":
    from pathlib import Path

    import networkx as nx

    print("=== 1. Generating Test Instance ===")
    # Minimum vertex cover test instance on a reproducible random graph.
    G = nx.gnp_random_graph(n=10, p=0.4, seed=42)
    n_vars = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    degree_sequence = [deg for _, deg in G.degree()]

    print(f"Created graph with {n_vars} nodes and {n_edges} edges.")
    print(f"Density: {density:.4f}")

    c = np.ones(n_vars, dtype=float)

    # For each edge (u, v): x_u + x_v >= 1  ->  -x_u - x_v <= -1.
    A_ub = np.zeros((n_edges, n_vars), dtype=float)
    b_ub = -np.ones(n_edges, dtype=float)

    for edge_idx, (u, v) in enumerate(G.edges()):
        A_ub[edge_idx, u] = -1.0
        A_ub[edge_idx, v] = -1.0

    output_dir = Path(__file__).resolve().parent / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_csv_path = output_dir / "mvc_graph_edges.csv"
    with edge_csv_path.open("w", encoding="utf-8") as edge_file:
        edge_file.write("u,v\n")
        for u, v in G.edges():
            edge_file.write(f"{u},{v}\n")

    graphml_path = output_dir / "mvc_graph.graphml"
    nx.write_graphml(G, graphml_path)

    summary_path = output_dir / "mvc_graph_summary.txt"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("Minimum Vertex Cover Test Graph Summary\n")
        summary_file.write(f"nodes: {n_vars}\n")
        summary_file.write(f"edges: {n_edges}\n")
        summary_file.write(f"density: {density:.6f}\n")
        summary_file.write(f"min_degree: {min(degree_sequence)}\n")
        summary_file.write(f"max_degree: {max(degree_sequence)}\n")
        summary_file.write(f"avg_degree: {float(np.mean(degree_sequence)):.6f}\n")

    print("\n=== 2. Building MILP Problem Space ===")
    problem = build_problem(c=c, A_ub=A_ub, b_ub=b_ub)

    print("\n=== 3. Initializing Solver Pipeline ===")
    solver = BranchAndBoundSolver(
        tolerance=1e-5,
        mwu_learning_rate=0.35,
        mwu_iterations=4,
    )

    print("\n=== 4. Executing Tree Search (Diving DFS) ===")
    solution = solver.solve(problem)

    print("\n=== 5. Optimization Performance Summary ===")
    print(f"Status:            {solution.status}")
    print(f"Optimal Value:     {solution.objective_value}")
    print(f"Total Nodes Seen:  {solution.node_count}")
    print(f"Explored (Popped): {solution.explored_nodes}")
    print(f"Pruned (Fathomed): {solution.pruned_nodes}")
    print(f"Graph Edge List CSV: {edge_csv_path}")
    print(f"Graph GraphML:       {graphml_path}")
    print(f"Graph Summary TXT:   {summary_path}")

    if solution.x is not None:
        selected_nodes = np.flatnonzero(solution.x > 0.5).tolist()
        print(f"Selected Vertex Set Indices: {selected_nodes}")