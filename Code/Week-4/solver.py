from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from lp import solve_lp_relaxation
from branching import BranchingStrategy, MostFractionalBranching
from problem import MILPProblem


TOL = 1e-6


@dataclass
class SolveResult:
    objective: float
    solution: np.ndarray
    nodes_explored: int


class BranchAndBoundSolver:

    def __init__(
        self,
        branching: Optional[BranchingStrategy] = None,
    ):
        self.branching = branching or MostFractionalBranching()

        self.best_obj = float("inf")
        self.best_solution = None

        self.nodes_explored = 0

    def solve(self, problem: MILPProblem) -> SolveResult:

        self.best_obj = float("inf")
        self.best_solution = None
        self.nodes_explored = 0

        self._dfs(problem, fixings={})

        return SolveResult(
            objective=self.best_obj,
            solution=self.best_solution,
            nodes_explored=self.nodes_explored,
        )

    def _dfs(
        self,
        problem: MILPProblem,
        fixings: Dict[int, int],
    ):

        self.nodes_explored += 1

        # --------------------------------------------------
        # Solve LP relaxation
        # --------------------------------------------------

        lp = solve_lp_relaxation(problem, fixings)

        if not lp.success:
            return

        lp_bound = lp.fun

        # --------------------------------------------------
        # Bound pruning
        # --------------------------------------------------

        if lp_bound >= self.best_obj - TOL:
            return

        x = lp.x

        # --------------------------------------------------
        # Integer solution?
        # --------------------------------------------------

        is_integer = np.all(
            np.abs(x - np.round(x)) < TOL
        )

        if is_integer:

            obj = float(problem.c @ x)

            if obj < self.best_obj:

                self.best_obj = obj
                self.best_solution = np.round(x).astype(int)

            return

        # --------------------------------------------------
        # Branch
        # --------------------------------------------------

        branch_var = self.branching.select(x)

        if branch_var == -1:
            return

        # Left child: x_i = 0

        left_fixings = dict(fixings)
        left_fixings[branch_var] = 0

        self._dfs(
            problem,
            left_fixings,
        )

        # Right child: x_i = 1

        right_fixings = dict(fixings)
        right_fixings[branch_var] = 1

        self._dfs(
            problem,
            right_fixings,
        )

import networkx as nx

from problem import build_vertex_cover_problem
#from solver import BranchAndBoundSolver

G = nx.cycle_graph(4)

problem = build_vertex_cover_problem(G)

solver = BranchAndBoundSolver()

result = solver.solve(problem)

print("objective:", result.objective)
print("solution:", result.solution)
print("nodes:", result.nodes_explored)
