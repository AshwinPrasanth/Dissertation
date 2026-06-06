from typing import Dict, Optional

import numpy as np
from scipy.optimize import linprog

from problem import MILPProblem


def solve_lp_relaxation(
    problem: MILPProblem,
    fixings: Optional[Dict[int, int]] = None,
):
    """
    Solve LP relaxation:

        min c^T x

        subject to:
            A_ub x <= b_ub
            A_eq x = b_eq
            0 <= x <= 1

    Some variables may be fixed:

        x_i = 0
        x_i = 1
    """

    if fixings is None:
        fixings = {}

    bounds = []

    for i in range(problem.num_variables):

        if i in fixings:
            value = float(fixings[i])
            bounds.append((value, value))

        else:
            bounds.append((0.0, 1.0))

    result = linprog(
        c=problem.c,
        A_ub=problem.A_ub,
        b_ub=problem.b_ub,
        A_eq=problem.A_eq,
        b_eq=problem.b_eq,
        bounds=bounds,
        method="highs",
    )

    return result

import networkx as nx

from problem import build_vertex_cover_problem
#from lp import solve_lp_relaxation

G = nx.cycle_graph(4)

problem = build_vertex_cover_problem(G)

result = solve_lp_relaxation(problem)

print("status:", result.success)
print("objective:", result.fun)
print("solution:", result.x)
