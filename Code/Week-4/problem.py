from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx


@dataclass
class MILPProblem:
    """
    Binary MILP:

        minimize     c^T x

        subject to   A_ub x <= b_ub
                     A_eq x  = b_eq
                     x_i ∈ {0,1}
    """

    c: np.ndarray

    A_ub: Optional[np.ndarray] = None
    b_ub: Optional[np.ndarray] = None

    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None

    variable_names: Optional[List[str]] = None

    @property
    def num_variables(self) -> int:
        return len(self.c)

    @property
    def num_constraints(self) -> int:
        n_ineq = 0 if self.A_ub is None else len(self.A_ub)
        n_eq = 0 if self.A_eq is None else len(self.A_eq)
        return n_ineq + n_eq


def build_vertex_cover_problem(G: nx.Graph) -> MILPProblem:
    """
    Minimum Vertex Cover

    Variables:
        x_v = 1 if vertex v is selected

    Objective:
        minimize sum_v x_v

    Constraints:
        x_u + x_v >= 1
        for every edge (u,v)

    Rewritten as:

        -x_u - x_v <= -1
    """

    vertices = list(G.nodes())
    n = len(vertices)

    vertex_to_idx = {
        v: i
        for i, v in enumerate(vertices)
    }

    # objective
    c = np.ones(n)

    A_ub = []
    b_ub = []

    for u, v in G.edges():

        row = np.zeros(n)

        row[vertex_to_idx[u]] = -1.0
        row[vertex_to_idx[v]] = -1.0

        A_ub.append(row)
        b_ub.append(-1.0)

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)

    return MILPProblem(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        variable_names=[str(v) for v in vertices],
    )

import networkx as nx
#from problem import build_vertex_cover_problem

G = nx.cycle_graph(4)

problem = build_vertex_cover_problem(G)

print(problem.num_variables)
print(problem.num_constraints)
