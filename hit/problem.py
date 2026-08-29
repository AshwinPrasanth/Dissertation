from dataclasses import dataclass
from typing import List, Optional
from hgr_reader import HittingSetInstance
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
#inequality constraints
    A_ub: Optional[np.ndarray] = None
    b_ub: Optional[np.ndarray] = None
#equality constraints
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None
#variable names
    variable_names: Optional[List[str]] = None
    # add graph structure for vertex cover problems (greedy oracle)
    graph: Optional[nx.Graph] = None

    hyperedges: Optional[List[np.ndarray]] = None
    num_hyperedges: int = 0
    
    problem_type: str = ""

    @property
    def num_variables(self) -> int:
        return len(self.c)

    @property
    def num_constraints(self) -> int:
        n_ineq = 0 if self.A_ub is None else len(self.A_ub)
        n_eq = 0 if self.A_eq is None else len(self.A_eq)
        return n_ineq + n_eq


def build_hitting_set_problem(
    instance: HittingSetInstance,
) -> MILPProblem:

    num_vertices = instance.num_vertices
    hyperedges = instance.hyperedges

    c = np.ones(
        num_vertices,
        dtype=float,
    )

    A_ub = []
    b_ub = []

    processed_edges = []

    for hyperedge in hyperedges:

        row = np.zeros(
            num_vertices,
            dtype=float,
        )

        edge = np.array(
            hyperedge,
            dtype=int,
        )

        row[edge] = -1.0

        A_ub.append(row)

        b_ub.append(-1.0)

        processed_edges.append(edge)

    A_ub = np.array(
        A_ub,
        dtype=float,
    )

    b_ub = np.array(
        b_ub,
        dtype=float,
    )

    return MILPProblem(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        hyperedges=processed_edges,
        num_hyperedges=len(processed_edges),
        variable_names=[
            str(i)
            for i in range(num_vertices)
        ],
        problem_type="hittingset",
    )

'''def build_mis_problem(
    G,
):
    """
    Maximum Independent Set

    maximize:
        sum x_v

    subject to:
        x_u + x_v <= 1
        for every edge (u,v)

    converted to minimization:

        minimize:
            -sum x_v
    """

    vertices = list(G.nodes())

    node_to_idx = {
        v: i
        for i, v in enumerate(vertices)
    }

    n = len(vertices)

    c = -np.ones(n)

    A_ub = []
    b_ub = []

    for u, v in G.edges():

        row = np.zeros(n)

        row[
            node_to_idx[u]
        ] = 1.0

        row[
            node_to_idx[v]
        ] = 1.0

        A_ub.append(row)

        b_ub.append(1.0)

    A_ub = np.array(
        A_ub,
        dtype=float,
    )

    b_ub = np.array(
        b_ub,
        dtype=float,
    )

    return MILPProblem(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        variable_names=[
            str(v)
            for v in vertices
        ],
        graph=G, problem_type="mis"
    )'''

'''def build_mis_problem(G):

    vertices = list(G.nodes())

    node_to_idx = {
        v: i
        for i, v in enumerate(vertices)
    }

    n = len(vertices)

    c = -np.ones(n)

    edges = []

    for u, v in G.edges():

        edges.append(
            (
                node_to_idx[u],
                node_to_idx[v]
            )
        )

    return MILPProblem(
        c=c,

        A_ub=None,
        b_ub=None,

        edges=edges,

        variable_names=[
            str(v)
            for v in vertices
        ],

        graph=G,

        problem_type="mis",
    )
print(problem.num_variables)
print(problem.num_constraints)
'''
