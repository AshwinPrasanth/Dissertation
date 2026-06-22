'''from pyscipopt import Model, quicksum


def solve_mis(n, edges):

    model = Model("MIS")

    # Binary variable for each vertex
    x = {}

    for v in range(n):
        x[v] = model.addVar(
            name=f"x_{v}",
            vtype="B",
        )

    # Maximize independent set size
    model.setObjective(
        quicksum(x[v] for v in range(n)),
        "maximize"
    )

    # Edge constraints
    for u, v in edges:
        model.addCons(
            x[u] + x[v] <= 1
        )

    model.optimize()

    print("Status:", model.getStatus())
    print("Objective:", model.getObjVal())

    solution = []

    for v in range(n):
        solution.append(
            round(
                model.getVal(x[v])
            )
        )

    print("Solution:", solution)


if __name__ == "__main__":

    n = 4

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
    ]

    solve_mis(
        n,
        edges,
    )'''
    
import networkx as nx
from pyscipopt import Model, quicksum


def solve_graph(G):

    n = G.number_of_nodes()

    model = Model("MIS")
    model.setPresolve(False)
    model.setSeparating(False)
    model.setHeuristics(0)

    x = {}

    for v in range(n):
        x[v] = model.addVar(
            name=f"x_{v}",
            vtype="B",
        )

    model.setObjective(
        quicksum(x[v] for v in range(n)),
        "maximize"
    )

    for u, v in G.edges():
        model.addCons(
            x[u] + x[v] <= 1
        )

    model.optimize()

    print("Status:", model.getStatus())
    print("Objective:", model.getObjVal())
    print("Nodes:", model.getNNodes())


if __name__ == "__main__":

    #G = nx.erdos_renyi_graph(30,0.2,seed=0,)
    G = nx.barabasi_albert_graph(
    500000,
    3,
    seed=0,
)
    print("Vertices:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    solve_graph(G)