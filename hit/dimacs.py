import networkx as nx


def load_dimacs_clq(path):

    G = nx.Graph()

    with open(path, "r") as f:

        for line in f:

            if line.startswith("c"):
                continue

            if line.startswith("p"):

                parts = line.split()

                n = int(parts[2])

                G.add_nodes_from(
                    range(1, n + 1)
                )

                continue

            if line.startswith("e"):

                _, u, v = line.split()

                G.add_edge(
                    int(u),
                    int(v),
                )

    return G