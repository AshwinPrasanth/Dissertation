import networkx as nx


def load_snap_graph(path):

    G = nx.Graph()

    with open(path, "r") as f:

        for line in f:

            if line.startswith("#"):
                continue

            parts = line.strip().split()

            if len(parts) != 2:
                continue

            u = int(parts[0])
            v = int(parts[1])

            G.add_edge(u, v)

    return G

graphs = [
    "CA-GrQc.txt",
    "CA-HepTh.txt",
    "Wiki-Vote.txt",
    "Email-Enron.txt",
]

for graph in graphs:

    G = load_snap_graph(
        f"graphs/{graph}"
    )

    print(
        graph,
        G.number_of_nodes(),
        G.number_of_edges(),
    )