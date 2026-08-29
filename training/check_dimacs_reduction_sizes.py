import sys
from pathlib import Path

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "training" / "graphs" / "Dimacs"
KAMIS_BUILD = PROJECT_ROOT / "CHSZLabLib" / "build-kamis"

sys.path.insert(0, str(KAMIS_BUILD))

import _kamis


def load_mtx_graph(path):
    G = nx.Graph()
    dimensions_read = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("%"):
                continue

            parts = line.split()

            if not dimensions_read:
                n = int(parts[0])
                G.add_nodes_from(range(n))
                dimensions_read = True
                continue

            if len(parts) < 2:
                continue

            u = int(parts[0]) - 1
            v = int(parts[1]) - 1

            if u != v:
                G.add_edge(u, v)

    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def graph_to_csr(G):
    G = nx.convert_node_labels_to_integers(
        G,
        ordering="sorted",
    )

    xadj = [0]
    adjncy = []

    for v in range(G.number_of_nodes()):
        adjncy.extend(sorted(G.neighbors(v)))
        xadj.append(len(adjncy))

    return (
        np.asarray(xadj, dtype=np.int32),
        np.asarray(adjncy, dtype=np.int32),
        np.asarray([], dtype=np.int32),
    )


def family_name(instance_name):
    name = instance_name.lower()

    for prefix in [
        "brock",
        "c-fat",
        "dsjc",
        "gen",
        "hamming",
        "johnson",
        "keller",
        "mann",
        "p-hat",
        "sanr",
        "san",
    ]:
        if name.startswith(prefix):
            return prefix

    if name.startswith("c"):
        return "C"

    return "other"


def main():
    graph_files = sorted(GRAPH_DIR.rglob("*.mtx"))

    if not graph_files:
        raise RuntimeError(
            f"No .mtx files found under {GRAPH_DIR}"
        )

    print(f"DIMACS .mtx files found: {len(graph_files)}")
    print()

    print("=" * 132)
    print("DIMACS COMPLEMENT -> REDUMIS IRREDUCIBLE CORE CHECK")
    print("=" * 132)

    print(
        f"{'Instance':<24}"
        f"{'Family':<10}"
        f"{'Clique n':>10}"
        f"{'Clique m':>12}"
        f"{'MIS m':>12}"
        f"{'Core n':>10}"
        f"{'Core m':>12}"
        f"{'Remain %':>12}"
        f"{'Reduction %':>14}"
    )

    print("-" * 132)

    for graph_file in graph_files:
        instance_name = graph_file.parent.name
        family = family_name(instance_name)

        try:
            clique_graph = load_mtx_graph(graph_file)
            mis_graph = nx.complement(clique_graph)

            clique_n = clique_graph.number_of_nodes()
            clique_m = clique_graph.number_of_edges()
            mis_m = mis_graph.number_of_edges()

            xadj, adjncy, vwgt = graph_to_csr(mis_graph)

            core_xadj, core_adjncy, reverse_mapping = (
                _kamis.redumis_kernel(
                    xadj,
                    adjncy,
                    vwgt,
                )
            )

            core_xadj = np.asarray(
                core_xadj,
                dtype=np.int32,
            )

            core_adjncy = np.asarray(
                core_adjncy,
                dtype=np.int32,
            )

            core_n = len(core_xadj) - 1
            core_m = len(core_adjncy) // 2

            remain_ratio = (
                core_n / clique_n
                if clique_n
                else 0.0
            )

            reduction_ratio = 1.0 - remain_ratio

            print(
                f"{instance_name:<24}"
                f"{family:<10}"
                f"{clique_n:>10}"
                f"{clique_m:>12}"
                f"{mis_m:>12}"
                f"{core_n:>10}"
                f"{core_m:>12}"
                f"{100.0 * remain_ratio:>11.2f}%"
                f"{100.0 * reduction_ratio:>13.2f}%"
            )

        except Exception as error:
            print(
                f"{instance_name:<24}"
                f"{family:<10}"
                f" ERROR: {repr(error)}"
            )


if __name__ == "__main__":
    main()
