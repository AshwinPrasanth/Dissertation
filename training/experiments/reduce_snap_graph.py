import sys
import time
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAMIS_BUILD = PROJECT_ROOT / "CHSZLabLib" / "build-kamis"

sys.path.insert(0, str(KAMIS_BUILD))

import _kamis


DATASET_PATH = PROJECT_ROOT / "training copy"/ "graphs" / "ca_family" /"delaunay_n10.txt"


def graph_to_csr(G):
    G = nx.convert_node_labels_to_integers(
        G,
        first_label=0,
        ordering="sorted",
    )

    n = G.number_of_nodes()

    xadj = np.zeros(n + 1, dtype=np.int32)
    adjncy = []

    for node in range(n):
        neighbors = sorted(G.neighbors(node))
        adjncy.extend(neighbors)
        xadj[node + 1] = len(adjncy)

    adjncy = np.asarray(adjncy, dtype=np.int32)
    vwgt = np.asarray([], dtype=np.int32)

    return G, xadj, adjncy, vwgt


def load_snap_graph(path):
    print("Loading SNAP graph...")

    start = time.perf_counter()

    G = nx.read_edgelist(
        path,
        comments="#",
        nodetype=int,
        create_using=nx.Graph(),
        data=False
    )

    G.remove_edges_from(nx.selfloop_edges(G))

    elapsed = time.perf_counter() - start

    print(f"Load time          : {elapsed:.6f} s")

    return G


def main():
    print("=" * 70)
    print("SNAP REDUMIS KERNELIZATION")
    print("=" * 70)

    print()
    print(f"Dataset: {DATASET_PATH.name}")

    G = load_snap_graph(DATASET_PATH)

    original_vertices = G.number_of_nodes()
    original_edges = G.number_of_edges()

    average_degree = (
        2.0 * original_edges / original_vertices
        if original_vertices > 0
        else 0.0
    )

    print()
    print("ORIGINAL GRAPH")
    print("-" * 70)

    print(f"Vertices           : {original_vertices}")
    print(f"Edges              : {original_edges}")
    print(f"Average degree     : {average_degree:.4f}")

    print()
    print("Converting graph to CSR...")

    csr_start = time.perf_counter()

    G, xadj, adjncy, vwgt = graph_to_csr(G)

    csr_time = time.perf_counter() - csr_start

    print(f"CSR conversion time: {csr_time:.6f} s")
    print(f"CSR adjacency size : {len(adjncy)}")

    print()
    print("Running ReduMIS kernelization...")

    reduction_start = time.perf_counter()

    core_xadj, core_adjncy, reverse_mapping = (
        _kamis.redumis_kernel(
            xadj,
            adjncy,
            vwgt,
        )
    )

    reduction_time = time.perf_counter() - reduction_start

    core_vertices = len(core_xadj) - 1
    core_edges = len(core_adjncy) // 2

    core_ratio = (
        core_vertices / original_vertices
        if original_vertices > 0
        else 0.0
    )

    removed_ratio = 1.0 - core_ratio

    edge_core_ratio = (
        core_edges / original_edges
        if original_edges > 0
        else 0.0
    )

    print()
    print("IRREDUCIBLE CORE")
    print("-" * 70)

    print(f"Core vertices      : {core_vertices}")
    print(f"Core edges         : {core_edges}")
    print(f"Core ratio         : {core_ratio:.6f}")
    print(f"Vertices removed   : {removed_ratio:.2%}")
    print(f"Edge core ratio    : {edge_core_ratio:.6f}")
    print(f"Reduction time     : {reduction_time:.6f} s")

    print()
    print("MAPPING")
    print("-" * 70)

    print(f"Mapping size       : {len(reverse_mapping)}")
    print("First 20 mappings  :")
    print(reverse_mapping[:20])

    print()
    print("=" * 70)

    if core_vertices == 0:
        print("RESULT: Graph completely reduced.")
        print("MWUA and branching are unnecessary.")

    elif core_ratio <= 0.25:
        print("RESULT: Strong kernelization.")
        print("MWUA computational domain reduced substantially.")

    elif core_ratio <= 0.75:
        print("RESULT: Partial kernelization.")
        print("Suitable candidate for raw-vs-core MWUA comparison.")

    else:
        print("RESULT: Weak kernelization.")
        print("Most vertices remain in the irreducible core.")

    print("=" * 70)


if __name__ == "__main__":
    main()