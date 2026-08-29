import csv
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHSZLABLIB_ROOT = PROJECT_ROOT / "CHSZLabLib"
KAMIS_BUILD = CHSZLABLIB_ROOT / "build-kamis"

sys.path.insert(0, str(KAMIS_BUILD))

import _kamis


GRAPH_SIZES = [1000, 5000, 10000]
TARGET_DEGREES = [2, 5, 10, 20]
SEEDS = [42, 43, 44]

OUTPUT_DIR = PROJECT_ROOT / "anytime" / "results"
OUTPUT_FILE = OUTPUT_DIR / "redumis_kernelization_results.csv"


def graph_to_csr(G):
    nodes = list(G.nodes())
    node_mapping = {node: i for i, node in enumerate(nodes)}

    xadj = [0]
    adjncy = []

    for node in nodes:
        neighbors = sorted(
            node_mapping[neighbor]
            for neighbor in G.neighbors(node)
        )

        adjncy.extend(neighbors)
        xadj.append(len(adjncy))

    return (
        np.asarray(xadj, dtype=np.int32),
        np.asarray(adjncy, dtype=np.int32),
        np.asarray([], dtype=np.int32),
    )


def run_redumis_kernel(G):
    xadj, adjncy, vwgt = graph_to_csr(G)

    start = time.perf_counter()

    core_xadj, core_adjncy, reverse_mapping = (
        _kamis.redumis_kernel(
            xadj,
            adjncy,
            vwgt,
        )
    )

    reduction_time = time.perf_counter() - start

    core_vertices = len(core_xadj) - 1
    core_edges = len(core_adjncy) // 2

    return {
        "core_vertices": core_vertices,
        "core_edges": core_edges,
        "reduction_time": reduction_time,
        "reverse_mapping": reverse_mapping,
    }


def generate_erdos_renyi(n, target_degree, seed):
    p = target_degree / (n - 1)

    return nx.fast_gnp_random_graph(
        n,
        p,
        seed=seed,
    )


def generate_barabasi_albert(n, target_degree, seed):
    m = max(1, round(target_degree / 2))
    m = min(m, n - 1)

    return nx.barabasi_albert_graph(
        n,
        m,
        seed=seed,
    )


def generate_watts_strogatz(n, target_degree, seed):
    k = max(2, int(target_degree))

    if k % 2 != 0:
        k += 1

    k = min(k, n - 1)

    if k % 2 != 0:
        k -= 1

    return nx.watts_strogatz_graph(
        n,
        k,
        0.1,
        seed=seed,
    )


def generate_random_regular(n, target_degree, seed):
    degree = min(int(target_degree), n - 1)

    if (n * degree) % 2 != 0:
        degree -= 1

    if degree <= 0:
        degree = 2

    return nx.random_regular_graph(
        degree,
        n,
        seed=seed,
    )


GRAPH_GENERATORS = {
    "erdos_renyi": generate_erdos_renyi,
    "barabasi_albert": generate_barabasi_albert,
    "watts_strogatz": generate_watts_strogatz,
    "random_regular": generate_random_regular,
}


def analyze_graph(
    family,
    n,
    target_degree,
    seed,
    G,
):
    original_vertices = G.number_of_nodes()
    original_edges = G.number_of_edges()

    if original_vertices > 0:
        actual_average_degree = (
            2.0 * original_edges / original_vertices
        )
    else:
        actual_average_degree = 0.0

    kernel_result = run_redumis_kernel(G)

    core_vertices = kernel_result["core_vertices"]
    core_edges = kernel_result["core_edges"]
    reduction_time = kernel_result["reduction_time"]

    if original_vertices > 0:
        core_ratio = core_vertices / original_vertices
    else:
        core_ratio = 0.0

    removed_ratio = 1.0 - core_ratio

    if original_edges > 0:
        edge_core_ratio = core_edges / original_edges
    else:
        edge_core_ratio = 0.0

    return {
        "family": family,
        "n": n,
        "target_degree": target_degree,
        "actual_average_degree": actual_average_degree,
        "seed": seed,
        "original_vertices": original_vertices,
        "original_edges": original_edges,
        "core_vertices": core_vertices,
        "core_edges": core_edges,
        "core_ratio": core_ratio,
        "removed_ratio": removed_ratio,
        "edge_core_ratio": edge_core_ratio,
        "reduction_time": reduction_time,
    }


def print_result(result):
    print()
    print("=" * 70)

    print(
        f"Family             : {result['family']}"
    )

    print(
        f"n                  : {result['n']}"
    )

    print(
        f"Target degree      : {result['target_degree']}"
    )

    print(
        f"Actual avg degree  : "
        f"{result['actual_average_degree']:.4f}"
    )

    print(
        f"Seed               : {result['seed']}"
    )

    print("-" * 70)

    print(
        f"Original vertices  : "
        f"{result['original_vertices']}"
    )

    print(
        f"Original edges     : "
        f"{result['original_edges']}"
    )

    print(
        f"Core vertices      : "
        f"{result['core_vertices']}"
    )

    print(
        f"Core edges         : "
        f"{result['core_edges']}"
    )

    print(
        f"Core ratio         : "
        f"{result['core_ratio']:.4f}"
    )

    print(
        f"Vertices removed   : "
        f"{result['removed_ratio']:.2%}"
    )

    print(
        f"Edge core ratio    : "
        f"{result['edge_core_ratio']:.4f}"
    )

    print(
        f"Reduction time     : "
        f"{result['reduction_time']:.6f} s"
    )


def save_results(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "family",
        "n",
        "target_degree",
        "actual_average_degree",
        "seed",
        "original_vertices",
        "original_edges",
        "core_vertices",
        "core_edges",
        "core_ratio",
        "removed_ratio",
        "edge_core_ratio",
        "reduction_time",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    print()
    print()
    print("=" * 100)
    print("REDUMIS KERNELIZATION SUMMARY")
    print("=" * 100)

    header = (
        f"{'Family':<20}"
        f"{'n':>8}"
        f"{'Degree':>10}"
        f"{'Seed':>8}"
        f"{'Original':>12}"
        f"{'Core':>12}"
        f"{'Core Ratio':>14}"
        f"{'Removed':>12}"
        f"{'Time':>12}"
    )

    print(header)
    print("-" * 100)

    for result in results:
        row = (
            f"{result['family']:<20}"
            f"{result['n']:>8}"
            f"{result['target_degree']:>10}"
            f"{result['seed']:>8}"
            f"{result['original_vertices']:>12}"
            f"{result['core_vertices']:>12}"
            f"{result['core_ratio']:>14.4f}"
            f"{result['removed_ratio']:>11.2%}"
            f"{result['reduction_time']:>11.4f}s"
        )

        print(row)


def main():
    results = []

    total_experiments = (
        len(GRAPH_GENERATORS)
        * len(GRAPH_SIZES)
        * len(TARGET_DEGREES)
        * len(SEEDS)
    )

    experiment_number = 0

    print("=" * 70)
    print("REDUMIS KERNELIZATION ANALYSIS")
    print("=" * 70)

    print(
        f"Total experiments: {total_experiments}"
    )

    for family, generator in GRAPH_GENERATORS.items():
        for n in GRAPH_SIZES:
            for target_degree in TARGET_DEGREES:
                for seed in SEEDS:
                    experiment_number += 1

                    print()
                    print(
                        f"[{experiment_number}/"
                        f"{total_experiments}]"
                    )

                    print(
                        "Generating "
                        f"{family}, "
                        f"n={n}, "
                        f"degree={target_degree}, "
                        f"seed={seed}"
                    )

                    try:
                        generation_start = time.perf_counter()

                        G = generator(
                            n,
                            target_degree,
                            seed,
                        )

                        generation_time = (
                            time.perf_counter()
                            - generation_start
                        )

                        print(
                            "Graph generation time: "
                            f"{generation_time:.6f} s"
                        )

                        result = analyze_graph(
                            family,
                            n,
                            target_degree,
                            seed,
                            G,
                        )

                        print_result(result)

                        results.append(result)

                        del G

                    except Exception as error:
                        print()
                        print("EXPERIMENT FAILED")

                        print(
                            f"Family: {family}"
                        )

                        print(
                            f"n: {n}"
                        )

                        print(
                            f"Target degree: "
                            f"{target_degree}"
                        )

                        print(
                            f"Seed: {seed}"
                        )

                        print(
                            f"Error: {error}"
                        )

    save_results(results)
    print_summary(results)

    print()
    print(
        f"Results written to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()