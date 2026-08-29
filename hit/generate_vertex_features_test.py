from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hgr_reader import read_hgr
from problem import build_hitting_set_problem
from mwua import MWUAFeatureExtractor

KERNEL_DIR = Path("../PACE2025-instances/private/hs/exact")
OUTPUT = Path("data/vertex_features_test.csv")
WORKERS = 16


def extract_vertex_features(path):
    instance = read_hgr(str(path))
    problem = build_hitting_set_problem(instance)

    n = problem.num_variables
    edges = problem.hyperedges

    mwua = MWUAFeatureExtractor(
        rounds=50000,
        eps=0.25,
        delta=1e-6,
        time_limit=90.0,
        verbose=False,
    )

    result = mwua.compute(problem)

    weights = result.final_weights
    x_avg = result.x_avg
    certainty = result.certainty

    degree = np.zeros(n, dtype=np.float64)
    weighted_degree = np.zeros(n, dtype=np.float64)
    participation = np.zeros(n, dtype=np.float64)
    mwu_weighted_degree = np.zeros(n, dtype=np.float64)
    mwu_weight_sum = np.zeros(n, dtype=np.float64)
    mwu_weight_max = np.zeros(n, dtype=np.float64)
    edge_size_sum = np.zeros(n, dtype=np.float64)
    edge_size_max = np.zeros(n, dtype=np.float64)
    singleton_count = np.zeros(n, dtype=np.float64)
    binary_count = np.zeros(n, dtype=np.float64)

    for edge_idx, edge in enumerate(edges):
        size = len(edge)
        weight = weights[edge_idx]

        for v in edge:
            degree[v] += 1.0
            participation[v] += 1.0

            weighted_degree[v] += size

            mwu_weighted_degree[v] += weight
            mwu_weight_sum[v] += weight
            mwu_weight_max[v] = max(
                mwu_weight_max[v],
                weight,
            )

            edge_size_sum[v] += size
            edge_size_max[v] = max(
                edge_size_max[v],
                size,
            )

            if size == 1:
                singleton_count[v] += 1.0

            if size == 2:
                binary_count[v] += 1.0

    mean_incident_mwu = np.divide(
        mwu_weight_sum,
        participation,
        out=np.zeros(n, dtype=np.float64),
        where=participation > 0,
    )

    mean_incident_edge_size = np.divide(
        edge_size_sum,
        participation,
        out=np.zeros(n, dtype=np.float64),
        where=participation > 0,
    )

    rows = []

    for v in range(n):
        rows.append(
            {
                "instance": path.stem.replace("private_", ""),
                "vertex": v + 1,
                "degree": degree[v],
                "weighted_degree": weighted_degree[v],
                "hyperedge_participation": participation[v],
                "mwu_weighted_degree": mwu_weighted_degree[v],
                "mean_incident_mwu": mean_incident_mwu[v],
                "max_incident_mwu": mwu_weight_max[v],
                "mean_incident_edge_size": mean_incident_edge_size[v],
                "max_incident_edge_size": edge_size_max[v],
                "singleton_participation": singleton_count[v],
                "binary_participation": binary_count[v],
                "x_avg": x_avg[v],
                "certainty": certainty[v],
            }
        )

    return rows


def process_instance(path):
    return extract_vertex_features(path)


def main():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        KERNEL_DIR.glob("*.hgr")
    )

    print(
        f"Found {len(files)} kernels"
    )

    rows = []

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        for i, result in enumerate(
            executor.map(
                process_instance,
                files,
            ),
            start=1,
        ):
            rows.extend(result)

            print(
                f"[{i:3d}/{len(files)}] "
                f"{files[i - 1].name} "
                f"({len(result)} vertices)",
                flush=True,
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )
    print(
        f"Shape: {df.shape}"
    )
    print()

    print(
        df.head()
    )

    print()
    print(
        "Missing values:"
    )
    print(
        df.isna().sum()
    )

    print()
    print(
        "Instances:",
        df["instance"].nunique(),
    )

    print(
        "Vertices:",
        len(df),
    )


if __name__ == "__main__":
    main()
