from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from hgr_reader import read_hgr
from problem import build_hitting_set_problem
from mwua import MWUAFeatureExtractor

KERNEL_DIR = Path("../PACE2025-reduced/exact")
OUTPUT = Path("data/instance_features.csv")

def extract_features(path):
    instance = read_hgr(str(path))
    problem = build_hitting_set_problem(instance)

    n = problem.num_variables
    edges = problem.hyperedges
    m = len(edges)

    degrees = np.zeros(n, dtype=np.float64)
    for edge in edges:
        for v in edge:
            degrees[v] += 1.0

    edge_sizes = np.array([len(edge) for edge in edges], dtype=np.float64)

    mean_degree = float(np.mean(degrees))
    degree_variance = float(np.var(degrees))
    max_degree = float(np.max(degrees))

    mean_edge_size = float(np.mean(edge_sizes))
    edge_size_variance = float(np.var(edge_sizes))
    min_edge_size = float(np.min(edge_sizes))
    max_edge_size = float(np.max(edge_sizes))

    singleton_fraction = float(np.mean(edge_sizes == 1))
    binary_fraction = float(np.mean(edge_sizes == 2))
    incidence_density = float(np.sum(degrees)) / float(n * m) if (n * m) > 0 else 0.0

    # FIX 1: Efficient memory-safe component extraction
    components = connected_components_bipartite(n, edges)

    mwua = MWUAFeatureExtractor(
        rounds=50000,
        eps=0.25,
        delta=1e-6,
        time_limit=90.0,
        verbose=False,
    )

    result = mwua.compute(problem)
    weights = result.final_weights

    mwu_mean = float(np.mean(weights))
    mwu_variance = float(np.var(weights))
    mwu_max = float(np.max(weights))
    mwu_concentration = mwu_max / float(np.sum(weights)) if np.sum(weights) > 0 else 0.0

    positive_weights = weights[weights > 0]
    
    # Normalize positive weights so they sum to 1.0 for strict Shannon Entropy
    if len(positive_weights) > 0:
        p = positive_weights / np.sum(positive_weights)
        mwu_entropy = float(-np.sum(p * np.log(p)))
    else:
        mwu_entropy = 0.0

    return {
        "n_vertices": n,
        "n_edges": m,
        "vertex_edge_ratio": n / m if m > 0 else 0.0,
        "mean_degree": mean_degree,
        "degree_variance": degree_variance,
        "max_degree": max_degree,
        "mean_edge_size": mean_edge_size,
        "edge_size_variance": edge_size_variance,
        "min_edge_size": min_edge_size,
        "max_edge_size": max_edge_size,
        "singleton_fraction": singleton_fraction,
        "binary_fraction": binary_fraction,
        "incidence_density": incidence_density,
        "connected_components": components,
        "mwu_mean": mwu_mean,
        "mwu_variance": mwu_variance,
        "mwu_max": mwu_max,
        "mwu_concentration": mwu_concentration,
        "mwu_entropy": mwu_entropy,
    }

def connected_components_bipartite(n, edges):
    """
    Memory-safe component finder. Traverses Vertex -> Edge -> Vertex.
    Prevents clique memory explosion on large hyperedges.
    """
    # Map vertices to the indices of hyperedges they belong to
    v_to_e = [[] for _ in range(n)]
    for e_idx, edge in enumerate(edges):
        for v in edge:
            v_to_e[v].append(e_idx)

    v_visited = np.zeros(n, dtype=bool)
    e_visited = np.zeros(len(edges), dtype=bool)
    components = 0

    for start_v in range(n):
        if v_visited[start_v]:
            continue

        components += 1
        stack = [start_v]
        v_visited[start_v] = True

        while stack:
            u = stack.pop()
            # Find all hyperedges connected to this vertex
            for e_idx in v_to_e[u]:
                if e_visited[e_idx]:
                    continue
                e_visited[e_idx] = True
                
                # Find all neighbors inside those hyperedges
                for v in edges[e_idx]:
                    if not v_visited[v]:
                        v_visited[v] = True
                        stack.append(v)
                        
    return components

def process_instance(path):
    features = extract_features(path)
    return {
        "instance": path.stem,
        **features,
    }

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(KERNEL_DIR.glob("*.hgr"))

    print(f"Found {len(files)} kernels")
    
    # FIX 2: Dynamic optimization to harness your full 32 cores safely
    import multiprocessing
    system_cores = multiprocessing.cpu_count()
    workers = min(12, system_cores) 
    print(f"Using {workers} worker processes out of {system_cores} available cores.")

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, row in enumerate(
            executor.map(process_instance, files), start=1
        ):
            rows.append(row)
            print(f"[{i:3d}/{len(files)}] {row['instance']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}\nShape: {df.shape}\n")
    print(df.head())
    print("\n", df.describe())

if __name__ == "__main__":
    main()
