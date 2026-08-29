from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from problem import build_vertex_cover_problem
from lp import solve_lp_relaxation
from mwua import MWUAFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_DIR = PROJECT_ROOT / "training copy 2"/ "graphs" / "Dimacs"

KAMIS_BUILD = (
    PROJECT_ROOT
    / "CHSZLabLib"
    / "build-kamis"
)

sys.path.insert(0, str(KAMIS_BUILD))

import _kamis


def load_mtx_graph(path):
    G = nx.Graph()

    with open(path) as f:

        dimensions = False

        for line in f:

            line = line.strip()

            if not line or line.startswith("%"):
                continue

            parts = line.split()

            if not dimensions:

                n = int(parts[0])

                G.add_nodes_from(range(n))

                dimensions = True

                continue

            u = int(parts[0]) - 1
            v = int(parts[1]) - 1

            if u != v:
                G.add_edge(u, v)

    return G


def graph_to_csr(G):

    G = nx.convert_node_labels_to_integers(
        G,
        ordering="sorted",
    )

    xadj = [0]
    adjncy = []

    for v in range(G.number_of_nodes()):

        adjncy.extend(
            sorted(G.neighbors(v))
        )

        xadj.append(len(adjncy))

    return (
        np.asarray(xadj, np.int32),
        np.asarray(adjncy, np.int32),
        np.asarray([], np.int32),
    )


def csr_to_graph(xadj, adjncy):

    G = nx.Graph()

    n = len(xadj) - 1

    G.add_nodes_from(range(n))

    for u in range(n):

        start = int(xadj[u])
        end = int(xadj[u + 1])

        for v in adjncy[start:end]:

            v = int(v)

            if u < v:
                G.add_edge(u, v)

    return G


def reduce_graph(G):

    xadj, adjncy, vwgt = graph_to_csr(G)

    core_xadj, core_adjncy, reverse_mapping = (
        _kamis.redumis_kernel(
            xadj,
            adjncy,
            vwgt,
        )
    )

    return csr_to_graph(
        core_xadj,
        core_adjncy,
    )


def cosine_similarity(a, b):

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return np.nan

    return np.dot(a, b) / (na * nb)

#graph_files = sorted(GRAPH_DIR.rglob("*.mtx"))
graph_files = [
    GRAPH_DIR / "hamming6-4" / "hamming6-4.mtx"
]

rows = []
convergence_rows = []

for graph_file in tqdm(graph_files):

    G = load_mtx_graph(graph_file)

    G = reduce_graph(G)

    if G.number_of_edges() == 0:
        continue

    problem = build_vertex_cover_problem(G)

    lp = solve_lp_relaxation(problem)

    '''dual = np.asarray(lp.dual)
    values, counts = np.unique(dual, return_counts=True)

    print("\nDual distribution")
    for v, c in zip(values, counts):
        print(f"{v:.1f}: {c}")

    dual = dual / dual.sum()'''
    
    lp_primal = np.asarray(lp.x)
    
    num_fractional = np.sum(
    (lp_primal > 1e-6) &
    (lp_primal < 1 - 1e-6)
    )

    print(
        f"{graph_file.stem}: "
        f"{num_fractional} fractional vertices "
        f"out of {len(lp_primal)}"
    )

    mwua = MWUAFeatureExtractor(
        rounds=100000,
    )

    result = mwua.compute(problem)
    
    convergence_rows.append(
    {
        "graph": graph_file.stem,
        "convergence_iteration": result.convergence_iteration,
    }
)
    
    for iteration, x_avg in zip(
        result.history_iterations,
        result.history_x_avg,
    ):

        cosine = cosine_similarity(
            lp_primal,
            x_avg,
        )

        spearman = spearmanr(
            lp_primal,
            x_avg,
        ).correlation

        if np.std(lp_primal) < 1e-12 or np.std(x_avg) < 1e-12:
            pearson = np.nan
        else:
            pearson = pearsonr(
                lp_primal,
                x_avg,
            )[0]

        mae = np.mean(
            np.abs(lp_primal - x_avg)
        )

        rows.append(
            {
                "graph": graph_file.stem,
                "iteration": iteration,
                "cosine": cosine,
                "spearman": spearman,
                "pearson": pearson,
                "mae": mae,
            }
        )
        
    '''for iteration, weights in zip(
        result.history_iterations,
        result.history_weights,
    ):

        weights = weights / weights.sum()

        spearman = spearmanr(
            dual,
            weights,
        ).correlation

        cosine = cosine_similarity(
            dual,
            weights,
        )

        active = dual > 0

        if active.sum() >= 2:

            active_spearman = spearmanr(
                dual[active],
                weights[active],
            ).correlation

            active_cosine = cosine_similarity(
                dual[active],
                weights[active],
            )

        else:

            active_spearman = np.nan
            active_cosine = np.nan



        rows.append(
    {
        "graph": graph_file.stem,
        "iteration": iteration,
        "spearman": spearman,
        "cosine": cosine,
        "active_spearman": active_spearman,
        "active_cosine": active_cosine,
        "num_active": int(active.sum()),
    }
)'''

df = pd.DataFrame(rows)

df.to_csv(
    "graph_results.csv",
    index=False,
)

pd.DataFrame(convergence_rows).to_csv(
    "convergence_results.csv",
    index=False,
)

average = (
    df.groupby("iteration")
    .agg(
        mean_cosine=("cosine","mean"),
        std_cosine=("cosine","std"),

        mean_spearman=("spearman","mean"),
        std_spearman=("spearman","std"),

        mean_pearson=("pearson","mean"),
        std_pearson=("pearson","std"),

        mean_mae=("mae","mean"),
        std_mae=("mae","std"),

        num_graphs=("graph","nunique"),
    )
    .reset_index()
)

average.to_csv(
    "average_results.csv",
    index=False,
)

plt.figure(figsize=(8,5))

plt.plot(
    average["iteration"],
    average["mean_cosine"],
    label="Cosine",
)

plt.plot(
    average["iteration"],
    average["mean_pearson"],
    label="Pearson",
)

plt.plot(
    average["iteration"],
    average["mean_spearman"],
    label="Spearman",
)

plt.xlabel("MWUA iteration")
plt.ylabel("Similarity")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "average_convergence.png",
    dpi=300,
)


plt.figure(figsize=(8,5))

plt.plot(
    average["iteration"],
    average["mean_mae"],
    label="MAE",
)

plt.xlabel("MWUA iteration")
plt.ylabel("Mean Absolute Error")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "mae_convergence.png",
    dpi=300,
)

plt.show()