# benchmark_analysis.py 

import numpy as np
import networkx as nx

from problem import build_mis_problem

from features import (
    DegreeFeatureExtractor,
    CentralityFeatureExtractor,
    MWUAVertexFeatureExtractor,
    LPFeatureExtractor,
    LubyFeatureExtractor,
)

# benchmark analysis of features for different graph types
def feature_stats(
    name,
    values,
):

    unique = len(
        np.unique(
            np.round(values, 4)
        )
    )

    print(
        f"{name:<25}"
        f"unique={unique:<4}"
        f"mean={np.mean(values):.4f}   "
        f"std={np.std(values):.4f}"
    )


def analyze_graph(
    G,
    name,
):
    #analysis of features for a given graph 

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Vertices:",
        len(G)
    )

    print(
        "Edges:",
        G.number_of_edges()
    )

    print(
        "Density:",
        round(
            nx.density(G),
            4,
        )
    )

    print()

    problem = build_mis_problem(G)

    degree = (
        DegreeFeatureExtractor()
        .compute(G)
    )

    centrality = (
        CentralityFeatureExtractor()
        .compute(G)
    )

    mwua = (
        MWUAVertexFeatureExtractor()
        .compute(problem)
    )

    lp = (
        LPFeatureExtractor()
        .compute(problem)
    )

    luby = (
        LubyFeatureExtractor(
            runs=100,
        )
        .compute(G)
    )

    print("DEGREE FEATURES")
    print("-" * 70)

    feature_stats(
        "degree_rank",
        degree.degree_rank,
    )

    feature_stats(
        "nbr_min_rank",
        degree.nbr_min_rank,
    )

    feature_stats(
        "nbr_max_rank",
        degree.nbr_max_rank,
    )

    feature_stats(
        "nbr_avg_rank",
        degree.nbr_avg_rank,
    )

    print()

    print("CENTRALITY FEATURES")
    print("-" * 70)

    feature_stats(
        "pagerank",
        centrality.pagerank,
    )

    feature_stats(
        "core_number",
        centrality.core_number,
    )

    feature_stats(
        "clustering",
        centrality.clustering,
    )

    feature_stats(
        "degree_centrality",
        centrality.degree_centrality,
    )

    print()

    print("MWUA FEATURES")
    print("-" * 70)

    feature_stats(
        "mwua_xavg",
        mwua.x_avg,
    )

    feature_stats(
        "mwua_weight_min",
        mwua.weight_min,
    )

    feature_stats(
        "mwua_weight_max",
        mwua.weight_max,
    )

    feature_stats(
        "mwua_weight_avg",
        mwua.weight_avg,
    )
    
    print(
    np.unique(
        np.round(
            mwua.weight_min,
            4
        )
    )
)

    print()

    print("LP FEATURES")
    print("-" * 70)

    feature_stats(
        "lp_value",
        lp.lp_value,
    )

    feature_stats(
        "lp_certainty",
        lp.lp_certainty,
    )
    
    print(
    np.unique(
        lp.lp_value
    )
)

    print()

    print("LUBY FEATURES")
    print("-" * 70)

    feature_stats(
        "luby_frequency",
        luby.frequency,
    )


def benchmark_features():

    n = 100

    er = nx.erdos_renyi_graph(
        n=n,
        p=0.2,
        seed=42,
    )

    ba = nx.barabasi_albert_graph(
        n=n,
        m=3,
        seed=42,
    )

    ws = nx.watts_strogatz_graph(
        n=n,
        k=6,
        p=0.1,
        seed=42,
    )

    analyze_graph(
        er,
        "Erdos-Renyi",
    )

    analyze_graph(
        ba,
        "Barabasi-Albert",
    )

    analyze_graph(
        ws,
        "Watts-Strogatz",
    )


if __name__ == "__main__":

    benchmark_features()