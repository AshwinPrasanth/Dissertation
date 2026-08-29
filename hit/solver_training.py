import time
from pathlib import Path

import networkx as nx

from pyscipopt import (
    Model,
    SCIP_PARAMSETTING,
    quicksum,
)

from branching import SCIPSBDataCollector
from dataset_writer import DatasetWriter
from features import (
    CentralityFeatureExtractor,
    LubyFeatureExtractor,
    MWUAVertexFeatureExtractor,
)
from problem import build_vertex_cover_problem


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ltb_training"
)


def build_root_static_features(
    G,
):

    problem = build_vertex_cover_problem(
        G
    )

    centrality = (
        CentralityFeatureExtractor()
        .compute(G)
    )

    luby = (
        LubyFeatureExtractor()
        .compute(G)
    )

    mwua = (
        MWUAVertexFeatureExtractor()
        .compute(problem)
    )

    static_features = {}

    for v in G.nodes():

        static_features[v] = {
            "pagerank":
                float(
                    centrality.pagerank[v]
                ),

            "mwua_xavg":
                float(
                    mwua.x_avg[v]
                ),

            "mwua_weight_min":
                float(
                    mwua.weight_min[v]
                ),

            "mwua_weight_max":
                float(
                    mwua.weight_max[v]
                ),

            "mwua_weight_avg":
                float(
                    mwua.weight_avg[v]
                ),

            "luby_frequency":
                float(
                    luby.frequency[v]
                ),
        }

    return static_features


def solve_training_instance(
    G,
    graph_name,
    family,
    max_sb_nodes=10,
    candidate_limit=None,
    strongbranch_itlim=100,
    time_limit=None,
    output_dir=None,
):

    total_start = time.perf_counter()

    G = nx.convert_node_labels_to_integers(
        G,
        first_label=0,
        ordering="sorted",
    )

    static_start = time.perf_counter()

    static_features = (
        build_root_static_features(
            G
        )
    )

    static_feature_time = (
        time.perf_counter()
        - static_start
    )

    writer = DatasetWriter(
        graph_name=graph_name,
        output_dir=OUTPUT_DIR if output_dir is None else Path(output_dir),
    )

    model = Model(
        problemName=str(
            graph_name
        )
    )

    model.setPresolve(
        SCIP_PARAMSETTING.OFF
    )

    model.setHeuristics(
        SCIP_PARAMSETTING.OFF
    )

    model.setSeparating(
        SCIP_PARAMSETTING.OFF
    )

    if time_limit is not None:

        model.setRealParam(
            "limits/time",
            float(time_limit),
        )

    x = {}

    for v in G.nodes():

        x[v] = model.addVar(
            name=f"x_{v}",
            vtype="B",
        )

    for u, v in G.edges():

        model.addCons(
            x[u] + x[v] <= 1
        )

    model.setObjective(
        quicksum(
            x[v]
            for v in G.nodes()
        ),
        "maximize",
    )

    collector = SCIPSBDataCollector(
        static_features=static_features,
        graph=G,
        graph_name=graph_name,
        writer=writer,
        max_sb_nodes=max_sb_nodes,
        candidate_limit=candidate_limit,
        strongbranch_itlim=strongbranch_itlim,
    )

    model.includeBranchrule(
        collector,
        "SBDataCollector",
        "6 static global plus 9 dynamic local strong branching collector",
        priority=1000000,
        maxdepth=-1,
        maxbounddist=1.0,
    )

    solve_start = time.perf_counter()

    model.optimize()
    
    collection_complete = (
    collector.sb_nodes
    >= max_sb_nodes)

    solve_time = (
        time.perf_counter()
        - solve_start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    result = {
        "graph_name":
            graph_name,

        "family":
            family,

        "n":
            G.number_of_nodes(),

        "m":
            G.number_of_edges(),

        "status":
            (
                "collection_complete"
                if collection_complete
                else str(model.getStatus())
            ),
            
        "collection_complete":
            collection_complete,

        "objective":
            model.getObjVal()
            if model.getNSols()
            else None,

        "solve_nodes":
            model.getNNodes(),

        "sb_nodes":
            collector.sb_nodes,

        "sb_candidates":
            collector.sb_candidates,

        "branch_count":
            collector.branch_count,

        "static_feature_time":
            static_feature_time,

        "solve_time":
            solve_time,

        "total_time":
            total_time,

        "dataset_file":
            str(writer.filename),
    }

    print()

    print(
        "=" * 70
    )

    print(
        "LTB TRAINING INSTANCE"
    )

    print(
        "=" * 70
    )

    for key, value in result.items():

        print(
            f"{key:<20}: {value}"
        )

    return result
