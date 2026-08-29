import time
from pathlib import Path

from pyscipopt import (
    Model,
    SCIP_PARAMSETTING,
    quicksum,
)

from branching import SCIPSBDataCollector
from dataset_writer import DatasetWriter
from features import (
    HypergraphFeatureExtractor,
    MWUAElementFeatureExtractor,
)
from problem import build_hitting_set_problem



PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ltb_training_dynamic"
)


def build_root_static_features(
    instance,
):

    problem = build_hitting_set_problem(
        instance
    )

    hypergraph = (
        HypergraphFeatureExtractor()
        .compute(problem)
    )
    mwua_start = time.perf_counter()
    mwua = (
        MWUAElementFeatureExtractor()
        .compute(problem)
    )
    
    mwua_time = (
    time.perf_counter()
    - mwua_start
)

    print(f"MWUA time: {mwua_time:.4f}s")

    static_features = {}

    for v in range(
        problem.num_variables
    ):

        static_features[v] = {

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

        "bipartite_core_number":
            float(
                hypergraph.bipartite_core_number[v]
            ),

        "bipartite_pagerank":
            float(
                hypergraph.bipartite_pagerank[v]
            ),

        "frequency_rank":
            float(
                hypergraph.frequency_rank[v]
            ),

        "min_set_size":
            float(
                hypergraph.min_set_size[v]
            ),

        "max_set_size":
            float(
                hypergraph.max_set_size[v]
            ),

        "pair_count":
            float(
                hypergraph.pair_count[v]
            ),
    }

    return static_features


def solve_training_instance(

    instance,

    graph_name,

    family,

    max_sb_nodes=500,

    candidate_limit=None,

    strongbranch_itlim=100,

    time_limit=None,

    output_dir=None,

):

    total_start = time.perf_counter()

    problem = build_hitting_set_problem(
        instance
    )

    static_start = time.perf_counter()

    static_features = (
        build_root_static_features(
            instance
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

    for v in range(
        problem.num_variables
    ):

        x[v] = model.addVar(
            name=f"x_{v}",
            vtype="B",
        )


    for hyperedge in problem.hyperedges:

        model.addCons(

            quicksum(

                x[v]
                for v in hyperedge

            ) >= 1

        )


    model.setObjective(

        quicksum(

            x[v]
            for v in range(
                problem.num_variables
            )

        ),

        "minimize",

)

    collector = SCIPSBDataCollector(
    static_features=static_features,
    hyperedges=problem.hyperedges,
    graph_name=graph_name,
    writer=writer,
    max_sb_nodes=max_sb_nodes,
    candidate_limit=candidate_limit,
    strongbranch_itlim=strongbranch_itlim,
)

    model.includeBranchrule(
        collector,
        "SBDataCollector",
        "Hitting Set Strong Branching Data Collector",
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

        "variables":
            problem.num_variables,

        "hyperedges":
            len(problem.hyperedges),

        "status":
            (
                "collection_complete"
                if collection_complete
                else str(model.getStatus())
            ),
            
        "collection_complete":
            collection_complete,

        "objective":
            (
                float(model.getObjVal())
                if model.getNSols() > 0
                else None
            ),
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
