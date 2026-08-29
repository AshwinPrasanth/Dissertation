import sys
import time
from pathlib import Path

import numpy as np

from pyscipopt import (
    Eventhdlr,
    Model,
    SCIP_EVENTTYPE,
    SCIP_PARAMSETTING,
    quicksum,
)

from xgb_branching import (
    SCIPXGBBranchRule,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "xgb_reduced_19_100"
    / "xgb_ranker_full_100.json"
)


class IncumbentEventHandler(
    Eventhdlr
):

    def __init__(self):

        self.history = []


    def eventinit(self):

        self.model.catchEvent(
            SCIP_EVENTTYPE.BESTSOLFOUND,
            self,
        )


    def eventexit(self):

        self.model.dropEvent(
            SCIP_EVENTTYPE.BESTSOLFOUND,
            self,
        )


    def eventexec(
        self,
        event,
    ):

        sol = self.model.getBestSol()

        if sol is not None:

            obj = self.model.getSolObjVal(
                sol
            )

            self.history.append(
                (
                    self.model.getSolvingTime(),
                    obj,
                )
            )


def load_hitting_set(
    path,
):

    hyperedges = []

    num_variables = None

    with open(
        path,
        "r",
    ) as f:

        for line in f:

            line = line.strip()

            if (
                not line
                or line.startswith("c")
            ):

                continue

            parts = line.split()

            if parts[0] == "p":

                num_variables = int(
                    parts[2]
                )

                continue

            edge = [
                int(x) - 1
                for x in parts
            ]

            edge = sorted(
                set(edge)
            )

            if edge:

                hyperedges.append(
                    edge
                )


    if num_variables is None:

        raise ValueError(
            "Could not determine number of variables"
        )


    return (
        num_variables,
        hyperedges,
    )


def build_hitting_set_problem(
    num_variables,
    hyperedges,
):

    model = Model(
        "HittingSet"
    )

    variables = {}

    for v in range(
        num_variables
    ):

        variables[v] = (
            model.addVar(
                name=f"x_{v}",
                vtype="B",
            )
        )


    for edge in hyperedges:

        model.addCons(
            quicksum(
                variables[v]
                for v in edge
            ) >= 1
        )


    model.setObjective(
        quicksum(
            variables[v]
            for v in range(
                num_variables
            )
        ),
        "minimize",
    )


    return (
        model,
        variables,
    )


def build_static_features(
    num_variables,
    hyperedges,
):

    from types import SimpleNamespace

    from features import (
        HypergraphFeatureExtractor,
        MWUAElementFeatureExtractor,
    )


    problem = SimpleNamespace()

    problem.num_variables = (
        num_variables
    )

    problem.hyperedges = (
        hyperedges
    )


    hypergraph = (
        HypergraphFeatureExtractor()
        .compute(
            problem
        )
    )


    mwua = (
        MWUAElementFeatureExtractor()
        .compute(
            problem
        )
    )


    static_features = {}


    for v in range(
        num_variables
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
                    hypergraph
                    .bipartite_core_number[v]
                ),

            "bipartite_pagerank":
                float(
                    hypergraph
                    .bipartite_pagerank[v]
                ),

            "frequency_rank":
                float(
                    hypergraph
                    .frequency_rank[v]
                ),

            "min_set_size":
                float(
                    hypergraph
                    .min_set_size[v]
                ),

            "max_set_size":
                float(
                    hypergraph
                    .max_set_size[v]
                ),

            "pair_count":
                float(
                    hypergraph
                    .pair_count[v]
                ),
        }


    return static_features


def solve_xgb_hitting_set(
    problem_path,
    time_limit=1800,
    xgb_max_depth=5,
):

    start = time.perf_counter()

    print()
    print("=" * 70)
    print("XGB HITTING SET SOLVER")
    print("=" * 70)


    num_variables, hyperedges = (
        load_hitting_set(
            problem_path
        )
    )


    print(
        "Variables:",
        num_variables,
    )

    print(
        "Hyperedges:",
        len(hyperedges),
    )


    print()
    print(
        "Computing root static features..."
    )


    feature_start = (
        time.perf_counter()
    )


    static_features = (
        build_static_features(
            num_variables,
            hyperedges,
        )
    )


    feature_time = (
        time.perf_counter()
        -
        feature_start
    )


    print(
        "Static feature time:",
        feature_time,
    )


    model, variables = (
        build_hitting_set_problem(
            num_variables,
            hyperedges,
        )
    )


    inc_handler = (
        IncumbentEventHandler()
    )


    model.includeEventhdlr(
        inc_handler,
        "IncumbentLogger",
        "records incumbent solutions",
    )


    model.setIntParam(
        "display/verblevel",
        4,
    )


    model.setPresolve(
        SCIP_PARAMSETTING.DEFAULT
    )


    model.setHeuristics(
        SCIP_PARAMSETTING.DEFAULT
    )


    model.setSeparating(
        SCIP_PARAMSETTING.DEFAULT
    )


    model.setRealParam(
        "limits/time",
        float(time_limit),
    )


    branchrule = (
        SCIPXGBBranchRule(
            static_features=static_features,
            hyperedges=hyperedges,
            model_path=str(
                MODEL_PATH
            ),
            max_depth=xgb_max_depth,
        )
    )


    model.includeBranchrule(
        branchrule,
        "XGBBranchRule",
        "XGBoost ranking branching",
        priority=1000000,
        maxdepth=xgb_max_depth,
        maxbounddist=1.0,
    )


    print()
    print(
        "Starting SCIP..."
    )


    solve_start = (
        time.perf_counter()
    )


    model.optimize()


    solve_time = (
        time.perf_counter()
        -
        solve_start
    )


    status = str(
        model.getStatus()
    )


    objective = None

    if model.getNSols() > 0:

        objective = (
            model.getObjVal()
        )


    dual_bound = (
        model.getDualbound()
    )


    gap = None

    if objective is not None:

        if objective != 0:

            gap = (
                abs(
                    objective
                    - dual_bound
                )
                /
                abs(objective)
                *
                100
            )

        else:

            gap = 0.0


    total_time = (
        time.perf_counter()
        - start
    )


    print()
    print(
        "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )


    print(
        "Status:",
        status,
    )


    print(
        "Objective:",
        objective,
    )


    print(
        "Dual bound:",
        dual_bound,
    )


    print(
        "Gap:",
        gap,
    )


    print(
        "Nodes:",
        model.getNNodes(),
    )


    print(
        "Branches:",
        branchrule.branch_count,
    )


    print(
        "Solve time:",
        solve_time,
    )


    print(
        "Total time:",
        total_time,
    )


    print()
    print(
        "INCUMBENT HISTORY"
    )


    for t, obj in (
        inc_handler.history
    ):

        print(
            f"time={t:.3f}, "
            f"objective={obj}"
        )


    return {

        "status":
            status,

        "objective":
            objective,

        "dual_bound":
            dual_bound,

        "gap":
            gap,

        "nodes":
            model.getNNodes(),

        "branches":
            branchrule.branch_count,

        "solve_time":
            solve_time,

        "total_time":
            total_time,

        "incumbent_history":
            inc_handler.history,
    }


if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "instance"
    )


    parser.add_argument(
        "--time-limit",
        type=float,
        default=1200,
    )


    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=-1,
    )


    args = parser.parse_args()


    solve_xgb_hitting_set(
        args.instance,
        time_limit=args.time_limit,
        xgb_max_depth=args.xgb_max_depth,
    )