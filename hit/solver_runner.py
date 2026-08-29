import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

from pyscipopt import (
    Model,
    SCIP_PARAMSETTING,
    quicksum,
)

from dataset import DatasetBuilder
from features import MWUAVertexFeatureExtractor
from problem import build_vertex_cover_problem
from branching import SCIPMWUABranchRule


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KAMIS_BUILD = (
    PROJECT_ROOT
    / "CHSZLabLib"
    / "build-kamis"
)

sys.path.insert(
    0,
    str(KAMIS_BUILD),
)

import _kamis


def load_snap_graph(path):

    G = nx.Graph()

    with open(path, "r") as f:

        for line in f:

            if line.startswith("#"):
                continue

            parts = line.strip().split()

            if len(parts) != 2:
                continue

            u = int(parts[0])
            v = int(parts[1])

            if u != v:
                G.add_edge(u, v)

    G = nx.convert_node_labels_to_integers(
        G,
        first_label=0,
        ordering="sorted",
    )

    return G


def graph_to_csr(G):

    n = G.number_of_nodes()

    xadj = np.zeros(
        n + 1,
        dtype=np.int32,
    )

    adjncy = []

    for node in range(n):

        neighbors = sorted(
            G.neighbors(node)
        )

        adjncy.extend(neighbors)

        xadj[node + 1] = len(adjncy)

    adjncy = np.asarray(
        adjncy,
        dtype=np.int32,
    )

    vwgt = np.asarray(
        [],
        dtype=np.int32,
    )

    return (
        xadj,
        adjncy,
        vwgt,
    )


def csr_to_graph(
    xadj,
    adjncy,
):

    n = len(xadj) - 1

    G = nx.Graph()

    G.add_nodes_from(
        range(n)
    )

    for u in range(n):

        start = int(xadj[u])
        end = int(xadj[u + 1])

        for index in range(
            start,
            end,
        ):

            v = int(
                adjncy[index]
            )

            if u < v:

                G.add_edge(
                    u,
                    v,
                )

    return G


def reduce_graph(G):

    print()

    print(
        "==================================="
    )

    print(
        "KAMIS KERNELIZATION"
    )

    print(
        "==================================="
    )

    original_vertices = (
        G.number_of_nodes()
    )

    original_edges = (
        G.number_of_edges()
    )

    print(
        "Original vertices:",
        original_vertices,
    )

    print(
        "Original edges:",
        original_edges,
    )

    print()

    print(
        "Converting raw graph to CSR..."
    )

    csr_start = time.perf_counter()

    xadj, adjncy, vwgt = (
        graph_to_csr(G)
    )

    csr_time = (
        time.perf_counter()
        - csr_start
    )

    print(
        f"CSR conversion time: "
        f"{csr_time:.6f} s"
    )

    print()

    print(
        "Running KaMIS reductions..."
    )

    reduction_start = (
        time.perf_counter()
    )

    (
        core_xadj,
        core_adjncy,
        reverse_mapping,
    ) = _kamis.redumis_kernel(
        xadj,
        adjncy,
        vwgt,
    )

    reduction_time = (
        time.perf_counter()
        - reduction_start
    )

    core_vertices = (
        len(core_xadj) - 1
    )

    core_edges = (
        len(core_adjncy) // 2
    )

    core_ratio = (
        core_vertices
        / original_vertices
        if original_vertices
        else 0.0
    )

    print()

    print(
        "Core vertices:",
        core_vertices,
    )

    print(
        "Core edges:",
        core_edges,
    )

    print(
        f"Core ratio: "
        f"{core_ratio:.6f}"
    )

    print(
        f"Vertices removed: "
        f"{1.0 - core_ratio:.2%}"
    )

    print(
        f"Reduction time: "
        f"{reduction_time:.6f} s"
    )

    print()

    print(
        "Constructing core graph..."
    )

    core_build_start = (
        time.perf_counter()
    )

    G_core = csr_to_graph(
        core_xadj,
        core_adjncy,
    )

    core_build_time = (
        time.perf_counter()
        - core_build_start
    )

    print(
        f"Core graph build time: "
        f"{core_build_time:.6f} s"
    )

    return (
        G_core,
        reverse_mapping,
        {
            "original_vertices":
                original_vertices,

            "original_edges":
                original_edges,

            "core_vertices":
                core_vertices,

            "core_edges":
                core_edges,

            "core_ratio":
                core_ratio,

            "csr_time":
                csr_time,

            "reduction_time":
                reduction_time,

            "core_build_time":
                core_build_time,
        },
    )


def solve_instance(
    graph_path,
    use_mwua=True,
    selector="mwua",
    use_mwua_direction=True,
    max_spine_length=1,
    certainty_threshold=0.0,
):

    total_start = time.perf_counter()

    print()

    print(
        "Loading SNAP graph..."
    )

    load_start = time.perf_counter()

    G = load_snap_graph(
        graph_path
    )

    load_time = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Graph load time: "
        f"{load_time:.6f} s"
    )

    (
        G,
        reverse_mapping,
        kernel_stats,
    ) = reduce_graph(G)

    if G.number_of_nodes() == 0:

        total_time = (
            time.perf_counter()
            - total_start
        )

        return {
            "strategy":
                "MWUA"
                if use_mwua
                else "SCIP",

            '''"depth_limit":
                depth_limit
                if use_mwua
                else "default",'''
            
            "certainty_threshold":
                certainty_threshold
                if use_mwua
                else None,

            "runtime":
                0.0,

            "nodes":
                0,

            "objective":
                None,

            "status":
                "kernel_empty",

            "solutions":
                0,

            "search_depth":
                None,

            "branch_calls":
                None,

            "mwua_branches":
                None,

            "predicted_one":
                None,

            "predicted_zero":
                None,

            "load_time":
                load_time,

            "kernel_time":
                kernel_stats[
                    "reduction_time"
                ],

            "mwua_time":
                0.0,

            "solve_time":
                0.0,

            "total_time":
                total_time,

            **kernel_stats,
        }

    print()

    print(
        "==================================="
    )

    print(
        "MWUA FEATURE EXTRACTION"
    )

    print(
        "==================================="
    )

    mwua_start = time.perf_counter()

    problem = (
        build_vertex_cover_problem(G)
    )

    builder = DatasetBuilder()

    dataset = builder.build(
        problem
    )

    xavg_idx = (
        dataset.feature_names.index(
            "mwua_xavg"
        )
    )

    mis_xavg = np.asarray(
        dataset.X[:, xavg_idx],
        dtype=float,
    )

    scores = np.abs(
        mis_xavg - 0.5
    )

    predictions = (
        mis_xavg > 0.5
    ).astype(
        np.int8
    )

    mwua_time = (
        time.perf_counter()
        - mwua_start
    )

    print(
        f"MWUA time: "
        f"{mwua_time:.6f} s"
    )

    print(
        "MWUA score count:",
        len(scores),
    )

    print(
        "MWUA score min:",
        scores.min(),
    )

    print(
        "MWUA score max:",
        scores.max(),
    )

    print(
        "MWUA score mean:",
        scores.mean(),
    )

    print(
        "Predicted x=0:",
        int(
            np.sum(
                predictions == 0
            )
        ),
    )

    print(
        "Predicted x=1:",
        int(
            np.sum(
                predictions == 1
            )
        ),
    )

    print(
        "MIS x_avg min:",
        mis_xavg.min(),
    )

    print(
        "MIS x_avg max:",
        mis_xavg.max(),
    )

    print(
        "MIS x_avg mean:",
        mis_xavg.mean(),
    )

    print()

    print(
        "==================================="
    )

    print(
        "SCIP CORE SOLVE"
    )

    print(
        "==================================="
    )

    model = Model()

    model.setPresolve(
        SCIP_PARAMSETTING.OFF
    )

    model.setHeuristics(
        SCIP_PARAMSETTING.OFF
    )

    model.setSeparating(
        SCIP_PARAMSETTING.OFF
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

    if use_mwua:

        '''branchrule = (
            SCIPMWUABranchRule(
                mwua_certainty=scores,
                mwua_prediction=predictions,
                depth_limit=depth_limit,
                certainty_threshold=0.0,
            )
        )'''
        
        branchrule = (
    SCIPMWUABranchRule(
        mwua_certainty=scores,
        mwua_prediction=predictions,
        graph=G,
        selector=selector,
        use_mwua_direction=use_mwua_direction,
        max_spine_length=max_spine_length,
        certainty_threshold=certainty_threshold,
    )
)

        model.includeBranchrule(
            branchrule,
            "MWUA",
            "MWUA Anytime Branching",
            priority=1000000,
            maxdepth=-1,
            maxbounddist=1.0,
        )

    else:

        branchrule = None

    solve_start = time.perf_counter()

    model.optimize()

    solve_time = (
        time.perf_counter()
        - solve_start
    )

    if branchrule is not None:

        branchrule.stats.finish()

    total_time = (
        time.perf_counter()
        - total_start
    )

    if branchrule is not None:
        print()

        print(
            "========== MWUA SPINE =========="
        )

        print(
            "Branch rule calls:",
            branchrule.call_count,
        )

        print(
            "MWUA branches:",
            branchrule.branch_count,
        )

        print(
            "Predicted x=1 branches:",
            branchrule.predicted_one_count,
        )

        print(
            "Predicted x=0 branches:",
            branchrule.predicted_zero_count,
        )

        print(
            "Maximum callback depth:",
            branchrule.stats.max_depth,
        )

        print(
            "Spine length:",
            branchrule.spine_length,
        )

        print(
            "Length stops:",
            branchrule.length_stop_count,
        )

        print(
            "Uncertainty stops:",
            branchrule.uncertainty_stop_count,
        )

        print(
            "Off-spine callbacks:",
            branchrule.off_spine_count,
        )

        print(
            "Spine vertices:",
            branchrule.spine_vertices,
        )

        print(
            "Spine certainties:",
            branchrule.spine_certainties,
        )

        print(
            "Spine predictions:",
            branchrule.spine_predictions,
        )
        
        print(
            "Selector:",
            branchrule.selector,
        )

        print(
            "MWUA direction:",
            branchrule.use_mwua_direction,
        )

        print(
            "Spine residual degrees:",
            branchrule.spine_residual_degrees,
        )

    return {
        "strategy":
            "MWUA"
            if use_mwua
            else "SCIP",

        '''"depth_limit":
            depth_limit
            if use_mwua
            else "default",'''
            
        "certainty_threshold":
            certainty_threshold
            if use_mwua
            else None,

        "runtime":
            solve_time,

        "nodes":
            model.getNNodes(),

        "objective":
            model.getObjVal()
            if model.getNSols()
            else None,

        "status":
            str(model.getStatus()),

        "solutions":
            model.getNSols(),

        "search_depth":
            branchrule.stats.max_depth
            if branchrule
            else None,

        "branch_calls":
            branchrule.call_count
            if branchrule
            else None,

        "mwua_branches":
            branchrule.branch_count
            if branchrule
            else None,

        "predicted_one":
            branchrule.predicted_one_count
            if branchrule
            else None,

        "predicted_zero":
            branchrule.predicted_zero_count
            if branchrule
            else None,
            
        "max_spine_length":
            max_spine_length
            if use_mwua
            else None,

        "spine_length":
            branchrule.spine_length
            if branchrule
            else None,

        "length_stops":
            branchrule.length_stop_count
            if branchrule
            else None,

        "uncertainty_stops":
            branchrule.uncertainty_stop_count
            if branchrule
            else None,
            
        "selector":
            selector
            if use_mwua
            else None,

        "use_mwua_direction":
            use_mwua_direction
            if use_mwua
            else None,

        "spine_residual_degrees":
            branchrule.spine_residual_degrees
            if branchrule
            else None,
            
        "load_time":
            load_time,

        "kernel_time":
            kernel_stats[
                "reduction_time"
            ],

        "mwua_time":
            mwua_time,

        "solve_time":
            solve_time,

        "total_time":
            total_time,

        **kernel_stats,
    }