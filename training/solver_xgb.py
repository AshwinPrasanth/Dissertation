import sys
import time
from pathlib import Path
from problem import build_vertex_cover_problem
import networkx as nx
import numpy as np
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
import pandas as pd

from pyscipopt import Model, quicksum, SCIP_PARAMSETTING


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KAMIS_BUILD = (
    PROJECT_ROOT
    / "CHSZLabLib"
    / "build-kamis"
)

sys.path.insert(
    0,
    str(KAMIS_BUILD)
)

import _kamis


from xgb_branching import SCIPXGBBranchRule


from features import (
    CentralityFeatureExtractor,
    LubyFeatureExtractor,
    MWUAVertexFeatureExtractor,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "xgb_ranker.json"
)

class IncumbentEventHandler(Eventhdlr):

    def __init__(self):
        self.history = []


    def eventinit(self):

        self.model.catchEvent(
            SCIP_EVENTTYPE.BESTSOLFOUND,
            self
        )


    def eventexit(self):

        self.model.dropEvent(
            SCIP_EVENTTYPE.BESTSOLFOUND,
            self
        )


    def eventexec(self, event):

        sol = self.model.getBestSol()

        if sol is not None:

            obj = self.model.getSolObjVal(
                sol
            )

            self.history.append(
                (
                    self.model.getSolvingTime(),
                    obj
                )
            )
            
        



def load_mtx_graph(path):

    G = nx.Graph()

    dimensions_read = False


    with open(path, "r") as f:

        for line in f:

            line = line.strip()


            if not line or line.startswith("%"):

                continue


            parts = line.split()


            if not dimensions_read:

                n = int(parts[0])

                G.add_nodes_from(
                    range(n)
                )

                dimensions_read = True

                continue


            if len(parts) < 2:

                continue


            u = int(parts[0]) - 1
            v = int(parts[1]) - 1


            if u != v:

                G.add_edge(
                    u,
                    v
                )


    G.remove_edges_from(
        nx.selfloop_edges(G)
    )


    return G



def graph_to_csr(G):

    G = nx.convert_node_labels_to_integers(
        G,
        ordering="sorted",
    )


    xadj = [0]

    adjncy = []


    for v in range(
        G.number_of_nodes()
    ):

        adjncy.extend(
            sorted(
                G.neighbors(v)
            )
        )

        xadj.append(
            len(adjncy)
        )


    return (

        np.asarray(
            xadj,
            dtype=np.int32
        ),

        np.asarray(
            adjncy,
            dtype=np.int32
        ),

        np.asarray(
            [],
            dtype=np.int32
        )
    )



def reduce_to_core(
    G
):

    xadj, adjncy, vwgt = graph_to_csr(
        G
    )


    core_xadj, core_adjncy, reverse_mapping = (
        _kamis.redumis_kernel(
            xadj,
            adjncy,
            vwgt,
        )
    )


    n = len(core_xadj)-1


    core_graph = nx.Graph()


    core_graph.add_nodes_from(
        range(n)
    )


    for u in range(n):

        start = int(
            core_xadj[u]
        )

        end = int(
            core_xadj[u+1]
        )


        for idx in range(
            start,
            end
        ):

            v = int(
                core_adjncy[idx]
            )

            if u < v:

                core_graph.add_edge(
                    u,
                    v
                )


    return (
        core_graph,
        reverse_mapping
    )



def build_static_features(
    G
):
    problem = build_vertex_cover_problem(G)

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


    features = {}


    for v in G.nodes():

        features[v] = {

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


    return features



def build_mis_model(
    G
):

    model = Model(
        "MIS"
    )


    x = {}


    for v in G.nodes():

        x[v] = model.addVar(
            name=f"x_{v}",
            vtype="B"
        )



    for u,v in G.edges():

        model.addCons(
            x[u] + x[v] <= 1
        )


    model.setObjective(
        quicksum(
            x[v]
            for v in G.nodes()
        ),
        "maximize"
    )


    return model


def load_snap_graph(path):
    print("Loading SNAP graph...")

    start = time.perf_counter()

    if path.suffix.lower() == ".csv":

        df = pd.read_csv(path)

        source = df.columns[0]
        target = df.columns[1]

        G = nx.from_pandas_edgelist(
            df,
            source=source,
            target=target,
            create_using=nx.Graph(),
            data=False
        )

    else:

        G = nx.read_edgelist(
            path,
            comments="#",
            nodetype=int,
            create_using=nx.Graph(),
            data=False
            
        )

    G.remove_edges_from(
        nx.selfloop_edges(G)
    )

    elapsed = time.perf_counter() - start

    print(f"Load time          : {elapsed:.6f} s")

    return G

def solve_xgb_snap(
    graph_path,
    time_limit=12000,
    xgb_max_depth=-1,
):

    start = time.perf_counter()

    print()
    print("=" * 70)
    print("XGB SNAP SOLVER")
    print("=" * 70)

    mis_graph = load_snap_graph(
        graph_path
    )

    print(
        "MIS graph:",
        mis_graph.number_of_nodes(),
        mis_graph.number_of_edges()
    )

    reduction_start = time.perf_counter()

    core_graph, reverse_mapping = reduce_to_core(
        mis_graph
    )

    reduction_time = (
        time.perf_counter()
        - reduction_start
    )

    print(
        "Core:",
        core_graph.number_of_nodes(),
        core_graph.number_of_edges()
    )

    print(
        "Reduction time:",
        reduction_time
    )

    if core_graph.number_of_nodes() == 0:

        print("Fully reduced")

        return None

    static_features = build_static_features(
        core_graph
    )

    model = build_mis_model(
        core_graph
    )

    inc_handler = IncumbentEventHandler()

    model.includeEventhdlr(
        inc_handler,
        "IncumbentLogger",
        "records incumbent solutions"
    )

    model.setIntParam(
        "display/verblevel",
        4
    )

    model.setBoolParam(
        "display/lpinfo",
        False
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
        time_limit
    )

    branchrule = SCIPXGBBranchRule(
        static_features,
        core_graph,
        str(MODEL_PATH),
        max_depth=xgb_max_depth
    )

    model.includeBranchrule(
        branchrule,
        "XGBBranchRule",
        "XGBoost branching",
        priority=1000000,
        maxdepth=xgb_max_depth,
        maxbounddist=1.0,
    )

    solve_start = time.perf_counter()

    model.optimize()
    
    status = model.getStatus()

    objective = (
        model.getObjVal()
        if model.getNSols() > 0
        else None
    )

    dual_bound = model.getDualbound()
    gap = model.getGap()
    solve_time = (
            time.perf_counter()
            - solve_start
        )

    print()
    print("INCUMBENT HISTORY")

    for t, obj in inc_handler.history:
        print(
            f"time={t:.3f}, objective={obj}"
        )
        print()
        print("=" * 70)
        print("SCIP SUMMARY")
        print("=" * 70)

        print(f"Status          : {status}")
        print(f"Objective       : {objective}")
        print(f"Dual bound      : {dual_bound}")
        print(f"Gap             : {gap:.6f}")
        print(f"Nodes explored  : {model.getNNodes()}")
        print(f"Branches (XGB)  : {branchrule.branch_count}")
        print(f"LP iterations   : {model.getNLPIterations()}")
        print(f"Solutions found : {model.getNSols()}")
        print(f"Solve time      : {solve_time:.3f} s")
        print("=" * 70)

    result = {

    "graph":
        Path(graph_path).stem,

    "original_n":
        mis_graph.number_of_nodes(),

    "core_n":
        core_graph.number_of_nodes(),

    "status":
        str(status),

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

    "lp_iterations":
        model.getNLPIterations(),

    "solutions":
        model.getNSols(),

    "solve_time":
        solve_time,

    "total_time":
        time.perf_counter() - start,

    "incumbent_history":
        inc_handler.history,
}

    print(result)

    return result


def solve_xgb_dimacs(
    graph_path,
    time_limit=1200,
    xgb_max_depth=-1
):


    start = time.perf_counter()


    print()
    print("="*70)
    print("XGB DIMACS SOLVER")
    print("="*70)


    clique_graph = load_mtx_graph(
        graph_path
    )


    print(
        "Clique graph:",
        clique_graph.number_of_nodes(),
        clique_graph.number_of_edges()
    )



    mis_graph = nx.complement(
        clique_graph
    )


    print(
        "MIS graph:",
        mis_graph.number_of_nodes(),
        mis_graph.number_of_edges()
    )



    reduction_start = time.perf_counter()


    core_graph, reverse_mapping = reduce_to_core(
        mis_graph
    )


    reduction_time = (
        time.perf_counter()
        -
        reduction_start
    )


    print(
        "Core:",
        core_graph.number_of_nodes(),
        core_graph.number_of_edges()
    )


    print(
        "Reduction time:",
        reduction_time
    )


    if core_graph.number_of_nodes()==0:

        print(
            "Fully reduced"
        )

        return None



    static_features = build_static_features(
        core_graph
    )


    model = build_mis_model(
        core_graph
    )
    
    inc_handler = IncumbentEventHandler()

    model.includeEventhdlr(
        inc_handler,
        "IncumbentLogger",
        "records incumbent solutions"
    )
    
    model.setIntParam(
    "display/verblevel",
    4
    )

    model.setBoolParam(
        "display/lpinfo",
        False
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
        time_limit
    )


    branchrule = SCIPXGBBranchRule(
        static_features,
        core_graph,
        str(MODEL_PATH),
        max_depth=xgb_max_depth
    )


    model.includeBranchrule(
        branchrule,
        "XGBBranchRule",
        "XGBoost branching",
        priority=1000000,
        maxdepth=xgb_max_depth,
        maxbounddist=1.0,
    )


    solve_start = time.perf_counter()
    incumbent_history = []


    model.optimize()
    
    print()
    print("INCUMBENT HISTORY")

    for t,obj in inc_handler.history:
        print(
            f"time={t:.3f}, objective={obj}"
        )


    solve_time = (
        time.perf_counter()
        -
        solve_start
    )


    result = {

        "graph":
            Path(graph_path).parent.name,

        "original_n":
            clique_graph.number_of_nodes(),

        "core_n":
            core_graph.number_of_nodes(),

        "objective":
            model.getObjVal()
            if model.getNSols()
            else None,

        "nodes":
            model.getNNodes(),

        "branches":
            branchrule.branch_count,

        "solve_time":
            solve_time,

        "total_time":
            time.perf_counter()-start,
            
        "incumbent_history":
            inc_handler.history
    }


    print(result)


    return result