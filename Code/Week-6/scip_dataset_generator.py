import networkx as nx
import numpy as np
from pyscipopt import Model

from dataset import DatasetBuilder
from problem import build_vertex_cover_problem
from features import MWUAVertexFeatureExtractor
from dataset_branchrule import SCIPDatasetCollector
from pyscipopt import SCIP_PARAMSETTING, quicksum


import time

from pyscipopt import (
    Model,
    SCIP_PARAMSETTING,
    quicksum,
)

from dimacs import load_dimacs_clq

graph_name = "frb30-15-1"
# ---------------------------------
# Graph
# ---------------------------------

G = load_dimacs_clq(f"graphs/bhoslib/{graph_name}.clq")

#G = nx.complement(G)

G = nx.convert_node_labels_to_integers(G,first_label=0)



print(
    "Nodes:",
    G.number_of_nodes()
)

print(
    "Edges:",
    G.number_of_edges()
)
# ---------------------------------
# MWUA features
# ---------------------------------

problem = build_vertex_cover_problem(G)
#print(problem)
builder = DatasetBuilder()

dataset = builder.build(problem)

scores = (
    MWUAVertexFeatureExtractor
    .compute_mwua_score(dataset)
)


# ---------------------------------
# SCIP model
# ---------------------------------

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

model.setParam(
    "limits/time",
    7200
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
# ---------------------------------
# Dataset Collection
# ---------------------------------

collector = SCIPDatasetCollector(
    dataset,
    scores,
    graph_name
)

model.includeBranchrule(

    collector,

    "COLLECT",

    "Dataset Collector",

    priority=1000000,

    maxdepth=-1,

    maxbounddist=1.0,
)

# ---------------------------------
# Solve
# ---------------------------------

print(
    "Vars:",
    model.getNVars()
)

print(
    "Conss:",
    model.getNConss()
)

start = time.time()

print("\nCollecting branch-node dataset...")
model.optimize()

end = time.time()

# ---------------------------------
# Results
# ---------------------------------

print("\n===== RESULTS =====")

print(
    "Status:",
    model.getStatus()
)

if model.getNSols() > 0:

    print(
        "Objective:",
        model.getObjVal()
    )

print(
    "Nodes:",
    model.getNNodes()
)

print(
    "Runtime:",
    end - start
)

