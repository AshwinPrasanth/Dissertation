import networkx as nx
import numpy as np
from pyscipopt import Model

from dataset import DatasetBuilder
from problem import build_vertex_cover_problem
from features import MWUAVertexFeatureExtractor
from branching import SCIPMWUABranchRule
from pyscipopt import SCIP_PARAMSETTING, quicksum


'''# ---------------------------------
# Graph
# ---------------------------------

G = nx.erdos_renyi_graph(
    100,
    0.25,
    seed=0,
)

# ---------------------------------
# MWUA features
# ---------------------------------

problem = build_mis_problem(G)

builder = DatasetBuilder()

dataset = builder.build(problem)

scores = (
    MWUAVertexFeatureExtractor
    .compute_mwua_score(dataset)
)
np.savetxt(
    "mwua_scores.csv",
    scores,
    delimiter=","
)
print("\n===== MWUA SCORE STATS =====")

print("Min:", scores.min())
print("Max:", scores.max())
print("Mean:", scores.mean())

print("\nTop 10 vertices:")

top = sorted(
    range(len(scores)),
    key=lambda v: scores[v],
    reverse=True,
)[:10]

for v in top:
    print(
        f"Vertex {v}: "
        f"{scores[v]}"
    )
print(
    sorted(
        range(len(scores)),
        key=lambda v: scores[v],
        reverse=True,
    )[:20]
)

# ---------------------------------
# SCIP model
# ---------------------------------

model = Model()
model.setPresolve(SCIP_PARAMSETTING.OFF)
model.setHeuristics(SCIP_PARAMSETTING.OFF)
model.setSeparating(SCIP_PARAMSETTING.OFF)
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

model.setObjective(quicksum(x[v] for v in G.nodes()),"maximize")

# ---------------------------------
# MWUA branchrule
# ---------------------------------

branchrule = SCIPMWUABranchRule(
    scores
)

model.includeBranchrule(

    branchrule,

    "MWUA",

    "MWUA branching",

    priority=1000000,

    maxdepth=-1,

    maxbounddist=1.0,
)

# ---------------------------------
# Solve
# ---------------------------------
print("Vars:", model.getNVars())
print("Conss:", model.getNConss())
model.optimize()'''


import time

from pyscipopt import (
    Model,
    SCIP_PARAMSETTING,
    quicksum,
)

from dimacs import load_dimacs_clq


# ---------------------------------
# Graph
# ---------------------------------

G = load_dimacs_clq(
    "graphs/bhoslib/frb40-19-2.clq"
)

#G = nx.complement(G)

G = nx.convert_node_labels_to_integers(
    G,
    first_label=0
)



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
np.savetxt(
    "mwua_scores.csv",
    scores,
    delimiter=","
)
print("\n===== MWUA SCORE STATS =====")

print("Min:", scores.min())
print("Max:", scores.max())
print("Mean:", scores.mean())

print("\nTop 10 vertices:")

top = sorted(
    range(len(scores)),
    key=lambda v: scores[v],
    reverse=True,
)[:10]

for v in top:
    print(
        f"Vertex {v}: "
        f"{scores[v]}"
    )
print(
    sorted(
        range(len(scores)),
        key=lambda v: scores[v],
        reverse=True,
    )[:20]
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
# MWUA branchrule
# ---------------------------------

branchrule = SCIPMWUABranchRule(
    scores
)

model.includeBranchrule(

    branchrule,

    "MWUA",

    "MWUA branching",

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

'''
import time
import networkx as nx

from pyscipopt import (
    Model,
    SCIP_PARAMSETTING,
    quicksum,
)


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

            G.add_edge(u, v)

    return G


# ---------------------------------
# Graph
# ---------------------------------

G = load_snap_graph(
    "graphs/email-Enron.txt"
)

G.remove_edges_from(
    nx.selfloop_edges(G)
)

G = nx.convert_node_labels_to_integers(
    G,
    first_label=0
)
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
t0 = time.time()
problem = build_mis_problem(G)
print("build_mis_problem:", time.time() - t0)

t0 = time.time()
builder = DatasetBuilder()
dataset = builder.build(problem)
print("dataset build:", time.time() - t0)

t0 = time.time()
scores = (
    MWUAVertexFeatureExtractor
    .compute_mwua_score(dataset)
)
print("MWUA score computation:", time.time() - t0)

np.savetxt(
    "mwua_scores.csv",
    scores,
    delimiter=","
)
print("\n===== MWUA SCORE STATS =====")

print("Min:", scores.min())
print("Max:", scores.max())
print("Mean:", scores.mean())

print("\nTop 10 vertices:")

top = sorted(
    range(len(scores)),
    key=lambda v: scores[v],
    reverse=True,
)[:10]

for v in top:
    print(
        f"Vertex {v}: "
        f"{scores[v]}"
    )
print(
    sorted(
        range(len(scores)),
        key=lambda v: scores[v],
        reverse=True,
    )[:20]
)

# ---------------------------------
# SCIP model
# ---------------------------------

model = Model()
model.setParam(
    "limits/time",
    300
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
# MWUA branchrule
# ---------------------------------

branchrule = SCIPMWUABranchRule(
    scores
)

model.includeBranchrule(

    branchrule,

    "MWUA",

    "MWUA branching",

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

model.optimize()

end = time.time()

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
)'''