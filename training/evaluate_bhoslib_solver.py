from pathlib import Path
import time
import torch
import networkx as nx
import csv
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

from model import BranchingMLP
from ml_branching import MLBranchRule
from feature_builder import MLFeatureBuilder

BHOSLIB_DIR = Path("graphs/bhoslib")
CHECKPOINT = "checkpoints/best_model.pt"


def load_mtx(path):
    G = nx.Graph()
    first = True
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            p=line.split()
            if first:
                G.add_nodes_from(range(int(p[0])))
                first=False
            else:
                G.add_edge(int(p[0])-1,int(p[1])-1)
    return G


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=BranchingMLP().to(device)
    ck=torch.load(CHECKPOINT,map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    print("="*70)
    print("BHOSLIB ML BRANCHING")
    print("="*70)
    
    results = []
    for folder in [BHOSLIB_DIR / "frb30-15-1"]:
        mtx=folder/(folder.name+".mtx")
        if not mtx.exists():
            continue

        clique=load_mtx(mtx)
        G=nx.complement(clique)

        scip=Model(folder.name)
        #scip.hideOutput(True)
        scip.setPresolve(SCIP_PARAMSETTING.DEFAULT)
        scip.setHeuristics(SCIP_PARAMSETTING.DEFAULT)
        scip.setSeparating(SCIP_PARAMSETTING.DEFAULT)

        x={}
        for v in G.nodes():
            x[v]=scip.addVar(name=f"x_{v}",vtype="B")

        for u,v in G.edges():
            scip.addCons(x[u]+x[v]<=1)

        scip.setObjective(
            quicksum(x[v] for v in G.nodes()),
            "maximize"
        )
        
        feature_builder = MLFeatureBuilder(G)

        branch_rule = MLBranchRule(
            model,
            feature_builder,
            device
        )

        scip.includeBranchrule(
            branch_rule,
            "MLBranch",
            "ML branching rule",
            priority=1000000,
            maxdepth=-1,
            maxbounddist=1.0
        )

        print("Processing:",folder.name)

        start=time.time()

        scip.optimize()

        elapsed = time.time() - start

        result = {
            "graph": folder.name,
            "time": elapsed,
            "nodes": scip.getNNodes(),
            "objective": (
                scip.getObjVal()
                if scip.getNSols() > 0
                else None
            ),
            "gap": scip.getGap(),
            "ml_calls": branch_rule.calls
        }

        results.append(result)

        print(result)
    
    with open(
        "bhoslib_ml_depth10_results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

if __name__=="__main__":
    main()
