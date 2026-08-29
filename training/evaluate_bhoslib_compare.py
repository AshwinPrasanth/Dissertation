import csv
import time
from pathlib import Path

import torch
import networkx as nx

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

            p = line.split()

            if first:
                G.add_nodes_from(range(int(p[0])))
                first = False

            else:
                u = int(p[0])-1
                v = int(p[1])-1

                if u != v:
                    G.add_edge(u, v)

    return G



def build_problem(G, name):

    scip = Model(name)

    scip.setPresolve(SCIP_PARAMSETTING.DEFAULT)
    scip.setHeuristics(SCIP_PARAMSETTING.DEFAULT)
    scip.setSeparating(SCIP_PARAMSETTING.DEFAULT)

    x = {}

    for v in G.nodes():

        x[v] = scip.addVar(
            name=f"x_{v}",
            vtype="B"
        )


    for u,v in G.edges():

        scip.addCons(
            x[u]+x[v] <= 1
        )


    scip.setObjective(
        quicksum(x[v] for v in G.nodes()),
        "maximize"
    )

    return scip



def run_scip(G, name):

    print("\nSCIP DEFAULT:", name)

    scip = build_problem(
        G,
        name
    )

    start = time.time()

    scip.optimize()

    result = {

        "graph": name,
        "method": "SCIP",

        "time":
            time.time()-start,

        "nodes":
            scip.getNNodes(),

        "objective":
            scip.getObjVal()
            if scip.getNSols()
            else None,

        "gap":
            scip.getGap(),

        "ml_calls": 0
    }


    print(result)

    return result



def run_ml(G, name, model, device):

    print("\nML DEPTH 5:", name)

    scip = build_problem(
        G,
        name
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
        "ML depth 5",
        priority=1000000,
        maxdepth=-1,
        maxbounddist=1.0
    )


    start = time.time()

    scip.optimize()


    result = {

        "graph": name,
        "method": "ML-depth5",

        "time":
            time.time()-start,

        "nodes":
            scip.getNNodes(),

        "objective":
            scip.getObjVal()
            if scip.getNSols()
            else None,

        "gap":
            scip.getGap(),

        "ml_calls":
            branch_rule.calls
    }


    print(result)

    return result



def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model = BranchingMLP().to(device)

    ck = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        ck["model"]
    )

    model.eval()


    results = []


    for folder in sorted(BHOSLIB_DIR.iterdir()):

        mtx = folder / f"{folder.name}.mtx"

        if not mtx.exists():
            continue


        print("="*70)
        print(folder.name)
        print("="*70)


        clique = load_mtx(mtx)

        G = nx.complement(
            clique
        )


        results.append(
            run_scip(
                G,
                folder.name
            )
        )


        results.append(
            run_ml(
                G,
                folder.name,
                model,
                device
            )
        )


        with open(
            "bhoslib_scip_vs_ml_depth5.csv",
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=results[0].keys()
            )

            writer.writeheader()
            writer.writerows(results)



if __name__ == "__main__":
    main()