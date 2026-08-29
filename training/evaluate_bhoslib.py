# BHOSLIB evaluator
# Uses SCIPSBDataCollector directly.
# No solver_training.solve_training_instance.
# No separate BHOSLIB dataset.

import numpy as np
import torch
import networkx as nx
from metrics import compute_metrics
from pyscipopt import Model, SCIP_PARAMSETTING, quicksum

from branching import SCIPSBDataCollector
from features import CentralityFeatureExtractor, LubyFeatureExtractor, MWUAVertexFeatureExtractor
from model import BranchingMLP
from problem import build_vertex_cover_problem


BHOSLIB_DIR = "graphs/bhoslib"
CHECKPOINT = "checkpoints/best_model.pt"


class MemoryWriter:
    def __init__(self):
        self.samples = []

    def save(self, sample):
        self.samples.append(sample)


def load_mtx_graph(path):
    G = nx.Graph()
    first = True

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
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


def static_features(G):

    c = CentralityFeatureExtractor().compute(G)
    l = LubyFeatureExtractor().compute(G)
    m = MWUAVertexFeatureExtractor().compute(
        build_vertex_cover_problem(G)
    )

    out = {}

    for v in G.nodes():
        out[v] = {
            "pagerank": float(c.pagerank[v]),
            "mwua_xavg": float(m.x_avg[v]),
            "mwua_weight_min": float(m.weight_min[v]),
            "mwua_weight_max": float(m.weight_max[v]),
            "mwua_weight_avg": float(m.weight_avg[v]),
            "luby_frequency": float(l.frequency[v]),
        }

    return out


def collect_samples(G, name):

    writer = MemoryWriter()

    scip = Model(name)

    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)

    x = {}

    for v in G.nodes():
        x[v] = scip.addVar(
            name=f"x_{v}",
            vtype="B"
        )

    for u, v in G.edges():
        scip.addCons(x[u] + x[v] <= 1)

    scip.setObjective(
        quicksum(x[v] for v in G.nodes()),
        "maximize"
    )

    collector = SCIPSBDataCollector(
        static_features(G),
        G,
        name,
        writer,
        max_sb_nodes=50,
        strongbranch_itlim=100
    )

    scip.includeBranchrule(
        collector,
        "SBDataCollector",
        "BHOSLIB collector",
        priority=1000000,
        maxdepth=-1,
        maxbounddist=1.0
    )

    scip.optimize()

    return writer.samples


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BranchingMLP().to(device)

    ck = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(ck["model"])
    model.eval()

    metrics_sum = {
    "top1": 0.0,
    "top3": 0.0,
    "top5": 0.0,
    "mrr": 0.0,
    "rank": 0.0,
    "spearman": 0.0,
    "kendall": 0.0,
}

    total = 0

    for folder in sorted(__import__("pathlib").Path(BHOSLIB_DIR).iterdir()):

        mtx = folder / f"{folder.name}.mtx"

        if not mtx.exists():
            continue

        print("Processing:", folder.name)

        G = nx.complement(load_mtx_graph(mtx))

        samples = collect_samples(
            G,
            folder.name
        )

        for s in samples:

            x = torch.tensor(
                s.candidate_features,
                dtype=torch.float32,
                device=device
            )

            with torch.no_grad():
                pred = model(x).cpu().numpy()

            target = np.where(
                s.candidate_ids == s.chosen_variable
            )[0][0]

            m = compute_metrics(
                pred,
                np.asarray(
                    s.sb_scores,
                    dtype=np.float32
                ),
                int(target)
            )

            for k in metrics_sum:
                metrics_sum[k] += m[k]

            total += 1

    print("=" * 70)
    print("BHOSLIB RESULTS")
    print("=" * 70)

    print("Samples:", total)

    for k, v in metrics_sum.items():
        value = v / total

        if k in ["top1", "top3", "top5"]:
            print(
                f"{k}: {value*100:.2f}%"
            )
        else:
            print(
                f"{k}: {value:.4f}"
            )


if __name__ == "__main__":
    main()
