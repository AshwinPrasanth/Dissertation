
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from solver_training import solve_training_instance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "training" / "graphs" / "Dimacs"
RESULT_DIR = PROJECT_ROOT / "results" / "dimacs_ltb_training500"
SUMMARY_FILE = RESULT_DIR / "dimacs_training_summary500.csv"
KAMIS_BUILD = PROJECT_ROOT / "CHSZLabLib" / "build-kamis"

sys.path.insert(0, str(KAMIS_BUILD))
import _kamis

MAX_SB_NODES = 500
CANDIDATE_LIMIT = None
STRONGBRANCH_ITLIM = 100
TIME_LIMIT = None
RANDOM_SEED = 42


def load_mtx_graph(path):
    G = nx.Graph()
    with open(path) as f:
        dims = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if not dims:
                n = int(parts[0])
                G.add_nodes_from(range(n))
                dims = True
                continue
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            if u != v:
                G.add_edge(u, v)
    return G


def graph_to_csr(G):
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    xadj = [0]
    adjncy = []
    for v in range(G.number_of_nodes()):
        adjncy.extend(sorted(G.neighbors(v)))
        xadj.append(len(adjncy))
    return (
        np.asarray(xadj, np.int32),
        np.asarray(adjncy, np.int32),
        np.asarray([], np.int32),
    )


def csr_to_graph(xadj, adjncy):
    G = nx.Graph()
    n = len(xadj) - 1
    G.add_nodes_from(range(n))
    for v in range(n):
        for u in adjncy[int(xadj[v]):int(xadj[v+1])]:
            u = int(u)
            if u > v:
                G.add_edge(v, u)
    return G


def family(name):
    s = name.lower()
    for p in ["brock","c-fat","dsjc","gen","hamming","johnson","keller","mann","p-hat","sanr","san"]:
        if s.startswith(p):
            return p
    if s.startswith("c"):
        return "C"
    return "other"


def discover():
    groups = defaultdict(list)
    for f in GRAPH_DIR.rglob("*.mtx"):
        groups[family(f.parent.name)].append(f)
    rng = random.Random(RANDOM_SEED)
    for v in groups.values():
        rng.shuffle(v)
    order = []
    while any(groups.values()):
        for k in sorted(groups):
            if groups[k]:
                order.append(groups[k].pop())
    return order


def load_summary():
    if not SUMMARY_FILE.exists():
        return []
    with open(SUMMARY_FILE) as f:
        return list(csv.DictReader(f))


def save(rows):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(SUMMARY_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_summary()
    done = {r["graph_name"] for r in rows}
    graphs = discover()

    print(f"Found {len(graphs)} DIMACS instances")

    for i, path in enumerate(graphs, 1):
        name = path.parent.name
        if name in done:
            continue

        print("=" * 80)
        print(f"[{i}/{len(graphs)}] {name}")

        clique = load_mtx_graph(path)
        mis = nx.complement(clique)

        raw_n = mis.number_of_nodes()
        raw_m = mis.number_of_edges()

        xadj, adjncy, vwgt = graph_to_csr(mis)
        cx, ca, rm = _kamis.redumis_kernel(xadj, adjncy, vwgt)
        core = csr_to_graph(np.asarray(cx, np.int32), np.asarray(ca, np.int32))

        result = solve_training_instance(
            G=core,
            graph_name=name,
            family=family(name),
            max_sb_nodes=MAX_SB_NODES,
            candidate_limit=CANDIDATE_LIMIT,
            strongbranch_itlim=STRONGBRANCH_ITLIM,
            time_limit=TIME_LIMIT,
            output_dir=RESULT_DIR,
        )

        result["original_n"] = raw_n
        result["original_m"] = raw_m
        result["core_n"] = core.number_of_nodes()
        result["core_m"] = core.number_of_edges()
        result["reduction_ratio"] = 1.0 - (core.number_of_nodes()/raw_n if raw_n else 0)
        result["source_file"] = str(path.relative_to(PROJECT_ROOT))

        rows.append(result)
        done.add(name)
        save(rows)

    print("Finished.")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
