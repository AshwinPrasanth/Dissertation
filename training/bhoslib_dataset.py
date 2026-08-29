from pathlib import Path
import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

from features import CentralityFeatureExtractor, LubyFeatureExtractor, MWUAVertexFeatureExtractor
from problem import build_vertex_cover_problem


def load_mtx_graph(path):
    G = nx.Graph()
    first = True
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if first:
                G.add_nodes_from(range(int(parts[0])))
                first = False
            else:
                u, v = int(parts[0])-1, int(parts[1])-1
                if u != v:
                    G.add_edge(u, v)
    return G


def build_features(G):
    centrality = CentralityFeatureExtractor().compute(G)
    luby = LubyFeatureExtractor().compute(G)
    mwua = MWUAVertexFeatureExtractor().compute(
        build_vertex_cover_problem(G)
    )

    return np.asarray([
        [
            centrality.pagerank[v],
            mwua.x_avg[v],
            mwua.weight_min[v],
            mwua.weight_max[v],
            mwua.weight_avg[v],
            luby.frequency[v],
        ]
        for v in G.nodes()
    ], dtype=np.float32)


class BHOSLIBDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        root = Path(root)

        for folder in sorted(root.iterdir()):
            mtx = folder / f"{folder.name}.mtx"
            if not mtx.exists():
                continue

            clique = load_mtx_graph(mtx)
            mis = nx.complement(clique)

            self.samples.append({
                "name": folder.name,
                "features": torch.tensor(
                    build_features(mis),
                    dtype=torch.float32
                )
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
