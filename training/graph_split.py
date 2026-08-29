from collections import defaultdict
import random
from torch.utils.data import Subset

def graph_train_val_split(dataset, train_fraction=0.8, seed=42):
    graph_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        graph_to_indices[sample.graph_name].append(idx)
    graphs = list(graph_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(graphs)
    split = int(len(graphs) * train_fraction)
    train_graphs = set(graphs[:split])
    train_idx, val_idx = [], []
    for g, idxs in graph_to_indices.items():
        (train_idx if g in train_graphs else val_idx).extend(idxs)
    return Subset(dataset, train_idx), Subset(dataset, val_idx), sorted(train_graphs), sorted(set(graphs)-train_graphs)
