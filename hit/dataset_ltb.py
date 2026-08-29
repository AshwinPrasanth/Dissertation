import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BranchingDataset(Dataset):

    def __init__(
        self,
        dataset_dir,
        graph_names=None,
        normalize_targets=False,
        return_metadata=True,
    ):

        self.dataset_dir = Path(dataset_dir)

        self.normalize_targets = normalize_targets

        self.return_metadata = return_metadata

        self.samples = []

        self._load_dataset(
            graph_names
        )

    def _load_dataset(
        self,
        graph_names,
    ):

        if graph_names is None:

            pkl_files = sorted(
                self.dataset_dir.glob(
                    "*.pkl"
                )
            )

        else:

            pkl_files = []

            for name in graph_names:

                pkl_files.append(
                    self.dataset_dir /
                    f"{name}.pkl"
                )
                
        filtered = 0
        for pkl in pkl_files:

            if not pkl.exists():

                continue

            with open(
                pkl,
                "rb",
            ) as f:

                while True:
                    try:
                        sample = pickle.load(f)
                    except EOFError:
                        break

                    # Skip samples with no branching signal
                    if len(np.unique(sample.sb_scores)) == 1:
                        filtered += 1
                        continue

                    self.samples.append(sample)

        print()

        print(
            "=" * 70
        )

        print(
            "BRANCHING DATASET"
        )

        print(
            "=" * 70
        )

        print(
            f"Directory : {self.dataset_dir}"
        )

        print(
            f"Graphs    : {len(pkl_files)}"
        )

        print(
            f"Samples   : {len(self.samples)}"
        )
        
        print(
            f"Filtered : {filtered}"
        )

    def __len__(
        self,
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx,
    ):

        sample = self.samples[
            idx
        ]

        x = torch.tensor(
            sample.candidate_features,
            dtype=torch.float32,
        )

        sb_scores = np.asarray(
            sample.sb_scores,
            dtype=np.float32,
        )

        if self.normalize_targets:

            maximum = np.max(
                sb_scores
            )

            if maximum > 0:

                sb_scores = (
                    sb_scores /
                    maximum
                )

        y_score = torch.tensor(
            sb_scores,
            dtype=torch.float32,
        )

        chosen = int(
            sample.chosen_variable
        )

        y_class = torch.tensor(
            chosen,
            dtype=torch.long,
        )

        output = {

            "features":
                x,

            "scores":
                y_score,

            "chosen":
                y_class,
        }

        if self.return_metadata:

            output[
                "graph_name"
            ] = sample.graph_name

            output[
                "depth"
            ] = sample.depth

            output[
                "node_number"
            ] = sample.node_number

            output[
                "residual_n"
            ] = sample.residual_n

            output[
                "residual_m"
            ] = sample.residual_m

            output[
                "candidate_ids"
            ] = torch.tensor(
                sample.candidate_ids,
                dtype=torch.long,
            )

            output[
                "feature_names"
            ] = list(
                sample.feature_names
            )

            output[
                "best_sb_score"
            ] = float(
                sample.best_sb_score
            )

        return output


def collate_fn(
    batch,
):

    return batch


def load_dataset(
    dataset_dir,
    graph_names=None,
    normalize_targets=False,
):

    return BranchingDataset(

        dataset_dir=dataset_dir,

        graph_names=graph_names,

        normalize_targets=normalize_targets,

        return_metadata=True,
    )
    
def train_val_split(
    dataset,
    train_fraction=0.8,
    seed=42,
):

    rng = np.random.default_rng(
        seed
    )

    indices = np.arange(
        len(dataset)
    )

    rng.shuffle(
        indices
    )

    split = int(
        train_fraction *
        len(indices)
    )

    train_indices = indices[
        :split
    ]

    val_indices = indices[
        split:
    ]

    train = torch.utils.data.Subset(
        dataset,
        train_indices.tolist(),
    )

    val = torch.utils.data.Subset(
        dataset,
        val_indices.tolist(),
    )

    return (
        train,
        val,
    )


def graph_split(
    dataset,
    train_graphs,
    test_graphs,
):

    train_samples = []

    test_samples = []

    for i, sample in enumerate(
        dataset.samples
    ):

        if sample.graph_name in train_graphs:

            train_samples.append(
                i
            )

        elif sample.graph_name in test_graphs:

            test_samples.append(
                i
            )

    train = torch.utils.data.Subset(
        dataset,
        train_samples,
    )

    test = torch.utils.data.Subset(
        dataset,
        test_samples,
    )

    return (
        train,
        test,
    )


if __name__ == "__main__":

    dataset = BranchingDataset(

        dataset_dir=
        "../results/dimacs_ltb_training",

        normalize_targets=False,
    )

    print()

    print(
        "=" * 70
    )

    print(
        "FIRST SAMPLE"
    )

    print(
        "=" * 70
    )

    sample = dataset[
        0
    ]

    print(
        "Graph:",
        sample["graph_name"],
    )

    print(
        "Features:",
        sample["features"].shape,
    )

    print(
        "Scores:",
        sample["scores"].shape,
    )

    print(
        "Chosen:",
        sample["chosen"],
    )

    print(
        "Residual:",
        sample["residual_n"],
    )

    print(
        "Depth:",
        sample["depth"],
    )

    train, val = train_val_split(
        dataset
    )

    print()

    print(
        "Train samples:",
        len(train),
    )

    print(
        "Validation samples:",
        len(val),
    )