from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(
    "~/dissertation_ashwin/ml_training_data"
).expanduser()


FEATURE_COLUMNS = list(
    range(3, 22)
)


FEATURE_NAMES = np.asarray([
    "degree",
    "weighted_degree",
    "hyperedge_participation",
    "mwu_weighted_degree",
    "mean_incident_mwu",
    "max_incident_mwu",
    "mean_incident_edge_size",
    "max_incident_edge_size",
    "singleton_participation",
    "binary_participation",
    "x_avg",
    "certainty",
    "decision_level",
    "trail_size",
    "propagated",
    "conflicts",
    "evsids",
    "assigned",
    "assignment_level"
])


def process_split(
    split
):

    input_dir = BASE_DIR / split
    output_dir = BASE_DIR / "ranking" / split

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_features = []
    all_labels = []
    all_groups = []
    all_instance_ids = []

    total_states = 0
    total_rows = 0

    files = sorted(
        input_dir.glob("*.csv")
    )

    print(
        "\nProcessing",
        split
    )

    for file in files:

        print(
            "  ",
            file.name
        )

        df = pd.read_csv(
            file,
            header=None
        )

        valid_states = 0

        for _, group in df.groupby(
            0,
            sort=False
        ):

            if len(group) != 32:
                continue

            labels = group[22].to_numpy(
                dtype=np.int32
            )

            if labels.sum() != 1:
                continue

            features = group[
                FEATURE_COLUMNS
            ].to_numpy(
                dtype=np.float32
            )

            all_features.append(
                features
            )

            all_labels.append(
                labels
            )

            all_groups.append(
                32
            )

            all_instance_ids.append(
                file.stem
            )

            valid_states += 1

        print(
            "    valid states:",
            valid_states
        )

        total_states += valid_states
        total_rows += valid_states * 32

    X = np.vstack(
        all_features
    )

    y = np.concatenate(
        all_labels
    )

    groups = np.asarray(
        all_groups,
        dtype=np.int32
    )

    instance_ids = np.asarray(
        all_instance_ids
    )

    np.save(
        output_dir / "X.npy",
        X
    )

    np.save(
        output_dir / "y.npy",
        y
    )

    np.save(
        output_dir / "groups.npy",
        groups
    )

    np.save(
        output_dir / "instance_ids.npy",
        instance_ids
    )

    np.save(
        output_dir / "feature_names.npy",
        FEATURE_NAMES
    )

    print()
    print(
        split,
        "X:",
        X.shape
    )

    print(
        split,
        "y:",
        y.shape
    )

    print(
        split,
        "groups:",
        len(groups)
    )

    print(
        split,
        "rows:",
        groups.sum()
    )

    print(
        split,
        "positive labels:",
        y.sum()
    )

    print(
        split,
        "negative labels:",
        len(y) - y.sum()
    )

    print(
        split,
        "feature dimension:",
        X.shape[1]
    )

    assert (
        groups.sum() == len(X)
    )

    assert (
        y.sum() == len(groups)
    )

    assert np.all(
        groups == 32
    )


def main():

    process_split(
        "val"
    )

    process_split(
        "test"
    )


if __name__ == "__main__":
    main()
