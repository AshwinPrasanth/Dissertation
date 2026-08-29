from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/train"
).expanduser()

OUTPUT_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/ranking"
).expanduser()


FEATURE_COLUMNS = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21
]


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_groups = []
    all_features = []
    all_labels = []

    total_groups = 0
    total_rows = 0

    files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    print(
        "Training instances:",
        len(files)
    )

    for file in files:

        print(
            "Processing:",
            file.name
        )

        df = pd.read_csv(
            file,
            header=None
        )

        groups = df.groupby(
            0,
            sort=False
        )

        valid_groups = 0

        for group_id, group in groups:

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

            valid_groups += 1

        total_groups += valid_groups
        total_rows += valid_groups * 32

        print(
            "  valid states:",
            valid_groups
        )

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

    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )

    np.save(
        OUTPUT_DIR / "groups.npy",
        groups
    )

    feature_names = np.asarray([
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

    np.save(
        OUTPUT_DIR / "feature_names.npy",
        feature_names
    )

    print()
    print(
        "Final X shape:",
        X.shape
    )

    print(
        "Final y shape:",
        y.shape
    )

    print(
        "Groups:",
        len(groups)
    )

    print(
        "Rows:",
        groups.sum()
    )

    print(
        "Feature dimension:",
        X.shape[1]
    )

    print(
        "Positive labels:",
        y.sum()
    )

    print(
        "Negative labels:",
        len(y) - y.sum()
    )


if __name__ == "__main__":
    main()
