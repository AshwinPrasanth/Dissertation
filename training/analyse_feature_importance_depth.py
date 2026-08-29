from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRanker


DATA_DIR = Path(
    "../results/ranking_dataset_xgb"
)

OUTPUT_DIR = Path(
    "../results/feature_importance_depth"
)


DEPTH_BUCKETS = {
    "0-5": (0, 5),
    "5-10": (6, 10),
    "10-20": (11, 20),
    "20+": (21, 999),
}



def train_depth_model(
    X,
    y,
    groups
):

    model = XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.05,
        n_estimators=200,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
    )


    model.fit(
        X,
        y,
        group=groups,
        verbose=False
    )


    return model



def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    X = np.load(
        DATA_DIR / "X.npy"
    )

    y = np.load(
        DATA_DIR / "y.npy"
    )

    groups = np.load(
        DATA_DIR / "groups.npy"
    )


    depths = np.load(
        DATA_DIR / "depths.npy"
    )


    feature_names = np.load(
        DATA_DIR / "feature_names.npy",
        allow_pickle=True
    )


    print(
        "Candidates:",
        X.shape[0]
    )

    print(
        "Ranking nodes:",
        len(groups)
    )

    print(
        "Depth nodes:",
        len(depths)
    )


    assert len(groups) == len(depths)

    assert groups.sum() == X.shape[0]


    rows = []


    for bucket, (low, high) in DEPTH_BUCKETS.items():

        print()
        print("=" * 60)

        print(
            "Training depth:",
            bucket
        )


        indices = []

        selected_groups = []


        ptr = 0


        for node_group, node_depth in zip(
            groups,
            depths
        ):

            start = ptr

            end = ptr + int(node_group)


            if low <= node_depth <= high:

                indices.extend(
                    range(
                        start,
                        end
                    )
                )


                selected_groups.append(
                    int(node_group)
                )


            ptr = end



        if len(indices) == 0:

            print(
                "No samples found"
            )

            continue



        indices = np.asarray(
            indices,
            dtype=np.int64
        )


        X_depth = X[
            indices
        ]

        y_depth = y[
            indices
        ]


        groups_depth = np.asarray(
            selected_groups,
            dtype=np.int32
        )


        assert (
            groups_depth.sum()
            ==
            X_depth.shape[0]
        )


        print(
            "Candidates:",
            len(X_depth)
        )

        print(
            "Groups:",
            len(groups_depth)
        )


        model = train_depth_model(
            X_depth,
            y_depth,
            groups_depth
        )


        importance = (
            model
            .get_booster()
            .get_score(
                importance_type="gain"
            )
        )


        row = {
            "depth": bucket,
            "samples": X_depth.shape[0],
            "groups": len(groups_depth)
        }


        for i, name in enumerate(feature_names):

            row[name] = importance.get(
                f"f{i}",
                0.0
            )


        rows.append(
            row
        )


    df = pd.DataFrame(
        rows
    )


    df.to_csv(
        OUTPUT_DIR / "feature_importance_depth.csv",
        index=False
    )


    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE BY DEPTH")
    print("=" * 70)

    print(df)



if __name__ == "__main__":

    main()