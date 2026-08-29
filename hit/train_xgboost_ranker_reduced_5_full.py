from pathlib import Path

import numpy as np

from xgboost import XGBRanker

DATA_DIR = Path(
    "../results/ranking_dataset_xgb19_reduced_100"
)

MODEL_DIR = Path(
    "../checkpoints/xgb_reduced_19_100"
)


def main():

    MODEL_DIR.mkdir(
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

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    print(
        "Ranking groups:",
        len(groups)
    )

    print(
        "Total candidates:",
        groups.sum()
    )

    assert X.shape[0] == y.shape[0]

    assert X.shape[0] == groups.sum()

    print()
    print("=" * 60)
    print("TRAINING FINAL 100-BIN MODEL ON 100% DATA")
    print("=" * 60)

    model = XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.05,
        n_estimators=500,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        verbosity=1,
        n_jobs=16,
    )

    model.fit(
        X,
        y,
        group=groups,
        verbose=True
    )

    model.save_model(
        MODEL_DIR / "xgb_ranker_full_100.json"
    )

    print()
    print("Saved:")

    print(
        MODEL_DIR / "xgb_ranker_full_100.json"
    )


if __name__ == "__main__":

    main()