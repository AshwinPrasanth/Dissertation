from pathlib import Path

import json
import numpy as np

from scipy.stats import spearmanr, kendalltau

from xgboost import XGBRanker


BASE_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/ranking"
).expanduser()

MODEL_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/models"
).expanduser()


FEATURE_NAMES = np.load(
    BASE_DIR / "feature_names.npy"
).tolist()


def ranking_metrics(
    y_true,
    y_pred,
    groups
):

    top1 = []
    hit3 = []
    hit5 = []
    mrr = []
    ndcg1 = []
    ndcg3 = []
    ndcg5 = []
    spearman = []
    kendall = []

    start = 0

    for size in groups:

        end = start + size

        true = y_true[start:end]
        pred = y_pred[start:end]

        order = np.argsort(
            -pred
        )

        positive = np.argmax(
            true
        )

        rank = (
            np.where(
                order == positive
            )[0][0] + 1
        )

        top1.append(
            int(rank == 1)
        )

        hit3.append(
            int(rank <= 3)
        )

        hit5.append(
            int(rank <= 5)
        )

        mrr.append(
            1.0 / rank
        )

        rho, _ = spearmanr(
            true,
            pred
        )

        tau, _ = kendalltau(
            true,
            pred
        )

        if np.isfinite(rho):
            spearman.append(
                rho
            )

        if np.isfinite(tau):
            kendall.append(
                tau
            )

        for k, values in [
            (1, ndcg1),
            (3, ndcg3),
            (5, ndcg5)
        ]:

            selected = order[:k]

            dcg = 0.0

            for i, index in enumerate(selected):

                dcg += (
                    (2 ** true[index] - 1)
                    /
                    np.log2(i + 2)
                )

            ideal = np.sort(
                true
            )[::-1][:k]

            idcg = 0.0

            for i, value in enumerate(ideal):

                idcg += (
                    (2 ** value - 1)
                    /
                    np.log2(i + 2)
                )

            if idcg > 0:

                values.append(
                    dcg / idcg
                )

            else:

                values.append(
                    0.0
                )

        start = end

    return {
        "Top1 Accuracy": float(
            np.mean(top1)
        ),
        "Hit@3": float(
            np.mean(hit3)
        ),
        "Hit@5": float(
            np.mean(hit5)
        ),
        "MRR": float(
            np.mean(mrr)
        ),
        "NDCG@1": float(
            np.mean(ndcg1)
        ),
        "NDCG@3": float(
            np.mean(ndcg3)
        ),
        "NDCG@5": float(
            np.mean(ndcg5)
        ),
        "Spearman rho": float(
            np.mean(spearman)
        ),
        "Kendall tau": float(
            np.mean(kendall)
        )
    }


def load_split(
    split
):

    if split == "train":
        directory = BASE_DIR
    else:
        directory = BASE_DIR / split

    X = np.load(
        directory / "X.npy"
    )

    y = np.load(
        directory / "y.npy"
    )

    groups = np.load(
        directory / "groups.npy"
    )

    return X, y, groups

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Loading training data..."
    )

    X_train, y_train, groups_train = load_split(
        "train"
    )

    print(
        "Loading validation data..."
    )

    X_val, y_val, groups_val = load_split(
        "val"
    )

    print()

    print(
        "Train:",
        X_train.shape,
        "groups:",
        len(groups_train)
    )

    print(
        "Validation:",
        X_val.shape,
        "groups:",
        len(groups_val)
    )

    model=XGBRanker(
    objective="rank:ndcg",
    eval_metric="ndcg@5",
    tree_method="hist",
    max_depth=2,
    learning_rate=0.05,
    n_estimators=500,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=16,
    early_stopping_rounds=20
)

    print()
    print(
        "Training..."
    )

    model.fit(
        X_train,
        y_train,
        group=groups_train,
        eval_set=[
            (X_val, y_val)
        ],
        eval_group=[
            groups_val
        ],
        verbose=True
    )

    print()
    print(
        "Predicting validation set..."
    )

    predictions = model.predict(
        X_val
    )

    metrics = ranking_metrics(
        y_val,
        predictions,
        groups_val
    )

    print()
    print(
        "Validation metrics"
    )

    for name, value in metrics.items():

        print(
            f"{name}: {value:.6f}"
        )

    model.save_model(
        MODEL_DIR / "xgb_ranker_baseline.json"
    )

    np.save(
        MODEL_DIR / "validation_predictions.npy",
        predictions
    )

    with open(
        MODEL_DIR / "baseline_metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    with open(
        MODEL_DIR / "feature_names.json",
        "w"
    ) as f:

        json.dump(
            FEATURE_NAMES,
            f,
            indent=2
        )

    print()
    print(
        "Model saved to:",
        MODEL_DIR / "xgb_ranker_baseline.json"
    )


if __name__ == "__main__":
    main()
