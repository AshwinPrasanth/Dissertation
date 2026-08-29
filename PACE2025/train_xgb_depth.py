from pathlib import Path

import json
import numpy as np

from xgboost import XGBRanker


BASE_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/ranking"
).expanduser()

MODEL_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/models/depth"
).expanduser()


FEATURE_NAMES = [
    "degree",
    "weighted_degree",
    "hyperedge_participation",
    "mwu_weighted_degree",
    "mean_incident_mwu",
    "max_incident_mwu",
    "mean_incident_edge_size",
    "max_incident_edge_size"
]


DEPTHS = [
1,2
]


def ranking_metrics(
    y_true,
    y_pred,
    groups
):

    top1 = []
    hit3 = []
    hit5 = []
    mrr = []
    ndcg5 = []

    start = 0

    for size in groups:

        end = start + size

        true = y_true[start:end]
        pred = y_pred[start:end]

        order = np.argsort(-pred)

        positive = np.argmax(true)

        rank = (
            np.where(order == positive)[0][0] + 1
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

        selected = order[:5]

        dcg = 0.0

        for i, index in enumerate(selected):

            dcg += (
                (2 ** true[index] - 1)
                /
                np.log2(i + 2)
            )

        ideal = np.sort(true)[::-1][:5]

        idcg = 0.0

        for i, value in enumerate(ideal):

            idcg += (
                (2 ** value - 1)
                /
                np.log2(i + 2)
            )

        if idcg > 0:

            ndcg5.append(
                dcg / idcg
            )

        else:

            ndcg5.append(0.0)

        start = end

    return {
        "Top1 Accuracy": float(np.mean(top1)),
        "Hit@3": float(np.mean(hit3)),
        "Hit@5": float(np.mean(hit5)),
        "MRR": float(np.mean(mrr)),
        "NDCG@5": float(np.mean(ndcg5))
    }


def load_split(split):

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

    print("Loading data...")

    X_train, y_train, groups_train = load_split(
        "train"
    )

    X_val, y_val, groups_val = load_split(
        "val"
    )

    X_test, y_test, groups_test = load_split(
        "test"
    )

    print()
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    results = {}

    for depth in DEPTHS:

        print()
        print("=" * 70)
        print(
            "MAX DEPTH:",
            depth
        )
        print("=" * 70)

        model = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@5",
            tree_method="hist",
            max_depth=depth,
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
        print("Training...")

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
            "Best iteration:",
            model.best_iteration
        )

        print(
            "Best validation score:",
            model.best_score
        )

        val_predictions = model.predict(
            X_val
        )

        test_predictions = model.predict(
            X_test
        )

        val_metrics = ranking_metrics(
            y_val,
            val_predictions,
            groups_val
        )

        test_metrics = ranking_metrics(
            y_test,
            test_predictions,
            groups_test
        )

        print()
        print("Validation metrics")

        for name, value in val_metrics.items():

            print(
                f"{name}: {value:.6f}"
            )

        print()
        print("Test metrics")

        for name, value in test_metrics.items():

            print(
                f"{name}: {value:.6f}"
            )

        model.save_model(
            MODEL_DIR /
            f"xgb_depth_{depth}.json"
        )

        results[str(depth)] = {
            "max_depth": depth,
            "best_iteration": int(
                model.best_iteration
            ),
            "best_validation_score": float(
                model.best_score
            ),
            "validation": val_metrics,
            "test": test_metrics
        }

    with open(
        MODEL_DIR / "depth_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("FINAL DEPTH RESULTS")
    print("=" * 70)

    for depth, result in results.items():

        print()
        print(
            f"depth={depth}"
        )

        print(
            "  Best iteration:",
            result["best_iteration"]
        )

        print(
            "  Validation NDCG@5:",
            f"{result['validation']['NDCG@5']:.6f}"
        )

        print(
            "  Test Top1:",
            f"{result['test']['Top1 Accuracy']:.6f}"
        )

        print(
            "  Test Hit@3:",
            f"{result['test']['Hit@3']:.6f}"
        )

        print(
            "  Test Hit@5:",
            f"{result['test']['Hit@5']:.6f}"
        )

        print(
            "  Test MRR:",
            f"{result['test']['MRR']:.6f}"
        )

        print(
            "  Test NDCG@5:",
            f"{result['test']['NDCG@5']:.6f}"
        )

    print()
    print(
        "Results saved to:",
        MODEL_DIR / "depth_results.json"
    )


if __name__ == "__main__":
    main()