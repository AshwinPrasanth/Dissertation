from pathlib import Path

import json
import numpy as np

from scipy.stats import spearmanr, kendalltau

from xgboost import XGBRanker


BASE_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/ranking"
).expanduser()

MODEL_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/models/ablation"
).expanduser()


FEATURE_NAMES = np.load(
    BASE_DIR / "feature_names.npy",
    allow_pickle=True
).tolist()


'''FEATURE_SETS = {

    "structural": [
        "degree",
        "weighted_degree",
        "hyperedge_participation",
        "mean_incident_edge_size",
        "max_incident_edge_size"
    ],

    "structural_mwu": [
        "degree",
        "weighted_degree",
        "hyperedge_participation",
        "mwu_weighted_degree",
        "mean_incident_mwu",
        "max_incident_mwu",
        "mean_incident_edge_size",
        "max_incident_edge_size"
    ],

    "structural_mwu_certainty": [
        "degree",
        "weighted_degree",
        "hyperedge_participation",
        "mwu_weighted_degree",
        "mean_incident_mwu",
        "max_incident_mwu",
        "mean_incident_edge_size",
        "max_incident_edge_size",
        "x_avg",
        "certainty"
    ],

    "static": [
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
        "certainty"
    ],

    "static_dynamic": FEATURE_NAMES
}'''
FEATURE_SETS = {

    "structural": [
        "degree",
        "weighted_degree",
        "hyperedge_participation",
        "mean_incident_edge_size",
        "max_incident_edge_size"
    ],

    "structural_mwu": [
        "degree",
        "weighted_degree",
        "hyperedge_participation",
        "mwu_weighted_degree",
        "mean_incident_mwu",
        "max_incident_mwu",
        "mean_incident_edge_size",
        "max_incident_edge_size"
    ],

    "structural_mwu_xavg": [
        "degree",
        "weighted_degree",
        "hyperedge_participation",
        "mwu_weighted_degree",
        "mean_incident_mwu",
        "max_incident_mwu",
        "mean_incident_edge_size",
        "max_incident_edge_size",
        "x_avg"
    ],

    "structural_mwu_certainty": [
        "degree",
        "weighted_degree",
        "hyperedge_participation",
        "mwu_weighted_degree",
        "mean_incident_mwu",
        "max_incident_mwu",
        "mean_incident_edge_size",
        "max_incident_edge_size",
        "certainty"
    ],

    "structural_mwu_xavg_certainty": [
        "degree",
        "weighted_degree",
        "hyperedge_participation",
        "mwu_weighted_degree",
        "mean_incident_mwu",
        "max_incident_mwu",
        "mean_incident_edge_size",
        "max_incident_edge_size",
        "x_avg",
        "certainty"
    ]
}

def get_feature_indices(
    feature_names
):

    return [
        FEATURE_NAMES.index(name)
        for name in feature_names
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
            spearman.append(rho)

        if np.isfinite(tau):
            kendall.append(tau)

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


def train_experiment(
    name,
    feature_names,
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    groups_val,
    X_test,
    y_test,
    groups_test
):

    indices = get_feature_indices(
        feature_names
    )

    Xtr = X_train[:, indices]
    Xv = X_val[:, indices]
    Xt = X_test[:, indices]

    print()
    print("=" * 70)
    print(
        "EXPERIMENT:",
        name
    )
    print(
        "Features:",
        len(indices)
    )

    print(
        feature_names
    )

    print("=" * 70)

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        tree_method="hist",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=1.0,
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
        Xtr,
        y_train,
        group=groups_train,
        eval_set=[
            (Xv, y_val)
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

    print()
    print(
        "Predicting validation..."
    )

    val_predictions = model.predict(
        Xv
    )

    val_metrics = ranking_metrics(
        y_val,
        val_predictions,
        groups_val
    )

    print()
    print(
        "Validation metrics"
    )

    for metric, value in val_metrics.items():

        print(
            f"{metric}: {value:.6f}"
        )

    print()
    print(
        "Predicting test..."
    )

    test_predictions = model.predict(
        Xt
    )

    test_metrics = ranking_metrics(
        y_test,
        test_predictions,
        groups_test
    )

    print()
    print(
        "Test metrics"
    )

    for metric, value in test_metrics.items():

        print(
            f"{metric}: {value:.6f}"
        )

    model_path = (
        MODEL_DIR /
        f"xgb_{name}.json"
    )

    model.save_model(
        model_path
    )

    np.save(
        MODEL_DIR /
        f"{name}_val_predictions.npy",
        val_predictions
    )

    np.save(
        MODEL_DIR /
        f"{name}_test_predictions.npy",
        test_predictions
    )

    return {
        "features": feature_names,
        "feature_count": len(indices),
        "best_iteration": int(
            model.best_iteration
        ),
        "best_validation_score": float(
            model.best_score
        ),
        "validation": val_metrics,
        "test": test_metrics
    }


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

    print(
        "Loading test data..."
    )

    X_test, y_test, groups_test = load_split(
        "test"
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

    print(
        "Test:",
        X_test.shape,
        "groups:",
        len(groups_test)
    )

    results = {}

    for name, feature_names in FEATURE_SETS.items():

        results[name] = train_experiment(
            name,
            feature_names,
            X_train,
            y_train,
            groups_train,
            X_val,
            y_val,
            groups_val,
            X_test,
            y_test,
            groups_test
        )

    with open(
        MODEL_DIR / "ablation_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("FINAL ABLATION RESULTS")
    print("=" * 70)

    for name, result in results.items():

        print()
        print(name)

        print(
            "  Validation Top1:",
            f"{result['validation']['Top1 Accuracy']:.6f}"
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
        MODEL_DIR / "ablation_results.json"
    )


if __name__ == "__main__":
    main()