import json
from pathlib import Path

import numpy as np

from xgboost import XGBRanker

from scipy.stats import spearmanr, kendalltau

from sklearn.model_selection import train_test_split


DATA_DIR = Path("../results/ranking_dataset_xgb")

MODEL_DIR = Path(
    "../checkpoints"
)


def split_by_graph(
    graph_ids,
    test_size=0.2,
    seed=42
):

    graphs = np.unique(
        graph_ids
    )

    train_graphs, test_graphs = train_test_split(
        graphs,
        test_size=test_size,
        random_state=seed
    )

    train_mask = np.isin(
        graph_ids,
        train_graphs
    )

    test_mask = np.isin(
        graph_ids,
        test_graphs
    )

    return (
        train_mask,
        test_mask,
        train_graphs,
        test_graphs
    )


def ndcg_at_k(
    true_scores,
    pred_scores,
    k
):

    order = np.argsort(
        -pred_scores
    )[:k]

    ideal_order = np.argsort(
        -true_scores
    )[:k]


    discounts = np.log2(
        np.arange(len(order)) + 2
    )


    dcg = np.sum(
        (2 ** true_scores[order] - 1)
        / discounts
    )


    idcg = np.sum(
        (2 ** true_scores[ideal_order] - 1)
        / discounts
    )


    if idcg == 0:
        return 0.0

    return float(
        dcg / idcg
    )



def ranking_metrics(
    y_true,
    y_pred,
    groups
):

    ndcg1 = []
    ndcg3 = []
    ndcg5 = []

    hit1 = []
    hit3 = []
    hit5 = []

    mrr = []

    top1 = []


    start = 0


    for size in groups:

        end = start + size


        true = y_true[start:end]
        pred = y_pred[start:end]


        ranking = np.argsort(
            -pred
        )


        best_true = np.argmax(
            true
        )


        predicted_rank = np.where(
            ranking == best_true
        )[0][0] + 1


        top1.append(
            int(predicted_rank == 1)
        )


        hit1.append(
            int(predicted_rank <= 1)
        )

        hit3.append(
            int(predicted_rank <= 3)
        )

        hit5.append(
            int(predicted_rank <= 5)
        )


        mrr.append(
            1.0 / predicted_rank
        )


        ndcg1.append(
            ndcg_at_k(
                true,
                pred,
                1
            )
        )

        ndcg3.append(
            ndcg_at_k(
                true,
                pred,
                3
            )
        )

        ndcg5.append(
            ndcg_at_k(
                true,
                pred,
                5
            )
        )


        start = end


    return {

        "Top1 Accuracy":
            np.mean(top1),

        "Hit@1":
            np.mean(hit1),

        "Hit@3":
            np.mean(hit3),

        "Hit@5":
            np.mean(hit5),

        "MRR":
            np.mean(mrr),

        "NDCG@1":
            np.mean(ndcg1),

        "NDCG@3":
            np.mean(ndcg3),

        "NDCG@5":
            np.mean(ndcg5),
    }



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

    node_graph_ids = np.load(
        DATA_DIR / "graph_ids.npy"
    )
    
    depths = np.load(
    DATA_DIR / "depths.npy"
)


    candidate_graph_ids = np.repeat(
        node_graph_ids,
        groups
    )
    
    candidate_depths = np.repeat(
    depths,
    groups
)


    print(
        "X:",
        X.shape
    )

    print(
        "candidate graph ids:",
        candidate_graph_ids.shape
    )


    train_mask, test_mask, train_graphs, test_graphs = split_by_graph(
        np.unique(node_graph_ids)
    )


    train_mask = np.isin(
        candidate_graph_ids,
        train_graphs
    )

    test_mask = np.isin(
        candidate_graph_ids,
        test_graphs
    )


    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]


    train_node_mask = np.isin(
        node_graph_ids,
        train_graphs
    )

    test_node_mask = np.isin(
        node_graph_ids,
        test_graphs
    )


    train_groups = groups[
        train_node_mask
    ]

    test_groups = groups[
        test_node_mask
    ]


    print()
    print("="*60)
    print("DATA SPLIT")
    print("="*60)

    print(
        "Train graphs:",
        len(train_graphs)
    )

    print(
        "Test graphs:",
        len(test_graphs)
    )

    print(
        "Train rows:",
        X_train.shape
    )

    print(
        "Test rows:",
        X_test.shape
    )

    print(
        "Train ranking groups:",
        len(train_groups)
    )

    print(
        "Test ranking groups:",
        len(test_groups)
    )


    assert X_train.shape[0] == train_groups.sum()
    assert X_test.shape[0] == test_groups.sum()


    model = XGBRanker(
    objective="rank:pairwise",
    learning_rate=0.05,
    n_estimators=500,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    eval_metric="ndcg@5",
    verbosity=1,
)


    model.fit(
    X_train,
    y_train,
    group=train_groups,
    eval_set=[
        (
            X_test,
            y_test
        )
    ],
    eval_group=[
        test_groups
    ],
    verbose=True
)


    history = model.evals_result()


    with open(
        MODEL_DIR / "xgb_training_history.json",
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


    pred = model.predict(
        X_test
    )


    print()
    print("="*60)
    print("GLOBAL CORRELATION")
    print("="*60)

    metrics = ranking_metrics(
        y_test,
        pred,
        test_groups
    )


    print()

    print("="*60)
    print("RANKING METRICS")
    print("="*60)


    for key, value in metrics.items():

        print(
            f"{key:<15}: {value:.4f}"
        )


    print()

    print(
        "Spearman:",
        spearmanr(
            y_test,
            pred
        ).statistic
    )


    print(
        "Kendall:",
        kendalltau(
            y_test,
            pred
        ).statistic
    )


    


    model.save_model(
        MODEL_DIR / "xgb_ranker.json"
    )


    print()

    print(
        "Saved:"
    )

    print(
        MODEL_DIR / "xgb_ranker.json"
    )
    
if __name__=="__main__":
    main()