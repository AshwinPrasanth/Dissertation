import json
from pathlib import Path

import numpy as np

from xgboost import XGBRanker

from scipy.stats import spearmanr, kendalltau

from sklearn.model_selection import train_test_split


DATA_DIR = Path(
    "../results/ranking_dataset_xgb"
)

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

    gains = (
        2 ** true_scores[order]
        - 1
    )

    discounts = np.log2(
        np.arange(len(order))+2
    )

    dcg = np.sum(
        gains / discounts
    )

    ideal = np.argsort(
        -true_scores
    )[:k]

    ideal_gains = (
        2 ** true_scores[ideal]
        - 1
    )

    idcg = np.sum(
        ideal_gains / discounts
    )

    if idcg == 0:
        return 0

    return dcg/idcg



def main():

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


    candidate_graph_ids = np.repeat(
        node_graph_ids,
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


    MODEL_DIR.mkdir(
        exist_ok=True
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
