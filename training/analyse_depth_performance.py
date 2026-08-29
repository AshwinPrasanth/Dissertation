from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRanker
from scipy.stats import spearmanr, kendalltau


DATA_DIR = Path(
    "../results/ranking_dataset_xgb"
)

MODEL_PATH = Path(
    "../checkpoints/xgb_ranker.json"
)

OUTPUT_DIR = Path(
    "../results/depth_analysis"
)


DEPTH_BUCKETS = {
    "0-5": (0, 5),
    "5-10": (6, 10),
    "10-20": (11, 20),
    "20+": (21, 999),
}


def get_bucket(depth):

    for name, (low, high) in DEPTH_BUCKETS.items():

        if low <= depth <= high:
            return name

    return None



def ndcg_at_k(true, pred, k):

    order = np.argsort(
        -pred
    )[:k]

    gains = (
        2 ** true[order]
        - 1
    )

    discounts = np.log2(
        np.arange(len(order)) + 2
    )

    dcg = np.sum(
        gains / discounts
    )


    ideal = np.argsort(
        -true
    )[:k]


    ideal_gains = (
        2 ** true[ideal]
        - 1
    )


    idcg = np.sum(
        ideal_gains / discounts
    )


    if idcg == 0:
        return 0.0


    return dcg / idcg



def hit_at_k(true, pred, k):

    top = np.argsort(
        -pred
    )[:k]


    best = np.argmax(
        true
    )


    return int(
        best in top
    )



def reciprocal_rank(true, pred):

    order = np.argsort(
        -pred
    )


    best = np.argmax(
        true
    )


    rank = np.where(
        order == best
    )[0][0]


    return 1.0 / (rank + 1)



def pairwise_accuracy(true, pred):

    correct = 0
    total = 0

    n = len(true)


    for i in range(n):

        for j in range(i + 1, n):

            if true[i] == true[j]:
                continue


            true_order = (
                true[i] > true[j]
            )

            pred_order = (
                pred[i] > pred[j]
            )


            if true_order == pred_order:
                correct += 1


            total += 1


    if total == 0:
        return 0.0


    return correct / total



def sb_regret(true, pred):

    chosen = np.argmax(
        pred
    )


    best = np.max(
        true
    )


    return (
        best - true[chosen]
    )



def normalized_sb_score(true, pred):

    chosen = np.argmax(
        pred
    )


    selected = true[chosen]

    minimum = np.min(
        true
    )

    maximum = np.max(
        true
    )


    if maximum == minimum:
        return 1.0


    return (
        selected - minimum
    ) / (
        maximum - minimum
    )



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


    model = XGBRanker()

    model.load_model(
        MODEL_PATH
    )


    print(
        "Loaded:",
        MODEL_PATH
    )


    results = {
        key: []
        for key in DEPTH_BUCKETS
    }


    start = 0


    for i, size in enumerate(groups):

        end = start + size


        true = y[start:end]

        pred = model.predict(
            X[start:end]
        )


        bucket = get_bucket(
            int(depths[i])
        )


        if bucket is None:
            start = end
            continue


        results[bucket].append(
            {

                "top1":
                int(
                    np.argmax(pred)
                    ==
                    np.argmax(true)
                ),


                "hit3":
                hit_at_k(
                    true,
                    pred,
                    3
                ),


                "hit5":
                hit_at_k(
                    true,
                    pred,
                    5
                ),


                "hit10":
                hit_at_k(
                    true,
                    pred,
                    10
                ),


                "mrr":
                reciprocal_rank(
                    true,
                    pred
                ),


                "ndcg1":
                ndcg_at_k(
                    true,
                    pred,
                    1
                ),


                "ndcg3":
                ndcg_at_k(
                    true,
                    pred,
                    3
                ),


                "ndcg5":
                ndcg_at_k(
                    true,
                    pred,
                    5
                ),


                "ndcg10":
                ndcg_at_k(
                    true,
                    pred,
                    10
                ),


                "spearman":
                spearmanr(
                    true,
                    pred
                ).statistic,


                "kendall":
                kendalltau(
                    true,
                    pred
                ).statistic,


                "pairwise_accuracy":
                pairwise_accuracy(
                    true,
                    pred
                ),


                "sb_regret":
                sb_regret(
                    true,
                    pred
                ),


                "normalized_sb_score":
                normalized_sb_score(
                    true,
                    pred
                )

            }
        )


        start = end



    rows = []


    for bucket, values in results.items():

        if not values:
            continue


        df = pd.DataFrame(
            values
        )


        row = {
            "depth": bucket,
            "nodes": len(df)
        }


        for col in df.columns:

            row[col] = df[col].mean()


        rows.append(
            row
        )


    output = pd.DataFrame(
        rows
    )


    print()
    print("="*80)
    print("DEPTH ANALYSIS")
    print("="*80)

    print(output)


    output.to_csv(
        OUTPUT_DIR / "depth_metrics.csv",
        index=False
    )


    plot_metrics(
        output
    )


def plot_metrics(df):


    plots = [
        (
            ["ndcg5"],
            "NDCG@5 vs Depth",
            "ndcg5_depth.png"
        ),

        (
            ["hit5"],
            "Hit@5 vs Depth",
            "hit5_depth.png"
        ),

        (
            ["top1"],
            "Top-1 Accuracy vs Depth",
            "top1_depth.png"
        ),

        (
            [
                "top1",
                "hit5",
                "ndcg5",
                "mrr",
                "pairwise_accuracy"
            ],
            "Ranking Metrics vs Depth",
            "combined_depth_metrics.png"
        )
    ]


    for metrics, title, filename in plots:

        plt.figure(
            figsize=(9,5)
        )


        for metric in metrics:

            plt.plot(
                df["depth"],
                df[metric],
                marker="o",
                label=metric
            )


        plt.xlabel(
            "Depth"
        )

        plt.ylabel(
            "Score"
        )

        plt.title(
            title
        )

        plt.legend()

        plt.grid()


        plt.savefig(
            OUTPUT_DIR / filename,
            bbox_inches="tight"
        )

        plt.close()



if __name__ == "__main__":
    main()