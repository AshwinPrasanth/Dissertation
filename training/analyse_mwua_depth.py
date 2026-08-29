from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(
    "../results/ranking_dataset_xgb"
)

OUTPUT_DIR = Path(
    "../results/mwua_depth_analysis"
)


DEPTH_BUCKETS = {
    "0-5": (0,5),
    "5-10": (6,10),
    "10-20": (11,20),
    "20+": (21,999),
}


MWUA_FEATURES = [
    "mwua_xavg",
    "mwua_weight_min",
    "mwua_weight_max",
    "mwua_weight_avg",
]



def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    X = np.load(
        DATA_DIR/"X.npy"
    )


    depths = np.load(
        DATA_DIR/"candidate_depths.npy"
    )


    names = np.load(
        DATA_DIR/"feature_names.npy",
        allow_pickle=True
    )


    name_to_idx = {
        name:i
        for i,name in enumerate(names)
    }


    rows = []


    for bucket,(low,high) in DEPTH_BUCKETS.items():

        mask = (
            (depths>=low)
            &
            (depths<=high)
        )


        row = {
            "depth": bucket,
            "samples": int(mask.sum())
        }


        for feature in MWUA_FEATURES:

            idx = name_to_idx[feature]

            values = X[mask,idx]


            row[f"{feature}_mean"] = (
                np.mean(values)
            )

            row[f"{feature}_std"] = (
                np.std(values)
            )

            row[f"{feature}_median"] = (
                np.median(values)
            )


        rows.append(row)



    df = pd.DataFrame(
        rows
    )


    print("="*70)
    print("MWUA FEATURE EVOLUTION")
    print("="*70)

    print(df)


    df.to_csv(
        OUTPUT_DIR/"mwua_depth_statistics.csv",
        index=False
    )


    plot_means(df)

    plot_decay(df)



def plot_means(df):

    plt.figure(
        figsize=(9,5)
    )


    for f in MWUA_FEATURES:

        plt.plot(
            df["depth"],
            df[f+"_mean"],
            marker="o",
            label=f
        )


    plt.xlabel(
        "Depth"
    )

    plt.ylabel(
        "Mean value"
    )

    plt.title(
        "MWUA Feature Evolution Across Search Depth"
    )

    plt.legend()

    plt.grid()


    plt.savefig(
        OUTPUT_DIR/"mwua_mean_depth.png",
        bbox_inches="tight"
    )

    plt.close()



def plot_decay(df):

    plt.figure(
        figsize=(9,5)
    )


    for f in MWUA_FEATURES:

        base = df[f+"_mean"].iloc[0]


        decay = (
            df[f+"_mean"]
            /
            base
        )


        plt.plot(
            df["depth"],
            decay,
            marker="o",
            label=f
        )


    plt.xlabel(
        "Depth"
    )

    plt.ylabel(
        "Relative value (depth / root)"
    )

    plt.title(
        "MWUA Feature Relative Decay Across Depth"
    )

    plt.legend()

    plt.grid()


    plt.savefig(
        OUTPUT_DIR/"mwua_relative_decay.png",
        bbox_inches="tight"
    )

    plt.close()



if __name__=="__main__":
    main()