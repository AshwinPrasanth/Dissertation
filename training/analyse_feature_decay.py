from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(
    "../results/ranking_dataset_xgb"
)

OUTPUT_DIR = Path(
    "../results/feature_decay"
)


DEPTH_BUCKETS = {
    "0-5": (0,5),
    "5-10": (6,10),
    "10-20": (11,20),
    "20+": (21,999),
}



def get_bucket(depth):

    for name,(low,high) in DEPTH_BUCKETS.items():

        if low <= depth <= high:
            return name

    return None



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


    feature_names = np.load(
        DATA_DIR/"feature_names.npy",
        allow_pickle=True
    )


    results=[]


    for bucket,(low,high) in DEPTH_BUCKETS.items():

        mask = (
            (depths>=low)
            &
            (depths<=high)
        )


        X_depth = X[mask]


        row={
            "depth":bucket,
            "samples":len(X_depth)
        }


        for i,name in enumerate(feature_names):

            row[f"{name}_mean"] = np.mean(
                X_depth[:,i]
            )

            row[f"{name}_std"] = np.std(
                X_depth[:,i]
            )


        results.append(row)



    df=pd.DataFrame(
        results
    )


    df.to_csv(
        OUTPUT_DIR/"feature_depth_statistics.csv",
        index=False
    )


    print("="*70)
    print("FEATURE STATISTICS BY DEPTH")
    print("="*70)

    print(df)



    plot_means(
        df,
        feature_names
    )

    plot_std(
        df,
        feature_names
    )



def plot_means(df,features):

    plt.figure(
        figsize=(12,6)
    )


    for f in features:

        plt.plot(
            df["depth"],
            df[f"{f}_mean"],
            marker="o",
            label=f
        )


    plt.xlabel(
        "Depth"
    )

    plt.ylabel(
        "Mean Feature Value"
    )

    plt.title(
        "Feature Mean Evolution Across Search Depth"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid()


    plt.savefig(
        OUTPUT_DIR/"feature_mean_decay.png",
        bbox_inches="tight"
    )

    plt.close()



def plot_std(df,features):

    plt.figure(
        figsize=(12,6)
    )


    for f in features:

        plt.plot(
            df["depth"],
            df[f"{f}_std"],
            marker="o",
            label=f
        )


    plt.xlabel(
        "Depth"
    )

    plt.ylabel(
        "Feature Standard Deviation"
    )

    plt.title(
        "Feature Variance Evolution Across Search Depth"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid()


    plt.savefig(
        OUTPUT_DIR/"feature_variance_decay.png",
        bbox_inches="tight"
    )

    plt.close()



if __name__=="__main__":
    main()