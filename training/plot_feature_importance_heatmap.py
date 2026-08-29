from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = Path(
    "../results/feature_importance_depth/feature_importance_depth.csv"
)

OUTPUT_DIR = Path(
    "../results/feature_importance_depth"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    features = [
        col
        for col in df.columns
        if col not in [
            "depth",
            "samples",
            "groups"
        ]
    ]


    importance = df[
        features
    ].copy()


    # Normalize each depth row
    importance = (
        importance
        .div(
            importance.sum(axis=1),
            axis=0
        )
        * 100
    )


    importance.index = df["depth"]


    print()
    print("="*70)
    print("NORMALIZED FEATURE IMPORTANCE (%)")
    print("="*70)

    print(
        importance
    )


    importance.to_csv(
        OUTPUT_DIR /
        "feature_importance_depth_normalized.csv"
    )


    plt.figure(
        figsize=(12,7)
    )


    plt.imshow(
        importance.T,
        aspect="auto"
    )


    plt.xticks(
        range(len(importance.index)),
        importance.index
    )


    plt.yticks(
        range(len(features)),
        features
    )


    plt.xlabel(
        "Search Depth"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        "Feature Importance Evolution Across Branch-and-Bound Depth"
    )


    plt.colorbar(
        label="Normalized Gain (%)"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR /
        "feature_importance_depth_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



if __name__ == "__main__":

    main()