import pickle
from pathlib import Path

import numpy as np


INPUT_DIR = Path(
    "../results/dimacs_ltb_training500"
)

OUTPUT_DIR = Path(
    "../results/ranking_dataset_xgb"
)


def convert_scores_to_labels(scores):

    if np.max(scores) == np.min(scores):

        return np.zeros(
            len(scores),
            dtype=np.int32
        )

    ranks = (
        np.argsort(
            np.argsort(
                scores
            )
        )
    )

    percentile = (
        ranks
        /
        (len(scores)-1)
    )

    labels = np.floor(
        percentile * 4
    ).astype(
        np.int32
    )

    return labels



def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X = []
    y = []
    groups = []
    graph_ids = []
    depths = []
    candidate_depths = []

    feature_names = None

    nodes = 0
    candidates = 0

    files = sorted(
        INPUT_DIR.glob("*.pkl")
    )

    print(
        "Graphs:",
        len(files)
    )


    for file in files:

        print(
            "Processing:",
            file.name
        )

        graph_name = file.stem


        with open(
            file,
            "rb"
        ) as f:


            while True:

                try:

                    sample = pickle.load(
                        f
                    )

                except EOFError:

                    break


                scores = np.asarray(
                    sample.sb_scores,
                    dtype=np.float32
                )


                features = np.asarray(
                    sample.candidate_features,
                    dtype=np.float32
                )


                if len(scores) < 2:

                    continue


                if np.allclose(
                    scores,
                    scores[0]
                ):

                    continue


                labels = convert_scores_to_labels(
                    scores
                )


                X.append(
                    features
                )

                y.append(
                    labels
                )
                
                candidate_depths.append(
                    np.full(
                        len(scores),
                        sample.depth,
                        dtype=np.int32
                    )
                )


                groups.append(
                    len(scores)
                )


                graph_ids.append(
                    graph_name
                )


                depths.append(
                    sample.depth
                )


                nodes += 1

                candidates += len(scores)


                if feature_names is None:

                    feature_names = (
                        sample.feature_names
                    )


    X = np.vstack(
        X
    )


    y = np.concatenate(
        y
    )


    groups = np.asarray(
        groups,
        dtype=np.int32
    )


    depths = np.asarray(
        depths,
        dtype=np.int32
    )
    
    candidate_depths = np.concatenate(
        candidate_depths
    )


    graph_ids = np.asarray(
        graph_ids
    )


    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )


    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )
    
    np.save(
        OUTPUT_DIR / "candidate_depths.npy",
        candidate_depths
    )


    np.save(
        OUTPUT_DIR / "groups.npy",
        groups
    )


    np.save(
        OUTPUT_DIR / "feature_names.npy",
        feature_names
    )


    np.save(
        OUTPUT_DIR / "graph_ids.npy",
        graph_ids
    )


    np.save(
        OUTPUT_DIR / "depths.npy",
        depths
    )


    print()

    print(
        "Depth statistics"
    )


    print(
        "Depth range:",
        depths.min(),
        depths.max()
    )


    print(
        "Average depth:",
        depths.mean()
    )


    unique, counts = np.unique(
        depths,
        return_counts=True
    )


    print(
        "Depth distribution:"
    )


    print(
        dict(
            zip(
                unique,
                counts
            )
        )
    )


    print()

    print(
        "Label statistics"
    )


    unique, counts = np.unique(
        y,
        return_counts=True
    )


    print(
        dict(
            zip(
                unique,
                counts
            )
        )
    )


    print()

    print(
        "="*70
    )

    print(
        "XGBOOST RANKING DATASET"
    )

    print(
        "="*70
    )


    print(
        "Ranking nodes:",
        nodes
    )


    print(
        "Candidates:",
        candidates
    )


    print(
        "X shape:",
        X.shape
    )


    print(
        "y shape:",
        y.shape
    )


    print(
        "groups:",
        groups.shape
    )


    print(
        "Feature dimension:",
        X.shape[1]
    )


if __name__ == "__main__":

    main()