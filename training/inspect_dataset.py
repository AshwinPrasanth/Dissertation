import pickle
import sys

import numpy as np


filename = sys.argv[1]

samples = []

with open(
    filename,
    "rb",
) as file:

    while True:

        try:
            samples.append(
                pickle.load(file)
            )

        except EOFError:
            break


print(
    "Samples:",
    len(samples),
)

for sample in samples:

    print()

    print(
        "=" * 70
    )

    print(
        "Graph:",
        sample.graph_name,
    )

    print(
        "Node:",
        sample.node_number,
    )

    print(
        "Depth:",
        sample.depth,
    )

    print(
        "Residual:",
        sample.residual_n,
        sample.residual_m,
    )

    print(
        "Parent LP:",
        sample.parent_lp_obj,
    )

    print(
        "Candidates:",
        len(sample.candidate_ids),
    )

    print(
        "Feature matrix:",
        sample.candidate_features.shape,
    )

    print(
        "Unique SB scores:",
        len(
            np.unique(
                sample.sb_scores
            )
        ),
    )

    print(
        "SB score min:",
        np.min(
            sample.sb_scores
        ),
    )

    print(
        "SB score max:",
        np.max(
            sample.sb_scores
        ),
    )

    winner = int(
        np.argmax(
            sample.sb_scores
        )
    )

    print(
        "Chosen:",
        sample.chosen_variable,
    )

    print(
        "Argmax:",
        sample.candidate_ids[winner],
    )

    print()

    print(
        "Top 10 candidates"
    )

    order = np.argsort(
        -sample.sb_scores
    )

    for idx in order[:10]:

        print(
            f"v={sample.candidate_ids[idx]:>5} "
            f"down={sample.sb_down_bounds[idx]:>12.6f} "
            f"up={sample.sb_up_bounds[idx]:>12.6f} "
            f"dgain={sample.sb_down_gains[idx]:>10.6f} "
            f"ugain={sample.sb_up_gains[idx]:>10.6f} "
            f"score={sample.sb_scores[idx]:>12.8f}"
        )
