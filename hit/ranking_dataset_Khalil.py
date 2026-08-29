import glob
import os
import pickle

import numpy as np


ALPHA = 0.20
MAX_PAIRS_PER_NODE = 200
RANDOM_SEED = 42
SCORE_RTOL = 1e-9
SCORE_ATOL = 1e-12


class RankingDatasetBuilder:

    def __init__(
        self,
        alpha=ALPHA,
        max_pairs_per_node=MAX_PAIRS_PER_NODE,
        random_seed=RANDOM_SEED,
    ):

        self.alpha = float(alpha)

        self.max_pairs_per_node = (
            None
            if max_pairs_per_node is None
            else int(max_pairs_per_node)
        )

        self.rng = np.random.default_rng(
            random_seed
        )

        self.X = []
        self.y = []

        self.nodes_seen = 0
        self.nodes_used = 0
        self.nodes_skipped_flat = 0
        self.nodes_skipped_no_split = 0
        self.pairs_before_cap = 0
        self.pairs_after_cap = 0

    @staticmethod
    def build_pair(
        sample,
        first_idx,
        second_idx,
    ):

        first = sample.candidate_features[
            first_idx
        ]

        second = sample.candidate_features[
            second_idx
        ]

        return np.concatenate(
            [
                first,
                second,
            ]
        )

    @staticmethod
    def _is_flat(
        scores,
    ):

        return np.allclose(
            scores,
            scores[0],
            rtol=SCORE_RTOL,
            atol=SCORE_ATOL,
        )

    def _good_bad_indices(
        self,
        scores,
    ):

        best_score = float(
            np.max(scores)
        )

        threshold = (
            1.0 - self.alpha
        ) * best_score

        good = np.flatnonzero(
            scores >= threshold
        )

        bad = np.flatnonzero(
            scores < threshold
        )

        return (
            good,
            bad,
            best_score,
            threshold,
        )

    def _node_pairs(
        self,
        good,
        bad,
    ):

        pairs = [
            (
                int(good_idx),
                int(bad_idx),
            )
            for good_idx in good
            for bad_idx in bad
        ]

        self.pairs_before_cap += len(
            pairs
        )

        if (
            self.max_pairs_per_node is not None
            and len(pairs)
            > self.max_pairs_per_node
        ):

            selected = self.rng.choice(
                len(pairs),
                size=self.max_pairs_per_node,
                replace=False,
            )

            pairs = [
                pairs[int(idx)]
                for idx in selected
            ]

        self.pairs_after_cap += len(
            pairs
        )

        return pairs

    def convert_folder(
        self,
        folder,
    ):

        files = sorted(
            glob.glob(
                os.path.join(
                    folder,
                    "*.pkl",
                )
            )
        )

        for filename in files:

            with open(
                filename,
                "rb",
            ) as file:

                while True:

                    try:

                        sample = pickle.load(
                            file
                        )

                    except EOFError:

                        break

                    self.nodes_seen += 1

                    scores = np.asarray(
                        sample.sb_scores,
                        dtype=float,
                    )

                    if len(scores) < 2:

                        self.nodes_skipped_no_split += 1

                        continue

                    if self._is_flat(
                        scores
                    ):

                        self.nodes_skipped_flat += 1

                        continue

                    (
                        good,
                        bad,
                        best_score,
                        threshold,
                    ) = self._good_bad_indices(
                        scores
                    )

                    if (
                        len(good) == 0
                        or len(bad) == 0
                    ):

                        self.nodes_skipped_no_split += 1

                        continue

                    pairs = self._node_pairs(
                        good,
                        bad,
                    )

                    if not pairs:

                        self.nodes_skipped_no_split += 1

                        continue

                    self.nodes_used += 1

                    for (
                        good_idx,
                        bad_idx,
                    ) in pairs:

                        self.X.append(
                            self.build_pair(
                                sample,
                                good_idx,
                                bad_idx,
                            )
                        )

                        self.y.append(1)

                        self.X.append(
                            self.build_pair(
                                sample,
                                bad_idx,
                                good_idx,
                            )
                        )

                        self.y.append(0)

        self.X = np.asarray(
            self.X,
            dtype=float,
        )

        self.y = np.asarray(
            self.y,
            dtype=np.int8,
        )

        return (
            self.X,
            self.y,
        )

    def print_summary(
        self,
    ):

        print()

        print(
            "=" * 70
        )

        print(
            "KHALIL-STYLE RANKING DATASET"
        )

        print(
            "=" * 70
        )

        print(
            f"Alpha                  : {self.alpha}"
        )

        print(
            f"Max pairs per node     : {self.max_pairs_per_node}"
        )

        print(
            f"Nodes seen             : {self.nodes_seen}"
        )

        print(
            f"Nodes used             : {self.nodes_used}"
        )

        print(
            f"Flat nodes skipped     : {self.nodes_skipped_flat}"
        )

        print(
            f"No-split nodes skipped : {self.nodes_skipped_no_split}"
        )

        print(
            f"Pairs before cap       : {self.pairs_before_cap}"
        )

        print(
            f"Pairs after cap        : {self.pairs_after_cap}"
        )

        print(
            f"Training rows          : {len(self.y)}"
        )

        if len(self.y) > 0:

            print(
                f"Positive rows          : {int(np.sum(self.y == 1))}"
            )

            print(
                f"Negative rows          : {int(np.sum(self.y == 0))}"
            )

            print(
                f"Feature dimension      : {self.X.shape[1]}"
            )

    def save(
        self,
        filename,
    ):

        filename = os.fspath(
            filename
        )

        output_dir = os.path.dirname(
            filename
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        with open(
            filename,
            "wb",
        ) as file:

            pickle.dump(
                (
                    self.X,
                    self.y,
                ),
                file,
            )


if __name__ == "__main__":

    INPUT_DIR = (
        "results/ltb_training"
    )

    OUTPUT_FILE = (
        "results/ltb_training/"
        "ranking_dataset.pkl"
    )

    builder = RankingDatasetBuilder(
        alpha=0.20,
        max_pairs_per_node=200,
        random_seed=42,
    )

    builder.convert_folder(
        INPUT_DIR
    )

    builder.print_summary()

    builder.save(
        OUTPUT_FILE
    )

    print()

    print(
        "Saved ranking dataset to:"
    )

    print(
        OUTPUT_FILE
    )
