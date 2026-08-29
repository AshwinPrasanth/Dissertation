import pickle
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_ROOT
    / "results"
    / "ltb_training"
    / "ranking_dataset.pkl"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "results"
    / "ltb_training"
    / "ranking_model.pkl"
)

TEST_SIZE = 0.20
RANDOM_SEED = 42


def main():

    with open(
        DATASET_FILE,
        "rb",
    ) as file:

        X, y = pickle.load(
            file
        )

    print(
        "=" * 70
    )

    print(
        "LTB RANKING MODEL SANITY CHECK"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset shape      : {X.shape}"
    )

    print(
        f"Positive samples   : {np.sum(y == 1)}"
    )

    print(
        f"Negative samples   : {np.sum(y == 0)}"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    train_predictions = model.predict(
        X_train_scaled
    )

    test_predictions = model.predict(
        X_test_scaled
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    print()

    print(
        f"Train rows         : {len(y_train)}"
    )

    print(
        f"Test rows          : {len(y_test)}"
    )

    print(
        f"Train accuracy     : {train_accuracy:.6f}"
    )

    print(
        f"Test accuracy      : {test_accuracy:.6f}"
    )

    print()

    print(
        classification_report(
            y_test,
            test_predictions,
            digits=6,
        )
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MODEL_FILE,
        "wb",
    ) as file:

        pickle.dump(
            {
                "model":
                    model,

                "scaler":
                    scaler,

                "input_dimension":
                    X.shape[1],

                "representation":
                    "concat",

                "test_size":
                    TEST_SIZE,

                "random_seed":
                    RANDOM_SEED,
            },
            file,
        )

    print(
        "Saved model to:"
    )

    print(
        MODEL_FILE
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is only a row-wise pipeline sanity check."
    )

    print(
        "It is not a graph-wise generalisation result."
    )


if __name__ == "__main__":
    main()
