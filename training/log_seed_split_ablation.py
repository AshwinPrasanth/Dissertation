import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "seed_split_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEPTHS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5+",
]

RESULTS = {
    "static_only": {
        "feature_count": 6,
        "train_accuracy": 0.688471,
        "validation_accuracy": 0.677967,
        "test_accuracy": 0.650282,
        "validation_depth": [
            0.998110,
            0.814624,
            0.736620,
            0.706944,
            0.618872,
            0.558731,
        ],
        "test_depth": [
            0.996988,
            0.804082,
            0.717409,
            0.655680,
            0.570526,
            0.505372,
        ],
    },
    "dynamic_only": {
        "feature_count": 9,
        "train_accuracy": 0.840662,
        "validation_accuracy": 0.807758,
        "test_accuracy": 0.844631,
        "validation_depth": [
            0.992756,
            0.990998,
            0.943281,
            0.887902,
            0.782276,
            0.631809,
        ],
        "test_depth": [
            0.989960,
            0.991981,
            0.935904,
            0.871564,
            0.779549,
            0.713891,
        ],
    },
    "hybrid": {
        "feature_count": 15,
        "train_accuracy": 0.838119,
        "validation_accuracy": 0.803767,
        "test_accuracy": 0.837818,
        "validation_depth": [
            0.994331,
            0.986037,
            0.937464,
            0.883244,
            0.779184,
            0.627736,
        ],
        "test_depth": [
            0.993976,
            0.987243,
            0.928843,
            0.857456,
            0.765865,
            0.711972,
        ],
    },
}

DATASET_AUDIT = {
    "train": {
        "graphs": 68,
        "nodes_seen": 1270,
        "nodes_used": 1079,
        "flat_nodes_skipped": 181,
        "no_split_nodes_skipped": 10,
        "pairs_before_cap": 332515,
        "pairs_after_cap": 142314,
        "ranking_rows": 284628,
    },
    "validation": {
        "graphs": 24,
        "nodes_seen": 420,
        "nodes_used": 355,
        "flat_nodes_skipped": 64,
        "no_split_nodes_skipped": 1,
        "pairs_before_cap": 99306,
        "pairs_after_cap": 45604,
        "ranking_rows": 91208,
    },
    "test": {
        "graphs": 23,
        "nodes_seen": 426,
        "nodes_used": 357,
        "flat_nodes_skipped": 63,
        "no_split_nodes_skipped": 6,
        "pairs_before_cap": 106442,
        "pairs_after_cap": 43741,
        "ranking_rows": 87482,
    },
}


def save_json():
    output = {
        "experiment": "strict_seed_split_ablation",
        "alpha": 0.2,
        "max_pairs_per_node": 200,
        "train_seeds": [42, 43, 44],
        "validation_seeds": [45],
        "test_seeds": [46],
        "dataset_audit": DATASET_AUDIT,
        "results": RESULTS,
    }

    filename = (
        OUTPUT_DIR
        / "seed_split_ablation_results.json"
    )

    with open(
        filename,
        "w",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )


def save_overall_csv():
    filename = (
        OUTPUT_DIR
        / "overall_accuracy.csv"
    )

    with open(
        filename,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "model",
                "feature_count",
                "train_accuracy",
                "validation_accuracy",
                "test_accuracy",
            ]
        )

        for model, result in RESULTS.items():
            writer.writerow(
                [
                    model,
                    result["feature_count"],
                    result["train_accuracy"],
                    result["validation_accuracy"],
                    result["test_accuracy"],
                ]
            )


def save_depth_csv():
    filename = (
        OUTPUT_DIR
        / "depth_accuracy.csv"
    )

    with open(
        filename,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "model",
                "split",
                "depth",
                "accuracy",
            ]
        )

        for model, result in RESULTS.items():
            for split in [
                "validation",
                "test",
            ]:
                values = result[
                    f"{split}_depth"
                ]

                for depth, accuracy in zip(
                    DEPTHS,
                    values,
                ):
                    writer.writerow(
                        [
                            model,
                            split,
                            depth,
                            accuracy,
                        ]
                    )


def plot_test_depth():
    plt.figure(
        figsize=(8, 5)
    )

    for model, result in RESULTS.items():
        label = model.replace(
            "_",
            " ",
        ).title()

        plt.plot(
            DEPTHS,
            [
                value * 100
                for value in result["test_depth"]
            ],
            marker="o",
            label=label,
        )

    plt.xlabel(
        "Branch-and-bound depth"
    )

    plt.ylabel(
        "Pairwise ranking accuracy (%)"
    )

    plt.title(
        "Test Accuracy by Search Depth"
    )

    plt.ylim(
        45,
        102,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "test_accuracy_by_depth.png",
        dpi=300,
    )

    plt.close()


def plot_validation_depth():
    plt.figure(
        figsize=(8, 5)
    )

    for model, result in RESULTS.items():
        label = model.replace(
            "_",
            " ",
        ).title()

        plt.plot(
            DEPTHS,
            [
                value * 100
                for value in result[
                    "validation_depth"
                ]
            ],
            marker="o",
            label=label,
        )

    plt.xlabel(
        "Branch-and-bound depth"
    )

    plt.ylabel(
        "Pairwise ranking accuracy (%)"
    )

    plt.title(
        "Validation Accuracy by Search Depth"
    )

    plt.ylim(
        45,
        102,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "validation_accuracy_by_depth.png",
        dpi=300,
    )

    plt.close()


def plot_overall_test():
    models = [
        "Static only",
        "Dynamic only",
        "Hybrid",
    ]

    values = [
        RESULTS["static_only"][
            "test_accuracy"
        ] * 100,
        RESULTS["dynamic_only"][
            "test_accuracy"
        ] * 100,
        RESULTS["hybrid"][
            "test_accuracy"
        ] * 100,
    ]

    plt.figure(
        figsize=(7, 5)
    )

    bars = plt.bar(
        models,
        values,
    )

    plt.ylabel(
        "Test accuracy (%)"
    )

    plt.title(
        "Overall Test Ranking Accuracy"
    )

    plt.ylim(
        0,
        100,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "overall_test_accuracy.png",
        dpi=300,
    )

    plt.close()


def plot_static_decay():
    validation = [
        value * 100
        for value in RESULTS[
            "static_only"
        ]["validation_depth"]
    ]

    test = [
        value * 100
        for value in RESULTS[
            "static_only"
        ]["test_depth"]
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        DEPTHS,
        validation,
        marker="o",
        label="Validation",
    )

    plt.plot(
        DEPTHS,
        test,
        marker="o",
        label="Test",
    )

    plt.xlabel(
        "Branch-and-bound depth"
    )

    plt.ylabel(
        "Static-only ranking accuracy (%)"
    )

    plt.title(
        "Decay of Frozen Static Feature Predictiveness"
    )

    plt.ylim(
        45,
        102,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "static_feature_decay.png",
        dpi=300,
    )

    plt.close()


def main():
    save_json()
    save_overall_csv()
    save_depth_csv()

    plot_test_depth()
    plot_validation_depth()
    plot_overall_test()
    plot_static_decay()

    print(
        "=" * 70
    )

    print(
        "RESULT LOGGING AND PLOTTING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()

    for filename in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            filename.name
        )


if __name__ == "__main__":
    main()
