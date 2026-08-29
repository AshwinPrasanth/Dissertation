from pathlib import Path

import json
import numpy as np
from xgboost import XGBRanker


BASE_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/ranking"
).expanduser()

MODEL_DIR = Path(
    "~/dissertation_ashwin/ml_training_data/models/ablation"
).expanduser()


FEATURE_NAMES = [
    "degree",
    "weighted_degree",
    "hyperedge_participation",
    "mwu_weighted_degree",
    "mean_incident_mwu",
    "max_incident_mwu",
    "mean_incident_edge_size",
    "max_incident_edge_size"
]


def main():

    model = XGBRanker()

    model.load_model(
        MODEL_DIR / "xgb_structural_mwu.json"
    )

    booster = model.get_booster()

    gain = booster.get_score(
        importance_type="gain"
    )

    weight = booster.get_score(
        importance_type="weight"
    )

    cover = booster.get_score(
        importance_type="cover"
    )

    results = []

    for index, name in enumerate(FEATURE_NAMES):

        key = f"f{index}"

        results.append({
            "feature": name,
            "gain": float(gain.get(key, 0.0)),
            "weight": float(weight.get(key, 0.0)),
            "cover": float(cover.get(key, 0.0))
        })

    total_gain = sum(
        x["gain"] for x in results
    )

    for x in results:

        if total_gain > 0:
            x["gain_percent"] = (
                100.0 * x["gain"] / total_gain
            )
        else:
            x["gain_percent"] = 0.0

    results.sort(
        key=lambda x: x["gain"],
        reverse=True
    )

    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE — STRUCTURAL + MWU")
    print("=" * 70)

    print()

    for rank, x in enumerate(results, 1):

        print(
            f"{rank:2d}. "
            f"{x['feature']:30s} "
            f"Gain={x['gain_percent']:8.3f}% "
            f"Weight={x['weight']:8.0f} "
            f"Cover={x['cover']:10.3f}"
        )

    output = (
        MODEL_DIR /
        "structural_mwu_feature_importance.json"
    )

    with open(
        output,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(
        "Saved to:",
        output
    )


if __name__ == "__main__":
    main()