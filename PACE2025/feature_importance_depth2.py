from pathlib import Path

import json

from xgboost import XGBRanker


MODEL_PATH = Path(
    "~/dissertation_ashwin/ml_training_data/models/depth/xgb_depth_2.json"
).expanduser()

OUTPUT_PATH = Path(
    "~/dissertation_ashwin/ml_training_data/models/depth/depth2_feature_importance.json"
).expanduser()


FEATURE_NAMES = [
    "degree",
    "weighted_degree",
    "hyperedge_participation",
    "mwu_weighted_degree",
    "mean_incident_mwu",
    "max_incident_mwu",
    "mean_incident_edge_size",
    "max_incident_edge_size",
    "singleton_participation",
    "binary_participation",
    "x_avg",
    "certainty",
    "decision_level",
    "trail_size",
    "propagated",
    "conflicts",
    "evsids",
    "assigned",
    "assignment_level"
]


model = XGBRanker()

model.load_model(
    MODEL_PATH
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


for i, name in enumerate(FEATURE_NAMES):

    key = f"f{i}"

    results.append({
        "feature": name,
        "gain": float(gain.get(key, 0.0)),
        "weight": int(weight.get(key, 0)),
        "cover": float(cover.get(key, 0.0))
    })


total_gain = sum(
    item["gain"]
    for item in results
)


for item in results:

    if total_gain > 0:
        item["gain_percent"] = (
            100.0 *
            item["gain"] /
            total_gain
        )
    else:
        item["gain_percent"] = 0.0


results.sort(
    key=lambda x: x["gain_percent"],
    reverse=True
)


print()
print("=" * 70)
print("FEATURE IMPORTANCE — FINAL DEPTH-2 MODEL")
print("=" * 70)

for rank, item in enumerate(results, 1):

    print(
        f"{rank:2d}. "
        f"{item['feature']:<30} "
        f"Gain={item['gain_percent']:8.3f}% "
        f"Weight={item['weight']:8d} "
        f"Cover={item['cover']:12.3f}"
    )


with open(
    OUTPUT_PATH,
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
    OUTPUT_PATH
)
