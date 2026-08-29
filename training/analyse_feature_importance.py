from xgboost import XGBRanker
import numpy as np
from pathlib import Path


MODEL = Path(
    "../checkpoints/xgb_ranker.json"
)

DATA = Path(
    "../results/ranking_dataset_xgb"
)


model = XGBRanker()

model.load_model(
    MODEL
)


features = np.load(
    DATA/"feature_names.npy",
    allow_pickle=True
)


importance = model.get_booster().get_score(
    importance_type="gain"
)


results = []

for i, name in enumerate(features):

    key = f"f{i}"

    results.append(
        (
            name,
            importance.get(key,0)
        )
    )


results.sort(
    key=lambda x:x[1],
    reverse=True
)


print("="*60)
print("FEATURE IMPORTANCE (GAIN)")
print("="*60)


for name,value in results:
    print(
        f"{name:<30} {value:.4f}"
    )