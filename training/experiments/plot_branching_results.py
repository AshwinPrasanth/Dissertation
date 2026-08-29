import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv(

    "results/branching_comparison.csv"

)

plt.figure(figsize=(8,5))

plt.bar(

    df["Strategy"],

    df["Runtime"],

)

plt.ylabel("Runtime (seconds)")

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig(

    "results/runtime_comparison.png",

    dpi=300,

)

plt.close()

plt.figure(figsize=(8,5))

plt.bar(

    df["Strategy"],

    df["Nodes"],

)

plt.ylabel("Nodes Explored")

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig(

    "results/nodes_comparison.png",

    dpi=300,

)

plt.close()

plt.figure(figsize=(8,5))

plt.bar(

    df["Strategy"],

    df["SearchReduction"],

)

plt.ylabel("Search Reduction (%)")

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig(

    "results/search_reduction.png",

    dpi=300,

)

plt.close()

print("Plots saved.")