import pickle
import numpy as np

samples = []

with open("dataset/frb30-15-1.pkl", "rb") as f:

    while True:

        try:
            samples.append(
                pickle.load(f)
            )

        except EOFError:
            break

print("Samples:", len(samples))

s = samples[0]

print("Graph:", s.graph_name)
print("Depth:", s.depth)
print("Candidates:", len(s.candidate_ids))
print("Feature matrix:", s.candidate_features.shape)
print("MWUA:", s.mwua_scores.shape)
print("Chosen:", s.chosen_variable)

depths = [s.depth for s in samples]

print("Min depth:", min(depths))
print("Max depth:", max(depths))
print("Unique depths:", sorted(set(depths))[:20])

sizes = [len(s.candidate_ids) for s in samples]

print("Minimum candidates :", min(sizes))
print("Maximum candidates :", max(sizes))
print("Average candidates :", np.mean(sizes))