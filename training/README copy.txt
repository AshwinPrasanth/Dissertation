Corrected LTB node-ranking data collector.

Feature architecture:
6 root-static global features:
pagerank, mwua_xavg, mwua_weight_min, mwua_weight_max,
mwua_weight_avg, luby_frequency

9 dynamic local features:
degree_rank, nbr_min_rank, nbr_max_rank, nbr_avg_rank,
core_number, clustering, degree_centrality, lp_value, lp_certainty

The seven structural features are recomputed once per selected SCIP node
on the residual graph induced by locally unfixed variables.

Strong branching follows the PySCIPOpt tutorial pattern:
parent_lp_obj = getLPObjVal()
down_gain = max(down - parent_lp_obj, 0) when valid and feasible
up_gain = max(up - parent_lp_obj, 0) when valid and feasible
score = getBranchScoreMultiple(var, [down_gain, up_gain])

One BranchSample is written per SB-labelled B&B node.
Each sample stores the full candidate feature matrix and SB ranking targets.

Pilot:
GRAPH_SIZES = [50]
TARGET_DEGREES = [10]
SEEDS = [42]
MAX_SB_NODES = 10
CANDIDATE_LIMIT = None
STRONGBRANCH_ITLIM = 100

Run:
python generate_ltb_training_data.py

Inspect:
python inspect_dataset.py results/ltb_training/erdos_renyi_n50_d10_s42.pkl
