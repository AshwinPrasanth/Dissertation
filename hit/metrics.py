import numpy as np
from scipy.stats import spearmanr, kendalltau

def compute_metrics(pred_scores, target_scores, chosen_index):
    pred_scores=np.asarray(pred_scores)
    target_scores=np.asarray(target_scores)
    ranking=np.argsort(pred_scores)[::-1]
    top1=int(ranking[0]==chosen_index)
    top3=int(chosen_index in ranking[:3])
    top5=int(chosen_index in ranking[:5])
    rank=int(np.where(ranking==chosen_index)[0][0])+1
    rho=spearmanr(pred_scores,target_scores).correlation
    tau=kendalltau(pred_scores,target_scores).correlation
    if np.isnan(rho): rho=0.0
    if np.isnan(tau): tau=0.0
    return {
        'top1':top1,
        'top3':top3,
        'top5':top5,
        'rank':rank,
        'mrr':1.0/rank,
        'spearman':float(rho),
        'kendall':float(tau)
    }
