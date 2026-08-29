import argparse
import torch
from torch.utils.data import DataLoader

from dataset_ltb import load_dataset
from model import BranchingMLP
from metrics import compute_metrics

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="../results/dimacs_ltb_training")
    ap.add_argument("--checkpoint",default="checkpoints/best_model.pt")
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck=torch.load(
        args.checkpoint,
        map_location=device
    )

    val_graphs = ck["val_graphs"]

    print("="*70)
    print("VALIDATION GRAPHS")
    print("="*70)

    for g in val_graphs:
        print(g)

    ds=load_dataset(
        args.dataset,
        graph_names=val_graphs,
    )

    loader=DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x:x
    )

    model=BranchingMLP().to(device)

    model.load_state_dict(
        ck["model"]
    )

    model.eval()

    totals={"top1":0,"top3":0,"top5":0,"mrr":0.0,"rank":0.0,"spearman":0.0,"kendall":0.0}
    n=0

    for batch in loader:
        s=batch[0]
        pred=model(s["features"].to(device)).cpu().numpy()
        m=compute_metrics(pred,s["scores"].numpy(),int(s["chosen"]))
        for k in totals:
            totals[k]+=m[k]
        n+=1

    print("="*60)
    print("EVALUATION")
    print("="*60)
    for k,v in totals.items():
        if k.startswith("top"):
            print(f"{k}: {100*v/n:.2f}%")
        else:
            print(f"{k}: {v/n:.4f}")

if __name__=="__main__":
    main()
