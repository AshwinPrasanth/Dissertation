import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset_ltb import load_dataset
from graph_split import graph_train_val_split
from model import BranchingMLP, BranchingLoss
from metrics import compute_metrics

def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    loss_sum = 0.0
    metric_sum = {"top1":0,"top3":0,"top5":0,"mrr":0.0,"rank":0.0}
    total = 0

    for batch in loader:
        s = batch[0]
        x = s["features"].to(device)
        chosen = s["chosen"].to(device)
        sb = s["scores"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = criterion(pred, chosen, sb)
            if train:
                loss.backward()
                optimizer.step()

        m = compute_metrics(pred.detach().cpu().numpy(),
                            sb.cpu().numpy(),
                            int(chosen.cpu()))
        for k in metric_sum:
            metric_sum[k] += m[k]
        loss_sum += loss.item()
        total += 1

    out = {k:v/total for k,v in metric_sum.items()}
    out["loss"] = loss_sum/total
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="../results/dimacs_ltb_training")
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--loss",default="cross_entropy",choices=["cross_entropy","mse","pairwise"])
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset=load_dataset(args.dataset)
    train_set,val_set,train_graphs,val_graphs=graph_train_val_split(dataset,0.8,args.seed)
    overlap = set(train_graphs) & set(val_graphs)

    print("=" * 60)
    print("Overlap:", overlap)
    print("Train graphs:", len(train_graphs))
    print("Validation graphs:", len(val_graphs))
    assert len(overlap) == 0, f"Leakage detected: {overlap}"
    print("=" * 60)

    print("Train graphs:",train_graphs)
    print("Validation graphs:",val_graphs)

    train_loader=DataLoader(train_set,batch_size=1,shuffle=True,collate_fn=lambda x:x)
    val_loader=DataLoader(val_set,batch_size=1,shuffle=False,collate_fn=lambda x:x)

    model=BranchingMLP().to(device)
    crit=BranchingLoss(mode=args.loss)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)

    Path("checkpoints").mkdir(exist_ok=True)
    best_top1=-1
    best_loss=1e9

    for e in range(1,args.epochs+1):
        tr=run_epoch(model,train_loader,crit,device,opt)
        va=run_epoch(model,val_loader,crit,device)

        print(f'Epoch {e:03d} | TL {tr["loss"]:.4f} VL {va["loss"]:.4f} | '
              f'Top1 {va["top1"]*100:.2f}% Top3 {va["top3"]*100:.2f}% '
              f'Top5 {va["top5"]*100:.2f}% MRR {va["mrr"]:.3f}')

        if va["top1"]>best_top1 or (va["top1"]==best_top1 and va["loss"]<best_loss):
            best_top1=va["top1"]; best_loss=va["loss"]
            torch.save({
                "model":model.state_dict(),
                "epoch":e,
                "top1":best_top1,
                "loss":best_loss,
                "train_graphs":train_graphs,
                "val_graphs":val_graphs,
            },"checkpoints/best_model.pt")

if __name__=="__main__":
    main()
