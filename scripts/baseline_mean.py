"""M0: always predict the training-set average. No learning. The floor every real model must beat.
  python scripts/baseline_mean.py --corpus give --split test
Correlations are undefined for a constant prediction (shown as nan); compare the MSE columns."""
import argparse, json, os, sys
import torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from data_common import collate
from data_give import GiveDataset
from evaluate import evaluate, short

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/GiVe")
ap.add_argument("--variant", default="calibrated")
ap.add_argument("--split", default="test")
args = ap.parse_args()
cfg = yaml.safe_load(open("configs/common.yaml"))

train = DataLoader(GiveDataset(args.data, args.variant, "train", cfg["flag_below"]), batch_size=16, collate_fn=collate)
sent, w, p = [], [], []
for b in train:
    sent.append(b["sent"]); w.append(b["w_acc"][b["w_mask"]]); p.append(b["p_score"][b["p_loss_mask"]])
sent_mean, w_mean, p_mean = torch.cat(sent).mean(0), torch.cat(w).mean(), torch.cat(p).mean()
print("training means  sentence:", [round(float(x) * 10, 2) for x in sent_mean], "| word acc:", round(float(w_mean) * 10, 2),
      "| phone:", round(float(p_mean) * 2, 3))


class MeanModel(nn.Module):
    def score_batch(self, b):
        B, W, P = b["w_acc"].shape[0], b["w_acc"].shape[1], b["p_id"].shape[1]
        return dict(sent=sent_mean.expand(B, 4), w_acc=w_mean.expand(B, W), w_stress=torch.full((B, W), 10.0),
                    p_score=p_mean.expand(B, P))        # stress: always predict "correct" (the majority class)


ds = GiveDataset(args.data, args.variant, args.split, cfg["flag_below"])
res = evaluate(MeanModel(), DataLoader(ds, batch_size=16, collate_fn=collate), torch.device("cpu"))
print(f"M0 mean baseline on give/{args.split} ({res['n_utts']} utts)")
print(short(res))
print({k: round(v, 3) for k, v in res.items() if k.startswith("mse") or k == "stress_acc"})
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/m0_give_{args.split}.json", "w"), indent=2)
print(f"saved results/m0_give_{args.split}.json")