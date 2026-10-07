import sys
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from data_give import GiveDataset, collate

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/GiVe"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "calibrated"
tot = dict(utts=0, words=0, stress_valid=0, phones=0, phones_in_loss=0)
for split in ["train", "validation", "test"]:
    ds = GiveDataset(ROOT, VARIANT, split)
    n_utt = n_w = n_sv = n_p = n_pl = 0
    for batch in DataLoader(ds, batch_size=8, collate_fn=collate):      # loads EVERY file in the split
        n_utt += len(batch["utt"]); n_w += int(batch["w_mask"].sum()); n_sv += int(batch["w_stress_mask"].sum())
        n_p += int(batch["p_mask"].sum()); n_pl += int(batch["p_loss_mask"].sum())
    print(f"{split:10s} utts {n_utt:4d} | words {n_w:5d} (valid stress {n_sv}) | phones {n_p:5d} (in phone loss {n_pl}, {n_pl/n_p:.0%})")
b = next(iter(DataLoader(GiveDataset(ROOT, VARIANT, "train"), batch_size=4, collate_fn=collate)))
for k, v in b.items():
    print(f"{k:13s}", tuple(v.shape) if hasattr(v, "shape") else v)
print("sentence targets of first item (acc, flu, pros, compl):", [round(x, 3) for x in b["sent"][0].tolist()])
print("ranges  sent:", float(b["sent"].min()), float(b["sent"].max()),
      "| word acc:", float(b["w_acc"][b["w_mask"]].min()), float(b["w_acc"][b["w_mask"]].max()),
      "| phone:", float(b["p_score"][b["p_mask"]].min()), float(b["p_score"][b["p_mask"]].max()))
