"""M3 stage 1: teach Wav2Vec2 to recognise English letters (CTC) on Speechocean762. Resumable.
  python scripts/train_ctc.py                       # real run (about 15 epochs)
  python scripts/train_ctc.py --tiny --epochs 2     # smoke test
Rerun the same command after an interruption: it continues from checkpoints/m3_ctc/last.pt."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F, yaml
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, "src")
from data_speechocean import SpeechoceanDataset
from model import build_backbone
from train import get_device, load_checkpoint, save_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/speechocean762")
ap.add_argument("--split", default="splits/speechocean_split.json")
ap.add_argument("--out", default="checkpoints/m3_ctc")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--freeze_steps", type=int, default=300, help="only the output layer learns for the first steps")
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--device", default="auto")
args = ap.parse_args()

cfg = yaml.safe_load(open("configs/common.yaml"))
torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
device = get_device("cpu" if args.tiny else args.device)
VOCAB = ["<blank>", "|"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
CH = {c: i for i, c in enumerate(VOCAB)}


def encode(text):
    out = []
    for ch in " ".join(text.upper().split()):
        if ch == " ":
            out.append(CH["|"])
        elif ch in CH:
            out.append(CH[ch])
    return out


class CtcData(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        u = self.ds.utts[i]
        return self.ds[i]["audio"], torch.tensor(encode(self.ds.scores[u]["text"]))


def collate(batch):
    n = max(len(a) for a, _ in batch); s = max(len(t) for _, t in batch)
    audio = torch.zeros(len(batch), n); mask = torch.zeros(len(batch), n, dtype=torch.bool)
    tgt = torch.zeros(len(batch), s, dtype=torch.long); tl = torch.zeros(len(batch), dtype=torch.long)
    for i, (a, t) in enumerate(batch):
        audio[i, :len(a)] = a; mask[i, :len(a)] = True; tgt[i, :len(t)] = t; tl[i] = len(t)
    return audio, mask, tgt, tl


class CTCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = build_backbone(cfg["backbone"], args.tiny)
        self.use_attn = self.wav2vec2.config.feat_extract_norm == "layer"
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.wav2vec2.config.hidden_size, len(VOCAB))

    def forward(self, audio, mask):
        h = self.wav2vec2(audio, attention_mask=mask.long() if self.use_attn else None).last_hidden_state
        logits = self.head(self.drop(h))
        f_len = self.wav2vec2._get_feat_extract_output_lengths(mask.sum(1)).clamp(min=1, max=logits.shape[1])
        return logits, f_len


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def greedy(logits, f_len):
    texts = []
    for lg, n in zip(logits.argmax(-1).cpu(), f_len.cpu()):
        ids, last = [], -1
        for t in lg[:n].tolist():
            if t != last and t != 0:
                ids.append(t)
            last = t
        texts.append("".join(VOCAB[i] if i != 1 else " " for i in ids).strip())
    return texts


@torch.no_grad()
def val_cer(model, loader, val_ds):
    model.eval(); errs = chars = 0; k = 0; sample = None
    for audio, mask, tgt, tl in loader:
        logits, f_len = model(audio.to(device), mask.to(device))
        for hyp in greedy(logits, f_len):
            ref = " ".join(val_ds.scores[val_ds.utts[k]]["text"].upper().split())
            ref = "".join(c for c in ref if c in CH or c == " ")
            errs += edit_distance(hyp, ref); chars += len(ref); k += 1
            sample = sample or (ref, hyp)
    return errs / max(chars, 1), sample


split = json.load(open(args.split))
train_ds = SpeechoceanDataset(args.data, "train", speakers=split["train"])
val_ds = SpeechoceanDataset(args.data, "train", speakers=split["val"])
val_loader = DataLoader(CtcData(val_ds), batch_size=args.batch, collate_fn=collate)
model = CTCModel().to(device)
head_ids = {id(p) for p in model.head.parameters()}
back = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
opt = torch.optim.AdamW([{"params": list(model.head.parameters()), "lr": args.lr_head, "base_lr": args.lr_head},
                         {"params": back, "lr": args.lr, "base_lr": args.lr}], weight_decay=0.0)
steps_per_epoch = (len(train_ds) + args.batch - 1) // args.batch
total = args.epochs * steps_per_epoch
os.makedirs(args.out, exist_ok=True)
last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")
epoch0, step, best = 0, 0, float("inf")
if os.path.exists(last_path):
    meta = load_checkpoint(last_path, model, opt, map_location=device)
    epoch0, step, best = meta["epoch"], meta["step"], meta["best"] if meta["best"] is not None else float("inf")
    print(f"resuming after epoch {epoch0} (step {step}, best CER {best:.3f})")


def lr_factor(s):
    warm = max(1, int(0.1 * total))
    return (s + 1) / warm if s < warm else max(0.05, (total - s) / max(1, total - warm))


loader_all = CtcData(train_ds)
for epoch in range(epoch0, args.epochs):
    g = torch.Generator().manual_seed(cfg["seed"] + epoch)
    loader = DataLoader(loader_all, batch_size=args.batch, shuffle=True, generator=g, collate_fn=collate)
    model.train(); t0 = time.time(); run = 0.0
    for i, (audio, mask, tgt, tl) in enumerate(loader):
        for gr in opt.param_groups:
            gr["lr"] = gr["base_lr"] * lr_factor(step) * (0.0 if (gr["base_lr"] == args.lr and step < args.freeze_steps) else 1.0)
        audio, mask = audio.to(device), mask.to(device)
        logits, f_len = model(audio, mask)
        lp = logits.float().log_softmax(-1).transpose(0, 1)
        loss = F.ctc_loss(lp, tgt.to(device), f_len, tl.to(device), blank=0, reduction="mean", zero_infinity=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        step += 1; run += float(loss.detach())
        if (i + 1) % 50 == 0:
            print(f"  epoch {epoch+1} batch {i+1}/{steps_per_epoch} ctc loss {run/(i+1):.3f}", flush=True)
    cer, sample = val_cer(model, val_loader, val_ds)
    print(f"epoch {epoch+1}/{args.epochs} | train loss {run/steps_per_epoch:.3f} | val CER {cer:.3f} | {(time.time()-t0)/60:.1f} min")
    print(f"   ref: {sample[0]}\n   hyp: {sample[1]}")
    if cer < best:
        best = cer; save_checkpoint(best_path, model, None, epoch + 1, step, best)
    save_checkpoint(last_path, model, opt, epoch + 1, step, best)
    if device.type == "mps":
        torch.mps.empty_cache()
print(f"done. best val CER {best:.3f}. Backbone for the scoring stage: {best_path}")
