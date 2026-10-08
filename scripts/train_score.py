"""Train the scoring model. Resumable. Used for M3 (Speechocean), M2 (Indonesian from scratch) and M4 (Indonesian from M3).
  M3: python scripts/train_score.py --corpus speechocean --init checkpoints/m3_ctc/best.pt --init_kind ctc --out checkpoints/m3_score
  M2: python scripts/train_score.py --corpus give --out checkpoints/m2
  M4: python scripts/train_score.py --corpus give --init checkpoints/m3_score/best.pt --init_kind scorer --out checkpoints/m4
Rerun the same command after an interruption: it continues from <out>/last.pt."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, math, sys, time
import torch, yaml
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, "src")
from data_common import collate
from data_give import GiveDataset
from data_speechocean import SpeechoceanDataset
from evaluate import evaluate, short
from losses import compute_loss
from model import PronunciationScorer
from train import get_device, load_checkpoint, save_checkpoint, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True, choices=["speechocean", "give"])
ap.add_argument("--data", default=None)
ap.add_argument("--variant", default="calibrated")
ap.add_argument("--out", required=True)
ap.add_argument("--init", default=None)
ap.add_argument("--init_kind", default="scorer", choices=["ctc", "scorer"])
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr_head", type=float, default=5e-4)
ap.add_argument("--lr_backbone", type=float, default=3e-5)
ap.add_argument("--freeze_epochs", type=int, default=2, help="epochs with the Wav2Vec2 backbone frozen")
ap.add_argument("--top_epochs", type=int, default=2, help="epochs where only the top layers are trained")
ap.add_argument("--top_layers", type=int, default=4)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--limit", type=int, default=None, help="use only N utterances per split (smoke tests)")
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--device", default="auto")
args = ap.parse_args()

cfg = yaml.safe_load(open("configs/common.yaml"))
seed = cfg["seed"] if args.seed is None else args.seed
torch.manual_seed(seed)
device = get_device("cpu" if args.tiny else args.device)

if args.corpus == "speechocean":
    root = args.data or "data/speechocean762"
    sp = json.load(open("splits/speechocean_split.json"))
    train_ds = SpeechoceanDataset(root, "train", speakers=sp["train"])
    val_ds = SpeechoceanDataset(root, "train", speakers=sp["val"])
else:
    root = args.data or "data/GiVe"
    train_ds = GiveDataset(root, args.variant, "train", cfg["flag_below"])
    val_ds = GiveDataset(root, args.variant, "validation", cfg["flag_below"])
if args.limit:
    train_ds, val_ds = Subset(train_ds, range(min(args.limit, len(train_ds)))), Subset(val_ds, range(min(args.limit, len(val_ds))))
val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)
print(f"corpus {args.corpus} | train {len(train_ds)} utts, val {len(val_ds)} utts | device {device}")

model = PronunciationScorer(cfg["backbone"], tiny=args.tiny, **cfg["model"]).to(device)
os.makedirs(args.out, exist_ok=True)
last_path, best_path, log_path = (os.path.join(args.out, n) for n in ("last.pt", "best.pt", "log.jsonl"))


def stage_of(e):
    return "frozen" if e < args.freeze_epochs else ("top" if e < args.freeze_epochs + args.top_epochs else "all")


def make_opt(stage):
    {"frozen": model.freeze_backbone, "top": lambda: model.unfreeze_top(args.top_layers), "all": model.unfreeze_all}[stage]()
    o = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head), weight_decay=0.01)
    for g in o.param_groups:
        g["base_lr"] = g["lr"]
    return o


epoch0, step, best, best_epoch, saved_stage = 0, 0, -9.0, 0, None
if os.path.exists(last_path):
    meta = load_checkpoint(last_path, model, None, map_location=device)
    epoch0, step = meta["epoch"], meta["step"]
    best = meta["best"] if meta["best"] is not None else -9.0
    best_epoch, saved_stage = meta["extra"].get("best_epoch", 0), meta["extra"].get("stage")
    print(f"resuming after epoch {epoch0} (step {step}, best val score {best:.3f})")
elif args.init:
    sd = torch.load(args.init, map_location="cpu", weights_only=True)["model"]
    if args.init_kind == "ctc":
        sub = {k[len("wav2vec2."):]: v for k, v in sd.items() if k.startswith("wav2vec2.")}
        model.wav2vec2.load_state_dict(sub, strict=True)
        print("loaded Wav2Vec2 backbone from the CTC stage:", args.init)
    else:
        model.load_state_dict(sd, strict=True)
        print("loaded the full scoring model from:", args.init)

steps_per_epoch = math.ceil(len(train_ds) / args.batch)
total = args.epochs * steps_per_epoch


def lr_factor(s):
    warm = max(1, int(0.05 * total))
    return (s + 1) / warm if s < warm else max(0.1, (total - s) / max(1, total - warm))


opt, cur_stage = None, None
for epoch in range(epoch0, args.epochs):
    stage = stage_of(epoch)
    if stage != cur_stage:
        opt, cur_stage = make_opt(stage), stage
        if epoch == epoch0 and saved_stage == stage and os.path.exists(last_path):
            state = torch.load(last_path, map_location=device, weights_only=True)["optimizer"]
            if state is not None:
                opt.load_state_dict(state)
        print(f"--- epoch {epoch+1}: backbone stage '{stage}'")
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g, collate_fn=collate)
    model.train()
    if stage == "frozen":
        model.wav2vec2.eval()
    t0, run, n = time.time(), 0.0, 0
    for i, batch in enumerate(loader):
        for gr in opt.param_groups:
            gr["lr"] = gr["base_lr"] * lr_factor(step)
        b = to_device(batch, device)
        loss, _ = compute_loss(model.score_batch(b), b, cfg["loss_weights"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        step += 1; run += float(loss.detach()); n += 1
        if (i + 1) % 50 == 0:
            print(f"  epoch {epoch+1} batch {i+1}/{steps_per_epoch} loss {run/n:.4f}", flush=True)
    res = evaluate(model, val_loader, device)
    print(f"epoch {epoch+1}/{args.epochs} | train loss {run/max(n,1):.4f} | val {short(res)} | {(time.time()-t0)/60:.1f} min")
    improved = res["val_score"] == res["val_score"] and res["val_score"] > best
    if improved:
        best, best_epoch = res["val_score"], epoch + 1
        save_checkpoint(best_path, model, None, epoch + 1, step, best)
    save_checkpoint(last_path, model, opt, epoch + 1, step, best, extra={"stage": stage, "best_epoch": best_epoch})
    with open(log_path, "a") as f:
        f.write(json.dumps(dict(epoch=epoch + 1, stage=stage, train_loss=run / max(n, 1), **res)) + "\n")
    if device.type == "mps":
        torch.mps.empty_cache()
    if stage == "all" and epoch + 1 - best_epoch >= args.patience:
        print(f"no improvement for {args.patience} epochs: stopping early"); break
print(f"done. best val score {best:.3f} at epoch {best_epoch}. Best model: {best_path}")
