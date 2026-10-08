"""Step 8a: how fast and how big? Times training steps and checks that CTC loss works on this device.
  python scripts/bench.py                  # real model, batch sizes 4 and 8
  python scripts/bench.py --tiny --device cpu
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")     # unsupported GPU ops run on CPU instead of crashing
import argparse, json, random, sys, time, warnings
import torch, torch.nn.functional as F, yaml
sys.path.insert(0, "src")
from data_speechocean import SpeechoceanDataset, collate
from losses import compute_loss
from model import PronunciationScorer
from train import get_device, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--device", default="auto")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--batch_sizes", type=int, nargs="+", default=[4, 8])
args = ap.parse_args()

cfg = yaml.safe_load(open("configs/common.yaml"))
device = get_device("cpu" if args.tiny else args.device)
print("device:", device)
sync = torch.mps.synchronize if device.type == "mps" else (torch.cuda.synchronize if device.type == "cuda" else (lambda: None))
mem = (lambda: torch.mps.driver_allocated_memory() / 1e9) if device.type == "mps" else \
      (lambda: torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan"))

split = json.load(open("splits/speechocean_split.json"))
ds = SpeechoceanDataset("data/speechocean762", "train", speakers=split["train"])
random.seed(0)
idx = random.sample(range(len(ds)), min(len(ds), 64))
items = [ds[i] for i in idx]
secs = [len(x["audio"]) / 16000 for x in items]
print(f"utterance length in a 64-sample: mean {sum(secs)/len(secs):.1f}s, max {max(secs):.1f}s")

for bs in args.batch_sizes:
    try:
        torch.manual_seed(0)
        model = PronunciationScorer(cfg["backbone"], tiny=args.tiny, **cfg["model"]).to(device)
        model.unfreeze_all(); model.train()
        opt = torch.optim.AdamW(model.param_groups(3e-5, 1e-3))
        times = []
        for s in range(args.steps + 3):
            b = to_device(collate([items[(s * bs + j) % len(items)] for j in range(bs)]), device)
            sync(); t0 = time.time()
            opt.zero_grad()
            loss, _ = compute_loss(model.score_batch(b), b, cfg["loss_weights"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            sync()
            if s >= 3:                                            # first 3 steps are warm-up
                times.append(time.time() - t0)
        per = sum(times) / len(times)
        n_train = len(ds)
        print(f"batch {bs}: {per:.2f} s/step | memory ~{mem():.1f} GB | one epoch of {n_train} utts ~ {n_train/bs*per/60:.0f} min")
    except (RuntimeError, MemoryError) as e:
        print(f"batch {bs}: FAILED ({str(e)[:120]})")
    del model, opt
    if device.type == "mps": torch.mps.empty_cache()

# CTC loss on this device (needed for the CTC stage)
print("CTC check ...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    logits = torch.randn(120, 2, 32, device=device, requires_grad=True)
    lp = logits.log_softmax(-1)
    tgt = torch.randint(1, 32, (2, 20), device=device)
    loss = F.ctc_loss(lp, tgt, torch.tensor([120, 100]), torch.tensor([20, 15]), blank=0, zero_infinity=True)
    loss.backward()
    fb = [str(x.message)[:100] for x in w if "fall" in str(x.message).lower() or "cpu" in str(x.message).lower()]
print("CTC loss OK, value", round(float(loss.detach()), 3), "| fell back to CPU:" , "yes" if fb else "no")