"""Step 6 tests: (1) forward pass, (2) overfit 8 examples, (3) checkpoint save + reload.
  python scripts/test_model.py                # real Wav2Vec2-base (downloads ~360 MB the first time)
  python scripts/test_model.py --tiny         # tiny random network, 1 minute smoke test
"""
import argparse, json, os, sys, tempfile
import torch, yaml
sys.path.insert(0, "src")
from data_give import GiveDataset, collate
from losses import compute_loss
from model import PronunciationScorer
from train import get_device, load_checkpoint, save_checkpoint, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/GiVe")
ap.add_argument("--variant", default="calibrated")
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--device", default="auto")
args = ap.parse_args()

cfg = yaml.safe_load(open("configs/common.yaml"))
torch.manual_seed(cfg["seed"])
device = get_device("cpu" if args.tiny else args.device)
print("device:", device)
build = lambda: PronunciationScorer(cfg["backbone"], tiny=args.tiny, **cfg["model"]).to(device)

# 8 short Indonesian utterances (short = fast test)
ds = GiveDataset(args.data, args.variant, "train")
items = sorted([ds[i] for i in range(40)], key=lambda x: len(x["audio"]))[:8]
batch = to_device(collate(items), device)
print("audio batch:", tuple(batch["audio"].shape), "| phones in loss:", int(batch["p_loss_mask"].sum()))

# ---------- Test 1: forward pass ----------
model = build().eval()
with torch.no_grad():
    out = model.score_batch(batch)
B, W, P = batch["w_acc"].shape[0], batch["w_acc"].shape[1], batch["p_id"].shape[1]
assert out["sent"].shape == (B, 4) and out["w_acc"].shape == (B, W)
assert out["w_stress"].shape == (B, W) and out["p_score"].shape == (B, P)
assert all(torch.isfinite(v).all() for v in out.values()), "NaN or inf in outputs"
for k in ("sent", "w_acc", "p_score"):
    assert 0 <= out[k].min() and out[k].max() <= 1, f"{k} outside [0,1]"
loss, parts = compute_loss(out, batch, cfg["loss_weights"])
assert torch.isfinite(loss)
print("TEST 1 PASSED  forward pass. initial loss", round(float(loss), 4), parts)

# optional: same check on a Speechocean batch
split_file, so_root = "splits/speechocean_split.json", "data/speechocean762"
if os.path.exists(split_file) and os.path.exists(so_root) and not args.tiny:
    from data_speechocean import SpeechoceanDataset
    sp = json.load(open(split_file))
    so = SpeechoceanDataset(so_root, "train", speakers=sp["train"])
    sb = to_device(collate([so[i] for i in range(4)]), device)
    with torch.no_grad():
        so_out = model.score_batch(sb)
    assert all(torch.isfinite(v).all() for v in so_out.values())
    print("TEST 1b PASSED forward pass on a Speechocean762 batch")

# ---------- Test 2: overfit 8 examples ----------
model = build()
model.unfreeze_all()
opt = torch.optim.AdamW(model.param_groups(lr_backbone=3e-5, lr_head=1e-3), weight_decay=0.0)
model.train()
history = []
for step in range(1, args.steps + 1):
    opt.zero_grad()
    loss, parts = compute_loss(model.score_batch(batch), batch, cfg["loss_weights"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    history.append(float(loss))
    if step % 10 == 0 or step == 1:
        print(f"  step {step:4d} loss {float(loss):.4f}  " + " ".join(f"{k} {v:.3f}" for k, v in parts.items()))
model.eval()
with torch.no_grad():
    out = model.score_batch(batch)
m = batch["p_loss_mask"]
pred, tgt = out["p_score"][m], batch["p_score"][m]
r2 = 1 - float(((pred - tgt) ** 2).mean() / tgt.var(unbiased=False).clamp(min=1e-8))
first, last = sum(history[:3]) / 3, sum(history[-5:]) / 5
print(f"  loss {first:.4f} -> {last:.4f} | phone R^2 on the 8 training examples: {r2:.2f}")
assert last < 0.2 * first, "loss did not fall enough: the model cannot memorize 8 examples"
assert r2 > 0.5, "phone scores not memorized: check masks and phone alignment"
print("TEST 2 PASSED  overfits 8 examples")

# ---------- Test 3: checkpoint save + reload ----------
with torch.no_grad():
    before = model.score_batch(batch)
path = os.path.join(tempfile.mkdtemp(), "ckpt", "last.pt")
save_checkpoint(path, model, opt, epoch=3, step=args.steps, best=0.5)
assert os.path.exists(path) and not os.path.exists(path + ".tmp")
model2 = build().eval()
opt2 = torch.optim.AdamW(model2.param_groups(lr_backbone=3e-5, lr_head=1e-3), weight_decay=0.0)
meta = load_checkpoint(path, model2, opt2, map_location=device)
with torch.no_grad():
    after = model2.score_batch(batch)
diff = max(float((before[k] - after[k]).abs().max()) for k in before)
assert diff < 1e-5, f"reloaded model differs by {diff}"
assert meta["epoch"] == 3 and meta["step"] == args.steps and meta["best"] == 0.5
assert len(opt2.state) == len(opt.state) > 0, "optimizer state not restored"
model2.train(); opt2.zero_grad()
l2, _ = compute_loss(model2.score_batch(batch), batch, cfg["loss_weights"]); l2.backward(); opt2.step()
print(f"TEST 3 PASSED  checkpoint reload (max output difference {diff:.2e}), training resumes")
print("ALL 3 TESTS PASSED")
