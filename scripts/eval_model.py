"""Evaluate a trained model on a test set, e.g. M3 zero-shot on the Indonesian test speakers.
  python scripts/eval_model.py --ckpt checkpoints/m3_score/best.pt --corpus give --split test"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, sys
import torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from data_common import collate
from data_give import GiveDataset
from data_speechocean import SpeechoceanDataset
from evaluate import evaluate, short
from model import PronunciationScorer
from train import get_device, load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--corpus", required=True, choices=["speechocean", "give"])
ap.add_argument("--split", default="test", help="give: train/validation/test; speechocean: train/test (official folders)")
ap.add_argument("--variant", default="calibrated")
ap.add_argument("--data", default=None)
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--device", default="auto")
args = ap.parse_args()

cfg = yaml.safe_load(open("configs/common.yaml"))
device = get_device("cpu" if args.tiny else args.device)
if args.corpus == "give":
    ds = GiveDataset(args.data or "data/GiVe", args.variant, args.split, cfg["flag_below"])
else:
    ds = SpeechoceanDataset(args.data or "data/speechocean762", args.split)
model = PronunciationScorer(cfg["backbone"], tiny=args.tiny, **cfg["model"]).to(device)
load_checkpoint(args.ckpt, model, None, map_location=device)
res = evaluate(model, DataLoader(ds, batch_size=8, collate_fn=collate), device)
print(f"{args.ckpt} on {args.corpus}/{args.split} ({res['n_utts']} utts, {res['n_words']} words, {res['n_phones']} phones)")
print(short(res))
name = os.path.basename(os.path.dirname(args.ckpt)) + f"_{args.corpus}_{args.split}"
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/{name}.json", "w"), indent=2)
print("saved results/" + name + ".json")
