import json, sys
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from data_speechocean import SpeechoceanDataset, collate

ROOT = "data/speechocean762"
split = json.load(open("splits/speechocean_split.json"))
train = SpeechoceanDataset(ROOT, "train", speakers=split["train"])
val = SpeechoceanDataset(ROOT, "train", speakers=split["val"])
print("train utterances:", len(train), "| val utterances:", len(val))
print("speaker overlap between train and val:", set(train.utt2spk[u] for u in train.utts) & set(val.utt2spk[u] for u in val.utts))
b = next(iter(DataLoader(train, batch_size=4, shuffle=True, collate_fn=collate)))
for k, v in b.items():
    print(f"{k:13s}", tuple(v.shape) if hasattr(v, "shape") else v)
print("value ranges  sent:", float(b["sent"].min()), float(b["sent"].max()),
      "| word acc:", float(b["w_acc"][b["w_mask"]].min()), float(b["w_acc"][b["w_mask"]].max()),
      "| phone:", float(b["p_score"][b["p_mask"]].min()), float(b["p_score"][b["p_mask"]].max()))
print("phones per utterance in the batch:", b["p_mask"].sum(1).tolist())
