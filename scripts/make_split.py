UTT2SPK = "data/speechocean762/train/utt2spk"  

import json, random

speakers = sorted({line.split()[1] for line in open(UTT2SPK) if line.strip()})
random.seed(42)                      # fixed seed so the split never changes
random.shuffle(speakers)

val = sorted(speakers[:25])          # 25 validation speakers
train = sorted(speakers[25:])        # the rest for training
assert not set(train) & set(val), "a speaker is in both sets!"

with open("splits/speechocean_split.json", "w") as f:
    json.dump({"train": train, "val": val}, f, indent=1)

print(len(train), "train speakers,", len(val), "validation speakers")


