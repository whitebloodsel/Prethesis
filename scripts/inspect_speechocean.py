import json, glob, collections
import soundfile as sf

ROOT = "data/speechocean762"
found = glob.glob(f"{ROOT}/**/scores.json", recursive=True)
print("scores.json found at:", found)
scores = json.load(open(found[0]))
print("number of utterances:", len(scores))

utt = next(iter(scores))
print("example id:", utt)
print(json.dumps(scores[utt], indent=1)[:1500])

def rng(v):
    return (min(v), max(v)) if v else None

sent = collections.defaultdict(list)
word = collections.defaultdict(list)
phone = []
for e in scores.values():
    for k, v in e.items():
        if isinstance(v, (int, float)):
            sent[k].append(v)
    for w in e["words"]:
        for k, v in w.items():
            if isinstance(v, (int, float)):
                word[k].append(v)
        phone += w.get("phones-accuracy", [])

print("sentence-level ranges:", {k: rng(v) for k, v in sent.items()})
print("word-level ranges:", {k: rng(v) for k, v in word.items()})
print("phone score values:", sorted(set(phone)))

wavs = glob.glob(f"{ROOT}/**/*.WAV", recursive=True) + glob.glob(f"{ROOT}/**/*.wav", recursive=True)
print("wav files:", len(wavs))
x, sr = sf.read(wavs[0])
print(wavs[0], "| sample rate:", sr, "| seconds:", round(len(x) / sr, 2), "| array dims:", x.ndim)
