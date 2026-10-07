import glob, json, os, re
from torch.utils.data import Dataset
from phones import PHONE_ID, PHONE_LIST
from data_common import read_two_col, build_item, collate   # collate is re-exported for convenience


def strip_stress(phone):
    return re.sub(r"\d", "", phone)          # 'IY0' -> 'IY'


class SpeechoceanDataset(Dataset):
    def __init__(self, root, split="train", speakers=None):
        """split: official folder ('train' or 'test'). speakers: optional speaker ids to keep
        (use the ids in splits/speechocean_split.json)."""
        self.scores = json.load(open(os.path.join(root, "resource", "scores.json")))
        self.utt2spk = read_two_col(os.path.join(root, split, "utt2spk"))
        keep = set(speakers) if speakers is not None else None
        self.utts = sorted(u for u, s in self.utt2spk.items() if (keep is None or s in keep) and u in self.scores)
        wavs = glob.glob(os.path.join(root, "WAVE", "**", "*.[Ww][Aa][Vv]"), recursive=True)
        self.wav_path = {os.path.splitext(os.path.basename(p))[0]: p for p in wavs}
        missing = [u for u in self.utts if u not in self.wav_path]
        assert not missing, f"{len(missing)} utterances have no audio file, e.g. {missing[:3]}"
        self.phone_list = PHONE_LIST

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, i):
        u = self.utts[i]
        return build_item(u, self.utt2spk[u], self.scores[u], self.wav_path[u], strip_stress, PHONE_ID)
