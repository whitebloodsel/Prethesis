import json, os
from torch.utils.data import Dataset
from phones import PHONE_ID, PHONE_LIST, IPA_TO_ARPABET
from data_common import read_two_col, build_item, collate   # collate is re-exported for convenience


class GiveDataset(Dataset):
    """Indonesian corpus. variant: 'calibrated' (severity-corrected sentence scores) or 'raw_median'.
    split: 'train', 'validation' or 'test' (fixed speaker split from the data folder).
    Audio paths come from wav.scp, because 15 file names contain a typo (a double dot)."""

    def __init__(self, root, variant="calibrated", split="train", flag_below=9):
        base = os.path.join(root, variant)
        self.scores = json.load(open(os.path.join(base, "resource", "scores.json"), encoding="utf-8"))
        self.utt2spk = read_two_col(os.path.join(base, split, "utt2spk"))
        self.wav_path = {u: os.path.normpath(os.path.join(base, split, rel))
                         for u, rel in read_two_col(os.path.join(base, split, "wav.scp")).items()}
        self.utts = sorted(self.wav_path)
        missing = [u for u in self.utts if not os.path.exists(self.wav_path[u]) or u not in self.scores]
        assert not missing, f"{len(missing)} utterances missing audio or labels, e.g. {missing[:3]}"
        self.flag_below = flag_below
        self.phone_list = PHONE_LIST

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, i):
        u = self.utts[i]
        return build_item(u, self.utt2spk[u], self.scores[u], self.wav_path[u],
                          IPA_TO_ARPABET.__getitem__, PHONE_ID, flag_below=self.flag_below)
