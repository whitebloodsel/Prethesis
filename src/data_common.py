"""Shared pieces for both corpora, so Speechocean762 and the Indonesian data produce identical batches.

Normalization rules:
  sentence accuracy / fluency / prosody / completeness : score / 10  (both corpora store completeness on 0-10)
  word accuracy                                         : score / 10
  word stress                                           : class 1 = correct (10), 0 = incorrect (5); anything else is masked out
  phone score                                           : score / 2   (soft value, kept as an average)
  phone loss mask                                       : Speechocean762 = all phones; Indonesian = phones in words with accuracy < flag_below
"""
import numpy as np
import soundfile as sf
import torch

SENT_FIELDS = ["accuracy", "fluency", "prosodic", "completeness"]


def read_two_col(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
    return out


def build_item(utt, speaker, entry, wav_path, to_arpabet, phone_id, flag_below=None):
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000 and audio.ndim == 1, f"{utt}: expected 16 kHz mono, got {sr} Hz, {audio.ndim} dims"
    audio = (audio - audio.mean()) / (audio.std() + 1e-7)       # what Wav2Vec2 expects
    sent = torch.tensor([entry[k] / 10.0 for k in SENT_FIELDS], dtype=torch.float32)
    w_acc, w_stress, w_ok, p_score, p_id, p_word, p_use = [], [], [], [], [], [], []
    for wi, w in enumerate(entry["words"]):
        acc = w["accuracy"]
        w_acc.append(acc / 10.0)
        st = w.get("stress")
        w_stress.append(1 if st == 10 else 0)
        w_ok.append(st in (5, 10))
        use = True if flag_below is None else acc < flag_below
        for ph, sc in zip(w["phones"], w["phones-accuracy"]):
            p_id.append(phone_id[to_arpabet(ph)])
            p_score.append(sc / 2.0)
            p_word.append(wi)
            p_use.append(use)
    return dict(utt=utt, speaker=speaker, audio=torch.from_numpy(audio), sent=sent,
                w_acc=torch.tensor(w_acc, dtype=torch.float32), w_stress=torch.tensor(w_stress),
                w_stress_ok=torch.tensor(w_ok), p_score=torch.tensor(p_score, dtype=torch.float32),
                p_id=torch.tensor(p_id), p_word=torch.tensor(p_word), p_use=torch.tensor(p_use))


def _pad(seqs, value=0):
    n = max(len(s) for s in seqs)
    out = torch.full((len(seqs), n), value, dtype=seqs[0].dtype)
    mask = torch.zeros(len(seqs), n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
        mask[i, :len(s)] = True
    return out, mask


def collate(batch):
    audio, audio_mask = _pad([b["audio"] for b in batch])
    w_acc, w_mask = _pad([b["w_acc"] for b in batch])
    w_stress, _ = _pad([b["w_stress"] for b in batch])
    w_ok, _ = _pad([b["w_stress_ok"] for b in batch], value=False)
    p_score, p_mask = _pad([b["p_score"] for b in batch])
    p_id, _ = _pad([b["p_id"] for b in batch])
    p_word, _ = _pad([b["p_word"] for b in batch])
    p_use, _ = _pad([b["p_use"] for b in batch], value=False)
    return dict(utt=[b["utt"] for b in batch], speaker=[b["speaker"] for b in batch],
                audio=audio, audio_mask=audio_mask, sent=torch.stack([b["sent"] for b in batch]),
                w_acc=w_acc, w_stress=w_stress, w_mask=w_mask,
                w_stress_mask=w_mask & w_ok,                    # stress loss only where the label is valid
                p_score=p_score, p_id=p_id, p_word=p_word, p_mask=p_mask,
                p_loss_mask=p_mask & p_use)                    # phone loss only where a phone label was collected
