"""Shared scoring model (used by M2, M3 and M4 so the comparison is fair).

audio -> Wav2Vec2 -> weighted sum of all layers -> BiLSTM -> acoustic memory
sentence head : mean-pooled memory -> 4 scores (accuracy, fluency, prosody, completeness)
phone/word    : canonical phone sequence (ids + positions + word index) attends to the acoustic memory
                (cross-attention, so NO forced alignment is needed) -> phone scores
                word vector = mean of its phone states -> word accuracy and word stress
All scores are in [0, 1] (sigmoid); stress is a logit for a binary loss.
"""
import warnings

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from phones import PHONE_LIST

MAX_PHONES = 256
MAX_WORDS = 64


class WeightedLayerSum(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))          # softmax(0) = uniform at the start

    def forward(self, hidden_states):
        a = torch.softmax(self.w, dim=0)
        return sum(a[i] * h for i, h in enumerate(hidden_states))


class PronunciationScorer(nn.Module):
    def __init__(self, backbone="facebook/wav2vec2-base", tiny=False, lstm_hidden=256, d_model=256,
                 n_heads=4, n_dec_layers=2, dropout=0.1):
        super().__init__()
        if tiny:                                              # random small network, only for quick tests
            cfg = Wav2Vec2Config(hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                                 intermediate_size=128, conv_dim=(32,) * 7, num_conv_pos_embeddings=16,
                                 num_conv_pos_embedding_groups=4, apply_spec_augment=False, layerdrop=0.0)
            self.wav2vec2 = Wav2Vec2Model(cfg)
        else:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(backbone, layerdrop=0.0)
        cfg = self.wav2vec2.config
        self.wav2vec2.freeze_feature_encoder()               # the CNN stays frozen in every model
        # base models were trained without an attention mask; passing one hurts them
        self.use_attn_mask = cfg.feat_extract_norm == "layer"
        self.layer_sum = WeightedLayerSum(cfg.num_hidden_layers + 1)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cfg.hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
        self.mem_proj = nn.Linear(2 * lstm_hidden, d_model)
        self.sent_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.phone_emb = nn.Embedding(len(PHONE_LIST), d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_PHONES, d_model)
        self.word_emb = nn.Embedding(MAX_WORDS, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_dec_layers)
        self.phone_head = nn.Linear(d_model, 1)
        self.word_acc_head = nn.Linear(d_model, 1)
        self.word_stress_head = nn.Linear(d_model, 1)
        self._pack_ok = True

    # ---- freezing helpers (used for gradual unfreezing in M3/M4) ----
    def backbone_params(self):
        return [p for p in self.wav2vec2.parameters()]

    def freeze_backbone(self):
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

    def unfreeze_top(self, k):
        """Train only the top k transformer layers of Wav2Vec2 (the CNN stays frozen)."""
        self.freeze_backbone()
        for layer in self.wav2vec2.encoder.layers[-k:]:
            for p in layer.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.wav2vec2.parameters():
            p.requires_grad = True
        self.wav2vec2.freeze_feature_encoder()

    def param_groups(self, lr_backbone, lr_head):
        bb = {id(p) for p in self.wav2vec2.parameters()}
        back = [p for p in self.wav2vec2.parameters() if p.requires_grad]
        head = [p for p in self.parameters() if id(p) not in bb and p.requires_grad]
        groups = [{"params": head, "lr": lr_head}]
        if back:
            groups.append({"params": back, "lr": lr_backbone})
        return groups

    # ---- forward ----
    def _lstm(self, h, f_len):
        if self._pack_ok:
            try:
                packed = nn.utils.rnn.pack_padded_sequence(h, f_len.cpu(), batch_first=True, enforce_sorted=False)
                out, _ = self.lstm(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
                return out
            except (NotImplementedError, RuntimeError) as e:
                warnings.warn(f"packed LSTM failed on this device ({e}); using the unpacked LSTM instead")
                self._pack_ok = False
        out, _ = self.lstm(h)
        return out

    def forward(self, audio, audio_mask, p_id, p_word, p_mask, n_words):
        lengths = audio_mask.sum(1)
        out = self.wav2vec2(audio, attention_mask=audio_mask.long() if self.use_attn_mask else None,
                            output_hidden_states=True)
        h = self.layer_sum(out.hidden_states)
        T = h.shape[1]
        f_len = self.wav2vec2._get_feat_extract_output_lengths(lengths).clamp(min=1, max=T)
        f_mask = torch.arange(T, device=h.device)[None, :] < f_len[:, None]
        h = self.drop(h * f_mask.unsqueeze(-1))
        mem = self.mem_proj(self._lstm(h, f_len))

        pooled = (mem * f_mask.unsqueeze(-1)).sum(1) / f_len.unsqueeze(-1).to(mem.dtype)
        sent = torch.sigmoid(self.sent_head(pooled))

        P = p_id.shape[1]
        assert P <= MAX_PHONES, f"utterance has {P} phones, raise MAX_PHONES"
        pos = torch.arange(P, device=p_id.device)[None, :]
        x = self.phone_emb(p_id) + self.pos_emb(pos) + self.word_emb(p_word.clamp(max=MAX_WORDS - 1))
        x = self.decoder(tgt=x, memory=mem, tgt_key_padding_mask=~p_mask, memory_key_padding_mask=~f_mask)
        p_score = torch.sigmoid(self.phone_head(x)).squeeze(-1)

        sel = (p_word.unsqueeze(1) == torch.arange(n_words, device=x.device)[None, :, None]) & p_mask.unsqueeze(1)
        sel = sel.to(x.dtype)                                          # (B, words, phones)
        wv = sel @ x / sel.sum(-1, keepdim=True).clamp(min=1)
        w_acc = torch.sigmoid(self.word_acc_head(wv)).squeeze(-1)
        w_stress = self.word_stress_head(wv).squeeze(-1)
        return dict(sent=sent, w_acc=w_acc, w_stress=w_stress, p_score=p_score)

    def score_batch(self, batch):
        return self(batch["audio"], batch["audio_mask"], batch["p_id"], batch["p_word"], batch["p_mask"],
                    n_words=batch["w_acc"].shape[1])
