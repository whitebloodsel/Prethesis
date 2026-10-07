"""Multi-task loss (Eq. 1). Every term is averaged over its own valid positions only."""
import torch
import torch.nn.functional as F


def masked_mean(x, mask):
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def compute_loss(out, batch, weights):
    parts = dict(
        sent=F.mse_loss(out["sent"], batch["sent"]),
        word=masked_mean((out["w_acc"] - batch["w_acc"]) ** 2, batch["w_mask"]),
        stress=masked_mean(F.binary_cross_entropy_with_logits(out["w_stress"], batch["w_stress"].float(),
                                                              reduction="none"), batch["w_stress_mask"]),
        phone=masked_mean((out["p_score"] - batch["p_score"]) ** 2, batch["p_loss_mask"]),
    )
    total = sum(weights[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
