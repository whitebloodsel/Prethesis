"""Training utilities shared by every model: device choice, batch moving, safe resumable checkpoints."""
import os
import torch


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def save_checkpoint(path, model, optimizer, epoch, step, best=None, extra=None):
    """Write to a temporary file first, then rename, so a crash can never leave a broken checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                 epoch=epoch, step=step, best=best, extra=extra or {}, torch_rng=torch.get_rng_state())
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    return dict(epoch=state["epoch"], step=state["step"], best=state["best"], extra=state["extra"])
