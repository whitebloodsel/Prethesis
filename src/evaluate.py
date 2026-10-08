"""One evaluation function for every model (M2, M3, M4), so the numbers are comparable.
Phones are scored only where a human phone label exists (Indonesian: words with accuracy below the flag; Speechocean: all).
MSE is reported on the original score scales (sentence/word 0-10, phone 0-2)."""
import numpy as np
import torch
from data_common import SENT_FIELDS
from train import to_device


def pcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    acc = {k: [] for k in ("sent_p", "sent_t", "w_p", "w_t", "st_p", "st_t", "p_p", "p_t")}
    for batch in loader:
        b = to_device(batch, device)
        out = model.score_batch(b)
        m, ms, mp = b["w_mask"], b["w_stress_mask"], b["p_loss_mask"]
        acc["sent_p"].append(out["sent"].cpu()); acc["sent_t"].append(b["sent"].cpu())
        acc["w_p"].append(out["w_acc"][m].cpu()); acc["w_t"].append(b["w_acc"][m].cpu())
        acc["st_p"].append((out["w_stress"][ms] > 0).long().cpu()); acc["st_t"].append(b["w_stress"][ms].cpu())
        acc["p_p"].append(out["p_score"][mp].cpu()); acc["p_t"].append(b["p_score"][mp].cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def metrics(pr):
    r = {}
    for i, name in enumerate(SENT_FIELDS):
        r[f"pcc_sent_{name}"] = pcc(pr["sent_p"][:, i], pr["sent_t"][:, i])
        r[f"mse_sent_{name}"] = float(((pr["sent_p"][:, i] - pr["sent_t"][:, i]) ** 2).mean() * 100)
    r["pcc_word_acc"] = pcc(pr["w_p"], pr["w_t"])
    r["mse_word_acc"] = float(((pr["w_p"] - pr["w_t"]) ** 2).mean() * 100) if len(pr["w_p"]) else float("nan")
    r["pcc_phone"] = pcc(pr["p_p"], pr["p_t"])
    r["mse_phone"] = float(((pr["p_p"] - pr["p_t"]) ** 2).mean() * 4) if len(pr["p_p"]) else float("nan")
    r["stress_acc"] = float((pr["st_p"] == pr["st_t"]).mean()) if len(pr["st_p"]) else float("nan")
    r["stress_n_incorrect"] = int((pr["st_t"] == 0).sum())
    r["n_utts"], r["n_words"], r["n_phones"] = len(pr["sent_p"]), len(pr["w_p"]), len(pr["p_p"])
    main = [r["pcc_sent_accuracy"], r["pcc_sent_fluency"], r["pcc_sent_prosodic"], r["pcc_word_acc"], r["pcc_phone"]]
    r["val_score"] = float(np.nanmean(main)) if not np.all(np.isnan(main)) else float("nan")
    return r


def evaluate(model, loader, device):
    return metrics(predict(model, loader, device))


def short(r):
    f = lambda x: "nan" if x != x else f"{x:.3f}"
    return (f"score {f(r['val_score'])} | sent acc {f(r['pcc_sent_accuracy'])} flu {f(r['pcc_sent_fluency'])} "
            f"pros {f(r['pcc_sent_prosodic'])} | word {f(r['pcc_word_acc'])} | phone {f(r['pcc_phone'])} "
            f"| stress acc {f(r['stress_acc'])}")
