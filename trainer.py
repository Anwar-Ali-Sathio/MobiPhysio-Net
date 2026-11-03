"""Generic mini-batch trainer (AdamW + class-weighted cross-entropy, validation
macro-F1 checkpoint selection, cosine schedule, early stopping)."""
from __future__ import annotations

import copy
import logging
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from ..evaluation.metrics import macro_f1

log = logging.getLogger(__name__)


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def iterate(idx: np.ndarray, batch: int, shuffle: bool, rng: np.random.Generator):
    idx = rng.permutation(idx) if shuffle else idx
    for s in range(0, len(idx), batch):
        yield idx[s:s + batch]


@torch.no_grad()
def predict(model: nn.Module, idx: np.ndarray, forward: Callable, batch: int = 64,
            collect: Callable | None = None) -> dict:
    model.eval()
    probs, extra = [], []
    for b in iterate(idx, batch, False, np.random.default_rng(0)):
        out = forward(model, b, False)
        logits = out["logits"] if isinstance(out, dict) else out
        probs.append(torch.softmax(logits.float(), 1).numpy())
        if collect is not None and isinstance(out, dict):
            extra.append(collect(out))
    p = np.concatenate(probs)
    res = {"proba": p, "pred": p.argmax(1)}
    if extra:
        res["extra"] = np.concatenate(extra)
    return res


def fit(model: nn.Module, train_idx: np.ndarray, val_idx: np.ndarray, labels: np.ndarray,
        forward: Callable, cfg: dict, num_classes: int, seed: int,
        on_epoch_end: Callable | None = None, on_start: Callable | None = None) -> dict:
    """``forward(model, batch_idx, train) -> logits | {"logits": ...}``.

    ``on_start(model)`` / ``on_epoch_end(model)`` run before the first epoch and
    after every training epoch (before validation) - used by MobiPhysio-Net to
    re-encode the reference bank and refit the training-only distance scaler.
    """
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    epochs = cfg["epochs"]
    steps_per_epoch = int(np.ceil(len(train_idx) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * steps_per_epoch,
                                                pct_start=cfg.get("warmup_frac", 0.1))
    w = class_weights(labels[train_idx], num_classes) if cfg.get("class_weighting", True) else None
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.0))
    best = {"score": -1.0, "epoch": -1, "state": None}
    history, patience, bad = [], cfg.get("patience", 8), 0
    if on_start:
        on_start(model)
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for b in iterate(train_idx, cfg["batch_size"], True, rng):
            out = forward(model, b, True)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = loss_fn(logits, torch.as_tensor(labels[b], dtype=torch.long))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()
            sched.step()
            losses.append(loss.item())
        train_s = time.time() - t0
        if on_epoch_end:
            on_epoch_end(model)
        pv = predict(model, val_idx, forward)
        f1 = macro_f1(labels[val_idx], pv["pred"], num_classes)
        acc = float((pv["pred"] == labels[val_idx]).mean())
        vloss = float(nn.functional.nll_loss(torch.log(torch.tensor(pv["proba"]).clamp_min(1e-12)),
                                             torch.as_tensor(labels[val_idx])).item())
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": vloss,
                        "val_macro_f1": f1, "val_accuracy": acc, "epoch_train_s": round(train_s, 2),
                        "lr_end": float(sched.get_last_lr()[0])})
        log.info("epoch %d loss=%.4f val_f1=%.4f val_acc=%.4f (%.1fs)", ep, history[-1]["train_loss"],
                 f1, acc, train_s)
        if f1 > best["score"]:
            best = {"score": f1, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info("early stopping at epoch %d (best %d)", ep, best["epoch"])
                break
    model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "best_val_macro_f1": best["score"],
            "selection_rule": "max validation macro F1 (first occurrence)"}
