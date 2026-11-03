"""Per-model training/evaluation runners (Algorithms 7 and 8).

Every runner uses the SAME participant-independent split and returns test
probabilities/predictions plus training history, parameter counts and latency.
Latency = median wall time to classify one video from its cached 16 x 224 x 224
frames on CPU (batch of one video, includes the backbone; excludes MP4 decoding,
which is common to all models and reported separately).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..evaluation.metrics import macro_f1
from ..models.references import build_reference_bank, reference_ids, reference_labels
from ..preprocessing.cache import FeatureStore, FrameCache
from .trainer import fit, predict

log = logging.getLogger(__name__)


@dataclass
class Bundle:
    manifest: pd.DataFrame          # usable videos, aligned with feature arrays
    data_dir: Path
    num_classes: int
    class_codes: list[str]
    t: int
    size: int
    experiment: str = "full"

    @property
    def ids(self) -> np.ndarray:
        # Plain NumPy strings (pandas >= 3 returns a string extension array from .values,
        # which does not broadcast in comparisons).
        return self.manifest.video_id.to_numpy(dtype=str)

    @property
    def labels(self) -> np.ndarray:
        return self.manifest.label.values.astype(int)

    def idx(self, split: str) -> np.ndarray:
        return np.where(self.manifest.split.values == split)[0]

    @property
    def store(self) -> FeatureStore:
        return FeatureStore(self.data_dir, self.experiment)

    @property
    def frame_cache(self) -> FrameCache:
        return FrameCache(self.data_dir, self.t, self.size)

    def aligned(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        z = self.store.load(name)
        pos = {v: i for i, v in enumerate(z["video_ids"])}
        sel = np.array([pos[v] for v in self.ids])
        return z["feats"][sel], z["mask"][sel]

    def aligned_memmap(self, fname: str, ids_name: str | None = None):
        arr = np.load(self.store.root / fname, mmap_mode="r")
        ids = np.load(self.store.root / ids_name) if ids_name else self.store.load("vit_s16")["video_ids"]
        pos = {v: i for i, v in enumerate(ids)}
        return arr, np.array([pos[v] for v in self.ids])


def n_params(m: nn.Module, trainable_only: bool = False) -> int:
    return int(sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only)))


def _median_latency(fn, video_ids, cache: FrameCache, warmup: int = 2) -> dict:
    times = []
    for i, v in enumerate(video_ids):
        fr, mk = cache.load(v)
        t0 = time.perf_counter()
        fn(fr, mk)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return {"median_ms": float(np.median(times) * 1000), "p90_ms": float(np.quantile(times, 0.9) * 1000),
            "n_timed": len(times), "threads": torch.get_num_threads(),
            "definition": "cached 16x224x224 frames -> class probabilities, batch=1, CPU"}


def _result(b: Bundle, test_out: dict, fit_out: dict | None, extra: dict) -> dict:
    te = b.idx("test")
    return {"test_idx": te, "proba": test_out["proba"], "pred": test_out["pred"],
            "y_true": b.labels[te], "fit": fit_out, **extra}


# ----------------------------------------------------------------------------- SVM
def run_svm(b: Bundle, cfg: dict, seed: int, latency_ids: list[str]) -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from ..models.svm import video_descriptor
    X, _ = b.aligned("svm_descriptors")
    y = b.labels
    tr, va, te = b.idx("train"), b.idx("val"), b.idx("test")
    scaler = StandardScaler().fit(X[tr])  # TRAIN only
    Xs = scaler.transform(X)
    grid = []
    for C in cfg["C"]:
        for gmul in cfg["gamma_multipliers"]:
            gamma = gmul / X.shape[1]  # multiples of sklearn "scale" for standardised data
            m = SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced", random_state=seed)
            m.fit(Xs[tr], y[tr])
            grid.append({"C": C, "gamma_mult": gmul, "val_macro_f1": macro_f1(y[va], m.predict(Xs[va]), b.num_classes)})
            log.info("svm C=%s gmul=%s val_f1=%.4f", C, gmul, grid[-1]["val_macro_f1"])
    bestp = max(grid, key=lambda g: g["val_macro_f1"])
    final = SVC(C=bestp["C"], gamma=bestp["gamma_mult"] / X.shape[1], kernel="rbf",
                class_weight="balanced", probability=True, random_state=seed).fit(Xs[tr], y[tr])
    proba = final.predict_proba(Xs[te])
    # Predicted label = argmax of the reported probabilities (keeps metrics reproducible
    # from the prediction file); agreement with SVC.predict() is recorded.
    pred = proba.argmax(1)
    agree = float((final.predict(Xs[te]) == pred).mean())

    def lat(fr, mk):
        final.predict_proba(scaler.transform(video_descriptor(fr, mk)[None]))

    return _result(b, {"proba": proba, "pred": pred},
                   {"grid": grid, "best": bestp, "selection_rule": "max validation macro F1",
                    "history": []},
                   {"latency": _median_latency(lat, latency_ids, b.frame_cache),
                    "params_trainable": int(final.support_vectors_.size + final.dual_coef_.size),
                    "params_total": int(final.support_vectors_.size + final.dual_coef_.size),
                    "notes": {"svc_predict_vs_argmax_proba_agreement": agree,
                              "descriptor_dim": int(X.shape[1])}})


# ----------------------------------------------------------------------------- feature-sequence models
def run_feature_model(b: Bundle, model: nn.Module, feat_name: str, cfg: dict, seed: int,
                      latency_ids: list[str], backbone: nn.Module, backbone_norm) -> dict:
    X, M = b.aligned(feat_name)
    X = torch.from_numpy(X.astype(np.float32))
    M = torch.from_numpy(M)

    def forward(m, idx, train):
        return m(X[idx], M[idx])

    out = fit(model, b.idx("train"), b.idx("val"), b.labels, forward, cfg, b.num_classes, seed)
    pt = predict(model, b.idx("test"), forward)
    model.eval()

    def lat(fr, mk):
        with torch.no_grad():
            f = backbone(_norm(fr, *backbone_norm)).unsqueeze(0)
            model(f, torch.from_numpy(mk)[None])

    return _result(b, pt, out, {"latency": _median_latency(lat, latency_ids, b.frame_cache),
                                "params_trainable": n_params(model, True),
                                "params_total": n_params(model) + n_params(backbone)})


def _norm(frames_u8, mean, std):
    x = torch.from_numpy(frames_u8).float().div_(255).permute(0, 3, 1, 2)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- from-scratch CNNs
def subsample_frames(x: np.ndarray, mask: torch.Tensor, k: int, rng: np.random.Generator):
    """Sorted random subset of ``k`` of the T frames per video (temporal order kept).

    Used only during training (temporal jitter + CPU budget). Afterwards valid
    frames are moved before padded ones (stable), so padding stays a suffix as the
    packed LSTMs require.
    """
    t = x.shape[1]
    sel = np.stack([np.sort(rng.choice(t, k, replace=False)) for _ in range(len(x))])
    x = np.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 2)), axis=1)
    mk = torch.gather(mask, 1, torch.from_numpy(sel))
    order = torch.argsort((~mk).int(), dim=1, stable=True)
    mk = torch.gather(mk, 1, order)
    x = np.take_along_axis(x, order.numpy().reshape(order.shape + (1,) * (x.ndim - 2)), axis=1)
    return x, mk


SCRATCH_MEAN, SCRATCH_STD = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)


def run_scratch_cnn(b: Bundle, model: nn.Module, cfg: dict, seed: int, latency_ids: list[str]) -> dict:
    import cv2
    arr, sel = b.aligned_memmap("frames112.u8.npy", "frames112_ids.npy")
    frames = np.ascontiguousarray(arr[sel])  # uint8 in RAM (~2.2 GB for all videos)
    masks = torch.from_numpy(np.load(b.store.root / "frames112_mask.npy")[sel])
    mean = torch.tensor(SCRATCH_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(SCRATCH_STD).view(1, 1, 3, 1, 1)
    rng = np.random.default_rng(seed)
    pad = cfg.get("aug_pad", 8)

    def to_tensor(x):
        x = torch.from_numpy(x).float().div_(255).permute(0, 1, 4, 2, 3)
        return (x - mean) / std

    k_train = cfg.get("train_frames", frames.shape[1])

    def forward(m, idx, train):
        x, mk = frames[idx], masks[idx]
        if train and k_train < x.shape[1]:
            x, mk = subsample_frames(x, mk, k_train, rng)
        if train and cfg.get("augment", True):
            # Same random shift-crop + horizontal flip for all frames of a video.
            out = np.empty_like(x)
            for k in range(len(x)):
                p = np.pad(x[k], ((0, 0), (pad, pad), (pad, pad), (0, 0)))
                dy, dx = rng.integers(0, 2 * pad + 1, 2)
                v = p[:, dy:dy + x.shape[2], dx:dx + x.shape[3]]
                out[k] = v[:, :, ::-1] if rng.random() < 0.5 else v
            x = out
        return m(to_tensor(x), mk)

    out = fit(model, b.idx("train"), b.idx("val"), b.labels, forward, cfg, b.num_classes, seed)
    pt = predict(model, b.idx("test"), forward, batch=16)
    model.eval()
    small = frames.shape[2]

    def lat(fr, mk):
        with torch.no_grad():
            x = np.stack([cv2.resize(f, (small, small), interpolation=cv2.INTER_AREA) for f in fr])[None]
            model(to_tensor(x), torch.from_numpy(mk)[None])

    return _result(b, pt, out, {"latency": _median_latency(lat, latency_ids, b.frame_cache),
                                "params_trainable": n_params(model, True), "params_total": n_params(model)})


# ----------------------------------------------------------------------------- MobiPhysio-Net
def build_or_load_reference_bank(b: Bundle, cfg: dict, seed: int, path: Path) -> dict:
    import json
    if path.exists():
        bank = json.loads(path.read_text())
        from ..models.references import verify_reference_provenance
        verify_reference_provenance(bank, set(b.ids[b.idx("train")]),
                                    set(b.ids[b.idx("val")]) | set(b.ids[b.idx("test")]))
        return bank
    Z, M = b.aligned("vit_s16")
    Zn = Z.astype(np.float32)
    Zn /= np.linalg.norm(Zn, axis=-1, keepdims=True).clip(1e-8)
    t0 = time.time()
    bank = build_reference_bank(Zn, M, b.labels, b.ids, set(b.ids[b.idx("train")]),
                                set(b.ids[b.idx("val")]) | set(b.ids[b.idx("test")]),
                                b.num_classes, cfg["refs_per_class"], cfg["reference_sample_limit"],
                                cfg.get("dtw_band"), seed)
    bank["build_seconds"] = time.time() - t0
    bank["aggregation"] = cfg["dtw_aggregation"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, indent=2))
    return bank


def run_mobiphysio(b: Bundle, cfg: dict, seed: int, latency_ids: list[str], bank: dict,
                   variant: dict) -> dict:
    """variant: {"vit_mode": frozen|last_block, "use_dtw": bool, "use_temporal": bool}."""
    from ..models.mobiphysio_net import LastBlockViT, MobiPhysioNet
    from ..preprocessing.features import load_vit, vit_preprocess
    vit = load_vit(True)
    for p in vit.parameters():
        p.requires_grad_(False)
    Zc, M = b.aligned("vit_s16")
    M = torch.from_numpy(M)
    last_block = variant["vit_mode"] == "last_block"
    if last_block:
        tok, sel = b.aligned_memmap("vit_s16_lastblock_in.f16.npy")
        spatial = LastBlockViT(vit)

        def inputs(idx):
            return torch.from_numpy(np.asarray(tok[sel[idx]], dtype=np.float32))
    else:
        spatial = None
        Z = torch.from_numpy(Zc.astype(np.float32))

        def inputs(idx):
            return Z[idx]

    model = MobiPhysioNet(vit_dim=vit.num_features, num_classes=b.num_classes, d_h=cfg["d_h"],
                          layers=cfg["layers"], heads=cfg["heads"], ff=cfg["ff"], dropout=cfg["dropout"],
                          head_hidden=cfg["head_hidden"], use_dtw=variant["use_dtw"],
                          use_temporal=variant["use_temporal"], dtw_band=cfg.get("dtw_band"),
                          dtw_aggregation=cfg["dtw_aggregation"], spatial=spatial)
    pos = {v: i for i, v in enumerate(b.ids)}
    rids = reference_ids(bank)
    ridx = np.array([pos[v] for v in rids])
    if variant["use_dtw"]:
        model.set_references(inputs(ridx), M[ridx], reference_labels(bank))
    ref_id_arr = np.array(rids)

    def exclude_for(idx):
        # Leave-one-out: a training video never matches itself as a reference.
        return torch.from_numpy(b.ids[idx][:, None] == ref_id_arr[None, :])

    def forward(m, idx, train):
        return m(inputs(idx), M[idx], exclude=exclude_for(idx) if m.use_dtw else None, return_all=True)

    fit_rng = np.random.default_rng(seed)
    tr = b.idx("train")

    def refit(m):
        """Re-encode references with current weights; refit scaler on TRAIN only."""
        if not m.use_dtw:
            return
        m.refresh_reference_encodings()
        sub = tr if len(tr) <= cfg["scaler_fit_max_videos"] else np.sort(
            fit_rng.choice(tr, cfg["scaler_fit_max_videos"], replace=False))
        qs = []
        was = m.training
        m.eval()
        with torch.no_grad():
            for s in range(0, len(sub), 64):
                idx = sub[s:s + 64]
                h, _ = m.encode(inputs(idx), M[idx])
                qs.append(m.dtw_distances(h, M[idx], exclude_for(idx)))
        m.train(was)
        q = torch.cat(qs)
        if not torch.isfinite(q).all():
            raise FloatingPointError("non-finite DTW training distances")
        m.scaler.fit(q)

    out = fit(model, tr, b.idx("val"), b.labels, forward, cfg, b.num_classes, seed,
              on_epoch_end=refit, on_start=refit)
    if model.use_dtw:
        model.refresh_reference_encodings()  # buffers restored from best epoch
    collect = (lambda o: o["q"].numpy()) if model.use_dtw else None
    pt = predict(model, b.idx("test"), forward, collect=collect)
    model.eval()
    cfgv = vit.pretrained_cfg

    def lat(fr, mk):
        with torch.no_grad():
            x = vit_preprocess(fr, cfgv["mean"], cfgv["std"])
            if last_block:
                from ..preprocessing.features import vit_forward_split
                pre, _ = vit_forward_split(vit, x, run_last=False)
                z = pre.unsqueeze(0)
            else:
                z = vit(x).unsqueeze(0)
            model(z, torch.from_numpy(mk)[None])

    lat_res = _median_latency(lat, latency_ids, b.frame_cache)
    frozen_vit = n_params(vit) - (n_params(spatial) if last_block else 0)
    return _result(b, pt, out, {"latency": lat_res, "dtw_test": pt.get("extra"),
                                "params_trainable": n_params(model, True),
                                "params_total": n_params(model) + frozen_vit,
                                "model": model})
