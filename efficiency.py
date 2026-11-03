"""Controlled efficiency benchmark (replaces the in-run latency measurements).

Why: in-run latency was timed inside each training process on a shared VM (30 videos) and is not
comparable across runs (e.g. the same no-DTW configuration logged 512 ms and 765 ms). Here every model
is timed under identical conditions in its own fresh subprocess:

* input: cached 16 x 224 x 224 RGB uint8 frames of the same N test videos (seed-42 checkpoints);
* warm-up videos are discarded; median and p95 over the remaining videos; batch = 1 video;
* torch thread count fixed per measurement (4 = all cores, 1 = constrained setting);
* per-component wall time with ``time.perf_counter``: preprocessing (normalise/resize/descriptor),
  backbone, temporal encoder, DTW (query vs the cached reference encodings), head;
* peak resident memory of the subprocess (``ru_maxrss``) and of an idle baseline process that only
  imports torch + the package;
* MP4 decoding is NOT included in model latency (it is identical for all models); it is measured
  separately by :func:`decode_benchmark` on freshly downloaded, checksum-verified test videos.

The DTW reference *bank construction* (k-medoids) is a one-off training step and is reported separately
from the build time recorded in ``reference_bank.json``; reference *encoding* is computed once when a
model is loaded (cached), as in deployment, and reported as a load-time cost.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import torch


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t.setdefault(k, []).append(dt)


def _norm(fr, mean, std):
    x = torch.from_numpy(fr).float().div_(255).permute(0, 3, 1, 2)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def build_runner(key: str, run_dir: Path, repo: Path, data_dir: Path, bundle):
    """Return (infer(frames, mask, timer) -> probs, info) for one registered seed-42 run."""
    from ..models.references import reference_labels
    from ..training.checkpoint import load_checkpoint
    man = json.loads((run_dir / "run_manifest.json").read_text())
    cfg = man["config"]
    info = {"run_id": man["run_id"]}
    ck = man.get("outputs", {}).get("checkpoint")
    if ck:
        info["checkpoint_bytes"] = Path(ck["path"]).stat().st_size
    nc = 9
    if key == "svm":
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        from ..models.svm import video_descriptor
        hist = json.loads((run_dir / "history.json").read_text())
        X, _ = bundle.aligned("svm_descriptors")
        tr = bundle.idx("train")
        sc = StandardScaler().fit(X[tr])
        t0 = time.time()
        svm = SVC(C=hist["best"]["C"], gamma=hist["best"]["gamma_mult"] / X.shape[1], kernel="rbf",
                  class_weight="balanced", probability=True, random_state=man["seed"]).fit(sc.transform(X[tr]), bundle.labels[tr])
        info["refit_seconds"] = time.time() - t0
        te = bundle.idx("test")
        import pandas as pd
        saved = pd.read_csv(run_dir / "test_predictions.csv")
        refit_pred = svm.predict_proba(sc.transform(X[te])).argmax(1)
        info["refit_reproduces_saved_predictions"] = float((refit_pred == saved.y_pred.values).mean())
        info["model_values"] = int(svm.support_vectors_.size + svm.dual_coef_.size)
        info["n_support_vectors"] = int(svm.support_vectors_.shape[0])

        def infer(fr, mk, T):
            t = time.perf_counter()
            d = video_descriptor(fr, mk)
            T.add("preprocess+descriptor", time.perf_counter() - t)
            t = time.perf_counter()
            p = svm.predict_proba(sc.transform(d[None]))
            T.add("head", time.perf_counter() - t)
            return p
        return infer, info

    if key in ("lstm", "resnet50", "bilstm"):
        from ..models.lstm import LSTMClassifier
        from ..models.resnet50 import ResNet50Head, resnet50_backbone
        backbone = resnet50_backbone(True)
        if key == "resnet50":
            m = ResNet50Head(2048, nc, cfg["dropout"])
        else:
            from ..models.bilstm import BiLSTMClassifier
            kw = dict(in_dim=2048, proj_dim=cfg.get("proj_dim", 256), hidden=cfg.get("hidden", 256),
                      layers=cfg.get("layers", 1), num_classes=nc, dropout=cfg["dropout"])
            m = LSTMClassifier(**kw) if key == "lstm" else BiLSTMClassifier(**kw)
        load_checkpoint(Path(ck["path"]), m)
        m.eval()
        info["params_trainable"] = sum(p.numel() for p in m.parameters())
        info["params_frozen_backbone"] = sum(p.numel() for p in backbone.parameters())
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        def infer(fr, mk, T):
            with torch.no_grad():
                t = time.perf_counter()
                x = _norm(fr, mean, std)
                T.add("preprocess", time.perf_counter() - t)
                t = time.perf_counter()
                f = backbone(x).unsqueeze(0)
                T.add("backbone", time.perf_counter() - t)
                t = time.perf_counter()
                p = torch.softmax(m(f, torch.from_numpy(mk)[None]), 1)
                T.add("temporal+head", time.perf_counter() - t)
            return p
        return infer, info

    if key in ("cnn2d", "cnn_lstm"):
        import cv2

        from ..models.cnn2d import CNN2D
        from ..models.cnn_lstm import CNNLSTM
        from ..training.runners import SCRATCH_MEAN, SCRATCH_STD
        m = (CNN2D(nc, tuple(cfg["widths"]), cfg["feat_dim"], cfg["dropout"]) if key == "cnn2d"
             else CNNLSTM(nc, tuple(cfg["widths"]), cfg["feat_dim"], cfg["hidden"], cfg["dropout"]))
        load_checkpoint(Path(ck["path"]), m)
        m.eval()
        info["params_trainable"] = sum(p.numel() for p in m.parameters())
        info["params_frozen_backbone"] = 0

        def infer(fr, mk, T):
            with torch.no_grad():
                t = time.perf_counter()
                small = np.stack([cv2.resize(f, (112, 112), interpolation=cv2.INTER_AREA) for f in fr])
                x = _norm(small, SCRATCH_MEAN, SCRATCH_STD).unsqueeze(0)
                T.add("preprocess", time.perf_counter() - t)
                t = time.perf_counter()
                p = torch.softmax(m(x, torch.from_numpy(mk)[None]), 1)
                T.add("cnn(+lstm)+head", time.perf_counter() - t)
            return p
        return infer, info

    # MobiPhysio-Net variants
    from ..models.dtw import dtw_batch
    from ..models.mobiphysio_net import LastBlockViT, MobiPhysioNet
    from ..models.references import aggregate_class_distances
    from ..preprocessing.features import load_vit, vit_forward_split, vit_preprocess
    # Primary runs recorded variant=None (default applied after the manifest was written); the stored
    # config's ablations["full"] is that default.
    variant = man["variant"] or cfg["ablations"]["full"]
    vit = load_vit(True)
    last_block = variant["vit_mode"] == "last_block"
    spatial = LastBlockViT(vit) if last_block else None
    m = MobiPhysioNet(vit_dim=vit.num_features, num_classes=nc, d_h=cfg["d_h"], layers=cfg["layers"],
                      heads=cfg["heads"], ff=cfg["ff"], dropout=cfg["dropout"], head_hidden=cfg["head_hidden"],
                      use_dtw=variant["use_dtw"], use_temporal=variant["use_temporal"],
                      dtw_band=cfg.get("dtw_band"), dtw_aggregation=cfg["dtw_aggregation"], spatial=spatial)
    sd = torch.load(ck["path"], map_location="cpu", weights_only=False)["state_dict"]
    m.ref_inputs = torch.zeros_like(sd["ref_inputs"])
    m.ref_mask = torch.zeros_like(sd["ref_mask"])
    m.load_state_dict(sd)
    m.eval()
    if variant["use_dtw"]:
        bank = json.loads((run_dir.parents[1] / "reference_bank.json").read_text())
        m.ref_labels = reference_labels(bank)
        t = time.perf_counter()
        m.refresh_reference_encodings()
        info["load_time_reference_encoding_s"] = time.perf_counter() - t
        info["n_references"] = int(m.ref_mask.shape[0])
    info["params_trainable"] = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("spatial."))
    info["params_trainable"] += sum(p.numel() for p in spatial.parameters()) if spatial is not None else 0
    info["params_frozen_backbone"] = sum(p.numel() for p in vit.parameters()) - (
        sum(p.numel() for p in spatial.parameters()) if spatial is not None else 0)
    cfgv = vit.pretrained_cfg

    def infer(fr, mk, T):
        with torch.no_grad():
            mask = torch.from_numpy(mk)[None]
            t = time.perf_counter()
            x = vit_preprocess(fr, cfgv["mean"], cfgv["std"])
            T.add("preprocess", time.perf_counter() - t)
            t = time.perf_counter()
            if last_block:
                pre, _ = vit_forward_split(vit, x, run_last=False)
                z = m.spatial(pre.unsqueeze(0))
            else:
                z = vit(x).unsqueeze(0)
            T.add("backbone (ViT)", time.perf_counter() - t)
            t = time.perf_counter()
            if m.use_temporal:
                h, g = m.temporal(z, mask)
            else:
                h = torch.nn.functional.normalize(z, dim=-1) * mask.unsqueeze(-1).float()
                from ..models.mobiphysio_net import masked_mean
                g = masked_mean(z, mask)
            T.add("temporal encoder", time.perf_counter() - t)
            if m.use_dtw:
                t = time.perf_counter()
                d = dtw_batch(h, mask, m._ref_h_cache, m.ref_mask, m.dtw_band)
                q = aggregate_class_distances(d, m.ref_labels, nc, m.dtw_aggregation)
                u = torch.cat([g, m.scaler(q)], 1)
                T.add("DTW (27 refs) + scaling", time.perf_counter() - t)
            else:
                u = g
            t = time.perf_counter()
            p = torch.softmax(m.head(u), 1)
            T.add("head", time.perf_counter() - t)
        return p
    return infer, info


def benchmark_one(key: str, threads: int, n_videos: int, warmup: int, repo: Path, exp: str) -> dict:
    """Runs inside a fresh subprocess (see scripts/benchmark_efficiency.py)."""
    from .. import pipeline as P
    torch.set_num_threads(threads)
    ctx = P.Context.create(None, "experiment", {})
    b = P.load_bundle(ctx)
    reg = json.loads((ctx.run_root / "registry.json").read_text())
    run_dir = repo / reg["seeds"][key][str(ctx.exp_cfg["seed"])]
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    infer, info = build_runner(key, run_dir, repo, ctx.data_dir, b)
    info["model_load_s"] = time.perf_counter() - t0
    te = b.idx("test")
    ids = list(b.ids[te[:: max(1, len(te) // (n_videos + warmup))][: n_videos + warmup]])
    cache = b.frame_cache
    T, total, preds = Timer(), [], []
    for i, v in enumerate(ids):
        fr, mk = cache.load(v)
        Ti = Timer()
        t = time.perf_counter()
        p = infer(fr, mk, Ti)
        dt = time.perf_counter() - t
        if i >= warmup:
            total.append(dt)
            for k, vv in Ti.t.items():
                T.add(k, sum(vv))
        preds.append(int(np.asarray(p).argmax()))
    comp = {k: {"median_ms": float(np.median(v) * 1e3), "p95_ms": float(np.quantile(v, 0.95) * 1e3)} for k, v in T.t.items()}
    return {"key": key, "threads": threads, "n_timed": len(total), "warmup": warmup,
            "median_ms": float(np.median(total) * 1e3), "p95_ms": float(np.quantile(total, 0.95) * 1e3),
            "mean_ms": float(np.mean(total) * 1e3), "components": comp,
            "peak_rss_mb": _rss_mb(), "rss_before_model_mb": rss0, **info,
            "video_ids": ids[warmup:], "timing_definition":
                "cached 16x224x224 uint8 frames -> class probabilities; batch=1; excludes MP4 decoding"}


def decode_benchmark(records: list[dict], workdir: Path, t: int = 16, size: int = 224) -> dict:
    """Download (MD5-verified) a few test videos and time single-process PyAV decoding (1 thread)."""
    import os

    from ..dataset.download import download_file
    from ..preprocessing.frames import read_sampled_frames_pyav
    out = []
    workdir.mkdir(parents=True, exist_ok=True)
    for r in records:
        dest = workdir / r["filename"]
        st = download_file(r["download_url"], dest, r["size_bytes"], r["checksum"], r["checksum_type"])
        if st["status"] != "ok":
            out.append({"video": r["filename"], "error": st["error"]})
            continue
        t0 = time.perf_counter()
        res = read_sampled_frames_pyav(str(dest), t, size)
        dt = time.perf_counter() - t0
        out.append({"video": r["filename"], "decode_s": dt, "ok": res["ok"], "size_mb": r["size_bytes"] / 1e6,
                    "width": res.get("width"), "height": res.get("height")})
        os.remove(dest)
    ds = [o["decode_s"] for o in out if "decode_s" in o]
    return {"videos": out, "median_s": float(np.median(ds)) if ds else None,
            "p95_s": float(np.quantile(ds, 0.95)) if ds else None,
            "definition": "PyAV accurate seek (thread_type=AUTO), 16 frames, letterbox to 224, one process; download time excluded"}
