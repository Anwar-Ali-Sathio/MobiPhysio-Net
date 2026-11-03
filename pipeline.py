"""Stage implementations behind the CLI."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config as C
from .dataset.inventory import build_inventory, load_inventory

log = logging.getLogger(__name__)


@dataclass
class Context:
    data_dir: Path
    manifest_dir: Path
    results_dir: Path
    dataset_cfg: dict
    prep_cfg: dict
    exp_cfg: dict
    overrides: dict

    @classmethod
    def create(cls, data_dir: str | None, experiment: str, overrides: dict) -> "Context":
        exp = C.deep_update(C.experiment_config(experiment), overrides.get("experiment", {}))
        prep = C.deep_update(C.load_yaml(C.CONFIG_DIR / "preprocessing.yaml"),
                             overrides.get("preprocessing", {}))
        ds = C.deep_update(C.dataset_config(), overrides.get("dataset", {}))
        return cls(C.data_dir(data_dir), C.manifest_dir(), C.results_dir(), ds, prep, exp, overrides)

    @property
    def run_root(self) -> Path:
        """Experiments are written per configuration name; pilot never mixes with full."""
        p = C.REPO_ROOT / "experiments" / self.exp_cfg["name"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def model_cfg(self, model: str) -> dict:
        return C.deep_update(C.model_config(model), self.overrides.get(model, {}))


# ----------------------------------------------------------------------------- stage 2
def cmd_inventory(ctx: Context) -> dict:
    res = build_inventory(ctx.dataset_cfg, ctx.manifest_dir)
    m = res["metadata"]
    log.info("inventory: %d files, %d videos, %.2f GB, version %s", m["n_files"],
             m["n_videos_in_inventory"], m["total_size_bytes"] / 1e9, m["version"])
    return {k: m[k] for k in ["doi", "landing_page", "version", "n_files", "n_videos_in_inventory",
                              "total_size_bytes", "n_restricted", "license"]}


def _non_video_records(inv):
    return [r for r in inv if not (r["content_type"] or "").startswith("video/")]


def cmd_download(ctx: Context, limit: int | None = None, non_video_only: bool = False) -> dict:
    """Explicit raw download (files are kept). Always fetches README + annotations."""
    from .dataset.download import download_file
    inv = load_inventory(ctx.manifest_dir)
    dl = ctx.dataset_cfg["download"]
    targets = _non_video_records(inv)
    if not non_video_only:
        vids = [r for r in inv if (r["content_type"] or "").startswith("video/")]
        targets += vids[:limit] if limit else vids
    report = []
    for r in targets:
        dest = ctx.data_dir / "raw" / r["filename"]
        st = download_file(r["download_url"], dest, r["size_bytes"], r["checksum"], r["checksum_type"],
                           timeout_s=dl["timeout_s"], max_retries=dl["max_retries"],
                           backoff_s=dl["backoff_s"], chunk_bytes=dl["chunk_bytes"])
        st["filename"] = r["filename"]
        report.append(st)
    ok = sum(s["status"] == "ok" for s in report)
    return {"requested": len(report), "ok": ok, "failed": [s for s in report if s["status"] != "ok"]}


# ----------------------------------------------------------------------------- stage 3/4
def cmd_preprocess(ctx: Context, limit: int | None = None, workers: int | None = None) -> dict:
    from .preprocessing.video import run_preprocessing
    inv = load_inventory(ctx.manifest_dir)
    dl = ctx.dataset_cfg["download"]
    cmd_download(ctx, non_video_only=True)
    t, s = ctx.prep_cfg["num_frames"], ctx.prep_cfg["image_size"]
    res = run_preprocessing(inv, ctx.data_dir, t, s, dl, workers=workers or dl["workers"], limit=limit)
    out = write_download_report(ctx)
    out["processed_this_call"] = len(res)
    return out


def write_download_report(ctx: Context) -> dict:
    """Aggregate per-video status records into download_report.json (never over-claims)."""
    from .preprocessing.cache import FrameCache
    inv = load_inventory(ctx.manifest_dir)
    cache = FrameCache(ctx.data_dir, ctx.prep_cfg["num_frames"], ctx.prep_cfg["image_size"])
    status = cache.all_status()
    vids = [r for r in inv if (r["content_type"] or "").startswith("video/")]
    rows = []
    for r in vids:
        vid = Path(r["filename"]).stem
        s = status.get(vid)
        rows.append({"video_id": vid, "file_id": r["file_id"],
                     "status": s.get("status") if s else "not_attempted",
                     "checksum_verified": bool(s and s.get("checksum_verified")),
                     "decoded": bool(s and s.get("decode_status") == "ok" and cache.has(vid)),
                     "error": (s.get("download_error") or s.get("decode_error") or "") if s else ""})
    df = pd.DataFrame(rows)
    nv = _non_video_records(inv)
    aux = []
    for r in nv:
        from .dataset.download import verify_file
        ok, why = verify_file(ctx.data_dir / "raw" / r["filename"], r["size_bytes"], r["checksum"],
                              r["checksum_type"])
        aux.append({"filename": r["filename"], "verified": ok, "detail": why})
    rep = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "stream-process: raw MP4 deleted after checksum verification + frame decoding"
                if not ctx.dataset_cfg["download"].get("keep_raw_videos") else "raw videos retained",
        "inventory_videos": len(vids),
        "attempted": int((df.status != "not_attempted").sum()),
        "downloaded_checksum_verified": int(df.checksum_verified.sum()),
        "decoded_and_cached": int(df.decoded.sum()),
        "download_failed": df[df.status == "download_failed"].video_id.tolist(),
        "decode_failed": df[df.status == "decode_failed"].video_id.tolist(),
        "worker_errors": df[df.status == "worker_error"].video_id.tolist(),
        "not_attempted": int((df.status == "not_attempted").sum()),
        "complete": bool(df.decoded.all()),
        "auxiliary_files": aux,
    }
    (ctx.manifest_dir / "download_report.json").write_text(json.dumps(rep, indent=2))
    return {k: (len(v) if isinstance(v, list) and k != "auxiliary_files" else v) for k, v in rep.items()}


# ----------------------------------------------------------------------------- helpers
def _class_codes(ctx: Context) -> list[str]:
    return sorted(ctx.dataset_cfg["exercises"])


def _class_names(ctx: Context) -> list[str]:
    return [ctx.dataset_cfg["exercises"][c] for c in _class_codes(ctx)]


def _results_root(ctx: Context) -> Path:
    """Full experiment -> results/ ; pilot -> experiments/pilot/results (never mixed)."""
    root = ctx.results_dir if not ctx.exp_cfg.get("pilot") else ctx.run_root / "results"
    for sub in ["metrics", "predictions", "figures", "reports", "logs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, obj) -> None:
    import numpy as np

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default))


def _sha(path: Path) -> str | None:
    from .training.checkpoint import sha256_file
    return sha256_file(path) if path.exists() else None


# ----------------------------------------------------------------------------- validate-data
def cmd_validate_data(ctx: Context) -> dict:
    from .dataset.labels import FilenameParser, parse_eaaq_csv
    from .dataset.manifest import build_video_manifest, find_annotation_orphans
    from .preprocessing.cache import FrameCache
    inv = load_inventory(ctx.manifest_dir)
    parser = FilenameParser.from_config(ctx.dataset_cfg)
    status = FrameCache(ctx.data_dir, ctx.prep_cfg["num_frames"], ctx.prep_cfg["image_size"]).all_status()
    ann_path = ctx.data_dir / "raw" / ctx.dataset_cfg["annotation_file"]
    eaaq, ann_summary = (None, {"available": False})
    if ann_path.exists():
        eaaq, ann_summary = parse_eaaq_csv(ann_path)
        ann_summary["available"] = True
        eaaq.to_csv(ctx.manifest_dir / "eaaq_annotations_parsed.csv", index=False)
    df = build_video_manifest(inv, parser, status, eaaq, _class_codes(ctx))
    df.to_csv(ctx.manifest_dir / "video_manifest.csv", index=False)
    if eaaq is not None:
        ann_summary["orphan_named_blocks_not_in_inventory"] = find_annotation_orphans(eaaq, set(df.video_id))
        ann_summary["videos_with_linked_score"] = int(df.eaaq_avg_score.notna().sum())
        ann_summary["videos_without_linked_score"] = int(df.eaaq_avg_score.isna().sum())
    _write_json(ctx.manifest_dir / "annotation_audit.json", ann_summary)
    out = {
        "inventory_videos": int(len(df)),
        "filename_parse_ok": int(df.parse_ok.sum()),
        "checksum_verified": int(df.checksum_verified.sum()),
        "decoded": int((df.preprocess_status == "ok").sum()),
        "usable_task_a": int(df.usable_task_a.sum()),
        "participants": int(df.participant.nunique()),
        "class_counts_inventory": df.exercise.value_counts().sort_index().to_dict(),
        "excluded": df[df.exclusion_reason != ""].groupby("exclusion_reason").size().to_dict(),
        "eaaq": {k: v for k, v in ann_summary.items() if k != "issues"},
    }
    _write_json(ctx.manifest_dir / "data_validation.json", out)
    return out


# ----------------------------------------------------------------------------- split
def cmd_split(ctx: Context) -> dict:
    import numpy as np

    from .dataset.splits import audit_split, grouped_stratified_split, split_hash
    df = pd.read_csv(ctx.manifest_dir / "video_manifest.csv")
    use = df[df.usable_task_a].copy()
    sc = ctx.exp_cfg["split"]
    res = grouped_stratified_split(use, sc["fractions"], ctx.exp_cfg["num_classes"],
                                   sc["search_trials"], sc["seed"])
    use["split"] = use.participant.map(res["participant_to_split"])
    if ctx.exp_cfg.get("max_videos"):  # pilot: subsample within splits, keep participant split
        rng = np.random.default_rng(ctx.exp_cfg["seed"])
        frac = ctx.exp_cfg["max_videos"] / len(use)
        keep = [g.sample(frac=frac, random_state=int(rng.integers(1 << 31))) for _, g in use.groupby(["split", "label"])]
        use = pd.concat(keep).sort_values("video_id")
    audit = audit_split(use)
    if audit["participant_overlap_any"] or audit["video_in_multiple_splits"] or audit["cross_split_duplicate_md5"]:
        raise RuntimeError(f"split leakage detected: {audit}")
    cols = ["video_id", "participant", "exercise", "label", "angle", "variation", "gender", "md5", "split"]
    out_csv = ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv"
    use[cols].to_csv(out_csv, index=False)
    dist = use.groupby(["split", "exercise"]).size().unstack(fill_value=0)
    info = {"experiment": ctx.exp_cfg["name"], "search": {k: v for k, v in res.items() if k != "participant_to_split"},
            "participant_to_split": res["participant_to_split"], "audit": audit,
            "class_distribution": dist.to_dict(orient="index"), "split_sha256": split_hash(use),
            "file": str(out_csv.relative_to(C.REPO_ROOT))}
    _write_json(ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}_audit.json", info)
    return {"counts": audit["counts"], "participants": {k: len(v) for k, v in audit["participants"].items()},
            "split_sha256": info["split_sha256"], "participant_overlap_any": audit["participant_overlap_any"]}


# ----------------------------------------------------------------------------- features
def cmd_extract_features(ctx: Context, backbone: str = "all") -> dict:
    import torch

    from .preprocessing import features as F
    from .preprocessing.cache import FeatureStore, FrameCache
    torch.set_num_threads(ctx.exp_cfg["training"]["num_threads"])
    sp_path = ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv"
    if not sp_path.exists():
        raise FileNotFoundError(f"{sp_path} missing - run `split` first")
    ids = sorted(pd.read_csv(sp_path).video_id)  # exactly this experiment's videos
    t, s = ctx.prep_cfg["num_frames"], ctx.prep_cfg["image_size"]
    cache, store = FrameCache(ctx.data_dir, t, s), FeatureStore(ctx.data_dir, ctx.exp_cfg["name"])
    out = {}

    def stale(name):
        if not store.exists(name):
            return True
        return not set(ids) <= set(store.load(name)["video_ids"].tolist())

    if backbone in ("all", "frames"):
        if stale("svm_descriptors"):
            out["frames112_svm"] = F.extract_small_frames_and_svm(ids, cache, store, t, ctx.prep_cfg["small_image_size"])
    if backbone in ("all", "vit_s16") and stale("vit_s16"):
        out["vit_s16"] = F.extract_vit(ids, cache, store, t)
    if backbone in ("all", "resnet50") and stale("resnet50"):
        out["resnet50"] = F.extract_resnet50(ids, cache, store, t)
    if store.exists("vit_s16"):
        out["near_duplicates"] = near_duplicate_check(ctx)
    return out or {"status": "all features up to date", "n_videos": len(ids)}


def near_duplicate_check(ctx: Context) -> dict:
    """Cross-split near-duplicate audit on mean-pooled frozen ViT embeddings."""
    from .dataset.splits import near_duplicate_audit
    from .preprocessing.cache import FeatureStore
    z = FeatureStore(ctx.data_dir, ctx.exp_cfg["name"]).load("vit_s16")
    sp = pd.read_csv(ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv")
    pos = {v: i for i, v in enumerate(z["video_ids"])}
    sel = [pos[v] for v in sp.video_id]
    f, m = z["feats"][sel].astype("float32"), z["mask"][sel]
    pooled = (f * m[..., None]).sum(1) / m.sum(1, keepdims=True).clip(1)
    res = near_duplicate_audit(pooled, sp.split.values, sp.video_id.to_numpy(dtype=str))
    import numpy as np

    from .preprocessing.cache import FrameCache
    cache = FrameCache(ctx.data_dir, ctx.prep_cfg["num_frames"], ctx.prep_cfg["image_size"])
    for pr in res["pairs_above_threshold"]:
        fa, fb = cache.load(pr["a"])[0].astype(np.int16), cache.load(pr["b"])[0].astype(np.int16)
        pr["mean_abs_pixel_diff"] = float(np.abs(fa - fb).mean())
    res["pixel_candidate_threshold"] = 2.0
    res["n_pixel_candidates"] = sum(pr["mean_abs_pixel_diff"] < 2.0 for pr in res["pairs_above_threshold"])
    review = C.REPO_ROOT / "data" / "manifests" / "near_duplicate_manual_review.json"
    res["manual_review"] = json.loads(review.read_text()) if review.exists() else None
    res["note"] = ("Embedding cosine and mean absolute pixel difference only nominate CANDIDATES: frozen ViT CLS "
                   "embeddings are highly anisotropic and many recordings share one room, camera position and "
                   "static background, and low-light videos have small pixel differences. Exact duplicates are "
                   "established by MD5 (handled in the manifest); candidates are reviewed visually.")
    _write_json(ctx.manifest_dir / f"near_duplicates_{ctx.exp_cfg['name']}.json", res)
    return {k: v for k, v in res.items() if k != "pairs_above_threshold"}



# ----------------------------------------------------------------------------- training
def load_bundle(ctx: Context):
    from .training.runners import Bundle
    sp = pd.read_csv(ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv")
    return Bundle(sp.sort_values("video_id").reset_index(drop=True), ctx.data_dir,
                  ctx.exp_cfg["num_classes"], _class_codes(ctx), ctx.prep_cfg["num_frames"],
                  ctx.prep_cfg["image_size"], ctx.exp_cfg["name"])


def _epochs(ctx, cfg):
    o = ctx.exp_cfg["training"].get("max_epochs_override")
    if o and "epochs" in cfg:
        cfg = dict(cfg, epochs=min(cfg["epochs"], o), patience=min(cfg.get("patience", 8),
                                                                    ctx.exp_cfg["training"]["early_stopping_patience"]))
    return cfg


def _registry_path(ctx) -> Path:
    return ctx.run_root / "registry.json"


def _load_registry(ctx) -> dict:
    p = _registry_path(ctx)
    return json.loads(p.read_text()) if p.exists() else {"primary": {}, "seeds": {}}


def _register(ctx, key: str, run_dir: Path, seed: int, primary: bool) -> None:
    reg = _load_registry(ctx)
    rel = str(run_dir.relative_to(C.REPO_ROOT))
    if primary:
        reg["primary"][key] = rel
    reg["seeds"].setdefault(key, {})[str(seed)] = rel
    _write_json(_registry_path(ctx), reg)


def _latency_ids(b, n):
    te = b.idx("test")
    return list(b.ids[te[:: max(1, len(te) // n)][:n]])


def train_one(ctx: Context, key: str, seed: int, variant: dict | None = None) -> dict:
    """Train + evaluate a single model (or MobiPhysio-Net variant); writes a run dir."""
    import torch

    from .evaluation.metrics import classification_metrics
    from .evaluation.predictions import save_predictions
    from .reporting.environment import environment_info
    from .training import runners as R
    from .training.checkpoint import save_checkpoint
    from .training.reproducibility import new_run_id, set_seed, utc_now
    torch.set_num_threads(ctx.exp_cfg["training"]["num_threads"])
    model_name = "mobiphysio_net" if key.startswith("mobiphysio_net") else key
    cfg = _epochs(ctx, ctx.model_cfg(model_name))
    if variant and variant.get("vit_mode") == "last_block":
        cfg = _epochs(ctx, C.deep_update(cfg, cfg.get("finetune_overrides", {})))
    run_id = new_run_id(key)
    run_dir = ctx.run_root / key / run_id
    run_dir.mkdir(parents=True)
    started = utc_now()
    seed_info = set_seed(seed)
    b = load_bundle(ctx)
    lat_ids = _latency_ids(b, ctx.exp_cfg["evaluation"]["latency_videos"])
    manifest = {"run_id": run_id, "model": key, "experiment": ctx.exp_cfg["name"],
                "pilot": bool(ctx.exp_cfg.get("pilot")), "started_utc": started, **seed_info,
                "config": cfg, "variant": variant, "task": "A: 9-class exercise recognition",
                "dataset": json.loads((ctx.manifest_dir / "dataset_metadata.json").read_text())["doi"],
                "dataset_version": json.loads((ctx.manifest_dir / "dataset_metadata.json").read_text())["version"],
                "inventory_sha256": _sha(ctx.manifest_dir / "dataverse_inventory.json"),
                "video_manifest_sha256": _sha(ctx.manifest_dir / "video_manifest.csv"),
                "split_file_sha256": _sha(ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv"),
                "split_counts": {s: int(len(b.idx(s))) for s in ["train", "val", "test"]},
                "environment": environment_info(str(ctx.data_dir)),
                "stages": {}}
    _write_json(run_dir / "run_manifest.json", manifest | {"exit_status": "running"})
    try:
        nc = b.num_classes
        model_obj = None
        if key == "svm":
            res = R.run_svm(b, cfg, seed, lat_ids)
        elif key in ("lstm", "bilstm", "resnet50"):
            from .models.bilstm import BiLSTMClassifier
            from .models.lstm import LSTMClassifier
            from .models.resnet50 import ResNet50Head, resnet50_backbone
            kw = dict(in_dim=2048, proj_dim=cfg.get("proj_dim", 256), hidden=cfg.get("hidden", 256),
                      layers=cfg.get("layers", 1), num_classes=nc, dropout=cfg["dropout"])
            model_obj = {"lstm": lambda: LSTMClassifier(**kw), "bilstm": lambda: BiLSTMClassifier(**kw),
                         "resnet50": lambda: ResNet50Head(2048, nc, cfg["dropout"])}[key]()
            res = R.run_feature_model(b, model_obj, "resnet50", cfg, seed, lat_ids, resnet50_backbone(True),
                                      ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif key in ("cnn2d", "cnn_lstm"):
            from .models.cnn2d import CNN2D
            from .models.cnn_lstm import CNNLSTM
            model_obj = (CNN2D(nc, tuple(cfg["widths"]), cfg["feat_dim"], cfg["dropout"]) if key == "cnn2d"
                         else CNNLSTM(nc, tuple(cfg["widths"]), cfg["feat_dim"], cfg["hidden"], cfg["dropout"]))
            res = R.run_scratch_cnn(b, model_obj, cfg, seed, lat_ids)
        else:
            bank_path = ctx.run_root / "reference_bank.json"
            bank = R.build_or_load_reference_bank(b, cfg, ctx.exp_cfg["seed"], bank_path)
            manifest["reference_bank"] = {"path": str(bank_path.relative_to(C.REPO_ROOT)),
                                          "sha256": _sha(bank_path),
                                          "reference_ids_sha256": bank["reference_ids_sha256"],
                                          "training_ids_sha256": bank["training_ids_sha256"],
                                          "provenance": "training split only (verified)"}
            variant = variant or {k: v for k, v in cfg["ablations"]["full"].items()}
            manifest["variant"] = variant
            res = R.run_mobiphysio(b, cfg, seed, lat_ids, bank, variant)
            model_obj = res.pop("model")
        manifest["stages"]["train"] = "completed"
        te = res["test_idx"]
        m = classification_metrics(res["y_true"], res["pred"], res["proba"], nc, _class_names(ctx))
        save_predictions(run_dir / "test_predictions.csv", b.ids[te], b.manifest.participant.values[te],
                         res["y_true"], res["pred"], res["proba"], _class_codes(ctx), res.get("dtw_test"),
                         extra={"angle": b.manifest.angle.values[te], "variation": b.manifest.variation.values[te]})
        manifest["stages"]["evaluate"] = "completed"
        fit_out = res.get("fit") or {}
        _write_json(run_dir / "history.json", fit_out)
        _write_json(run_dir / "metrics.json", m | {"latency": res["latency"],
                                                    "params_trainable": res["params_trainable"],
                                                    "params_total": res["params_total"],
                                                    "notes": res.get("notes", {})})
        outputs = {"test_predictions.csv": _sha(run_dir / "test_predictions.csv"),
                   "metrics.json": _sha(run_dir / "metrics.json"), "history.json": _sha(run_dir / "history.json")}
        if model_obj is not None:
            ck = ctx.data_dir / "checkpoints" / key / f"{run_id}.pt"
            outputs["checkpoint"] = {"path": str(ck), "sha256": save_checkpoint(ck, model_obj, {"run_id": run_id, "config": cfg, "variant": variant})}
        manifest.update(exit_status="completed", ended_utc=utc_now(), outputs=outputs,
                        best_epoch=fit_out.get("best_epoch"), best_val_macro_f1=fit_out.get("best_val_macro_f1"),
                        selection_rule=fit_out.get("selection_rule", "max validation macro F1"))
        _write_json(run_dir / "run_manifest.json", manifest)
        _register(ctx, key, run_dir, seed, primary=(seed == ctx.exp_cfg["seed"]))
        log.info("%s seed=%d test macroF1=%.4f acc=%.4f", key, seed, m["macro_f1"], m["accuracy"])
        return {"run_id": run_id, "run_dir": str(run_dir), "test_macro_f1": m["macro_f1"],
                "test_accuracy": m["accuracy"], "best_epoch": fit_out.get("best_epoch")}
    except Exception as exc:
        import traceback
        manifest.update(exit_status="failed", ended_utc=utc_now(), error=repr(exc),
                        traceback=traceback.format_exc())
        _write_json(run_dir / "run_manifest.json", manifest)
        log.exception("run %s failed", run_id)
        return {"run_id": run_id, "exit_status": "failed", "error": repr(exc)}


ABLATION_KEYS = {"no_dtw": "mobiphysio_net__no_dtw", "dtw_no_temporal": "mobiphysio_net__dtw_no_temporal",
                 "finetuned_vit": "mobiphysio_net__finetuned_vit"}


def cmd_train(ctx: Context, model: str, seed: int | None = None) -> dict:
    seed = ctx.exp_cfg["seed"] if seed is None else seed
    if model == "ablations":
        abl = ctx.model_cfg("mobiphysio_net")["ablations"]
        return {k: train_one(ctx, key, seed, abl[k]) for k, key in ABLATION_KEYS.items()}
    return train_one(ctx, model, seed)


# ----------------------------------------------------------------------------- evaluate / compare
DEFAULT_FINAL = ["svm", "cnn2d", "lstm", "cnn_lstm", "resnet50", "mobiphysio_net"]
DEFAULT_ABLATIONS = ["mobiphysio_net__no_dtw", "mobiphysio_net__dtw_no_temporal", "mobiphysio_net__finetuned_vit"]


def final_models(ctx: Context) -> list[str]:
    return list(ctx.exp_cfg.get("final_comparison_models", DEFAULT_FINAL))


def ablation_keys(ctx: Context) -> list[str]:
    return list(ctx.exp_cfg.get("ablation_keys", DEFAULT_ABLATIONS))


def historical_models(ctx: Context) -> list[str]:
    """Models trained earlier but outside the final scope (decision log); supplementary only."""
    return list(ctx.exp_cfg.get("historical_models", []))


def cmd_evaluate(ctx: Context) -> dict:
    from .evaluation.comparison import ablation_table, evaluate_all
    res_root = _results_root(ctx)
    ev = ctx.exp_cfg["evaluation"]
    keys = final_models(ctx) + ablation_keys(ctx)
    out = evaluate_all(C.REPO_ROOT, ctx.run_root, res_root, _class_codes(ctx), _class_names(ctx),
                       ev["bootstrap_iterations"], ev["bootstrap_seed"], keys)
    comp = out["comparison"]
    md = res_root / "metrics"
    prim = comp[comp.key.isin(final_models(ctx))]
    prim.to_csv(md / "model_comparison.csv", index=False)
    _write_json(md / "model_comparison.json", prim.to_dict(orient="records"))
    comp.to_csv(md / "all_runs_comparison.csv", index=False)
    out["per_class"].to_csv(md / "per_class_metrics.csv", index=False)
    out["stratified"].to_csv(md / "metrics_by_angle_variation.csv", index=False)
    if not out["seeds"].empty:
        out["seeds"].to_csv(md / "seed_variability.csv", index=False)
    abl = ablation_table(out["predictions"], comp, ev["bootstrap_iterations"], ev["bootstrap_seed"],
                         ctx.exp_cfg["num_classes"])
    if not abl.empty:
        abl.to_csv(md / "ablation_comparison.csv", index=False)
    sp = pd.read_csv(ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv")
    dist = sp.groupby(["exercise", "split"]).size().unstack(fill_value=0)
    dist["total"] = dist.sum(1)
    dist.insert(0, "exercise_name", [ctx.dataset_cfg["exercises"][e] for e in dist.index])
    dist.to_csv(md / "class_distribution.csv")
    summary = {"experiment": ctx.exp_cfg["name"], "pilot": bool(ctx.exp_cfg.get("pilot")),
               "models": {r.key: r.status for r in comp.itertuples()},
               "test_videos": int((sp.split == "test").sum()),
               "test_participants": int(sp[sp.split == "test"].participant.nunique()),
               "bootstrap": f"participant cluster bootstrap, {ev['bootstrap_iterations']} iterations"}
    hist = historical_models(ctx)
    if hist:
        # Historical (out-of-scope) models: evaluated identically, written ONLY to results/supplementary/.
        sup = res_root / "supplementary"
        (sup / "predictions").mkdir(parents=True, exist_ok=True)
        ho = evaluate_all(C.REPO_ROOT, ctx.run_root, sup, _class_codes(ctx), _class_names(ctx),
                          ev["bootstrap_iterations"], ev["bootstrap_seed"], hist)
        ho["comparison"].to_csv(sup / "historical_models_comparison.csv", index=False)
        if not ho["seeds"].empty:
            ho["seeds"].to_csv(sup / "historical_models_seed_variability.csv", index=False)
        ho["per_class"].to_csv(sup / "historical_models_per_class.csv", index=False)
        (sup / "README.md").write_text(
            "# Supplementary: historical models outside the final comparison scope\n\n"
            "These models were trained and evaluated with the same split, test videos and code as the final "
            "models, then removed from the final comparison scope by decision D4 "
            "(`reports/experiment_decision_log.md`). They are reported here for transparency and are "
            "never included in the main tables or figures.\n")
        summary["historical_models_supplementary"] = {r.key: r.status for r in ho["comparison"].itertuples()}
    summary["final_comparison_models"] = final_models(ctx)
    _write_json(md / "evaluation_summary.json", summary)
    return summary


def cmd_compare(ctx: Context) -> dict:
    from .evaluation import plots
    from .reporting.report import write_reports
    if not (_results_root(ctx) / "metrics" / "all_runs_comparison.csv").exists():
        cmd_evaluate(ctx)
    res_root = _results_root(ctx)
    md, fd = res_root / "metrics", res_root / "figures"
    comp = pd.read_csv(md / "all_runs_comparison.csv")
    prim = pd.read_csv(md / "model_comparison.csv")
    suffix = " - PILOT" if ctx.exp_cfg.get("pilot") else ""
    plots.model_comparison(prim, fd / "model_comparison.png", suffix)
    plots.confusion_matrices(prim, _class_codes(ctx), fd / "confusion_matrices.png")
    pc = pd.read_csv(md / "per_class_metrics.csv")
    plots.per_class_f1(pc[pc.key.isin(prim.key)], _class_codes(ctx), fd / "per_class_f1.png")
    reg = _load_registry(ctx)
    from .evaluation.comparison import DISPLAY
    hist = {}
    for k in final_models(ctx) + ablation_keys(ctx):
        if k in reg["primary"] and k != "svm":
            h = json.loads((C.REPO_ROOT / reg["primary"][k] / "history.json").read_text())
            hist[DISPLAY[k]] = h.get("history", [])
    plots.training_curves(hist, fd / "training_curves.png")
    if (md / "ablation_comparison.csv").exists():
        plots.dtw_ablation(pd.read_csv(md / "ablation_comparison.csv"), fd / "dtw_ablation.png")
    reports = write_reports(ctx, res_root)
    return {"figures": sorted(p.name for p in fd.glob("*.png")), "reports": reports,
            "completed": int(comp.status.str.startswith("COMPLETED").sum())}


# ----------------------------------------------------------------------------- task B
def cmd_task_b(ctx: Context) -> dict:
    from .preprocessing.cache import FeatureStore
    from .tasks.eaaq import run_task_b
    res_root = _results_root(ctx)
    ann = ctx.manifest_dir / "eaaq_annotations_parsed.csv"
    if not ann.exists():
        out = {"status": "BLOCKED", "reason": "annotation CSV not parsed / unavailable"}
        _write_json(res_root / "metrics" / "task_b_eaaq.json", out)
        return out
    eaaq = pd.read_csv(ann, keep_default_na=False, na_values=[""])
    eaaq["video_name"] = eaaq.video_name.fillna("")
    sp = pd.read_csv(ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv")
    vit = FeatureStore(ctx.data_dir, ctx.exp_cfg["name"]).load("vit_s16")
    r = run_task_b(sp, eaaq, vit, ctx.exp_cfg["seed"])
    r["metrics"].to_csv(res_root / "metrics" / "task_b_eaaq_metrics.csv", index=False)
    r["predictions"].to_csv(res_root / "predictions" / "task_b_eaaq_test_predictions.csv", index=False)
    r["score_by_variation"].to_csv(res_root / "metrics" / "task_b_score_by_variation.csv", index=False)
    out = {"status": "COMPLETED (exploratory)", "n": r["n"], "ridge_alpha": r["ridge_alpha"],
           "inter_judge_icc2_1": r["inter_judge_icc2_1"], "mean_judge_sd": r["mean_judge_sd"],
           "score_summary": r["score_summary"], "metrics": r["metrics"].to_dict(orient="records"),
           "target": "per-video mean of 3 physiotherapists' EAAQ scores (0-100), explicit-name linkage only"}
    _write_json(res_root / "metrics" / "task_b_eaaq.json", out)
    return out


# ----------------------------------------------------------------------------- verify
def cmd_verify(ctx: Context) -> dict:
    """Recompute every reported metric from the saved predictions; audit provenance."""
    import numpy as np

    from .evaluation.metrics import classification_metrics
    from .evaluation.predictions import load_predictions
    from .models.references import verify_reference_provenance
    res_root = _results_root(ctx)
    checks, problems = [], []
    reg = _load_registry(ctx)
    sp = pd.read_csv(ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}.csv")
    test_ids = set(sp.loc[sp.split == "test", "video_id"])
    train_ids = set(sp.loc[sp.split == "train", "video_id"])
    held = set(sp.loc[sp.split != "train", "video_id"])
    comp_path = res_root / "metrics" / "all_runs_comparison.csv"
    comp = pd.read_csv(comp_path).set_index("key") if comp_path.exists() else pd.DataFrame()
    hist_path = res_root / "supplementary" / "historical_models_comparison.csv"
    if hist_path.exists():
        comp = pd.concat([comp, pd.read_csv(hist_path).set_index("key")])
    for key, rel in reg["primary"].items():
        run_dir = C.REPO_ROOT / rel
        man = json.loads((run_dir / "run_manifest.json").read_text())
        df, proba = load_predictions(run_dir / "test_predictions.csv", _class_codes(ctx))
        m = classification_metrics(df.y_true, df.y_pred, proba, ctx.exp_cfg["num_classes"])
        stored = json.loads((run_dir / "metrics.json").read_text())
        ok_metrics = all(abs(m[k] - stored[k]) < 1e-9 for k in ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1"])
        ok_table = key in comp.index and abs(comp.loc[key, "macro_f1"] - m["macro_f1"]) < 1e-9
        ok_set = set(df.video_id) == test_ids
        ok_labels = (df.set_index("video_id").y_true.reindex(sp.set_index("video_id").loc[list(df.video_id)].index).values
                     == sp.set_index("video_id").loc[list(df.video_id), "label"].values).all()
        ok_argmax = proba is None or bool((np.asarray(proba).argmax(1) == df.y_pred.values).all())
        ok_hash = all((_sha(run_dir / f) == h) for f, h in man.get("outputs", {}).items() if f.endswith((".csv", ".json")))
        ok_ckpt = True
        ck = man.get("outputs", {}).get("checkpoint")
        if ck:
            ok_ckpt = Path(ck["path"]).exists() and _sha(Path(ck["path"])) == ck["sha256"]
        ok_leak = True
        if "reference_bank" in man:
            bank = json.loads((C.REPO_ROOT / man["reference_bank"]["path"]).read_text())
            try:
                verify_reference_provenance(bank, train_ids, held)
            except Exception as exc:  # noqa: BLE001
                ok_leak = False
                problems.append(f"{key}: {exc}")
        row = {"key": key, "run_id": man["run_id"], "metrics_recomputed_match": ok_metrics,
               "comparison_table_match": bool(ok_table), "test_set_identical": ok_set,
               "labels_match_split_file": bool(ok_labels), "pred_is_argmax_proba": ok_argmax,
               "output_hashes_match": ok_hash, "checkpoint_hash_match": ok_ckpt,
               "reference_bank_training_only": ok_leak, "exit_status": man["exit_status"]}
        checks.append(row)
        for k, v in row.items():
            if v is False:
                problems.append(f"{key}: {k} failed")
    audit = json.loads((ctx.manifest_dir / f"splits_{ctx.exp_cfg['name']}_audit.json").read_text())
    if audit["audit"]["participant_overlap_any"]:
        problems.append("participant overlap between splits")
    cited = ["metrics/model_comparison.csv", "metrics/model_comparison.json", "metrics/per_class_metrics.csv",
             "metrics/class_distribution.csv", "metrics/evaluation_summary.json"]
    missing = [c for c in cited if not (res_root / c).exists()]
    out = {"checks": checks, "problems": problems, "missing_outputs": missing,
           "participant_overlap": audit["audit"]["participant_overlap_any"],
           "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "exit_status": "passed" if not problems and not missing else "failed"}
    _write_json(res_root / "reports" / "verification.json", out)
    return out


# ----------------------------------------------------------------------------- run-all
def mark_interrupted_runs(ctx: Context) -> list[str]:
    """Run dirs left in 'running' state by a killed process -> 'interrupted' (never deleted)."""
    out = []
    for m in ctx.run_root.glob("*/*/run_manifest.json"):
        j = json.loads(m.read_text())
        if j.get("exit_status") == "running":
            j.update(exit_status="interrupted", note="process killed (container restart); run not used")
            _write_json(m, j)
            out.append(j["run_id"])
    return out


def _done(ctx: Context, key: str, seed: int) -> bool:
    return str(seed) in _load_registry(ctx).get("seeds", {}).get(key, {})


def cmd_run_all(ctx: Context, skip_preprocess: bool = False, resume: bool = False) -> dict:
    from .reporting.environment import environment_info
    from .training.reproducibility import new_run_id, utc_now
    res_root = _results_root(ctx)
    man = {"run_id": new_run_id(f"runall-{ctx.exp_cfg['name']}"), "started_utc": utc_now(),
           "experiment": ctx.exp_cfg["name"], "pilot": bool(ctx.exp_cfg.get("pilot")),
           "environment": environment_info(str(ctx.data_dir)), "stages": {}}
    if resume:
        prev = res_root / "run_manifest.json"
        if prev.exists():
            old = json.loads(prev.read_text())
            man["resumed_from"] = {"run_id": old.get("run_id"), "started_utc": old.get("started_utc"),
                                   "stages": old.get("stages")}
        man["interrupted_runs"] = mark_interrupted_runs(ctx)
    seed0 = ctx.exp_cfg["seed"]

    def train_stage(key, seed, fn):
        if resume and _done(ctx, key, seed):
            return {"skipped": "already completed (resume)"}
        return fn()
    stages = []
    if not skip_preprocess:
        stages.append(("inventory", lambda: cmd_inventory(ctx)))
        stages.append(("preprocess", lambda: cmd_preprocess(ctx)))
    stages += [("validate-data", lambda: cmd_validate_data(ctx)), ("split", lambda: cmd_split(ctx)),
               ("extract-features", lambda: cmd_extract_features(ctx))]
    for m in final_models(ctx):
        stages.append((f"train:{m}", lambda m=m: train_stage(m, seed0, lambda: cmd_train(ctx, m))))
    abl_cfg = ctx.model_cfg("mobiphysio_net")["ablations"]
    for k, key in ABLATION_KEYS.items():
        stages.append((f"train:{k}", lambda k=k, key=key: train_stage(key, seed0, lambda: train_one(ctx, key, seed0, abl_cfg[k]))))
    ev = ctx.exp_cfg["evaluation"]
    seed_models = ev.get("extra_seed_models", final_models(ctx))
    seed_ablations = ev.get("extra_seed_ablations", ["no_dtw", "dtw_no_temporal"])
    for s in ev.get("extra_training_seeds", []):
        for m in seed_models:
            stages.append((f"train:{m}:seed{s}", lambda m=m, s=s: train_stage(m, s, lambda: cmd_train(ctx, m, s))))
        for k in seed_ablations:
            abl = ctx.model_cfg("mobiphysio_net")["ablations"][k]
            stages.append((f"train:{k}:seed{s}", lambda k=k, s=s, abl=abl: train_stage(
                ABLATION_KEYS[k], s, lambda: train_one(ctx, ABLATION_KEYS[k], s, abl))))
    stages += [("task-b", lambda: cmd_task_b(ctx)), ("evaluate", lambda: cmd_evaluate(ctx)),
               ("compare", lambda: cmd_compare(ctx)), ("verify-results", lambda: cmd_verify(ctx))]
    for name, fn in stages:
        t0 = time.time()
        try:
            r = fn()
            failed = isinstance(r, dict) and (r.get("exit_status") == "failed" or any(
                isinstance(v, dict) and v.get("exit_status") == "failed" for v in r.values()))
            man["stages"][name] = {"status": "failed" if failed else "completed", "seconds": round(time.time() - t0, 1)}
            if isinstance(r, dict) and r.get("skipped"):
                man["stages"][name]["note"] = r["skipped"]
        except Exception as exc:  # noqa: BLE001
            log.exception("stage %s failed", name)
            man["stages"][name] = {"status": "failed", "error": repr(exc), "seconds": round(time.time() - t0, 1)}
        _write_json(res_root / "run_manifest.json", man)
    man["ended_utc"] = utc_now()
    man["completed_stages"] = [k for k, v in man["stages"].items() if v["status"] == "completed"]
    man["failed_stages"] = [k for k, v in man["stages"].items() if v["status"] != "completed"]
    man["exit_status"] = "completed" if not man["failed_stages"] else "completed_with_failures"
    _write_json(res_root / "run_manifest.json", man)
    return {k: man[k] for k in ["run_id", "exit_status", "completed_stages", "failed_stages"]}
