"""Command-line interface: ``python -m mobiphysio.cli <command> [--help]``."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config as C

MODELS = ["svm", "cnn2d", "lstm", "bilstm", "cnn_lstm", "resnet50", "mobiphysio_net"]


def _setup_logging(verbose: bool = False) -> None:
    log_dir = C.results_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stderr),
                logging.FileHandler(log_dir / "execution.log", encoding="utf-8")]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt,
                        handlers=handlers, force=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mobiphysio", description=__doc__)
    p.add_argument("--data-dir", help="external data directory (overrides env/config)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. --set training.epochs=5 (repeatable)")
    p.add_argument("--experiment", default="experiment",
                   help="experiment config name in configs/ (experiment | pilot)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("inventory", help="resolve DOI, fetch metadata + file inventory")
    d = sub.add_parser("download", help="download raw files (keeps them; ~158.7 GB for all videos)")
    d.add_argument("--limit", type=int)
    d.add_argument("--non-video-only", action="store_true", help="only README + annotation CSV")
    sub.add_parser("validate-data", help="parse filenames + annotations, build video manifest")
    pp = sub.add_parser("preprocess", help="stream download -> verify -> decode -> cache -> delete")
    pp.add_argument("--limit", type=int)
    pp.add_argument("--workers", type=int)
    sub.add_parser("split", help="participant-independent train/val/test split + leakage audit")
    fe = sub.add_parser("extract-features", help="frozen backbone features (vit_s16, resnet50)")
    fe.add_argument("--backbone", choices=["vit_s16", "resnet50", "all"], default="all")
    tr = sub.add_parser("train", help="train + evaluate one model family")
    tr.add_argument("--model", choices=MODELS + ["ablations"], required=True,
                    help="bilstm is a historical model outside the final comparison scope (see decision log)")
    tr.add_argument("--seed", type=int)
    sub.add_parser("evaluate", help="recompute metrics for all saved test predictions")
    sub.add_parser("compare", help="comparison table, figures, report")
    sub.add_parser("task-b", help="EAAQ score regression (Task B)")
    ra = sub.add_parser("run-all", help="run every stage end to end")
    ra.add_argument("--skip-preprocess", action="store_true")
    ra.add_argument("--resume", action="store_true",
                    help="skip (model, seed) runs already completed and registered; mark killed runs interrupted")
    sub.add_parser("verify-results", help="recompute metrics from predictions + audit artifacts")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    from . import pipeline as P
    ctx = P.Context.create(args.data_dir, args.experiment, C.parse_overrides(args.set))
    cmd = args.command
    if cmd == "inventory":
        out = P.cmd_inventory(ctx)
    elif cmd == "download":
        out = P.cmd_download(ctx, limit=args.limit, non_video_only=args.non_video_only)
    elif cmd == "validate-data":
        out = P.cmd_validate_data(ctx)
    elif cmd == "preprocess":
        out = P.cmd_preprocess(ctx, limit=args.limit, workers=args.workers)
    elif cmd == "split":
        out = P.cmd_split(ctx)
    elif cmd == "extract-features":
        out = P.cmd_extract_features(ctx, args.backbone)
    elif cmd == "train":
        out = P.cmd_train(ctx, args.model, seed=args.seed)
    elif cmd == "evaluate":
        out = P.cmd_evaluate(ctx)
    elif cmd == "compare":
        out = P.cmd_compare(ctx)
    elif cmd == "task-b":
        out = P.cmd_task_b(ctx)
    elif cmd == "run-all":
        out = P.cmd_run_all(ctx, skip_preprocess=args.skip_preprocess, resume=args.resume)
    elif cmd == "verify-results":
        out = P.cmd_verify(ctx)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command {cmd}")
    print(json.dumps(out, indent=2, default=str)[:4000])
    return 0 if not (isinstance(out, dict) and out.get("exit_status") == "failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
