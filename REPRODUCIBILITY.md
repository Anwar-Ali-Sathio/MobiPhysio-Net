# Reproducibility

```bash
pip install -r requirements-dev.txt          # CPU wheels
export MOBIPHYSIO_DATA_DIR=/path/with/30GB/free
export PYTHONPATH=src
python scripts/setup_environment.py           # checks deps, RAM, disk
python -m mobiphysio.cli inventory            # metadata + inventory manifests
python -m mobiphysio.cli preprocess           # ~158.7 GB streamed; resumable; ~2.5 h on 4 cores
python -m mobiphysio.cli run-all --skip-preprocess
python -m pytest -q
```

Per-stage commands: `validate-data`, `split`, `extract-features`, `train --model {svm,cnn2d,lstm,bilstm,
cnn_lstm,resnet50,mobiphysio_net,ablations} [--seed N]`, `task-b`, `evaluate`, `compare`,
`verify-results`. Config overrides: `--set mobiphysio_net.lr=0.001`, `--set experiment.seed=7`.

## What every run records (`experiments/<experiment>/<model>/<run_id>/run_manifest.json`)
Run ID, UTC start/end, exit status, completed stages, dataset DOI + version, SHA-256 of the inventory,
video manifest and split file, seed + determinism notes, full model config/variant, split counts,
Python/OS/CPU/RAM/GPU/package versions, best epoch + selection rule, reference-bank path/hashes
(MobiPhysio-Net), and SHA-256 of the prediction, metric and history files and the checkpoint.
Runs never overwrite each other; `experiments/<experiment>/registry.json` points to the primary
(seed 42) run per model and lists seed reruns. `results/run_manifest.json` records the `run-all` stages.

## Verification (`verify-results`)
For every registered run: metrics recomputed from `test_predictions.csv` equal the stored metrics and
the comparison table; the prediction file covers exactly the split's test videos with matching labels;
`y_pred = argmax(proba)`; output and checkpoint hashes match; the reference bank contains only training
videos. Result: `results/reports/verification.json`.

## Known nondeterminism
Multi-threaded CPU float reductions; SVC Platt scaling (internal 5-fold CV, seeded); `torch` deterministic
mode is warn-only. Re-running produces new run IDs; numbers may differ in late decimals.
