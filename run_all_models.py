"""Run every stage after preprocessing: validate, split, features, 7 models, ablations,
seed reruns, Task B, evaluation, figures/reports and verification.
Pass `--experiment pilot` (before other args) for the reduced pilot configuration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mobiphysio.cli import main  # noqa: E402

sys.exit(main([*sys.argv[1:], "run-all", "--skip-preprocess"]))
