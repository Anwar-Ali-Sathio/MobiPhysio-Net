"""Recompute all metrics from saved predictions and audit provenance/hashes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mobiphysio.cli import main  # noqa: E402

sys.exit(main([*sys.argv[1:], "verify-results"]))
