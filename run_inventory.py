"""Resolve the DOI and save Dataverse metadata + inventory manifests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mobiphysio.cli import main  # noqa: E402

sys.exit(main(["inventory", *sys.argv[1:]]))
