"""Stream-process all videos (download -> MD5 verify -> decode -> cache -> delete raw).
Use `python -m mobiphysio.cli download` instead to keep raw MP4s (~158.7 GB)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mobiphysio.cli import main  # noqa: E402

sys.exit(main(["preprocess", *sys.argv[1:]]))
