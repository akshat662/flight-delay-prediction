"""Download the Kaggle flight-delay dataset into data/raw/.

Run as: python -m src.data.download [--force]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from src.config import Config, load_config

logger = logging.getLogger(__name__)


def download_dataset(config: Config | None = None, force: bool = False) -> Path:
    """Fetch the dataset via kagglehub and copy the target CSV into data/raw/.

    Skips the download if the CSV already exists in data/raw/ unless `force`
    is set. Returns the path to the local CSV.
    """
    config = config or load_config()
    dest_path = config.paths.raw_csv

    if dest_path.exists() and not force:
        logger.info("Raw CSV already present at %s (force=False); skipping download", dest_path)
        _log_file_stats(dest_path)
        return dest_path

    import kagglehub

    logger.info("Downloading dataset %s via kagglehub", config.dataset.kaggle_slug)
    download_root = Path(kagglehub.dataset_download(config.dataset.kaggle_slug))
    logger.info("kagglehub download root: %s", download_root)

    src_path = download_root / config.dataset.filename
    if not src_path.exists():
        matches = list(download_root.rglob(config.dataset.filename))
        if not matches:
            raise FileNotFoundError(
                f"{config.dataset.filename} not found under {download_root}"
            )
        src_path = matches[0]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dest_path)
    logger.info("Copied %s -> %s", src_path, dest_path)

    _log_file_stats(dest_path)
    return dest_path


def _log_file_stats(path: Path) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    n_rows = sum(1 for _ in open(path, "rb")) - 1  # exclude header
    logger.info("%s: %.1f MB, %d rows", path, size_mb, n_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the flight-delay dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    download_dataset(force=args.force)
