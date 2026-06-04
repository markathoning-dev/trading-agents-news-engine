"""
Placeholder for the PIXIU dataset downloader.

PIXIU is a large-scale Chinese financial dataset containing news articles,
financial reports, and associated stock data.

In the full pipeline this module would handle:
  - Downloading the raw data files.
  - Extracting and caching.
  - Returning a pandas DataFrame or HuggingFace Dataset.

Runtime loading is delegated to
:func:`market_news.training.dataset.load_pixiu`.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PIXIU_HF_PATH = "pixiu/pixiu"


def download_pixiu(
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download the PIXIU dataset.

    Args:
        output_dir: Local path to store the downloaded data (default:
            ``./data/pixiu``).
        cache_dir: HuggingFace datasets cache directory.

    Returns:
        Path to the downloaded data directory.
    """
    if output_dir is None:
        output_dir = "./data/pixiu"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "PIXIU download delegated to ``datasets.load_dataset('%s')``. "
        "Use ``market_news.training.dataset.load_pixiu()`` at runtime.",
        _PIXIU_HF_PATH,
    )

    return output_path


def load_pixiu_raw(
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Load raw PIXIU data (delegates to the training loader)."""
    from ..training.dataset import load_pixiu
    return load_pixiu(cache_dir=cache_dir)