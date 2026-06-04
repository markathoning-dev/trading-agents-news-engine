"""
Placeholder for the FinMultiTime dataset downloader.

FinMultiTime is a multi-timeframe financial news dataset with
sentiment and economic-indicator annotations sourced from multiple
time horizons (daily, weekly, monthly).

In the full pipeline this module would handle:
  - Downloading the raw data files.
  - Extracting and caching.
  - Returning a pandas DataFrame or HuggingFace Dataset.

Runtime loading is delegated to
:func:`market_news.training.dataset.load_finmultitime`.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FINMULTITIME_HF_PATH = "finmultitime/finmultitime"


def download_finmultitime(
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download the FinMultiTime dataset.

    Args:
        output_dir: Local path to store the downloaded data (default:
            ``./data/finmultitime``).
        cache_dir: HuggingFace datasets cache directory.

    Returns:
        Path to the downloaded data directory.
    """
    if output_dir is None:
        output_dir = "./data/finmultitime"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "FinMultiTime download delegated to ``datasets.load_dataset('%s')``. "
        "Use ``market_news.training.dataset.load_finmultitime()`` at runtime.",
        _FINMULTITIME_HF_PATH,
    )

    return output_path


def load_finmultitime_raw(
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Load raw FinMultiTime data (delegates to the training loader)."""
    from ..training.dataset import load_finmultitime
    return load_finmultitime(cache_dir=cache_dir)