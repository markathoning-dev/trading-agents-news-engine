"""
Placeholder for the FNSPID dataset downloader.

FNSPID (Financial News Sentiment with Price Impact Dataset) provides
financial news articles paired with sentiment labels and price-impact
signals.

In the full pipeline this module would handle:
  - Downloading the raw CSV/Parquet files from a remote source.
  - Extracting and caching them locally.
  - Returning a pandas DataFrame or HuggingFace Dataset.

For now the heavy lifting is delegated to the HuggingFace datasets library
(:func:`market_news.training.dataset.load_fnspid`).
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FNSPID_HF_PATH = "fnspid/fnspid"


def download_fnspid(
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download the FNSPID dataset.

    Args:
        output_dir: Local path to store the downloaded data (default:
            ``./data/fnspid``).
        cache_dir: HuggingFace datasets cache directory.

    Returns:
        Path to the downloaded data directory.
    """
    if output_dir is None:
        output_dir = "./data/fnspid"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "FNSPID download delegated to ``datasets.load_dataset('%s')``. "
        "Use ``market_news.training.dataset.load_fnspid()`` at runtime.",
        _FNSPID_HF_PATH,
    )

    # In production, add explicit download logic here.
    # Example:
    #   from datasets import load_dataset
    #   ds = load_dataset(_FNSPID_HF_PATH, cache_dir=cache_dir)
    #   ds.save_to_disk(str(output_path))

    return output_path


def load_fnspid_raw(
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Load raw FNSPID data (delegates to the training loader)."""
    from ..training.dataset import load_fnspid
    return load_fnspid(cache_dir=cache_dir)