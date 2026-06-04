"""
FNSPID, FinMultiTime, and PIXIU dataset loaders with weighted mixing.

Each ``load_*`` function returns a HuggingFace ``Dataset`` whose samples
contain at minimum a ``"text"`` key with the formatted instruction-style
prompt.

Dataset mixing is handled by :func:`mix_datasets`.
"""

from typing import Dict, List, Optional

from datasets import Dataset as HFDataset
from datasets import DatasetDict, concatenate_datasets, load_dataset


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def load_fnspid(
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """Load the FNSPID dataset from HuggingFace.

    FNSPID (Financial News Sentiment with Price Impact Dataset) contains
    financial news articles with associated sentiment labels and price-impact
    signals.

    Args:
        split: Dataset split — ``"train"``, ``"validation"``, or ``"test"``.
        cache_dir: Optional cache directory for HF datasets.

    Returns:
        HuggingFace ``Dataset``.
    """
    ds: DatasetDict = load_dataset(
        "fnspid/fnspid",
        split=split,
        cache_dir=cache_dir,
    )
    return ds  # type: ignore[return-value]


def load_finmultitime(
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """Load the FinMultiTime dataset from HuggingFace.

    FinMultiTime is a multi-timeframe financial news dataset with
    sentiment and economic-indicator annotations.

    Args:
        split: Dataset split — ``"train"``, ``"validation"``, or ``"test"``.
        cache_dir: Optional cache directory for HF datasets.

    Returns:
        HuggingFace ``Dataset``.
    """
    ds: DatasetDict = load_dataset(
        "finmultitime/finmultitime",
        split=split,
        cache_dir=cache_dir,
    )
    return ds  # type: ignore[return-value]


def load_pixiu(
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """Load the PIXIU dataset from HuggingFace.

    PIXIU is a large-scale Chinese financial dataset with news, reports,
    and stock data.

    Args:
        split: Dataset split — ``"train"``, ``"validation"``, or ``"test"``.
        cache_dir: Optional cache directory for HF datasets.

    Returns:
        HuggingFace ``Dataset``.
    """
    ds: DatasetDict = load_dataset(
        "pixiu/pixiu",
        split=split,
        cache_dir=cache_dir,
    )
    return ds  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Dataset mixing
# ---------------------------------------------------------------------------

def mix_datasets(
    datasets: Dict[str, HFDataset],
    weights: Dict[str, float],
    seed: int = 42,
) -> HFDataset:
    """Mix multiple datasets with weighted sampling.

    Args:
        datasets: Mapping of ``name -> Dataset``.
        weights: Mapping of ``name -> sampling weight``.  Weights are
            normalised automatically.
        seed: Random seed for reproducible mixing.

    Returns:
        A combined ``Dataset`` with weighted contributions from each source.
    """
    total_weight = sum(weights.values())
    normalized = {k: v / total_weight for k, v in weights.items()}

    sampled_parts: List[HFDataset] = []
    for name, weight in normalized.items():
        ds = datasets[name]
        n = int(len(ds) * weight)
        sampled = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        sampled_parts.append(sampled)

    combined = concatenate_datasets(sampled_parts)
    return combined.shuffle(seed=seed)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_INSTRUCTION_TEMPLATE = """\
### Instruction
You are a financial news analyst. Given the event context and economic indicators below, generate a concise headline and a sentiment score.

### Event Context
{event_context}

### Economic Indicators
{econ_indicators}

### Response
Headline: {headline}
Sentiment Score: {sentiment_score}
"""


def format_fields(
    event_context: str,
    econ_indicators: str = "N/A",
    headline: str = "",
    sentiment_score: float = 0.0,
) -> str:
    """Format fields into the standard instruction-template string.

    Use this to transform raw dataset rows into the ``"text"`` field
    expected by the SFT trainer.

    Args:
        event_context:  Free-text description of the financial event.
        econ_indicators: Key economic indicators at the time of the event.
        headline:  The ground-truth headline (empty string during inference).
        sentiment_score:  Ground-truth sentiment score (0.0 during inference).

    Returns:
        A formatted instruction string.
    """
    return _INSTRUCTION_TEMPLATE.format(
        event_context=event_context,
        econ_indicators=econ_indicators,
        headline=headline,
        sentiment_score=sentiment_score,
    )