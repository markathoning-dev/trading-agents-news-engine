"""
Model loading with QLoRA — supports Gemma 4 12B, 4-bit quantization via
bitsandbytes, and tokenizer setup with padding.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import QLoRAConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_bnb_config(qlora: QLoRAConfig) -> BitsAndBytesConfig:
    """Build a ``BitsAndBytesConfig`` from a ``QLoRAConfig``."""
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return BitsAndBytesConfig(
        load_in_4bit=qlora.use_4bit,
        bnb_4bit_compute_dtype=compute_dtype_map.get(
            qlora.bnb_4bit_compute_dtype, torch.float16
        ),
        bnb_4bit_quant_type=qlora.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=qlora.bnb_4bit_use_double_quant,
    )


def _build_lora_config(qlora: QLoRAConfig) -> LoraConfig:
    """Build a PEFT ``LoraConfig`` from a ``QLoRAConfig``."""
    return LoraConfig(
        r=qlora.r,
        lora_alpha=qlora.lora_alpha,
        target_modules=qlora.target_modules,
        lora_dropout=qlora.lora_dropout,
        bias=qlora.bias,
        task_type=qlora.task_type,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    model_id: str,
    qlora: Optional[QLoRAConfig] = None,
    device_map: str = "auto",
    token: Optional[str] = None,
    **kwargs,
) -> transformers.PreTrainedModel:
    """Load a base model with 4-bit QLoRA quantization.

    Args:
        model_id: HuggingFace model identifier (e.g. ``"google/gemma-4-12b-it"``).
        qlora: QLoRA configuration.  Defaults to ``QLoRAConfig()``.
        device_map: Device map passed to ``from_pretrained``.
        token: HF auth token for gated models.
        **kwargs: Additional keyword arguments forwarded to ``from_pretrained``.

    Returns:
        The base model wrapped for k-bit training (but **not** yet wrapped
        with LoRA adapters — call :func:`apply_lora` for that, or use
        :func:`load_model_and_tokenizer`).
    """
    if qlora is None:
        qlora = QLoRAConfig()

    bnb_config = _build_bnb_config(qlora)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        token=token,
        torch_dtype=torch.float16,
        **kwargs,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required for gradient checkpointing
    return model


def apply_lora(model: transformers.PreTrainedModel, qlora: Optional[QLoRAConfig] = None):
    """Wrap a base model with LoRA adapters (in-place)."""
    if qlora is None:
        qlora = QLoRAConfig()
    lora_config = _build_lora_config(qlora)
    model = get_peft_model(model, lora_config)
    return model


def load_tokenizer(
    model_id: str,
    token: Optional[str] = None,
    padding_side: str = "right",
    **kwargs,
) -> AutoTokenizer:
    """Load and configure a tokenizer with left/right padding.

    Args:
        model_id: HuggingFace model identifier.
        token: HF auth token.
        padding_side: ``"right"`` (default, recommended for autoregressive
            generation) or ``"left"`` (for encoder-style tasks).
        **kwargs: Additional keyword arguments for ``from_pretrained``.

    Returns:
        Configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        padding_side=padding_side,
        **kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model_and_tokenizer(
    model_id: str,
    qlora: Optional[QLoRAConfig] = None,
    apply_lora_adapters: bool = True,
    device_map: str = "auto",
    token: Optional[str] = None,
    padding_side: str = "right",
    **kwargs,
) -> Tuple[transformers.PreTrainedModel, AutoTokenizer]:
    """Convenience — load model + tokenizer in one call.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    if qlora is None:
        qlora = QLoRAConfig()

    model = load_model(model_id, qlora=qlora, device_map=device_map, token=token, **kwargs)
    if apply_lora_adapters:
        model = apply_lora(model, qlora=qlora)

    tokenizer = load_tokenizer(model_id, token=token, padding_side=padding_side)
    return model, tokenizer