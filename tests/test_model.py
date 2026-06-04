#!/usr/bin/env python3
"""
Tests for market_news — model loading, tokenization, and forward pass.

All external/slow operations are mocked so tests run quickly without a GPU.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def mock_model():
    """Return a MagicMock that quacks like a GemmaForCausalLM."""
    model = MagicMock()
    model.device.type = "cpu"
    model.config.hidden_size = 1024
    model.config.num_attention_heads = 16
    model.config.num_hidden_layers = 12
    model.config.vocab_size = 256000
    model.config.use_cache = False

    # Simulate forward pass output
    fake_output = MagicMock()
    fake_output.logits = MagicMock()
    fake_output.logits.shape = (1, 32, 256000)  # (batch, seq_len, vocab)
    fake_output.loss = None
    model.return_value = fake_output
    model.__call__ = MagicMock(return_value=fake_output)
    return model


@pytest.fixture
def mock_tokenizer():
    """Return a MagicMock that quacks like a Gemma tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.vocab_size = 256000
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenizer.encode = MagicMock(
        side_effect=lambda text, **kw: list(range(1, len(text.split()) + 1))
    )
    tokenizer.decode = MagicMock(
        side_effect=lambda ids, **kw: "decoded mock text"
    )
    tokenizer.__call__ = MagicMock(
        return_value=MagicMock(
            input_ids=[[1, 2, 3]],
            attention_mask=[[1, 1, 1]],
        )
    )
    return tokenizer


# ──────────────────────────────────────────────
# Model Loading (market_news.llm.model)
# ──────────────────────────────────────────────


@patch("market_news.llm.model.AutoModelForCausalLM.from_pretrained")
def test_load_model_calls_from_pretrained(mock_from_pretrained, mock_model):
    mock_from_pretrained.return_value = mock_model

    from market_news.llm.model import load_model
    from market_news.llm.config import QLoRAConfig

    qlora = QLoRAConfig(use_4bit=True)
    model = load_model("google/gemma-4-12b-it", qlora=qlora, device_map="cpu")

    mock_from_pretrained.assert_called_once()
    _, kwargs = mock_from_pretrained.call_args
    assert kwargs.get("quantization_config") is not None
    assert model is mock_model


@patch("market_news.llm.model.AutoModelForCausalLM.from_pretrained")
def test_load_model_no_quant(mock_from_pretrained, mock_model):
    """Verify load_model with default config (no quantization if use_4bit=False)."""
    mock_from_pretrained.return_value = mock_model

    from market_news.llm.model import load_model

    model = load_model("google/gemma-4-12b-it", device_map="cpu")
    mock_from_pretrained.assert_called_once()
    assert model is mock_model


@patch("market_news.llm.model.AutoTokenizer.from_pretrained")
def test_load_tokenizer(mock_from_pretrained, mock_tokenizer):
    mock_from_pretrained.return_value = mock_tokenizer

    from market_news.llm.model import load_tokenizer

    tokenizer = load_tokenizer("google/gemma-4-12b-it")
    mock_from_pretrained.assert_called_once_with(
        "google/gemma-4-12b-it",
        token=None,
        padding_side="right",
    )
    assert tokenizer is mock_tokenizer


@patch("market_news.llm.model.AutoModelForCausalLM.from_pretrained")
@patch("market_news.llm.model.AutoTokenizer.from_pretrained")
def test_load_model_and_tokenizer(
    mock_tok_load, mock_model_load, mock_model, mock_tokenizer
):
    mock_tok_load.return_value = mock_tokenizer
    mock_model_load.return_value = mock_model

    from market_news.llm.model import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        "google/gemma-4-12b-it", apply_lora_adapters=False
    )
    assert model is mock_model
    assert tokenizer is mock_tokenizer


# ──────────────────────────────────────────────
# Forward Pass
# ──────────────────────────────────────────────


@patch("market_news.llm.model.AutoModelForCausalLM.from_pretrained")
@patch("market_news.llm.model.AutoTokenizer.from_pretrained")
def test_forward_pass(mock_tok_load, mock_model_load, mock_model, mock_tokenizer):
    """Verify that a forward pass returns logits of expected shape."""
    mock_tok_load.return_value = mock_tokenizer
    mock_model_load.return_value = mock_model

    from market_news.llm.model import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        "google/gemma-4-12b-it", apply_lora_adapters=False
    )

    text = "What is the market outlook for Q3?"
    inputs = tokenizer(text)
    output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    assert output.logits is not None
    assert output.logits.shape[-1] == model.config.vocab_size


# ──────────────────────────────────────────────
# Error Handling
# ──────────────────────────────────────────────


@patch("market_news.llm.model.AutoModelForCausalLM.from_pretrained")
def test_load_model_missing_model(mock_from_pretrained):
    mock_from_pretrained.side_effect = OSError("Model not found")

    from market_news.llm.model import load_model

    with pytest.raises(OSError, match="Model not found"):
        load_model("nonexistent/model")