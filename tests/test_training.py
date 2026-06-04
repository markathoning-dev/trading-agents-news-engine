#!/usr/bin/env python3
"""
Tests for market_news training — dataset loading, formatting, and training loop.

All external/slow operations are mocked so tests run quickly without a GPU.
"""

import pytest
from unittest.mock import patch, MagicMock

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def mock_hf_dataset():
    """Simulate a HuggingFace Dataset with text field."""
    ds = MagicMock()
    ds.__len__.return_value = 100
    ds.__getitem__.side_effect = lambda i: {
        "text": f"Sample training example {i}",
        "event_context": f"Event context {i}",
        "econ_indicators": "CPI: 3.2%, GDP: 2.1%",
        "headline": f"Headline {i}",
        "sentiment_score": 0.5 if i % 2 == 0 else -0.3,
    }
    ds.shuffle.return_value = ds
    ds.select.return_value = ds
    return ds


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device.type = "cpu"
    model.config.hidden_size = 1024
    model.config.vocab_size = 256000
    model.config.use_cache = False
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.vocab_size = 256000
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.__call__ = MagicMock(
        return_value=MagicMock(
            input_ids=[[1, 2, 3]],
            attention_mask=[[1, 1, 1]],
        )
    )
    return tokenizer


# ──────────────────────────────────────────────
# Dataset Formatting
# ──────────────────────────────────────────────


def test_format_fields():
    """Verify format_fields produces the expected instruction template."""
    from market_news.training.dataset import format_fields

    result = format_fields(
        event_context="AAPL reports record revenue",
        econ_indicators="CPI: 3.2%",
        headline="Apple revenue surges",
        sentiment_score=0.85,
    )
    assert "AAPL reports record revenue" in result
    assert "Apple revenue surges" in result
    assert "0.85" in result
    assert "Instruction" in result
    assert "Event Context" in result
    assert "Economic Indicators" in result
    assert "Response" in result


def test_format_fields_defaults():
    """Verify default values for econ_indicators, headline, sentiment_score."""
    from market_news.training.dataset import format_fields

    result = format_fields(event_context="Test event")
    assert "N/A" in result
    assert "Test event" in result


# ──────────────────────────────────────────────
# Dataset Loading
# ──────────────────────────────────────────────


@patch("market_news.training.dataset.load_dataset")
def test_load_fnspid(mock_load_dataset, mock_hf_dataset):
    mock_load_dataset.return_value = mock_hf_dataset

    from market_news.training.dataset import load_fnspid

    ds = load_fnspid(split="train")
    mock_load_dataset.assert_called_once_with("fnspid/fnspid", split="train", cache_dir=None)
    assert len(ds) == 100


@patch("market_news.training.dataset.load_dataset")
def test_load_finmultitime(mock_load_dataset, mock_hf_dataset):
    mock_load_dataset.return_value = mock_hf_dataset

    from market_news.training.dataset import load_finmultitime

    ds = load_finmultitime(split="validation")
    mock_load_dataset.assert_called_once_with(
        "finmultitime/finmultitime", split="validation", cache_dir=None
    )


@patch("market_news.training.dataset.load_dataset")
def test_load_pixiu(mock_load_dataset, mock_hf_dataset):
    mock_load_dataset.return_value = mock_hf_dataset

    from market_news.training.dataset import load_pixiu

    ds = load_pixiu(split="test")
    mock_load_dataset.assert_called_once_with("pixiu/pixiu", split="test", cache_dir=None)


# ──────────────────────────────────────────────
# Dataset Mixing
# ──────────────────────────────────────────────


@patch("market_news.training.dataset.load_dataset")
def test_mix_datasets(mock_load_dataset, mock_hf_dataset):
    mock_load_dataset.return_value = mock_hf_dataset

    from market_news.training.dataset import load_fnspid, mix_datasets

    fnspid_ds = load_fnspid()
    datasets = {"fnspid": fnspid_ds}
    weights = {"fnspid": 1.0}

    combined = mix_datasets(datasets, weights, seed=42)
    assert combined is not None
    # mix_datasets calls shuffle, select, and concatenate_datasets
    mock_hf_dataset.shuffle.assert_called()


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────


@patch("market_news.training.trainer.load_model_and_tokenizer")
@patch("market_news.training.trainer.SFTTrainer")
def test_train_function(
    mock_sft_trainer_cls,
    mock_load_model_and_tokenizer,
    mock_model,
    mock_tokenizer,
    mock_hf_dataset,
):
    """Verify that the train() function calls SFTTrainer correctly."""
    mock_load_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
    mock_trainer_instance = MagicMock()
    mock_sft_trainer_cls.return_value = mock_trainer_instance

    from market_news.training.trainer import train

    trainer = train(
        model_id="google/gemma-4-12b-it",
        dataset=mock_hf_dataset,
        output_dir="/tmp/test_train_out",
    )

    mock_load_model_and_tokenizer.assert_called_once()
    mock_sft_trainer_cls.assert_called_once()
    mock_trainer_instance.train.assert_called_once()
    mock_trainer_instance.save_model.assert_called_once()
    assert trainer is mock_trainer_instance


@patch("market_news.training.trainer.load_model_and_tokenizer")
@patch("market_news.training.trainer.SFTTrainer")
def test_train_saves_tokenizer(
    mock_sft_trainer_cls,
    mock_load_model_and_tokenizer,
    mock_model,
    mock_tokenizer,
    mock_hf_dataset,
):
    """Verify tokenizer.save_pretrained is called after training."""
    mock_load_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
    mock_trainer_instance = MagicMock()
    mock_sft_trainer_cls.return_value = mock_trainer_instance

    from market_news.training.trainer import train

    train(
        model_id="google/gemma-4-12b-it",
        dataset=mock_hf_dataset,
        output_dir="/tmp/test_save_out",
    )

    mock_tokenizer.save_pretrained.assert_called_once()


@patch("market_news.training.trainer.load_model_and_tokenizer")
@patch("market_news.training.trainer.SFTTrainer")
def test_train_with_config_overrides(
    mock_sft_trainer_cls,
    mock_load_model_and_tokenizer,
    mock_model,
    mock_tokenizer,
    mock_hf_dataset,
):
    """Verify custom QLoRAConfig and TrainingConfig are passed through."""
    mock_load_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)
    mock_trainer_instance = MagicMock()
    mock_sft_trainer_cls.return_value = mock_trainer_instance

    from market_news.training.trainer import train
    from market_news.llm.config import QLoRAConfig, TrainingConfig

    qlora = QLoRAConfig(r=32, lora_alpha=64, use_4bit=True)
    training = TrainingConfig(num_train_epochs=5, per_device_train_batch_size=2)

    train(
        model_id="google/gemma-4-12b-it",
        dataset=mock_hf_dataset,
        output_dir="/tmp/test_custom",
        qlora=qlora,
        training=training,
        max_seq_length=4096,
    )

    # The qlora config should be passed to load_model_and_tokenizer
    _, call_kwargs = mock_load_model_and_tokenizer.call_args
    assert call_kwargs["qlora"].r == 32

    # The training config should be converted to HF TrainingArguments
    mock_trainer_instance.train.assert_called_once()