#!/usr/bin/env python3
"""
Tests for market_news inference — FastAPI HTTP endpoints and gRPC service.

All external/slow operations are mocked so tests run quickly without a GPU.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_request():
    """A typical inference request matching the GenerateRequest schema."""
    return {
        "event_context": "AAPL reports record quarterly revenue of $120B",
        "econ_indicators": "CPI: 3.2%, GDP: 2.1%, Fed Rate: 5.25%",
        "max_new_tokens": 128,
        "temperature": 0.7,
    }


@pytest.fixture
def mock_pipeline():
    """Mocked text-generation pipeline."""
    pipeline = MagicMock()
    pipeline.return_value = [
        {
            "generated_text": (
                "Headline: Apple revenue surges past expectations\n"
                "Sentiment Score: 0.85"
            )
        }
    ]
    return pipeline


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device.type = "cpu"
    model.config.hidden_size = 1024
    model.config.vocab_size = 256000
    model.config.use_cache = False
    model.eval = MagicMock()
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# ──────────────────────────────────────────────
# FastAPI App Creation
# ──────────────────────────────────────────────


@patch("market_news.inference.server.load_model")
@patch("market_news.inference.server.AutoTokenizer.from_pretrained")
@patch("market_news.inference.server.pipeline")
def test_create_app(mock_pipeline_cls, mock_tok_load, mock_model_load,
                    mock_model, mock_tokenizer, mock_pipeline):
    """Verify create_app builds a FastAPI app with health + generate endpoints."""
    mock_model_load.return_value = mock_model
    mock_tok_load.return_value = mock_tokenizer
    mock_pipeline_cls.return_value = mock_pipeline

    from market_news.inference.server import create_app

    app = create_app(
        base_model_id="google/gemma-4-12b-it",
        adapter_path=None,
        device="cpu",
    )

    assert app.title == "Market News Generator"

    # Check routes exist
    routes = [r.path for r in app.routes]
    assert "/health" in routes
    assert "/generate" in routes


# ──────────────────────────────────────────────
# HTTP Endpoints
# ──────────────────────────────────────────────


@patch("market_news.inference.server.load_model")
@patch("market_news.inference.server.AutoTokenizer.from_pretrained")
@patch("market_news.inference.server.pipeline")
def test_health_endpoint(mock_pipeline_cls, mock_tok_load, mock_model_load,
                         mock_model, mock_tokenizer, mock_pipeline):
    """GET /health should return 200 with status ok."""
    mock_model_load.return_value = mock_model
    mock_tok_load.return_value = mock_tokenizer
    mock_pipeline_cls.return_value = mock_pipeline

    from market_news.inference.server import create_app

    app = create_app(base_model_id="test", device="cpu")
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("market_news.inference.server.load_model")
@patch("market_news.inference.server.AutoTokenizer.from_pretrained")
@patch("market_news.inference.server.pipeline")
def test_generate_endpoint(mock_pipeline_cls, mock_tok_load, mock_model_load,
                           mock_model, mock_tokenizer, mock_pipeline, sample_request):
    """POST /generate should return headline and sentiment_score."""
    mock_model_load.return_value = mock_model
    mock_tok_load.return_value = mock_tokenizer
    mock_pipeline_cls.return_value = mock_pipeline

    from market_news.inference.server import create_app

    app = create_app(base_model_id="test", device="cpu")
    client = TestClient(app)

    response = client.post("/generate", json=sample_request)
    assert response.status_code == 200
    data = response.json()
    assert data["headline"] == "Apple revenue surges past expectations"
    assert data["sentiment_score"] == 0.85


@patch("market_news.inference.server.load_model")
@patch("market_news.inference.server.AutoTokenizer.from_pretrained")
@patch("market_news.inference.server.pipeline")
def test_generate_without_indicators(mock_pipeline_cls, mock_tok_load, mock_model_load,
                                     mock_model, mock_tokenizer):
    """econ_indicators defaults to 'N/A' when not provided."""
    mock_model_load.return_value = mock_model
    mock_tok_load.return_value = mock_tokenizer
    pipeline = MagicMock()
    pipeline.return_value = [{"generated_text": "Headline: Test\nSentiment Score: 0.5"}]
    mock_pipeline_cls.return_value = pipeline

    from market_news.inference.server import create_app

    app = create_app(base_model_id="test", device="cpu")
    client = TestClient(app)

    response = client.post("/generate", json={"event_context": "Test event"})
    assert response.status_code == 200


# ──────────────────────────────────────────────
# Error Handling
# ──────────────────────────────────────────────


@patch("market_news.inference.server.load_model")
@patch("market_news.inference.server.AutoTokenizer.from_pretrained")
@patch("market_news.inference.server.pipeline")
def test_generate_missing_event_context(mock_pipeline_cls, mock_tok_load, mock_model_load,
                                        mock_model, mock_tokenizer):
    """Missing required event_context should return 422."""
    mock_model_load.return_value = mock_model
    mock_tok_load.return_value = mock_tokenizer
    mock_pipeline_cls.return_value = MagicMock()

    from market_news.inference.server import create_app

    app = create_app(base_model_id="test", device="cpu")
    client = TestClient(app)

    response = client.post("/generate", json={})
    assert response.status_code == 422


@patch("market_news.inference.server.load_model")
@patch("market_news.inference.server.AutoTokenizer.from_pretrained")
@patch("market_news.inference.server.pipeline")
def test_generate_pipeline_error(mock_pipeline_cls, mock_tok_load, mock_model_load,
                                 mock_model, mock_tokenizer):
    """Pipeline failure should return 500."""
    mock_model_load.return_value = mock_model
    mock_tok_load.return_value = mock_tokenizer
    pipeline = MagicMock()
    pipeline.side_effect = RuntimeError("Inference failed")
    mock_pipeline_cls.return_value = pipeline

    from market_news.inference.server import create_app

    app = create_app(base_model_id="test", device="cpu")
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={"event_context": "Test event"},
    )
    assert response.status_code == 500


# ──────────────────────────────────────────────
# gRPC Service (market_news.inference.grpc_service)
# ──────────────────────────────────────────────


@patch("market_news.inference.grpc_service.create_app")
def test_grpc_service_import(mock_create_app):
    """Verify the gRPC module can be imported and creates an app."""
    mock_create_app.return_value = MagicMock()

    from market_news.inference.grpc_service import serve_grpc

    # serve_grpc should be a callable
    assert callable(serve_grpc)


@patch("market_news.inference.grpc_service.grpc")
@patch("market_news.inference.grpc_service.create_app")
def test_grpc_serve_call(mock_create_app, mock_grpc):
    """Verify serve_grpc sets up gRPC server with correct params."""
    mock_create_app.return_value = MagicMock()
    mock_server = MagicMock()
    mock_grpc.server.return_value = mock_server

    from market_news.inference.grpc_service import serve_grpc

    serve_grpc(base_model_id="google/gemma-4-12b-it", port=50051)
    mock_grpc.server.assert_called_once()
    mock_server.add_insecure_port.assert_called_once_with("[::]:50051")


# ──────────────────────────────────────────────
# Server Runner
# ──────────────────────────────────────────────


@patch("market_news.inference.server.create_app")
@patch("market_news.inference.server.uvicorn")
def test_run_server(mock_uvicorn, mock_create_app):
    """Verify run_server creates an app and runs it via uvicorn."""
    mock_create_app.return_value = MagicMock()

    from market_news.inference.server import run_server

    run_server(base_model_id="google/gemma-4-12b-it", host="0.0.0.0", port=8000)
    mock_create_app.assert_called_once_with(
        base_model_id="google/gemma-4-12b-it",
        adapter_path=None,
    )
    mock_uvicorn.run.assert_called_once()
    _, kwargs = mock_uvicorn.run.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000