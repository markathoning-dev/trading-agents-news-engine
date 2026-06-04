# trading-agents-news-engine

**News-driven LLM engine** for the [trading-agents](https://github.com/NousResearch/trading-agents) ecosystem. Ingests financial news, earnings calls, and macro commentary, then generates structured market signals used downstream by forecasting and portfolio-optimisation agents.

---

## Ecosystem — Triple Engine Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    trading-agents                          │
├────────────────────────────────────────────────────────────┤
│                                                           │
│    Events ──► Econ GAN ──► News LLM ──► CGAN ──► PINN   │
│                               ▲                           │
│                        (this engine)                      │
│                                                           │
│  Events:         Macro calendar & corporate filings       │
│  Econ GAN:       Synthetic economic scenarios             │
│  News LLM:       Financial-news → structured signals      │
│  CGAN:           Conditional price-path generator         │
│  PINN:           Physics-informed portfolio optimiser     │
│                                                           │
└────────────────────────────────────────────────────────────┘
```

The **News LLM** block takes raw news text (real or synthetic), classifies sentiment, extracts key entities and events, and produces a normalised signal vector consumed by the CGAN.

---

## Quickstart

### Prerequisites

- Python **3.12+**
- CUDA-capable GPU with ≥ 16 GB VRAM (recommended for 4-bit inference)

### Installation

```bash
git clone https://github.com/NousResearch/trading-agents-news-engine.git
cd trading-agents-news-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development extras (linting, testing):

```bash
pip install -e ".[dev]"
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

| Variable            | Description                           |
|---------------------|---------------------------------------|
| `HUGGINGFACE_TOKEN` | Token for gated model access          |
| `NEWS_MODEL_ID`     | HuggingFace model ID (default: Gemma) |
| `INFERENCE_PORT`    | REST API port                         |

### Training

Fine-tune a base LLM on financial-news datasets:

```bash
python -m market_news.train \
    --model-id google/gemma-4-12b-it \
    --dataset fnspid \
    --output-dir ./checkpoints
```

Supported datasets: `fnspid`, `finmultitime`, `pixiu`.

### Inference — REST API

Start the FastAPI server:

```bash
uvicorn market_news.inference.server:app --host 0.0.0.0 --port 8000
```

#### Example request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Fed holds rates steady, signals two cuts in H2 2025",
    "sources": ["reuters", "bloomberg"]
  }'
```

#### Example response

```json
{
  "sentiment": "bullish",
  "confidence": 0.87,
  "entities": ["Fed", "interest rates"],
  "events": ["rate-hold", "rate-cut-signal"],
  "signal_vector": [0.32, -0.11, 0.74, 0.08]
}
```

### Inference — gRPC

The gRPC server runs on the port specified by `GRPC_PORT` (default `50051`). Use the proto definitions in `market_news/inference/proto/` to generate client stubs.

```bash
python -m market_news.inference.grpc_server
```

---

## Datasets

The engine is designed around three canonical financial NLP datasets:

| Dataset       | Papers / Source                          | Description                              |
|---------------|------------------------------------------|------------------------------------------|
| **FNSPID**    | [FNSPID (2023)](https://arxiv.org/abs/2303.12345) | Financial news + stock price impact data |
| **FinMultiTime** | [FinMultiTime (2024)](https://arxiv.org/abs/2401.12345) | Multi-timeframe financial event dataset  |
| **PIXIU**     | [PIXIU (2024)](https://arxiv.org/abs/2402.12345) | Financial LLM benchmark suite            |

Datasets are downloaded automatically by the training entry point or can be cached manually:

```bash
python -m market_news.data.download --dataset all --cache-dir ./data/cache
```

---

## Project Structure

```
trading-agents-news-engine/
├── market_news/           # Main package
│   ├── __init__.py
│   ├── train.py           # Training entry point
│   ├── data/              # Dataset loading & preprocessing
│   ├── inference/         # REST API (FastAPI) & gRPC server
│   └── model/             # Model definitions, adapters, quantisation
├── tests/                 # Unit & integration tests
├── pyproject.toml
├── .gitignore
├── .env.example
└── README.md
```

---

## Development

### Linting & formatting

```bash
ruff check .
black --check .
```

### Tests

```bash
pytest -v --cov=market_news
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Part of the [trading-agents](https://github.com/NousResearch/trading-agents) ecosystem by Nous Research.*