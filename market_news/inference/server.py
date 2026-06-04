"""
FastAPI inference server for the market-news model.

Exposes ``/generate`` (POST) and ``/health`` (GET) endpoints.
"""

from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, pipeline
from peft import PeftModel

from ..llm.model import load_model
from ..llm.config import QLoRAConfig
from ..training.dataset import format_fields


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    event_context: str = Field(..., description="Description of the financial event")
    econ_indicators: str = Field("N/A", description="Key economic indicators")
    max_new_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    headline: str
    sentiment_score: float


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    base_model_id: str,
    adapter_path: Optional[str] = None,
    qlora: Optional[QLoRAConfig] = None,
    device: str = "cuda:0",
) -> FastAPI:
    """Create a configured FastAPI application ready to serve.

    Args:
        base_model_id: HuggingFace model identifier (e.g.
            ``"google/gemma-4-12b-it"``).
        adapter_path: Path to saved LoRA adapter weights.  If ``None``
            the base model is used without adapters.
        qlora: QLoRA configuration matching the one used during training.
        device: Torch device string.

    Returns:
        A ``FastAPI`` instance.
    """
    if qlora is None:
        qlora = QLoRAConfig()

    # Load base model
    model = load_model(base_model_id, qlora=qlora, device_map=device)

    # Optionally attach adapters
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build text-generation pipeline
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
    )

    app = FastAPI(title="Market News Generator")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse()

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        try:
            prompt = format_fields(
                event_context=req.event_context,
                econ_indicators=req.econ_indicators,
                headline="",
                sentiment_score=0.0,
            )

            outputs = gen_pipeline(
                prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
            )

            generated_text: str = outputs[0]["generated_text"]

            # Naive parse: assume the model outputs
            # "Headline: ...\nSentiment Score: ..."
            headline = ""
            sentiment_score = 0.0

            for line in generated_text.strip().split("\n"):
                if line.startswith("Headline:"):
                    headline = line[len("Headline:"):].strip()
                elif line.startswith("Sentiment Score:"):
                    raw = line[len("Sentiment Score:"):].strip()
                    try:
                        sentiment_score = float(raw)
                    except ValueError:
                        sentiment_score = 0.0

            return GenerateResponse(
                headline=headline or generated_text.strip(),
                sentiment_score=sentiment_score,
            )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return app


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_server(
    base_model_id: str = "google/gemma-4-12b-it",
    adapter_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Run the FastAPI server via uvicorn.

    This is a blocking call — suitable for a dedicated entry-point script.
    """
    import uvicorn

    app = create_app(
        base_model_id=base_model_id,
        adapter_path=adapter_path,
    )
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market News inference server")
    parser.add_argument("--model", default="google/gemma-4-12b-it", help="Base model ID or path")
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter weights")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--grpc-port", type=int, default=None, help="gRPC port (if specified)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="GPU memory utilization (vLLM compatible flag)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max model context length")
    parser.add_argument("--dtype", default="auto", help="Model dtype")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallelism")
    args = parser.parse_args()

    if args.grpc_port:
        from .grpc_service import serve_grpc
        import threading
        grpc_thread = threading.Thread(
            target=serve_grpc,
            kwargs={
                "base_model_id": args.model,
                "adapter_path": args.adapter_path,
                "port": args.grpc_port,
            },
            daemon=True,
        )
        grpc_thread.start()

    run_server(
        base_model_id=args.model,
        adapter_path=args.adapter_path,
        host=args.host,
        port=args.port,
    )