"""Evaluation runner — computes perplexity, sentiment accuracy, and coherence.

Usage:
    python -m market_news.evaluation.evaluator \
        --model ./outputs/quantized \
        --data ./data/eval.jsonl \
        --metrics perplexity,sentiment,coherence
"""

import argparse
import json
from pathlib import Path
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import compute_perplexity, compute_sentiment_accuracy, compute_coherence_score


def run_evaluation(
    model_path: str,
    data_path: str,
    metrics: List[str],
    batch_size: int = 4,
    output_dir: str = "./outputs/evaluation",
) -> dict:
    """Run the specified metrics on a model using evaluation data.

    Args:
        model_path: Path or HF hub ID of the model to evaluate.
        data_path: JSONL file with evaluation samples.
        metrics: List of metric names ('perplexity', 'sentiment', 'coherence').
        batch_size: Batch size (used where applicable).
        output_dir: Directory to write results.

    Returns:
        Dictionary of metric names to scores.
    """
    results = {}

    # Load model and tokenizer (set to eval mode)
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation data
    with open(data_path) as f:
        records = [json.loads(line.strip()) for line in f]

    texts = []
    predicted_scores = []
    ground_truth_scores = []
    headlines = []
    contexts = []

    for rec in records:
        texts.append(rec.get("text", rec.get("instruction", "")))
        predicted_scores.append(rec.get("predicted_sentiment", 0.0))
        ground_truth_scores.append(rec.get("sentiment_score", 0.0))
        headlines.append(rec.get("headline", ""))
        contexts.append(rec.get("event_context", rec.get("input", "")))

    if "perplexity" in metrics:
        print("Computing perplexity...")
        ppl = compute_perplexity(model, tokenizer, texts)
        results["perplexity"] = round(ppl, 4)

    if "sentiment" in metrics:
        print("Computing sentiment accuracy...")
        acc = compute_sentiment_accuracy(predicted_scores, ground_truth_scores)
        results["sentiment_accuracy"] = round(acc, 4)

    if "coherence" in metrics:
        print("Computing coherence score...")
        coh = compute_coherence_score(headlines, contexts)
        results["coherence_score"] = round(coh, 4)

    # Write results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market news model evaluation")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--data", required=True, help="JSONL evaluation data")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output", default="./outputs/evaluation", help="Output directory")
    parser.add_argument("--metrics", default="perplexity,sentiment,coherence",
                        help="Comma-separated metric names")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        data_path=args.data,
        metrics=[m.strip() for m in args.metrics.split(",")],
        batch_size=args.batch_size,
        output_dir=args.output,
    )