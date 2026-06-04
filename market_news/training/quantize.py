"""Post-training quantization (bitsandbytes 4-bit / 8-bit, or GPTQ)."""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quantize_bitsandbytes(
    model_dir: str,
    output_dir: str,
    bits: int = 4,
    compute_dtype: str = "float16",
    double_quant: bool = True,
    quant_type: str = "nf4",
) -> None:
    """Load a model from ``model_dir`` and save a quantized copy to ``output_dir``.

    Args:
        model_dir: Path to the source model checkpoint.
        output_dir: Where to save the quantized model.
        bits: Number of bits (4 or 8).
        compute_dtype: Computation dtype ('float16', 'bfloat16', 'float32').
        double_quant: Whether to use double quantization (4-bit only).
        quant_type: 4-bit quantization type ('nf4' or 'fp4').
    """
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    compute_dtype = dtype_map.get(compute_dtype, torch.float16)

    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=double_quant,
        )
    elif bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported bit count: {bits}. Use 4 or 8.")

    print(f"Loading model from {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving quantized model to {output_path} ...")
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    print("Done.")


def quantize_gptq(
    model_dir: str,
    output_dir: str,
    calibration_data: Optional[str] = None,
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
) -> None:
    """Quantize a model using GPTQ (via auto-gptq).

    Args:
        model_dir: Path to the source model checkpoint.
        output_dir: Where to save the quantized model.
        calibration_data: Path to JSONL calibration data. If None, a small
            default calibration set is used.
        bits: Bit width (4 or 8).
        group_size: Group size for GPTQ.
        damp_percent: Damping percentage.
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        raise ImportError(
            "auto-gptq is required for GPTQ quantization. "
            "Install with: pip install auto-gptq"
        )

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=False,
    )

    print(f"Loading model from {model_dir} for GPTQ quantization ...")
    model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Prepare calibration data
    if calibration_data:
        with open(calibration_data) as f:
            samples = [json.loads(line.strip()) for line in f]
        texts = [
            s.get("text", s.get("instruction", s.get("input", json.dumps(s))))
            for s in samples
        ]
    else:
        texts = [
            "AAPL reports record revenue of $120B in Q4.",
            "Fed holds rates steady at 5.25% amid inflation concerns.",
            "Oil prices surge 8% on OPEC+ production cut announcement.",
        ]

    def tokenize_calibration(examples):
        return tokenizer(examples, return_tensors="pt", padding=True, truncation=True, max_length=512)

    print("Running GPTQ quantization ...")
    model.quantize(
        tokenize_calibration(texts),
        use_triton=False,
        autotune_warmup_iters=100,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving GPTQ quantized model to {output_path} ...")
    model.save_quantized(str(output_path), use_safetensors=True)
    tokenizer.save_pretrained(str(output_path))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training quantization")
    parser.add_argument("--model-dir", required=True, help="Path to source model checkpoint")
    parser.add_argument("--output-dir", default="./outputs/quantized", help="Output directory")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Bit width")
    parser.add_argument("--method", default="bitsandbytes", choices=["bitsandbytes", "gptq"],
                        help="Quantization method")
    parser.add_argument("--dtype", default="float16", help="Compute dtype")
    parser.add_argument("--calibration-data", default=None, help="JSONL calibration file (GPTQ)")
    args = parser.parse_args()

    if args.method == "bitsandbytes":
        quantize_bitsandbytes(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            bits=args.bits,
            compute_dtype=args.dtype,
        )
    elif args.method == "gptq":
        quantize_gptq(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            calibration_data=args.calibration_data,
            bits=args.bits,
        )