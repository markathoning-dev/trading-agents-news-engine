"""
SFTTrainer setup and ``train()`` convenience for fine-tuning with QLoRA.
"""

from pathlib import Path
from typing import Optional, Union

from datasets import Dataset as HFDataset
from transformers import TrainingArguments as HFTrainingArguments
from trl import SFTTrainer as TRL_SFTTrainer

from ..llm.config import QLoRAConfig, TrainingConfig
from ..llm.model import load_model_and_tokenizer


class SFTTrainer(TRL_SFTTrainer):
    """Thin project-specific subclass of TRL's ``SFTTrainer``.

    All standard ``SFTTrainer`` functionality is inherited.  Use the
    module-level :func:`train` function for a streamlined entry-point.
    """

    pass


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def train(
    model_id: str,
    dataset: HFDataset,
    output_dir: Union[str, Path],
    *,
    qlora: Optional[QLoRAConfig] = None,
    training: Optional[TrainingConfig] = None,
    max_seq_length: int = 2048,
    dataset_text_field: str = "text",
    token: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> SFTTrainer:
    """Load a QLoRA model and train on the provided dataset.

    After training completes only the **LoRA adapter weights** are saved
    to ``output_dir`` (via ``trainer.save_model()``).

    Args:
        model_id: HuggingFace model identifier.
        dataset: HuggingFace ``Dataset`` containing a ``text`` field (or
            the field indicated by ``dataset_text_field``).
        output_dir: Directory to save the adapter weights into.
        qlora: QLoRA configuration.  Defaults to ``QLoRAConfig()``.
        training: Training hyper-parameters.  Defaults to ``TrainingConfig()``.
        max_seq_length: Maximum sequence length for the SFT trainer.
        dataset_text_field: Column name containing the formatted text.
        token: HF auth token for gated models.
        resume_from_checkpoint: Optional checkpoint path to resume from.

    Returns:
        The trained ``SFTTrainer`` instance (model weights already saved).
    """
    if qlora is None:
        qlora = QLoRAConfig()
    if training is None:
        training = TrainingConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_id, qlora=qlora, apply_lora_adapters=True, token=token
    )

    # 2. Build HF TrainingArguments
    hf_args: HFTrainingArguments = training.to_hf()
    hf_args.output_dir = str(output_dir)

    # 3. Create trainer
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=hf_args,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        dataset_text_field=dataset_text_field,
        packing=False,
    )

    # 4. Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 5. Save only the LoRA adapter weights
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return trainer


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    from datasets import Dataset as HFDataset

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for market-news")
    parser.add_argument("--model", default="google/gemma-4-12b-it", help="Base model ID")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--output-dir", default="./outputs/train", help="Output directory")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None,
                        help="Quantization mode")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target-modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="LoRA target modules")
    parser.add_argument("--deepspeed", default=None, help="DeepSpeed config file")
    args = parser.parse_args()

    qlora_config = QLoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        use_4bit=(args.quantize == "4bit"),
    )

    training_config = TrainingConfig(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
    )

    # Load dataset from JSONL
    records = []
    with open(args.dataset) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    dataset = HFDataset.from_list(records)

    train(
        model_id=args.model,
        dataset=dataset,
        output_dir=args.output_dir,
        qlora=qlora_config,
        training=training_config,
        max_seq_length=args.max_seq_length,
    )