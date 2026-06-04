"""
QLoRA configuration and HuggingFace TrainingArguments / SFTConfig for
fine-tuning Gemma 4 12B on financial news data.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments as HFTrainingArguments


# ---------------------------------------------------------------------------
# QLoRA hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class QLoRAConfig:
    """Configuration for BitsAndBytes 4-bit QLoRA."""

    r: int = 16
    """LoRA rank."""

    lora_alpha: int = 32
    """LoRA alpha scaling factor."""

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    """Transformer modules to attach LoRA adapters to."""

    lora_dropout: float = 0.1
    """Dropout probability for LoRA layers."""

    bias: str = "none"
    """Bias setting for LoRA ('none', 'all', 'lora_only')."""

    task_type: str = "CAUSAL_LM"
    """Type of task for the PeftModel."""

    # BitsAndBytes 4-bit config
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """High-level training hyper-parameters mapped to HF TrainingArguments."""

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 25
    save_steps: int = 250
    save_total_limit: int = 2
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.3
    seed: int = 42

    def to_hf(self) -> HFTrainingArguments:
        """Convert to a HuggingFace ``TrainingArguments`` instance."""
        return HFTrainingArguments(
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            gradient_checkpointing=self.gradient_checkpointing,
            optim=self.optim,
            max_grad_norm=self.max_grad_norm,
            seed=self.seed,
            output_dir=".",  # placeholder — caller overrides
            report_to="none",
        )


# ---------------------------------------------------------------------------
# SFTConfig (TRL)
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Thin wrapper over TRL's SFTConfig exposed for the project namespace."""

    max_seq_length: int = 2048
    dataset_text_field: str = "text"
    packing: bool = False

    # Pack the fields that differ from TrainingConfig here; actual
    # instantiation is done inside the trainer module.