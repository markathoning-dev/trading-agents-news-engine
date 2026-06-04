# Training Guide: Fine-tuning Gemma 4 12B on Financial News

## Overview

This guide walks through end-to-end QLoRA fine-tuning of **Google's Gemma 4 12B Instruct model** on financial news data using the `market-news` toolkit.

**Hardware requirements:** Single A100-80GB (or 2× A10G-24GB via DeepSpeed ZeRO-3).

---

## 1. Environment Setup

### 1.1 Clone the repository

```bash
git clone <your-repo-url> trading-agents-news-engine
cd trading-agents-news-engine
```

### 1.2 Option A: Docker (Recommended)

Build the training image:

```bash
docker build -f docker/Dockerfile.training -t market-news-training .
```

Run training with dataset mounted:

```bash
docker run --gpus all \
    -v /path/to/your/dataset:/data \
    -v /path/to/checkpoints:/outputs \
    market-news-training \
    --model google/gemma-4-12b-it \
    --dataset /data/financial_news.jsonl \
    --output-dir /outputs/qlora-checkpoint \
    --quantize 4bit
```

### 1.3 Option B: Native (Manual venv)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch (CUDA 12.4)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install transformers>=4.46.0 accelerate>=1.0.0 datasets>=3.0.0 \
    peft>=0.13.0 trl>=0.12.0 bitsandbytes>=0.44.0

# Flash attention (optional, ~2× speedup)
pip install flash-attn>=2.6.0

# Install market-news package (editable)
pip install -e .
```

---

## 2. Dataset Preparation

### 2.1 Format

Training data must be a **JSONL file** where each line is:

```json
{
  "instruction": "Analyze the sentiment of this headline.",
  "input": "AAPL up 3% after earnings beat.",
  "output": "positive"
}
```

Or in chat format:

```json
{
  "messages": [
    {"role": "user", "content": "What is the outlook for oil?"},
    {"role": "assistant", "content": "Oil prices face headwinds from oversupply..."}
  ]
}
```

### 2.2 Validation split

The trainer automatically reserves 5% of the JSONL rows for validation. You can control this with `--eval-ratio 0.1`.

---

## 3. Running Training

### 3.1 Quick start (defaults)

```bash
./scripts/train.sh \
    --model google/gemma-4-12b-it \
    --dataset ./data/financial_news.jsonl \
    --output-dir ./outputs/train-v1
```

### 3.2 With 4-bit QLoRA

```bash
./scripts/train.sh \
    --model google/gemma-4-12b-it \
    --dataset ./data/financial_news.jsonl \
    --output-dir ./outputs/train-qlora \
    --quantize 4bit \
    --lr 2e-4 \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 8
```

### 3.3 Custom LoRA config

```bash
./scripts/train.sh \
    --model google/gemma-4-12b-it \
    --dataset ./data/financial_news.jsonl \
    --output-dir ./outputs/train-custom-lora \
    --quantize 4bit \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.1 \
    --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### 3.4 Multi-GPU with DeepSpeed

```bash
export CUDA_VISIBLE_DEVICES=0,1
./scripts/train.sh \
    --model google/gemma-4-12b-it \
    --dataset ./data/financial_news.jsonl \
    --output-dir ./outputs/train-deepspeed \
    --quantize 4bit \
    --deepspeed ./configs/deepspeed_zero3.json
```

---

## 4. Post-training Quantization

After training, quantize the full-precision adapter merge to 4-bit for efficient serving:

```bash
./scripts/quantize.sh \
    --model-dir ./outputs/train-v1/final \
    --output-dir ./outputs/quantized-v1 \
    --bits 4 \
    --method bitsandbytes
```

Or use GPTQ for slightly better quality:

```bash
./scripts/quantize.sh \
    --model-dir ./outputs/train-v1/final \
    --output-dir ./outputs/quantized-v1-gptq \
    --bits 4 \
    --method gptq \
    --calibration ./data/calib.jsonl
```

---

## 5. Evaluation

Run the full evaluation suite:

```bash
./scripts/evaluate.sh \
    --model ./outputs/quantized-v1 \
    --data ./data/eval.jsonl \
    --metrics perplexity,sentiment,coherence \
    --output ./outputs/evaluation-v1
```

### Individual metrics

| Metric | Description | Expected range |
|--------|-------------|----------------|
| Perplexity | Cross-entropy on held-out news | < 8 (good), < 5 (excellent) |
| Sentiment accuracy | Classification on financial sentiment test set | > 0.85 |
| Coherence | LLM-as-judge faithfulness score (1–5) | > 4.0 |

---

## 6. Serving

Launch the vLLM inference server:

```bash
./scripts/serve.sh \
    --model ./outputs/quantized-v1 \
    --port 8000 \
    --gpu-memory 0.90
```

Test with a curl:

```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Analyze the sentiment: TSLA up 8%", "max_tokens": 64}'
```

---

## 7. Docker Inference

Build the inference image:

```bash
docker build -f docker/Dockerfile.inference -t market-news-inference .
```

Run with quantized model mounted:

```bash
docker run --gpus all \
    -v /path/to/quantized/model:/model \
    -p 8000:8000 -p 50051:50051 \
    market-news-inference
```

---

## 8. Hyperparameter Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `google/gemma-4-12b-it` | Base model name or path |
| `--dataset` | (required) | Path to JSONL training data |
| `--output-dir` | `./outputs/train` | Checkpoint output directory |
| `--quantize` | None | `4bit` or `8bit` for QLoRA |
| `--lr` | `2e-4` | Learning rate |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--gradient-accumulation` | `8` | Gradient accumulation steps |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha scaling |
| `--lora-dropout` | `0.05` | LoRA dropout rate |
| `--max-seq-length` | `2048` | Maximum sequence length |
| `--warmup-ratio` | `0.03` | LR warmup fraction |
| `--logging-steps` | `25` | Logging interval |
| `--save-steps` | `500` | Checkpoint save interval |
| `--eval-steps` | `500` | Validation interval |
| `--target-modules` | `q_proj,v_proj` | LoRA target modules |

---

## 9. Troubleshooting

**OOM (Out of Memory)**
- Lower `--batch-size` to 1 or 2
- Increase `--gradient-accumulation` to compensate
- Enable `--quantize 4bit`
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Slow training**
- Install `flash-attn` for ~2× speedup
- Use `--bf16` if your GPU supports it (A100, H100)
- Increase `--batch-size` to saturate GPU

**Poor model quality**
- Increase `--epochs` to 5–10
- Increase `--lora-r` to 32–64
- Ensure dataset has ≥ 1000 examples
- Check for data leakage between train and eval splits