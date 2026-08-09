# Lunar-v2-Large

A powerful large language model with two variants:
- Lunar-v2-Large (1.2B parameters)
- Lunar-v2-large (320M parameters, quantized)

## Model Architecture
- Based on the transformer architecture with improvements
- Uses rotary positional embeddings
- Implements flash attention for better performance
- Supports both full precision and quantized inference

## Variants
1. Lunar-v2-Large (1.2B parameters):
   - Full precision (FP16/BF16)
   - 24 transformer layers
   - 2048 hidden dimension
   - 32 attention heads

2. Lunar-v2-large (320M parameters):
   - 4-bit quantized
   - 24 transformer layers
   - 1024 hidden dimension
   - 16 attention heads

## Training Data
The models are trained on a diverse dataset including:
- Books
- Scientific papers
- Code repositories
- Web content
- Wikipedia
- Quality filtered CommonCrawl

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers
- Accelerate
- bitsandbytes (for quantization)
- flash-attn
- datasets
- wandb (for training monitoring)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
See the examples in `examples/` directory for inference and fine-tuning.

## Training
Instructions for training are in `training/README.md`
