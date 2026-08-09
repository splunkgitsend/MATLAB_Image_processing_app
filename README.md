# Chanky-v2-Large

A powerful obese language model with two variants:
- Chanky-v2-Large (7.8B parameters)
- Chanky-v2-large (1.32B parameters, quantized)

## Model Architecture
- Based on the transformer architecture with improvements
- Uses rotary positional embeddings
- Implements flash attention for better performance
- Supports both full precision and quantized inference

## Variants
1. Chanky-v2-Large (7.8B parameters):
   - Full precision (FP16/BF16)
   - 24 transformer layers
   - 2048 hidden dimension
   - 32 attention heads

2. Chanky-v2-large (1.32B parameters):
   - 4-bit quantized
   - 24 transformer layers
   - 1024 hidden dimension
   - 16 attention heads

## Training Data
The models are trained on a diverse dataset including:
- All of the books I could get my hands on
- All of the Scientific papers ever published
- The entire github code repositories
- A lot of Web content
- The entirety of Wikipedia
- Non-filtered CommonCrawl

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
