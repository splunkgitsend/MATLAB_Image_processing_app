import torch
import bitsandbytes as bnb
from model import LunarConfig, LunarForCausalLM

def quantize_model(model_path: str, output_path: str):
    """
    Quantize the model to 4-bit precision using bitsandbytes
    """
    # Load the original model
    checkpoint = torch.load(model_path)
    
    # Create medium config
    config = LunarConfig(
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
    )
    
    # Initialize new model with medium config
    model = LunarForCausalLM(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Convert linear layers to 4-bit
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float16,
                compress_statistics=True,
                device=None,
            )
    
    # Save quantized model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    quantize_model(args.model_path, args.output_path)
