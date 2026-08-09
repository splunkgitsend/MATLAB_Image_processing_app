import torch
from transformers import AutoTokenizer
from model import LunarForCausalLM

def generate_text(
    model_path: str,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
):
    # Load model
    checkpoint = torch.load(model_path)
    model = LunarForCausalLM(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()
    
    generated_text = generate_text(**vars(args))
    print(generated_text)
