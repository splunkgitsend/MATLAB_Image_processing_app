import os
import argparse
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
from accelerate import Accelerator
from tqdm import tqdm

from model import LunarConfig, LunarForCausalLM

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

def get_dataset(tokenizer, seq_length: int = 2048):
    # Load and combine multiple datasets
    datasets = []
    
    # Books dataset (BookCorpus)
    books = load_dataset("bookcorpus", split="train")
    datasets.append(books)
    
    # Wikipedia
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    datasets.append(wiki)
    
    # Scientific papers (S2ORC)
    papers = load_dataset("scientific_papers", "arxiv", split="train")
    datasets.append(papers)
    
    # Code (The Stack)
    code = load_dataset("bigcode/the-stack", "data", split="train")
    datasets.append(code)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Tokenize all datasets
    tokenized_datasets = [dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    ) for dataset in datasets]
    
    # Combine all datasets
    combined_dataset = torch.utils.data.ConcatDataset(tokenized_datasets)
    return combined_dataset

def train(
    config_path: Optional[str] = None,
    output_dir: str = "checkpoints",
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 3,
    warmup_steps: int = 1000,
    logging_steps: int = 100,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    max_grad_norm: float = 1.0,
):
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Setup wandb
    if accelerator.is_main_process:
        wandb.init(project="lunar-v2", name="training")
    
    # Initialize model and tokenizer
    if config_path:
        config = torch.load(config_path)
    else:
        config = LunarConfig()
    
    model = LunarForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # We'll use GPT2's tokenizer
    
    # Load dataset
    dataset = get_dataset(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Prepare everything with accelerator
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    # Training loop
    model.train()
    completed_steps = 0
    num_update_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs["loss"]
                accelerator.backward(loss)
                
                if step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                
                if completed_steps % logging_steps == 0:
                    if accelerator.is_main_process:
                        wandb.log({
                            "loss": loss.item(),
                            "step": completed_steps,
                        })
                
                if completed_steps % save_steps == 0:
                    if accelerator.is_main_process:
                        accelerator.save_state(f"{output_dir}/step_{completed_steps}")
    
    # Save final model
    if accelerator.is_main_process:
        accelerator.save_state(f"{output_dir}/final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    args = parser.parse_args()
    train(**vars(args))
