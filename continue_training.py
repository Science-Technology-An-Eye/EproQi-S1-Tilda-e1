#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model import Transformer, ModelArgs

class PackedDataset(Dataset):
    def __init__(self, data_file: str, max_seq_len: int):
        self.max_seq_len = max_seq_len
        print(f"Loading data from {data_file}...")
        self.data = torch.load(data_file, map_location='cpu')
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data)
        self.data = self.data.long()
        self.num_tokens = len(self.data)
        print(f"✓ Loaded {self.num_tokens:,} tokens")
    
    def __len__(self):
        return max(1, (self.num_tokens - 1) // self.max_seq_len)
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_len
        end_idx = min(start_idx + self.max_seq_len + 1, self.num_tokens)
        tokens = self.data[start_idx:end_idx]
        if len(tokens) < 2:
            tokens = self.data[:2]
        x = tokens[:-1]
        y = tokens[1:]
        if len(x) < self.max_seq_len:
            pad_len = self.max_seq_len - len(x)
            x = F.pad(x, (0, pad_len), value=0)
            y = F.pad(y, (0, pad_len), value=-100)
        return x, y

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✓ Loaded model weights")
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("✓ Loaded optimizer state")
        except:
            print("⚠️  Starting with fresh optimizer")
    start_epoch = checkpoint.get('epoch', 0) + 1
    global_step = checkpoint.get('step', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    print(f"✓ Resuming from epoch {start_epoch}, step {global_step}")
    return start_epoch, global_step, best_loss

def train_epoch(model, train_loader, optimizer, scheduler, epoch, global_step, args):
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.cuda() if torch.cuda.is_available() else x
        y = y.cuda() if torch.cuda.is_available() else y
        logits = model(x, start_pos=0, return_all_logits=True)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        batch_tokens = (y != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        global_step += 1
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            print(f"Epoch {epoch} | Step {global_step} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Tokens/s: {tokens_per_sec:.0f}")
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity, global_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--max-seq-len', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--save-every', type=int, default=5)
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    if "_comments" in config:
        del config["_comments"]
    model_args = ModelArgs(**config)
    model_args.max_seq_len = args.max_seq_len
    
    if model_args.dtype == "bf16":
        torch.set_default_dtype(torch.bfloat16)
    # REMOVED: torch.set_default_device(device)
    
    model = Transformer(model_args)
    model = model.to(device)  # Explicitly move model to device
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    start_epoch, global_step, best_loss = load_checkpoint(args.checkpoint, model, optimizer)
    
    train_dataset = PackedDataset(args.data_file, args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps, global_step - 1)
    
    print(f"\nTraining from epoch {start_epoch} to {start_epoch + args.epochs - 1}")
    print(f"Batch size: {args.batch_size}, Steps/epoch: {len(train_loader)}\n")
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        avg_loss, perplexity, global_step = train_epoch(model, train_loader, optimizer, scheduler, epoch, global_step, args)
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}\n")
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        if (epoch % args.save_every == 0) or is_best:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'step': global_step,
                'loss': avg_loss,
                'best_loss': best_loss,
                'config': config,
            }
            checkpoint_path = Path(args.output_dir) / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved: {checkpoint_path}")
            if is_best:
                best_path = Path(args.output_dir) / 'best.pt'
                torch.save(checkpoint, best_path)
                print(f"✓ New best! Loss: {best_loss:.4f}\n")

if __name__ == '__main__':
    main()
