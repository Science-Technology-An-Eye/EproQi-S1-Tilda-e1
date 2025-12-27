"""
Memory-efficient training script with chunked sequence processing
FIXED: Import issues resolved
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import time
import gc

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import model (try different methods)
try:
    from model import Transformer, ModelArgs
except ImportError:
    print("Warning: Could not import from 'model', trying alternative...")
    import model as model_module

    Transformer = model_module.Transformer
    ModelArgs = model_module.ModelArgs


class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset that doesn't load all data into RAM"""

    def __init__(self, data_path: str, max_seq_len: int, vocab_size: int):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        print(f"Loading dataset from {data_path}...")

        # Memory-map the file instead of loading it
        if data_path.endswith('.bin'):
            self.data = torch.load(data_path, map_location='cpu', mmap=True)
        else:
            # Fallback for other formats
            self.data = torch.load(data_path, map_location='cpu')

        # Calculate number of samples
        self.num_samples = max(1, (len(self.data) - 1) // max_seq_len)

        print(f"✓ Dataset loaded: {len(self.data):,} tokens, {self.num_samples:,} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_len
        end_idx = min(start_idx + self.max_seq_len + 1, len(self.data))

        # Load only this sequence (not entire file)
        tokens = self.data[start_idx:end_idx]

        # Handle sequences that are too short
        if len(tokens) < 2:
            # Return a dummy sample
            x = torch.zeros(self.max_seq_len, dtype=torch.long)
            y = torch.zeros(self.max_seq_len, dtype=torch.long)
            return x, y

        # Ensure we have exactly max_seq_len tokens
        if len(tokens) < self.max_seq_len + 1:
            # Pad if needed
            pad_len = self.max_seq_len + 1 - len(tokens)
            tokens = torch.cat([tokens, torch.zeros(pad_len, dtype=tokens.dtype)])

        x = tokens[:-1].clone()
        y = tokens[1:].clone()

        # Validate and clamp
        x = torch.clamp(x, 0, self.vocab_size - 1)
        y = torch.clamp(y, 0, self.vocab_size - 1)

        return x, y


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return rank, world_size, local_rank
    return 0, 1, 0


def check_model_health(model, sample_input, vocab_size):
    """Check if model has common issues"""
    print("\n" + "=" * 60)
    print("MODEL HEALTH CHECK")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        logits = model(sample_input)

        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"\nLogits statistics:")
        print(f"  Mean: {logits.mean().item():.4f}")
        print(f"  Std: {logits.std().item():.4f}")
        print(f"  Min: {logits.min().item():.4f}")
        print(f"  Max: {logits.max().item():.4f}")

        # Check for issues
        issues = []

        if logits.abs().max() > 100:
            issues.append("⚠️  Logits very large (>100) - may cause saturation")

        if logits.std() < 0.01:
            issues.append("⚠️  Logits have very low variance - model may not learn")

        # Check probabilities
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        avg_max_prob = max_probs.mean().item()

        print(f"\nProbability statistics:")
        print(f"  Average max probability: {avg_max_prob:.4f}")

        if avg_max_prob > 0.99:
            issues.append("⚠️  Model too confident (avg max prob > 0.99)")
        elif avg_max_prob < 0.01:
            issues.append("⚠️  Model too uncertain (avg max prob < 0.01)")

        if issues:
            print(f"\n{'=' * 60}")
            print("ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
            print("Consider adjusting model initialization or learning rate")
        else:
            print(f"\n✓ Model health looks good!")

        print("=" * 60 + "\n")

    model.train()


def train_step(model, x, y, criterion):
    """Single training step"""
    # Forward pass - CRITICAL: must use return_all_logits=True
    logits = model(x, start_pos=0, return_all_logits=True)

    # Compute loss
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )

    return loss


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, rank, epoch,
                grad_accum_steps, max_grad_norm=1.0):
    """Memory-efficient training epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    step = 0
    nan_count = 0

    optimizer.zero_grad()

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Training step
        loss = train_step(model, x, y, criterion)

        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps

        # Check for NaN
        if not torch.isfinite(loss):
            nan_count += 1
            if nan_count < 10:
                if rank == 0:
                    print(f"⚠️  NaN/Inf loss at batch {batch_idx}, skipping...")
            elif nan_count == 10:
                if rank == 0:
                    print(f"❌ Too many NaN losses ({nan_count}). Training unstable!")
                return float('inf')
            optimizer.zero_grad()
            continue

        # Backward pass
        loss.backward()

        # Accumulate gradients
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Periodic cleanup
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            step += 1

        # Logging
        total_loss += loss.item() * grad_accum_steps * x.numel()
        total_tokens += x.numel()

        if rank == 0 and batch_idx % 50 == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            current_lr = scheduler.get_last_lr()[0]
            mem_allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024 ** 3

            print(f"Epoch {epoch} | Step {step} | Batch {batch_idx}/{len(dataloader)}")
            print(f"  Loss: {loss.item() * grad_accum_steps:.4f} | Avg: {avg_loss:.4f}")
            print(f"  LR: {current_lr:.6f} | Grad Norm: {grad_norm:.4f}" if (
                                                                                        batch_idx + 1) % grad_accum_steps == 0 else f"  LR: {current_lr:.6f}")
            print(f"  Memory: {mem_allocated:.2f}GB / {mem_reserved:.2f}GB")

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--data-file', type=str, default='./preprocessed_data/train_packed.bin')
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--check-health', action='store_true', help='Check model health before training')

    args = parser.parse_args()

    # Setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'=' * 60}")
        print(f"Config: {args.config}")
        print(f"Data: {args.data_file}")
        print(f"Output: {args.output_dir}")
        print(f"GPUs: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.grad_accum}")
        print(f"Effective batch size: {args.batch_size * args.grad_accum * world_size}")
        print(f"Max sequence length: {args.max_seq_len}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Warmup steps: {args.warmup_steps}")
        print(f"Max gradient norm: {args.max_grad_norm}")
        print(f"{'=' * 60}\n")

    # Load config
    if rank == 0:
        print(f"Loading config from {args.config}...")

    with open(args.config) as f:
        config = json.load(f)

    # Filter out comments
    model_config = {k: v for k, v in config.items() if not k.startswith('_')}
    model_args = ModelArgs(**model_config)
    model_args.max_seq_len = args.max_seq_len

    if rank == 0:
        print(f"✓ Config loaded")
        print(f"  Vocab size: {model_args.vocab_size}")
        print(f"  Model dim: {model_args.dim}")
        print(f"  Layers: {model_args.n_layers}")
        print(f"  Heads: {model_args.n_heads}")

    # Create model
    if rank == 0:
        print(f"\nCreating model...")

    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(model_args).to(device)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        if rank == 0:
            print("✓ Gradient checkpointing enabled")
        from torch.utils.checkpoint import checkpoint
        for layer in model.layers:
            original_forward = layer.forward
            layer.forward = lambda *a, l=layer, of=original_forward, **k: checkpoint(
                of, *a, use_reentrant=False, **k
            )

    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {params:,} parameters ({params / 1e6:.1f}M)\n")

    # Health check
    if args.check_health and rank == 0:
        sample_input = torch.randint(0, model_args.vocab_size, (2, 128), device=device)
        check_model_health(model, sample_input, model_args.vocab_size)

    # Dataset
    if rank == 0:
        print(f"Loading dataset...")

    dataset = MemoryMappedDataset(args.data_file, args.max_seq_len, model_args.vocab_size)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    if rank == 0:
        print(f"✓ Dataloader ready: {len(dataloader):,} batches per epoch\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Scheduler with warmup
    total_steps = len(dataloader) // args.grad_accum * args.epochs

    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(progress * math.pi)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Create output dir
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Starting training for {args.epochs} epochs...")
        print(f"Total optimization steps: {total_steps:,}\n")

    # Training loop
    best_loss = float('inf')

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch + 1}/{args.epochs}")
            print(f"{'=' * 60}")

        epoch_start = time.time()
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, criterion,
            device, rank, epoch, args.grad_accum, args.max_grad_norm
        )
        epoch_time = time.time() - epoch_start

        # Stop if training is unstable
        if not torch.isfinite(torch.tensor(avg_loss)):
            if rank == 0:
                print("\n❌ Training unstable (NaN/Inf loss), stopping early")
            break

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch + 1} COMPLETE")
            print(f"{'=' * 60}")
            print(f"Time: {epoch_time:.1f}s ({epoch_time / 60:.1f}m)")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"{'=' * 60}\n")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config,
                'args': vars(args)
            }

            # Save latest
            latest_path = os.path.join(args.output_dir, 'latest.pt')
            torch.save(checkpoint, latest_path)
            print(f"✓ Checkpoint saved: {latest_path}")

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, 'best.pt')
                torch.save(checkpoint, best_path)
                print(f"✓ New best model saved: {best_path} (loss: {best_loss:.4f})\n")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE!")
        print(f"{'=' * 60}")
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Checkpoints saved in: {args.output_dir}")
        print(f"{'=' * 60}\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()