#!/usr/bin/env python3
"""
Complete pipeline: Load train_packed.json → Split train/val → Convert to .bin
Optimized for format: [{"input_ids": [...], "length": ...}, ...]
"""

import os
import json
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import sys


class DataPipeline:
    """Handle the complete data processing pipeline"""
    
    def __init__(self, input_json: str, output_dir: str, val_ratio: float = 0.05, 
                 chunk_size: int = 1000000, vocab_size: int = 64000):
        self.input_json = input_json
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        
        self.train_tokens = []
        self.val_tokens = []
    
    def load_and_flatten(self):
        """Step 1: Load JSON and flatten tokens"""
        print("\n" + "="*60)
        print("STEP 1: Loading and Flattening Data")
        print("="*60)
        
        if not os.path.exists(self.input_json):
            raise FileNotFoundError(f"Input file not found: {self.input_json}")
        
        print(f"Loading {self.input_json}...")
        with open(self.input_json, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data):,} documents")
        
        # Extract all tokens from input_ids
        print("Extracting tokens from input_ids...")
        all_tokens = []
        
        for i, item in enumerate(tqdm(data, desc="Processing documents")):
            if isinstance(item, dict) and 'input_ids' in item:
                tokens = item['input_ids']
                # Filter out special tokens if needed (e.g., -1)
                tokens = [int(t) for t in tokens if t >= 0]
                all_tokens.extend(tokens)
            elif i < 5:  # Show warning for first few items only
                print(f"⚠️  Warning: Item {i} has unexpected format: {type(item)}")
        
        if not all_tokens:
            raise ValueError("No tokens found! Check your JSON structure.")
        
        print(f"✓ Extracted {len(all_tokens):,} tokens")
        
        # Validate tokens
        print("Validating tokens...")
        max_token = max(all_tokens)
        min_token = min(all_tokens)
        unique_tokens = len(set(all_tokens))
        
        print(f"  Token range: [{min_token}, {max_token}]")
        print(f"  Unique tokens: {unique_tokens:,}")
        print(f"  Sample (first 20): {all_tokens[:20]}")
        
        if max_token >= self.vocab_size:
            print(f"⚠️  WARNING: Max token {max_token} >= vocab_size {self.vocab_size}")
            print(f"    Consider setting --vocab-size to at least {max_token + 1}")
        
        return all_tokens
    
    def split_data(self, all_tokens):
        """Step 2: Split into train/val"""
        print("\n" + "="*60)
        print("STEP 2: Splitting Train/Validation")
        print("="*60)
        
        split_idx = int(len(all_tokens) * (1 - self.val_ratio))
        self.train_tokens = all_tokens[:split_idx]
        self.val_tokens = all_tokens[split_idx:]
        
        print(f"Split index: {split_idx:,}")
        print(f"Training tokens: {len(self.train_tokens):,} ({(1-self.val_ratio)*100:.1f}%)")
        print(f"Validation tokens: {len(self.val_tokens):,} ({self.val_ratio*100:.1f}%)")
        print(f"✓ Split complete")
    
    def save_binary(self):
        """Step 3: Convert and save as .bin files"""
        print("\n" + "="*60)
        print("STEP 3: Converting to Binary Format")
        print("="*60)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save training data (single packed file)
        print("\n[Training Data]")
        print(f"Converting {len(self.train_tokens):,} tokens to tensor...")
        train_tensor = torch.tensor(self.train_tokens, dtype=torch.long)
        train_path = os.path.join(self.output_dir, "train_packed.bin")
        
        print(f"Saving to {train_path}...")
        torch.save(train_tensor, train_path)
        
        train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
        print(f"✓ Saved train_packed.bin ({train_size_mb:.2f} MB)")
        
        # Save validation data (multiple chunks)
        print("\n[Validation Data]")
        print(f"Converting {len(self.val_tokens):,} tokens to tensor...")
        val_tensor = torch.tensor(self.val_tokens, dtype=torch.long)
        num_chunks = (len(self.val_tokens) + self.chunk_size - 1) // self.chunk_size
        
        print(f"Splitting into {num_chunks} chunks of {self.chunk_size:,} tokens each...")
        
        for i in tqdm(range(0, len(self.val_tokens), self.chunk_size), 
                      desc="Saving val chunks", unit="chunk"):
            chunk = val_tensor[i:i + self.chunk_size]
            chunk_idx = i // self.chunk_size
            chunk_path = os.path.join(self.output_dir, f"val_chunk_{chunk_idx:04d}.bin")
            torch.save(chunk, chunk_path)
        
        print(f"✓ Saved {num_chunks} validation chunks")
        
        return num_chunks
    
    def save_validation_json(self):
        """Step 4: Save validation JSON (optional)"""
        print("\n" + "="*60)
        print("STEP 4: Saving Validation JSON (Optional)")
        print("="*60)
        
        val_json_path = os.path.join(self.output_dir, "val_unpacked.json")
        print(f"Saving to {val_json_path}...")
        
        with open(val_json_path, 'w') as f:
            json.dump(self.val_tokens, f)
        
        val_json_size_mb = os.path.getsize(val_json_path) / (1024 * 1024)
        print(f"✓ Saved val_unpacked.json ({val_json_size_mb:.2f} MB)")
    
    def save_dataset_info(self, num_chunks):
        """Step 5: Save dataset metadata"""
        print("\n" + "="*60)
        print("STEP 5: Saving Dataset Info")
        print("="*60)
        
        info = {
            "source_file": self.input_json,
            "train_tokens": len(self.train_tokens),
            "val_tokens": len(self.val_tokens),
            "total_tokens": len(self.train_tokens) + len(self.val_tokens),
            "val_ratio": self.val_ratio,
            "num_val_chunks": num_chunks,
            "chunk_size": self.chunk_size,
            "vocab_size": self.vocab_size,
            "format": "PyTorch tensors (.bin)",
            "files": {
                "train": "train_packed.bin",
                "val_chunks": [f"val_chunk_{i:04d}.bin" for i in range(num_chunks)],
                "val_json": "val_unpacked.json"
            }
        }
        
        info_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Saved dataset_info.json")
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE!")
        print("="*60)
        print(f"Training tokens:   {len(self.train_tokens):>12,}")
        print(f"Validation tokens: {len(self.val_tokens):>12,}")
        print(f"Total tokens:      {len(self.train_tokens) + len(self.val_tokens):>12,}")
        print(f"Validation ratio:  {self.val_ratio:>12.1%}")
        print(f"Output directory:  {self.output_dir}")
        print("="*60)
        
        # List all created files
        print("\nCreated files:")
        all_files = sorted(Path(self.output_dir).glob("*"))
        
        total_size = 0
        for f in all_files:
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  {f.name:<30} {size_mb:>8.2f} MB")
        
        print(f"  {'─'*30} {'─'*11}")
        print(f"  {'Total:':<30} {total_size:>8.2f} MB")
        
        print("\n" + "="*60)
        print("Next Steps:")
        print("="*60)
        print("1. Verify the data:")
        print(f"   python verify_data.py --data-dir {self.output_dir}")
        print("\n2. Start training:")
        print(f"   python train.py --data-dir {self.output_dir}")
        print("="*60 + "\n")
    
    def run(self, save_json: bool = True):
        """Run the complete pipeline"""
        try:
            # Step 1: Load and flatten
            all_tokens = self.load_and_flatten()
            
            # Step 2: Split
            self.split_data(all_tokens)
            
            # Step 3: Save binary
            num_chunks = self.save_binary()
            
            # Step 4: Save JSON (optional)
            if save_json:
                self.save_validation_json()
            
            # Step 5: Save metadata
            self.save_dataset_info(num_chunks)
            
            # Summary
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


def inspect_json(json_path: str, max_items: int = 3):
    """Inspect JSON structure"""
    print("\n" + "="*60)
    print("JSON STRUCTURE INSPECTION")
    print("="*60)
    print(f"File: {json_path}\n")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Root type: {type(data)}")
    print(f"Number of items: {len(data) if isinstance(data, (list, dict)) else 'N/A'}\n")
    
    if isinstance(data, list):
        print(f"Showing first {max_items} items:\n")
        for i in range(min(max_items, len(data))):
            item = data[i]
            print(f"Item [{i}]:")
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, list):
                        print(f"  '{key}': list with {len(value)} items")
                        if len(value) > 0:
                            sample = value[:10]
                            print(f"    Sample: {sample}")
                    else:
                        print(f"  '{key}': {type(value).__name__} = {value}")
            else:
                print(f"  Type: {type(item)}, Value: {str(item)[:100]}")
            print()
    
    # Token statistics
    if isinstance(data, list) and len(data) > 0:
        print("\n" + "="*60)
        print("TOKEN STATISTICS")
        print("="*60)
        
        total_tokens = 0
        total_docs = 0
        min_len = float('inf')
        max_len = 0
        
        for item in data:
            if isinstance(item, dict) and 'input_ids' in item:
                tokens = [t for t in item['input_ids'] if t >= 0]
                doc_len = len(tokens)
                total_tokens += doc_len
                total_docs += 1
                min_len = min(min_len, doc_len)
                max_len = max(max_len, doc_len)
        
        if total_docs > 0:
            avg_len = total_tokens / total_docs
            print(f"Total documents: {total_docs:,}")
            print(f"Total tokens: {total_tokens:,}")
            print(f"Average doc length: {avg_len:.1f} tokens")
            print(f"Min doc length: {min_len:,} tokens")
            print(f"Max doc length: {max_len:,} tokens")


def verify_data(data_dir: str):
    """Verify converted binary data"""
    print("\n" + "="*60)
    print("VERIFYING CONVERTED DATA")
    print("="*60)
    
    # Load train data
    train_path = os.path.join(data_dir, "train_packed.bin")
    if os.path.exists(train_path):
        print(f"\n[Training Data]")
        print(f"Loading {train_path}...")
        train_data = torch.load(train_path)
        print(f"✓ Shape: {train_data.shape}")
        print(f"  Dtype: {train_data.dtype}")
        print(f"  Min: {train_data.min().item()}")
        print(f"  Max: {train_data.max().item()}")
        print(f"  Sample: {train_data[:20].tolist()}")
    
    # Load val chunks
    val_chunks = sorted(Path(data_dir).glob("val_chunk_*.bin"))
    if val_chunks:
        print(f"\n[Validation Data]")
        print(f"Found {len(val_chunks)} chunks")
        print(f"Loading first chunk: {val_chunks[0].name}...")
        val_data = torch.load(val_chunks[0])
        print(f"✓ Shape: {val_data.shape}")
        print(f"  Dtype: {val_data.dtype}")
        print(f"  Min: {val_data.min().item()}")
        print(f"  Max: {val_data.max().item()}")
        print(f"  Sample: {val_data[:20].tolist()}")
    
    print("\n✓ Data verification complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline: Split train_packed.json and convert to .bin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect JSON structure first
  python pipeline.py --input train_packed.json --inspect
  
  # Basic usage (5% validation)
  python pipeline.py --input train_packed.json --output-dir ./data
  
  # Custom validation split (10%)
  python pipeline.py --input train_packed.json --val-ratio 0.1
  
  # Skip saving validation JSON to save disk space
  python pipeline.py --input train_packed.json --no-json
  
  # Verify converted data
  python pipeline.py --verify --output-dir ./data
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='./preprocessed_data/train_packed.json',
        help='Path to input train_packed.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./preprocessed_data',
        help='Output directory for .bin files (default: ./preprocessed_data)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.05,
        help='Validation ratio (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000000,
        help='Tokens per validation chunk (default: 1M)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=64000,
        help='Vocabulary size (default: 64000)'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip saving val_unpacked.json (saves disk space)'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Only inspect JSON structure without processing'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify converted binary data'
    )
    
    args = parser.parse_args()
    
    # Verify mode
    if args.verify:
        verify_data(args.output_dir)
        sys.exit(0)
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Inspect mode
    if args.inspect:
        inspect_json(args.input, max_items=5)
        sys.exit(0)
    
    # Run pipeline
    print(f"\n🚀 Starting data pipeline...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    
    pipeline = DataPipeline(
        input_json=args.input,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        chunk_size=args.chunk_size,
        vocab_size=args.vocab_size
    )
    
    success = pipeline.run(save_json=not args.no_json)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()