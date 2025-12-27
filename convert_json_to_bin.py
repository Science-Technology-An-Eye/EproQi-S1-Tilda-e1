#!/usr/bin/env python3
"""
Convert train_packed.json and val_unpacked.json to .bin format for training
"""

import os
import json
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np


def load_json_data(json_path: str):
    """Load tokenized data from JSON file"""
    print(f"Loading {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data


def flatten_tokens(tokens):
    """Recursively flatten nested token lists"""
    flat_tokens = []
    
    if isinstance(tokens, (int, np.integer)):
        return [int(tokens)]
    elif isinstance(tokens, list):
        # Check if this is a list of dicts (like [{"input_ids": [...]}, ...])
        if len(tokens) > 0 and isinstance(tokens[0], dict):
            for item in tokens:
                if isinstance(item, dict):
                    # Try common keys
                    for key in ['input_ids', 'tokens', 'data', 'token_ids']:
                        if key in item:
                            # Recursively flatten the values
                            flat_tokens.extend(flatten_tokens(item[key]))
                            break
        else:
            # Regular list, flatten recursively
            for item in tokens:
                if isinstance(item, (int, np.integer)):
                    flat_tokens.append(int(item))
                else:
                    flat_tokens.extend(flatten_tokens(item))
    elif isinstance(tokens, dict):
        # Try common keys
        for key in ['input_ids', 'tokens', 'data', 'token_ids']:
            if key in tokens:
                flat_tokens.extend(flatten_tokens(tokens[key]))
                break
    
    return flat_tokens


def convert_packed_json(json_path: str, output_dir: str, split: str = "train"):
    """
    Convert packed JSON format to .bin chunks
    Handles various JSON structures flexibly
    """
    data = load_json_data(json_path)
    
    print(f"Data type: {type(data)}")
    
    # Flatten and extract all tokens
    all_tokens = flatten_tokens(data)
    
    if not all_tokens:
        raise ValueError("No tokens found in JSON file. Please check the file format.")
    
    print(f"Total tokens after flattening: {len(all_tokens):,}")
    
    # Validate tokens are integers
    print("Validating token format...")
    try:
        # Check first few tokens
        sample = all_tokens[:10]
        print(f"Sample tokens: {sample}")
        
        # Convert to integers if needed
        all_tokens = [int(t) for t in all_tokens]
        
    except (ValueError, TypeError) as e:
        print(f"Error converting tokens to integers: {e}")
        print(f"Sample problematic data: {all_tokens[:5]}")
        raise
    
    # Convert to tensor
    print("Converting to tensor...")
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    
    # Save as single bin file for packed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}_packed.bin")
    
    print(f"Saving to {output_path}...")
    torch.save(tokens_tensor, output_path)
    print(f"✓ Saved {len(all_tokens):,} tokens")
    
    return len(all_tokens)


def convert_unpacked_json(json_path: str, output_dir: str, split: str = "val", chunk_size: int = 1000000):
    """
    Convert unpacked JSON format to .bin chunks
    Handles various JSON structures flexibly
    """
    data = load_json_data(json_path)
    
    print(f"Data type: {type(data)}")
    
    # Flatten and extract all tokens
    all_tokens = flatten_tokens(data)
    
    if not all_tokens:
        raise ValueError("No tokens found in JSON file. Please check the file format.")
    
    print(f"Total tokens after flattening: {len(all_tokens):,}")
    
    # Validate tokens are integers
    print("Validating token format...")
    try:
        # Check first few tokens
        sample = all_tokens[:10]
        print(f"Sample tokens: {sample}")
        
        # Convert to integers if needed
        all_tokens = [int(t) for t in all_tokens]
        
    except (ValueError, TypeError) as e:
        print(f"Error converting tokens to integers: {e}")
        print(f"Sample problematic data: {all_tokens[:5]}")
        raise
    
    # Convert to tensor
    print("Converting to tensor...")
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    
    # Split into chunks
    os.makedirs(output_dir, exist_ok=True)
    num_chunks = (len(all_tokens) + chunk_size - 1) // chunk_size
    
    print(f"Splitting into {num_chunks} chunks...")
    for i in tqdm(range(0, len(all_tokens), chunk_size), desc="Saving chunks"):
        chunk = tokens_tensor[i:i + chunk_size]
        chunk_idx = i // chunk_size
        output_path = os.path.join(output_dir, f"{split}_chunk_{chunk_idx:04d}.bin")
        torch.save(chunk, output_path)
    
    print(f"✓ Saved {num_chunks} chunk files")
    
    return len(all_tokens)


def inspect_json_structure(json_path: str, max_depth: int = 3):
    """Inspect and print the structure of the JSON file"""
    print(f"\n--- Inspecting {json_path} ---")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    def describe_structure(obj, depth=0, key="root"):
        indent = "  " * depth
        if depth > max_depth:
            return
        
        if isinstance(obj, dict):
            print(f"{indent}{key}: dict with {len(obj)} keys")
            for k, v in list(obj.items())[:3]:  # Show first 3 keys
                describe_structure(v, depth + 1, k)
            if len(obj) > 3:
                print(f"{indent}  ... ({len(obj) - 3} more keys)")
        elif isinstance(obj, list):
            print(f"{indent}{key}: list with {len(obj)} items")
            if len(obj) > 0:
                print(f"{indent}  First item:")
                describe_structure(obj[0], depth + 1, "[0]")
                if len(obj) > 1 and isinstance(obj[0], type(obj[1])):
                    print(f"{indent}  (remaining {len(obj) - 1} items have similar structure)")
                elif len(obj) > 1:
                    print(f"{indent}  Second item:")
                    describe_structure(obj[1], depth + 1, "[1]")
        else:
            print(f"{indent}{key}: {type(obj).__name__} = {str(obj)[:100]}")
    
    describe_structure(data)
    print()


def create_dataset_info(output_dir: str, train_tokens: int, val_tokens: int, vocab_size: int = 64000):
    """Create dataset info file"""
    info = {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": train_tokens + val_tokens,
        "vocab_size": vocab_size,
        "format": "converted from JSON"
    }
    
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Dataset info saved to {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON tokenized data to .bin format'
    )
    parser.add_argument(
        '--train-json',
        type=str,
        default='train_packed.json',
        help='Path to training data JSON'
    )
    parser.add_argument(
        '--val-json',
        type=str,
        default='val_unpacked.json',
        help='Path to validation data JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./preprocessed_data',
        help='Output directory for .bin files'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=64000,
        help='Vocabulary size (from config)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000000,
        help='Tokens per chunk for validation data'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Only inspect JSON structure without converting'
    )
    
    args = parser.parse_args()
    
    # If inspect mode, just show structure and exit
    if args.inspect:
        if os.path.exists(args.train_json):
            inspect_json_structure(args.train_json)
        if os.path.exists(args.val_json):
            inspect_json_structure(args.val_json)
        return
    
    print("="*60)
    print("Converting JSON to Binary Format")
    print("="*60)
    print(f"Training data: {args.train_json}")
    print(f"Validation data: {args.val_json}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    train_tokens = 0
    val_tokens = 0
    
    # Convert training data (packed)
    if os.path.exists(args.train_json):
        print("\n--- Converting Training Data ---")
        try:
            train_tokens = convert_packed_json(
                args.train_json,
                args.output_dir,
                split="train"
            )
        except Exception as e:
            print(f"\n❌ Error converting training data: {e}")
            print("\nRun with --inspect flag to see the JSON structure:")
            print(f"python convert_json_to_bin.py --train-json {args.train_json} --inspect")
            raise
    else:
        print(f"Warning: {args.train_json} not found, skipping training data")
    
    # Convert validation data (unpacked)
    if os.path.exists(args.val_json):
        print("\n--- Converting Validation Data ---")
        try:
            val_tokens = convert_unpacked_json(
                args.val_json,
                args.output_dir,
                split="val",
                chunk_size=args.chunk_size
            )
        except Exception as e:
            print(f"\n❌ Error converting validation data: {e}")
            print("\nRun with --inspect flag to see the JSON structure:")
            print(f"python convert_json_to_bin.py --val-json {args.val_json} --inspect")
            raise
    else:
        print(f"Warning: {args.val_json} not found, skipping validation data")
    
    # Create dataset info
    create_dataset_info(args.output_dir, train_tokens, val_tokens, args.vocab_size)
    
    print("\n" + "="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    print(f"Training tokens: {train_tokens:,}")
    print(f"Validation tokens: {val_tokens:,}")
    print(f"Total tokens: {train_tokens + val_tokens:,}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # List created files
    print("\nCreated files:")
    bin_files = sorted(Path(args.output_dir).glob("*.bin"))
    for f in bin_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()