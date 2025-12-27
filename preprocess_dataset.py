#!/usr/bin/env python3
"""
Preprocess and pack dataset separately before training.
WITH CHECKPOINT SUPPORT - Resume tokenization and packing separately!
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sentencepiece as spm


def preprocess_dataset(
        input_file: str,
        tokenizer_path: str,
        output_dir: str,
        max_seq_len: int = 4096,
        pack: bool = True,
        val_split: float = 0.1
):
    """Preprocess dataset into tokenized and optionally packed format."""

    print(f"\n{'=' * 70}")
    print("DATASET PREPROCESSING WITH CHECKPOINTS")
    print(f"{'=' * 70}")
    print(f"Input: {input_file}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Output dir: {output_dir}")
    print(f"Max seq len: {max_seq_len}")
    print(f"Packing: {pack}")
    print(f"Val split: {val_split}")
    print(f"{'=' * 70}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()
    print(f"✅ Tokenizer loaded (vocab_size={sp.get_piece_size()})\n")

    # Read and split dataset
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    split_idx = int(len(lines) * (1 - val_split))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    print(f"✅ Read {len(lines):,} documents")
    print(f"   Train: {len(train_lines):,}")
    print(f"   Val: {len(val_lines):,}\n")

    # Process splits
    for split_name, split_lines in [('train', train_lines), ('val', val_lines)]:
        print(f"\n{'=' * 70}")
        print(f"Processing {split_name} split...")
        print(f"{'=' * 70}\n")

        # Define checkpoint files
        tokenized_checkpoint = os.path.join(
            output_dir,
            f"{split_name}_tokenized_checkpoint.json"
        )

        # ========================================
        # PHASE 1: TOKENIZATION (with resume)
        # ========================================

        # Check if tokenization is already complete
        if os.path.exists(tokenized_checkpoint):
            print(f"📂 Found tokenized checkpoint: {tokenized_checkpoint}")
            response = input(f"Resume from this checkpoint? (y/n): ").strip().lower()

            if response == 'y':
                print("✅ Loading tokenized data from checkpoint...")
                with open(tokenized_checkpoint, 'r') as f:
                    checkpoint_data = json.load(f)
                    tokenized = checkpoint_data['tokenized']
                    skipped_short = checkpoint_data.get('skipped_short', 0)
                    skipped_long = checkpoint_data.get('skipped_long', 0)

                print(f"✅ Loaded {len(tokenized):,} tokenized examples")
                print(f"   Skipped (short): {skipped_short:,}")
                print(f"   Skipped (long): {skipped_long:,}")
                print("\n⏭️  Skipping tokenization phase...\n")
            else:
                print("🔄 Re-tokenizing from scratch...")
                tokenized = None
        else:
            tokenized = None

        # Tokenize if not loaded from checkpoint
        if tokenized is None:
            print(f"🔄 Tokenizing {len(split_lines):,} documents...")
            print("💾 Progress will be saved to checkpoint file\n")

            tokenized = []
            skipped_short = 0
            skipped_long = 0

            for line in tqdm(split_lines, desc=f"Tokenizing {split_name}"):
                # Tokenize
                tokens = sp.encode(line, out_type=int)

                # Add special tokens
                if not tokens or tokens[0] != bos_id:
                    tokens = [bos_id] + tokens
                if not tokens or tokens[-1] != eos_id:
                    tokens = tokens + [eos_id]

                # Filter
                if len(tokens) < 10:
                    skipped_short += 1
                    continue
                if len(tokens) > max_seq_len:
                    skipped_long += 1
                    continue

                tokenized.append({
                    'input_ids': tokens,
                    'length': len(tokens)
                })

            print(f"\n📊 {split_name.capitalize()} Tokenization Statistics:")
            print(f"   Valid: {len(tokenized):,}")
            print(f"   Skipped (short): {skipped_short:,}")
            print(f"   Skipped (long): {skipped_long:,}")

            # Save tokenization checkpoint
            print(f"\n💾 Saving tokenization checkpoint...")
            try:
                with open(tokenized_checkpoint, 'w') as f:
                    json.dump({
                        'tokenized': tokenized,
                        'skipped_short': skipped_short,
                        'skipped_long': skipped_long,
                        'total_processed': len(split_lines)
                    }, f)
                print(f"✅ Tokenization checkpoint saved!")
                print(f"   Location: {tokenized_checkpoint}")
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
                print("   Continuing anyway...")

        # ========================================
        # PHASE 2: PACKING (separate step)
        # ========================================

        if pack and split_name == 'train':
            print(f"\n{'=' * 70}")
            print("PACKING PHASE")
            print(f"{'=' * 70}\n")

            packing_checkpoint = os.path.join(
                output_dir,
                f"{split_name}_packing_checkpoint.json"
            )

            # Check for packing checkpoint
            if os.path.exists(packing_checkpoint):
                print(f"📂 Found packing checkpoint: {packing_checkpoint}")
                response = input(f"Resume packing from checkpoint? (y/n): ").strip().lower()

                if response == 'y':
                    print("✅ Loading partial packing results...")
                    with open(packing_checkpoint, 'r') as f:
                        pack_data = json.load(f)
                        packed = pack_data['packed']
                        last_processed = pack_data['last_processed']
                        current_tokens = pack_data.get('current_tokens', [])
                        current_length = pack_data.get('current_length', 0)

                    print(f"✅ Resuming from document {last_processed}/{len(tokenized)}")
                    print(f"   Already packed: {len(packed):,} sequences\n")
                else:
                    print("🔄 Starting packing from scratch...")
                    packed = []
                    last_processed = 0
                    current_tokens = []
                    current_length = 0
            else:
                packed = []
                last_processed = 0
                current_tokens = []
                current_length = 0

            print(f"🔄 Packing documents into sequences of {max_seq_len} tokens...")
            print(f"💾 Progress will be saved every 50,000 documents\n")

            save_interval = 1000000

            # Create progress bar
            pbar = tqdm(
                total=len(tokenized),
                initial=last_processed,
                desc="Packing"
            )

            for idx in range(last_processed, len(tokenized)):
                example = tokenized[idx]
                tokens = example['input_ids']

                if current_length + len(tokens) > max_seq_len:
                    if current_tokens:
                        packed.append({
                            'input_ids': current_tokens,
                            'length': current_length
                        })
                    current_tokens = tokens.copy()
                    current_length = len(tokens)
                else:
                    current_tokens.extend(tokens)
                    current_length += len(tokens)

                pbar.update(1)

                # Save checkpoint periodically
                if (idx + 1) % save_interval == 0:
                    try:
                        temp_file = packing_checkpoint + '.tmp'
                        with open(temp_file, 'w') as f:
                            json.dump({
                                'packed': packed,
                                'last_processed': idx + 1,
                                'current_tokens': current_tokens,
                                'current_length': current_length
                            }, f)
                        os.replace(temp_file, packing_checkpoint)
                        pbar.set_postfix({'saved': f'{len(packed)} seqs'})
                    except Exception as e:
                        pbar.write(f"⚠️  Failed to save packing checkpoint: {e}")

            pbar.close()

            # Add remaining tokens
            if current_tokens:
                packed.append({
                    'input_ids': current_tokens,
                    'length': current_length
                })

            print(f"\n✅ Packing complete!")
            print(f"   Packed into: {len(packed):,} sequences")
            print(f"   Compression: {len(packed) / len(tokenized) * 100:.1f}%")

            tokenized = packed

            # Clean up packing checkpoint after success
            if os.path.exists(packing_checkpoint):
                try:
                    os.remove(packing_checkpoint)
                    print(f"   Cleaned up packing checkpoint")
                except:
                    pass

        # ========================================
        # PHASE 3: SAVE FINAL OUTPUT
        # ========================================

        output_file = os.path.join(
            output_dir,
            f"{split_name}_{'packed' if pack and split_name == 'train' else 'unpacked'}.json"
        )

        print(f"\n💾 Saving final output to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(tokenized, f)
        print(f"✅ Saved {len(tokenized):,} examples")

        # Clean up tokenization checkpoint after final save
        if os.path.exists(tokenized_checkpoint):
            try:
                os.remove(tokenized_checkpoint)
                print(f"   Cleaned up tokenization checkpoint")
            except:
                pass

    print(f"\n{'=' * 70}")
    print("✅ PREPROCESSING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nProcessed files saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  • {output_dir}/train_packed.json")
    print(f"  • {output_dir}/val_unpacked.json")
    print(f"\nUse in training with:")
    print(f"  python train_i.py --dataset {output_dir}/train_packed.json ...")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for training with checkpoint support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with packing
  python preprocess_dataset.py \\
      --input corpus_math.txt \\
      --tokenizer eproqi_e1_tilda_i.model

  # Custom settings
  python preprocess_dataset.py \\
      --input corpus_math.txt \\
      --tokenizer eproqi_e1_tilda_i.model \\
      --output-dir ./my_preprocessed \\
      --max-seq-len 4096 \\
      --val-split 0.05

  # Without packing
  python preprocess_dataset.py \\
      --input corpus_math.txt \\
      --tokenizer eproqi_e1_tilda_i.model \\
      --no-pack

Resume Support:
  If the process is killed, just run the same command again!
  The script will detect checkpoints and ask if you want to resume.
        """
    )
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece model")
    parser.add_argument("--output-dir", default="./preprocessed", help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--no-pack", action="store_true", help="Don't pack documents")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")

    args = parser.parse_args()

    preprocess_dataset(
        args.input,
        args.tokenizer,
        args.output_dir,
        args.max_seq_len,
        not args.no_pack,
        args.val_split
    )