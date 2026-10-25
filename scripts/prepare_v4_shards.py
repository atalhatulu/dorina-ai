import os
import sys
import time
import argparse
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

# Constants
TOKENIZER_PATH = "tokenizers/dorina_tokenizer_v4_32k.json"
OUT_DIR = "data/v4_shards"

def tokenize_and_pack(dataset_stream, tokenizer, target_tokens, out_path, text_column="text"):
    print(f"\nTargeting {target_tokens:,} tokens -> {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    tokens_buffer = []
    total_written = 0
    start_time = time.time()
    
    # We will append to a binary file directly
    with open(out_path, "wb") as f_out:
        for idx, item in enumerate(dataset_stream):
            text = item.get(text_column, "")
            if not text or len(text.strip()) < 10:
                continue
                
            enc = tokenizer.encode(text)
            # Prepend EOS (0) or wrap properly
            tokens_buffer.extend(enc.ids)
            tokens_buffer.append(0)  # <|endoftext|>
            
            if len(tokens_buffer) >= 1_000_000:
                arr = np.array(tokens_buffer, dtype=np.uint16)
                f_out.write(arr.tobytes())
                total_written += len(tokens_buffer)
                tokens_buffer = []
                
                elapsed = time.time() - start_time
                speed = total_written / max(1, elapsed)
                print(f"  Processed {total_written:,} / {target_tokens:,} tokens ({total_written/target_tokens*100:.1f}%) | Speed: {speed:,.0f} tok/s", end="\r")
                
                if total_written >= target_tokens:
                    break
                    
        if len(tokens_buffer) > 0 and total_written < target_tokens:
            arr = np.array(tokens_buffer, dtype=np.uint16)
            f_out.write(arr.tobytes())
            total_written += len(tokens_buffer)

    print(f"\nFinished: {total_written:,} tokens written to {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
    return total_written

def main():
    parser = argparse.ArgumentParser(description="Dorina V4 Multi-Shard Generator")
    parser.add_argument("--tokens-per-shard", type=int, default=500_000_000, help="Tokens per shard (default: 500M ~ 1GB bin)")
    parser.add_argument("--shards", type=int, default=1, help="Number of shards to generate")
    args = parser.parse_args()

    print("=" * 65)
    print("DORINA V4 HIGH-QUALITY ENGLISH SHARD PIPELINE")
    print("Dataset: HuggingFaceFW/fineweb-edu (Sample-10BT, Score >= 3.0)")
    print(f"Target: {args.shards} Shard(s) of {args.tokens_per_shard:,} tokens each")
    print("=" * 65)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # Streaming dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # Filter high educational score >= 3
    filtered_ds = (item for item in ds if item.get("score", 0) >= 3.0)

    for shard_idx in range(1, args.shards + 1):
        shard_path = os.path.join(OUT_DIR, f"v4_shard_{shard_idx}.bin")
        tokenize_and_pack(filtered_ds, tokenizer, args.tokens_per_shard, shard_path, text_column="text")

if __name__ == "__main__":
    main()
