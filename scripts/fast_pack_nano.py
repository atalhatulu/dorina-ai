import os
import sys
import time
import numpy as np
from tokenizers import Tokenizer

def pack_stories_slice(input_path, out_bin_path, start_mb, size_mb, tokenizer):
    print(f"\n[PACKER] Extracting {size_mb}MB from offset {start_mb}MB -> {out_bin_path}...")
    t0 = time.time()
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(start_mb * 1024 * 1024)
        text = f.read(size_mb * 1024 * 1024)
        
    stories = [s.strip() for s in text.split('\n\n') if len(s.strip()) > 10]
    del text
    print(f"  Parsed {len(stories):,} stories in {time.time()-t0:.2f}s. Batch tokenizing...")
    
    t1 = time.time()
    encs = tokenizer.encode_batch(stories)
    all_ids = []
    for e in encs:
        all_ids.extend(e.ids)
        all_ids.append(0) # EOS
        
    arr = np.array(all_ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    arr.tofile(out_bin_path)
    tok_s = len(all_ids) / max(1e-4, time.time() - t1)
    print(f"  SUCCESS: {len(all_ids):,} tokens packed ({len(all_ids)/(1024*1024)*2:.1f} MB) in {time.time()-t0:.1f}s ({tok_s:,.0f} tok/s)!")

def main():
    tok = Tokenizer.from_file("tokenizers/dorina_tokenizer_v4_32k.json")
    src = "data/TinyStoriesV2-GPT4-train.txt"
    # Stage 1: Part 1 to 4 (Each 150MB text ~ 40M tokens -> ~160M tokens)
    pack_stories_slice(src, "data/nano_pack/nano_part2_40m.bin", 150, 150, tok)
    pack_stories_slice(src, "data/nano_pack/nano_part3_40m.bin", 300, 150, tok)
    pack_stories_slice(src, "data/nano_pack/nano_part4_40m.bin", 450, 150, tok)
    pack_stories_slice(src, "data/nano_pack/nano_part5_40m.bin", 600, 150, tok)

if __name__ == "__main__":
    main()
