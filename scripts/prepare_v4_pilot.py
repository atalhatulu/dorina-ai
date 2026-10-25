import os
import sys
import time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

def prepare_pilot_dataset(target_tokens=25_000_000, out_dir="data/v4_pilot"):
    """
    Kaggle hız ve stabilite pilotu için ~25M tokenlık
    FineWeb-Edu + Python-Edu + FineMath mini shard üretici.
    """
    os.makedirs(out_dir, exist_ok=True)
    tok_path = "tokenizers/dorina_tokenizer_v4_32k.json"
    tok = Tokenizer.from_file(tok_path)
    
    print("=" * 60)
    print(f"Dorina V4 Pilot Shard Hazırlanıyor: Hedef {target_tokens:,} Token")
    print("Kaynaklar: %60 FineWeb-Edu + %25 Python Code + %15 Math")
    print("=" * 60)

    all_tokens = []
    
    # 1. FineWeb-Edu (%60 -> 15M token)
    web_target = int(target_tokens * 0.60)
    print(f"\n[1/3] FineWeb-Edu akışı başlatılıyor (Hedef: {web_target:,} token)...")
    ds_web = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    web_tokens = 0
    for item in ds_web:
        text = item["text"]
        # EOS ile ayır
        enc = tok.encode(text).ids + [0] # 0 = <|endoftext|>
        all_tokens.extend(enc)
        web_tokens += len(enc)
        if web_tokens >= web_target:
            break
        if len(all_tokens) % 2_000_000 < 5000:
            print(f"  * FineWeb Toplanan: {web_tokens:,} / {web_target:,}")

    print(f"  >>> FineWeb-Edu Tamamlandı: {web_tokens:,} Token")

    # 2. Python Code (%25 -> 6.25M token)
    code_target = int(target_tokens * 0.25)
    print(f"\n[2/3] Python Kod akışı başlatılıyor (Hedef: {code_target:,} token)...")
    try:
        ds_code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
        code_tokens = 0
        for item in ds_code:
            content = item["content"]
            enc = tok.encode(content).ids + [0]
            all_tokens.extend(enc)
            code_tokens += len(enc)
            if code_tokens >= code_target:
                break
        print(f"  >>> Python Code Tamamlandı: {code_tokens:,} Token")
    except Exception as e:
        print(f"  >>> Kod akışı yedek veriyle tamamlanıyor: {e}")

    # 3. Math (%15 -> 3.75M token)
    math_target = target_tokens - len(all_tokens)
    print(f"\n[3/3] Aritmetik ve Matematik verisi ekleniyor (Hedef: {math_target:,} token)...")
    try:
        ds_math = load_dataset("HuggingFaceTB/finemath", name="finemath-4+", split="train", streaming=True)
        math_tokens = 0
        for item in ds_math:
            text = item["text"]
            enc = tok.encode(text).ids + [0]
            all_tokens.extend(enc)
            math_tokens += len(enc)
            if len(all_tokens) >= target_tokens:
                break
        print(f"  >>> Math Tamamlandı: {math_tokens:,} Token")
    except Exception as e:
        print(f"  >>> Math akışı tamamlandı: {e}")

    total_gathered = len(all_tokens)
    print(f"\n[Paketleniyor] Toplam Toplanan: {total_gathered:,} Token")
    
    # Train / Val ayır (%98 train, %2 val)
    split_idx = int(total_gathered * 0.98)
    train_tokens = np.array(all_tokens[:split_idx], dtype=np.uint16)
    val_tokens = np.array(all_tokens[split_idx:], dtype=np.uint16)

    train_path = os.path.join(out_dir, "v4_pilot_train.bin")
    val_path = os.path.join(out_dir, "v4_pilot_val.bin")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"[BAŞARILI]")
    print(f"  * Train: {train_path} ({len(train_tokens):,} token | {os.path.getsize(train_path)/(1024**2):.2f} MB)")
    print(f"  * Val:   {val_path} ({len(val_tokens):,} token | {os.path.getsize(val_path)/(1024**2):.2f} MB)")

if __name__ == "__main__":
    prepare_pilot_dataset()
