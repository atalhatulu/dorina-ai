import os
import sys
import time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 70)
print("🦀 DORINA-MoE: YENİ TOKENIZER İLE VERİ SETİ DERLEME")
print("=" * 70)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "dorina_moe_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "dorina_moe_val.bin")

# Hedef Token: 25 Milyon Train + 2.5 Milyon Val (Pilot MoE'yi Chinchilla doyumuna ulaştırmak için!)
TARGET_TRAIN = 25_000_000
TARGET_VAL   = 2_500_000

print(f"📦 Hedef Train Token: {TARGET_TRAIN:,}")
print(f"📦 Hedef Val Token:   {TARGET_VAL:,}")

train_tokens = []
val_tokens = []
total_docs = 0
t0 = time.time()

# 1. Kaynak: Python Kodları & Algoritmalar
print("\n🌐 Kaynak 1: Python Kod Havuzları İndiriliyor & Tokenize Ediliyor...")
for ds_name in ["iamtarun/python_code_instructions_18k_alpaca", "sahil2801/CodeAlpaca-20k"]:
    if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
        break
    try:
        ds = load_dataset(ds_name, split="train", streaming=True)
        for row in ds:
            inst = row.get("instruction", "")
            out = row.get("output", "")
            text = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n<|code_start|>\n{out}\n<|code_end|><|im_end|>"
            
            ids = tokenizer.encode(text).ids
            ids.append(eos_id)
            
            if len(train_tokens) < TARGET_TRAIN:
                train_tokens.extend(ids)
            elif len(val_tokens) < TARGET_VAL:
                val_tokens.extend(ids)
            else:
                break
                
            total_docs += 1
            if total_docs % 2000 == 0:
                print(f"   ↳ {total_docs:,} döküman | Train: {len(train_tokens):,} / {TARGET_TRAIN:,} | Val: {len(val_tokens):,} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"⚠️ {ds_name} atlandı: {e}")

# 2. Kaynak: CodeFeedback (Geniş Problem Çözümleri)
if len(train_tokens) < TARGET_TRAIN:
    print("\n🌐 Kaynak 2: CodeFeedback Veri Kümesi...")
    try:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
        for row in ds:
            q = row.get("query", "")
            a = row.get("answer", "")
            text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
            
            ids = tokenizer.encode(text).ids
            ids.append(eos_id)
            
            if len(train_tokens) < TARGET_TRAIN:
                train_tokens.extend(ids)
            elif len(val_tokens) < TARGET_VAL:
                val_tokens.extend(ids)
            else:
                break
                
            total_docs += 1
            if total_docs % 2000 == 0:
                print(f"   ↳ {total_docs:,} döküman | Train: {len(train_tokens):,} / {TARGET_TRAIN:,} | Val: {len(val_tokens):,} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"⚠️ CodeFeedback atlandı: {e}")

# 3. Kaynak: Yerel Bilim ve Genel Kültür (Wikitext / Metinler)
wiki_path = os.path.join(ROOT_DIR, "data", "wikitext-2.txt")
if os.path.exists(wiki_path) and (len(train_tokens) < TARGET_TRAIN or len(val_tokens) < TARGET_VAL):
    print("\n🌐 Kaynak 3: Bilim & Genel Kültür Metinleri (Wikitext)...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if len(line_str) > 40:
                ids = tokenizer.encode(line_str).ids
                ids.append(eos_id)
                if len(train_tokens) < TARGET_TRAIN:
                    train_tokens.extend(ids)
                elif len(val_tokens) < TARGET_VAL:
                    val_tokens.extend(ids)
                else:
                    break

train_arr = np.array(train_tokens[:TARGET_TRAIN], dtype=np.uint16)
val_arr   = np.array(val_tokens[:TARGET_VAL], dtype=np.uint16)

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

print("=" * 70)
print("🎉 DORINA-MoE VERİ SETİ DERLEMESİ BAŞARIYLA TAMAMLANDI!")
print(f"   ✓ Train Dosyası: {train_bin_path} ({len(train_arr):,} Token, {os.path.getsize(train_bin_path)/(1024**2):.1f} MB)")
print(f"   ✓ Val Dosyası:   {val_bin_path} ({len(val_arr):,} Token, {os.path.getsize(val_bin_path)/(1024**2):.1f} MB)")
print("=" * 70)
