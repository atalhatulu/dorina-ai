import os
import sys
import time
import numpy as np
import tiktoken
from datasets import load_dataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 70)
print("🐍 DORINA-278M: AŞAMA 3 KOD & MANTIK VERİ SETİ HAZIRLIĞI")
print("=" * 70)

enc = tiktoken.get_encoding("gpt2")
eos_id = enc.eot_token

train_bin_path = os.path.join(DATA_DIR, "dorina278m_phase3_code_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "dorina278m_phase3_code_val.bin")

# Hedef: ~10M Token (RTX 4050 için 1 saatlik hızlı ve tam doyumlu kod eğitimi)
TARGET_TRAIN_TOKENS = 10_000_000
TARGET_VAL_TOKENS   = 1_000_000

print(f"📦 Hedef Train Token: {TARGET_TRAIN_TOKENS:,}")
print(f"📦 Hedef Val Token:   {TARGET_VAL_TOKENS:,}")

datasets_sources = [
    ("iamtarun/python_code_instructions_18k_alpaca", "train"),
    ("sahil2801/CodeAlpaca-20k", "train"),
    ("m-a-p/CodeFeedback-Filtered-Instruction", "train")
]

train_tokens = []
val_tokens = []
total_docs = 0
t0 = time.time()

for ds_name, split in datasets_sources:
    if len(train_tokens) >= TARGET_TRAIN_TOKENS and len(val_tokens) >= TARGET_VAL_TOKENS:
        break
    print(f"\n🌐 İndiriliyor & Tokenize Ediliyor: {ds_name}...")
    try:
        ds = load_dataset(ds_name, split=split, streaming=True)
        for row in ds:
            # Formatlama: Kod + Açıklama + Fonksiyon
            if "prompt" in row and row["prompt"]:
                text = row["prompt"]
            elif "instruction" in row:
                inst = row.get("instruction", "")
                inp = row.get("input", "")
                out = row.get("output", "")
                text = f"### Instruction:\n{inst}\n"
                if inp:
                    text += f"### Input:\n{inp}\n"
                text += f"### Response:\n{out}"
            elif "query" in row and "answer" in row:
                text = f"### Problem:\n{row['query']}\n### Solution:\n{row['answer']}"
            else:
                continue

            if len(text.strip()) < 30:
                continue

            tokens = enc.encode_ordinary(text)
            tokens.append(eos_id)

            if len(train_tokens) < TARGET_TRAIN_TOKENS:
                train_tokens.extend(tokens)
            elif len(val_tokens) < TARGET_VAL_TOKENS:
                val_tokens.extend(tokens)
            else:
                break

            total_docs += 1
            if total_docs % 1000 == 0:
                print(f"   ↳ {total_docs:,} kod dökümanı | Train: {len(train_tokens):,} / {TARGET_TRAIN_TOKENS:,} | Val: {len(val_tokens):,} / {TARGET_VAL_TOKENS:,} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"⚠️ {ds_name} okunurken hata: {e}")

print(f"\nToplam toplanan döküman: {total_docs:,}")
train_arr = np.array(train_tokens[:TARGET_TRAIN_TOKENS], dtype=np.uint16)
val_arr   = np.array(val_tokens[:TARGET_VAL_TOKENS], dtype=np.uint16)

print("💾 İkili (uint16) dosyalar kaydediliyor...")
train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

print("=" * 70)
print("🎉 AŞAMA 3 KOD VERİ SETİ HAZIR!")
print(f"   ✓ Train: {train_bin_path} ({len(train_arr):,} Token, {os.path.getsize(train_bin_path)/(1024**2):.1f} MB)")
print(f"   ✓ Val:   {val_bin_path} ({len(val_arr):,} Token, {os.path.getsize(val_bin_path)/(1024**2):.1f} MB)")
print("=" * 70)
