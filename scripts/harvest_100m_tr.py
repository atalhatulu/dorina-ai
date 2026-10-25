import os
import re
import sys
import time
import random
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

DATA_DIR = "data/nano_tr_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_BIN = os.path.join(DATA_DIR, "dorina_nano_tr_100m_train.bin")
VAL_BIN = os.path.join(DATA_DIR, "dorina_nano_tr_val.bin")
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_8k.json"

TARGET_TOKENS = 100_000_000  # Exactly 100M fresh unique Turkish tokens
TARGET_VAL_TOKENS = 1_500_000

print("=" * 70)
print(f"HARVESTING {TARGET_TOKENS/1e6:.0f}M FRESH & DIVERSE TURKISH TOKENS")
print("Sources: 50% Wikipedia TR (Formal Grammar) + 50% C4 TR (Web & Everyday)")
print("Includes: 70% Next-Token Flow + 30% Synthetic Typo/Grammar Correction Pairs")
print("NO EPOCH REPEAT - 100% UNSEEN UNIQUE DATA STREAM")
print("=" * 70, flush=True)

tok = Tokenizer.from_file(TOKENIZER_FILE)

TYPO_DICT = {
    "herkes": "herkez", "yalnız": "yanlız", "yanlış": "yalnış",
    "laboratuvar": "laboratuar", "orijinal": "orjinal", "sürpriz": "süpriz",
    "şarj": "şarz", "egzoz": "egsoz", "kılavuz": "klavuz",
    "birtakım": "bir takım", "hiçbir": "hiç bir", "herhangi": "her hangi",
    "karnabahar": "karnıbahar", "şoför": "şöför", "doküman": "döküman",
    "pantolon": "pantul", "rastgele": "rasgele", "tespit": "tesbit"
}

def inject_turkish_errors(text):
    words = text.split()
    new_words = []
    for w in words:
        clean_w = w.lower().strip(".,!?\"':;")
        if clean_w in TYPO_DICT and random.random() < 0.6:
            w = w.replace(clean_w, TYPO_DICT[clean_w])
        elif w.endswith("de") and len(w) > 4 and random.random() < 0.2:
            w = w[:-2] + " de"
        elif w.endswith("da") and len(w) > 4 and random.random() < 0.2:
            w = w[:-2] + " da"
        elif w == "ki" and new_words and random.random() < 0.3:
            new_words[-1] = new_words[-1] + "ki"
            continue
        new_words.append(w)
    return " ".join(new_words)

# Stream writers directly to disk binary (using preallocated memmap or chunk appending)
train_f = open(TRAIN_BIN, "wb")
val_f = open(VAL_BIN, "wb")

total_train_tokens = 0
total_val_tokens = 0
t0 = time.time()

# 1. STREAM FROM WIKIPEDIA TR (Target: ~50M tokens)
print("\n>>> [1/2] Streaming Wikipedia TR...", flush=True)
ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)

buffer_tokens = []

for item in ds_wiki:
    text = item.get("text", "").strip()
    if len(text) < 120:
        continue

    # Pure flow
    ids = tok.encode(text).ids
    ids.append(3) # </s>
    buffer_tokens.extend(ids)

    # Synthetic correction pairs from sentences
    if random.random() < 0.5:
        sentences = re.split(r'(?<=[.!?]) +', text[:1500])
        for s in sentences:
            if 25 < len(s) < 180:
                err = inject_turkish_errors(s)
                if err != s:
                    pair = f"Hatalı: {err} | Düzeltme: {s}\n"
                    pair_ids = tok.encode(pair).ids
                    pair_ids.append(3)
                    buffer_tokens.extend(pair_ids)

    if len(buffer_tokens) >= 500_000:
        arr = np.array(buffer_tokens, dtype=np.uint16)
        train_f.write(arr.tobytes())
        total_train_tokens += len(buffer_tokens)
        buffer_tokens = []
        dt = time.time() - t0
        print(f"Harvested {total_train_tokens/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M tokens ({len(arr)/(dt):,.0f} tok/s)...", flush=True)

    if total_train_tokens >= 50_000_000:
        break

# 2. STREAM FROM C4 TR (Target: remaining ~50M tokens)
print("\n>>> [2/2] Streaming C4 Turkish Web...", flush=True)
ds_c4 = load_dataset("allenai/c4", "tr", split="train", streaming=True)

for item in ds_c4:
    text = item.get("text", "").strip()
    if len(text) < 120:
        continue

    ids = tok.encode(text).ids
    ids.append(3)
    buffer_tokens.extend(ids)

    if random.random() < 0.5:
        sentences = re.split(r'(?<=[.!?]) +', text[:1500])
        for s in sentences:
            if 25 < len(s) < 180:
                err = inject_turkish_errors(s)
                if err != s:
                    pair = f"Hatalı: {err} | Düzeltme: {s}\n"
                    pair_ids = tok.encode(pair).ids
                    pair_ids.append(3)
                    buffer_tokens.extend(pair_ids)

    if len(buffer_tokens) >= 500_000:
        arr = np.array(buffer_tokens, dtype=np.uint16)
        train_f.write(arr.tobytes())
        total_train_tokens += len(buffer_tokens)
        buffer_tokens = []
        dt = time.time() - t0
        print(f"Harvested {total_train_tokens/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M tokens...", flush=True)

    if total_train_tokens >= TARGET_TOKENS:
        break

# Write remainder
if buffer_tokens:
    arr = np.array(buffer_tokens, dtype=np.uint16)
    train_f.write(arr.tobytes())
    total_train_tokens += len(buffer_tokens)

train_f.close()

# Harvest clean Val set (~1.5M tokens)
print("\nHarvesting Held-Out Validation tokens...", flush=True)
val_buffer = []
for item in ds_c4:
    text = item.get("text", "").strip()
    if len(text) > 120:
        ids = tok.encode(text).ids
        ids.append(3)
        val_buffer.extend(ids)
        if len(val_buffer) >= TARGET_VAL_TOKENS:
            break

val_arr = np.array(val_buffer[:TARGET_VAL_TOKENS], dtype=np.uint16)
val_f.write(val_arr.tobytes())
val_f.close()

print("\n" + "=" * 70)
print(f"[SUCCESS] 100M TURKISH DATASET READY!")
print(f"Train File: {TRAIN_BIN} ({total_train_tokens:,} tokens, {os.path.getsize(TRAIN_BIN)/(1024*1024):.1f} MB)")
print(f"Val File:   {VAL_BIN} ({len(val_arr):,} tokens, {os.path.getsize(VAL_BIN)/(1024*1024):.1f} MB)")
print("=" * 70, flush=True)
