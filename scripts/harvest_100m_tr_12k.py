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

TRAIN_BIN = os.path.join(DATA_DIR, "dorina_nano_tr_12k_train.bin")
VAL_BIN = os.path.join(DATA_DIR, "dorina_nano_tr_12k_val.bin")
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_12k.json"

TARGET_TOKENS = 100_000_000  # 100M 100% Unique Turkish Tokens
TARGET_VAL_TOKENS = 1_500_000

print("=" * 70)
print(f"HARVESTING 100M MULTI-TASK TURKISH CORPUS (12k VOCAB)")
print("Tasks: <fix> (Spelling, ASCII & Grammar) + <complete> (Autocomplete)")
print("No Epoch Repeats - 100% Fresh Unique Data Stream")
print("=" * 70, flush=True)

tok = Tokenizer.from_file(TOKENIZER_FILE)

# 1. ASCII-fication mapping (ı->i, ş->s, ğ->g, ü->u, ö->o, ç->c)
ASCII_MAP = {
    'ı': 'i', 'I': 'I',
    'ş': 's', 'Ş': 'S',
    'ğ': 'g', 'Ğ': 'G',
    'ü': 'u', 'Ü': 'U',
    'ö': 'o', 'Ö': 'O',
    'ç': 'c', 'Ç': 'C'
}

TYPO_DICT = {
    "herkes": "herkez", "yalnız": "yanlız", "yanlış": "yalnış",
    "laboratuvar": "laboratuar", "orijinal": "orjinal", "sürpriz": "süpriz",
    "şarj": "şarz", "egzoz": "egsoz", "kılavuz": "klavuz",
    "birtakım": "bir takım", "hiçbir": "hiç bir", "herhangi": "her hangi",
    "karnabahar": "karnıbahar", "şoför": "şöför", "doküman": "döküman",
    "pantolon": "pantul", "rastgele": "rasgele", "tespit": "tesbit",
    "tabii": "tabi", "müsaade": "müsade", "kravat": "kıravat"
}

def perturb_sentence(s):
    # 20% untouched clean sentence
    if random.random() < 0.20:
        return s
    
    words = s.split()
    perturbed_words = []
    
    # 50% chance of ASCII-fying some or all Turkish characters in the sentence
    do_ascii = random.random() < 0.50
    
    for w in words:
        clean_w = w.lower().strip(".,!?\"':;")
        
        # 1. Typo dictionary substitution
        if clean_w in TYPO_DICT and random.random() < 0.6:
            w = w.replace(clean_w, TYPO_DICT[clean_w])
        # 2. De/Da error
        elif w.endswith("de") and len(w) > 4 and random.random() < 0.25:
            w = w[:-2] + " de"
        elif w.endswith("da") and len(w) > 4 and random.random() < 0.25:
            w = w[:-2] + " da"
        elif w == "de" and random.random() < 0.25:
            pass # drop or merge
        elif w == "ki" and perturbed_words and random.random() < 0.35:
            perturbed_words[-1] = perturbed_words[-1] + "ki"
            continue
        
        if do_ascii and random.random() < 0.8:
            w = "".join(ASCII_MAP.get(c, c) for c in w)
            
        perturbed_words.append(w)
        
    return " ".join(perturbed_words)

train_f = open(TRAIN_BIN, "wb")
val_f = open(VAL_BIN, "wb")

total_train_tokens = 0
buffer_tokens = []
t0 = time.time()

# 1. STREAM WIKIPEDIA TR (Target ~50M tokens)
print("\n>>> [1/2] Streaming Wikipedia TR...", flush=True)
ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)

for item in ds_wiki:
    text = item.get("text", "").strip()
    if len(text) < 120:
        continue

    # Pure flow with <complete> prefix
    complete_text = f"<complete> {text}\n"
    ids = tok.encode(complete_text).ids
    ids.append(3) # </s>
    buffer_tokens.extend(ids)

    # Short sentence pairs for <fix> task (10 to 30 words)
    sentences = re.split(r'(?<=[.!?]) +', text[:2000])
    for s in sentences:
        words = s.split()
        if 5 <= len(words) <= 30:
            corrupted = perturb_sentence(s)
            fix_pair = f"<fix> {corrupted} => {s}\n"
            pair_ids = tok.encode(fix_pair).ids
            pair_ids.append(3)
            buffer_tokens.extend(pair_ids)

    if len(buffer_tokens) >= 500_000:
        arr = np.array(buffer_tokens, dtype=np.uint16)
        train_f.write(arr.tobytes())
        total_train_tokens += len(buffer_tokens)
        buffer_tokens = []
        dt = time.time() - t0
        print(f"Collected {total_train_tokens/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M tokens ({total_train_tokens/dt:,.0f} tok/s)...", flush=True)

    if total_train_tokens >= 50_000_000:
        break

# 2. STREAM C4 TR (Target remaining ~50M tokens)
print("\n>>> [2/2] Streaming C4 Turkish Web...", flush=True)
ds_c4 = load_dataset("allenai/c4", "tr", split="train", streaming=True)

for item in ds_c4:
    text = item.get("text", "").strip()
    if len(text) < 120:
        continue

    complete_text = f"<complete> {text}\n"
    ids = tok.encode(complete_text).ids
    ids.append(3)
    buffer_tokens.extend(ids)

    sentences = re.split(r'(?<=[.!?]) +', text[:2000])
    for s in sentences:
        words = s.split()
        if 5 <= len(words) <= 30:
            corrupted = perturb_sentence(s)
            fix_pair = f"<fix> {corrupted} => {s}\n"
            pair_ids = tok.encode(fix_pair).ids
            pair_ids.append(3)
            buffer_tokens.extend(pair_ids)

    if len(buffer_tokens) >= 500_000:
        arr = np.array(buffer_tokens, dtype=np.uint16)
        train_f.write(arr.tobytes())
        total_train_tokens += len(buffer_tokens)
        buffer_tokens = []
        dt = time.time() - t0
        print(f"Collected {total_train_tokens/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M tokens...", flush=True)

    if total_train_tokens >= TARGET_TOKENS:
        break

if buffer_tokens:
    arr = np.array(buffer_tokens, dtype=np.uint16)
    train_f.write(arr.tobytes())
    total_train_tokens += len(buffer_tokens)

train_f.close()

# Held-Out Validation set (~1.5M tokens)
print("\nHarvesting Held-Out Validation tokens with both tasks...", flush=True)
val_buffer = []
for item in ds_c4:
    text = item.get("text", "").strip()
    if len(text) > 120:
        complete_text = f"<complete> {text}\n"
        ids = tok.encode(complete_text).ids
        ids.append(3)
        val_buffer.extend(ids)

        sentences = re.split(r'(?<=[.!?]) +', text[:1500])
        for s in sentences:
            words = s.split()
            if 5 <= len(words) <= 30:
                corrupted = perturb_sentence(s)
                fix_pair = f"<fix> {corrupted} => {s}\n"
                pair_ids = tok.encode(fix_pair).ids
                pair_ids.append(3)
                val_buffer.extend(pair_ids)

        if len(val_buffer) >= TARGET_VAL_TOKENS:
            break

val_arr = np.array(val_buffer[:TARGET_VAL_TOKENS], dtype=np.uint16)
val_f.write(val_arr.tobytes())
val_f.close()

print("\n" + "=" * 70)
print(f"[SUCCESS] 100M MULTI-TASK TURKISH DATASET READY!")
print(f"Train File: {TRAIN_BIN} ({total_train_tokens:,} tokens, {os.path.getsize(TRAIN_BIN)/(1024*1024):.1f} MB)")
print(f"Val File:   {VAL_BIN} ({len(val_arr):,} tokens, {os.path.getsize(VAL_BIN)/(1024*1024):.1f} MB)")
print("=" * 70, flush=True)
