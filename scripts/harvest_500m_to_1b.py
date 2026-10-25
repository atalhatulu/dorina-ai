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

# 500M fresh tokens target (from token 500M to 1 Billion)
OUT_BIN = os.path.join(DATA_DIR, "dorina_nano_tr_12k_train_part3_500m.bin")
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_12k.json"

TARGET_TOKENS = 500_000_000

print("=" * 70)
print("HARVESTING 500M FRESH TURKISH CORPUS (500M -> 1 BILLION)")
print("Target: 1:100 Ultimate Chinchilla Doyum (10.4M Model -> 1B Token)")
print("Dataset: FineWeb-2 (tur_Latn) + C4 Turkish + Synthetic Fixes")
print("=" * 70, flush=True)

tok = Tokenizer.from_file(TOKENIZER_FILE)

ASCII_MAP = {
    'ı': 'i', 'I': 'I', 'ş': 's', 'Ş': 'S', 'ğ': 'g', 'Ğ': 'G',
    'ü': 'u', 'Ü': 'U', 'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
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

SPOKEN_MAP = {
    "geliyorum": "geliyom", "gidiyorum": "gidiyom", "yapıyorum": "yapıyom",
    "ediyorum": "ediyom", "biliyorum": "biliyom", "istiyorum": "istiyom",
    "geleceğim": "gelcem", "gideceğim": "gitcem", "yapacağım": "yapcam",
    "edeceğim": "etcem", "olacağım": "olcam", "göreceğim": "görcem",
    "nerede": "nerde", "burada": "burda", "orada": "orda",
    "bir şey": "bişey"
}

def perturb_sentence(s):
    if random.random() < 0.20:
        return s
    words = s.split()
    perturbed_words = []
    do_ascii = random.random() < 0.55
    for w in words:
        clean_w = w.lower().strip(".,!?\"':;")
        if clean_w in SPOKEN_MAP and random.random() < 0.7:
            w = w.replace(clean_w, SPOKEN_MAP[clean_w])
        elif clean_w in TYPO_DICT and random.random() < 0.7:
            w = w.replace(clean_w, TYPO_DICT[clean_w])
        elif w.endswith("de") and len(w) > 4 and random.random() < 0.3:
            w = w[:-2] + " de"
        elif w.endswith("da") and len(w) > 4 and random.random() < 0.3:
            w = w[:-2] + " da"
        elif do_ascii and random.random() < 0.8:
            w = "".join(ASCII_MAP.get(c, c) for c in w)
        perturbed_words.append(w)
    return " ".join(perturbed_words)

out_f = open(OUT_BIN, "wb")
total_tokens = 0
buffer_tokens = []
t0 = time.time()

# 1. STREAM FINEWEB-2 - Target 300M fresh tokens
print("\n>>> [1/2] Streaming FineWeb-2 (tur_Latn)...", flush=True)
ds_fw = load_dataset("HuggingFaceFW/fineweb-2", "tur_Latn", split="train", streaming=True)

# Skip first 60k items to guarantee 100% unseen fresh text
ds_fw_iter = iter(ds_fw)
print("Skipping seen documents to ensure 100% fresh unseen data...", flush=True)
for _ in range(60_000):
    try:
        next(ds_fw_iter)
    except StopIteration:
        break

for item in ds_fw_iter:
    text = item.get("text", "").strip()
    if len(text) < 100:
        continue

    complete_text = f"<complete> {text}\n"
    ids = tok.encode(complete_text).ids
    ids.append(3)
    buffer_tokens.extend(ids)

    sentences = re.split(r'(?<=[.!?]) +', text[:2500])
    for s in sentences:
        words = s.split()
        if 5 <= len(words) <= 30:
            corrupted = perturb_sentence(s)
            pair_ids = tok.encode(f"<fix> {corrupted} => {s}\n").ids
            pair_ids.append(3)
            buffer_tokens.extend(pair_ids)

    if len(buffer_tokens) >= 500_000:
        arr = np.array(buffer_tokens, dtype=np.uint16)
        out_f.write(arr.tobytes())
        total_tokens += len(buffer_tokens)
        buffer_tokens = []
        dt = time.time() - t0
        print(f"[FineWeb-2] Harvested: {total_tokens/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M tokens ({total_tokens/dt:,.0f} tok/s)...", flush=True)

    if total_tokens >= 300_000_000:
        break

# 2. STREAM C4 TURKISH - Target remaining 200M fresh tokens (to hit 500M)
print("\n>>> [2/2] Streaming C4 Turkish...", flush=True)
ds_c4 = load_dataset("allenai/c4", "tr", split="train", streaming=True)
ds_c4_iter = iter(ds_c4)

# Skip first 40k items
for _ in range(40_000):
    try:
        next(ds_c4_iter)
    except StopIteration:
        break

for item in ds_c4_iter:
    text = item.get("text", "").strip()
    if len(text) < 100:
        continue

    complete_text = f"<complete> {text}\n"
    ids = tok.encode(complete_text).ids
    ids.append(3)
    buffer_tokens.extend(ids)

    sentences = re.split(r'(?<=[.!?]) +', text[:2500])
    for s in sentences:
        words = s.split()
        if 5 <= len(words) <= 30:
            corrupted = perturb_sentence(s)
            pair_ids = tok.encode(f"<fix> {corrupted} => {s}\n").ids
            pair_ids.append(3)
            buffer_tokens.extend(pair_ids)

    if len(buffer_tokens) >= 500_000:
        arr = np.array(buffer_tokens, dtype=np.uint16)
        out_f.write(arr.tobytes())
        total_tokens += len(buffer_tokens)
        buffer_tokens = []
        dt = time.time() - t0
        print(f"[Total Harvest] Harvested: {total_tokens/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M tokens ({total_tokens/dt:,.0f} tok/s)...", flush=True)

    if total_tokens >= TARGET_TOKENS:
        break

if buffer_tokens:
    arr = np.array(buffer_tokens, dtype=np.uint16)
    out_f.write(arr.tobytes())
    total_tokens += len(buffer_tokens)

out_f.close()
print(f"\n[HARVEST COMPLETE] Successfully saved {total_tokens/1e6:.2f}M fresh tokens to {OUT_BIN}!", flush=True)
