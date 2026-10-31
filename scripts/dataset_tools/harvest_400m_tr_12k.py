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

# 400M fresh tokens target (from token 100M to 500M)
TRAIN_BIN = os.path.join(DATA_DIR, "dorina_nano_tr_12k_train_part2.bin")
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_12k.json"

TARGET_TOKENS = 400_000_000

print("=" * 70)
print(f"HARVESTING 400M MULTI-TASK TURKISH CORPUS (FineWeb-2 + C4 + Speech Fixes)")
print("Target: 400M Fresh Tokens (Completing 500M Journey)")
print("=" * 70, flush=True)

tok = Tokenizer.from_file(TOKENIZER_FILE)

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
    "tabii": "tabi", "müsaade": "müsade", "kravat": "kıravat",
    "profesör": "profösör", "kravat": "kıravat", "muhatap": "muhattap",
    "kolej": "kolej", "karnaval": "karnaval", "katalog": "katolog",
    "makine": "makina", "meyve": "meyva", "silahşor": "silahşör",
    "hıristiyan": "hristiyan", "dinazor": "dinozor"
}

# Colloquial spoken forms to formal written forms
SPOKEN_MAP = {
    "geliyorum": "geliyom", "gidiyorum": "gidiyom", "yapıyorum": "yapıyom",
    "ediyorum": "ediyom", "biliyorum": "biliyom", "istiyorum": "istiyom",
    "geleceğim": "gelcem", "gideceğim": "gitcem", "yapacağım": "yapcam",
    "edeceğim": "etcem", "olacağım": "olcam", "göreceğim": "görcem",
    "nerede": "nerde", "burada": "burda", "orada": "orda",
    "bir şey": "bişey", "şey": "şey"
}

def perturb_sentence(s):
    # 20% clean/untouched
    if random.random() < 0.20:
        return s
    
    words = s.split()
    perturbed_words = []
    
    do_ascii = random.random() < 0.55
    
    for w in words:
        clean_w = w.lower().strip(".,!?\"':;")
        
        # 1. Colloquial / Spoken noise (e.g. geliyom -> geliyorum)
        if clean_w in SPOKEN_MAP and random.random() < 0.7:
            w = w.replace(clean_w, SPOKEN_MAP[clean_w])
        # 2. Common Typo substitution
        elif clean_w in TYPO_DICT and random.random() < 0.7:
            w = w.replace(clean_w, TYPO_DICT[clean_w])
        # 3. De/Da separate/combine errors
        elif w.endswith("de") and len(w) > 4 and random.random() < 0.3:
            w = w[:-2] + " de"
        elif w.endswith("da") and len(w) > 4 and random.random() < 0.3:
            w = w[:-2] + " da"
        elif w == "de" and random.random() < 0.3:
            pass
        elif w == "da" and random.random() < 0.3:
            pass
        elif w == "ki" and perturbed_words and random.random() < 0.35:
            perturbed_words[-1] = perturbed_words[-1] + "ki"
            continue
        
        # 4. ASCII-fication
        if do_ascii and random.random() < 0.8:
            w = "".join(ASCII_MAP.get(c, c) for c in w)
            
        perturbed_words.append(w)
        
    return " ".join(perturbed_words)

train_f = open(TRAIN_BIN, "wb")
total_train_tokens = 0
buffer_tokens = []
t0 = time.time()

# 1. STREAM FINEWEB-2 (TURKISH LATN) - Target 250M tokens
print("\n>>> [1/2] Streaming FineWeb-2 (tur_Latn)...", flush=True)
ds_fw = load_dataset("HuggingFaceFW/fineweb-2", "tur_Latn", split="train", streaming=True)

for item in ds_fw:
    text = item.get("text", "").strip()
    if len(text) < 100:
        continue

    # LM completion stream
    complete_text = f"<complete> {text}\n"
    ids = tok.encode(complete_text).ids
    ids.append(3) # </s>
    buffer_tokens.extend(ids)

    # Synthetic correction pairs
    sentences = re.split(r'(?<=[.!?]) +', text[:2500])
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
        print(f"[FineWeb-2] Progress: {total_train_tokens/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M tokens ({total_train_tokens/dt:,.0f} tok/s)...", flush=True)

    if total_train_tokens >= 250_000_000:
        break

# 2. STREAM C4 TURKISH - Target remaining 150M tokens (to hit 400M)
print("\n>>> [2/2] Streaming C4 Turkish...", flush=True)
ds_c4 = load_dataset("allenai/c4", "tr", split="train", streaming=True)

for item in ds_c4:
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
        print(f"[C4 + FineWeb] Total: {total_train_tokens/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M tokens ({total_train_tokens/dt:,.0f} tok/s)...", flush=True)

    if total_train_tokens >= TARGET_TOKENS:
        break

if buffer_tokens:
    arr = np.array(buffer_tokens, dtype=np.uint16)
    train_f.write(arr.tobytes())
    total_train_tokens += len(buffer_tokens)

train_f.close()
print(f"\n🎉 Successfully harvested {total_train_tokens/1e6:.2f}M tokens into {TRAIN_BIN}!")
