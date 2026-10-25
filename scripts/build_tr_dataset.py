import os
import re
import random
import numpy as np
from tokenizers import Tokenizer

RAW_FILE = "data/nano_tr_data/turkish_corpus.txt"
OUT_BIN = "data/nano_tr_data/dorina_nano_tr_train.bin"
VAL_BIN = "data/nano_tr_data/dorina_nano_tr_val.bin"
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_8k.json"

print("=" * 65)
print("CREATING TURKISH TRAINING DATASET (AUTOCOMPLETE + SYNTHETIC SPELLCHECK)")
print("=" * 65, flush=True)

tok = Tokenizer.from_file(TOKENIZER_FILE)

# Common Turkish typo & grammar perturbation engine
TYPO_DICT = {
    "herkes": "herkez", "yalnız": "yanlız", "yanlış": "yalnış",
    "laboratuvar": "laboratuar", "orijinal": "orjinal", "sürpriz": "süpriz",
    "şarj": "şarz", "egzoz": "egsoz", "kılavuz": "klavuz",
    "birtakım": "bir takım", "hiçbir": "hiç bir", "herhangi": "her hangi"
}

def inject_turkish_errors(text):
    words = text.split()
    new_words = []
    for w in words:
        clean_w = w.lower().strip(".,!?\"':;")
        # 1. Common typo substitution
        if clean_w in TYPO_DICT and random.random() < 0.6:
            rep = TYPO_DICT[clean_w]
            w = w.replace(clean_w, rep)
        # 2. De/Da error: "evde" -> "ev de" OR "sen de" -> "sende"
        elif w.endswith("de") and len(w) > 4 and random.random() < 0.15:
            w = w[:-2] + " de"
        elif w.endswith("da") and len(w) > 4 and random.random() < 0.15:
            w = w[:-2] + " da"
        elif w == "de" and random.random() < 0.2:
            pass # sometimes dropped or attached
        # 3. Ki error: "biliyorum ki" -> "biliyorumki"
        elif w == "ki" and new_words and random.random() < 0.3:
            new_words[-1] = new_words[-1] + "ki"
            continue
        new_words.append(w)
    return " ".join(new_words)

all_tokens = []
with open(RAW_FILE, "r", encoding="utf-8") as f:
    text_data = f.read()

articles = [a.strip() for a in text_data.split("\n\n") if len(a.strip()) > 100]
print(f"Total clean articles to process: {len(articles)}")

# 1. 70% Pure Clean Turkish for next-token prediction & autocomplete
for a in articles:
    enc = tok.encode(a).ids
    all_tokens.extend(enc)
    all_tokens.append(3) # </s> end of text token

# 2. 30% Grammar Correction Pair formatting: "Hatalı: {bozuk} -> Düzeltilmiş: {temiz}"
print("Generating synthetic grammar/spellcheck pairs...")
for a in articles[:4000]:
    sentences = re.split(r'(?<=[.!?]) +', a)
    for s in sentences:
        if 20 < len(s) < 180:
            err_s = inject_turkish_errors(s)
            if err_s != s:
                pair_text = f"Hatalı: {err_s} | Düzeltme: {s}\n"
                enc = tok.encode(pair_text).ids
                all_tokens.extend(enc)
                all_tokens.append(3)

print(f"\nTotal Turkish Tokens ready: {len(all_tokens):,}")

# Split 95% train, 5% val
split_pt = int(len(all_tokens) * 0.95)
train_tokens = np.array(all_tokens[:split_pt], dtype=np.uint16)
val_tokens = np.array(all_tokens[split_pt:], dtype=np.uint16)

train_tokens.tofile(OUT_BIN)
val_tokens.tofile(VAL_BIN)

print(f"[SUCCESS] Saved Train Tokens: {OUT_BIN} ({len(train_tokens):,} tokens, {os.path.getsize(OUT_BIN)/(1024*1024):.1f} MB)")
print(f"[SUCCESS] Saved Val Tokens:   {VAL_BIN} ({len(val_tokens):,} tokens, {os.path.getsize(VAL_BIN)/(1024*1024):.1f} MB)")
