import os
import sys
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.decoders import ByteLevel

CORPUS_DIR = "data/nano_tr_data"
os.makedirs(CORPUS_DIR, exist_ok=True)
RAW_FILE = os.path.join(CORPUS_DIR, "turkish_corpus.txt")
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_8k.json"
os.makedirs("tokenizers", exist_ok=True)

print("=" * 65)
print("1. STREAMING CLEAN TURKISH CORPUS (Wikipedia TR + CulturaX TR)")
print("=" * 65, flush=True)

doc_count = 0
with open(RAW_FILE, "w", encoding="utf-8") as f:
    # 1. Wikipedia TR
    print("Streaming Wikipedia TR...", flush=True)
    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    for item in ds_wiki:
        t = item.get("text", "").strip()
        if len(t) > 150:
            f.write(t + "\n\n")
            doc_count += 1
            if doc_count >= 12000:
                break
    print(f"Collected {doc_count} docs from Wikipedia TR.", flush=True)

    # 2. CulturaX TR
    print("Streaming CulturaX TR...", flush=True)
    ds_cultura = load_dataset("uonlp/CulturaX", "tr", split="train", streaming=True)
    c_count = 0
    for item in ds_cultura:
        t = item.get("text", "").strip()
        if len(t) > 150:
            f.write(t + "\n\n")
            doc_count += 1
            c_count += 1
            if c_count >= 8000:
                break
    print(f"Collected {c_count} docs from CulturaX TR.", flush=True)

sz_mb = os.path.getsize(RAW_FILE) / (1024 * 1024)
print(f"\nCorpus ready: {RAW_FILE} ({sz_mb:.2f} MB, {doc_count} documents).", flush=True)

print("\n" + "=" * 65)
print("2. TRAINING TURKISH 8,192 VOCAB BYTE-LEVEL BPE TOKENIZER")
print("=" * 65, flush=True)

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

tokenizer.train([RAW_FILE], trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = ByteLevel()
tokenizer.save(TOKENIZER_FILE)
print(f"[SUCCESS] Turkish 8k Tokenizer saved to: {TOKENIZER_FILE}", flush=True)
