import os
import sys
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

OUT_DIR = "data/nano_v2_bench"
os.makedirs(OUT_DIR, exist_ok=True)
RAW_FILE = os.path.join(OUT_DIR, "sample_raw.txt")
TOKENIZER_8K = "tokenizers/dorina_nano_8k.json"
os.makedirs("tokenizers", exist_ok=True)

print("=" * 65)
print("1. BUILDING REPRESENTATIVE CORPUS FOR 8k TOKENIZER & BENCHMARK")
print("Target: ~40% FineWeb-Edu, ~30% Cosmopedia, ~30% Simple Wikipedia")
print("=" * 65)

doc_count = 0
with open(RAW_FILE, "w", encoding="utf-8") as f:
    # 1. FineWeb-Edu (score >= 3)
    print("Streaming from FineWeb-Edu...", flush=True)
    ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for item in ds_fw:
        score = item.get("score", 0)
        text = item.get("text", "").strip()
        if score >= 3.0 and len(text) > 150:
            f.write(text + "\n\n")
            doc_count += 1
            if doc_count >= 4000:
                break
    print(f"Collected {doc_count} docs from FineWeb-Edu.", flush=True)

    # 2. Cosmopedia 100k
    print("Streaming from Cosmopedia...", flush=True)
    ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train", streaming=True)
    cosmo_cnt = 0
    for item in ds_cosmo:
        text = item.get("text", "").strip()
        if len(text) > 150:
            f.write(text + "\n\n")
            doc_count += 1
            cosmo_cnt += 1
            if cosmo_cnt >= 3000:
                break
    print(f"Collected {cosmo_cnt} docs from Cosmopedia.", flush=True)

    # 3. Simple Wikipedia
    print("Streaming from Simple Wikipedia...", flush=True)
    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    wiki_cnt = 0
    for item in ds_wiki:
        text = item.get("text", "").strip()
        if len(text) > 150:
            f.write(text + "\n\n")
            doc_count += 1
            wiki_cnt += 1
            if wiki_cnt >= 3000:
                break
    print(f"Collected {wiki_cnt} docs from Simple Wikipedia.", flush=True)

sz_mb = os.path.getsize(RAW_FILE) / (1024 * 1024)
print(f"\nCorpus ready: {RAW_FILE} ({sz_mb:.2f} MB, {doc_count} documents).", flush=True)

print("\n" + "=" * 65)
print("2. TRAINING 8,192 VOCAB BYTE-LEVEL BPE TOKENIZER")
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
tokenizer.save(TOKENIZER_8K)
print(f"[SUCCESS] 8k Vocab Tokenizer trained and saved to: {TOKENIZER_8K}", flush=True)
