"""
================================================================================
DORINA V3: 1 MİLYAR TOKENLIK GECE MARATONU PAKETLEYİCİ (TITAN 1B RUN)
================================================================================
Müfredat Dağılımı (Bilişsel & Kod Sıçraması - 1.000.000.000 Token = ~1.9 GB):
  - %45 FineWeb-Edu (450M Token - Saf İngilizce İleri Bilim, Mantık, Dünya Bilgisi)
  - %35 CodeParrot (350M Token - Saf Algoritmalar, OOP, Standart Kütüphaneler)
  - %10 Elit Master Gold (100M Token - 4.192 Adet Damıtılmış Soru-Cevap)
  - %10 Türkçe Keep-Alive (100M Token - Wikipedia Keep-Alive Gramer)
================================================================================
"""

import os
import sys
import json
import random
import time
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TOK_PATH = os.path.join(ROOT_DIR, "tokenizers", "dorina_tokenizer_v3_32k.json")
OUTPUT_BIN = os.path.join(DATA_DIR, "v3_titan_1b_train.bin")
OUTPUT_VAL_BIN = os.path.join(DATA_DIR, "v3_titan_1b_val.bin")

TOTAL_TARGET_TOKENS = 1_000_000_000  # Tam 1.000.000.000 (1 Milyar Token)
VAL_TARGET_TOKENS   =     5_000_000  # 5 Milyon Doğrulama Tokenı

TARGETS = {
    "edu": int(TOTAL_TARGET_TOKENS * 0.45),       # 450M
    "code": int(TOTAL_TARGET_TOKENS * 0.35),      # 350M
    "teachers": int(TOTAL_TARGET_TOKENS * 0.10),   # 100M
    "turkish": int(TOTAL_TARGET_TOKENS * 0.10)    # 100M
}

print("=" * 80)
print("🚀 DORINA V3: TAM 1 MİLYAR (1B) TOKENLIK TITAN GECE PAKETLEME MOTORU")
print("=" * 80)
print(f"Hedef Token Sayısı: {TOTAL_TARGET_TOKENS:,} (~1.9 GB)")
print(f"Tokenizer: {TOK_PATH}")
print(f"Çıktı Dosyası: {OUTPUT_BIN}")
print("-" * 80)

tokenizer = Tokenizer.from_file(TOK_PATH)
eos_id = tokenizer.token_to_id("<|endoftext|>")

def collect_tokens_to_mmap(name, target_tokens, token_generator, mmap_array, offset):
    t0 = time.time()
    count = 0
    written = 0
    print(f"\n[Veri Toplama] '{name}' kategorisi için {target_tokens:,} token toplanıyor...", flush=True)
    for t_list in token_generator:
        n = len(t_list) + 1  # tokens + eos
        if written + n > target_tokens:
            needed = target_tokens - written
            if needed > 0:
                chunk = (t_list + [eos_id])[:needed]
                mmap_array[offset + written : offset + written + needed] = chunk
                written += needed
            break
        
        mmap_array[offset + written : offset + written + len(t_list)] = t_list
        mmap_array[offset + written + len(t_list)] = eos_id
        written += n
        count += 1
        
        if count % 3000 == 0:
            dt = time.time() - t0
            speed = written / max(dt, 1e-4)
            pct = (written / target_tokens) * 100
            print(f"  * {name}: {written:,}/{target_tokens:,} (%{pct:.1f}) - {speed:,.0f} tok/s", end="\r", flush=True)
            
    print(f"\n  ✅ {name} tamamlandı: {written:,} token ({time.time() - t0:.1f} sn)", flush=True)
    return written

# 1. Teachers (Master Gold SFT + DeepSeek sanitized)
def get_teacher_tokens():
    master_gold_path = os.path.join(DATA_DIR, "distilled_teachers", "dorina_gold_sft_master.json")
    with open(master_gold_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    print(f"  Kullanılan Altın Öğretmen Örneği: {len(samples):,}", flush=True)
    idx = 0
    while True:
        sample = samples[idx % len(samples)]
        text_parts = []
        for msg in sample.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        full_text = "\n".join(text_parts)
        yield tokenizer.encode(full_text).ids
        idx += 1

# 2. Turkish (Wikipedia keep-alive)
def get_turkish_tokens():
    while True:
        ds = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
        skipped = 0
        for item in ds:
            skipped += 1
            if skipped < 40000:
                continue
            text = item.get("text", "")
            if len(text.strip()) > 100:
                yield tokenizer.encode(text).ids

# 3. Code (Codeparrot clean valid)
def get_code_tokens():
    while True:
        ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
        skipped = 0
        for item in ds:
            skipped += 1
            if skipped < 45000:
                continue
            code = item.get("content", "")
            if len(code.strip()) > 50:
                yield tokenizer.encode(code).ids

# 4. Edu (Fineweb-Edu sample-10BT)
def get_edu_tokens():
    while True:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        skipped = 0
        for item in ds:
            skipped += 1
            if skipped < 80000:
                continue
            text = item.get("text", "")
            if len(text.strip()) > 100:
                yield tokenizer.encode(text).ids

def main():
    print("\n--- Titan 1.000.000.000 (1B) Token Paketi Başlıyor ---", flush=True)
    RAW_TEMP_BIN = os.path.join(DATA_DIR, "v3_titan_1b_raw.bin")
    
    # 1B tokenlık geçici raw dosya aç (2 GB disk)
    total_tokens_with_val = TOTAL_TARGET_TOKENS + VAL_TARGET_TOKENS
    print(f"[Disk Bellek Eşleme] {RAW_TEMP_BIN} oluşturuluyor ({total_tokens_with_val:,} token)...", flush=True)
    raw_mmap = np.memmap(RAW_TEMP_BIN, dtype=np.uint16, mode='w+', shape=(total_tokens_with_val,))
    
    offset = 0
    for name, gen, tgt in [
        ("teachers", get_teacher_tokens(), TARGETS["teachers"]),
        ("turkish", get_turkish_tokens(), TARGETS["turkish"]),
        ("code", get_code_tokens(), TARGETS["code"]),
        ("edu", get_edu_tokens(), TARGETS["edu"] + VAL_TARGET_TOKENS)
    ]:
        written = collect_tokens_to_mmap(name, tgt, gen, raw_mmap, offset)
        offset += written
        raw_mmap.flush()

    print("\n--- Disk Üzerinde Blok Karıştırma (Block Shuffling) ---", flush=True)
    CHUNK_SIZE = 1024
    num_chunks = total_tokens_with_val // CHUNK_SIZE
    print(f"Toplam {num_chunks:,} blok (her biri 1024 token) rastgele karıştırılıyor...", flush=True)
    
    indices = list(range(num_chunks))
    random.seed(42 + 100) # Titan 1B seed
    random.shuffle(indices)

    val_chunks = VAL_TARGET_TOKENS // CHUNK_SIZE
    train_chunks = num_chunks - val_chunks

    print(f"[Yazılıyor] Train: {train_chunks * CHUNK_SIZE:,} token -> {OUTPUT_BIN}", flush=True)
    train_mmap = np.memmap(OUTPUT_BIN, dtype=np.uint16, mode='w+', shape=(train_chunks * CHUNK_SIZE,))
    for i, idx in enumerate(indices[:train_chunks]):
        train_mmap[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] = raw_mmap[idx * CHUNK_SIZE : (idx + 1) * CHUNK_SIZE]
        if i % 100000 == 0 and i > 0:
            print(f"  * Train Yazılıyor: {i:,}/{train_chunks:,} blok...", flush=True)
    train_mmap.flush()
    del train_mmap

    print(f"[Yazılıyor] Val: {val_chunks * CHUNK_SIZE:,} token -> {OUTPUT_VAL_BIN}", flush=True)
    val_mmap = np.memmap(OUTPUT_VAL_BIN, dtype=np.uint16, mode='w+', shape=(val_chunks * CHUNK_SIZE,))
    for i, idx in enumerate(indices[train_chunks:]):
        val_mmap[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] = raw_mmap[idx * CHUNK_SIZE : (idx + 1) * CHUNK_SIZE]
    val_mmap.flush()
    del val_mmap
    del raw_mmap

    if os.path.exists(RAW_TEMP_BIN):
        os.remove(RAW_TEMP_BIN)

    print(f"\n✅ Train Titan 1B kaydedildi: {OUTPUT_BIN} ({os.path.getsize(OUTPUT_BIN)/(1024*1024):.1f} MB)", flush=True)
    print(f"✅ Val Titan 1B kaydedildi:   {OUTPUT_VAL_BIN} ({os.path.getsize(OUTPUT_VAL_BIN)/(1024*1024):.1f} MB)", flush=True)

if __name__ == "__main__":
    main()
