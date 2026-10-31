"""
================================================================================
DORINA V3: 600 MİLYON TOKENLIK GECE MARATONU PAKETLEYİCİ (TITAN RUN)
================================================================================
Müfredat Dağılımı (Bilişsel & Kod Sıçraması - 600M Token):
  - %40 FineWeb-Edu (240M Token - İleri Bilim, Mantık, Dünya Bilgisi)
  - %35 CodeParrot (210M Token - Saf Algoritmalar, OOP, Standart Kütüphaneler)
  - %15 Türkçe Derin Metinler (90M Token - Wikipedia Keep-Alive Gramer)
  - %10 Elit Master Gold (60M Token - 4.192 Adet Damıtılmış Soru-Cevap)
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
OUTPUT_BIN = os.path.join(DATA_DIR, "v3_titan_600m_train.bin")
OUTPUT_VAL_BIN = os.path.join(DATA_DIR, "v3_titan_val.bin")

TOTAL_TARGET_TOKENS = 600_000_000  # 600 Milyon Token (Kümülatif 1.2 Milyar Token Hedefi)
VAL_TARGET_TOKENS   =   4_000_000  # 4 Milyon Doğrulama Tokenı

TARGETS = {
    "edu": int(TOTAL_TARGET_TOKENS * 0.40),       # 240M
    "code": int(TOTAL_TARGET_TOKENS * 0.35),      # 210M
    "turkish": int(TOTAL_TARGET_TOKENS * 0.15),   # 90M
    "teachers": int(TOTAL_TARGET_TOKENS * 0.10)   # 60M
}

print("=" * 80)
print("🚀 DORINA V3: 600M TOKENLIK TITAN GECE PAKETLEME MOTORU")
print("=" * 80)
print(f"Hedef Token Sayısı: {TOTAL_TARGET_TOKENS:,} (~1.2 GB)")
print(f"Tokenizer: {TOK_PATH}")
print(f"Çıktı Dosyası: {OUTPUT_BIN}")
print("-" * 80)

tokenizer = Tokenizer.from_file(TOK_PATH)
eos_id = tokenizer.token_to_id("<|endoftext|>")

def collect_tokens(name, target_tokens, token_generator):
    tokens = []
    t0 = time.time()
    count = 0
    print(f"\n[Veri Toplama] '{name}' kategorisi için {target_tokens:,} token toplanıyor...")
    for t_list in token_generator:
        tokens.extend(t_list)
        tokens.append(eos_id)
        count += 1
        if len(tokens) >= target_tokens:
            break
        if count % 3000 == 0:
            dt = time.time() - t0
            speed = len(tokens) / max(dt, 1e-4)
            pct = (len(tokens) / target_tokens) * 100
            print(f"  * {name}: {len(tokens):,}/{target_tokens:,} (%{pct:.1f}) - {speed:,.0f} tok/s", end="\r", flush=True)
    
    tokens = tokens[:target_tokens]
    print(f"\n  ✅ {name} tamamlandı: {len(tokens):,} token ({time.time() - t0:.1f} sn)", flush=True)
    return tokens

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

# 2. Turkish (Wikipedia skip 40000 for Titan run)
def get_turkish_tokens():
    ds = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    skipped = 0
    for item in ds:
        skipped += 1
        if skipped < 40000:
            continue
        text = item.get("text", "")
        if len(text.strip()) > 100:
            yield tokenizer.encode(text).ids

# 3. Code (Codeparrot clean valid, skip 45000)
def get_code_tokens():
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    skipped = 0
    for item in ds:
        skipped += 1
        if skipped < 45000:
            continue
        code = item.get("content", "")
        if len(code.strip()) > 50:
            yield tokenizer.encode(code).ids

# 4. Edu (Fineweb-Edu sample-10BT, skip 80000)
def get_edu_tokens():
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
    print("\n--- Titan 600M Veri Çekme Başlıyor ---", flush=True)
    teacher_tokens = collect_tokens("teachers", TARGETS["teachers"], get_teacher_tokens())
    turkish_tokens = collect_tokens("turkish", TARGETS["turkish"], get_turkish_tokens())
    code_tokens    = collect_tokens("code", TARGETS["code"], get_code_tokens())
    edu_tokens     = collect_tokens("edu", TARGETS["edu"], get_edu_tokens())

    print("\n--- Homojen Karıştırma (Interleaving) ---", flush=True)
    all_chunks = []
    CHUNK_SIZE = 1024

    for src in [edu_tokens, code_tokens, turkish_tokens, teacher_tokens]:
        for i in range(0, len(src), CHUNK_SIZE):
            chunk = src[i:i + CHUNK_SIZE]
            if len(chunk) == CHUNK_SIZE:
                all_chunks.append(chunk)

    random.seed(42 + 99) # Titan seed
    random.shuffle(all_chunks)

    final_train = []
    for c in all_chunks:
        final_train.extend(c)

    val_tokens = final_train[:VAL_TARGET_TOKENS]
    train_tokens = final_train[VAL_TARGET_TOKENS:TOTAL_TARGET_TOKENS + VAL_TARGET_TOKENS]

    print(f"\nToplam Train Token: {len(train_tokens):,}", flush=True)
    print(f"Toplam Val Token:   {len(val_tokens):,}", flush=True)

    np.array(train_tokens, dtype=np.uint16).tofile(OUTPUT_BIN)
    print(f"✅ Train Titan kaydedildi: {OUTPUT_BIN} ({os.path.getsize(OUTPUT_BIN)/(1024*1024):.1f} MB)", flush=True)

    np.array(val_tokens, dtype=np.uint16).tofile(OUTPUT_VAL_BIN)
    print(f"✅ Val Titan kaydedildi:   {OUTPUT_VAL_BIN} ({os.path.getsize(OUTPUT_VAL_BIN)/(1024*1024):.1f} MB)", flush=True)

if __name__ == "__main__":
    main()
