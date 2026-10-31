"""
================================================================================
DORINA-MoE: AŞAMA 2 SHARD 8 (150M BÜYÜK FİNAL DERLEYİCİSİ - 1 MİLYAR TOKEN ZİRVESİ)
================================================================================
Hedeflenen Şok Tedavi İlaç Reçetesi:
  - %45 Bilgisayar Sistemleri, Bellek Mimarisi & Fen Bilimleri (FineWeb-Edu sample-10BT)
  - %40 Matematik, Teoremler & Sayısal Muhakeme (FineWeb-Edu Math)
  - %15 Koruyucu Python & Kodlama
================================================================================
"""

import os
import sys
import time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 80)
print("⚡ DORINA-MoE: SHARD 8 (150M TOKEN BÜYÜK FİNAL DERLEYİCİSİ BAŞLADI)")
print("   Hedef: 1 MİLYAR KÜMÜLATİF TOKEN TAMAMLAMA!")
print("=" * 80)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "stage2_shard8_150m_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "stage2_shard8_150m_val.bin")

TARGET_TRAIN = 150_000_000
TARGET_VAL   = 4_000_000
TOTAL_TARGET = TARGET_TRAIN + TARGET_VAL

all_tokens = []
total_docs = 0
t0 = time.time()

def add_batch(texts):
    global total_docs
    encodings = tokenizer.encode_batch(texts)
    for enc in encodings:
        all_tokens.extend(enc.ids)
        all_tokens.append(eos_id)
        total_docs += 1
    
    if len(all_tokens) % 3_000_000 < 60_000 or total_docs % 8000 == 0:
        elapsed = max(time.time() - t0, 1)
        speed = len(all_tokens) / elapsed
        print(f"   ↳ {len(all_tokens):,} / {TOTAL_TARGET:,} Token derlendi ({speed:.0f} tok/s) - %{len(all_tokens)/TOTAL_TARGET*100:.1f}")

# 1. BİLGİSAYAR SİSTEMLERİ & DONANIM (%45 -> ~69M Token)
print("\n🌐 [1/3] Bilgisayar Sistemleri, Bellek Mimarisi & Fen Bilimleri (%45)...")
target_part1 = 69_000_000
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf = []
    doc_count = 0
    for row in ds_edu:
        doc_count += 1
        if doc_count < 450_000:
            continue  # Shard 6 ve 7'de kullanılanları atlayıp yepyeni makaleler alıyoruz
        txt = row.get("text", "")
        if any(w in txt.lower() for w in ["memory", "cache", "operating system", "hardware", "cpu", "compiler", "dna", "gravity", "physics", "disk", "ssd", "ram", "stack", "heap"]):
            if len(txt) > 150:
                buf.append(txt)
        if len(buf) >= 1000:
            add_batch(buf)
            buf = []
        if len(all_tokens) >= target_part1:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ Bilim & Sistem yükleme uyarısı: {e}")

print(f"   ✓ Sistem & Fen tamamlandı: {len(all_tokens):,} Token")

# 2. MATEMATİK, TEOREMLER & SAYISAL AKIL (%40 -> Toplam 130M Token)
print("\n🌐 [2/3] İleri Matematik, Teoremler & Sayısal Muhakeme (%40)...")
target_part2 = 130_000_000
try:
    ds_math = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf = []
    doc_count = 0
    for row in ds_math:
        doc_count += 1
        if doc_count < 850_000:
            continue
        txt = row.get("text", "")
        if any(w in txt.lower() for w in ["math", "theorem", "calculate", "algorithm", "equation", "formula", "matrix", "function", "probability", "algebra", "arithmetic", "gcd", "prime"]):
            if len(txt) > 100:
                buf.append(txt)
        if len(buf) >= 1000:
            add_batch(buf)
            buf = []
        if len(all_tokens) >= target_part2:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ Matematik yükleme uyarısı: {e}")

print(f"   ✓ Matematik tamamlandı: {len(all_tokens):,} Token")

# 3. KORUYUCU KOD VE FORMAT (%15 -> Toplam 154M Token)
print("\n🌐 [3/3] Koruyucu Kod, Fonksiyonlar ve Talimat Disiplini (%15)...")
try:
    ds_py = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    buf = []
    for row in ds_py:
        out = row.get("output", "")
        if out and len(out) > 50:
            buf.append(out)
        if len(buf) >= 1000:
            add_batch(buf)
            buf = []
        if len(all_tokens) >= TOTAL_TARGET:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ Kod yükleme uyarısı: {e}")

# Eksik kalırsa yerel tampondan tamamla
if len(all_tokens) < TOTAL_TARGET:
    print("   ↳ Tamamlama tamponu (Yerel veri)...")
    local_path = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            chunk = []
            for line in f:
                if line.strip():
                    chunk.append(line)
                if len(chunk) >= 5000:
                    add_batch(chunk)
                    chunk = []
                if len(all_tokens) >= TOTAL_TARGET:
                    break
            if chunk:
                add_batch(chunk)

print(f"\n✨ TOPLAM DERLENEN TOKEN: {len(all_tokens):,}")

# Array'e dönüştür ve kaydet
all_tokens_np = np.array(all_tokens[:TOTAL_TARGET], dtype=np.uint16)
val_tokens = all_tokens_np[:TARGET_VAL]
train_tokens = all_tokens_np[TARGET_VAL:TARGET_VAL + TARGET_TRAIN]

print(f"💾 Train dosyası yazılıyor: {train_bin_path} ({len(train_tokens):,} Token)...")
train_tokens.tofile(train_bin_path)

print(f"💾 Val dosyası yazılıyor: {val_bin_path} ({len(val_tokens):,} Token)...")
val_tokens.tofile(val_bin_path)

elapsed_total = time.time() - t0
print(f"\n🎉 SHARD 8 (150M BÜYÜK FİNAL) HAZIR! Süre: {elapsed_total/60:.1f} dakika.")
