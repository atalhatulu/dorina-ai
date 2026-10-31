"""
================================================================================
DORINA-MoE: AŞAMA 2 SHARD 7 (TURBO DERLEYİCİ - 100M TOKEN)
================================================================================
Hedeflenen İlaç Reçetesi (Loss-Weighted Targeted Mining):
  - %35 Bilgisayar Sistemleri, Bellek Mimarisi & İleri Bilim (FineWeb-Edu sample-10BT)
  - %35 İleri Algoritmalar, Veri Yapıları & Hata Yakalama (flytech/python-codes + CodeAlpaca)
  - %20 Matematik, Teoremler & Sayısal Mantık (FineWeb-Edu Math)
  - %10 Koruyucu Sentaks ve Talimat Formatı
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
print("⚡ DORINA-MoE: SHARD 7 (100M TOKEN TURBO VERİ DERLEYİCİ BAŞLADI)")
print("=" * 80)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "stage2_shard7_100m_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "stage2_shard7_100m_val.bin")

TARGET_TRAIN = 100_000_000
TARGET_VAL   = 3_000_000

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
    
    if len(all_tokens) % 2_000_000 < 50_000 or total_docs % 5000 == 0:
        elapsed = max(time.time() - t0, 1)
        speed = len(all_tokens) / elapsed
        print(f"   ↳ {len(all_tokens):,} / {TARGET_TRAIN + TARGET_VAL:,} Token derlendi ({speed:.0f} tok/s) - %{len(all_tokens)/(TARGET_TRAIN+TARGET_VAL)*100:.1f}")

# 1. BİLGİSAYAR SİSTEMLERİ & DONANIM (%35 -> 36M Token)
print("\n🌐 [1/3] Bilgisayar Sistemleri, Bellek Mimarisi & Fen...")
target_part1 = 36_000_000
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf = []
    doc_count = 0
    for row in ds_edu:
        doc_count += 1
        if doc_count < 250_000:
            continue  # Shard 6'da kullanılan metinleri atlayıp yeni içerik alıyoruz
        txt = row.get("text", "")
        if len(txt) > 200:
            buf.append(txt)
        if len(buf) >= 1000:
            add_batch(buf)
            buf = []
        if len(all_tokens) >= target_part1:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ FineWeb-Edu uyarısı: {e}")

print(f"   ✓ Sistem & Fen tamamlandı: {len(all_tokens):,} Token")

# 2. KOD & ALGORİTMA (%35 -> Toplam 72M Token)
print("\n🌐 [2/3] İleri Algoritmalar, Veri Yapıları & Fonksiyonlar...")
target_part2 = 72_000_000
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
        if len(all_tokens) >= target_part2:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ Kod yükleme uyarısı: {e}")

if len(all_tokens) < target_part2:
    try:
        ds_alp = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        buf = []
        for row in ds_alp:
            out = row.get("output", "")
            if out and len(out) > 50:
                buf.append(out)
            if len(buf) >= 1000:
                add_batch(buf)
                buf = []
            if len(all_tokens) >= target_part2:
                break
        if buf:
            add_batch(buf)
    except Exception as e:
        print(f"⚠️ CodeAlpaca uyarısı: {e}")

print(f"   ✓ Kod & Algoritma tamamlandı: {len(all_tokens):,} Token")

# 3. MATEMATİK & SAYISAL AKIL (%30 -> 103M Token)
print("\n🌐 [3/3] Matematik, Sayısal Mantık & Genel Zeka...")
target_part3 = TARGET_TRAIN + TARGET_VAL
try:
    ds_math = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf = []
    doc_count = 0
    for row in ds_math:
        doc_count += 1
        if doc_count < 600_000:
            continue
        txt = row.get("text", "")
        if any(w in txt.lower() for w in ["math", "theorem", "calculate", "algorithm", "equation", "formula", "matrix", "function", "probability"]):
            if len(txt) > 100:
                buf.append(txt)
        if len(buf) >= 1000:
            add_batch(buf)
            buf = []
        if len(all_tokens) >= target_part3:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ Math yükleme uyarısı: {e}")

# Eksik kalırsa yerel TinyStories veya metinlerle tamamla
if len(all_tokens) < target_part3:
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
                if len(all_tokens) >= target_part3:
                    break
            if chunk:
                add_batch(chunk)

print(f"\n✨ TOPLAM DERLENEN TOKEN: {len(all_tokens):,}")

# Array'e dönüştür ve kaydet
all_tokens_np = np.array(all_tokens[:target_part3], dtype=np.uint16)
val_tokens = all_tokens_np[:TARGET_VAL]
train_tokens = all_tokens_np[TARGET_VAL:TARGET_VAL + TARGET_TRAIN]

print(f"💾 Train dosyası yazılıyor: {train_bin_path} ({len(train_tokens):,} Token)...")
train_tokens.tofile(train_bin_path)

print(f"💾 Val dosyası yazılıyor: {val_bin_path} ({len(val_tokens):,} Token)...")
val_tokens.tofile(val_bin_path)

elapsed_total = time.time() - t0
print(f"\n🎉 SHARD 7 HAZIR! Süre: {elapsed_total/60:.1f} dakika.")
