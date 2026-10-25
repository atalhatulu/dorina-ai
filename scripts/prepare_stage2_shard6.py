"""
================================================================================
DORINA-MoE: AŞAMA 2 SHARD 6 (TURBO DERLEYİCİ - 100M TOKEN)
================================================================================
Reçete:
  - %35 Algoritmalar, Veri Yapıları & Hata Yakalama (flytech/python-codes-25k + CodeAlpaca)
  - %35 Bilgisayar Mimarisi, Bellek & Fen (FineWeb-Edu sample-10BT)
  - %20 Sayısal Mantık & Aritmetik (FineWeb-Edu Math)
  - %10 Koruyucu Sentaks Dozu
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
print("⚡ DORINA-MoE: SHARD 6 (100M TOKEN TURBO VERİ DERLEYİCİ BAŞLADI)")
print("=" * 80)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "stage2_shard6_100m_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "stage2_shard6_100m_val.bin")

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

# 1. KOD & ALGORİTMA (%35 -> ~36M Token)
print("\n🌐 [1/3] Kod & Algoritmalar yükleniyor...")
target_part1 = 36_000_000
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
        if len(all_tokens) >= target_part1:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ Kod yükleme uyarısı: {e}")

if len(all_tokens) < target_part1:
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
            if len(all_tokens) >= target_part1:
                break
        if buf:
            add_batch(buf)
    except Exception as e:
        print(f"⚠️ CodeAlpaca uyarısı: {e}")

print(f"   ✓ Kod & Algoritma tamamlandı: {len(all_tokens):,} Token")

# 2. BİLGİSAYAR SİSTEMLERİ & DONANIM (%35 -> Toplam 72M Token)
print("\n🌐 [2/3] Bilgisayar Sistemleri, Bellek Mimarisi & Fen...")
target_part2 = 72_000_000
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf = []
    doc_count = 0
    for row in ds_edu:
        doc_count += 1
        if doc_count < 450000:
            continue
        txt = row.get("text", "")
        if len(txt) > 200:
            buf.append(txt)
        if len(buf) >= 500:
            add_batch(buf)
            buf = []
        if len(all_tokens) >= target_part2:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ Sistem verisi uyarısı: {e}")

print(f"   ✓ Sistem & Fen tamamlandı: {len(all_tokens):,} Token")

# 3. SAYISAL MANTIK & MATEMATİK (%30 -> 103M Token)
print("\n🌐 [3/3] Sayısal Mantık, Matematik & Teoremler...")
target_total = TARGET_TRAIN + TARGET_VAL
try:
    buf = []
    for row in ds_edu:
        txt = row.get("text", "")
        if len(txt) > 200:
            buf.append(txt)
        if len(buf) >= 500:
            add_batch(buf)
            buf = []
        if len(all_tokens) >= target_total:
            break
    if buf:
        add_batch(buf)
except Exception as e:
    print(f"⚠️ Matematik verisi uyarısı: {e}")

print(f"   ✓ Toplam derlenen token: {len(all_tokens):,}")

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr = np.array(all_tokens[:TARGET_TRAIN], dtype=np.uint16)
val_arr   = np.array(all_tokens[TARGET_TRAIN:TARGET_TRAIN + TARGET_VAL], dtype=np.uint16)

train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

print("=" * 80)
print(f"🎉 AŞAMA 2 - SHARD 6 (100M TOKEN) KUSURSUZ ŞEKİLDE DİSKE YAZILDI!")
print(f"   ✓ Train: {train_bin_path} ({len(train_arr):,} Token)")
print(f"   ✓ Val:   {val_bin_path} ({len(val_arr):,} Token)")
print(f"   ⏱️ Süre:  {(time.time() - t0) / 60:.1f} dakika")
print("=" * 80)
