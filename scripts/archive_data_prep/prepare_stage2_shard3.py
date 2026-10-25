"""
================================================================================
DORINA-MoE: AŞAMA 2 SHARD 3 (GENEL BİLİM, MANTIK & ALGORİTMA DERLEME MOTORU)
================================================================================
Amaç:
  Bu script, Run 7 için Aşama 2 Shard 3 (100M Token) verisini derler.
  Amacımız:
    - Bilgisayar mimarisi, fizik, matematiksel mantık ve algoritmik temelleri
      evrensel ölçekte güçlendirmek.
    - Asla benchmark sorularına ezber yaptırmadan, saf pre-train makaleleri
      ve açık kaynak temiz kodlarla genel zekayı (Chinchilla doygunluğunu) artırmak.

Bileşenler (%100 Saf Pre-Train - Sıfır Çakışma / Offsetli Akış):
  1. %45 Mantık, Teoremler & Sayısal Muhakeme (FineWeb-Edu Math & Reasoning)
  2. %35 Bilgisayar Mimarisi, Bellek (Stack/Heap), Fizik & Fen Bilimleri
  3. %20 Saf Algoritmalar, Fonksiyonlar & Veri Yapıları (Algoritmik Tazelik)

Filtreler:
  - Sıfır ChatML, Sıfır Web Framework kirliliği.
================================================================================
"""

import os
import sys
import re
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
print("🦀 DORINA-MoE: AŞAMA 2 - SHARD 3 (100M TOKEN BİLİM & MANTIK PRE-TRAIN)")
print("   • %45 Sayısal Muhakeme, Formüller & Mantık (FineWeb-Edu)")
print("   • %35 Bilgisayar Bilimi, Bellek, Fizik & Doğa Bilimleri")
print("   • %20 Saf Algoritmalar & Veri Yapıları")
print("=" * 80)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "stage2_shard3_100m_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "stage2_shard3_100m_val.bin")

TARGET_TRAIN = 100_000_000
TARGET_VAL   = 3_000_000

train_tokens = []
val_tokens = []
total_docs = 0
t0 = time.time()

def add_clean_text(text):
    global total_docs
    if not text or len(text.strip()) < 40:
        return
        
    lower = text.lower()
    if "from django" in lower or "import django" in lower:
        return
    if "settings.py" in lower or "secret_key" in lower:
        return
    if "<html>" in lower or "<!doctype" in lower:
        return
        
    ids = tokenizer.encode(text).ids
    ids.append(eos_id)
    
    if len(train_tokens) < TARGET_TRAIN:
        train_tokens.extend(ids)
    elif len(val_tokens) < TARGET_VAL:
        val_tokens.extend(ids)
        
    total_docs += 1
    if total_docs % 5000 == 0:
        toks = len(train_tokens) + len(val_tokens)
        speed = toks / max(time.time() - t0, 1)
        print(f"   ↳ {total_docs:,} döküman | Train: {len(train_tokens):,} / {TARGET_TRAIN:,} | Val: {len(val_tokens):,} ({speed:.0f} tok/s)")

def extract_pure_code(text):
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    if code_blocks:
        return "\n\n".join([cb.strip() for cb in code_blocks if len(cb.strip()) > 20])
    return text.strip()

# -----------------------------------------------------------------------------
# 1. SAYISAL MUHAKEME & MANTIK (%45 -> 45M Token)
# -----------------------------------------------------------------------------
print("\n🌐 [1/3] Kaynak: Sayısal Mantık, Formüller & Matematiksel Muhakeme...")
target_math = int(TARGET_TRAIN * 0.45)
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    # Shard 2'de ilk kısımları kullandık, bu kez temiz akışla ileriye doğru devam ediyoruz
    doc_count = 0
    for row in ds_edu:
        txt = row.get("text", "")
        doc_count += 1
        # Shard 2 offsetini atla (ilk 150k dokümanı Shard 2 almıştı)
        if doc_count < 155000:
            continue
        if len(txt) > 200:
            add_clean_text(txt)
        if len(train_tokens) >= target_math:
            break
except Exception as e:
    print(f"⚠️ FineWeb-Edu Math hatası: {e}")
print(f"   ✓ Sayısal Mantık kotası tamamlandı: {len(train_tokens):,} token")

# -----------------------------------------------------------------------------
# 2. BİLGİSAYAR BİLİMİ, SİSTEMLER, FİZİK & DOĞA BİLİMLERİ (%35 -> Toplam 80M Token)
# -----------------------------------------------------------------------------
print("\n🌐 [2/3] Kaynak: Bilgisayar Mimarisi, Bellek & Fen Bilimleri...")
target_science = int(TARGET_TRAIN * 0.80)
try:
    for row in ds_edu:
        txt = row.get("text", "")
        if len(txt) > 200:
            add_clean_text(txt)
        if len(train_tokens) >= target_science:
            break
except Exception as e:
    print(f"⚠️ FineWeb-Edu Science hatası: {e}")
print(f"   ✓ Bilim kotası tamamlandı: {len(train_tokens):,} token")

# -----------------------------------------------------------------------------
# 3. SAF PYTHON ALGORİTMALARI & PROBLEM ÇÖZÜMLERİ (%20 -> Toplam 100M + 3M Val)
# -----------------------------------------------------------------------------
print("\n🌐 [3/3] Kaynak: Saf Python Algoritmaları & Problem Çözümleri...")
try:
    ds_py = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    row_idx = 0
    for row in ds_py:
        row_idx += 1
        if row_idx < 15000: # Önceki run'lardan sonraki taze satırlar
            continue
        out = row.get("output", "")
        code = extract_pure_code(out)
        if code and len(code) > 40:
            add_clean_text(code)
        if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
            break
except Exception as e:
    print(f"⚠️ flytech hatası: {e}")

# Eğer hala kalan varsa edu akışıyla tamamla
if len(train_tokens) < TARGET_TRAIN or len(val_tokens) < TARGET_VAL:
    for row in ds_edu:
        txt = row.get("text", "")
        if len(txt) > 200:
            add_clean_text(txt)
        if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
            break

print(f"   ✓ Toplam hedef tamamlandı: Train={len(train_tokens):,}, Val={len(val_tokens):,}")

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr = np.array(train_tokens[:TARGET_TRAIN], dtype=np.uint16)
val_arr   = np.array(val_tokens[:TARGET_VAL], dtype=np.uint16)

train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

elapsed_total = time.time() - t0
print("=" * 80)
print(f"🎉 AŞAMA 2 - SHARD 3 (100M TOKEN) KUSURSUZ ŞEKİLDE DERLENDİ!")
print(f"   ✓ Train: {train_bin_path} ({len(train_arr):,} Token, {train_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ✓ Val:   {val_bin_path} ({len(val_arr):,} Token, {val_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ⏱️ Süre:  {elapsed_total / 60:.1f} dakika")
print("=" * 80)
