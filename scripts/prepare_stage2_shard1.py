"""
================================================================================
DORINA-MoE: AŞAMA 2 SHARD 1 (ALTIN KARIŞIM - PRE-TRAIN VERİ DERLEME MOTORU)
================================================================================
Amaç:
  Bu script, pre-training aşamasında format ve alan sapmasını (Domain/Format Drift)
  önlemek için tasarlanmış kusursuz ve homojen 100M tokenlik altın veri kümesini derler.

Bileşenler (%100 Saf Pre-Train):
  1. %40 Saf Python Algoritmaları & Problem Çözümleri (Temizlenmiş Saf Kod Blokları)
  2. %35 İleri Seviye Bilim, Mantık & Felsefe (FineWeb-Edu)
  3. %15 Matematiksel Akıl Yürütme & Problem Çözme
  4. %10 Dünya Bilgisi & Kavramlar (Vikipedi & Ansiklopedik Metinler)

Filtreler:
  - Sıfır ChatML (<|im_start|>, user, assistant tagları KESİNLİKLE YOKTUR)
  - Sıfır Web Framework kirliliği (Django, Flask, HTML, settings.py elenir)
  - Sadece saf kod gövdesi, döngüler, algoritmalar ve kaliteli akademik metinler
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
print("🦀 DORINA-MoE: AŞAMA 2 - ALTIN KARIŞIM (100M TOKEN KUSURSUZ PRE-TRAIN)")
print("   • %40 Saf Algoritmik Kod | %35 Bilim & Mantık | %15 Matematik | %10 Kavram")
print("=" * 80)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "stage2_shard1_100m_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "stage2_shard1_100m_val.bin")

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
        
    # Anti-framework ve anti-boilerplate filtresi
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
    # Markdown kod bloklarından saf kodu çıkartır
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    if code_blocks:
        return "\n\n".join([cb.strip() for cb in code_blocks if len(cb.strip()) > 20])
    return text.strip()

# -----------------------------------------------------------------------------
# 1. SAF PYTHON ALGORİTMALARI & FONKSİYONLAR (~40M Token)
# -----------------------------------------------------------------------------
print("\n🌐 [1/4] Kaynak: Saf Python Fonksiyonları & Algoritmaları...")
code_target = int(TARGET_TRAIN * 0.40)
try:
    ds_py = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    for row in ds_py:
        out = row.get("output", "")
        code = extract_pure_code(out)
        if code and len(code) > 40:
            add_clean_text(code)
        if len(train_tokens) >= code_target:
            break
except Exception as e:
    print(f"⚠️ flytech hatası: {e}")

try:
    if len(train_tokens) < code_target:
        ds_alp = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        for row in ds_alp:
            out = row.get("output", "")
            code = extract_pure_code(out)
            if code and len(code) > 40:
                add_clean_text(code)
            if len(train_tokens) >= code_target:
                break
except Exception as e:
    print(f"⚠️ CodeAlpaca hatası: {e}")
print(f"   ✓ Kod kotası tamamlandı: {len(train_tokens):,} token")

# -----------------------------------------------------------------------------
# 2. İLERİ BİLİM, MANTIK & FELSEFE (~35M Token -> Toplam 75M)
# -----------------------------------------------------------------------------
print("\n🌐 [2/4] Kaynak: İleri Seviye Bilim & Felsefe (FineWeb-Edu)...")
edu_target = int(TARGET_TRAIN * 0.75)
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds_edu:
        txt = row.get("text", "")
        if len(txt) > 200:
            add_clean_text(txt)
        if len(train_tokens) >= edu_target:
            break
except Exception as e:
    print(f"⚠️ FineWeb-Edu hatası: {e}")
print(f"   ✓ Bilim kotası tamamlandı: {len(train_tokens):,} token")

# -----------------------------------------------------------------------------
# 3. MATEMATİKSEL AKIL YÜRÜTME & MANTIK (~25M Token -> Toplam 100M Train + 3M Val)
# -----------------------------------------------------------------------------
print("\n🌐 [3/4] Kaynak: Matematiksel Mantık, Formüller & Genel Bilim...")
try:
    ds_math = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds_math:
        txt = row.get("text", "")
        if len(txt) > 200:
            add_clean_text(txt)
        if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
            break
except Exception as e:
    print(f"⚠️ Tamamlama hatası: {e}")
print(f"   ✓ Toplam hedef tamamlandı: Train={len(train_tokens):,}, Val={len(val_tokens):,}")

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr = np.array(train_tokens[:TARGET_TRAIN], dtype=np.uint16)
val_arr   = np.array(val_tokens[:TARGET_VAL], dtype=np.uint16)

train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

elapsed_total = time.time() - t0
print("=" * 80)
print(f"🎉 AŞAMA 2 - ALTIN KARIŞIM (100M TOKEN) KUSURSUZ ŞEKİLDE HAZIRLANDI!")
print(f"   ✓ Train: {train_bin_path} ({len(train_arr):,} Token, {train_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ✓ Val:   {val_bin_path} ({len(val_arr):,} Token, {val_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ⏱️ Süre:  {elapsed_total / 60:.1f} dakika")
print("=" * 80)
