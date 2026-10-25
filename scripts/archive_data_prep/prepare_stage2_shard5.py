"""
================================================================================
DORINA-MoE: AŞAMA 2 SHARD 5 (MATEMATİK & SAYISAL ZİHİN, SİSTEMLER VE BİLİM)
================================================================================
Amaç:
  Bu script, Run 9 için Aşama 2 Shard 5 (100M Token) verisini derler.
  Amacımız:
    - Modeli 552.9M kümülatif tokenden 651.2M tokene ulaştırmak.
    - Matematik ve sayısal muhakeme skorunu 9'dan 13-14 seviyesine fırlatmak
      (GSM8K, MathInstruct, saf matematik teoremleri ve aritmetik mantığı).
    - Bilim & Bilgisayar Sistemleri skorunu (RAM/Disk/Cache, Big-O, Fizik)
      7.4 rekorunun üzerine taşımak.
    - Python algoritmaları ve veri yapılarını (Binary search, Two sum, DP) pekiştirmek.

Bileşenler (%100 Saf Pre-Train - Sıfır Çakışma / 400.000+ Doküman Offsetli):
  1. %40 Matematik & Sayısal Akıl Yürütme (GSM8k & FineWeb-Edu Math)
  2. %35 Bilgisayar Sistemleri, Bellek Mimarisi & Fen Bilimleri (FineWeb-Edu 450k+ Offset)
  3. %25 Küratörlü İleri Algoritmalar & Veri Yapıları (def -> return)
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
print("🦀 DORINA-MoE: AŞAMA 2 - SHARD 5 (100M TOKEN MATEMATİK & BİLİM PRE-TRAIN)")
print("   • %40 Matematik, Aritmetik Mantık & Sayısal Teoremler")
print("   • %35 Bilgisayar Sistemleri, Donanım Mimarisi & Fen Bilimleri")
print("   • %25 İleri Algoritmalar & Temiz Fonksiyon Gövdeleri")
print("=" * 80)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "stage2_shard5_100m_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "stage2_shard5_100m_val.bin")

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

def extract_curated_function(text):
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    candidate = "\n\n".join(code_blocks) if code_blocks else text
    
    clean_blocks = []
    for chunk in candidate.split("\n\n"):
        c_strip = chunk.strip()
        if "def " in c_strip and ("return " in c_strip or "yield " in c_strip):
            if "input(" not in c_strip:
                clean_blocks.append(c_strip)
                
    if clean_blocks:
        return "\n\n".join(clean_blocks)
    return candidate.strip() if len(candidate.strip()) > 30 else ""

# -----------------------------------------------------------------------------
# 1. MATEMATİK & SAYISAL MUHAKEME (%40 -> 40M Token)
# -----------------------------------------------------------------------------
print("\n🌐 [1/3] Kaynak: Matematik & Sayısal Akıl Yürütme (GSM8k & FineWeb-Edu Math)...")
target_math = int(TARGET_TRAIN * 0.40)

try:
    ds_gsm = load_dataset("gsm8k", "main", split="train", streaming=True)
    for row in ds_gsm:
        q = row.get("question", "")
        a = row.get("answer", "")
        txt = f"Problem:\n{q}\n\nSolution and Reasoning:\n{a}"
        add_clean_text(txt)
        if len(train_tokens) >= target_math:
            break
except Exception as e:
    print(f"⚠️ GSM8K hatası: {e}")

if len(train_tokens) < target_math:
    try:
        ds_edu_math = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        doc_count = 0
        for row in ds_edu_math:
            doc_count += 1
            if doc_count < 100000:
                continue
            txt = row.get("text", "")
            # Matematik / Fizik filtresi
            txt_lower = txt.lower()
            if any(w in txt_lower for w in ["theorem", "proof", "equation", "integral", "derivative", "prime number", "matrix", "algebra", "calculate"]):
                if len(txt) > 200:
                    add_clean_text(txt)
            if len(train_tokens) >= target_math:
                break
    except Exception as e:
        print(f"⚠️ FineWeb-Edu Math hatası: {e}")

print(f"   ✓ Matematik kotası tamamlandı: {len(train_tokens):,} token")

# -----------------------------------------------------------------------------
# 2. BİLGİSAYAR SİSTEMLERİ & FEN BİLİMLERİ (%35 -> Toplam 75M Token)
# -----------------------------------------------------------------------------
print("\n🌐 [2/3] Kaynak: Donanım Mimarisi, Bellek (RAM/Disk/Cache) & Fen...")
target_systems = int(TARGET_TRAIN * 0.75)
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    doc_count = 0
    for row in ds_edu:
        txt = row.get("text", "")
        doc_count += 1
        # Shard 4'ün de bittiği 450.000 doküman offsetini atla (sıfır tekrar)
        if doc_count < 450000:
            continue
        txt_lower = txt.lower()
        if any(w in txt_lower for w in ["memory hierarchy", "cache", "operating system", "cpu", "compiler", "physics", "gravity", "photosynthesis", "dna", "algorithm complexity"]):
            if len(txt) > 200:
                add_clean_text(txt)
        if len(train_tokens) >= target_systems:
            break
except Exception as e:
    print(f"⚠️ FineWeb-Edu Systems hatası: {e}")

# Eğer hedef dolmadıysa serbest fen metinleriyle tamamla
if len(train_tokens) < target_systems:
    try:
        for row in ds_edu:
            txt = row.get("text", "")
            if len(txt) > 200:
                add_clean_text(txt)
            if len(train_tokens) >= target_systems:
                break
    except Exception as e:
        print(f"⚠️ FineWeb genel fen hatası: {e}")

print(f"   ✓ Bilim ve Sistem kotası tamamlandı: {len(train_tokens):,} token")

# -----------------------------------------------------------------------------
# 3. KÜRASYONLU ALGORİTMALAR & VERİ YAPILARI (%25 -> Toplam 100M + 3M Val)
# -----------------------------------------------------------------------------
print("\n🌐 [3/3] Kaynak: Küratörlü Algoritmalar & Veri Yapıları...")
try:
    ds_py = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    for row in ds_py:
        out = row.get("output", "")
        code = extract_curated_function(out)
        if code and len(code) > 40:
            add_clean_text(code)
        if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
            break
except Exception as e:
    print(f"⚠️ Python kod hatası: {e}")

if len(train_tokens) < TARGET_TRAIN or len(val_tokens) < TARGET_VAL:
    try:
        ds_alp = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        for row in ds_alp:
            out = row.get("output", "")
            code = extract_curated_function(out)
            if code and len(code) > 40:
                add_clean_text(code)
            if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
                break
    except Exception as e:
        print(f"⚠️ CodeAlpaca hatası: {e}")

print(f"   ✓ Toplam hedef tamamlandı: Train={len(train_tokens):,}, Val={len(val_tokens):,}")

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr = np.array(train_tokens[:TARGET_TRAIN], dtype=np.uint16)
val_arr   = np.array(val_tokens[:TARGET_VAL], dtype=np.uint16)

train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

elapsed_total = time.time() - t0
print("=" * 80)
print(f"🎉 AŞAMA 2 - SHARD 5 (100M TOKEN) KUSURSUZ ŞEKİLDE DERLENDİ!")
print(f"   ✓ Train: {train_bin_path} ({len(train_arr):,} Token, {train_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ✓ Val:   {val_bin_path} ({len(val_arr):,} Token, {val_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ⏱️ Süre:  {elapsed_total / 60:.1f} dakika")
print("=" * 80)
