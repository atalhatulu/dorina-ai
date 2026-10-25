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

print("=" * 70)
print("🦀 DORINA-MoE: BÜYÜK DAHİ MÜFREDATI (100M TOKEN PRE-TRAIN)")
print("   • %35 Kod & Algoritma | %35 Mantık & Bilim | %20 Türkçe Ansiklopedi | %10 Yapılandırılmış")
print("=" * 70)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "dorina_moe_grand_100m_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "dorina_moe_grand_100m_val.bin")

TARGET_TRAIN = 100_000_000
TARGET_VAL   = 5_000_000

train_tokens = []
val_tokens = []
total_docs = 0
t0 = time.time()

def add_text(text):
    global total_docs
    if not text or len(text.strip()) < 20:
        return
    ids = tokenizer.encode(text).ids
    ids.append(eos_id)
    if len(train_tokens) < TARGET_TRAIN:
        train_tokens.extend(ids)
    elif len(val_tokens) < TARGET_VAL:
        val_tokens.extend(ids)
    total_docs += 1
    if total_docs % 5000 == 0:
        print(f"   ↳ {total_docs:,} döküman | Train: {len(train_tokens):,} / {TARGET_TRAIN:,} | Val: {len(val_tokens):,} ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------------
# 1. TÜRKÇE ANSİKLOPEDİ & GENEL KAVRAMLAR (~20M Token -> %20)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 1: Türkçe Vikipedi & Ansiklopedik Derlem (Kavramlar & Tanımlar)...")
try:
    ds_tr_wiki = load_dataset("umarigan/turkish_wikipedia", split="train", streaming=True)
    for row in ds_tr_wiki:
        t = row.get("title", "")
        c = row.get("text", "")
        if c and len(c) > 100:
            doc = f"{t}\n\n{c}" if t else c
            add_text(doc)
        if len(train_tokens) >= int(TARGET_TRAIN * 0.20):
            break
except Exception as e:
    print(f"⚠️ Türkçe Vikipedi hatası: {e}")

# -----------------------------------------------------------------------------
# 2. MANTIK, BİLİM & EĞİTİCİ METİNLER (~35M Token -> Toplam %55)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 2: FineWeb-Edu Bilim, Felsefe, Mantık & Matematik...")
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds_edu:
        txt = row.get("text", "")
        if len(txt) > 200:
            add_text(txt)
        if len(train_tokens) >= int(TARGET_TRAIN * 0.55):
            break
except Exception as e:
    print(f"⚠️ FineWeb-Edu hatası: {e}")

# -----------------------------------------------------------------------------
# 3. KODLAMA, ALGORİTMALAR & PROBLEM ÇÖZÜMLERİ (~35M Token -> Toplam %90)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 3: Python Algoritmaları, The Stack & StarCoder...")
try:
    ds_code = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
    for row in ds_code:
        q = row.get("query", "")
        a = row.get("answer", "")
        if q and a:
            text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<|code_start|>\n{a}\n<|code_end|><|im_end|>"
            add_text(text)
        if len(train_tokens) >= int(TARGET_TRAIN * 0.90):
            break
except Exception as e:
    print(f"⚠️ CodeFeedback hatası: {e}")

# -----------------------------------------------------------------------------
# 4. YAPILANDIRILMIŞ FORMATLAR & TEMİZ KODLAR (~10M Token -> Toplam %100)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 4: Temiz Python Kütüphaneleri & Kod Dokümantasyonları...")
try:
    ds_parrot = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    for row in ds_parrot:
        c = row.get("content", "")
        if c:
            add_text(f"<|code_start|>\n{c}\n<|code_end|>")
        if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
            break
except Exception as e:
    print(f"⚠️ CodeParrot hatası: {e}")

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr = np.array(train_tokens[:TARGET_TRAIN], dtype=np.uint16)
val_arr   = np.array(val_tokens[:TARGET_VAL], dtype=np.uint16)

train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

print("=" * 70)
print("🎉 100M BÜYÜK DAHİ MÜFREDATI BAŞARIYLA DERLENDİ!")
print(f"   ✓ Train Dosyası: {train_bin_path} ({len(train_arr):,} Token, {train_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ✓ Val Dosyası:   {val_bin_path} ({len(val_arr):,} Token, {val_arr.nbytes / (1024*1024):.1f} MB)")
print("=" * 70)
