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
print("🦀 DORINA-MoE: FAZ 2 (35M TOKEN) MANTIK, BİLİM & TÜRKÇE AKIL VERİ SETİ")
print("   • %40 Mantık, Bilim & FineWeb-Edu | %30 Türkçe Muhakeme | %30 Algoritmik Problem Çözme")
print("=" * 70)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "dorina_moe_phase2_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "dorina_moe_phase2_val.bin")

TARGET_TRAIN = 35_000_000
TARGET_VAL   = 2_500_000

train_tokens = []
val_tokens = []
total_docs = 0
t0 = time.time()

def add_text(text):
    global total_docs
    if not text or len(text.strip()) < 15:
        return
    ids = tokenizer.encode(text).ids
    ids.append(eos_id)
    if len(train_tokens) < TARGET_TRAIN:
        train_tokens.extend(ids)
    elif len(val_tokens) < TARGET_VAL:
        val_tokens.extend(ids)
    total_docs += 1
    if total_docs % 2500 == 0:
        print(f"   ↳ {total_docs:,} döküman | Train: {len(train_tokens):,} / {TARGET_TRAIN:,} | Val: {len(val_tokens):,} ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------------
# 1. TÜRKÇE MUHAKEME & DİL ZENGİNLİĞİ (~10M Token)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 1: Türkçe Doğal Dil & Talimat Havuzları...")
try:
    ds_tr = load_dataset("merve/turkish_instructions", split="train")
    for row in ds_tr:
        inst = row.get("talimat", "") or ""
        inp = row.get(" giri\u015f", "") or row.get("giris", "") or ""
        out = row.get(" \u00e7\u0131kt\u0131", "") or row.get("cikti", "") or ""
        
        user_msg = f"{inst.strip()}\n{inp.strip()}".strip() if inp else inst.strip()
        if user_msg and out:
            text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{out.strip()}<|im_end|>"
            add_text(text)
        if len(train_tokens) >= int(TARGET_TRAIN * 0.30):
            break
except Exception as e:
    print(f"⚠️ merve/turkish_instructions atlandı/hata: {e}")

# -----------------------------------------------------------------------------
# 2. MANTIK, BİLİM VE EĞİTİCİ METİNLER (~15M Token)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 2: FineWeb-Edu Bilim, Felsefe & Mantık...")
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds_edu:
        txt = row.get("text", "")
        if len(txt) > 250:
            add_text(txt)
        if len(train_tokens) >= int(TARGET_TRAIN * 0.70):
            break
except Exception as e:
    print(f"⚠️ FineWeb-Edu hatası: {e}")

# -----------------------------------------------------------------------------
# 3. İLERİ SEVİYE ALGORİTMALAR & PROBLEM ÇÖZME (~10M Token)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 3: Kod Çözümleri & Çok Aşamalı Mantık...")
try:
    ds_code = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
    for row in ds_code:
        q = row.get("query", "")
        a = row.get("answer", "")
        if q and a:
            text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<|code_start|>\n{a}\n<|code_end|><|im_end|>"
            add_text(text)
        if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
            break
except Exception as e:
    print(f"⚠️ CodeFeedback hatası: {e}")

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr = np.array(train_tokens[:TARGET_TRAIN], dtype=np.uint16)
val_arr   = np.array(val_tokens[:TARGET_VAL], dtype=np.uint16)

train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

print("=" * 70)
print("🎉 DORINA-MoE FAZ 2 VERİ SETİ DERLEMESİ BAŞARIYLA TAMAMLANDI!")
print(f"   ✓ Train Dosyası: {train_bin_path} ({len(train_arr):,} Token, {train_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ✓ Val Dosyası:   {val_bin_path} ({len(val_arr):,} Token, {val_arr.nbytes / (1024*1024):.1f} MB)")
print("=" * 70)
