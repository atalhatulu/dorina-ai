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
print("🦀 DORINA-MoE: FAZ 1 (30M TOKEN) ÇOK YÖNLÜ VERİ SETİ DERLEME")
print("   • %40 Algoritma & Kodlama | %30 Mantık & Bilim | %15 Türkçe | %15 Yapılandırılmış")
print("=" * 70)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "dorina_moe_phase1_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "dorina_moe_phase1_val.bin")

TARGET_TRAIN = 30_000_000
TARGET_VAL   = 2_500_000

train_tokens = []
val_tokens = []
total_docs = 0
t0 = time.time()

def add_text_to_dataset(text):
    global total_docs
    if not text or len(text.strip()) < 10:
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
# 1. KOD & ALGORİTMA (~%40 -> 12M Token)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 1: Python Algoritmaları & Problem Çözümleri...")
code_sources = ["m-a-p/CodeFeedback-Filtered-Instruction", "iamtarun/python_code_instructions_18k_alpaca"]
for ds_name in code_sources:
    if len(train_tokens) >= int(TARGET_TRAIN * 0.40):
        break
    try:
        ds = load_dataset(ds_name, split="train", streaming=True)
        for row in ds:
            inst = row.get("query", "") or row.get("instruction", "")
            ans = row.get("answer", "") or row.get("output", "")
            if inst and ans:
                text = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n<|code_start|>\n{ans}\n<|code_end|><|im_end|>"
                add_text_to_dataset(text)
            if len(train_tokens) >= int(TARGET_TRAIN * 0.40):
                break
    except Exception as e:
        print(f"⚠️ {ds_name} atlandı: {e}")

# -----------------------------------------------------------------------------
# 2. MANTIK, BİLİM & EĞİTİM (~%30 -> 9M Token)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 2: Mantık, Eğitim & Bilim (FineWeb-Edu / Wiki)...")
try:
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds_edu:
        txt = row.get("text", "")
        # Kalite filtresi: 200 karakterden uzun, anlamlı eğitici metinler
        if len(txt) > 200:
            add_text_to_dataset(txt)
        if len(train_tokens) >= int(TARGET_TRAIN * 0.70):
            break
except Exception as e:
    print(f"⚠️ FineWeb-Edu atlandı/hata: {e}")
    # Yedek olarak wikitext
    try:
        ds_wiki = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        for row in ds_wiki:
            txt = row.get("text", "")
            if len(txt) > 100:
                add_text_to_dataset(txt)
            if len(train_tokens) >= int(TARGET_TRAIN * 0.70):
                break
    except Exception as e2:
        print(f"⚠️ Wikitext yedek hatası: {e2}")

# -----------------------------------------------------------------------------
# 3. TÜRKÇE DİL TEMELİ (~%15 -> 4.5M Token)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 3: Türkçe Doğal Dil & Kültür Verileri...")
try:
    # Bilgi ve doğal Türkçe içeren açık kaynak havuz
    ds_tr = load_dataset("malhajar/OpenHermes-2.5-tr", split="train", streaming=True)
    for row in ds_tr:
        convs = row.get("conversations", [])
        text_parts = []
        for c in convs:
            role = "user" if c.get("from") == "human" else "assistant"
            val = c.get("value", "")
            text_parts.append(f"<|im_start|>{role}\n{val}<|im_end|>")
        if text_parts:
            add_text_to_dataset("\n".join(text_parts))
        if len(train_tokens) >= int(TARGET_TRAIN * 0.85):
            break
except Exception as e:
    print(f"⚠️ Türkçe veri seti atlandı: {e}")

# -----------------------------------------------------------------------------
# 4. YAPILANDIRILMIŞ FORMATLAR & ALGORİTMA DÖNGÜLERİ (~%15 -> Kalan Kısım)
# -----------------------------------------------------------------------------
print("\n🌐 Kaynak 4: Kod Dokümantasyonu & Gelişmiş Python Kodları...")
try:
    ds_code = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    for row in ds_code:
        content = row.get("content", "")
        if content:
            add_text_to_dataset(f"<|code_start|>\n{content}\n<|code_end|>")
        if len(train_tokens) >= TARGET_TRAIN and len(val_tokens) >= TARGET_VAL:
            break
except Exception as e:
    print(f"⚠️ CodeParrot atlandı: {e}")

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr = np.array(train_tokens[:TARGET_TRAIN], dtype=np.uint16)
val_arr   = np.array(val_tokens[:TARGET_VAL], dtype=np.uint16)

train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

print("=" * 70)
print("🎉 DORINA-MoE FAZ 1 VERİ SETİ DERLEMESİ BAŞARIYLA TAMAMLANDI!")
print(f"   ✓ Train Dosyası: {train_bin_path} ({len(train_arr):,} Token, {train_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ✓ Val Dosyası:   {val_bin_path} ({len(val_arr):,} Token, {val_arr.nbytes / (1024*1024):.1f} MB)")
print("=" * 70)
