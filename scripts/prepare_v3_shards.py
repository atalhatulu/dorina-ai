"""
================================================================================
DORINA V3: HOMOJEN 10-80-10 SHARD 1 PAKETLEYİCİ (200 MİLYON TOKEN)
================================================================================
Müfredat Dağılımı:
  - %50 FineWeb-Edu (100M Token - Küresel Mantık & Bilim)
  - %30 CodeParrot / Python Codes (60M Token - Saf Kodlama & Algoritmalar)
  - %10 Türkçe Wikipedia (20M Token - Doğal Türkçe Dilbilgisi)
  - %10 Elit Öğretmen Verisi (20M Token - 2.008 Adet Damıtılmış Soru-Cevap)
Tüm metinler Tokenizer V3 (32.768 Byte-Level BPE) ile tokenlaştırılır
ve homojen şekilde karıştırılarak (interleaved/shuffled) binary olarak yazılır.
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
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TOK_PATH = os.path.join(ROOT_DIR, "tokenizers", "dorina_tokenizer_v3_32k.json")
OUTPUT_BIN = os.path.join(DATA_DIR, "v3_shard1_200m_train.bin")
OUTPUT_VAL_BIN = os.path.join(DATA_DIR, "v3_shard1_val.bin")

TOTAL_TARGET_TOKENS = 200_000_000  # 200 Milyon Token
VAL_TARGET_TOKENS   =   2_000_000  # 2 Milyon Doğrulama Tokenı

# Kategori Hedefleri (Train)
TARGETS = {
    "edu": int(TOTAL_TARGET_TOKENS * 0.50),       # 100M
    "code": int(TOTAL_TARGET_TOKENS * 0.30),      # 60M
    "turkish": int(TOTAL_TARGET_TOKENS * 0.10),   # 20M
    "teachers": int(TOTAL_TARGET_TOKENS * 0.10)   # 20M (Döngüsel aşı)
}

print("=" * 80)
print("🚀 DORINA V3: 200M TOKENLIK HOMOJEN SHARD 1 PAKETLEME MOTORU")
print("=" * 80)
print(f"Hedef Token Sayısı: {TOTAL_TARGET_TOKENS:,}")
print(f"Tokenizer: {TOK_PATH}")
print(f"Çıktı Dosyası: {OUTPUT_BIN}")
print("-" * 80)

# 1. Tokenizer Yükle
tokenizer = Tokenizer.from_file(TOK_PATH)
eos_id = tokenizer.token_to_id("<|endoftext|>")  # 0
im_start_id = tokenizer.token_to_id("<|im_start|>")
im_end_id = tokenizer.token_to_id("<|im_end|>")

# 2. Öğretmen Verilerini Oku ve Tokenlaştır (2.008 Örnek)
teacher_tokens = []
teacher_files = [
    "deepseek_code.jsonl", "gemini_turkish_identity.jsonl",
    "gpt4_math.jsonl", "kimi_logic.jsonl", "qwen_code.jsonl"
]

print("📚 1. Katman: 2.008 Adet Elit Öğretmen Verisi Okunuyor...")
for tf in teacher_files:
    p = os.path.join(DATA_DIR, "distilled_teachers", tf)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    u = d["messages"][0]["content"]
                    a = d["messages"][1]["content"]
                    # ChatML formatında tokenize et
                    chat_text = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
                    ids = tokenizer.encode(chat_text).ids
                    teacher_tokens.extend(ids)
                except Exception:
                    pass

print(f"   ✓ Toplam Öğretmen Token Havuzu: {len(teacher_tokens):,} Token")

# Öğretmen verisini 20M hedef için tekrarlı döngüye hazırla
def iter_teacher_tokens():
    while True:
        for t in teacher_tokens:
            yield t

teacher_gen = iter_teacher_tokens()

# 3. Akışlı (Streaming) Veri Kaynakları
print("\n🌐 2. Katman: HuggingFace Akışlı Kaynaklar Bağlanıyor...")

print("   • FineWeb-Edu (100M Token Hedefi)...")
ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

print("   • CodeParrot Python (60M Token Hedefi)...")
ds_code = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)

print("   • Türkçe Wikipedia (20M Token Hedefi)...")
ds_tr = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)

iter_edu = iter(ds_edu)
iter_code = iter(ds_code)
iter_tr = iter(ds_tr)

# 4. Homojen Karıştırma ve Binary Dosyaya Yazma
print("\n⚡ 3. Katman: Homojen Karıştırma (Data Interleaving) Başlıyor...")
t0 = time.time()

# Memmap tabanlı hızlı yazıcı
train_arr = np.memmap(OUTPUT_BIN, dtype=np.uint16, mode='w+', shape=(TOTAL_TARGET_TOKENS,))
val_arr   = np.memmap(OUTPUT_VAL_BIN, dtype=np.uint16, mode='w+', shape=(VAL_TARGET_TOKENS,))

written_train = 0
written_val = 0

batch_buffer = []
counts = {"edu": 0, "code": 0, "turkish": 0, "teachers": 0}

def get_tokens_from_stream(stream_iter, field_name):
    try:
        row = next(stream_iter)
        text = row.get(field_name, "")
        if len(text.strip()) > 30:
            return tokenizer.encode(text).ids + [eos_id]
    except Exception:
        pass
    return []

print(f"Yazım başladı... Hedef: {TOTAL_TARGET_TOKENS:,} token")

# Hızlı döngü
while written_train < TOTAL_TARGET_TOKENS:
    # Karışım Oranına Göre Çekim:
    # 5 Edu, 3 Kod, 1 Türkçe, 1 Öğretmen
    
    # 1. Edu (5 parça)
    for _ in range(5):
        toks = get_tokens_from_stream(iter_edu, "text")
        batch_buffer.extend(toks)
        counts["edu"] += len(toks)
        
    # 2. Kod (3 parça)
    for _ in range(3):
        toks = get_tokens_from_stream(iter_code, "content")
        batch_buffer.extend(toks)
        counts["code"] += len(toks)
        
    # 3. Türkçe (1 parça)
    toks = get_tokens_from_stream(iter_tr, "text")
    batch_buffer.extend(toks)
    counts["turkish"] += len(toks)
    
    # 4. Öğretmen (1 parça - ~500 token)
    t_chunk = [next(teacher_gen) for _ in range(500)]
    batch_buffer.extend(t_chunk)
    counts["teachers"] += len(t_chunk)
    
    # Tamponu diske boşalt
    if len(batch_buffer) >= 500_000:
        # Önce validation'ı doldur
        if written_val < VAL_TARGET_TOKENS:
            needed_val = min(len(batch_buffer), VAL_TARGET_TOKENS - written_val)
            val_arr[written_val:written_val + needed_val] = batch_buffer[:needed_val]
            written_val += needed_val
            batch_buffer = batch_buffer[needed_val:]
            
        # Kalanı train'e yaz
        needed_train = min(len(batch_buffer), TOTAL_TARGET_TOKENS - written_train)
        if needed_train > 0:
            train_arr[written_train:written_train + needed_train] = batch_buffer[:needed_train]
            written_train += needed_train
            batch_buffer = batch_buffer[needed_train:]
            
        elapsed = time.time() - t0
        speed = written_train / max(1, elapsed)
        pct = (written_train / TOTAL_TARGET_TOKENS) * 100
        print(f"İlerleme: %{pct:5.1f} | Yazılan: {written_train:,} / {TOTAL_TARGET_TOKENS:,} | "
              f"Hız: {speed:,.0f} tok/s | Geçen: {elapsed/60:.1f} dk")

train_arr.flush()
val_arr.flush()

total_time = (time.time() - t0) / 60
print("\n" + "=" * 80)
print(f"🎉 HOMOJEN SHARD 1 PAKETLEME TAMAMLANDI! Süre: {total_time:.1f} dk")
print(f"💾 Train Dosyası : {OUTPUT_BIN} ({os.path.getsize(OUTPUT_BIN) / 1024**2:.1f} MB)")
print(f"💾 Val Dosyası   : {OUTPUT_VAL_BIN} ({os.path.getsize(OUTPUT_VAL_BIN) / 1024**2:.1f} MB)")
print("Dağılım Özeti:")
print(f"   • FineWeb-Edu    : {counts['edu']:,} token")
print(f"   • Kodlama        : {counts['code']:,} token")
print(f"   • Türkçe Wiki    : {counts['turkish']:,} token")
print(f"   • Elit Öğretmen  : {counts['teachers']:,} token")
print("=" * 80)
