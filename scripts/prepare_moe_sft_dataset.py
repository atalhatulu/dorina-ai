import os
import sys
import time
import json
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
print("🦀 DORINA-MoE: SFT VERİ SETİ DERLEYİCİ (KİMLİK + MANTIK + TÜRKÇE KOD)")
print("=" * 70)

tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)
eos_id = tokenizer.token_to_id("<|endoftext|>")

train_bin_path = os.path.join(DATA_DIR, "dorina_moe_sft_train.bin")
val_bin_path   = os.path.join(DATA_DIR, "dorina_moe_sft_val.bin")

train_tokens = []
val_tokens = []

def add_dialogue(user_text, assistant_text, is_val=False):
    text = f"<|im_start|>user\n{user_text.strip()}<|im_end|>\n<|im_start|>assistant\n{assistant_text.strip()}<|im_end|>"
    ids = tokenizer.encode(text).ids
    ids.append(eos_id)
    if is_val:
        val_tokens.extend(ids)
    else:
        train_tokens.extend(ids)

# -----------------------------------------------------------------------------
# 1. DORINA ÖZEL KİMLİK & TÜRKÇE KÜLTÜR VERİLERİ (SENTETİK ÇEKİRDEK)
# -----------------------------------------------------------------------------
print("\n💎 Bölüm 1: Dorina Asistan Kimliği & Nezaket Şablonları Ekleniyor...")
identity_data = [
    ("Merhaba!", "Merhaba! Ben Dorina, yapay zeka asistanınım. Sana kodlama, algoritmalar ve genel konularda nasıl yardımcı olabilirim?"),
    ("Sen kimsin?", "Ben Dorina. Mixture of Experts (MoE) mimarisiyle ve Türkçe desteğiyle sıfırdan eğitilmiş akıllı bir yapay zeka modeliyim."),
    ("Adın nedir?", "Benim adım Dorina. Sana nasıl yardımcı olabilirim?"),
    ("Neler yapabilirsin?", "Python başta olmak üzere yazılım geliştirme, algoritmalar, hata ayıklama, matematiksel problemler ve genel soruları yanıtlama konularında sana yardımcı olabilirim."),
    ("Kim tarafından geliştirildin?", "Ben Talha tarafından geliştirilen Dorina-AI projesinin MoE mimarili yeni nesil dil modeliyim."),
    ("Nasılsın?", "Harikayım, teşekkürler! Seninle çalışmak için hazırım. Bugün ne inşa etmek istersin?"),
    ("Günaydın", "Günaydın! Umarım harika bir gün olur. Bugün ne üzerinde çalışıyoruz?"),
    ("İyi akşamlar", "İyi akşamlar! Kodlama veya araştırmalarında yardıma ihtiyacın olursa buradayım."),
    ("Teşekkür ederim", "Rica ederim, her zaman buradayım! Başka bir sorun olursa sormaktan çekinme."),
    ("Python nedir?", "Python; okunabilirliği yüksek, sade sözdizimine sahip, nesne yönelimli ve çok amaçlı popüler bir programlama dilidir."),
    ("Bir sayının çift olup olmadığını nasıl anlarım?", "Bir sayının 2'ye bölümünden kalan 0 ise o sayı çifttir. Python örneği:\n```python\ndef cift_mi(sayi):\n    return sayi % 2 == 0\n```"),
    ("Fibonacci serisi nedir?", "Fibonacci serisi, her sayının kendisinden önceki iki sayının toplamı olduğu sayı dizisidir (0, 1, 1, 2, 3, 5, 8, 13...)."),
    ("Asal sayı nedir?", "Asal sayı, sadece 1'e ve kendisine tam bölünebilen 1'den büyük doğal sayılardır (2, 3, 5, 7, 11...).")
]

# Kimlik verilerini ağırlıklandırmak için 30'ar kez tekrarla
for u, a in identity_data:
    for _ in range(25):
        add_dialogue(u, a, is_val=False)
    for _ in range(5):
        add_dialogue(u, a, is_val=True)

# -----------------------------------------------------------------------------
# 2. TÜRKÇE TALİMAT & DİYALOG HAVUZU (Merve Turkish Instructions)
# -----------------------------------------------------------------------------
print("\n🌐 Bölüm 2: Türkçe Soru-Cevap & Mantık Diyalogları...")
try:
    ds_tr = load_dataset("merve/turkish_instructions", split="train")
    for i, row in enumerate(ds_tr):
        inst = (row.get("talimat", "") or "").strip()
        inp = (row.get(" giri\u015f", "") or row.get("giris", "") or "").strip()
        out = (row.get(" \u00e7\u0131kt\u0131", "") or row.get("cikti", "") or "").strip()
        
        user_msg = f"{inst}\n{inp}".strip() if inp else inst
        if user_msg and out and len(out) > 10:
            is_val = (i % 10 == 0)
            add_dialogue(user_msg, out, is_val=is_val)
        if len(train_tokens) >= 3_500_000:
            break
except Exception as e:
    print(f"⚠️ Türkçe talimatlar hatası: {e}")

# -----------------------------------------------------------------------------
# 3. KOD TALİMATLARI & TÜRKÇE AÇIKLAMALI KODLAR
# -----------------------------------------------------------------------------
print("\n🌐 Bölüm 3: Açıklamalı Python Kodlama Talimatları...")
try:
    ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    for i, row in enumerate(ds_code):
        inst = (row.get("instruction", "") or "").strip()
        inp = (row.get("input", "") or "").strip()
        out = (row.get("output", "") or "").strip()
        
        u = f"{inst}\n{inp}".strip() if inp else inst
        if u and out:
            is_val = (i % 10 == 0)
            add_dialogue(u, f"```python\n{out}\n```", is_val=is_val)
        if len(train_tokens) >= 6_000_000:
            break
except Exception as e:
    print(f"⚠️ Kod talimatları hatası: {e}")

print("\n💾 Binary dosyalar diske yazılıyor...")
train_arr = np.array(train_tokens, dtype=np.uint16)
val_arr   = np.array(val_tokens, dtype=np.uint16)

train_arr.tofile(train_bin_path)
val_arr.tofile(val_bin_path)

print("=" * 70)
print("🎉 DORINA-MoE SFT VERİ SETİ DERLEMESİ TAMAMLANDI!")
print(f"   ✓ Train Token: {len(train_arr):,} ({train_arr.nbytes / (1024*1024):.1f} MB)")
print(f"   ✓ Val Token:   {len(val_arr):,} ({val_arr.nbytes / (1024*1024):.1f} MB)")
print("=" * 70)
