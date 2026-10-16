import os
import sys
import time
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Windows terminal utf-8 uyumluluğu
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot_token = enc.eot_token

# -----------------------------------------------------------------------------
# 300M Modeli Aşama 1: Çekirdek Ön Eğitim Verisi
# Hedef: ~250.000.000 Token (Kusursuz gramer, bilimsel kavramlar, ders kitapları)
# Kaynak: HuggingFaceTB/smollm-corpus (cosmopedia-v2)
# -----------------------------------------------------------------------------
TARGET_TOKENS = 250_000_000
CHUNK_FLUSH_TOKENS = 10_000_000  # RAM dolmasını engellemek için her 10M tokende diske yaz

def main():
    print("=" * 70)
    print("🌐 DORINA-300M VERİ HAZIRLAMA: AŞAMA 1 (ÇEKİRDEK PRE-TRAINING)")
    print(f"🎯 Hedef Token Hacmi: {TARGET_TOKENS:,} (~250 Milyon Token)")
    print("📦 Kaynak: HuggingFaceTB/smollm-corpus (cosmopedia-v2 alt kümesi)")
    print("=" * 70)

    train_bin_path = os.path.join(DATA_DIR, "dorina300m_phase1_train.bin")
    val_bin_path = os.path.join(DATA_DIR, "dorina300m_phase1_val.bin")

    # Temiz başlangıç için eski yarım kalanları temizle
    if os.path.exists(train_bin_path): os.remove(train_bin_path)
    if os.path.exists(val_bin_path): os.remove(val_bin_path)

    # %90 Train, %10 Val
    target_train_tokens = int(TARGET_TOKENS * 0.90)
    target_val_tokens = TARGET_TOKENS - target_train_tokens

    print(f"   • Hedef Train: {target_train_tokens:,} token (~{target_train_tokens*2/(1024*1024):.1f} MB)")
    print(f"   • Hedef Val:   {target_val_tokens:,} token (~{target_val_tokens*2/(1024*1024):.1f} MB)")
    print(f"   • Toplam Disk İhtiyacı: ~500 MB (Çok hafif ve optimize)")
    print("\n⏳ Akış (Streaming) başlatılıyor...")

    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)

    t0 = time.time()
    collected_total = 0
    buffer_tokens = []
    
    pbar = tqdm(total=TARGET_TOKENS, unit="tok", unit_scale=True, desc="Cosmopedia-v2")

    with open(train_bin_path, "ab") as f_train, open(val_bin_path, "ab") as f_val:
        for item in ds:
            text = item.get("text", "")
            if not text or len(text.strip()) < 50:
                continue

            # Tokenize et ve sonuna EOT ekle
            tokens = enc.encode_ordinary(text)
            tokens.append(eot_token)
            
            buffer_tokens.extend(tokens)
            collected_total += len(tokens)
            pbar.update(len(tokens))

            # Buffer dolduğunda veya hedefe ulaşıldığında diske akıt (RAM dostu)
            if len(buffer_tokens) >= CHUNK_FLUSH_TOKENS or collected_total >= TARGET_TOKENS:
                chunk_arr = np.array(buffer_tokens, dtype=np.uint16)
                
                # Henüz train hedefi dolmadıysa train dosyasına yaz
                cur_train_size = os.path.getsize(train_bin_path) // 2 if os.path.exists(train_bin_path) else 0
                if cur_train_size < target_train_tokens:
                    needed_for_train = target_train_tokens - cur_train_size
                    train_chunk = chunk_arr[:needed_for_train]
                    val_chunk = chunk_arr[needed_for_train:]
                    
                    if len(train_chunk) > 0:
                        train_chunk.tofile(f_train)
                    if len(val_chunk) > 0:
                        val_chunk.tofile(f_val)
                else:
                    chunk_arr.tofile(f_val)

                buffer_tokens = []

            if collected_total >= TARGET_TOKENS:
                break

    pbar.close()
    elapsed = time.time() - t0

    actual_train_tokens = os.path.getsize(train_bin_path) // 2
    actual_val_tokens = os.path.getsize(val_bin_path) // 2
    actual_total = actual_train_tokens + actual_val_tokens

    print("\n" + "=" * 70)
    print("🎉 AŞAMA 1 VERİ HAZIRLIĞI BAŞARIYLA TAMAMLANDI!")
    print(f"   • Toplam Token: {actual_total:,} ({actual_total/1_000_000:.1f}M Token)")
    print(f"   • Train Dosyası: {train_bin_path} ({actual_train_tokens:,} tok - {os.path.getsize(train_bin_path)/(1024*1024):.2f} MB)")
    print(f"   • Val Dosyası:   {val_bin_path} ({actual_val_tokens:,} tok - {os.path.getsize(val_bin_path)/(1024*1024):.2f} MB)")
    print(f"   • Geçen Süre:   {elapsed/60:.2f} dakika ({actual_total/max(1, elapsed):.0f} token/sn ortalama hız)")
    print("=" * 70)

if __name__ == "__main__":
    main()
