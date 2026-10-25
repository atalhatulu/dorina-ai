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
# Dorina-278M Aşama 2: FineWeb-Edu (Dünya Bilgisi, Tarih, Bilim, Ansiklopedi)
# Hedef: 50.000.000 Token (Yüksek kaliteli web metinleri, score >= 3.0)
# Boyut: ~100 MB (.bin uint16)
# -----------------------------------------------------------------------------
TARGET_TOKENS = 50_000_000
CHUNK_FLUSH_TOKENS = 5_000_000

def main():
    print("=" * 70)
    print("🌐 DORINA-278M VERİ HAZIRLAMA: AŞAMA 2 (FINEWEB-EDU GENİŞLEME)")
    print(f"🎯 Hedef Token Hacmi: {TARGET_TOKENS:,} (~50 Milyon Token)")
    print("📦 Kaynak: HuggingFaceFW/fineweb-edu (sample-10BT | score >= 3.0)")
    print("=" * 70)

    train_bin_path = os.path.join(DATA_DIR, "dorina278m_phase2_train.bin")
    val_bin_path = os.path.join(DATA_DIR, "dorina278m_phase2_val.bin")

    if os.path.exists(train_bin_path): os.remove(train_bin_path)
    if os.path.exists(val_bin_path): os.remove(val_bin_path)

    target_train_tokens = int(TARGET_TOKENS * 0.90)
    target_val_tokens = TARGET_TOKENS - target_train_tokens

    print(f"   • Hedef Train: {target_train_tokens:,} token (~{target_train_tokens*2/(1024*1024):.1f} MB)")
    print(f"   • Hedef Val:   {target_val_tokens:,} token (~{target_val_tokens*2/(1024*1024):.1f} MB)")
    print(f"   • Toplam Disk: ~100 MB")
    print("\n⏳ Akış (Streaming) başlatılıyor...")

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    t0 = time.time()
    collected_total = 0
    buffer_tokens = []
    skipped_low_score = 0
    
    pbar = tqdm(total=TARGET_TOKENS, unit="tok", unit_scale=True, desc="FineWeb-Edu (score>=3)")

    with open(train_bin_path, "ab") as f_train, open(val_bin_path, "ab") as f_val:
        for item in ds:
            # Sadece eğitim puanı 3 ve üzeri olan yüksek kaliteli veriler
            score = item.get("score", 0.0)
            if score is None or score < 3.0:
                skipped_low_score += 1
                continue

            text = item.get("text", "")
            if not text or len(text.strip()) < 80:
                continue

            tokens = enc.encode_ordinary(text)
            tokens.append(eot_token)
            
            buffer_tokens.extend(tokens)
            collected_total += len(tokens)
            pbar.update(len(tokens))

            if len(buffer_tokens) >= CHUNK_FLUSH_TOKENS or collected_total >= TARGET_TOKENS:
                chunk_arr = np.array(buffer_tokens, dtype=np.uint16)
                
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
    
    train_tokens = os.path.getsize(train_bin_path) // 2
    val_tokens = os.path.getsize(val_bin_path) // 2

    print("\n" + "=" * 70)
    print("✅ DORINA-278M AŞAMA 2 VERİ HAZIRLIĞI BAŞARIYLA TAMAMLANDI!")
    print(f"⏱️ Toplam Süre: {elapsed/60:.1f} dakika ({collected_total/elapsed:.1f} token/sn)")
    print(f"📁 Train Dosyası: {train_bin_path} ({train_tokens:,} token)")
    print(f"📁 Val Dosyası:   {val_bin_path} ({val_tokens:,} token)")
    print("=" * 70)

if __name__ == "__main__":
    main()
