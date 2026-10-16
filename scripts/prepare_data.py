import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot_token = enc.eot_token

# Toplam hedef: ~140 Milyon Token
SOURCES = [
    {
        "name": "TinyStories (Anlatı ve Diyalog)",
        "path": "roneneldan/TinyStories",
        "split": "train",
        "text_key": "text",
        "target_tokens": 45_000_000
    },
    {
        "name": "SciPhi Textbooks (Fen, Fizik, Bilim)",
        "path": "emozilla/pg19-sciphi-textbooks",
        "split": "train",
        "text_key": "text",
        "target_tokens": 55_000_000
    },
    {
        "name": "Simple Wikipedia (Ansiklopedik Tanımlar)",
        "path": "pszemraj/simple_wikipedia",
        "split": "train",
        "text_key": "text",
        "target_tokens": 25_000_000
    },
    {
        "name": "Tiny-Codes (Mantık ve Algoritma)",
        "path": "nampdn-ai/tiny-codes",
        "split": "train",
        "text_key": "response",
        "target_tokens": 15_000_000
    }
]

def stream_and_tokenize():
    all_tokens = []
    
    print("\n" + "=" * 65)
    print("🌐 DORINA DATA LAB: ÇOK KAYNAKLI VERİ İNDİRME VE TOKENIZE İŞLEMİ")
    print("=" * 65)

    for src in SOURCES:
        print(f"\n📥 Kaynak Çekiliyor: {src['name']}")
        print(f"🎯 Hedef Token: {src['target_tokens']:,}")
        
        try:
            ds = load_dataset(src["path"], split=src["split"], streaming=True)
        except Exception as e:
            print(f"⚠️ {src['name']} yüklenirken hata oluştu: {e}")
            continue

        collected_tokens = 0
        pbar = tqdm(total=src["target_tokens"], unit="tok", desc=src["name"][:20])

        for item in ds:
            text = item.get(src["text_key"], "")
            if not text or len(text.strip()) < 10:
                continue
                
            # Tokenize et ve cümlenin/belgenin sonuna EOT ekle
            tokens = enc.encode_ordinary(text)
            tokens.append(eot_token)
            
            all_tokens.extend(tokens)
            collected_tokens += len(tokens)
            pbar.update(len(tokens))

            if collected_tokens >= src["target_tokens"]:
                break
                
        pbar.close()

    total_count = len(all_tokens)
    print(f"\n✓ Toplam Toplanan Token Sayısı: {total_count:,}")

    # %90 Train, %10 Validation ayrımı
    split_idx = int(total_count * 0.90)
    train_tokens = np.array(all_tokens[:split_idx], dtype=np.uint16)
    val_tokens = np.array(all_tokens[split_idx:], dtype=np.uint16)

    print("\n💾 .bin Dosyaları Diske Yazılıyor...")
    train_tokens.tofile(os.path.join(DATA_DIR, "tinystories_train.bin"))
    val_tokens.tofile(os.path.join(DATA_DIR, "tinystories_val.bin"))

    print(f"✓ train.bin: {len(train_tokens):,} token ({os.path.getsize(os.path.join(DATA_DIR, 'tinystories_train.bin')) / (1024*1024):.2f} MB)")
    print(f"✓ val.bin:   {len(val_tokens):,} token ({os.path.getsize(os.path.join(DATA_DIR, 'tinystories_val.bin')) / (1024*1024):.2f} MB)")
    print("\n🎉 Veri seti hazır! Artık 70M modelin eğitimine başlayabilirsin.")

if __name__ == "__main__":
    stream_and_tokenize()