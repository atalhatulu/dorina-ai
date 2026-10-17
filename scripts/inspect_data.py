import os
import sys
import numpy as np
from tokenizers import Tokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def inspect_dataset(bin_file_name, sample_tokens=250):
    bin_path = os.path.join(DATA_DIR, bin_file_name)
    if not os.path.exists(bin_path):
        print(f"⚠️ Dosya bulunamadı: {bin_file_name}")
        return
        
    tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
    tokenizer = Tokenizer.from_file(tok_path)
    
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    
    print("\n" + "=" * 80)
    print(f"📦 VERİ SETİ: {bin_file_name}")
    print(f"   • Toplam Token Sayısı : {total_tokens:,} Token ({data.nbytes / (1024*1024):.1f} MB)")
    print(f"   • Örnek Alınan Aralık : İlk {sample_tokens} Token")
    print("-" * 80)
    
    sample_ids = data[:sample_tokens].tolist()
    text = tokenizer.decode(sample_ids)
    print(text)
    print("=" * 80)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "stage2_shard1_100m_train.bin"
    inspect_dataset(target)
