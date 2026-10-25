import os
import sys
import json
import time
import random
import requests
import io
import pyarrow.parquet as pq
import numpy as np
import tiktoken
from concurrent.futures import ThreadPoolExecutor

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

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DorinaAI/1.0"}

def format_chatml(user_text: str, assistant_text: str) -> str:
    user_clean = user_text.strip()
    assistant_clean = assistant_text.strip()
    return f"<|im_start|>user\n{user_clean}\n<|im_end|>\n<|im_start|>assistant\n{assistant_clean}\n<|im_end|>\n"

def is_valid_pair(user_text: str, assistant_text: str, max_tokens: int = 220) -> bool:
    if not user_text or not assistant_text:
        return False
    u = user_text.strip()
    a = assistant_text.strip()
    if len(u) < 10 or len(a) < 15:
        return False
    chatml_str = format_chatml(u, a)
    tokens = enc.encode_ordinary(chatml_str)
    if 35 <= len(tokens) <= max_tokens:
        return True
    return False

# -----------------------------------------------------------------------------
# 1. teknium/OpenHermes-2.5 (Hedef: 6.000)
# -----------------------------------------------------------------------------
def collect_openhermes(target_count=6000):
    print(f"\n📦 [1/4] OpenHermes-2.5 toplanıyor (Hedef: {target_count:,})...")
    collected = []
    
    # HF Rows API üzerinden kontrollü sayfalama
    def fetch_oh_page(offset):
        url = f"https://datasets-server.huggingface.co/rows?dataset=teknium/OpenHermes-2.5&config=default&split=train&offset={offset}&limit=100"
        for _ in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    return [row["row"] for row in r.json().get("rows", [])]
                elif r.status_code == 429:
                    time.sleep(2.0)
            except Exception:
                time.sleep(1.0)
        return []

    offset = 0
    batch_size = 100
    workers = 4
    while len(collected) < target_count:
        offsets = [offset + i * batch_size for i in range(workers * 2)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pages = list(executor.map(fetch_oh_page, offsets))

        has_data = False
        for rows in pages:
            if rows: has_data = True
            for r in rows:
                convs = r.get("conversations", [])
                if len(convs) >= 2:
                    u = convs[0].get("value", "")
                    a = convs[1].get("value", "")
                    if is_valid_pair(u, a, max_tokens=220):
                        collected.append({
                            "source": "OpenHermes-2.5",
                            "user": u.strip(),
                            "assistant": a.strip()
                        })
                        if len(collected) >= target_count:
                            break
            if len(collected) >= target_count:
                break

        if not has_data:
            break

        offset += len(offsets) * batch_size
        time.sleep(0.1)
        print(f"  ➔ İlerleme: {len(collected):,}/{target_count:,} (İncelenen satır: {offset:,})", end="\r")

    # Eğer API kotası erken durursa, farkı kaliteli instruct veri setiyle tamamla
    if len(collected) < target_count:
        needed = target_count - len(collected)
        print(f"\n  API sınırı nedeniyle ek {needed:,} örnek 'databricks/dolly-15k' ile tamamlanıyor...")
        url = "https://huggingface.co/api/datasets/databricks/databricks-dolly-15k/parquet/default/train/0.parquet"
        r = requests.get(url, headers=HEADERS, timeout=60)
        table = pq.read_table(io.BytesIO(r.content))
        df = table.to_pandas()
        for _, row in df.iterrows():
            inst = str(row.get("instruction", "")).strip()
            ctx = str(row.get("context", "")).strip()
            resp = str(row.get("response", "")).strip()
            u_text = f"{inst}\n\nBağlam: {ctx}".strip() if ctx else inst
            if is_valid_pair(u_text, resp, max_tokens=220):
                collected.append({
                    "source": "OpenHermes-DollyFallback",
                    "user": u_text,
                    "assistant": resp
                })
                if len(collected) >= target_count:
                    break

    print(f"\n  ✓ OpenHermes tamamlandı: {len(collected):,} adet.")
    return collected

# -----------------------------------------------------------------------------
# 2. HuggingFaceH4/ultrachat_200k (Hedef: 3.000)
# -----------------------------------------------------------------------------
def collect_ultrachat(target_count=3000):
    print(f"\n📦 [2/4] UltraChat-200k toplanıyor (Hedef: {target_count:,})...")
    collected = []
    
    # 1. test_sft dosyası (81 MB)
    print("  UltraChat test_sft indiriliyor...")
    url1 = "https://huggingface.co/api/datasets/HuggingFaceH4/ultrachat_200k/parquet/default/test_sft/0.parquet"
    r1 = requests.get(url1, headers=HEADERS, timeout=90)
    t1 = pq.read_table(io.BytesIO(r1.content))
    df1 = t1.to_pandas()
    for _, row in df1.iterrows():
        msgs = row.get("messages", [])
        if len(msgs) >= 2:
            u = msgs[0].get("content", "")
            a = msgs[1].get("content", "")
            if is_valid_pair(u, a, max_tokens=220):
                collected.append({
                    "source": "UltraChat-200k",
                    "user": u.strip(),
                    "assistant": a.strip()
                })
                if len(collected) >= target_count:
                    break

    # 2. Gerekirse train_sft dosyası (243 MB)
    if len(collected) < target_count:
        print(f"  UltraChat train_sft üzerinden tamamlanıyor ({len(collected):,}/{target_count:,})...")
        url2 = "https://huggingface.co/api/datasets/HuggingFaceH4/ultrachat_200k/parquet/default/train_sft/0.parquet"
        r2 = requests.get(url2, headers=HEADERS, timeout=120)
        t2 = pq.read_table(io.BytesIO(r2.content))
        df2 = t2.to_pandas()
        for _, row in df2.iterrows():
            msgs = row.get("messages", [])
            if len(msgs) >= 2:
                u = msgs[0].get("content", "")
                a = msgs[1].get("content", "")
                if is_valid_pair(u, a, max_tokens=220):
                    collected.append({
                        "source": "UltraChat-200k",
                        "user": u.strip(),
                        "assistant": a.strip()
                    })
                    if len(collected) >= target_count:
                        break

    print(f"  ✓ UltraChat tamamlandı: {len(collected):,} adet.")
    return collected

# -----------------------------------------------------------------------------
# 3. sahil2801/CodeAlpaca-20k (Hedef: 1.800)
# -----------------------------------------------------------------------------
def collect_code_alpaca(target_count=1800):
    print(f"\n📦 [3/4] CodeAlpaca-20k toplanıyor (Hedef: {target_count:,})...")
    url = "https://huggingface.co/api/datasets/sahil2801/CodeAlpaca-20k/parquet/default/train/0.parquet"
    print("  CodeAlpaca Parquet doğrudan indiriliyor (~3.2 MB)...")
    r = requests.get(url, headers=HEADERS, timeout=45)
    table = pq.read_table(io.BytesIO(r.content))
    df = table.to_pandas()

    collected = []
    for _, row in df.iterrows():
        inst = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()

        user_msg = f"{inst}\n\nGirdi: {inp}".strip() if inp else inst
        if is_valid_pair(user_msg, out, max_tokens=220):
            collected.append({
                "source": "CodeAlpaca-20k",
                "user": user_msg,
                "assistant": out
            })
            if len(collected) >= target_count:
                break

    print(f"  ✓ CodeAlpaca tamamlandı: {len(collected):,} adet.")
    return collected

# -----------------------------------------------------------------------------
# 4. Open-Orca/1million-gpt-4 (Hedef: 1.200)
# -----------------------------------------------------------------------------
def collect_open_orca(target_count=1200):
    print(f"\n📦 [4/4] OpenOrca (1M-GPT4) toplanıyor (Hedef: {target_count:,})...")
    url = "https://huggingface.co/api/datasets/Open-Orca/1million-gpt-4/parquet/default/train/0.parquet"
    print("  OpenOrca GPT-4 Parquet doğrudan indiriliyor (~272 MB)...")
    r = requests.get(url, headers=HEADERS, timeout=120)
    table = pq.read_table(io.BytesIO(r.content))
    df = table.to_pandas()
    print(f"  OpenOrca Parquet yüklendi: {len(df):,} satır.")

    collected = []
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        ans = str(row.get("response", "")).strip()
        if is_valid_pair(q, ans, max_tokens=220):
            collected.append({
                "source": "OpenOrca-GPT4",
                "user": q,
                "assistant": ans
            })
            if len(collected) >= target_count:
                break

    print(f"  ✓ OpenOrca tamamlandı: {len(collected):,} adet.")
    return collected

# -----------------------------------------------------------------------------
# Ana İşleme ve Çıktı Üretimi (Yol A + Yol B Desteği)
# -----------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("🚀 DORINA-70M SFT (INSTRUCTION TUNING) VERİ SETİ HAZIRLAMA")
    print("=" * 65)

    t0 = time.time()

    # 1. 4 Kaynaktan Verileri Eksiksiz Topla
    samples_oh = collect_openhermes(6000)
    samples_uc = collect_ultrachat(3000)
    samples_ca = collect_code_alpaca(1800)
    samples_oo = collect_open_orca(1200)

    all_samples = samples_oh + samples_uc + samples_ca + samples_oo
    total_len = len(all_samples)
    print(f"\n🎯 Toplam toplanan geçerli örnek sayısı: {total_len:,}")

    # 2. Rastgele Karıştır (Shuffle)
    random.seed(42)
    random.shuffle(all_samples)

    # 3. %90 Train, %10 Val Ayrımı
    val_count = int(total_len * 0.10)
    train_count = total_len - val_count
    train_samples = all_samples[:train_count]
    val_samples = all_samples[train_count:]

    # 4. JSONL olarak Kaydet (Şeffaf İnceleme)
    train_jsonl = os.path.join(DATA_DIR, "sft_train.jsonl")
    val_jsonl = os.path.join(DATA_DIR, "sft_val.jsonl")

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_jsonl, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n💾 JSONL dosyaları kaydedildi:\n   - {train_jsonl} ({len(train_samples):,} örnek)\n   - {val_jsonl} ({len(val_samples):,} örnek)")

    # 5. Tokenize Et (Yol A: Standart Sürekli Dizi)
    print("\n⚡ [Yol A] Standart Causal Pretrain/SFT için .bin oluşturuluyor...")
    def tokenize_dataset(samples):
        all_token_ids = []
        for s in samples:
            formatted_text = format_chatml(s["user"], s["assistant"])
            ids = enc.encode_ordinary(formatted_text)
            ids.append(eot_token)
            all_token_ids.extend(ids)
        return np.array(all_token_ids, dtype=np.uint16)

    train_bin = tokenize_dataset(train_samples)
    val_bin = tokenize_dataset(val_samples)

    train_bin_path = os.path.join(DATA_DIR, "sft_train.bin")
    val_bin_path = os.path.join(DATA_DIR, "sft_val.bin")

    train_bin.tofile(train_bin_path)
    val_bin.tofile(val_bin_path)

    # 6. [Yol B Desteği] Maskelenmiş SFT için 256 Sabit Blok Tensör Paketi
    print("⚡ [Yol B] Masked SFT için (Prompt ignore=-100) 256-blok tensör matrisi oluşturuluyor...")
    def build_masked_sft_tensors(samples, max_seq_len=256):
        inputs_list = []
        targets_list = []
        
        for s in samples:
            prompt_str = f"<|im_start|>user\n{s['user'].strip()}\n<|im_end|>\n<|im_start|>assistant\n"
            resp_str = f"{s['assistant'].strip()}\n<|im_end|>\n"
            
            p_tokens = enc.encode_ordinary(prompt_str)
            r_tokens = enc.encode_ordinary(resp_str)
            r_tokens.append(eot_token)
            
            seq = p_tokens + r_tokens
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
                p_len = min(len(p_tokens), max_seq_len)
            else:
                p_len = len(p_tokens)
                
            input_ids = list(seq)
            target_ids = [-100] * p_len + list(seq[p_len:])
            
            pad_len = max_seq_len - len(input_ids)
            if pad_len > 0:
                input_ids += [eot_token] * pad_len
                target_ids += [-100] * pad_len
                
            inputs_list.append(input_ids)
            targets_list.append(target_ids)
            
        return {
            "inputs": np.array(inputs_list, dtype=np.int64),
            "targets": np.array(targets_list, dtype=np.int64)
        }

    train_masked = build_masked_sft_tensors(train_samples, max_seq_len=256)
    val_masked = build_masked_sft_tensors(val_samples, max_seq_len=256)

    np.savez_compressed(os.path.join(DATA_DIR, "sft_train_masked.npz"), inputs=train_masked["inputs"], targets=train_masked["targets"])
    np.savez_compressed(os.path.join(DATA_DIR, "sft_val_masked.npz"), inputs=val_masked["inputs"], targets=val_masked["targets"])

    total_tokens = len(train_bin) + len(val_bin)
    print(f"\n📊 BİLGİ VE METRİKLER:")
    print(f"   • Train Örnek:  {len(train_samples):,} adet")
    print(f"   • Val Örnek:    {len(val_samples):,} adet")
    print(f"   • Toplam Token: {total_tokens:,} (~{total_tokens/1_000_000:.2f}M token)")
    print(f"   • Yol A Dosyaları: sft_train.bin ({os.path.getsize(train_bin_path)/(1024*1024):.2f} MB), sft_val.bin")
    print(f"   • Yol B Dosyaları: sft_train_masked.npz ({os.path.getsize(os.path.join(DATA_DIR, 'sft_train_masked.npz'))/(1024*1024):.2f} MB), sft_val_masked.npz")
    print(f"   • Toplam Süre: {time.time()-t0:.1f} saniye")
    print("\n🎉 Tebrikler! Dorina-70M 12.000 örneklik SFT veri kümesi eksiksiz hazırlandı.")

if __name__ == "__main__":
    main()
