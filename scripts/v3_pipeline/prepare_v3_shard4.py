import os
import sys
import json
import glob
import numpy as np
from tokenizers import Tokenizer

def pack_shard4():
    tok_path = "kaggle_upload/dorina-v3-dataset/dorina_tokenizer_v3_32k.json"
    tok = Tokenizer.from_file(tok_path)
    
    out_dir = "data/shard4_package"
    os.makedirs(out_dir, exist_ok=True)
    train_bin = os.path.join(out_dir, "v3_shard4_train.bin")
    val_bin = os.path.join(out_dir, "v3_shard4_val.bin")
    
    # 400M Token Hedefi (uint16 -> 800 MB train dosyası)
    TARGET_TRAIN_TOKENS = 400_000_000
    VAL_TOKENS = 2_000_000
    
    print("=" * 60)
    print("DORINA V3 SHARD 4 HEDEF KARIŞIM PAKETLEYİCİ")
    print(f"Hedef: {TARGET_TRAIN_TOKENS:,} Train Token + {VAL_TOKENS:,} Val Token")
    print("=" * 60)
    
    # uint16 memmap
    train_mmap = np.memmap(train_bin, dtype=np.uint16, mode='w+', shape=(TARGET_TRAIN_TOKENS,))
    val_mmap = np.memmap(val_bin, dtype=np.uint16, mode='w+', shape=(VAL_TOKENS,))
    
    train_pos = 0
    val_pos = 0
    
    # 1. En kritik: İlaç veri seti (Targeted Medicine) x 5 Epoch tekrar ile güçlü enjeksiyon
    medicine_file = "data/targeted_medicine_v3.jsonl"
    print(f"\n[1/4] Ilac Veri Seti Enjekte Ediliyor: {medicine_file}")
    med_tokens = []
    with open(medicine_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "messages" in obj:
                text = ""
                for m in obj["messages"]:
                    text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            else:
                text = obj.get("text", "") + "\n\n"
            enc = tok.encode(text).ids
            med_tokens.extend(enc)
            
    print(f"   Ilac baz token boyutu: {len(med_tokens):,} token")
    # 5 kez tekrarla (over-sampling)
    for _ in range(5):
        if train_pos + len(med_tokens) < TARGET_TRAIN_TOKENS:
            train_mmap[train_pos : train_pos + len(med_tokens)] = np.array(med_tokens, dtype=np.uint16)
            train_pos += len(med_tokens)
            
    print(f"   Ilac sonrasi Train Token: {train_pos:,}")
    
    # 2. Distilled Teachers & SFT Master Setleri (Reasoning, Math, Code)
    teacher_files = glob.glob("data/distilled_teachers/*.jsonl")
    print(f"\n[2/4] Ogretmen Modeller ve Akil Yurutme Setleri: {len(teacher_files)} dosya")
    for tf in teacher_files:
        with open(tf, "r", encoding="utf-8") as f:
            t_toks = []
            for line in f:
                try:
                    obj = json.loads(line)
                    if "messages" in obj:
                        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in obj["messages"])
                    elif "prompt" in obj and "response" in obj:
                        text = f"<|im_start|>user\n{obj['prompt']}<|im_end|>\n<|im_start|>assistant\n{obj['response']}<|im_end|>\n"
                    else:
                        text = obj.get("text", "") + "\n"
                    t_toks.extend(tok.encode(text).ids)
                except Exception:
                    continue
        # 3 kez tekrarla
        for _ in range(3):
            if train_pos + len(t_toks) < TARGET_TRAIN_TOKENS:
                train_mmap[train_pos : train_pos + len(t_toks)] = np.array(t_toks, dtype=np.uint16)
                train_pos += len(t_toks)
                
    print(f"   Ogretmenler sonrasi Train Token: {train_pos:,}")
    
    # 3. tok_v3 Math & Code & Turkish JSONL korpusu
    curated_files = [
        ("data/tok_v3/math.jsonl", 3),     # Math x 3 kat
        ("data/tok_v3/turkish.jsonl", 2),  # Turkish x 2 kat
        ("data/tok_v3/code.jsonl", 1),     # Code x 1 kat
        ("data/tok_v3/english.jsonl", 1)   # English x 1 kat
    ]
    print(f"\n[3/4] Matematik, Turkce ve Kod Korpusu Dolduruluyor...")
    for cfile, repeat in curated_files:
        if not os.path.exists(cfile):
            continue
        print(f"   Isleniyor: {cfile} (Carpan: x{repeat})")
        with open(cfile, "r", encoding="utf-8") as f:
            batch_toks = []
            for line in f:
                if train_pos >= TARGET_TRAIN_TOKENS:
                    break
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        batch_toks.extend(tok.encode(text).ids)
                        if len(batch_toks) >= 500_000:
                            take = min(len(batch_toks), TARGET_TRAIN_TOKENS - train_pos)
                            train_mmap[train_pos : train_pos + take] = np.array(batch_toks[:take], dtype=np.uint16)
                            train_pos += take
                            batch_toks = batch_toks[take:]
                except Exception:
                    continue
            if batch_toks and train_pos < TARGET_TRAIN_TOKENS:
                take = min(len(batch_toks), TARGET_TRAIN_TOKENS - train_pos)
                train_mmap[train_pos : train_pos + take] = np.array(batch_toks[:take], dtype=np.uint16)
                train_pos += take

    # 4. TinyStories ile kalan bosluk tam 400M'a tamamlanir
    if train_pos < TARGET_TRAIN_TOKENS:
        print(f"\n[4/4] Kalan {TARGET_TRAIN_TOKENS - train_pos:,} token TinyStories ile tamamlanıyor...")
        with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
            chunk = []
            for line in f:
                if train_pos >= TARGET_TRAIN_TOKENS:
                    break
                chunk.extend(tok.encode(line).ids)
                if len(chunk) >= 500_000:
                    take = min(len(chunk), TARGET_TRAIN_TOKENS - train_pos)
                    train_mmap[train_pos : train_pos + take] = np.array(chunk[:take], dtype=np.uint16)
                    train_pos += take
                    chunk = []

    # Val seti (1M Math + 1M Turkce)
    print("\n[Val Seti Olusturuluyor...]")
    val_files = glob.glob("data/tok_v3/*heldout.jsonl")
    v_toks = []
    for vf in val_files:
        with open(vf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    v_toks.extend(tok.encode(obj.get("text", "")).ids)
                    if len(v_toks) >= VAL_TOKENS:
                        break
                except Exception:
                    continue
        if len(v_toks) >= VAL_TOKENS:
            break
            
    val_mmap[:min(len(v_toks), VAL_TOKENS)] = np.array(v_toks[:VAL_TOKENS], dtype=np.uint16)
    
    train_mmap.flush()
    val_mmap.flush()
    
    print("\n" + "=" * 60)
    print(f"[BAŞARILI] Shard 4 Paketi Hazır:")
    print(f"Train Binary: {train_bin} ({os.path.getsize(train_bin)/(1024*1024):.2f} MB - {TARGET_TRAIN_TOKENS:,} token)")
    print(f"Val Binary  : {val_bin} ({os.path.getsize(val_bin)/(1024*1024):.2f} MB - {VAL_TOKENS:,} token)")
    print("=" * 60)

if __name__ == "__main__":
    pack_shard4()
