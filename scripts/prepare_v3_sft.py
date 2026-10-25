import os
import sys
import json
import numpy as np
from tokenizers import Tokenizer

def prepare_sft():
    tok_path = "kaggle_upload/dorina-v3-dataset/dorina_tokenizer_v3_32k.json"
    tok = Tokenizer.from_file(tok_path)
    
    IM_START = 2
    IM_END = 3
    IGNORE_INDEX = -100
    MAX_SEQ_LEN = 1024
    
    sft_files = [
        "archive/dorina_v2_44m/sft_data/dorina_perfect_sft_15k.jsonl",
        "archive/dorina_v2_44m/sft_data/dorina_broad_sft_15k.jsonl",
        "archive/dorina_v2_44m/sft_data/dorina_identity_sft_300.jsonl",
        "data/distilled_teachers/dorina_gold_sft_master.jsonl"
    ]
    
    all_x = []
    all_y = []
    
    total_valid_examples = 0
    total_tokens_count = 0
    
    print("[SFT Hazırlık] Veri dosyaları işleniyor ve maskeleme uygulanıyor...")
    
    for fpath in sft_files:
        if not os.path.exists(fpath):
            print(f"Uyarı: {fpath} bulunamadı, atlanıyor.")
            continue
            
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    if len(messages) < 2:
                        continue
                        
                    input_ids = []
                    target_ids = []
                    
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"].strip()
                        
                        # Role header: <|im_start|>role\n
                        header_ids = tok.encode(f"<|im_start|>{role}\n").ids
                        content_ids = tok.encode(content).ids
                        footer_ids = tok.encode("<|im_end|>\n").ids
                        
                        full_msg_ids = header_ids + content_ids + footer_ids
                        input_ids.extend(full_msg_ids)
                        
                        if role in ["system", "user"]:
                            # Loss maskeleme: user ve system tokenlarına loss hesaplanmaz!
                            target_ids.extend([IGNORE_INDEX] * len(full_msg_ids))
                        else:
                            # Assistant: Role header maskelenir, içerik ve <|im_end|> eğitilir!
                            target_ids.extend([IGNORE_INDEX] * len(header_ids))
                            target_ids.extend(content_ids + footer_ids)
                            
                    if len(input_ids) > MAX_SEQ_LEN:
                        input_ids = input_ids[:MAX_SEQ_LEN]
                        target_ids = target_ids[:MAX_SEQ_LEN]
                        
                    if len(input_ids) < 8:
                        continue
                        
                    # Model sonraki tokenı tahmin ettiği için:
                    # x = input_ids[:-1], y = target_ids[1:]
                    x_arr = np.array(input_ids[:-1], dtype=np.uint16)
                    y_arr = np.array(target_ids[1:], dtype=np.int16)
                    
                    all_x.append(x_arr)
                    all_y.append(y_arr)
                    total_valid_examples += 1
                    total_tokens_count += len(x_arr)
                    
                except Exception as e:
                    continue
                    
    print(f"[OK] Toplam Gecerli SFT Ornegi: {total_valid_examples:,d}")
    print(f"[OK] Toplam Token Hacmi: {total_tokens_count:,d} token")
    
    # Train / Val ayrımı (%98 Train, %2 Val)
    indices = np.random.RandomState(42).permutation(total_valid_examples)
    split_idx = int(total_valid_examples * 0.98)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    out_dir = "data/sft_v3"
    os.makedirs(out_dir, exist_ok=True)
    
    train_x = [all_x[i] for i in train_idx]
    train_y = [all_y[i] for i in train_idx]
    
    val_x = [all_x[i] for i in val_idx]
    val_y = [all_y[i] for i in val_idx]
    
    np.savez_compressed(os.path.join(out_dir, "sft_train.npz"), x=np.array(train_x, dtype=object), y=np.array(train_y, dtype=object))
    np.savez_compressed(os.path.join(out_dir, "sft_val.npz"), x=np.array(val_x, dtype=object), y=np.array(val_y, dtype=object))
    
    print(f"[KAYDEDILDI]: {out_dir}/sft_train.npz ({len(train_x):,d} ornek)")
    print(f"[KAYDEDILDI]: {out_dir}/sft_val.npz ({len(val_x):,d} ornek)")

if __name__ == "__main__":
    prepare_sft()
