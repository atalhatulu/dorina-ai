#!/usr/bin/env python3
import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, 'kaggle_kernel')
from model_moe_v3 import DorinaMoEV3, MoEConfig

class SFTDataset(Dataset):
    def __init__(self, npz_path, max_len=1024):
        data = np.load(npz_path, allow_pickle=True)
        self.x = data["x"]
        self.y = data["y"]
        self.max_len = max_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

def collate_fn(batch):
    # Dinamik Padding (Batch içindeki en uzun sequence'a göre)
    batch_x, batch_y = zip(*batch)
    max_len = max(len(seq) for seq in batch_x)
    
    padded_x = []
    padded_y = []
    for x, y in zip(batch_x, batch_y):
        pad_len = max_len - len(x)
        # Pad token: 0 (veya boşluk), target'ta pad: -100 (loss hesaplanmaz!)
        padded_x.append(torch.cat([x, torch.zeros(pad_len, dtype=torch.long)]))
        padded_y.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]))
        
    return torch.stack(padded_x), torch.stack(padded_y)

def main():
    parser = argparse.ArgumentParser(description="Dorina V3 Supervised Fine-Tuning (SFT)")
    parser.add_argument("--init-weights", type=str, default="kaggle_upload/dorina-v3-checkpoint/dorina_v3_step_4500.pt")
    parser.add_argument("--train-data", type=str, default="data/sft_v3/sft_train.npz")
    parser.add_argument("--val-data", type=str, default="data/sft_v3/sft_val.npz")
    parser.add_argument("--out-dir", type=str, default="models/sft_v3")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16) # Efektif batch: 32 sequence
    parser.add_argument("--lr", type=float, default=2e-5) # SFT için düşük LR
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Cihaz]: {device} | {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    # 1. Model Kurulumu
    cfg = MoEConfig()
    model = DorinaMoEV3(cfg).to(device)
    
    print(f"[Yukleniyor]: {args.init_weights}")
    sd = torch.load(args.init_weights, map_location=device, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    print("[OK] Step 4500 agirliklari basariyla aktarildi!")

    # 2. Veri Yukleyiciler
    train_dataset = SFTDataset(args.train_data)
    val_dataset = SFTDataset(args.val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"[Veri]: Train {len(train_dataset):,d} ornek | Val {len(val_dataset):,d} ornek")

    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Toplam adim
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    print(f"[Egitim]: {args.epochs} Epoch | Adim/Epoch: {steps_per_epoch} | Toplam Efektif Adim: {total_steps}")

    # 4. SFT Egitim Dongusu
    global_step = 0
    t0 = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        accum_loss = 0.0
        micro_step = 0
        optimizer.zero_grad(set_to_none=True)

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Gecerli token yoksa (hepsi maskeliyse) atla
            valid_tokens = (y != -100).sum()
            if valid_tokens == 0:
                continue

            with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % 20 == 0:
                    dt = time.time() - t0
                    t0 = time.time()
                    vram = torch.cuda.memory_allocated() / (1024**3)
                    print(f"Epoch {epoch+1}/{args.epochs} | Adim {global_step:4d}/{total_steps} | SFT Loss: {accum_loss:.4f} | VRAM: {vram:.2f}GB | {dt:.1f}s/20adim")
                accum_loss = 0.0

        # Epoch Sonu Dogrulama
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                if (vy != -100).sum() == 0:
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    _, vl = model(vx, vy)
                    val_loss += vl.item()
                    val_count += 1
                if val_count >= 50:
                    break
        val_loss /= max(1, val_count)
        print(f"\n>>> [EPOCH {epoch+1} BITTI] Val Loss: {val_loss:.4f} <<<\n")

        # Checkpoint
        ckpt_path = os.path.join(args.out_dir, f"dorina_v3_sft_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[CHECKPOINT]: {ckpt_path} kaydedildi!\n")

    print("[TAMAMLANDI] Dorina V3 SFT Egitimi Basariyla Sona Erdi!")

if __name__ == "__main__":
    main()
