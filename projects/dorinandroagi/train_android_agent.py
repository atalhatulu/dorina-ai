import os
import sys
import math
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from kaggle_v4_shard5_kernel.main import DorinaDenseV4, DenseConfig

class AndroidAgentDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        print(f"[*] Android Agent Veri Seti Yuklendi: {len(self.samples):,} ornek")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, lbl = self.samples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(lbl, dtype=torch.long)

def collate_fn(batch):
    max_len = max(len(x[0]) for x in batch)
    inps, lbls = [], []
    for inp, lbl in batch:
        pad = max_len - len(inp)
        inps.append(F.pad(inp, (0, pad), value=0))
        lbls.append(F.pad(lbl, (0, pad), value=-100))
    return torch.stack(inps), torch.stack(lbls)

def train_android_sft():
    print("=" * 70, flush=True)
    print("DORINA-V4 ANDROID OS AGENT EGITIMI (5.924 SAMPLES)", flush=True)
    print("=" * 70, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "data/dorina_gold_sft_v4/dorina_android_agent_dataset.json"
    dataset = AndroidAgentDataset(data_path)

    vocab_size = 32832
    cfg = DenseConfig(vocab_size=vocab_size, max_seq_len=2048)
    model = DorinaDenseV4(cfg)

    # Shard 6 Kristal Temel Modelinden basliyoruz!
    base_path = "models/dorina_v4/dorina_v4_shard6_sft_base_resized.pt"
    print(f"[*] Kristal Shard 6 Temeli Yukleniyor: {base_path}", flush=True)
    sd = torch.load(base_path, map_location="cpu", weights_only=False)
    state = sd["model"] if "model" in sd else sd
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(device)

    # Hiperparametreler: 2 Epoch, micro-batch 4, grad-accum 4 (Effective batch: 16)
    micro_batch = 4
    grad_accum = 4
    loader = DataLoader(dataset, batch_size=micro_batch, shuffle=True, collate_fn=collate_fn)

    epochs = 2
    peak_lr = 6.0e-5
    min_lr = 6.0e-6
    total_steps = (len(loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * 0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    def get_lr(step):
        if step < warmup_steps:
            return peak_lr * (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))

    print(f"\n[*] Egitim Basliyor (Toplam {total_steps} Adim ~ 3-4 Dakika):", flush=True)
    global_step = 0
    start_time = time.time()
    accum_loss = 0.0

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
                logits, _ = model(inputs)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                raw_loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
                loss = raw_loss / grad_accum

            scaler.scale(loss).backward()
            accum_loss += raw_loss.item() / grad_accum

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                cur_lr = get_lr(global_step)
                for pg in optimizer.param_groups: pg['lr'] = cur_lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    elapsed = (time.time() - start_time) / 60.0
                    print(f"Adim {global_step:3d}/{total_steps} | Android Loss: {accum_loss:.4f} | LR: {cur_lr:.2e} | Sure: {elapsed:.1f}dk", flush=True)
                accum_loss = 0.0

    out_final = "models/dorina_v4/dorina_v4_android_agent.pt"
    torch.save({"model": model.state_dict(), "vocab_size": vocab_size}, out_final)
    print(f"\n[+] DORINA-ANDROID AGENT EGITIMI TAMAMLANDI!")
    print(f"Kaydedilen Model: {out_final}")

if __name__ == "__main__":
    train_android_sft()
