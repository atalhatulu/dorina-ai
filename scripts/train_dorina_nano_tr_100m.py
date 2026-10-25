import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from tokenizers import Tokenizer
from scripts.dorina_nano_tr_model import NanoTRConfig, DorinaNanoTR

# =====================================================================
# 12,288 VOCAB CONFIGURATION FOR DORINA-NANO-TR (10.40M)
# =====================================================================

@dataclass
class NanoTR12kConfig:
    vocab_size: int = 12288
    max_seq_len: int = 256
    d_model: int = 256
    n_layer: int = 10
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 32
    intermediate_size: int = 688  # SwiGLU
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = True

TRAIN_BIN = "data/nano_tr_data/dorina_nano_tr_12k_train.bin"
VAL_BIN = "data/nano_tr_data/dorina_nano_tr_12k_val.bin"
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_12k.json"
MODEL_DIR = "models/dorina_nano_tr"
os.makedirs(MODEL_DIR, exist_ok=True)

FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "dorina_nano_tr_10m.pt")

print("=" * 70)
print("DORINA-NANO-TR (10M) OFFICIAL 100M MULTI-TASK TRAINER")
print("Tasks: <fix> (Spelling, ASCII & Grammar) + <complete> (Autocomplete)")
print("Vocabulary: 12,288 BPE | Architecture: 10 Layers, 256 Dim, Tied")
print("Target: 100,000,000 Tokens (100% Unique, Zero Repeat, Single Epoch)")
print("=" * 70, flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = NanoTR12kConfig()
model = DorinaNanoTR(cfg).to(device)

total_params = sum(p.numel() for p in model.parameters())
brain_params = sum(p.numel() for n, p in model.named_parameters() if "tok_embeddings" not in n)
print(f"Total Params: {total_params / 1e6:.2f} M | Transformer Brain: {brain_params / 1e6:.2f} M ({brain_params/total_params*100:.1f}%)")

train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')
print(f"Loaded Train Data: {len(train_data):,} tokens ({os.path.getsize(TRAIN_BIN)/(1024*1024):.1f} MB)")
print(f"Loaded Val Data:   {len(val_data):,} tokens ({os.path.getsize(VAL_BIN)/(1024*1024):.1f} MB)", flush=True)

tok = Tokenizer.from_file(TOKENIZER_FILE)

# Training Hyperparameters
MICRO_BATCH = 32
GRAD_ACCUM = 2
SEQ_LEN = cfg.max_seq_len
BATCH_TOKENS = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN  # 16,384 tokens/step
STEP_TOKENS = MICRO_BATCH * GRAD_ACCUM * (SEQ_LEN + 1)

TOTAL_STEPS = len(train_data) // STEP_TOKENS
print(f"Total Single-Pass Training Steps: {TOTAL_STEPS:,} | Batch Size: {BATCH_TOKENS:,} tokens/step", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda")

WARMUP_STEPS = 200
def get_lr(step):
    if step < WARMUP_STEPS:
        return 1.8e-3 * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 1.8e-4 + 0.5 * (1.8e-3 - 1.8e-4) * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate_val_loss(model, val_data, max_batches=40):
    model.eval()
    losses = []
    step_tokens = MICRO_BATCH * (SEQ_LEN + 1)
    for i in range(min(len(val_data) // step_tokens, max_batches)):
        start = i * step_tokens
        chunk = val_data[start : start + step_tokens].reshape(MICRO_BATCH, SEQ_LEN + 1)
        batch = torch.from_numpy(chunk.astype(np.int64))
        x = batch[:, :-1].to(device, non_blocking=True)
        y = batch[:, 1:].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

# Checkpoint milestones: 25M, 50M, 75M, 100M
MILESTONES = {
    int(TOTAL_STEPS * 0.25): "25m",
    int(TOTAL_STEPS * 0.50): "50m",
    int(TOTAL_STEPS * 0.75): "75m",
    TOTAL_STEPS: "100m"
}

t0 = time.time()
total_tokens_processed = 0

print("\n>>> LAUNCHING ZERO-REPEAT 100M TRAINING ON RTX 4050...", flush=True)

for step in range(1, TOTAL_STEPS + 1):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    start = (step - 1) * STEP_TOKENS
    chunk = train_data[start : start + STEP_TOKENS].reshape(MICRO_BATCH * GRAD_ACCUM, SEQ_LEN + 1)
    batch = torch.from_numpy(chunk.astype(np.int64))

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for m in range(GRAD_ACCUM):
        b = batch[m * MICRO_BATCH : (m + 1) * MICRO_BATCH]
        x = b[:, :-1].to(device, non_blocking=True)
        y = b[:, 1:].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        accum_loss += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    total_tokens_processed += BATCH_TOKENS

    if step % 100 == 0:
        dt = time.time() - t0
        t0 = time.time()
        tok_s = (BATCH_TOKENS * 100) / max(dt, 1e-4)
        vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"Step {step:4d}/{TOTAL_STEPS} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {tok_s:,.0f} tok/s | Tokens: {total_tokens_processed/1e6:.1f}M / 100M | VRAM: {vram:.2f} GB", flush=True)

    if step in MILESTONES:
        tag = MILESTONES[step]
        val_l = evaluate_val_loss(model, val_data)
        ckpt_path = os.path.join(MODEL_DIR, f"dorina_nano_tr_10m_{tag}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"\n[MILESTONE {tag.upper()}] Checkpoint saved: {ckpt_path} | Val Loss: {val_l:.4f}\n", flush=True)

# Save Final Master Model
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print("=" * 70)
print(f"[SUCCESS] DORINA-NANO-TR (10M) 100M TRAINING COMPLETED!")
print(f"Final Model Saved: {FINAL_MODEL_PATH}")
print("=" * 70, flush=True)
