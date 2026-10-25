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
# DORINA-NANO-TR (10.4M) STEPPING TRAINER: 300M -> 400M TOKEN RUN
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

PART2_BIN = "data/nano_tr_data/dorina_nano_tr_12k_train_part2.bin"
VAL_BIN = "data/nano_tr_data/dorina_nano_tr_12k_val.bin"
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_12k.json"
MODEL_DIR = "models/dorina_nano_tr"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_CKPT = os.path.join(MODEL_DIR, "dorina_nano_tr_10m_300m.pt")
SAVE_CKPT_400M = os.path.join(MODEL_DIR, "dorina_nano_tr_10m_400m.pt")
MASTER_FINAL = os.path.join(MODEL_DIR, "dorina_nano_tr_10m.pt")

print("=" * 70)
print("DORINA-NANO-TR (10.4M) LOCAL STEP 4: 300M -> 400M MILESTONE")
print("Target: 100,000,000 Fresh Unique Tokens (300M to 400M Slice)")
print("Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (Local)")
print("=" * 70, flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = NanoTR12kConfig()
model = DorinaNanoTR(cfg).to(device)

if os.path.exists(BASE_CKPT):
    print(f">>> Loading base weights from 300M model: {BASE_CKPT}", flush=True)
    ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
else:
    print(f"HATA: 300M modeli bulunamadı: {BASE_CKPT}")
    sys.exit(1)

total_params = sum(p.numel() for p in model.parameters())
brain_params = sum(p.numel() for n, p in model.named_parameters() if "tok_embeddings" not in n)
print(f"Total Params: {total_params / 1e6:.2f} M | Brain Body: {brain_params / 1e6:.2f} M ({brain_params/total_params*100:.1f}%)")

# Open memmap and take the THIRD 100M tokens (tokens 200,000,000 to 300,000,000 in part2)
full_stream = np.memmap(PART2_BIN, dtype=np.uint16, mode='r')
START_OFFSET = 200_000_000
END_OFFSET = 300_000_000
train_data = full_stream[START_OFFSET:END_OFFSET]
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')

print(f"Loaded Fresh Slice: {len(train_data):,} tokens (Exact 100M fresh stream, 300M-400M window)")
print(f"Loaded Val Data:    {len(val_data):,} tokens", flush=True)

tok = Tokenizer.from_file(TOKENIZER_FILE)

MICRO_BATCH = 32
GRAD_ACCUM = 2
SEQ_LEN = cfg.max_seq_len
STEP_TOKENS = MICRO_BATCH * GRAD_ACCUM * (SEQ_LEN + 1)
TOTAL_STEPS = len(train_data) // STEP_TOKENS

print(f"Total Steps: {TOTAL_STEPS:,} | Step Tokens: {STEP_TOKENS:,} tokens/step", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda")

def get_lr(step):
    progress = step / max(1, TOTAL_STEPS)
    return 8.0e-4 * 0.5 * (1.0 + math.cos(math.pi * progress))

t0 = time.time()
print("\n>>> TRAINING STARTED (300M -> 400M)...", flush=True)

model.train()
for step in range(TOTAL_STEPS):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro in range(GRAD_ACCUM):
        batch_offset = (step * GRAD_ACCUM + micro) * MICRO_BATCH * (SEQ_LEN + 1)
        raw_chunk = train_data[batch_offset : batch_offset + MICRO_BATCH * (SEQ_LEN + 1)]
        
        chunk = torch.from_numpy(raw_chunk.astype(np.int64)).view(MICRO_BATCH, SEQ_LEN + 1).to(device)
        x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, loss = model(x, y)
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        loss_accum += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if step % 50 == 0 or step == TOTAL_STEPS - 1:
        dt = time.time() - t0
        tok_seen = (step + 1) * STEP_TOKENS
        speed = tok_seen / max(1e-5, dt)
        total_seen = 300_000_000 + tok_seen
        eta_sec = (TOTAL_STEPS - step - 1) * (dt / max(1, step + 1))
        eta_min = eta_sec / 60
        print(f"Step {step:4d}/{TOTAL_STEPS:4d} | Loss: {loss_accum:.4f} | Total Seen: {total_seen/1e6:.1f}M | Speed: {speed:,.0f} tok/s | ETA: {eta_min:.1f} dk", flush=True)

    if (step + 1) % 1500 == 0 or step == TOTAL_STEPS - 1:
        model.eval()
        val_loss = 0.0
        val_steps = 30
        with torch.no_grad():
            for v_step in range(val_steps):
                v_off = v_step * MICRO_BATCH * (SEQ_LEN + 1)
                v_raw = val_data[v_off : v_off + MICRO_BATCH * (SEQ_LEN + 1)]
                v_chunk = torch.from_numpy(v_raw.astype(np.int64)).view(MICRO_BATCH, SEQ_LEN + 1).to(device)
                vx, vy = v_chunk[:, :-1].contiguous(), v_chunk[:, 1:].contiguous()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _, v_l = model(vx, vy)
                val_loss += v_l.item()
        val_loss /= val_steps
        model.train()
        print(f"\n>>> [EVAL] Step {step+1} | Val Loss: {val_loss:.4f} | Total Tokens: {(300_000_000 + (step+1)*STEP_TOKENS)/1e6:.1f}M\n", flush=True)

# Final Save
torch.save(model.state_dict(), SAVE_CKPT_400M)
torch.save(model.state_dict(), MASTER_FINAL)

total_time = (time.time() - t0) / 60
print("=" * 70)
print(f"[SUCCESS] 400M MILESTONE REACHED in {total_time:.1f} minutes!")
print(f"Saved: {SAVE_CKPT_400M}")
print("=" * 70, flush=True)
