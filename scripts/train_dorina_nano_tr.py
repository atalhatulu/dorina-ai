import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import numpy as np
import torch
from tokenizers import Tokenizer
from scripts.dorina_nano_tr_model import NanoTRConfig, DorinaNanoTR

TRAIN_BIN = "data/nano_tr_data/dorina_nano_tr_train.bin"
VAL_BIN = "data/nano_tr_data/dorina_nano_tr_val.bin"
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_8k.json"
MODEL_DIR = "models/dorina_nano_tr"
os.makedirs(MODEL_DIR, exist_ok=True)

FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "dorina_nano_tr_10m.pt")

print("=" * 65)
print("DORINA-NANO-TR (10M) TURKISH AUTOCOMPLETE & SPELLCHECK TRAINER")
print("Architecture: 8k Vocab Tied, 11 Layers, 256 Dim (10.08M Params)")
print("Hardware: Local NVIDIA RTX 4050 GPU")
print("=" * 65, flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = NanoTRConfig()
model = DorinaNanoTR(cfg).to(device)

total_params = sum(p.numel() for p in model.parameters())
brain_params = sum(p.numel() for n, p in model.named_parameters() if "tok_embeddings" not in n)
print(f"Total Parameters: {total_params / 1e6:.2f} M | Transformer Body: {brain_params / 1e6:.2f} M ({brain_params/total_params*100:.1f}%)")

train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')
print(f"Loaded Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens", flush=True)

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

# Training hyperparameters
# Context 256, micro_batch 32, grad_accum 2 -> 16,384 tokens per step
MICRO_BATCH = 32
GRAD_ACCUM = 2
SEQ_LEN = cfg.max_seq_len
BATCH_TOKENS = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN
STEP_TOKENS = MICRO_BATCH * GRAD_ACCUM * (SEQ_LEN + 1)

# Run 4 epochs over the 18M token corpus (~70M tokens total) to firmly saturate Turkish grammar rules
EPOCHS = 4
TOTAL_STEPS = (len(train_data) // STEP_TOKENS) * EPOCHS

optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda")

# Cosine LR schedule with warmup
WARMUP_STEPS = 150
def get_lr(step):
    if step < WARMUP_STEPS:
        return 1.8e-3 * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 1.8e-4 + 0.5 * (1.8e-3 - 1.8e-4) * (1.0 + math.cos(math.pi * progress))

print(f"Total Steps: {TOTAL_STEPS} ({EPOCHS} Epochs) | Batch: {BATCH_TOKENS:,} tokens/step", flush=True)

global_step = 0
total_tokens_processed = 0
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    print(f"\n>>> [EPOCH {epoch}/{EPOCHS}] Training Dorina-Nano-TR...", flush=True)
    steps_in_epoch = len(train_data) // STEP_TOKENS

    for step in range(steps_in_epoch):
        global_step += 1
        lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        start = step * STEP_TOKENS
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

        if global_step % 100 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_s = (BATCH_TOKENS * 100) / max(dt, 1e-4)
            vram = torch.cuda.memory_allocated() / (1024**3)
            print(f"Step {global_step:4d}/{TOTAL_STEPS} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {tok_s:,.0f} tok/s | Total Tok: {total_tokens_processed/1e6:.1f}M | VRAM: {vram:.2f} GB", flush=True)

    # Save epoch checkpoint
    epoch_ckpt = os.path.join(MODEL_DIR, f"dorina_nano_tr_10m_epoch{epoch}.pt")
    torch.save(model.state_dict(), epoch_ckpt)
    print(f"[CHECKPOINT] Saved: {epoch_ckpt}", flush=True)

# Save Final Model
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print("\n" + "=" * 65)
print(f"[SUCCESS] DORINA-NANO-TR (10M) COMPLETED!")
print(f"Final Weights: {FINAL_MODEL_PATH}")
print("=" * 65, flush=True)
