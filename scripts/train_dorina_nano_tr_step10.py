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
# DORINA-NANO-TR (10.4M) STEP 10: 900M -> 1.0B GRAND FINALE (COOL-DOWN)
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

PART3_BIN = "data/nano_tr_data/dorina_nano_tr_12k_train_part3_500m.bin"
VAL_BIN = "data/nano_tr_data/dorina_nano_tr_12k_val.bin"
SYNTHETIC_TXT = "data/nano_tr_data/synthetic_refinement_mix.txt"
TOKENIZER_FILE = "tokenizers/dorina_nano_tr_12k.json"
MODEL_DIR = "models/dorina_nano_tr"
LOGS_DIR = os.path.join(MODEL_DIR, "logs")

BASE_CKPT = os.path.join(MODEL_DIR, "dorina_nano_tr_10m_900m.pt")
SAVE_CKPT_1B = os.path.join(MODEL_DIR, "dorina_nano_tr_10m_1b_final.pt")
MASTER_FINAL = os.path.join(MODEL_DIR, "dorina_nano_tr_10m.pt")

print("=" * 70)
print("DORINA-NANO-TR (10.4M) STEP 10: 900M -> 1.0B GRAND FINALE")
print("Target: 1.0 Billion Total Tokens (1:100 Ultimate Over-Training)")
print("Cool-Down: LR 2.0e-4 -> Linear Decay to 1.0e-5 + 25% Synthetic Refinement")
print("=" * 70, flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = NanoTR12kConfig()
model = DorinaNanoTR(cfg).to(device)

if os.path.exists(BASE_CKPT):
    print(f">>> Loading base weights from 900M model: {BASE_CKPT}", flush=True)
    ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
else:
    print(f"HATA: 900M modeli bulunamadi: {BASE_CKPT}")
    sys.exit(1)

# Final 100M slice (400M -> 500M)
full_stream = np.memmap(PART3_BIN, dtype=np.uint16, mode='r')
START_OFFSET = 400_000_000
END_OFFSET = 500_000_000
train_raw_stream = full_stream[START_OFFSET:END_OFFSET]
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')

tok = Tokenizer.from_file(TOKENIZER_FILE)

# Tokenize Synthetic mix
with open(SYNTHETIC_TXT, "r", encoding="utf-8") as f:
    synth_lines = f.readlines()

synth_token_ids = []
for line in synth_lines:
    ids = tok.encode(line).ids
    synth_token_ids.extend(ids)
synth_token_arr = np.array(synth_token_ids, dtype=np.uint16)

MICRO_BATCH = 32
GRAD_ACCUM = 2
SEQ_LEN = cfg.max_seq_len
STEP_TOKENS = MICRO_BATCH * GRAD_ACCUM * (SEQ_LEN + 1)
TOTAL_STEPS = len(train_raw_stream) // STEP_TOKENS

# Cool-down schedule: Linear decay from 2.0e-4 down to 1e-5
LR_START = 2.0e-4
LR_END = 1.0e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda")

def get_lr(step):
    progress = step / max(1, TOTAL_STEPS)
    return LR_START - progress * (LR_START - LR_END)

REGRESSION_PROMPTS = [
    ("Ben ev gidiyrum", "Ben eve gidiyorum"),
    ("yarin gorusuruz insallah", "yarın görüşürüz inşallah"),
    ("bu dogrumu sence", "bu doğru mu sence"),
    ("bende gelicem", "ben de geleceğim"),
    ("turkiyenin baskenti ankaradir", "Türkiye'nin başkenti Ankara'dır."),
    ("geliyom bekleyin", "geliyorum bekleyin")
]

def evaluate_regression():
    model.eval()
    results = []
    with torch.no_grad():
        for corrupted, expected in REGRESSION_PROMPTS:
            prompt = f"<fix> {corrupted} =>"
            inp = torch.tensor([tok.encode(prompt).ids], device=device)
            gen = []
            for _ in range(25):
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                nid = torch.argmax(logits[0, -1]).item()
                if nid == tok.token_to_id("<eos>") or '\n' in tok.decode([nid]):
                    break
                gen.append(nid)
                inp = torch.cat([inp, torch.tensor([[nid]], device=device)], dim=1)
            pred = tok.decode(gen).strip()
            results.append((corrupted, pred))
    model.train()
    return results

t0 = time.time()
print("\n>>> TRAINING STARTED (900M -> 1.0B GRAND FINALE)...", flush=True)

synth_ptr = 0
model.train()

for step in range(TOTAL_STEPS):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro in range(GRAD_ACCUM):
        # 25% synthetic refinement for final stabilization
        is_synthetic = (np.random.rand() < 0.25)
        
        if is_synthetic and len(synth_token_arr) > MICRO_BATCH * (SEQ_LEN + 1):
            if synth_ptr + MICRO_BATCH * (SEQ_LEN + 1) >= len(synth_token_arr):
                synth_ptr = 0
            raw_chunk = synth_token_arr[synth_ptr : synth_ptr + MICRO_BATCH * (SEQ_LEN + 1)]
            synth_ptr += MICRO_BATCH * (SEQ_LEN + 1)
        else:
            batch_offset = (step * GRAD_ACCUM + micro) * MICRO_BATCH * (SEQ_LEN + 1)
            raw_chunk = train_raw_stream[batch_offset : batch_offset + MICRO_BATCH * (SEQ_LEN + 1)]

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
        total_seen = 900_000_000 + tok_seen
        eta_sec = (TOTAL_STEPS - step - 1) * (dt / max(1, step + 1))
        eta_min = eta_sec / 60
        print(f"Step {step:4d}/{TOTAL_STEPS:4d} | Loss: {loss_accum:.4f} | Total Seen: {total_seen/1e6:.1f}M | Speed: {speed:,.0f} tok/s | LR: {lr:.2e} | ETA: {eta_min:.1f} dk", flush=True)

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
        
        reg_tests = evaluate_regression()
        print(f"\n>>> [EVAL] Step {step+1} | Val Loss: {val_loss:.4f} | Total Tokens: {(900_000_000 + (step+1)*STEP_TOKENS)/1e6:.1f}M")
        print("    [REGRESYON KONTROLÜ]:")
        for inp_str, out_str in reg_tests:
            print(f"      * '{inp_str}' => '{out_str}'")
        print("", flush=True)
        model.train()

# Final Save
torch.save(model.state_dict(), SAVE_CKPT_1B)
torch.save(model.state_dict(), MASTER_FINAL)

total_time = (time.time() - t0) / 60
print("=" * 70)
print(f"🎉🎉 [ULTIMATE SUCCESS] 1.0 BILLION TOKENS COMPLETED in {total_time:.1f} minutes! 🎉🎉")
print(f"Saved Final Model: {SAVE_CKPT_1B}")
print(f"Updated Master:    {MASTER_FINAL}")
print("=" * 70, flush=True)
