import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from scripts.nano_ab_engine import (
    Nano8kConfig, Nano32kConfig, GenericNano, evaluate_metrics
)

DATA_RAW = "data/nano_v2_bench/sample_raw.txt"
with open(DATA_RAW, "r", encoding="utf-8") as f:
    full_text = f.read()

# Split into 90% train corpus, 10% heldout evaluation set
split_idx = int(len(full_text) * 0.90)
train_text = full_text[:split_idx]
heldout_text = full_text[split_idx:]

print("=" * 70)
print("DORINA-NANO ARCHITECTURE SHOWDOWN: 8k Tied (11L) vs 32k Untied (6L)")
print(f"Train Corpus: {len(train_text)/(1024*1024):.2f} MB | Held-Out Eval: {len(heldout_text)/(1024*1024):.2f} MB")
print("Target: Run 2,500 training steps (~10M tokens) on exact same text")
print("Measure: Actual tok/s, Bits-Per-Byte (BPB), Top-1 & Top-3 Autocomplete Accuracy")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 2000
MICRO_BATCH = 16
GRAD_ACCUM = 2
SEQ_LEN = 256
BATCH_TOKENS = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN  # 8,192 tokens per step

results = {}

# ----------------- EXPERIMENT 1: 8k Tied (11 Layers) -----------------
print("\n>>> [RUN 1/2] BENCHMARKING: 8k Vocab Tied Embeddings (11 Layers, 256 Dim)...")
tok_8k = Tokenizer.from_file("tokenizers/dorina_nano_8k.json")
train_ids_8k = np.array(tok_8k.encode(train_text).ids, dtype=np.uint16)
print(f"Total 8k Train Tokens: {len(train_ids_8k):,}")

cfg_8k = Nano8kConfig()
model_8k = GenericNano(cfg_8k).to(device)
params_8k = sum(p.numel() for p in model_8k.parameters())
body_8k = sum(p.numel() for n, p in model_8k.named_parameters() if "tok_embeddings" not in n)
print(f"Total Params: {params_8k/1e6:.2f}M | Brain Body: {body_8k/1e6:.2f}M ({body_8k/params_8k*100:.1f}%)")

opt_8k = torch.optim.AdamW(model_8k.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.1)
scaler_8k = torch.amp.GradScaler("cuda")

step_size = MICRO_BATCH * GRAD_ACCUM * (SEQ_LEN + 1)
t0 = time.time()
tok_count_8k = 0

for step in range(STEPS):
    start = (step * step_size) % (len(train_ids_8k) - step_size)
    chunk = train_ids_8k[start : start + step_size].reshape(MICRO_BATCH * GRAD_ACCUM, SEQ_LEN + 1)
    batch = torch.from_numpy(chunk.astype(np.int64))

    opt_8k.zero_grad(set_to_none=True)
    for m in range(GRAD_ACCUM):
        b = batch[m * MICRO_BATCH : (m + 1) * MICRO_BATCH]
        x = b[:, :-1].to(device, non_blocking=True)
        y = b[:, 1:].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model_8k(x, y)
            loss = loss / GRAD_ACCUM
        scaler_8k.scale(loss).backward()

    scaler_8k.unscale_(opt_8k)
    torch.nn.utils.clip_grad_norm_(model_8k.parameters(), 1.0)
    scaler_8k.step(opt_8k)
    scaler_8k.update()
    tok_count_8k += BATCH_TOKENS

dt_8k = time.time() - t0
tok_s_8k = tok_count_8k / dt_8k

bpb_8k, top1_8k, top3_8k = evaluate_metrics(model_8k, heldout_text, tok_8k, device)
print(f"[8k RESULT] Speed: {tok_s_8k:,.0f} tok/s | BPB: {bpb_8k:.4f} | Top-1 Acc: {top1_8k:.2f}% | Top-3 Acc: {top3_8k:.2f}%")
results["8k"] = {"tok_s": tok_s_8k, "bpb": bpb_8k, "top1": top1_8k, "top3": top3_8k, "body_pct": body_8k/params_8k*100}

# ----------------- EXPERIMENT 2: 32k Untied (6 Layers) -----------------
print("\n>>> [RUN 2/2] BENCHMARKING: 32k Vocab Untied (6 Layers, 216 Dim)...")
tok_32k = Tokenizer.from_file("tokenizers/dorina_tokenizer_v4_32k.json")
train_ids_32k = np.array(tok_32k.encode(train_text).ids, dtype=np.uint16)
print(f"Total 32k Train Tokens: {len(train_ids_32k):,}")

cfg_32k = Nano32kConfig()
model_32k = GenericNano(cfg_32k).to(device)
params_32k = sum(p.numel() for p in model_32k.parameters())
body_32k = sum(p.numel() for n, p in model_32k.named_parameters() if "tok_embeddings" not in n and "head" not in n)
print(f"Total Params: {params_32k/1e6:.2f}M | Brain Body: {body_32k/1e6:.2f}M ({body_32k/params_32k*100:.1f}%)")

opt_32k = torch.optim.AdamW(model_32k.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.1)
scaler_32k = torch.amp.GradScaler("cuda")

t0 = time.time()
tok_count_32k = 0

for step in range(STEPS):
    start = (step * step_size) % (len(train_ids_32k) - step_size)
    chunk = train_ids_32k[start : start + step_size].reshape(MICRO_BATCH * GRAD_ACCUM, SEQ_LEN + 1)
    batch = torch.from_numpy(chunk.astype(np.int64))

    opt_32k.zero_grad(set_to_none=True)
    for m in range(GRAD_ACCUM):
        b = batch[m * MICRO_BATCH : (m + 1) * MICRO_BATCH]
        x = b[:, :-1].to(device, non_blocking=True)
        y = b[:, 1:].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model_32k(x, y)
            loss = loss / GRAD_ACCUM
        scaler_32k.scale(loss).backward()

    scaler_32k.unscale_(opt_32k)
    torch.nn.utils.clip_grad_norm_(model_32k.parameters(), 1.0)
    scaler_32k.step(opt_32k)
    scaler_32k.update()
    tok_count_32k += BATCH_TOKENS

dt_32k = time.time() - t0
tok_s_32k = tok_count_32k / dt_32k

bpb_32k, top1_32k, top3_32k = evaluate_metrics(model_32k, heldout_text, tok_32k, device)
print(f"[32k RESULT] Speed: {tok_s_32k:,.0f} tok/s | BPB: {bpb_32k:.4f} | Top-1 Acc: {top1_32k:.2f}% | Top-3 Acc: {top3_32k:.2f}%")
results["32k"] = {"tok_s": tok_s_32k, "bpb": bpb_32k, "top1": top1_32k, "top3": top3_32k, "body_pct": body_32k/params_32k*100}

# ----------------- FINAL COMPARISON TABLE -----------------
print("\n" + "=" * 70)
print("FINAL BENCHMARK COMPARISON TABLE (Claude's Scientific Protocol)")
print("=" * 70)
print(f"{'Metric':<25} | {'8k Tied (11L)':<20} | {'32k Untied (6L)':<20}")
print("-" * 70)
print(f"{'Transformer Body %':<25} | {results['8k']['body_pct']:.1f}% {'(Daha zeki)':<10} | {results['32k']['body_pct']:.1f}%")
print(f"{'Speed (tok/s)':<25} | {results['8k']['tok_s']:,.0f} tok/s          | {results['32k']['tok_s']:,.0f} tok/s")
print(f"{'Bits-Per-Byte (BPB)':<25} | {results['8k']['bpb']:.4f} {'(Düşük iyi)':<10} | {results['32k']['bpb']:.4f}")
print(f"{'Top-1 Autocomplete Acc':<25} | {results['8k']['top1']:.2f}%               | {results['32k']['top1']:.2f}%")
print(f"{'Top-3 Autocomplete Acc':<25} | {results['8k']['top3']:.2f}% {'(Kazanan)':<10} | {results['32k']['top3']:.2f}%")
print("=" * 70)
