#!/usr/bin/env python3
"""
Dorina-Nano (10M) Quiet Overnight Mode:
- Target: ~25W-30W, whisper-quiet fans, cool GPU temp (<55C)
- micro_batch = 4, grad_accum = 4 (same mathematical batch, lower instant VRAM/power)
- Tiny sleep (8ms) between steps to prevent continuous 100% thermal stress
"""
import os
import sys
import time
import math
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tokenizers import Tokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

@dataclass
class NanoConfig:
    vocab_size: int = 32768
    n_layer: int = 6
    d_model: int = 216
    n_heads: int = 6
    n_kv_heads: int = 2
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    hidden_dim: int = 576
    norm_eps: float = 1e-6
    init_std: float = 0.02

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)

def build_rope(head_dim, max_seq_len, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq_len).float(), inv)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[None, None].float(), sin[None, None].float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w_in = nn.Linear(d, 2 * h, bias=False)
        self.w_out = nn.Linear(h, d, bias=False)

    def forward(self, x):
        g, u = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(g) * u)

class GQAttention(nn.Module):
    def __init__(self, cfg: NanoConfig):
        super().__init__()
        d, hd = cfg.d_model, cfg.head_dim
        self.q_proj = nn.Linear(d, cfg.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * hd, d, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = hd

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)

class NanoBlock(nn.Module):
    def __init__(self, cfg: NanoConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.hidden_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

class DorinaNano(nn.Module):
    def __init__(self, cfg: NanoConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([NanoBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = build_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.tok_embeddings(input_ids)
        cos, sin = self.rope_cos, self.rope_sin
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm_f(x)
        logits = F.linear(x, self.tok_embeddings.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=35, temperature=0.7):
        self.eval()
        idx = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        self.train()
        return idx

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NanoConfig()
    model = DorinaNano(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 65, flush=True)
    print("DORINA-NANO (10M) QUIET & COOL 25W OVERNIGHT TRAINER", flush=True)
    print(f"Device: {device} | Total Parameters: {total_params / 1e6:.2f} M", flush=True)
    print("Target Power: ~25W-30W (Whisper-Quiet Fan Mode)", flush=True)
    print("=" * 65, flush=True)

    tokenizer = Tokenizer.from_file("tokenizers/dorina_tokenizer_v4_32k.json")
    os.makedirs("models", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    # Quiet mode: micro_batch=4, grad_accum=4 -> batch 8,192 tokens
    micro_batch = 4
    grad_accum = 4
    seq_len = cfg.max_seq_len
    tokens_per_step = micro_batch * grad_accum * seq_len
    step_tokens = micro_batch * grad_accum * (seq_len + 1)

    bin_files = sorted(glob.glob("data/nano_pack/*.bin"))
    total_tokens_processed = 0
    global_step = 0
    t0 = time.time()

    for bf_idx, bin_path in enumerate(bin_files, 1):
        print(f"\n>>> [QUIET] Processing Part {bf_idx}/{len(bin_files)}: {bin_path}...", flush=True)
        data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        total_tokens = len(data)
        steps_in_file = total_tokens // step_tokens

        for step in range(steps_in_file):
            global_step += 1
            start = step * step_tokens
            raw_block = data[start : start + step_tokens].reshape(micro_batch * grad_accum, seq_len + 1)
            batch = torch.from_numpy(raw_block.astype(np.int64))

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for m in range(grad_accum):
                chunk = batch[m * micro_batch : (m + 1) * micro_batch]
                x = chunk[:, :-1].to(device, non_blocking=True)
                y = chunk[:, 1:].to(device, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_tokens_processed += tokens_per_step

            # Micro sleep to keep power low and fans whisper quiet
            time.sleep(0.12)

            if step % 200 == 0:
                dt = time.time() - t0
                t0 = time.time()
                tok_s = (tokens_per_step * 200) / max(dt, 1e-4) if step > 0 else tokens_per_step / max(dt, 1e-4)
                vram = torch.cuda.memory_allocated() / (1024**3)
                print(f"[Part {bf_idx}] Step {step:4d}/{steps_in_file} | Loss: {accum_loss:.4f} | Speed: {tok_s:,.0f} tok/s | Total: {total_tokens_processed/1e6:.1f}M Tok | VRAM: {vram:.2f} GB", flush=True)

    ckpt_200m = "models/dorina_nano_200m.pt"
    torch.save(model.state_dict(), ckpt_200m)
    print(f"\n[SUCCESS] Quiet Overnight Run Complete! Saved to {ckpt_200m}", flush=True)

if __name__ == "__main__":
    main()
