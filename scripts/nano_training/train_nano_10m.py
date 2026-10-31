#!/usr/bin/env python3
"""
Dorina-Nano 10M Toy Experiment:
- Architecture: 10.07M Dense Transformer (6 layers, d_model=216, 6 heads, GQA 3:1, SwiGLU=576)
- Trains on local RTX 4050 Laptop GPU
- Generates live sample text every 200 steps to watch intelligence emerge from scratch!
"""
import os
import sys
import time
import math
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
    def generate(self, prompt_tokens, max_new_tokens=30, temperature=0.8):
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
    print("=" * 65)
    print("DORINA-NANO (10M) TOY EXPERIMENT STARTED")
    print(f"Device: {device} | Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Layers: {cfg.n_layer} | d_model: {cfg.d_model} | Heads: {cfg.n_heads}/{cfg.n_kv_heads} | Context: {cfg.max_seq_len}")
    print("=" * 65)

    data_path = "data/v4_pilot/v4_pilot_train.bin"
    tokenizer_path = "tokenizers/dorina_tokenizer_v4_32k.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    seq_len = cfg.max_seq_len
    total_seqs = len(data) // (seq_len + 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler()

    micro_batch = 8
    grad_accum = 2
    max_steps = 1500

    prompts = [
        "The sun is",
        "Python is a",
        "Dorina is an AI"
    ]

    t0 = time.time()
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for m in range(grad_accum):
            idx = ((step * grad_accum + m) * micro_batch) % max(1, total_seqs - micro_batch)
            batch_x, batch_y = [], []
            for b in range(micro_batch):
                start = (idx + b) * (seq_len + 1)
                chunk = torch.from_numpy(data[start : start + seq_len + 1].astype(np.int64))
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])
            x = torch.stack(batch_x).to(device)
            y = torch.stack(batch_y).to(device)

            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_s = (micro_batch * grad_accum * seq_len * 50) / max(dt, 1e-4)
            vram = torch.cuda.memory_allocated() / (1024**3)
            print(f"Step {step:4d}/{max_steps} | Loss: {accum_loss:.4f} | Speed: {tok_s:,.0f} tok/s | VRAM: {vram:.2f} GB")

        # Live Text Generation every 250 steps!
        if step > 0 and step % 250 == 0:
            print("\n" + "-" * 50)
            print(f"[CANLI TEST - ADIM {step}] Model Kelime Üretimi:")
            for p in prompts:
                p_ids = torch.tensor([tokenizer.encode(p).ids], device=device)
                gen_ids = model.generate(p_ids, max_new_tokens=25, temperature=0.7)
                gen_text = tokenizer.decode(gen_ids[0].tolist())
                print(f"  Prompt: '{p}' -> {gen_text}")
            print("-" * 50 + "\n")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dorina_nano_10m.pt")
    print("\nTraining completed! Dorina-Nano 10M saved to models/dorina_nano_10m.pt")

if __name__ == "__main__":
    main()
