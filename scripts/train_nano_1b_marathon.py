#!/usr/bin/env python3
"""
Dorina-Nano (10M) Multi-Stage 1-Billion Token Marathon
High Performance Chunked Loader (Zero Lag, Direct RAM Streaming)
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
    print("DORINA-NANO (10M) MULTI-STAGE 1-BILLION TOKEN MARATHON", flush=True)
    print(f"Device: {device} | Total Parameters: {total_params / 1e6:.2f} M", flush=True)
    print("Stages: 5 Stages x 200M Tokens = 1,000,000,000 Tokens Total", flush=True)
    print("=" * 65, flush=True)

    tokenizer = Tokenizer.from_file("tokenizers/dorina_tokenizer_v4_32k.json")
    tinystories_path = "data/TinyStoriesV2-GPT4-train.txt"
    os.makedirs("models", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    micro_batch = 16
    grad_accum = 2
    seq_len = cfg.max_seq_len
    tokens_per_step = micro_batch * grad_accum * seq_len # 16,384 tokens

    prompts = [
        "Once upon a time",
        "The sun is",
        "Python is a",
        "What is 2 + 5?",
        "Dorina is an AI"
    ]

    steps_per_stage = 200_000_000 // tokens_per_step # ~12,207 steps

    file_size = os.path.getsize(tinystories_path)
    stage_file_chunk = file_size // 5 # ~425MB text per stage

    global_step = 0
    t_marathon_start = time.time()

    for stage_idx in range(1, 6):
        stage_token_start = (stage_idx - 1) * 200
        stage_token_end = stage_idx * 200
        ckpt_name = f"models/dorina_nano_{stage_token_end}m.pt" if stage_idx < 5 else "models/dorina_nano_1b_final.pt"

        print(f"\n{'='*65}", flush=True)
        print(f"STAGE {stage_idx}/5: {stage_token_start}M -> {stage_token_end}M TOKENS", flush=True)
        print(f"Target Checkpoint: {ckpt_name}", flush=True)
        print(f"{'='*65}", flush=True)

        # Read this stage's dedicated ~425MB text chunk into memory
        start_byte = (stage_idx - 1) * stage_file_chunk
        print(f"Loading stage {stage_idx} text stream from disk ({start_byte/(1024*1024):.1f}MB to {(start_byte+stage_file_chunk)/(1024*1024):.1f}MB)...", flush=True)
        
        with open(tinystories_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(start_byte)
            stage_text = f.read(stage_file_chunk)

        stories = stage_text.split("\n\n")
        del stage_text
        print(f"Found {len(stories):,} stories in Stage {stage_idx}. Streaming tokens...", flush=True)

        story_idx = 0
        token_buffer = []
        t0 = time.time()

        for step in range(steps_per_stage):
            global_step += 1

            # Ensure buffer has enough tokens for this step
            needed = tokens_per_step + (micro_batch * grad_accum) + 100
            while len(token_buffer) < needed and story_idx < len(stories):
                batch_stories = [s.strip() for s in stories[story_idx : story_idx + 250] if len(s.strip()) > 10]
                story_idx += 250
                if batch_stories:
                    encs = tokenizer.encode_batch(batch_stories)
                    for e in encs:
                        token_buffer.extend(e.ids)
                        token_buffer.append(0)

            if len(token_buffer) < needed:
                print(f"Stage {stage_idx} exhausted early at step {step}, wrapping up checkpoint.", flush=True)
                break

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for m in range(grad_accum):
                batch_x, batch_y = [], []
                for b in range(micro_batch):
                    chunk = torch.tensor(token_buffer[:seq_len + 1], dtype=torch.long)
                    token_buffer = token_buffer[seq_len + 1:]
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

            if step % 200 == 0:
                dt = time.time() - t0
                t0 = time.time()
                tok_s = (tokens_per_step * 200) / max(dt, 1e-4) if step > 0 else tokens_per_step / max(dt, 1e-4)
                vram = torch.cuda.memory_allocated() / (1024**3)
                total_tokens_so_far = global_step * tokens_per_step
                print(f"[Stage {stage_idx}] Step {step:5d}/{steps_per_stage} | Loss: {accum_loss:.4f} | Speed: {tok_s:,.0f} tok/s | Total: {total_tokens_so_far/1e6:.1f}M Tok | VRAM: {vram:.2f} GB", flush=True)

            if step > 0 and step % 1000 == 0:
                print("\n" + "-" * 55, flush=True)
                print(f"[CANLI TEST - STAGE {stage_idx} | ADIM {step}] Model Akıl Durumu:", flush=True)
                for p in prompts:
                    p_ids = torch.tensor([tokenizer.encode(p).ids], device=device)
                    gen_ids = model.generate(p_ids, max_new_tokens=30, temperature=0.7)
                    gen_text = tokenizer.decode(gen_ids[0].tolist())
                    print(f"  Prompt: '{p}' -> {gen_text}", flush=True)
                print("-" * 55 + "\n", flush=True)

        torch.save(model.state_dict(), ckpt_name)
        print(f"\n>>> [STAGE {stage_idx} COMPLETE!] Checkpoint saved: {ckpt_name}\n", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("ALL 5 STAGES FINISHED! 1 BILLION TOKENS MASTER 10M MODEL READY!", flush=True)
    print(f"Total Marathon Duration: {(time.time() - t_marathon_start) / 3600:.2f} hours", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
