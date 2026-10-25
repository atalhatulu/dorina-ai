import os
import io
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tokenizers import Tokenizer
from datasets import load_dataset

# =====================================================================
# 1. ARCHITECTURES: 8k Tied (11 layers) vs 32k Untied (6 layers)
# =====================================================================

@dataclass
class Nano8kConfig:
    vocab_size: int = 8192
    max_seq_len: int = 256
    d_model: int = 256
    n_layer: int = 11
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 32
    intermediate_size: int = 688  # SwiGLU
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = True

@dataclass
class Nano32kConfig:
    vocab_size: int = 32768
    max_seq_len: int = 256
    d_model: int = 216
    n_layer: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 36
    intermediate_size: int = 576  # SwiGLU
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = False

def build_rope(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    cos = cos[:T, :].unsqueeze(0).unsqueeze(0).to(x.device, dtype=x.dtype)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(0).to(x.device, dtype=x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)

class SwiGLUMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLUMLP(cfg)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class GenericNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        if not cfg.tied_embeddings:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.head = None
        cos, sin = build_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        x = self.tok_embeddings(input_ids)
        cos, sin = self.rope_cos, self.rope_sin
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm_f(x)
        if self.head is not None:
            logits = self.head(x)
        else:
            logits = F.linear(x, self.tok_embeddings.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

# =====================================================================
# 2. EVALUATION METRICS: BITS-PER-BYTE & TOP-3 ACCURACY
# =====================================================================

@torch.no_grad()
def evaluate_metrics(model, heldout_text, tokenizer, device, max_batches=20):
    model.eval()
    total_log_likelihood = 0.0
    total_bytes = 0
    total_tokens = 0
    top1_correct = 0
    top3_correct = 0

    tokens = tokenizer.encode(heldout_text).ids
    seq_len = model.cfg.max_seq_len
    num_samples = min(len(tokens) // (seq_len + 1), max_batches)

    raw_bytes = heldout_text.encode("utf-8")
    bytes_per_token = len(raw_bytes) / max(len(tokens), 1)

    for i in range(num_samples):
        chunk = tokens[i * (seq_len + 1) : (i + 1) * (seq_len + 1)]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

        with torch.autocast("cuda", dtype=torch.float16):
            logits, loss = model(x, y)

        # Cross Entropy Loss in nats
        ce_loss = loss.item()
        # Log-likelihood in base 2 (bits)
        bits = ce_loss / math.log(2.0)
        total_log_likelihood += bits * seq_len
        total_tokens += seq_len

        # Top-1 and Top-3 accuracy
        # logits: [1, seq_len, vocab_size]
        _, top3_preds = torch.topk(logits, k=3, dim=-1)
        y_expanded = y.unsqueeze(-1)
        top1_correct += (top3_preds[:, :, :1] == y_expanded).sum().item()
        top3_correct += (top3_preds == y_expanded).any(dim=-1).sum().item()

    model.train()
    # Bits-Per-Byte (BPB) = (total bits) / (total UTF-8 bytes)
    sample_bytes = total_tokens * bytes_per_token
    bpb = total_log_likelihood / max(sample_bytes, 1e-4)
    top1_acc = (top1_correct / total_tokens) * 100.0
    top3_acc = (top3_correct / total_tokens) * 100.0

    return bpb, top1_acc, top3_acc

print("Architecture & Eval module verified successfully.")
