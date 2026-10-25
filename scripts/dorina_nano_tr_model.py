import os
import io
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.decoders import ByteLevel

# =====================================================================
# 1. DORINA-NANO-TR (10.08M) OFFICIAL ARCHITECTURE
# =====================================================================

@dataclass
class NanoTRConfig:
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

def build_rope(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

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
        return self.out(out.transpose(1, 2).contiguous().view(B, T, -1))

class SwiGLUMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class NanoBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLUMLP(cfg)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))

class DorinaNanoTR(nn.Module):
    def __init__(self, cfg):
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
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        x = self.tok_embeddings(input_ids)
        cos, sin = self.rope_cos, self.rope_sin
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm_f(x)
        # Tied embeddings: head shares weights with tok_embeddings
        logits = F.linear(x, self.tok_embeddings.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=40, temperature=0.7):
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
