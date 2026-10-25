import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DenseConfig:
    vocab_size: int = 32768
    max_seq_len: int = 2048
    n_layers: int = 16
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4       # 3:1 Grouped Query Attention (GQA)
    hidden_dim: int = 2560    # SwiGLU FFN (~8/3 * d_model, rounded to multiple of 256)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, seq_len, n_heads, head_dim]
    seq_len = x.shape[1]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(2).to(x.device, x.dtype)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(2).to(x.device, x.dtype)
    
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    
    # cos, sin interleave
    cos_interleaved = torch.repeat_interleave(cos, 2, dim=-1)
    sin_interleaved = torch.repeat_interleave(sin, 2, dim=-1)
    return (x * cos_interleaved) + (rotated_x * sin_interleaved)

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.num_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        # QK-Norm (stabilite için - loss spike ve fp16 overflow engeller)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # GQA Repeat KV
        if self.num_rep > 1:
            k = torch.repeat_interleave(k, self.num_rep, dim=2)
            v = torch.repeat_interleave(v, self.num_rep, dim=2)

        # [B, n_heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch Flash / SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)

class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class DorinaDenseV4(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE
        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        # Tied embedding sayımı
        unique_params = sum(p.numel() for p in set(self.parameters()))
        return {
            "total_params": total,
            "unique_params": unique_params,
            "total_m": round(total / 1e6, 2),
            "unique_m": round(unique_params / 1e6, 2)
        }

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S = input_ids.shape
        x = self.tok_emb(input_ids)
        
        cos = self.rope_cos[:S]
        sin = self.rope_sin[:S]

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten cross entropy
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

if __name__ == "__main__":
    cfg = DenseConfig()
    model = DorinaDenseV4(cfg)
    p_info = model.count_parameters()
    print("=" * 60)
    print(f"Dorina V4 Architecture Verified!")
    print(f"Total Parameters: {p_info['total_m']}M (Unique: {p_info['unique_m']}M)")
    print(f"Layers: {cfg.n_layers} | d_model: {cfg.d_model} | Heads: {cfg.n_heads} (GQA: {cfg.n_kv_heads})")
    print(f"Context Length: {cfg.max_seq_len} tokens | Vocab Size: {cfg.vocab_size}")
    print("=" * 60)
    
    # Test forward pass
    dummy_x = torch.randint(0, cfg.vocab_size, (2, 128))
    dummy_y = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss = model(dummy_x, dummy_y)
    print(f"Forward Pass OK: Logits shape={logits.shape}, Loss={loss.item():.4f}")
