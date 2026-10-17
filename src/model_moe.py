import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# 1. Rotary Position Embeddings (RoPE)
# -----------------------------------------------------------------------------
def precompute_rope_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    # xq, xk: [B, T, n_head, head_dim]
    B, T, H, D = xq.shape
    d_half = D // 2
    
    cos = freqs_cos[:T].view(1, T, 1, d_half)
    sin = freqs_sin[:T].view(1, T, 1, d_half)
    
    q1, q2 = xq[..., :d_half], xq[..., d_half:]
    k1, k2 = xk[..., :d_half], xk[..., d_half:]
    
    out_q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    out_k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return out_q, out_k

# -----------------------------------------------------------------------------
# 2. Self-Attention with RoPE and SDPA (FlashAttention Backend)
# -----------------------------------------------------------------------------
class CausalSelfAttentionRoPE(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

# -----------------------------------------------------------------------------
# 3. SwiGLU Expert Feed-Forward
# -----------------------------------------------------------------------------
class SwiGLUExpert(nn.Module):
    def __init__(self, n_embd, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)  # Down
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)  # Up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: (Swish(w1(x)) * w3(x)) -> w2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# -----------------------------------------------------------------------------
# 4. Sparse Mixture of Experts (MoE) Layer with Router & Load Balancing
# -----------------------------------------------------------------------------
class MoELayer(nn.Module):
    def __init__(self, n_embd, hidden_dim, num_experts=4, top_k=2, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(n_embd, hidden_dim, dropout) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C]
        
        router_logits = self.router(x_flat)  # [B*T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)  # [B*T, num_experts]
        
        # Top-k uzman seçimi
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        out = torch.zeros_like(x_flat)
        
        # Her bir uzman için seçilen tokenları işle
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                selected_x = x_flat[mask]
                expert_out = expert(selected_x)
                
                # İlgili uzmanın ağırlığını çek
                # mask indekslerindeki topk_indices içinde i'nin nerede olduğunu bul
                weight = torch.zeros(selected_x.shape[0], 1, device=x.device, dtype=x.dtype)
                for k_idx in range(self.top_k):
                    match = (topk_indices[mask, k_idx] == i)
                    weight[match] += topk_weights[mask, k_idx][match].unsqueeze(-1)
                
                out[mask] += expert_out * weight

        # Load balancing auxiliary loss (Aux Loss) & Router Z-Loss
        if self.training:
            density = router_probs.mean(dim=0)  # [num_experts]
            # Her uzmana düşen toplam ağırlık (importance)
            importance = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
            for k_idx in range(self.top_k):
                importance.scatter_add_(0, topk_indices[:, k_idx], topk_weights[:, k_idx])
            importance = importance / (B * T)
            aux_loss = self.num_experts * torch.sum(density * importance)
            
            # Tier-1 Standardı: Router Z-Loss (Sayısal Kararlılık & Logit Overflow Önleme)
            # ST-MoE / PaLM MoE: z_loss = mean(log(sum(exp(logits)))^2)
            log_z = torch.logsumexp(router_logits, dim=-1)
            z_loss = torch.mean(log_z ** 2)
        else:
            aux_loss = 0.0
            z_loss = 0.0

        return out.view(B, T, C), aux_loss, z_loss

# -----------------------------------------------------------------------------
# 5. Transformer MoE Block
# -----------------------------------------------------------------------------
class MoEBlock(nn.Module):
    def __init__(self, n_embd, n_head, hidden_dim, num_experts=4, top_k=2, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttentionRoPE(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.moe = MoELayer(n_embd, hidden_dim, num_experts, top_k, dropout)

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn(self.ln1(x), freqs_cos, freqs_sin)
        moe_out, aux_loss, z_loss = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss, z_loss

# -----------------------------------------------------------------------------
# 6. Dorina-MoE GPT Model Architecture
# -----------------------------------------------------------------------------
class DorinaMoEGPT(nn.Module):
    def __init__(self, vocab_size=32768, context_length=512, n_embd=384, n_head=6, n_layer=6, 
                 num_experts=4, top_k=2, hidden_dim=None, dropout=0.0):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        
        # SwiGLU için standart hidden_dim: 8/3 * n_embd (256'ya yuvarlanmış)
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * n_embd) / 3)
            hidden_dim = ((hidden_dim + 255) // 256) * 256  # 1024
        self.hidden_dim = hidden_dim

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            MoEBlock(n_embd, n_head, hidden_dim, num_experts, top_k, dropout)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight Tying

        # Precompute RoPE freqs
        head_dim = n_embd // n_head
        freqs_cos, freqs_sin = precompute_rope_freqs_cis(head_dim, context_length)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Tier-1 AI Lab Standardı:
        - 2D tensörler (Linear ağırlıkları) weight decay alır.
        - 1D tensörler (LayerNorm, Bias, Embedding) weight decay'den hariç tutulur!
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # lm_head.weight tok_emb.weight ile tie edilmiş
        decay.discard('lm_head.weight')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Çakışan parametreler: {inter_params}"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.context_length, f"Girdi uzunluğu ({T}) bağlam sınırını ({self.context_length}) aştı!"
        
        x = self.tok_emb(idx)
        x = self.drop(x)
        
        cos = self.freqs_cos[:T].to(x.device)
        sin = self.freqs_sin[:T].to(x.device)
        
        total_aux_loss = 0.0
        total_z_loss = 0.0
        for block in self.blocks:
            x, aux_loss, z_loss = block(x, cos, sin)
            total_aux_loss = total_aux_loss + aux_loss
            total_z_loss = total_z_loss + z_loss
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross Entropy Loss + Load Balancing Aux Loss + Router Z-Loss
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # aux_loss katsayısı: 0.01, z_loss katsayısı: 0.0001 (ST-MoE standardı)
            num_b = len(self.blocks)
            loss = ce_loss + 0.01 * (total_aux_loss / num_b) + 0.0001 * (total_z_loss / num_b)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.7, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
