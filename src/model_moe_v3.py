#!/usr/bin/env python3
"""
Dorina V3: ince taneli (fine-grained) MoE dil modeli. Tek dosya, saf PyTorch.

Mimari
  * Pre-norm (RMSNorm), RoPE, GQA (n_kv_heads < n_heads), SwiGLU, weight tying, bias yok
  * Her katmanda: 32 yönlendirilmiş uzman (Top-2) + 1 paylaşımlı (always-on) uzman
  * Uzmanlar [E, d, hidden] 3B tensörlerinde; torch.bmm ile (for-döngüsü yok), statik şekiller
  * Router fp32; Z-loss + ST-MoE/Switch yük dengeleme kaybı
  * Parçalı cross-entropy (2048 token, parça başına checkpoint(use_reentrant=False))

Kullanım
  model = DorinaMoEV3(MoEConfig()).cuda()
  opt = model.configure_optimizers(weight_decay=0.1, lr=6e-4)
  with torch.autocast("cuda", dtype=torch.bfloat16):
      _, loss = model(x, y)          # y'de -100 = maskeli; loss = ce + aux + z
  loss.backward()
  model.moe_metrics()                # uzman yükü, düşme oranı, router entropisi (senkron yapar)

Hızlı kontrol
  python model_moe_v3.py                      # parametre raporu + CPU testleri + GPU 50 adım benchmark
  python model_moe_v3.py --sweep 2,4,8        # mikro-batch başına tepe VRAM / tok/s

Eğitim tarifi: fp32 master ağırlık + torch.autocast(bfloat16). Saf bf16 ağırlık (model.to(bf16))
RoPE tablolarının hassasiyetini düşürür, önerilmez.
"""
import argparse
import inspect
import math
import sys
import time
from dataclasses import dataclass, replace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ───────────────────────────── Konfigürasyon ─────────────────────────────
@dataclass
class MoEConfig:
    vocab_size: int = 32768
    n_layer: int = 12
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    n_experts: int = 32
    top_k: int = 2
    expert_hidden: int = 256
    shared_hidden: int = 256
    capacity_factor: Optional[float] = 1.25   # eğitimde; eval/üretimde dropless. None = eğitimde de dropless (senkron)
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 1e-3
    router_init_std: float = 0.01
    init_std: float = 0.02
    router_weight_decay: bool = False         # router'a weight decay verme (bilinçli karar)
    ce_chunk_size: int = 2048
    activation_checkpointing: bool = False    # katman başına checkpoint (VRAM az gelirse)
    norm_eps: float = 1e-6

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model n_heads'e bölünmeli"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads n_kv_heads'e bölünmeli"
        assert (self.d_model // self.n_heads) % 2 == 0, "head_dim çift olmalı (RoPE)"
        assert 1 <= self.top_k <= self.n_experts

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


# ───────────────────────────── Yardımcı katmanlar ─────────────────────────────
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
    freqs = torch.outer(torch.arange(max_seq_len).float(), inv)       # [T, hd/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, hd]; cos/sin: [T, hd/2] (rotate-half düzeni, fp32 hesap)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[None, None].float(), sin[None, None].float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w_in = nn.Linear(d, 2 * h, bias=False)    # gate | up birleşik
        self.w_out = nn.Linear(h, d, bias=False)

    def forward(self, x):
        g, u = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(g) * u)


class GQAttention(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        d, hd = cfg.d_model, cfg.head_dim
        self.nh, self.nkv, self.hd = cfg.n_heads, cfg.n_kv_heads, hd
        self.q_proj = nn.Linear(d, self.nh * hd, bias=False)
        self.k_proj = nn.Linear(d, self.nkv * hd, bias=False)
        self.v_proj = nn.Linear(d, self.nkv * hd, bias=False)
        self.o_proj = nn.Linear(self.nh * hd, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.nkv != self.nh:                                   # sorgu kafası i -> kv kafası i // grup
            g = self.nh // self.nkv
            k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.nh * self.hd))


# ───────────────────────────── MoE katmanı ─────────────────────────────
class MoEFFN(nn.Module):
    """32 yönlendirilmiş uzman (Top-k) + 1 paylaşımlı uzman. Sıralama tabanlı, statik şekilli dağıtım."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.E, self.k = cfg.n_experts, cfg.top_k
        self.capacity_factor = cfg.capacity_factor      # çalışma anında değiştirilebilir (örn. ilk 2K adım 2.0)
        d, h = cfg.d_model, cfg.expert_hidden
        self.router = nn.Linear(d, self.E, bias=False)
        self.w_in = nn.Parameter(torch.empty(self.E, d, 2 * h))    # [E, d, 2h] gate|up
        self.w_out = nn.Parameter(torch.empty(self.E, h, d))       # [E, h, d]
        self.shared = SwiGLU(d, cfg.shared_hidden)

    def forward(self, x):
        B, T, d = x.shape
        N, E, k = B * T, self.E, self.k
        dev = x.device
        xf = x.reshape(N, d)

        # 1) router: fp32
        with torch.autocast(device_type=dev.type, enabled=False):
            logits = F.linear(xf.float(), self.router.weight.float())          # [N, E]
        probs = logits.softmax(dim=-1)
        topv, topi = probs.topk(k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)                           # seçilenler üzerinde yeniden normalize

        # 2) uzmana göre sırala, uzman içi sıra (pos) bul
        flat_e = topi.reshape(-1)                                              # [N*k]
        tok = torch.arange(N, device=dev).repeat_interleave(k)
        order = flat_e.argsort(stable=True)
        e_s, t_s, w_s = flat_e[order], tok[order], topv.reshape(-1)[order]
        counts = torch.zeros(E, device=dev, dtype=torch.long).scatter_add_(0, flat_e, torch.ones_like(flat_e))
        pos = torch.arange(N * k, device=dev) - (counts.cumsum(0) - counts)[e_s]

        # 3) kapasite: eğitimde statik (senkron yok), eval/üretimde dropless
        if self.training and self.capacity_factor is not None:
            C = max(1, math.ceil(self.capacity_factor * N * k / E))
        else:
            C = max(1, int(counts.max().item()))
        keep = pos < C
        slot = torch.where(keep, e_s * C + pos, torch.full_like(pos, E * C))   # düşenler kukla satıra gider
        buf = xf.new_zeros(E * C + 1, d)
        buf[slot] = xf[t_s]

        # 4) uzmanlar: iki bmm
        h_in = buf[:-1].view(E, C, d).to(self.w_in.dtype)
        g, u = torch.bmm(h_in, self.w_in).chunk(2, dim=-1)
        y = torch.bmm(F.silu(g) * u, self.w_out).reshape(E * C, d)
        y = torch.cat([y, y.new_zeros(1, d)], dim=0)

        # 5) birleştir (fp32 toplama) + paylaşımlı uzman
        out = torch.zeros(N, d, device=dev, dtype=torch.float32)
        out.index_add_(0, t_s, y[slot].float() * (w_s * keep).unsqueeze(-1))
        out = out + self.shared(xf).float()

        # 6) yardımcı kayıplar ve izleme
        f = counts.float() / (N * k)                                           # atama payı (türevsiz)
        aux = E * (f * probs.mean(dim=0)).sum()                                # Switch / ST-MoE yük dengesi
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()                      # router z-loss
        with torch.no_grad():
            dropped = 1.0 - keep.float().mean()
            ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
            stats = torch.cat([counts.float(), torch.stack([dropped, ent, logits.max()])])   # [E + 3]
        return out.view(B, T, d).to(x.dtype), aux, z, stats


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = MoEFFN(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        m, aux, z, stats = self.moe(self.norm2(x))
        return x + m, aux, z, stats


# ───────────────────────────── Parçalı cross-entropy ─────────────────────────────
def chunked_cross_entropy(hidden, targets, lm_weight, chunk_size=2048, ignore_index=-100, normalizer=None):
    """hidden [N, d], targets [N]. Logits yalnızca chunk_size'lık dilimlerde üretilir ve geri yayılımda
    yeniden hesaplanır (checkpoint, use_reentrant=False). Toplam kayıp GEÇERLİ token sayısına bölünür
    (parça başına ortalama değil). normalizer verilirse (örn. gradient accumulation'daki toplam geçerli token)
    onunla bölünür. Geçerli token yoksa 0 döner (NaN yok)."""
    def _chunk_sum(h, t):
        logits = F.linear(h, lm_weight).float()
        return F.cross_entropy(logits, t, ignore_index=ignore_index, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    use_ckpt = torch.is_grad_enabled() and hidden.requires_grad
    for i in range(0, hidden.size(0), chunk_size):
        h, t = hidden[i:i + chunk_size], targets[i:i + chunk_size]
        total = total + (checkpoint(_chunk_sum, h, t, use_reentrant=False) if use_ckpt else _chunk_sum(h, t))
    denom = normalizer if normalizer is not None else (targets != ignore_index).sum().clamp(min=1)
    return total / denom


# ───────────────────────────── Model ─────────────────────────────
class DorinaMoEV3(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)          # lm_head ile paylaşılır (tying)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = build_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.last_stats: Dict[str, torch.Tensor] = {}
        self._init_weights()

    # --- init ---
    def _init_weights(self):
        c = self.cfg
        res_std = c.init_std / math.sqrt(2 * c.n_layer)                   # artık çıkışlar: GPT-2 ölçekleme
        nn.init.normal_(self.tok_emb.weight, 0.0, c.init_std)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if name.endswith("router"):
                    std = c.router_init_std
                elif name.endswith("o_proj") or name.endswith("w_out"):
                    std = res_std
                else:
                    std = c.init_std
                nn.init.normal_(m.weight, 0.0, std)
            elif isinstance(m, MoEFFN):
                nn.init.normal_(m.w_in, 0.0, c.init_std)
                nn.init.normal_(m.w_out, 0.0, res_std)

    # --- ileri geçiş ---
    def _trunk(self, idx):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"dizi {T} > max_seq_len {self.cfg.max_seq_len}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        auxs, zs, sts = [], [], []
        for blk in self.blocks:
            if self.training and self.cfg.activation_checkpointing:
                x, aux, z, st = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x, aux, z, st = blk(x, cos, sin)
            auxs.append(aux); zs.append(z); sts.append(st)
        return self.norm_f(x), torch.stack(auxs).mean(), torch.stack(zs).mean(), torch.stack(sts)

    def forward(self, idx, targets=None, ce_normalizer=None, last_only=False):
        """targets yoksa (logits, None); targets varsa (None, loss) döner (logits üretilmez).
        loss = ce + aux_loss_coef * aux + z_loss_coef * z. Ayrıntılar self.last_stats içinde."""
        h, aux, z, st = self._trunk(idx)
        stats = {"aux": aux.detach(), "z": z.detach(), "moe": st.detach()}
        if targets is None:
            if last_only:
                h = h[:, -1:, :]
            self.last_stats = stats
            return F.linear(h, self.tok_emb.weight), None
        ce = chunked_cross_entropy(h.reshape(-1, h.size(-1)), targets.reshape(-1), self.tok_emb.weight,
                                   self.cfg.ce_chunk_size, -100, ce_normalizer)
        stats["ce"] = ce.detach()
        self.last_stats = stats
        return None, ce + self.cfg.aux_loss_coef * aux + self.cfg.z_loss_coef * z

    # --- optimizer: 3B uzman tensörleri ve Linear'lar decay alır ---
    def configure_optimizers(self, weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, device_type="cuda"):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_emb = name.startswith("tok_emb")                            # tied lm_head dahil
            is_router = ("router" in name) and not self.cfg.router_weight_decay
            (decay if (p.dim() >= 2 and not is_emb and not is_router) else no_decay).append(p)
        assert len(decay) + len(no_decay) == len(list(self.parameters()))
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        print(f"optimizer: decay {sum(p.numel() for p in decay)/1e6:.1f}M param ({len(decay)} tensör), "
              f"no-decay {sum(p.numel() for p in no_decay)/1e6:.1f}M ({len(no_decay)} tensör), fused={fused}")
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, **({"fused": True} if fused else {}))

    # --- sayımlar ---
    def param_report(self):
        c = self.cfg
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel()
        moes = [m for m in self.modules() if isinstance(m, MoEFFN)]
        routed = sum(m.w_in.numel() + m.w_out.numel() for m in moes)
        router = sum(m.router.weight.numel() for m in moes)
        shared = sum(p.numel() for m in self.modules() if isinstance(m, SwiGLU) for p in m.parameters())
        attn = sum(p.numel() for m in self.modules() if isinstance(m, GQAttention) for p in m.parameters())
        active_routed = routed * c.top_k // c.n_experts
        active_body = total - emb - routed + active_routed
        return dict(total=total, embedding=emb, attention=attn, routed_experts=routed, shared_experts=shared,
                    router=router, active_body=active_body, active_with_head=active_body + emb)

    def flops_per_token(self, seq_len):
        r = self.param_report()
        return 6 * r["active_with_head"] + 6 * self.cfg.n_layer * self.cfg.d_model * seq_len   # causal attention ≈ yarım

    @torch.no_grad()
    def moe_metrics(self):
        """Son forward'ın MoE sağlık metrikleri (GPU senkronu yapar; her ~50 adımda çağır)."""
        s = self.last_stats
        if "moe" not in s:
            return {}
        E = self.cfg.n_experts
        m = s["moe"].float().cpu()                                         # [L, E+3]
        counts = m[:, :E]
        mean = counts.mean(dim=1, keepdim=True).clamp_min(1.0)
        out = dict(aux=float(s["aux"]), z=float(s["z"]),
                   load_max_over_mean_worst=float((counts.max(1).values / mean.squeeze(1)).max()),
                   load_max_over_mean_avg=float((counts.max(1).values / mean.squeeze(1)).mean()),
                   dead_experts=int((counts < 0.1 * mean).sum()),
                   dropped_pct=100 * float(m[:, E].mean()), router_entropy=float(m[:, E + 1].mean()),
                   router_entropy_uniform=math.log(E), logit_max=float(m[:, E + 2].max()))
        if "ce" in s:
            out["ce"] = float(s["ce"])
        return out

    # --- üretim (KV cache yok) ---
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=None, eos_id=None):
        was_training = self.training
        self.eval()
        dev = idx.device.type
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.cfg.max_seq_len:], last_only=True)
                logits = logits[:, -1, :].float()
                if temperature <= 0:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                    if top_p is not None and top_p < 1.0:
                        sl, si = torch.sort(logits, descending=True)
                        remove = F.softmax(sl, -1).cumsum(-1) > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        logits = logits.masked_fill(torch.zeros_like(remove).scatter(-1, si, remove), float("-inf"))
                    nxt = torch.multinomial(F.softmax(logits, -1), 1)
                idx = torch.cat([idx, nxt], dim=1)
                if eos_id is not None and bool((nxt == eos_id).all()):
                    break
        self.train(was_training)
        return idx


# ───────────────────────────── Testler (CPU'da çalışır) ─────────────────────────────
def run_selftests():
    torch.manual_seed(0)
    results = []

    def check(name, ok, detail="", warn_only=False):
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        results.append(status)
        print(f"[{status}] {name}  {detail}")

    cfg = MoEConfig(vocab_size=257, n_layer=2, d_model=64, n_heads=4, n_kv_heads=2, max_seq_len=64,
                    n_experts=8, top_k=2, expert_hidden=32, shared_hidden=32, capacity_factor=1.25)
    model = DorinaMoEV3(cfg)

    # 1) parametre sayısı: bağımsız formülle
    d, hd, L, E, h, hs, V = cfg.d_model, cfg.head_dim, cfg.n_layer, cfg.n_experts, cfg.expert_hidden, cfg.shared_hidden, cfg.vocab_size
    attn = d * cfg.n_heads * hd + 2 * d * cfg.n_kv_heads * hd + cfg.n_heads * hd * d
    expected = V * d + L * (attn + 2 * d + d * E + E * 3 * d * h + 3 * d * hs) + d
    check("parametre_sayisi", expected == sum(p.numel() for p in model.parameters()),
          f"beklenen {expected}, gerçek {sum(p.numel() for p in model.parameters())}")

    # 2) MoE == naif referans (dropless eval ve kapasite çok büyük eğitim yolu)
    moe = MoEFFN(cfg)
    nn.init.normal_(moe.w_in, 0, 0.2); nn.init.normal_(moe.w_out, 0, 0.2); nn.init.normal_(moe.router.weight, 0, 0.5)
    x = torch.randn(2, 16, cfg.d_model)

    def reference(m, x):
        xf = x.reshape(-1, x.size(-1))
        p = F.linear(xf, m.router.weight).softmax(-1)
        v, i = p.topk(m.k, -1)
        v = v / v.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for e in range(m.E):
            t, s = (i == e).nonzero(as_tuple=True)
            if t.numel():
                g, u = (xf[t] @ m.w_in[e]).chunk(2, -1)
                out.index_add_(0, t, ((F.silu(g) * u) @ m.w_out[e]) * v[t, s, None])
        return (out + m.shared(xf)).view_as(x)

    ref = reference(moe, x)
    moe.eval()
    check("moe_dropless_eval_referans", torch.allclose(moe(x)[0], ref, atol=1e-5), f"maks fark {(moe(x)[0]-ref).abs().max():.2e}")
    moe.train(); moe.capacity_factor = 1e3
    check("moe_egitim_yolu_referans", torch.allclose(moe(x)[0], ref, atol=1e-5), "kapasite çok büyük => düşme yok")

    # 3) kapasite düşmesi
    moe.capacity_factor = 0.25
    o, aux, z, st = moe(x)
    check("moe_kapasite_dusmesi", float(st[cfg.n_experts]) > 0 and torch.isfinite(o).all(),
          f"düşen oran {float(st[cfg.n_experts]):.2f}")
    moe.capacity_factor = cfg.capacity_factor

    # 4) nedensellik (gelecek token geçmişi değiştirmemeli)
    model.eval()
    idx = torch.randint(0, V, (1, 32))
    l1, _ = model(idx)
    idx2 = idx.clone(); idx2[0, 20] = (idx2[0, 20] + 1) % V
    l2, _ = model(idx2)
    check("nedensellik", torch.allclose(l1[:, :20], l2[:, :20], atol=1e-5) and not torch.allclose(l1[:, 20:], l2[:, 20:], atol=1e-5),
          "20. token değişince önceki logits sabit")

    # 5) RoPE göreli konum özelliği
    cos, sin = build_rope(16, 32, 10000.0)
    v = torch.randn(1, 1, 1, 16).expand(1, 1, 32, 16).contiguous()
    r = apply_rope(v, cos, sin)[0, 0]
    S = r @ r.T
    check("rope_goreli_konum", torch.allclose(S[3, 1], S[10, 8], atol=1e-4) and torch.allclose(S[5, 5], S[17, 17], atol=1e-4),
          "skor yalnızca (i-j)'ye bağlı")

    # 6) GQA == aynı ağırlıkları çoğaltılmış MHA
    cg = cfg
    cm = replace(cfg, n_kv_heads=cfg.n_heads)
    ag, am = GQAttention(cg), GQAttention(cm)
    grp = cfg.n_heads // cfg.n_kv_heads
    with torch.no_grad():
        am.q_proj.weight.copy_(ag.q_proj.weight); am.o_proj.weight.copy_(ag.o_proj.weight)
        for a, b in ((ag.k_proj, am.k_proj), (ag.v_proj, am.v_proj)):
            b.weight.copy_(a.weight.view(cfg.n_kv_heads, hd, d).repeat_interleave(grp, dim=0).reshape(cfg.n_heads * hd, d))
    xx = torch.randn(2, 16, d)
    c_, s_ = build_rope(hd, 16, cfg.rope_theta)
    check("gqa_esdegerligi", torch.allclose(ag(xx, c_, s_), am(xx, c_, s_), atol=1e-5), "GQA = kv çoğaltılmış MHA")

    # 7) parçalı CE == tam CE (değer ve gradyan, -100 maskeli)
    hid = torch.randn(37, 64, requires_grad=True)
    W = torch.randn(257, 64, requires_grad=True)
    tg = torch.randint(0, 257, (37,)); tg[::5] = -100
    full = F.cross_entropy(F.linear(hid, W), tg, ignore_index=-100)
    g_full = torch.autograd.grad(full, [hid, W])
    ch = chunked_cross_entropy(hid, tg, W, chunk_size=8)
    g_ch = torch.autograd.grad(ch, [hid, W])
    check("parcali_ce_deger", torch.allclose(full, ch, atol=1e-5), f"tam {full:.5f} parçalı {ch:.5f}")
    check("parcali_ce_gradyan", all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g_full, g_ch)), "hidden ve W gradyanları")
    allm = chunked_cross_entropy(hid, torch.full((37,), -100), W, chunk_size=8)
    check("parcali_ce_hepsi_maskeli", float(allm) == 0.0 and torch.isfinite(allm), "geçerli token yok => 0, NaN yok")

    # 8) model: ce istatistiği == elle hesaplanan ce; geri yayılım; uzman/router gradyanları
    model.eval(); lg, _ = model(idx)
    ce_ref = F.cross_entropy(lg[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1))
    for blk in model.blocks:
        blk.moe.capacity_factor = 1e3
    model.train()
    _, loss = model(idx[:, :-1], idx[:, 1:])
    check("model_ce_entegrasyon", abs(float(model.last_stats["ce"]) - float(ce_ref)) < 1e-4,
          f"ce {float(model.last_stats['ce']):.5f} vs {float(ce_ref):.5f}")
    loss.backward()
    gx = [blk.moe.w_in.grad for blk in model.blocks] + [blk.moe.w_out.grad for blk in model.blocks]
    gr = [blk.moe.router.weight.grad for blk in model.blocks]
    check("geri_yayilim", all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gx + gr),
          "uzman ve router gradyanları sonlu ve sıfırdan farklı")
    for blk in model.blocks:
        blk.moe.capacity_factor = cfg.capacity_factor

    # 9) optimizer grupları
    opt = model.configure_optimizers(weight_decay=0.1, lr=1e-3, device_type="cpu")
    dec = {id(p) for p in opt.param_groups[0]["params"]}
    names = {id(p): n for n, p in model.named_parameters()}
    three_d_ok = all(id(p) in dec for p in model.parameters() if p.dim() == 3)
    bad = [names[i] for i in dec if names[i].startswith("tok_emb") or "norm" in names[i] or "router" in names[i]]
    check("weight_decay_gruplari", three_d_ok and not bad, f"3B tensörler decay'de; decay'deki istenmeyenler: {bad}")

    # 10) tek batch'e aşırı uyum (bilgi amaçlı)
    torch.manual_seed(1)
    m2 = DorinaMoEV3(cfg); m2.train()
    o2 = m2.configure_optimizers(lr=3e-3, device_type="cpu")
    xb = torch.randint(0, V, (4, 33))
    first = None
    for _ in range(60):
        _, l = m2(xb[:, :-1], xb[:, 1:])
        first = first if first is not None else float(l)
        l.backward(); o2.step(); o2.zero_grad(set_to_none=True)
    check("tek_batch_overfit", float(l) < 0.8 * first, f"{first:.3f} -> {float(l):.3f} (60 adım)", warn_only=True)

    n_fail = results.count("FAIL")
    print(f"\nselftest: {results.count('PASS')} PASS, {results.count('WARN')} WARN, {n_fail} FAIL")
    return n_fail == 0


def get_amp_precision():
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            return torch.bfloat16, False
        else:
            return torch.float16, True
    return torch.float32, False


# ───────────────────────────── Mikro-benchmark (GPU) ─────────────────────────────
def bench_once(cfg, micro_batch, seq_len, steps, warmup=5, lr=6e-4):
    dev = "cuda"
    model = DorinaMoEV3(cfg).to(dev)
    opt = model.configure_optimizers(weight_decay=0.1, lr=lr, device_type="cuda")
    amp_dtype, use_scaler = get_amp_precision()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    x = torch.randint(0, cfg.vocab_size, (micro_batch, seq_len), device=dev)
    y = torch.randint(0, cfg.vocab_size, (micro_batch, seq_len), device=dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.time()
    for i in range(steps):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=amp_dtype):
            _, loss = model(x, y)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.detach())
    torch.cuda.synchronize()
    dt = max(1e-5, time.time() - t0)
    effective_steps = max(1, steps - warmup) if steps > warmup else steps
    toks = effective_steps * micro_batch * seq_len
    res = dict(tok_s=toks / dt, step_ms=1000 * dt / effective_steps,
               peak_alloc_gb=torch.cuda.max_memory_allocated() / 1024**3,
               peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
               loss_first=float(losses[0]), loss_last=float(losses[-1]),
               tflops=toks * model.flops_per_token(seq_len) / dt / 1e12, metrics=model.moe_metrics())
    del model, opt, x, y
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--sweep", default="", help="örn. 2,4,8: her mikro-batch için ayrı ölçüm")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--expert-hidden", type=int, default=256)
    ap.add_argument("--shared-hidden", type=int, default=256)
    ap.add_argument("--capacity", type=float, default=1.25)
    args = ap.parse_args()

    cfg = MoEConfig(n_experts=args.experts, top_k=args.top_k, expert_hidden=args.expert_hidden,
                    shared_hidden=args.shared_hidden, capacity_factor=args.capacity,
                    activation_checkpointing=args.grad_ckpt, max_seq_len=args.seq)
    meta = DorinaMoEV3(cfg)
    r = meta.param_report()
    print("=" * 70)
    print(f"Dorina V3 | d={cfg.d_model} L={cfg.n_layer} heads={cfg.n_heads}/{cfg.n_kv_heads}(kv) ctx={cfg.max_seq_len} "
          f"E={cfg.n_experts} top{cfg.top_k} h={cfg.expert_hidden} shared_h={cfg.shared_hidden}")
    print(f"toplam           : {r['total']/1e6:7.1f} M")
    print(f"  embedding(tied): {r['embedding']/1e6:7.1f} M | attention {r['attention']/1e6:.1f} M | router {r['router']/1e6:.2f} M")
    print(f"  routed uzmanlar: {r['routed_experts']/1e6:7.1f} M | shared uzmanlar {r['shared_experts']/1e6:.1f} M")
    print(f"aktif (gövde)    : {r['active_body']/1e6:7.1f} M | aktif (+lm_head): {r['active_with_head']/1e6:.1f} M")
    print(f"FLOP/token       : {meta.flops_per_token(cfg.max_seq_len)/1e6:.0f} M")
    print(f"statik VRAM tahmini (fp32 param+grad+Adam): {r['total']*16/1024**3:.2f} GB (+ aktivasyon, logits dilimi)")
    print("=" * 70)
    del meta

    ok = True
    if not args.skip_selftest:
        ok = run_selftests()

    if not torch.cuda.is_available():
        print("CUDA bulunamadı: GPU benchmark atlandı.")
        sys.exit(0 if ok else 1)
    if not torch.cuda.is_bf16_supported():
        print("Bu GPU bf16 desteklemiyor, benchmark atlandı.")
        sys.exit(1)

    sizes = [int(s) for s in args.sweep.split(",") if s] or [args.micro_batch]
    print(f"\nGPU: {torch.cuda.get_device_name(0)} | toplam VRAM {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    for mb in sizes:
        try:
            b = bench_once(cfg, mb, args.seq, args.steps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"mikro-batch {mb}: OOM (--grad-ckpt veya daha küçük mikro-batch dene)")
            continue
        m = b["metrics"]
        print(f"mikro-batch {mb} x {args.seq} | {b['tok_s']:,.0f} tok/s | {b['step_ms']:.0f} ms/adım | {b['tflops']:.1f} TFLOP/s | "
              f"tepe VRAM: ayrılan {b['peak_alloc_gb']:.2f} GB, rezerve {b['peak_reserved_gb']:.2f} GB | "
              f"loss {b['loss_first']:.2f} -> {b['loss_last']:.2f}")
        print(f"   MoE: yük max/ort (en kötü katman) {m['load_max_over_mean_worst']:.2f}, ölü uzman {m['dead_experts']}, "
              f"düşme %{m['dropped_pct']:.1f}, router entropi {m['router_entropy']:.2f}/{m['router_entropy_uniform']:.2f}, "
              f"logit max {m['logit_max']:.1f}")
    print("\nNot: Windows ekranı 0.3-0.8 GB VRAM kullanır; tepe rezerve değeri 5.3 GB altında kalmalı.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
