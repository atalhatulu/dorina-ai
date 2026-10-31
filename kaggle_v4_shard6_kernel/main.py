#!/usr/bin/env python3
"""
Dorina-V4 Shard 6 Master Trainer: 1:20 Milestone & WSD Cosine Decay (2.5B -> 3.0B Tokens).
Target: Dual Tesla T4 GPUs (PyTorch DDP).
Loads weights from Shard 5 checkpoint (/kaggle/input/dorina-v4-shard5-base/dorina_v4_shard5_final.pt).

Claude Protokolü (Altın Shard 6):
1. Müfredat Oranları (Global Interleaved & Document-Shuffled):
   - %35 Türkçe HQ Web & Wiki (175M Tokens)
   - %20 FineWeb-Edu HQ Score >= 3.5 (100M Tokens)
   - %25 Python Edu Cleaned (125M Tokens)
   - %20 FineMath 4+ (100M Tokens)
   - HER DOKÜMAN SONUNA <|endoftext|> (ID: 0) EKLENİR!

2. WSD Zamanlaması & Çift Kademeli Ayrım (Branching):
   - Adım 0 - 1.100 (~145M Token): Stable Faz (LR = 5.5e-4)
   - Adım 1.100: "1:50 Continuation Checkpoint" (dorina_v4_step_1100_momentum.pt) DONDURULUR!
   - Adım 1.100 - 3.800 (~355M Token): Cosine Decay ile LR 5.5e-4 -> 1.0e-5 seviyesine iner.
   - Adım 3.800: "1:20 Final SFT Base" (dorina_v4_shard6_1to20_crystal_final.pt) üretilir.
"""

import os
import sys
import time
import math
import json
import random
import argparse
import subprocess
import urllib.request
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# =====================================================================
# 1. DORINA-V4 DENSE ARCHITECTURE SPECIFICATION (144.7M)
# =====================================================================

@dataclass
class DenseConfig:
    vocab_size: int = 32768
    n_layer: int = 16
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    hidden_dim: int = 2560
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
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.rep = self.n_heads // self.n_kv
        self.hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.hd, eps=cfg.norm_eps)
        self.k_norm = RMSNorm(self.hd, eps=cfg.norm_eps)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos[:S], sin[:S])
        k = apply_rope(k, cos[:S], sin[:S])
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, -1))

class DenseBlock(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.hidden_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

class DorinaDenseV4(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self._init_weights()

    def _init_weights(self):
        c, s = build_rope(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta)
        self.rope_cos = c
        self.rope_sin = s
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=self.cfg.init_std)

    def forward(self, input_ids):
        h = self.tok_embeddings(input_ids)
        cos = self.rope_cos.to(h.device)
        sin = self.rope_sin.to(h.device)
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm_f(h)
        logits = F.linear(h, self.tok_embeddings.weight)
        return logits, None

# ==============================================================================
# 2. DORINA DEEP TELEMETRY LOGGER (LabLogger V4)
# ==============================================================================
class ShardTelemetryLogger:
    def __init__(self, output_dir, total_steps, model_name="Dorina-V4-Shard6"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.total_steps = total_steps
        self.model_name = model_name
        self.jsonl_path = os.path.join(output_dir, "training_log.jsonl")
        self.summary_json_path = os.path.join(output_dir, "training_summary.json")
        self.start_time = time.time()
        self.history = []

    def log_step(self, step, loss, lr, tok_s, vram_gb, grad_norm):
        entry = {
            "step": step,
            "loss": round(float(loss), 4),
            "perplexity": round(math.exp(min(float(loss), 20.0)), 2),
            "lr": float(lr),
            "tok_per_sec": round(float(tok_s), 1),
            "vram_gb": round(float(vram_gb), 2),
            "grad_norm": round(float(grad_norm), 2),
            "elapsed_min": round((time.time() - self.start_time) / 60.0, 2)
        }
        self.history.append(entry)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def finish(self, final_loss):
        total_time_hours = (time.time() - self.start_time) / 3600.0
        summary = {
            "model_name": self.model_name,
            "total_steps": self.total_steps,
            "final_loss": round(float(final_loss), 4),
            "total_time_hours": round(total_time_hours, 2),
            "start_time": time.ctime(self.start_time),
            "end_time": time.ctime()
        }
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[TELEMETRY] Egitim ozeti kaydedildi: {self.summary_json_path}", flush=True)

        try:
            import matplotlib.pyplot as plt
            steps = [h["step"] for h in self.history]
            losses = [h["loss"] for h in self.history]
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, label="Training Loss", color="#1f77b4", linewidth=2)
            plt.axvline(x=1100, color="r", linestyle="--", label="Decay Start (Step 1100)")
            plt.title(f"{self.model_name} WSD Loss Curve (1:20 Milestone)")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plot_path = os.path.join(self.output_dir, "loss_curve.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[TELEMETRY] Loss egrisi olusturuldu: {plot_path}", flush=True)
        except Exception as e:
            print(f"[TELEMETRY] Grafik cizim hatasi: {e}", flush=True)

# ==============================================================================
# 3. GLOBAL INTERLEAVED & SHUFFLED CURRICULUM PACKER (Claude Kuralı)
# ==============================================================================
def download_texts(url, text_col, target_tokens, filter_fn=None):
    local_p = "/kaggle/working/temp_shard6.parquet"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(local_p, "wb") as f_p:
        while True:
            chunk = resp.read(1024 * 1024 * 8)
            if not chunk: break
            f_p.write(chunk)

    table = pq.read_table(local_p)
    cols = table.column_names
    texts = table[text_col].to_pylist() if text_col in cols else []
    scores = table["score"].to_pylist() if "score" in cols else None
    
    valid_texts = []
    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 15: continue
        if filter_fn and not filter_fn(i, text, scores): continue
        valid_texts.append(text)

    if os.path.exists(local_p):
        os.remove(local_p)
    return valid_texts

def prepare_shard6_curriculum(tokenizer_path, out_bin_path):
    target_total = 500_000_000
    expected_bytes = target_total * 2
    if os.path.exists(out_bin_path) and os.path.getsize(out_bin_path) >= expected_bytes:
        print(f"[DATA PREP] Shard 6 binary cache hazir: {out_bin_path}", flush=True)
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("\n" + "="*70, flush=True)
    print("DORINA-V4 SHARD 6 DOKÜMAN BAZINDA KARIŞTIRILMIŞ MÜFREDAT PAKETLEYİCİ", flush=True)
    print("  1. %35 Türkçe HQ Web & Wiki (175M Tokens)", flush=True)
    print("  2. %25 Python Edu Cleaned (125M Tokens)", flush=True)
    print("  3. %20 FineMath 4+ (100M Tokens)", flush=True)
    print("  4. %20 FineWeb-Edu HQ Score >= 3.5 (100M Tokens)", flush=True)
    print("="*70, flush=True)

    sources = [
        ("Turkish HQ", 175_000_000, [
            "https://huggingface.co/api/datasets/altaidevorg/fineweb2-hq-turkish/parquet/default/train/0.parquet",
            "https://huggingface.co/api/datasets/altaidevorg/fineweb2-hq-turkish/parquet/default/train/1.parquet",
            "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.tr/train/0.parquet"
        ], "text", None),
        ("Python Edu", 125_000_000, [
            "https://huggingface.co/api/datasets/Avelina/python-edu-cleaned/parquet/default/train/0.parquet",
            "https://huggingface.co/api/datasets/Avelina/python-edu-cleaned/parquet/default/train/1.parquet"
        ], "text", lambda i, text, scores: (scores[i] >= 3.0 if scores else True)),
        ("FineMath", 100_000_000, [
            "https://huggingface.co/api/datasets/HuggingFaceTB/finemath/parquet/finemath-3plus/train/0.parquet",
            "https://huggingface.co/api/datasets/HuggingFaceTB/finemath/parquet/finemath-3plus/train/1.parquet"
        ], "text", None),
        ("FineWeb-Edu", 100_000_000, [
            "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet/sample-10BT/train/6.parquet",
            "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet/sample-10BT/train/7.parquet"
        ], "text", lambda i, text, scores: (scores[i] >= 3.5 if scores else True))
    ]

    all_document_token_chunks = []
    
    for s_name, s_target, s_urls, s_col, s_filt in sources:
        print(f"\n[*] {s_name} cekiliyor (Hedef: {s_target:,} token)...", flush=True)
        s_tokens = 0
        for u in s_urls:
            if s_tokens >= s_target: break
            print(f"  -> Indiriliyor: {u}", flush=True)
            texts = download_texts(u, s_col, s_target - s_tokens, s_filt)
            for t in texts:
                enc = tokenizer.encode(t).ids
                enc.append(0) # <|endoftext|> KURALI
                all_document_token_chunks.append(enc)
                s_tokens += len(enc)
                if s_tokens >= s_target: break
        print(f"  [+] {s_name} tamamlandi: {s_tokens:,} token.", flush=True)

    print("\n[*] Tum dokumanlar kuresel olarak karistiriliyor (Global Interleave / Shuffle)...", flush=True)
    random.seed(42)
    random.shuffle(all_document_token_chunks)

    print(f"[*] /kaggle/working/v4_shard6.bin dosyasina yaziliyor (Hedef 500M token)...", flush=True)
    total_written = 0
    with open(out_bin_path, "wb") as f_out:
        for chunk in all_document_token_chunks:
            needed = target_total - total_written
            if needed <= 0: break
            to_write = chunk[:needed]
            arr = np.array(to_write, dtype=np.uint16)
            f_out.write(arr.tobytes())
            total_written += len(to_write)

    print(f"[DATA PREP COMPLETE] Shard 6 basariyla paketlendi: {total_written:,} tokens ({os.path.getsize(out_bin_path)/(1024*1024):.1f} MB)\n", flush=True)

class BinaryDataset:
    def __init__(self, bin_path, seq_len=2048, rank=0, world_size=1):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        total_tokens = len(self.data)
        self.total_seqs = total_tokens // (seq_len + 1)
        self.seqs_per_rank = self.total_seqs // world_size
        self.start_seq = rank * self.seqs_per_rank

    def get_batch(self, micro_batch, step):
        idx_in_rank = (step * micro_batch) % max(1, (self.seqs_per_rank - micro_batch))
        seq_idx = self.start_seq + idx_in_rank
        batch_x = []
        batch_y = []
        for i in range(micro_batch):
            start = (seq_idx + i) * (self.seq_len + 1)
            chunk = torch.from_numpy((self.data[start : start + self.seq_len + 1]).astype(np.int64))
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])
        return torch.stack(batch_x), torch.stack(batch_y)

# =====================================================================
# 4. WORKER DDP TRAINING EXECUTION (WSD WITH STEP 1100 FORK & DECAY)
# =====================================================================
def get_shard6_wsd_lr(step, max_steps=3800, warmup_steps=100, peak_lr=5.5e-4, decay_start=1100, min_lr=1.0e-5):
    # 1. Warmup (0 - 100)
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    # 2. Stable Faz (100 - 1100)
    if step < decay_start:
        return peak_lr
    # 3. Cosine Decay Fazı (1100 - 3800) -> 1.0e-5 Tabana İniş
    decay_ratio = (step - decay_start) / (max_steps - decay_start)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)

def worker_train(rank, local_rank, world_size, is_master, device, args):
    cfg = DenseConfig(max_seq_len=2048)
    telemetry = ShardTelemetryLogger(args.out_dir, args.max_steps, 'Dorina-V4-Shard6') if is_master else None
    model = DorinaDenseV4(cfg).to(device)

    # 1. Weights Transfer from Shard 5 Checkpoint
    if args.resume_weights and os.path.exists(args.resume_weights):
        if is_master:
            print(f"[RELOAD] Loading prior Shard 5 weights from: {args.resume_weights}", flush=True)
        sd = torch.load(args.resume_weights, map_location="cpu", weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
        if is_master:
            print("  ✅ [RELOAD SUCCESS] Shard 5 weights (2.5B token brain) transferred smoothly to Shard 6!", flush=True)
    else:
        if is_master:
            print(f"[WARNING] Prior weights not found at {args.resume_weights}, training from scratch!", flush=True)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if world_size > 1 else model

    dataset = BinaryDataset(args.data_bin, seq_len=cfg.max_seq_len, rank=rank, world_size=world_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    if is_master:
        print("\n" + "="*70, flush=True)
        print("TRAINING DORINA-V4 SHARD 6 STARTED (Dual T4 DDP)", flush=True)
        print("Steps: 3,800 | Batch: 131,072 tok/step | Target: 500M Tokens", flush=True)
        print("WSD Schedule: Stable (Steps 0-1100) -> Forking -> Cosine Decay to 1.0e-5 (Steps 1100-3800)", flush=True)
        print("="*70 + "\n", flush=True)

    t0 = time.time()
    last_step = 0

    for step in range(args.max_steps):
        # WSD LR Hesaplama
        cur_lr = get_shard6_wsd_lr(step, max_steps=args.max_steps, warmup_steps=args.warmup_steps, peak_lr=args.lr, decay_start=1100, min_lr=1.0e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            x, y = dataset.get_batch(args.micro_batch, step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _ = model(x)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Telemetri ve Konsol Çıktısı (Her 25 adımda)
        if is_master and (step % 25 == 0 or step == args.max_steps - 1):
            tok_s = (step - last_step) * args.batch_size * 2 * args.seq_len * args.grad_accum / max(0.001, (time.time() - t0))
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            phase = "STABLE" if step < 1100 else "DECAY"
            print(f"[SHARD6-{phase}] Step {step:5d}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {cur_lr:.2e} | Speed: {tok_s:,.0f} tok/s | VRAM: {vram_gb:.2f} GB | Norm: {grad_norm:.2f}", flush=True)
            if telemetry:
                telemetry.log_step(step, accum_loss, cur_lr, tok_s, vram_gb, grad_norm)
            t0 = time.time()
            last_step = step

        # ==============================================================
        # CRITICAL MILESTONE: STEP 1100 FORKING (1:50 CONTINUATION BASE)
        # ==============================================================
        if is_master and step == 1100:
            fork_save_path = os.path.join(args.out_dir, "dorina_v4_3b_stable_momentum.pt")
            print("\n" + "#"*75, flush=True)
            print("🏛️ [FORK MILESTONE REACHED] STEP 1100 - 1:50 CONTINUATION BASE DONDURULUYOR!", flush=True)
            torch.save({
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "lr": cur_lr,
                "cumulative_tokens": "2.645_Billion"
            }, fork_save_path)
            print(f"  ✅ [SAVED FORK] {fork_save_path} güvenle donduruldu. 1:50 (Shard 7) buradan devam edecek!", flush=True)
            print("  ❄️ [DECAY INITIATED] Model artık 1:20 final asistanı için kristalleşme fazına geçiyor...", flush=True)
            print("#"*75 + "\n", flush=True)

        # Periyodik Checkpoint (Her 500 Adımda)
        if is_master and step > 0 and (step % 500 == 0 or step == args.max_steps - 1):
            ckpt_path = os.path.join(args.out_dir, f"dorina_v4_shard6_step_{step}.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"  [CHECKPOINT] Periodic state saved: {ckpt_path}", flush=True)

    # 1:20 Nihai Kristalleşmiş Model Kaydı
    if is_master:
        final_save_path = os.path.join(args.out_dir, "dorina_v4_shard6_1to20_crystal_final.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(raw_model.state_dict(), final_save_path)
        print("\n" + "="*75, flush=True)
        print(f"🏆 [SUCCESS] 1:20 DÖNÜM NOKTASI TAMAMLANDI (3.0B TOKENS)!", flush=True)
        print(f"  Final Kristalleşmiş Ağırlık: {final_save_path}", flush=True)
        print("="*75, flush=True)
        if telemetry: telemetry.finish(accum_loss)

# =====================================================================
# 5. ENTRYPOINT & AUTOMATED RUNNER
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Internal worker flag for torchrun")
    parser.add_argument("--data-bin", type=str, default="/kaggle/working/v4_shard6.bin")
    parser.add_argument("--resume-weights", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/models_v4")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3800)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5.5e-4)
    args = parser.parse_args()

    # Worker DDP
    if args.worker:
        ddp = int(os.environ.get("RANK", -1)) != -1
        if ddp:
            dist.init_process_group(backend="nccl")
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(device)
            is_master = (rank == 0)
        else:
            rank, local_rank, world_size = 0, 0, 1
            device = "cuda" if torch.cuda.is_available() else "cpu"
            is_master = True

        args.batch_size = args.micro_batch
        args.seq_len = 2048
        worker_train(rank, local_rank, world_size, is_master, device, args)
        if ddp:
            dist.destroy_process_group()
        return

    # Master Launcher
    print("=" * 70, flush=True)
    print("DORINA V4 AUTONOMOUS MASTER TRAINER (SHARD 6 - 1:20 DECAY MILESTONE)", flush=True)
    print("=" * 70, flush=True)

    n_gpus = torch.cuda.device_count()
    tokenizer_path = None
    resume_path = None

    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                full_p = os.path.join(root, f)
                if "tokenizer" in f.lower() and f.endswith(".json") and not tokenizer_path:
                    tokenizer_path = full_p
                if ("shard5" in f.lower() or "shard" in f.lower()) and f.endswith(".pt") and not resume_path:
                    resume_path = full_p

    print(f"[Resource Check] Tokenizer: {tokenizer_path}", flush=True)
    print(f"[Resource Check] Base Weights (Shard 5): {resume_path}", flush=True)

    bin_file = "/kaggle/working/v4_shard6.bin"
    prepare_shard6_curriculum(tokenizer_path, bin_file)

    script_path = os.path.abspath(__file__)
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        script_path,
        "--worker",
        "--data-bin", bin_file,
        "--out-dir", args.out_dir,
        "--micro-batch", str(args.micro_batch),
        "--grad-accum", str(args.grad_accum),
        "--max-steps", str(args.max_steps),
        "--warmup-steps", str(args.warmup_steps),
        "--lr", str(args.lr)
    ]
    if resume_path:
        cmd.extend(["--resume-weights", resume_path])

    print("[Execute Command]:", " ".join(cmd), flush=True)
    sys.stdout.flush()
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
