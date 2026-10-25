#!/usr/bin/env python3
"""
Dorina-V4 Shard 4 Autonomous Balanced Multi-Curriculum Trainer (500M Tokens -> 2.0B Cumulative).
Target: Dual Tesla T4 GPUs (PyTorch DDP).
Loads weights from Shard 3 checkpoint (/kaggle/input/dorina-v4-shard3-base/dorina_v4_shard3_final.pt).

Müfredat Dağılımı (500M Token):
  1. %35 FineWeb-Edu (175M Token) -> İngilizce genel kültür ve bilim hafızası koruma (Replay)
  2. %25 Python Edu Cleaned (125M Token) -> Sentaks, algoritmik mantık, girinti ve fonksiyon kurgusu
  3. %15 FineMath 3+ (75M Token) -> Matematiksel nedensellik ve adım adım akıl yürütme
  4. %25 Türkçe HQ Web & Wiki (125M Token) -> FineWeb2-HQ Turkish + Wikipedia TR ile taban Türkçe

Eğitim Rejimi:
  - Stable LR = 5.5e-4 (WSD Kararlı Fazı - Kesinlikle erken decay yok)
  - Batch: 131,072 tok/step (3,800 adım)
  - Her 500 adımda bir periyodik checkpoint
"""

import os
import sys
import time
import math
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
# 1. DORINA-V4 DENSE ARCHITECTURE SPECIFICATION
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

class DenseBlock(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
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

# =====================================================================
# 2. SHARD 4 BALANCED MULTI-CURRICULUM HARVESTER
# =====================================================================

def download_and_extract_tokens(url, text_col, tokenizer, target_tokens, filter_fn=None):
    local_p = "/kaggle/working/temp_curriculum.parquet"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(local_p, "wb") as f_p:
        while True:
            chunk = resp.read(1024 * 1024 * 8)
            if not chunk:
                break
            f_p.write(chunk)

    table = pq.read_table(local_p)
    cols = table.column_names
    texts = table[text_col].to_pylist() if text_col in cols else []
    
    # Filter if needed
    scores = table["score"].to_pylist() if "score" in cols else None
    
    extracted = []
    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 10:
            continue
        if filter_fn and not filter_fn(i, text, scores):
            continue
            
        enc = tokenizer.encode(text)
        extracted.extend(enc.ids)
        extracted.append(0) # EOS
        if len(extracted) >= target_tokens:
            break

    if os.path.exists(local_p):
        os.remove(local_p)

    return extracted[:target_tokens]

def prepare_shard4_curriculum(tokenizer_path, out_bin_path):
    target_total = 500_000_000
    expected_bytes = target_total * 2
    if os.path.exists(out_bin_path) and os.path.getsize(out_bin_path) >= expected_bytes:
        print(f"[DATA PREP] Shard 4 binary cache already complete: {out_bin_path} ({os.path.getsize(out_bin_path)/(1024*1024):.1f} MB)", flush=True)
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("\n" + "="*70, flush=True)
    print("DORINA-V4 SHARD 4 MULTI-CURRICULUM PACKER (500M TOKENS)", flush=True)
    print("  1. FineWeb-Edu (English Replay): 175M Tokens (35%)", flush=True)
    print("  2. Python Edu Cleaned (Coding Logic): 125M Tokens (25%)", flush=True)
    print("  3. FineMath 3+ (Mathematics & Reasoning): 75M Tokens (15%)", flush=True)
    print("  4. Turkish HQ Web & Wiki (Turkish Foundation): 125M Tokens (25%)", flush=True)
    print("="*70, flush=True)

    sources = [
        # (Name, TargetTokens, ParquetURLs, TextColumn, FilterFunction)
        (
            "FineWeb-Edu (EN Replay)",
            175_000_000,
            [
                "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet/sample-10BT/train/6.parquet",
                "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet/sample-10BT/train/7.parquet"
            ],
            "text",
            lambda i, text, scores: (scores[i] >= 3.0 if scores else True)
        ),
        (
            "Python Edu Cleaned (Coding)",
            125_000_000,
            [
                "https://huggingface.co/api/datasets/Avelina/python-edu-cleaned/parquet/default/train/0.parquet",
                "https://huggingface.co/api/datasets/Avelina/python-edu-cleaned/parquet/default/train/1.parquet"
            ],
            "text",
            lambda i, text, scores: (scores[i] >= 3.0 if scores else True)
        ),
        (
            "FineMath 3+ (Reasoning & Math)",
            75_000_000,
            [
                "https://huggingface.co/api/datasets/HuggingFaceTB/finemath/parquet/finemath-3plus/train/0.parquet",
                "https://huggingface.co/api/datasets/HuggingFaceTB/finemath/parquet/finemath-3plus/train/1.parquet"
            ],
            "text",
            None
        ),
        (
            "Turkish HQ Web & Wiki (Turkish)",
            125_000_000,
            [
                "https://huggingface.co/api/datasets/altaidevorg/fineweb2-hq-turkish/parquet/default/train/0.parquet",
                "https://huggingface.co/api/datasets/altaidevorg/fineweb2-hq-turkish/parquet/default/train/1.parquet",
                "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.tr/train/0.parquet"
            ],
            "text",
            None
        )
    ]

    total_packed = 0
    with open(out_bin_path, "wb") as f_out:
        for s_name, s_target, s_urls, s_col, s_filt in sources:
            print(f"\n>>> Harvesting {s_name} (Target: {s_target:,} tokens)...", flush=True)
            s_collected = 0
            for u_idx, url in enumerate(s_urls):
                needed = s_target - s_collected
                if needed <= 0:
                    break
                print(f"  Downloading & tokenizing part {u_idx+1}/{len(s_urls)}: {url}", flush=True)
                toks = download_and_extract_tokens(url, s_col, tokenizer, needed, s_filt)
                if len(toks) > 0:
                    arr = np.array(toks, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    s_collected += len(toks)
                    total_packed += len(toks)
                    print(f"    Collected for {s_name}: {s_collected:,} / {s_target:,} tokens | Total Shard 4: {total_packed:,} / {target_total:,}", flush=True)

    print(f"\n[DATA PREP COMPLETE] Shard 4 successfully packed: {total_packed:,} tokens into {out_bin_path} ({os.path.getsize(out_bin_path)/(1024*1024):.1f} MB)\n", flush=True)

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
# 3. WORKER DDP TRAINING EXECUTION (STABLE WSD REGIME)
# =====================================================================

def worker_train(rank, local_rank, world_size, is_master, device, args):
    cfg = DenseConfig(max_seq_len=2048)
    model = DorinaDenseV4(cfg).to(device)

    # 1. Weights Transfer from Shard 3 Checkpoint
    if args.resume_weights and os.path.exists(args.resume_weights):
        if is_master:
            print(f"[RELOAD] Loading prior Shard 3 weights from: {args.resume_weights}", flush=True)
        sd = torch.load(args.resume_weights, map_location="cpu", weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
        if is_master:
            print("  ✅ [RELOAD SUCCESS] Shard 3 weights (1.5B token brain) transferred smoothly to Shard 4!", flush=True)
    else:
        if is_master:
            print(f"[WARNING] Prior weights not found at {args.resume_weights}, training from scratch!", flush=True)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if world_size > 1 else model

    param_groups = [
        {"params": [p for p in raw_model.parameters() if p.requires_grad], "weight_decay": 0.1}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    dataset = BinaryDataset(args.data_bin, seq_len=cfg.max_seq_len, rank=rank, world_size=world_size)
    tokens_per_step = args.micro_batch * args.grad_accum * cfg.max_seq_len * world_size

    if is_master:
        print("=" * 70, flush=True)
        print("TRAINING DORINA-V4 SHARD 4 STARTED (Dual T4 DDP)", flush=True)
        print(f"Steps: {args.max_steps:,} | Batch: {tokens_per_step:,} tok/step | Target: {tokens_per_step * args.max_steps:,} tokens", flush=True)
        print(f"Learning Rate: {args.lr} (STABLE WSD PHASE - Constant Appetite)", flush=True)
        print(f"Cumulative Tokens upon completion: 2.0 Billion Tokens", flush=True)
        print("=" * 70, flush=True)

    t0 = time.time()
    for step in range(args.max_steps):
        # Stable LR regime (with gentle 100-step warm-up to absorb new data distribution)
        if step < args.warmup_steps:
            cur_lr = args.lr * (step + 1) / args.warmup_steps
        else:
            cur_lr = args.lr  # CONSTANT STABLE LR

        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(args.grad_accum):
            x, y = dataset.get_batch(args.micro_batch, step * args.grad_accum + micro_step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            is_last = (micro_step == args.grad_accum - 1)
            ctx = model.no_sync() if (world_size > 1 and not is_last) else torch.enable_grad()

            with ctx:
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Regular logging
        if is_master and (step % 25 == 0 or step == args.max_steps - 1):
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tok_s = (tokens_per_step * (25 if step > 0 else 1)) / max(dt, 1e-4)
            vram_gb = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"[SHARD4] Step {step:5d}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {cur_lr:.2e} | Speed: {tok_s:,.0f} tok/s | VRAM: {vram_gb:.2f} GB | Norm: {grad_norm:.2f}", flush=True)

        # Periodic checkpointing
        if is_master and step > 0 and (step % 500 == 0 or step == args.max_steps - 1):
            ckpt_path = os.path.join(args.out_dir, f"dorina_v4_shard4_step_{step}.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"  [CHECKPOINT] Periodic state saved: {ckpt_path}", flush=True)

    if is_master:
        final_save_path = os.path.join(args.out_dir, "dorina_v4_shard4_final.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(raw_model.state_dict(), final_save_path)
        print(f"\n=================================================================", flush=True)
        print(f"[SHARD4 COMPLETE] Model weights successfully saved to: {final_save_path}", flush=True)
        print(f"=================================================================", flush=True)

# =====================================================================
# 4. ENTRYPOINT & TORCHRUN DISPATCHER
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Internal worker flag for torchrun")
    parser.add_argument("--data-bin", type=str, default="/kaggle/working/v4_shard4.bin")
    parser.add_argument("--resume-weights", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/models_v4")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3800)  # 3800 * 131072 = 498,073,600 tokens (~500M)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5.5e-4)     # STABLE LR
    args = parser.parse_args()

    # --- Mode A: Subprocess DDP Worker ---
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
            rank = 0
            local_rank = 0
            world_size = 1
            device = "cuda" if torch.cuda.is_available() else "cpu"
            is_master = True

        worker_train(rank, local_rank, world_size, is_master, device, args)
        if ddp:
            dist.destroy_process_group()
        return

    # --- Mode B: Kaggle Master Launcher ---
    print("=" * 70, flush=True)
    print("DORINA V4 AUTONOMOUS MASTER TRAINER (SHARD 4 - 500M TOKENS)", flush=True)
    print("Target Architecture: 144.7M Dense Transformer (2048 Context, GQA 3:1)", flush=True)
    print("Target Hardware: Dual Tesla T4 (PyTorch DDP)", flush=True)
    print("=" * 70, flush=True)

    n_gpus = torch.cuda.device_count()
    print(f"[Hardware Scan] Detected {n_gpus} GPU(s) available.", flush=True)

    tokenizer_path = None
    resume_path = None

    print("[Filesystem Scan] Listing /kaggle/input...", flush=True)
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  Directory: {root} | files: {files}", flush=True)
            for f in files:
                full_p = os.path.join(root, f)
                if "tokenizer" in f.lower() and f.endswith(".json") and not tokenizer_path:
                    tokenizer_path = full_p
                if "shard3" in f.lower() and f.endswith(".pt") and not resume_path:
                    resume_path = full_p

    print(f"[Resource Check] Tokenizer: {tokenizer_path}", flush=True)
    print(f"[Resource Check] Base Weights (Shard 3): {resume_path}", flush=True)

    if not tokenizer_path or not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found! Resolved: {tokenizer_path}")

    # 1. Download & Prepare Shard 4 Multi-Curriculum Binary
    bin_file = "/kaggle/working/v4_shard4.bin"
    prepare_shard4_curriculum(tokenizer_path, bin_file)

    # 2. Launch DDP with torchrun
    print(f"\n>>> LAUNCHING SHARD 4 DDP TRAINING (Dual T4)...", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

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

    ret = subprocess.run(cmd)
    print(f"\n[Training Finished] Process exit code: {ret.returncode}", flush=True)
    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()
