import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from model_dense_v4 import DorinaDenseV4, DenseConfig

class BinaryDataset:
    def __init__(self, bin_path, seq_len=2048, rank=0, world_size=1):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
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
            
        x = torch.stack(batch_x)
        y = torch.stack(batch_y)
        return x, y

def main():
    parser = argparse.ArgumentParser(description="Dorina V4 Dense DDP Trainer / Speed Benchmark")
    parser.add_argument("--data-train", type=str, required=True)
    parser.add_argument("--data-val", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./checkpoints_v4")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    args = parser.parse_args()

    # DDP Setup
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

    if is_master:
        print("=" * 65)
        print("Dorina V4 Dense Pilot Engine Started")
        print(f"Device: {device} (World Size: {world_size}) | FP16 Scaler Active")
        print(f"Context: 2048 | Micro-batch: {args.micro_batch} | Grad-accum: {args.grad_accum}")
        tokens_per_step = args.micro_batch * args.grad_accum * world_size * 2048
        print(f"Tokens Per Step: {tokens_per_step:,} Tokens")
        print("=" * 65)

    os.makedirs(args.out_dir, exist_ok=True)

    # Dataset
    train_dataset = BinaryDataset(args.data_train, seq_len=2048, rank=rank, world_size=world_size)
    val_dataset = BinaryDataset(args.data_val, seq_len=2048, rank=rank, world_size=world_size)

    # Model
    cfg = DenseConfig(vocab_size=32768, max_seq_len=2048, n_layers=16, d_model=768, n_heads=12, n_kv_heads=4)
    model = DorinaDenseV4(cfg).to(device)

    # FP16 Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module

    t0 = time.time()
    step_times = []

    for step in range(args.max_steps):
        # WSD LR Scheduler
        if step < args.warmup_steps:
            current_lr = args.lr * (step + 1) / args.warmup_steps
        else:
            current_lr = args.lr
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(args.grad_accum):
            x, y = train_dataset.get_batch(args.micro_batch, step * args.grad_accum + micro_step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            is_last_micro_step = (micro_step == args.grad_accum - 1)
            context = model.no_sync() if (ddp and not is_last_micro_step) else torch.enable_grad()

            with context:
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum

                scaler.scale(loss).backward()
                accum_loss += loss.item()

        # Unscale & Clip
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()

        if is_master and (step % 5 == 0 or step == args.max_steps - 1):
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            step_times.append(dt)
            tok_s = (tokens_per_step * (5 if step > 0 else 1)) / max(dt, 1e-4)
            vram_gb = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"Step {step:4d}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {current_lr:.2e} | Speed: {tok_s:,.0f} tok/s | VRAM: {vram_gb:.2f} GB | Norm: {grad_norm:.2f}")

    if is_master:
        print("\n" + "=" * 65)
        print("PILOT BENCHMARK COMPLETE!")
        avg_speed = tokens_per_step / (sum(step_times[-10:]) / len(step_times[-10:]))
        print(f"Average Sustained Speed: {avg_speed:,.0f} tokens/second")
        print("Ready for full Kaggle deployment upon user signal.")
        print("=" * 65)

if __name__ == "__main__":
    main()
