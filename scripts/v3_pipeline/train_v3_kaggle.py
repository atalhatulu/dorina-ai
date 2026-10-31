import os
import sys
import time
import math
import argparse
from dataclasses import replace
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# model_moe_v3 dosyasını doğrudan import ediyoruz
from model_moe_v3 import DorinaMoEV3, MoEConfig, get_amp_precision

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if world_size > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank
            )
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class ShardedDataset:
    def __init__(self, bin_path, seq_len, rank, world_size):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        
        # Binary dosyayı uint16 memory-mapped olarak aç
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        total_tokens = len(self.data)
        
        # Toplam sequence sayısı (her sequence seq_len + 1 token içerir; x ve y için)
        self.total_seqs = total_tokens // (seq_len + 1)
        
        # Her rank için sıralı shard dilimi
        self.seqs_per_rank = self.total_seqs // world_size
        self.start_seq = rank * self.seqs_per_rank
        self.end_seq = self.start_seq + self.seqs_per_rank

        if rank == 0:
            print(f"[Dataset] Toplam Token: {total_tokens:,} | Toplam Seq ({seq_len}): {self.total_seqs:,} | Rank Başına Seq: {self.seqs_per_rank:,}")

    def get_batch(self, micro_batch, step):
        idx_in_rank = (step * micro_batch) % (self.seqs_per_rank - micro_batch)
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

def get_lr(step, max_steps, warmup_steps, max_lr, min_lr):
    # WSD Programı: Warmup -> Stable -> Decay
    # Claude & Dorina mimarisi: %5 warmup, %80 stable, %15 cosine decay
    decay_start = int(max_steps * 0.85)
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step < decay_start:
        return max_lr
    else:
        decay_steps = max_steps - decay_start
        current_decay = step - decay_start
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * current_decay / decay_steps))
        return min_lr + (max_lr - min_lr) * cosine_decay

def main():
    parser = argparse.ArgumentParser(description="Dorina V3 DDP Training")
    parser.add_argument("--data-train", type=str, default="/kaggle/input/dorina-v3-dataset/v3_shard1_200m_train.bin")
    parser.add_argument("--data-val", type=str, default="/kaggle/input/dorina-v3-dataset/v3_shard1_val.bin")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/checkpoints")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--init-weights", type=str, default="", help="Önceki shard'dan devam etmek için ağırlık dosyası")
    parser.add_argument("--smoke-test", action="store_true", help="10 adımlık doğrulama modu")
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_distributed()
    is_master = (rank == 0)


    if args.smoke_test:
        args.max_steps = 10
        args.eval_interval = 5
        args.save_interval = 10
        if is_master:
            print("[SMOKE TEST] 10 adımlık doğrulama modu devrede!")

    # Toplam efektif batch büyüklüğü: micro_batch * world_size * grad_accum * 1024 token
    tokens_per_step = args.micro_batch * world_size * args.grad_accum * 1024
    if is_master:
        print("=" * 70)
        print(f"Dorina V3 Training | World Size: {world_size} GPUs")
        print(f"Micro-batch: {args.micro_batch} | Grad Accum: {args.grad_accum}")
        print(f"Global Batch: {tokens_per_step:,} tokens/step")
        print(f"Max Steps: {args.max_steps:,} (Hedef ~{args.max_steps * tokens_per_step / 1e6:.1f}M token)")
        print("=" * 70)
        os.makedirs(args.out_dir, exist_ok=True)

    # 1. Model ve Precision
    cfg = MoEConfig(
        vocab_size=32768,
        n_layer=12,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=1024,
        n_experts=32,
        top_k=2,
        expert_hidden=256,
        shared_hidden=256,
        capacity_factor=1.25,
        aux_loss_coef=0.01,
        z_loss_coef=1e-3
    )

    model = DorinaMoEV3(cfg).to(device)
    amp_dtype, use_scaler = get_amp_precision()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if args.init_weights and os.path.exists(args.init_weights):
        if is_master:
            print(f"[Ağırlık Devralma] Önceki Shard ağırlıkları yükleniyor: {args.init_weights}")
        sd = torch.load(args.init_weights, map_location=device, weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
        if is_master:
            print("  ✅ Önceki Shard ağırlıkları başarıyla yüklendi! Eğitim kaldığı yerden devam ediyor.")

    if is_master:
        print(f"[Hardware] Device: {device} | AMP Precision: {amp_dtype} | GradScaler: {use_scaler}")


    if world_size > 1:
        # Claude uyarısı: Experts tek 3D tensörde olduğu için find_unused_parameters=False
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )

    # Optimizer (Weight decay sadece d >= 2 için uygulanır)
    raw_model = model.module if world_size > 1 else model
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, lr=args.lr, device_type="cuda")

    # 2. Veri Yükleyici
    train_dataset = ShardedDataset(args.data_train, seq_len=1024, rank=rank, world_size=world_size)
    val_dataset = ShardedDataset(args.data_val, seq_len=1024, rank=rank, world_size=world_size)

    # 3. Eğitim Döngüsü
    model.train()
    step = 0
    t0 = time.time()

    for step in range(args.max_steps):
        # LR güncellemesi
        lr = get_lr(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(args.grad_accum):
            x, y = train_dataset.get_batch(args.micro_batch, step * args.grad_accum + micro_step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Claude uyarısı: PCIe üzerinde gereksiz gradyan senkronizasyonunu engelle
            is_last_micro_step = (micro_step == args.grad_accum - 1)
            
            if world_size > 1 and not is_last_micro_step:
                context = model.no_sync()
            else:
                context = torch.enable_grad()

            with context:
                with torch.autocast("cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

        # Gradyan kırpma ve optimizer adımı
        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()

        # Loglama
        if is_master and (step % 10 == 0 or step == args.max_steps - 1):
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tok_s = (tokens_per_step * 10) / dt if step > 0 else tokens_per_step / max(dt, 1e-4)
            vram_alloc = torch.cuda.memory_allocated(device) / 1024**3
            vram_res = torch.cuda.memory_reserved(device) / 1024**3
            scale_info = f" | Scale: {scaler.get_scale():.0f}" if use_scaler else ""
            print(f"Adım {step:5d}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Hız: {tok_s:,.0f} tok/s | VRAM: {vram_alloc:.2f}GB (Rezerve: {vram_res:.2f}GB){scale_info}")

        # MoE Metrikleri
        if is_master and (step % 50 == 0 and step > 0):
            metrics = raw_model.moe_metrics()
            print(f"  [MoE Sağlık] Düşme: %{metrics['dropped_pct']:.2f} | Ölü Uzman: {metrics['dead_experts']} | Router Entropi: {metrics['router_entropy']:.2f}")

        # Eval
        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            eval_batches = 10
            with torch.no_grad():
                for v_step in range(eval_batches):
                    vx, vy = val_dataset.get_batch(args.micro_batch, v_step)
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=amp_dtype):
                        _, l = raw_model(vx, vy)
                        val_loss += l.item()
            val_loss /= eval_batches
            if is_master:
                print(f"  >>> [EVAL Adım {step}] Val Loss: {val_loss:.4f}")
            model.train()

        # Checkpoint (Claude uyarısı: Sadece rank 0 ve model.module olmadan temiz kaydet)
        if is_master and (step > 0 and step % args.save_interval == 0 or step == args.max_steps - 1):
            ckpt_path = os.path.join(args.out_dir, f"dorina_v3_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_scaler else None,
                "loss": accum_loss,
                "cfg": cfg
            }, ckpt_path)
            print(f"  [CHECKPOINT] Kaydedildi: {ckpt_path}")

    if is_master:
        final_path = os.path.join(args.out_dir, "dorina_v3_shard1_final.pt")
        torch.save(raw_model.state_dict(), final_path)
        print(f"\n[BİTTİ] Shard 1 eğitimi tamamlandı! Ağırlıklar kaydedildi: {final_path}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
