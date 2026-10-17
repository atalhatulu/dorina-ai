"""
================================================================================
DORINA-MoE: ANA EĞİTİM MOTORU (train.py)
================================================================================
Amaç:
  Bu script, Dorina-MoE (Sparse Mixture of Experts, 4 Uzman, Top-2 Routing)
  mimarisinin büyük gövde pre-train ve devam eğitimlerini yürüten ana motorudur.

Özellikler & Konfigürasyon:
  - Mimari: 6 Katman, 384 Boyut, 6 Başlık, RoPE, SwiGLU, SDPA Attention
  - Parametre: ~44M Toplam, ~30M Aktif (Router Z-Loss destekli)
  - Mod: Turbo Mod (Micro-batch: 12, ~24.700 tok/s, VRAM ~4.2GB)
  - Çıktı Rotası: models/dorina_44m_moe/runX/ altında model.pt, plots/ ve run_info.json
  - Otomasyon: Eğitim bitiminde 100 puanlık Benchmark sınavını otomatik tetikler.
================================================================================
"""

import os
import sys
import time
import math
import numpy as np
import torch
from tokenizers import Tokenizer

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.model_moe import DorinaMoEGPT
from src.utils.logger import LabLogger

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# 1. Dorina-MoE Aşama 2 (Shard 1 - 100M Turbo) Konfigürasyonu
# -----------------------------------------------------------------------------
# Turbo Mod: micro_batch_size = 12 (Batch 16'ya yakın, VRAM 3.6 GB, %100 güvenli hız)
CONFIG = {
    "name": "dorina_v0.31_moe_stage2_shard1",
    "phase": "stage2_body_shard1_100m",
    "n_layer": 6,
    "n_embd": 384,
    "n_head": 6,
    "context_length": 512,
    "vocab_size": 32768,
    "num_experts": 4,
    "top_k": 2,
    "micro_batch_size": 12,        # Turbo mod: Adım başına 6,144 token
    "grad_accum_steps": 1,
    "max_iters": 16000,            # 16.000 adım = ~98.3M token (~48 dakika)
    "eval_interval": 500,          # Her 500 adımda bir tam denetim
    "eval_iters": 25,
    "lr": 2.5e-4,                  # Büyük gövde öğrenme oranı
    "min_lr": 2.5e-5,
    "warmup_iters": 300,
    "weight_decay": 0.1,
    "dropout": 0.05,
    "max_loss_spike_factor": 2.5
}

# -----------------------------------------------------------------------------
# 2. Veri Kümelerini Memmap Olarak Bağlama
# -----------------------------------------------------------------------------
train_bin_path = os.path.join(ROOT_DIR, "data", "stage2_shard1_100m_train.bin")
val_bin_path   = os.path.join(ROOT_DIR, "data", "stage2_shard1_100m_val.bin")

print("\n" + "=" * 70)
print("🦀 DORINA-MoE: AŞAMA 2 - SHARD 1 (100M TURBO EĞİTİMİ)")
print("=" * 70)

if not (os.path.exists(train_bin_path) and os.path.exists(val_bin_path)):
    print("⏳ Shard 1 Veri seti derlemesi bekleniyor...")
    while not (os.path.exists(train_bin_path) and os.path.exists(val_bin_path)):
        time.sleep(3)

train_data = np.memmap(train_bin_path, dtype=np.uint16, mode='r')
val_data   = np.memmap(val_bin_path, dtype=np.uint16, mode='r')

print(f"📦 Train Token: {len(train_data):,} (~{len(train_data)/1e6:.1f}M)")
print(f"📦 Val Token:   {len(val_data):,} (~{len(val_data)/1e6:.1f}M)")

# -----------------------------------------------------------------------------
# 3. Model Başlatma ve Run 4'ten Devam (Warm Start)
# -----------------------------------------------------------------------------
tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)

model = DorinaMoEGPT(
    vocab_size=CONFIG["vocab_size"],
    context_length=CONFIG["context_length"],
    n_embd=CONFIG["n_embd"],
    n_head=CONFIG["n_head"],
    n_layer=CONFIG["n_layer"],
    num_experts=CONFIG["num_experts"],
    top_k=CONFIG["top_k"],
    dropout=CONFIG["dropout"]
).to(device)

prev_ckpt_path = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "run4", "model.pt")
if os.path.exists(prev_ckpt_path):
    print(f"🔄 Run 4'teki olgun ağırlıklar yükleniyor: {prev_ckpt_path}")
    ckpt = torch.load(prev_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print("   ✓ 160M Token'lık güçlü Aşama 1 temeli başarıyla devralındı!")
else:
    print("⚠️ Checkpoint bulunamadı, sıfırdan başlanıyor.")

total_params = sum(p.numel() for p in model.parameters())
shared_params = sum(p.numel() for p in model.tok_emb.parameters()) + sum(p.numel() for p in model.ln_f.parameters())
for b in model.blocks:
    shared_params += sum(p.numel() for p in b.attn.parameters())
    shared_params += sum(p.numel() for p in b.ln1.parameters())
    shared_params += sum(p.numel() for p in b.ln2.parameters())
    shared_params += sum(p.numel() for p in b.moe.router.parameters())
    for exp in b.moe.experts[:CONFIG["top_k"]]:
        shared_params += sum(p.numel() for p in exp.parameters())

print(f"⚡ MİMARİ: {total_params:,} Parametre ({shared_params:,} Aktif)")

# -----------------------------------------------------------------------------
# 4. Tier-1 Optimizer (Weight Decay Exclusions) & Batch Loader
# -----------------------------------------------------------------------------
optimizer = model.configure_optimizers(
    weight_decay=CONFIG["weight_decay"],
    learning_rate=CONFIG["lr"],
    betas=(0.9, 0.95),
    device_type=device
)
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

def get_lr(it):
    warmup = CONFIG["warmup_iters"]
    if it < warmup:
        return CONFIG["lr"] * (it + 1) / (warmup + 1)
    if it > CONFIG["max_iters"]:
        return CONFIG["min_lr"]
    decay_ratio = (it - warmup) / (CONFIG["max_iters"] - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return CONFIG["min_lr"] + coeff * (CONFIG["lr"] - CONFIG["min_lr"])

g_val = np.random.RandomState(42)
val_indices = g_val.randint(
    0,
    len(val_data) - CONFIG["context_length"] - 1,
    size=(CONFIG["eval_iters"], CONFIG["micro_batch_size"])
)

def get_batch(split, k=0):
    ctx = CONFIG["context_length"]
    bsz = CONFIG["micro_batch_size"]
    d = train_data if split == 'train' else val_data
    if split == 'train':
        ix = np.random.randint(0, len(d) - ctx - 1, size=(bsz,))
    else:
        ix = val_indices[k]
    x_np = np.stack([d[i : i + ctx] for i in ix])
    y_np = np.stack([d[i + 1 : i + ctx + 1] for i in ix])
    x = torch.from_numpy(x_np.astype(np.int64)).to(device, non_blocking=True)
    y = torch.from_numpy(y_np.astype(np.int64)).to(device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(CONFIG["eval_iters"])
        for k in range(CONFIG["eval_iters"]):
            X, Y = get_batch(split, k)
            with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.float16, enabled=(device=="cuda")):
                _, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# -----------------------------------------------------------------------------
# 5. Model Çıktı Klasörü: Doğrudan models/dorina_44m_moe/run5/
# -----------------------------------------------------------------------------
RUN5_DIR = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "run5")
os.makedirs(RUN5_DIR, exist_ok=True)

logger = LabLogger(
    model_name="dorina_44m_moe",
    run_tag="run5",
    config={
        **CONFIG,
        "total_parameters": total_params,
        "active_parameters": shared_params,
        "mode": "stage2_shard1_100m",
        "precision": "fp16_mixed" if device == "cuda" else "fp32"
    },
    total_steps=CONFIG["max_iters"],
    runs_dir=RUN5_DIR
)

print("\n" + "=" * 70)
print(f"🚀 DORINA-MoE SHARD 1 EĞİTİMİ BAŞLADI (Hedef: {CONFIG['max_iters']:,} Adım)")
print(f"   • Hız: Turbo Mod (Batch 12, Adım Başına 6,144 Token)")
print(f"   • Beklenen Süre: ~45 - 48 Dakika")
print("=" * 70)

best_val_loss = float('inf')
rolling_losses = []
last_healthy_state = None

for iter_step in range(1, CONFIG["max_iters"] + 1):
    lr = get_lr(iter_step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    xb, yb = get_batch('train')

    with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.float16, enabled=(device=="cuda")):
        _, loss = model(xb, targets=yb)

    loss_val = loss.item()
    
    if math.isnan(loss_val) or math.isinf(loss_val):
        print(f"\n🚨 [SpikeGuard] NaN/Inf! Son sağlıklı duruma dönülüyor...")
        if last_healthy_state is not None:
            model.load_state_dict(last_healthy_state)
        continue

    if len(rolling_losses) > 20:
        avg_loss = sum(rolling_losses[-20:]) / 20.0
        if loss_val > avg_loss * CONFIG["max_loss_spike_factor"]:
            print(f"\n⚠️ [SpikeGuard] Anormal sıçrama ({loss_val:.2f} vs ort. {avg_loss:.2f})! Adım atlanıyor.")
            optimizer.zero_grad(set_to_none=True)
            continue

    rolling_losses.append(loss_val)
    if len(rolling_losses) > 100:
        rolling_losses.pop(0)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if iter_step == 1 or iter_step % CONFIG["eval_interval"] == 0 or iter_step == CONFIG["max_iters"]:
        losses = estimate_loss()
        current_val = losses['val']
        last_healthy_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        torch.save({
            "model_state_dict": model.state_dict(),
            "step": iter_step,
            "val_loss": current_val,
            "config": CONFIG
        }, os.path.join(RUN5_DIR, "model.pt"))

        if current_val < best_val_loss:
            best_val_loss = current_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": iter_step,
                "val_loss": current_val,
                "config": CONFIG
            }, os.path.join(RUN5_DIR, "best.pt"))

        test_prompt = "def quick_sort(arr):\n"
        prompt_tokens = tokenizer.encode(test_prompt).ids
        ctx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            out_tokens = model.generate(ctx, max_new_tokens=40, temperature=0.6)[0].tolist()
        sample_text = tokenizer.decode(out_tokens, skip_special_tokens=False)

        step_tokens = CONFIG["micro_batch_size"] * CONFIG["context_length"] if iter_step == 1 else CONFIG["micro_batch_size"] * CONFIG["context_length"] * CONFIG["eval_interval"]

        logger.log(
            step=iter_step,
            train_loss=losses['train'],
            val_loss=current_val,
            grad_norm=None,
            lr=lr,
            batch_tokens=step_tokens,
            sample_text=sample_text
        )

        # ---------------------------------------------------------------------
        # HER ~20M TOKENDE BİR (%20) PERİYODİK BENCHMARK SINAVI (~3.200 Adım)
        # ---------------------------------------------------------------------
        if iter_step % 3200 == 0:
            print("\n" + "=" * 75)
            print(f"🎯 [CANLI ZEKA TESTİ] {iter_step:,}. Adım (~{(iter_step * CONFIG['micro_batch_size'] * CONFIG['context_length'])/1e6:.1f}M Token) 100 Puanlık Sınav Başlatılıyor...")
            print("=" * 75)
            from scripts.benchmark_suite import run_benchmark
            snap_path = os.path.join(RUN5_DIR, f"model_step_{iter_step}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": iter_step,
                "val_loss": current_val,
                "config": CONFIG
            }, snap_path)
            bench_out = os.path.join(RUN5_DIR, f"benchmark_step_{iter_step}.json")
            try:
                rep = run_benchmark(snap_path, bench_out)
                print(f"🏆 {iter_step:,}. Adım Sınav Skoru: {rep['total_score']} / 100 Puan")
                print(f"   • Kod Mantığı: {rep['category_scores']['Code Logic']}/25 | Bilim: {rep['category_scores']['Science Knowledge']}/25")
            except Exception as e:
                print(f"⚠️ Periyodik sınav hatası: {e}")
            print("=" * 75 + "\n")

# -----------------------------------------------------------------------------
# 6. Final Kapanış & Run Info Manifest
# -----------------------------------------------------------------------------
logger.finish(
    model=model,
    encode_fn=lambda s: tokenizer.encode(s).ids,
    decode_fn=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
    test_prompts=[
        "def quick_sort(arr):\n",
        "def binary_search(arr, target):\n",
        "<|im_start|>user\nWrite a python function to check if a string is palindrome.<|im_end|>\n<|im_start|>assistant\n"
    ],
    device=device
)

import json
run_info = {
    "model_architecture": "Dorina-MoE (Sparse Mixture of Experts)",
    "total_parameters": 44452608,
    "active_parameters": 30296832,
    "run_id": "run5",
    "phase_name": "Aşama 2: Büyük Gövde - Shard 1 (İleri Algoritma & Matematik)",
    "tokens_in_this_run": 98304000,
    "cumulative_tokens": 258048000,
    "target_tokens": 1000000000,
    "steps_completed": CONFIG["max_iters"],
    "best_val_loss": best_val_loss,
    "status": "SUCCESS"
}
with open(os.path.join(RUN5_DIR, "run_info.json"), "w", encoding="utf-8") as f:
    json.dump(run_info, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print(f"🎉 AŞAMA 2 - SHARD 1 EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
print(f"   • En İyi Val Loss: {best_val_loss:.4f}")
print("=" * 70)

# Otomatik 100 Puanlık Benchmark Sınavı
print("\n🏛️ EĞİTİM SONRASI 100 PUANLIK BENCHMARK SINAVI BAŞLATILIYOR...")
from scripts.benchmark_suite import run_benchmark
model_saved_path = os.path.join(RUN5_DIR, "model.pt")
benchmark_out = os.path.join(RUN5_DIR, "benchmark_score.json")
if os.path.exists(model_saved_path):
    run_benchmark(model_saved_path, benchmark_out)
