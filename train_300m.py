import os
import sys
import time
import math
import glob
import numpy as np
import torch
import tiktoken

# Proje kök dizinini Python yoluna ekle
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Windows terminal utf-8 uyumluluğu
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.model import DorinaGPT
from src.utils.logger import LabLogger

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# 1. Dorina-300M Mimari & Hiperparametre Konfigürasyonu
# -----------------------------------------------------------------------------
CONFIG = {
    "name": "dorina_v0.14_300m",
    "phase": "phase1_core",
    "n_layer": 18,
    "n_embd": 1024,
    "n_head": 16,
    "context_length": 512,
    "vocab_size": 50257,
    "micro_batch_size": 4,          # RTX 4050 (6GB) VRAM sınırını aşmamak için (~4.1 GB VRAM)
    "grad_accum_steps": 4,          # Etkili batch size = 4 * 4 = 16 (16 * 512 = 8.192 token/adım)
    "max_iters": 12000,             # 12.000 adım * 8.192 token ≈ ~100 Milyon Token ön-eğitim
    "eval_interval": 1200,          # Her 1.200 adımda bir (%10'luk periyotlarda) değerlendirme
    "eval_iters": 20,
    "lr": 4.0e-4,                   # 300M model için ideal cosine tepe öğrenme oranı
    "min_lr": 4.0e-5,
    "warmup_iters": 600,            # İlk %5 adım boyunca ısınma (warmup)
    "weight_decay": 0.1,
    "dropout": 0.05
}

# -----------------------------------------------------------------------------
# 2. Veri Kümesi Kontrolü (Aşama 1: Cosmopedia-v2)
# -----------------------------------------------------------------------------
train_bin = os.path.join(ROOT_DIR, "data", "dorina300m_phase1_train.bin")
val_bin = os.path.join(ROOT_DIR, "data", "dorina300m_phase1_val.bin")

if not (os.path.exists(train_bin) and os.path.exists(val_bin)):
    print("\n⚠️ 300M Aşama 1 veri seti (.bin) bulunamadı!")
    print("Lütfen önce 'python scripts/prepare_300m_phase1.py' çalıştırın.")
    sys.exit(1)

train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')

print("\n" + "=" * 70)
print("🔬 DORINA LAB: 300M BÜYÜK DİL MODELİ (AŞAMA 1 ÖN-EĞİTİM)")
print("=" * 70)
print(f"📦 Veri Seti: Cosmopedia-v2")
print(f"   • Train Tokens: {len(train_data):,} (~{len(train_data)/1e6:.1f}M Token)")
print(f"   • Val Tokens:   {len(val_data):,} (~{len(val_data)/1e6:.1f}M Token)")

# -----------------------------------------------------------------------------
# 3. Model Başlatma (Sıfırdan - Scratch)
# -----------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
model = DorinaGPT(
    vocab_size=CONFIG["vocab_size"],
    context_length=CONFIG["context_length"],
    n_embd=CONFIG["n_embd"],
    n_head=CONFIG["n_head"],
    n_layer=CONFIG["n_layer"],
    dropout=CONFIG["dropout"]
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"⚡ Model Mimarisi: {CONFIG['n_layer']} Katman | {CONFIG['n_embd']} Vektör | {CONFIG['n_head']} Kafa | {CONFIG['context_length']} Bağlam")
print(f"🚀 Toplam Parametre Sayısı: {total_params:,} (~{total_params/1e6:.1f}M Parametre)")

# -----------------------------------------------------------------------------
# 4. Optimizer, Scheduler ve Gradyan Yöneticisi
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    betas=(0.9, 0.95),
    weight_decay=CONFIG["weight_decay"]
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

# Sabit Değerlendirme Batch İndeksleri
g_val = torch.Generator().manual_seed(42)
val_indices = torch.randint(
    len(val_data) - CONFIG["context_length"] - 1,
    (CONFIG["eval_iters"], CONFIG["micro_batch_size"]),
    generator=g_val
)

def get_batch(split, k=0):
    d = train_data if split == 'train' else val_data
    ctx = CONFIG["context_length"]
    bsz = CONFIG["micro_batch_size"]
    
    if split == 'train':
        ix = torch.randint(len(d) - ctx - 1, (bsz,))
    else:
        ix = val_indices[k]
        
    x = torch.stack([torch.from_numpy(d[i:i+ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i+1:i+ctx+1].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
# 5. LabLogger ve Run Takibi
# -----------------------------------------------------------------------------
logger = LabLogger(
    model_name=CONFIG["name"],
    config={
        **CONFIG,
        "total_parameters": total_params,
        "mode": "scratch_pretrain",
        "precision": "fp16_mixed" if device == "cuda" else "fp32"
    },
    total_steps=CONFIG["max_iters"]
)

# -----------------------------------------------------------------------------
# 6. Ana Ön-Eğitim Döngüsü (Gradient Accumulation Destekli)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"🚀 DORINA-300M ÖN-EĞİTİMİ BAŞLADI (Hedef: {CONFIG['max_iters']:,} Adım)")
print(f"   • Etkili Batch: {CONFIG['micro_batch_size'] * CONFIG['grad_accum_steps']} ({CONFIG['micro_batch_size']} micro x {CONFIG['grad_accum_steps']} accum)")
print(f"   • Adım Başına Token: {CONFIG['micro_batch_size'] * CONFIG['grad_accum_steps'] * CONFIG['context_length']:,}")
print("=" * 70)

best_val_loss = float('inf')
accum_steps = CONFIG["grad_accum_steps"]

for iter_step in range(1, CONFIG["max_iters"] + 1):
    lr = get_lr(iter_step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    # Gradient Accumulation Döngüsü
    for micro_step in range(accum_steps):
        xb, yb = get_batch('train')
        with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.float16, enabled=(device=="cuda")):
            _, loss = model(xb, targets=yb)
            # Kaybı accum_steps sayısına bölerek biriktir
            loss = loss / accum_steps
            accum_loss += loss.item()

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # -------------------------------------------------------------------------
    # Değerlendirme & Kayıt Aşaması (Her %10 ve 1. adımda)
    # -------------------------------------------------------------------------
    if iter_step == 1 or iter_step % CONFIG["eval_interval"] == 0 or iter_step == CONFIG["max_iters"]:
        losses = estimate_loss()

        # Güncel checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": iter_step,
            "val_loss": losses['val'],
            "config": CONFIG
        }, os.path.join(logger.runs_dir, "model.pt"))

        # Rekor kırıldıysa en iyi anı dondur
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": iter_step,
                "val_loss": losses['val'],
                "config": CONFIG
            }, os.path.join(logger.runs_dir, "best.pt"))

        # Modelden canlı test örneği üret
        test_prompt = "In science, the law of gravity explains"
        prompt_tokens = enc.encode_ordinary(test_prompt)
        ctx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        out_tokens = model.generate(ctx, max_new_tokens=40, temperature=0.7)[0].tolist()
        sample_text = enc.decode(out_tokens)

        # İşlenen kümülatif token hacmi
        if iter_step == 1:
            step_tokens = CONFIG["micro_batch_size"] * accum_steps * CONFIG["context_length"]
        else:
            step_tokens = CONFIG["micro_batch_size"] * accum_steps * CONFIG["context_length"] * CONFIG["eval_interval"]

        logger.log(
            step=iter_step,
            train_loss=losses['train'],
            val_loss=losses['val'],
            grad_norm=None,
            lr=lr,
            batch_tokens=step_tokens,
            sample_text=sample_text
        )

# -----------------------------------------------------------------------------
# 7. Final Raporlama ve Release Paketi
# -----------------------------------------------------------------------------
logger.finish(
    model=model,
    encode_fn=enc.encode_ordinary,
    decode_fn=enc.decode,
    test_prompts=[
        "In science, the law of gravity explains",
        "The solar system consists of the Sun and",
        "During the Renaissance, art and culture flourished in"
    ],
    device=device
)

print("\n" + "=" * 70)
print(f"🎉 DORINA-300M AŞAMA 1 EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
print(f"   • En İyi Val Loss: {best_val_loss:.4f}")
print(f"   • Kayıt Dizini: {logger.runs_dir}")
print("=" * 70)
