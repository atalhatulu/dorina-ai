import os
import sys
import time
import math
import glob
import numpy as np
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader

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
# 1. Checkpoint Arama ve Seçim Menüsü
# -----------------------------------------------------------------------------
def get_base_checkpoint():
    candidates = []
    
    # 1. Releases altındaki resmi model
    rel_model = os.path.join(ROOT_DIR, "releases", "dorina_v0.13_70m", "model.pt")
    if os.path.exists(rel_model):
        candidates.append(rel_model)
        
    # 2. Runs altındaki en son best.pt / model.pt'ler
    run_pts = glob.glob(os.path.join(ROOT_DIR, "runs", "dorina_v0.13_70m", "*", "best.pt")) + \
              glob.glob(os.path.join(ROOT_DIR, "runs", "dorina_v0.13_70m", "*", "model.pt"))
    candidates.extend(sorted(list(set(run_pts))))

    if not candidates:
        print("❌ SFT yapılacak önceden eğitilmiş hiçbir 70M modeli bulunamadı!")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("🎯 DORINA-70M SFT: ÖNCEDEN EĞİTİLMİŞ BAZ MODELİ SEÇİN")
    print("=" * 65)
    for idx, c in enumerate(candidates, 1):
        rel = os.path.relpath(c, ROOT_DIR)
        size_mb = os.path.getsize(c) / (1024 * 1024)
        print(f" [{idx}] ➔ {rel} ({size_mb:.1f} MB)")
    
    choice = input(f"\nModel seçimi [Varsayılan sonuncu: {len(candidates)}]: ").strip()
    idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(candidates) else len(candidates) - 1
    return candidates[idx]

# -----------------------------------------------------------------------------
# 2. Hiperparametreler ve SFT Konfigürasyonu
# -----------------------------------------------------------------------------
CFG = {
    "name": "dorina_v0.13_70m_sft",
    "base_model": "dorina_v0.13_70m",
    "n_layer": 10,
    "n_embd": 512,
    "n_head": 8,
    "context_length": 256,
    "vocab_size": 50257,
    "batch_size": 16,
    "max_iters": 2500,          # ~10.800 örnek / 16 = 675 adım/epoch -> ~3.7 epoch
    "eval_interval": 150,
    "eval_iters": 25,
    "lr": 2.5e-5,               # Pre-training'e göre daha düşük (Catastrophic forgetting önleme)
    "min_lr": 2.5e-6,
    "warmup_iters": 100,
    "weight_decay": 0.05,
    "dropout": 0.05
}

# -----------------------------------------------------------------------------
# 3. Veri Kümesi Kontrolü ve Yükleme (Yol B - Masked SFT)
# -----------------------------------------------------------------------------
train_npz = os.path.join(ROOT_DIR, "data", "sft_train_masked.npz")
val_npz = os.path.join(ROOT_DIR, "data", "sft_val_masked.npz")

if not (os.path.exists(train_npz) and os.path.exists(val_npz)):
    print("\n⚠️ Maskelenmiş SFT veri setleri (.npz) bulunamadı!")
    print("Lütfen önce 'python scripts/prepare_sft_data.py' çalıştırın.")
    sys.exit(1)

class MaskedSFTDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.inputs = torch.from_numpy(data["inputs"]).long()
        self.targets = torch.from_numpy(data["targets"]).long()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

train_dataset = MaskedSFTDataset(train_npz)
val_dataset = MaskedSFTDataset(val_npz)

train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False)

print(f"\n📦 SFT Veri Seti Yüklendi:")
print(f"   • Train Örnek: {len(train_dataset):,} adet")
print(f"   • Val Örnek:   {len(val_dataset):,} adet")

# -----------------------------------------------------------------------------
# 4. Model ve Checkpoint Yükleme
# -----------------------------------------------------------------------------
base_ckpt_path = get_base_checkpoint()
print(f"\n🚀 Baz model yükleniyor: {os.path.relpath(base_ckpt_path, ROOT_DIR)}")

enc = tiktoken.get_encoding("gpt2")
model = DorinaGPT(
    vocab_size=CFG["vocab_size"],
    context_length=CFG["context_length"],
    n_embd=CFG["n_embd"],
    n_head=CFG["n_head"],
    n_layer=CFG["n_layer"],
    dropout=CFG["dropout"]
).to(device)

raw_dict = torch.load(base_ckpt_path, map_location=device)
state_dict = raw_dict.get("model_state_dict", raw_dict.get("model", raw_dict))
# Çift sarmalanmış state_dict koruması
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

model.load_state_dict(state_dict, strict=False)
print("✓ Önceden eğitilmiş ağırlıklar başarıyla yüklendi!")

# -----------------------------------------------------------------------------
# 5. Optimizer, Loss (Ignore Index -100) ve Scheduler
# -----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], betas=(0.9, 0.95))
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

def get_lr(it):
    warmup = CFG["warmup_iters"]
    if it < warmup:
        return CFG["lr"] * (it + 1) / (warmup + 1)
    if it > CFG["max_iters"]:
        return CFG["min_lr"]
    decay_ratio = (it - warmup) / (CFG["max_iters"] - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return CFG["min_lr"] + coeff * (CFG["lr"] - CFG["min_lr"])

# -----------------------------------------------------------------------------
# 6. Değerlendirme (Evaluation)
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_sft_loss():
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    
    # Train kaybı tahmini
    t_losses = []
    for i, (xb, yb) in enumerate(train_loader):
        if i >= CFG["eval_iters"]: break
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.float16, enabled=(device=="cuda")):
            logits, _ = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        t_losses.append(loss.item())
    losses["train"] = float(np.mean(t_losses)) if t_losses else 0.0

    # Val kaybı tahmini
    v_losses = []
    for i, (xb, yb) in enumerate(val_loader):
        if i >= CFG["eval_iters"]: break
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.float16, enabled=(device=="cuda")):
            logits, _ = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        v_losses.append(loss.item())
    losses["val"] = float(np.mean(v_losses)) if v_losses else 0.0

    model.train()
    return losses

# -----------------------------------------------------------------------------
# 7. LabLogger Başlatma
# -----------------------------------------------------------------------------
logger = LabLogger(
    model_name=CFG["name"],
    config={
        **CFG,
        "base_checkpoint": os.path.relpath(base_ckpt_path, ROOT_DIR),
        "loss_method": "masked_cross_entropy_ignore_minus_100",
        "precision": "fp16_mixed" if device == "cuda" else "fp32"
    },
    total_steps=CFG["max_iters"]
)

# -----------------------------------------------------------------------------
# 8. SFT Eğitim Döngüsü
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("🚀 DORINA-70M SFT (INSTRUCTION TUNING) EĞİTİMİ BAŞLADI")
print(f"   • Yöntem: Masked Cross-Entropy (Kullanıcı sorusu yoksayılır, hedef ignore=-100)")
print(f"   • Toplam Adım: {CFG['max_iters']} | Batch Size: {CFG['batch_size']}")
print("=" * 65)

best_val_loss = float("inf")
train_iter = iter(train_loader)

for iter_step in range(1, CFG["max_iters"] + 1):
    # Dataloader döngüsü (epoch bittiğinde otomatik baştan başlar)
    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)

    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

    # LR Güncelle
    lr = get_lr(iter_step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.float16, enabled=(device=="cuda")):
        logits, _ = model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # Değerlendirme & Kayıt
    if iter_step == 1 or iter_step % CFG["eval_interval"] == 0 or iter_step == CFG["max_iters"]:
        losses = estimate_sft_loss()

        # Checkpoint kaydet
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": iter_step,
            "val_loss": losses["val"],
            "config": CFG
        }, os.path.join(logger.runs_dir, "model.pt"))

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": iter_step,
                "val_loss": losses["val"],
                "config": CFG
            }, os.path.join(logger.runs_dir, "best.pt"))

        # ChatML formatında test üretimi
        test_prompt = "<|im_start|>user\nWhat is the difference between speed and velocity?\n<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = enc.encode_ordinary(test_prompt)
        ctx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        out_tokens = model.generate(ctx, max_new_tokens=50, temperature=0.7)[0].tolist()
        sample_text = enc.decode(out_tokens)

        step_tokens = CFG["batch_size"] * CFG["context_length"] if iter_step == 1 else CFG["batch_size"] * CFG["context_length"] * CFG["eval_interval"]

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
# 9. Eğitimi Tamamla, Dashboard Üret ve Release Kapsülü Oluştur
# -----------------------------------------------------------------------------
logger.finish(
    model=model,
    encode_fn=enc.encode_ordinary,
    decode_fn=enc.decode,
    test_prompts=[
        "<|im_start|>user\nWhat is the capital of France?\n<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a python function to add two numbers.\n<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhy do birds fly south for winter?\n<|im_end|>\n<|im_start|>assistant\n"
    ],
    device=device
)

print("\n" + "=" * 65)
print(f"🎉 DORINA-70M SFT EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
print(f"   • En İyi Val Loss: {best_val_loss:.4f}")
print(f"   • Kayıt Dizini: {logger.runs_dir}")
print("=" * 65)
