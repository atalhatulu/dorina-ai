import os
import sys
import glob
import math
import numpy as np
import torch
import tiktoken

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from src.model import DorinaGPT
from src.utils.logger import LabLogger

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIGS = {
    "1": {
        "name": "dorina_v0.10_240k",
        "n_layer": 4, 
        "n_embd": 64, 
        "n_head": 4, 
        "context_length": 64,
        "batch_size": 64, 
        "max_iters": 1500, 
        "eval_interval": 100,
        "eval_iters": 20,
        "lr": 1e-3, 
        "min_lr": 1e-4,
        "warmup_iters": 80,
        "weight_decay": 0.05
    },
    "2": {
        "name": "dorina_v0.11_4m",
        "n_layer": 4, 
        "n_embd": 128, 
        "n_head": 4, 
        "context_length": 128,
        "batch_size": 32, 
        "max_iters": 3000, 
        "eval_interval": 150,
        "eval_iters": 25,
        "lr": 5e-4, 
        "min_lr": 5e-5,
        "warmup_iters": 120,
        "weight_decay": 0.1
    },
    "3": {
        "name": "dorina_v0.12_32m",
        "n_layer": 8, 
        "n_embd": 256, 
        "n_head": 8, 
        "context_length": 256,
        "batch_size": 16, 
        "max_iters": 6000, 
        "eval_interval": 200,
        "eval_iters": 30,
        "lr": 3e-4, 
        "min_lr": 3e-5,
        "warmup_iters": 200,
        "weight_decay": 0.1
    },
    "4": {
        "name": "dorina_v0.13_70m",
        "n_layer": 10, 
        "n_embd": 512, 
        "n_head": 8, 
        "context_length": 256,
        "batch_size": 16, 
        "max_iters": 8000, 
        "eval_interval": 250,
        "eval_iters": 35,
        "lr": 2.5e-4, 
        "min_lr": 2.5e-5,
        "warmup_iters": 250,
        "weight_decay": 0.1
    }
}

print("\n" + "=" * 60)
print("🔬 DORINA LAB: BİRLEŞİK MODEL EĞİTİM MERKEZİ")
print("=" * 60)
print("[1] 240k Model (Hızlı Prototip)")
print("[2] 4M Model   (Doğrulama Testi)")
print("[3] 32M Model  (Ana Amiral Gemisi)")
print("[4] 70M Model  (Yeni Nesil Akıl Yürütme Modeli)")
mod_choice = input("Eğitilecek model (1, 2, 3 veya 4) [Varsayılan: 4]: ").strip()
cfg = CONFIGS.get(mod_choice, CONFIGS["4"]).copy()
base_model_name = cfg["name"]

train_bin = "data/tinystories_train.bin"
val_bin = "data/tinystories_val.bin"

if not (os.path.exists(train_bin) and os.path.exists(val_bin)):
    print("\n⚠️ .bin dosyaları bulunamadı! Lütfen önce 'prepare_data.py' çalıştırın.")
    sys.exit(1)

train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')

# -----------------------------------------------------------------------------
# Dinamik Run ve Soykütüğü Yönetimi
# -----------------------------------------------------------------------------
existing_runs = sorted(
    glob.glob(f"runs/{base_model_name}/run*"),
    key=lambda p: int(os.path.basename(p).replace("run", "")) if os.path.basename(p).replace("run", "").isdigit() else 0
)

valid_checkpoints = []
for r in existing_runs:
    best_p = os.path.join(r, "best.pt")
    model_p = os.path.join(r, "model.pt")
    if os.path.exists(best_p):
        valid_checkpoints.append((r, best_p, "best.pt"))
    elif os.path.exists(model_p):
        valid_checkpoints.append((r, model_p, "model.pt"))

is_continuation = False
load_path = None
parent_tag = None
best_val_loss = float('inf')

if valid_checkpoints:
    last_run_dir, default_cp, cp_name = valid_checkpoints[-1]
    last_tag = os.path.basename(last_run_dir)
    print(f"\n📦 Mevcut Kayıt Bulundu: {last_tag} ({cp_name})")
    print("[1] Sıfırdan Eğit (Scratch - Yeni Temiz Koşu)")
    print(f"[2] Kaldığı Yerden Devam Et (Continuous Pre-training - {last_tag})")
    secim = input("Seçiminiz (1/2) [Varsayılan: 2]: ").strip()
    
    if secim != "1":
        is_continuation = True
        load_path = default_cp
        parent_tag = last_tag
        print(f"\n🔄 Model devralınıyor: {load_path}")
        
        # Devam eğitiminde öğrenme oranını daha dengeli tutmak için ölçekle
        cfg["lr"] = cfg["lr"] * 0.7
        cfg["min_lr"] = cfg["min_lr"] * 0.7
        cfg["warmup_iters"] = int(cfg["warmup_iters"] * 0.5)

# Yeni run etiketini otomatik hesapla (run1, run2, run3...)
next_run_idx = 1
if existing_runs:
    run_numbers = [
        int(os.path.basename(p).replace("run", ""))
        for p in existing_runs
        if os.path.basename(p).replace("run", "").isdigit()
    ]
    if run_numbers:
        next_run_idx = max(run_numbers) + 1

new_run_tag = f"run{next_run_idx}"

if not is_continuation:
    print(f"\n🚀 Sıfırdan başlatılıyor ➔ Hedef: {new_run_tag}")
else:
    print(f"🚀 Devam eğitimi başlatılıyor ➔ {parent_tag} -> {new_run_tag}")

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

model = DorinaGPT(
    vocab_size=vocab_size,
    context_length=cfg["context_length"],
    n_embd=cfg["n_embd"],
    n_head=cfg["n_head"],
    n_layer=cfg["n_layer"],
    dropout=0.1
).to(device)

if is_continuation and load_path:
    print(f"📥 Ağırlıklar yükleniyor: {load_path}")
    raw_dict = torch.load(load_path, map_location=device)
    state_dict = raw_dict.get("model_state_dict", raw_dict)
    model.load_state_dict(state_dict, strict=False)
    # Önceki en iyi val_loss kaydedilmişse onu baz al
    if isinstance(raw_dict, dict) and "val_loss" in raw_dict:
        best_val_loss = float(raw_dict["val_loss"])
        print(f"🎯 Önceki en iyi val_loss rekoru devralındı: {best_val_loss:.4f}")
    print("✅ Model ağırlıkları başarıyla aktarıldı!")

def get_lr(it):
    warmup = cfg.get("warmup_iters", 100)
    if it < warmup:
        return cfg["lr"] * (it + 1) / warmup
    if it > cfg["max_iters"]:
        return cfg["min_lr"]
    decay_ratio = (it - warmup) / (cfg["max_iters"] - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg.get("weight_decay", 0.1))
scaler = torch.amp.GradScaler('cuda')

# Sabit Değerlendirme Batch İndeksleri
g_val = torch.Generator().manual_seed(42)
eval_iters = cfg.get("eval_iters", 25)
val_indices = torch.randint(len(val_data) - cfg["context_length"], (eval_iters, cfg["batch_size"]), generator=g_val)

def get_batch(split, k=0):
    d = train_data if split == 'train' else val_data
    if split == 'train':
        ix = torch.randint(len(d) - cfg["context_length"], (cfg["batch_size"],))
    else:
        ix = val_indices[k]
    
    x = torch.stack([torch.from_numpy(d[i:i+cfg["context_length"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i+1:i+cfg["context_length"]+1].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, k)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

logger = LabLogger(
    model_name=base_model_name,
    run_tag=new_run_tag,
    config={
        **cfg,
        "vocab_size": vocab_size,
        "mode": "continuous_pretrain" if is_continuation else "scratch",
        "resumed_from": parent_tag,
        "precision": "fp16_mixed"
    },
    total_steps=cfg["max_iters"]
)

# -----------------------------------------------------------------------------
# 1-Tabanlı Temiz Eğitim Döngüsü
# -----------------------------------------------------------------------------
for iter_step in range(1, cfg["max_iters"] + 1):
    # 1. Gradyan ve Parametre Güncellemesi
    lr = get_lr(iter_step)
    for pg in optimizer.param_groups: 
        pg['lr'] = lr

    xb, yb = get_batch('train')
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # 2. Değerlendirme & Kayıt Aşaması
    if iter_step == 1 or iter_step % cfg["eval_interval"] == 0 or iter_step == cfg["max_iters"]:
        losses = estimate_loss()
        
        # Her eval periyodunda güncel durumu model.pt olarak kaydet
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": iter_step,
            "val_loss": losses['val'],
            "config": cfg
        }, os.path.join(logger.runs_dir, "model.pt"))

        # Eğer önceki rekor aşıldıysa en iyi anı dondur
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": iter_step,
                "val_loss": losses['val'],
                "config": cfg
            }, os.path.join(logger.runs_dir, "best.pt"))

        prompt_tokens = enc.encode("Once upon a time")
        context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        out_tokens = model.generate(context, max_new_tokens=40)[0].tolist()
        sample_text = enc.decode(out_tokens)

        if iter_step == 1:
            step_tokens = cfg["batch_size"] * cfg["context_length"]
        else:
            step_tokens = cfg["batch_size"] * cfg["context_length"] * cfg["eval_interval"]

        logger.log(
            step=iter_step,
            train_loss=losses['train'],
            val_loss=losses['val'],
            grad_norm=None,
            lr=lr,
            batch_tokens=step_tokens,
            sample_text=sample_text
        )

logger.finish(
    model=model,
    encode_fn=enc.encode_ordinary,
    decode_fn=enc.decode,
    test_prompts=["Once upon a time", "In the solar system, the Sun is", "When water gets extremely cold, it turns into"],
    device=device
)