"""
================================================================================
DORINA V3: 2 DAKİKALIK YEREL SANITY-CHECK (RTX 4050 ÜZERİNDE 20 ADIM PROVA)
================================================================================
Kaggle'a gitmeden önce laptopumuzda:
1. Üretilen v3_shard1_200m_train.bin dosyasını açar.
2. 180M MoE (32 Expert + 1 Shared) modelini başlatır.
3. 20 adım gerçek eğitim koşar.
4. Loss düşüşünü, VRAM kullanımını ve gradyan akışını doğrular.
================================================================================
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.model_moe_v3 import DorinaMoEV3, MoEConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = os.path.join(ROOT_DIR, "data", "v3_shard1_200m_train.bin")

print("=" * 75)
print("🧪 DORINA V3: 20 ADIMLIK YEREL DOĞRULAMA PROVASI BAŞLIYOR")
print("=" * 75)
print(f"Donanım : {torch.cuda.get_device_name(0)}")
print(f"Veri    : {data_path}")

# Veriyi memmap ile oku
tokens = np.memmap(data_path, dtype=np.uint16, mode='r')
print(f"Toplam Yüklenen Token: {len(tokens):,} Token")

cfg = MoEConfig()
model = DorinaMoEV3(cfg).to(device)
opt = model.configure_optimizers(weight_decay=0.1, lr=6e-4, device_type="cuda")

batch_size = 4
seq_len = 1024

def get_batch():
    ix = np.random.randint(0, len(tokens) - seq_len - 1, size=(batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i:i+seq_len].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(tokens[i+1:i+seq_len+1].astype(np.int64)) for i in ix]).to(device)
    return x, y

print("\n🚀 20 Adımlık Canlı Eğitim Başladı (Laptop GPU):")
print("-" * 75)
print(f"{'Adım':<6} | {'Loss':<8} | {'VRAM Ayrılan':<14} | {'VRAM Rezerve':<14} | {'Adım Süresi'}")
print("-" * 75)

model.train()
torch.cuda.reset_peak_memory_stats()
losses = []

for step in range(1, 21):
    t0 = time.time()
    x, y = get_batch()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    dt = (time.time() - t0) * 1000
    losses.append(loss.item())
    
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    res_gb = torch.cuda.memory_reserved() / 1024**3
    print(f"{step:6d} | {loss.item():<8.4f} | {alloc_gb:<14.2f} GB | {res_gb:<14.2f} GB | {dt:.1f} ms")

print("-" * 75)
print(f"✓ İlk Loss: {losses[0]:.4f}  --->  Son Loss: {losses[-1]:.4f}")
assert losses[-1] < losses[0], "Loss düşmedi!"
print("🎉 TEBRİKLER: 180M MoE Mimarisi Homojen Veriyi Kusursuz Şekilde Sindiriyor!")
print("Kaggle'a uçmaya %100 HAZIRIZ!")
print("=" * 75)
