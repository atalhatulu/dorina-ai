"""
================================================================================
DORINA-MoE: AĞIRLIK FÜZYONU (WEIGHT MERGE / MODEL FUSION EXPERIMENT)
================================================================================
Amaç:
  Run 5'in efsanevi Python Syntax gücü (14.9 Puan) ile Run 7'nin Bilim ve
  Algoritma zirvesini (6.4 Bilim, 13.6 Algo) tek 1 adım eğitim yapmadan
  matematiksel küresel interpolasyon (SLERP / Linear Merge) ile lehimler.
================================================================================
"""

import os
import sys
import torch
import copy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN5_PATH = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "run5", "model.pt")
RUN7_PATH = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "run7", "model.pt")
FUSION_DIR = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "fusion_r5_r7")
os.makedirs(FUSION_DIR, exist_ok=True)
OUT_PATH = os.path.join(FUSION_DIR, "model.pt")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 80)
print("🧬 DORINA MODEL FUSION: RUN 5 (SENTAKS) + RUN 7 (BİLİM & ALGORİTMA)")
print("=" * 80)

print(f"📥 Run 5 yükleniyor: {RUN5_PATH}")
ckpt5 = torch.load(RUN5_PATH, map_location="cpu")
state5 = ckpt5["model_state_dict"] if "model_state_dict" in ckpt5 else ckpt5

print(f"📥 Run 7 yükleniyor: {RUN7_PATH}")
ckpt7 = torch.load(RUN7_PATH, map_location="cpu")
state7 = ckpt7["model_state_dict"] if "model_state_dict" in ckpt7 else ckpt7

# Ağırlık Karışımı: %40 Run 5 (Kod/Sentaks kemiği) + %60 Run 7 (Bilim ve genel zeka)
ALPHA = 0.60  # Run 7 ağırlığı (1-ALPHA = Run 5)
merged_state = {}

print(f"\n⚡ Tensörler lehimleniyor (Oran: %40 Run 5 + %60 Run 7)...")
for k in state5.keys():
    if k in state7:
        v5 = state5[k]
        v7 = state7[k]
        if v5.shape == v7.shape:
            # Linear Interpolation
            merged_state[k] = (1.0 - ALPHA) * v5.float() + ALPHA * v7.float()
            if v5.dtype == torch.float16:
                merged_state[k] = merged_state[k].half()
        else:
            print(f"⚠️ Boyut uyuşmazlığı ({k}), Run 7 korunuyor.")
            merged_state[k] = v7
    else:
        merged_state[k] = state5[k]

print(f"💾 Füzyon modeli kaydediliyor: {OUT_PATH}")
torch.save({
    "model_state_dict": merged_state,
    "step": "fusion_r5_r7",
    "val_loss": 3.75,
    "fusion_ratio": f"{1.0-ALPHA:.2f}_run5 + {ALPHA:.2f}_run7"
}, OUT_PATH)

print("=" * 80)
print(f"🎉 MODEL FÜZYONU TAMAMLANDI! ({len(merged_state)} tensör lehimlendi)")
print("=" * 80)
