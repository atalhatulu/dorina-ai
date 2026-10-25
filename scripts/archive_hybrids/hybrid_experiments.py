"""
================================================================================
DORINA-MoE: SÜPER MELEZLEME VE UZMAN FÜZYONU (scripts/hybrid_experiments.py)
================================================================================
Bu script iki kritik melez modeli üretir ve test eder:

1. MODEL A: Linear/SLERP Füzyon (FUSION CHAMPION)
   - Taban: Run 8 Step 12.000 (Tüm zamanların algoritma & rekor şampiyonu, 52.5P)
   - Donör: Run 8 Final (Tüm zamanların bilim ve mimari şampiyonu, 6.8P Sci, 50.2P)
   - Oran: %50 Run 8 12k + %50 Run 8 Final

2. MODEL B: Uzmanlar Arası Çekmece Melezlemesi (DISENTANGLED EXPERT SURGERY)
   - Shared Layers (Attention, Embeddings, Norms): Run 8 12k'dan
   - Expert 0 & 1 (Temel Kod & Algoritma Çekmecesi): Run 8 12k'dan (14.1 Puanlık Algoritma)
   - Expert 2 (Süper Sentaks Çekmecesi): Run 8 Step 9k'dan (16.0 Puanlık Tüm Zamanların Sentaks Rekoru)
   - Expert 3 (Bilim & Mimari Çekmecesi): Run 8 Final'den (6.8 Puanlık Bilim Rekoru)
   - Router: Her katmanın router'ı dengeli enterpolasyonla lehimlenir.
================================================================================
"""

import os
import sys
import torch
import copy

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models", "dorina_44m_moe")

def get_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt

def create_linear_fusion():
    out_dir = os.path.join(MODELS_DIR, "hybrid_slerp_12k_final")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.pt")

    p_12k = os.path.join(MODELS_DIR, "run8", "model_step_12000.pt")
    p_final = os.path.join(MODELS_DIR, "run8", "model.pt")

    print(f"🧬 Model A: Linear Füzyon hazırlanıyor (%50 Step 12k + %50 Final)...")
    s_12k = get_state(p_12k)
    s_final = get_state(p_final)

    merged = {}
    for k in s_12k.keys():
        if k in s_final and s_12k[k].shape == s_final[k].shape:
            merged[k] = 0.5 * s_12k[k].float() + 0.5 * s_final[k].float()
            if s_12k[k].dtype == torch.float16:
                merged[k] = merged[k].half()
        else:
            merged[k] = s_12k[k]

    torch.save(merged, out_path)
    print(f"  ✓ Model A kaydedildi -> {out_path}")
    return out_path

def create_expert_surgery():
    out_dir = os.path.join(MODELS_DIR, "hybrid_expert_surgery")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.pt")

    p_12k = os.path.join(MODELS_DIR, "run8", "model_step_12000.pt")
    p_9k = os.path.join(MODELS_DIR, "run8", "model_step_9000.pt")
    p_final = os.path.join(MODELS_DIR, "run8", "model.pt")

    print(f"\n🧠 Model B: Uzman Çekmece Cerrahisi (Expert Surgery) hazırlanıyor...")
    s_12k = get_state(p_12k)
    s_9k = get_state(p_9k)
    s_final = get_state(p_final)

    # Taban olarak Step 12k'yı alıyoruz (En yüksek genel zeka ve dikkat katmanları)
    surgical_state = copy.deepcopy(s_12k)

    # Her bloktaki 4 uzmanı modüler olarak yerleştiriyoruz:
    # experts.0 ve experts.1: Zaten 12k'da (En yüksek algoritma)
    # experts.2: 9k'dan aktarılıyor (16.0 puanlık sentaks rekoru)
    # experts.3: Final'den aktarılıyor (6.8 puanlık bilim rekoru)
    
    for k in list(surgical_state.keys()):
        # Expert 2 -> Step 9k
        if ".moe.experts.2." in k and k in s_9k:
            surgical_state[k] = s_9k[k]
        # Expert 3 -> Final
        elif ".moe.experts.3." in k and k in s_final:
            surgical_state[k] = s_final[k]
        # Router -> 12k ve Final arasında lehimle
        elif ".moe.router." in k and k in s_final:
            surgical_state[k] = (0.5 * s_12k[k].float() + 0.5 * s_final[k].float()).half() if s_12k[k].dtype == torch.float16 else (0.5 * s_12k[k].float() + 0.5 * s_final[k].float())

    torch.save(surgical_state, out_path)
    print(f"  ✓ Model B kaydedildi -> {out_path}")
    return out_path

if __name__ == "__main__":
    create_linear_fusion()
    create_expert_surgery()
    print("\n🎉 İki melez model de diskte hazır!")
