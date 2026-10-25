"""
================================================================================
DORINA-MoE: ŞAMPİYONLAR ŞAMPİYONU FÜZYONU (scripts/fuse_grand_champions.py)
================================================================================
Turnuvanın en iyi iki modelini lehimler:
  - Model 1: hybrid_12k_final_30_70 (52.5 Puan - Bilim & Talimat Zirvesi)
  - Model 2: hybrid_expert_surgery   (52.2 Puan - 15.8 Sentaks & 13.1 Algoritma Dahisi)
Oran: %50 - %50
Çıktı: models/dorina_44m_moe/hybrid_grand_champion/model.pt
================================================================================
"""

import os
import sys
import torch

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

def main():
    p1 = os.path.join(MODELS_DIR, "hybrid_12k_final_30_70", "model.pt")
    p2 = os.path.join(MODELS_DIR, "hybrid_expert_surgery", "model.pt")
    out_dir = os.path.join(MODELS_DIR, "hybrid_grand_champion")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.pt")

    print("👑 En iyi iki melez lehimleniyor: hybrid_12k_final_30_70 + hybrid_expert_surgery...")
    s1 = get_state(p1)
    s2 = get_state(p2)

    grand_state = {}
    for k in s1.keys():
        if k in s2 and s1[k].shape == s2[k].shape:
            grand_state[k] = 0.5 * s1[k].float() + 0.5 * s2[k].float()
            if s1[k].dtype == torch.float16:
                grand_state[k] = grand_state[k].half()
        else:
            grand_state[k] = s1[k]

    torch.save(grand_state, out_path)
    print(f"✓ hybrid_grand_champion kaydedildi -> {out_path}")

if __name__ == "__main__":
    main()
