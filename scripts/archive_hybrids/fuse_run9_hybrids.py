"""
================================================================================
DORINA-MoE: RUN 9 RECOVERY & SÜPER MELEZ TURNUVASI (scripts/fuse_run9_hybrids.py)
================================================================================
Amaç:
  Run 9'un 15.000. adımındaki efsanevi Bilim Zirvesi (8.0 Bilim, 14.6 Sentaks, 12.1 Algo)
  ile Turnuva Şampiyonumuz `hybrid_12k_final_30_70` (52.5P) ve `hybrid_expert_surgery` (52.2P)
  modellerini cerrahi olarak birleştirip skoru 53-55 bandına fırlatmak!

Aday Melezler:
  1. hybrid_r9_champ_70_30  : %70 Şampiyon Melez (52.5P) + %30 Run 9 15k (8.0 Bilim Aşısı)
  2. hybrid_r9_champ_50_50  : %50 Şampiyon Melez + %50 Run 9 15k
  3. hybrid_surgery_v2      : Uzman Cerrahisi V2
     - Shared & Router: Şampiyon Melezden (52.5P)
     - Expert 0 & 1: 12k (14.1 Algo)
     - Expert 2: Run 8 9k (16.0 Sentaks Rekortmeni)
     - Expert 3: Run 9 15k (8.0 Bilim Rekortmeni) -> TARİHİ BİLİM ÇEKMECESİ!
================================================================================
"""

import os
import sys
import torch
import copy
import json

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

def fuse_linear(s1, s2, alpha, out_name):
    out_dir = os.path.join(MODELS_DIR, out_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.pt")
    
    merged = {}
    for k in s1.keys():
        if k in s2 and s1[k].shape == s2[k].shape:
            merged[k] = (1.0 - alpha) * s1[k].float() + alpha * s2[k].float()
            if s1[k].dtype == torch.float16:
                merged[k] = merged[k].half()
        else:
            merged[k] = s1[k]
    
    torch.save(merged, out_path)
    print(f"  ✓ {out_name} oluşturuldu -> {out_path}")
    return out_path

def main():
    p_champ = os.path.join(MODELS_DIR, "hybrid_12k_final_30_70", "model.pt")
    p_r9_15k = os.path.join(MODELS_DIR, "run9", "model_step_15000.pt")
    p_r8_9k = os.path.join(MODELS_DIR, "run8", "model_step_9000.pt")
    p_r8_12k = os.path.join(MODELS_DIR, "run8", "model_step_12000.pt")

    print("📥 Checkpoint'ler hafızaya alınıyor...")
    s_champ = get_state(p_champ)
    s_r9_15k = get_state(p_r9_15k)
    s_r8_9k = get_state(p_r8_9k)
    s_r8_12k = get_state(p_r8_12k)

    # 1. %70 Şampiyon + %30 Run 9 15k (Hafif Bilim Aşısı)
    fuse_linear(s_champ, s_r9_15k, 0.30, "hybrid_r9_champ_70_30")

    # 2. %50 Şampiyon + %50 Run 9 15k (Dengeli Lehim)
    fuse_linear(s_champ, s_r9_15k, 0.50, "hybrid_r9_champ_50_50")

    # 3. Uzman Cerrahisi V2 (Tüm Zamanların En Güçlü Çekmeceleri)
    out_dir = os.path.join(MODELS_DIR, "hybrid_surgery_v2_ultimate")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.pt")

    surg_v2 = copy.deepcopy(s_champ)
    for k in list(surg_v2.keys()):
        # Expert 0 & 1 -> Run 8 12k (14.1 Algo)
        if (".moe.experts.0." in k or ".moe.experts.1." in k) and k in s_r8_12k:
            surg_v2[k] = s_r8_12k[k]
        # Expert 2 -> Run 8 9k (16.0 Sentaks Rekortmeni)
        elif ".moe.experts.2." in k and k in s_r8_9k:
            surg_v2[k] = s_r8_9k[k]
        # Expert 3 -> Run 9 15k (8.0 Bilim Rekortmeni)
        elif ".moe.experts.3." in k and k in s_r9_15k:
            surg_v2[k] = s_r9_15k[k]

    torch.save(surg_v2, out_path)
    print(f"  ✓ hybrid_surgery_v2_ultimate (Süper Çekmece) oluşturuldu -> {out_path}")
    print("\n🎉 Tüm Run 9 melezleri hazır!")

if __name__ == "__main__":
    main()
