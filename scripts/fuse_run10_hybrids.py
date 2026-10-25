"""
================================================================================
DORINA-MoE: RUN 10 SÜPER MELEZ OPERASYONU (scripts/fuse_run10_hybrids.py)
================================================================================
Kombinasyonlar:
  1. hybrid_r10_linear_70_30:
     - %70 hybrid_surgery_v2_ultimate (Math 11.0 Rekortmeni) + %30 Run 10 (Algo 13.6 Rekortmeni)
  2. hybrid_r10_linear_50_50:
     - %50 hybrid_12k_final_30_70 (52.5P Genel Rekortmen) + %50 Run 10 (Algo 13.6)
  3. hybrid_surgery_v3_titan:
     - Shared / Attention / Router: hybrid_12k_final_30_70 (Syntax 15.3, Inst 11.3)
     - Expert 0 & 1: Run 10 (Algo 13.6 Rekortmen Uzmanlar)
     - Expert 2 & 3: hybrid_surgery_v2_ultimate (Math 11.0 Rekortmen Uzmanlar)
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
    print(f"Loading {path}...")
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
    p_r10 = os.path.join(MODELS_DIR, "run10", "model.pt")
    p_surg_v2 = os.path.join(MODELS_DIR, "hybrid_surgery_v2_ultimate", "model.pt")
    p_champ_30_70 = os.path.join(MODELS_DIR, "hybrid_12k_final_30_70", "model.pt")

    s_r10 = get_state(p_r10)
    s_v2 = get_state(p_surg_v2)
    s_champ = get_state(p_champ_30_70)

    # 1. Linear 70/30: Math Rekortmeni + Run 10 Algo Aşısı
    fuse_linear(s_v2, s_r10, 0.30, "hybrid_r10_linear_70_30")

    # 2. Linear 50/50: Genel Rekortmen (52.5) + Run 10 Algo
    fuse_linear(s_champ, s_r10, 0.50, "hybrid_r10_linear_50_50")

    # 3. Cerrahi V3 TITAN:
    out_dir = os.path.join(MODELS_DIR, "hybrid_surgery_v3_titan")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.pt")

    surg_v3 = copy.deepcopy(s_champ)
    for k in list(surg_v3.keys()):
        # Expert 0 & 1 -> Run 10 (Algo 13.6)
        if (".moe.experts.0." in k or ".moe.experts.1." in k) and k in s_r10:
            surg_v3[k] = s_r10[k]
        # Expert 2 & 3 -> Surgery V2 (Math 11.0)
        elif (".moe.experts.2." in k or ".moe.experts.3." in k) and k in s_v2:
            surg_v3[k] = s_v2[k]

    torch.save(surg_v3, out_path)
    print(f"  ✓ hybrid_surgery_v3_titan oluşturuldu -> {out_path}")
    print("\n🎉 Tüm 3 Melez başarıyla üretildi!")

if __name__ == "__main__":
    main()
