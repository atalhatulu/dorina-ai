"""
================================================================================
DORINA-MoE: GENİŞ MELEZ KOMBİNASYONLARI (scripts/generate_all_hybrids.py)
================================================================================
Üretilen Melez Havuzu:
1. hybrid_12k_final_50_50   : Model A (%50 Step 12k + %50 Final) - (Zaten üretildi)
2. hybrid_expert_surgery    : Model B (Çekmece: Exp0,1=12k, Exp2=9k, Exp3=Final) - (Zaten üretildi)
3. hybrid_12k_final_70_30   : %70 Step 12k (Algoritma/Genel Zirve) + %30 Final (Bilim/Mimari Aşısı)
4. hybrid_12k_final_30_70   : %30 Step 12k + %70 Final (Daha derin Loss ve Bilim ağırlıklı)
5. hybrid_r7_r8_12k_50_50   : %50 Run 7 Şampiyonu + %50 Run 8 12k Şampiyonu (İki farklı Run'ın zirveleri)
6. hybrid_expert_r7_r8_9k   : Uzman Cerrahisi 2 (Run 7 Taban + Run 8 9k Sentaks Uzmanı + Run 8 Final Bilim Uzmanı)
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
    p_12k = os.path.join(MODELS_DIR, "run8", "model_step_12000.pt")
    p_9k = os.path.join(MODELS_DIR, "run8", "model_step_9000.pt")
    p_final = os.path.join(MODELS_DIR, "run8", "model.pt")
    p_r7 = os.path.join(MODELS_DIR, "run7", "model.pt")

    print("📥 Checkpoint'ler hafızaya alınıyor...")
    s_12k = get_state(p_12k)
    s_9k = get_state(p_9k)
    s_final = get_state(p_final)
    s_r7 = get_state(p_r7)

    # 3. %70 Step 12k + %30 Final
    fuse_linear(s_12k, s_final, 0.30, "hybrid_12k_final_70_30")

    # 4. %30 Step 12k + %70 Final
    fuse_linear(s_12k, s_final, 0.70, "hybrid_12k_final_30_70")

    # 5. %50 Run 7 + %50 Run 8 12k
    fuse_linear(s_r7, s_12k, 0.50, "hybrid_r7_r8_12k_50_50")

    # 6. Uzman Cerrahisi 2 (Run 7 Taban + Run 8 9k Sentaks + Run 8 Final Bilim)
    out_dir = os.path.join(MODELS_DIR, "hybrid_expert_r7_r8")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.pt")
    s_surg2 = copy.deepcopy(s_r7)
    for k in list(s_surg2.keys()):
        if ".moe.experts.2." in k and k in s_9k:
            s_surg2[k] = s_9k[k]
        elif ".moe.experts.3." in k and k in s_final:
            s_surg2[k] = s_final[k]
    torch.save(s_surg2, out_path)
    print(f"  ✓ hybrid_expert_r7_r8 oluşturuldu -> {out_path}")

    print("\n🎉 Tüm melez modeller başarıyla üretildi!")

if __name__ == "__main__":
    main()
