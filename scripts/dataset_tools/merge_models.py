import os
import sys
import torch

def merge_models(ckpt1_path, ckpt2_path, out_path, alpha=0.6):
    """
    Weighted average / Linear Interpolation:
    W_hybrid = alpha * W_ckpt1 + (1 - alpha) * W_ckpt2
    alpha = 0.6 (Step 4500 - Zirve Benchmark)
    1 - alpha = 0.4 (Step 7500 - Maksimum Veri Hacmi)
    """
    print(f"[Melezleme] Model 1 (Zirve Skor %60): {ckpt1_path}")
    print(f"[Melezleme] Model 2 (1.6B Token %40): {ckpt2_path}")
    
    sd1 = torch.load(ckpt1_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd1:
        sd1 = sd1["model_state_dict"]
    clean_sd1 = {k.replace("module.", ""): v for k, v in sd1.items()}
    
    sd2 = torch.load(ckpt2_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd2:
        sd2 = sd2["model_state_dict"]
    clean_sd2 = {k.replace("module.", ""): v for k, v in sd2.items()}
    
    hybrid_sd = {}
    keys = list(clean_sd1.keys())
    print(f"[Parametre Sayımı] Toplam tensör adedi: {len(keys)}")
    
    for k in keys:
        p1 = clean_sd1[k]
        p2 = clean_sd2[k]
        
        # Float ağırlıkları melezle, int buffer'ları aynen koru
        if p1.is_floating_point():
            hybrid_sd[k] = alpha * p1.float() + (1.0 - alpha) * p2.float()
        else:
            hybrid_sd[k] = p1
            
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(hybrid_sd, out_path)
    print(f"✅ [BAŞARILI] Melez Model Üretildi: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    ckpt1 = "kaggle_upload/dorina-v3-checkpoint/dorina_v3_step_4500.pt"
    ckpt2 = "kaggle_kernel/titan_final_run/checkpoints/dorina_v3_titan_1b_final.pt"
    out = "models/dorina_v3_titan_hybrid_4500_7500.pt"
    merge_models(ckpt1, ckpt2, out, alpha=0.6)
