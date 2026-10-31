import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("src"))
from model_moe_v3 import DorinaMoEV3, MoEConfig
from tokenizers import Tokenizer

def load_v3_model(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"[Yükleniyor] Model ağırlıkları: {ckpt_path} -> {device}")
    
    cfg = MoEConfig(
        vocab_size=32768,
        n_layer=12,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=1024,
        n_experts=32,
        top_k=2,
        expert_hidden=256,
        shared_hidden=256
    )
    
    model = DorinaMoEV3(cfg)
    
    # State dict yükle
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
        
    # Temiz yükleme (module. varsa kaldır)
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    print("[Başarılı] Dorina V3 (180.6M MoE) hafızaya alındı!")
    return model, cfg

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=40, device="cuda"):
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    print(f"\n--- PROMPT: {prompt} ---")
    generated = list(encoded.ids)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= 1024 else input_ids[:, -1024:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            generated.append(idx_next.item())
            
            # EOT token kontrolü
            if idx_next.item() == 3: # <|im_end|>
                break
                
    output_text = tokenizer.decode(generated)
    print(f"--- MODEL CEVABI ---\n{output_text}\n" + "=" * 50)
    return output_text

def main():
    ckpt_path = "models/dorina_v3_180m/shard1/dorina_v3_shard1_final.pt"
    tok_path = "tokenizers/dorina_tokenizer_v3_32k.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(tok_path)
    model, cfg = load_v3_model(ckpt_path, device=device)
    
    test_prompts = [
        "def fibonacci(n):",
        "Türkiye Cumhuriyeti'nin kurucusu",
        "class BinarySearchTree:\n    def __init__(self):",
        "Yapay zeka modellerinde Mixture of Experts (MoE) mimarisi"
    ]
    
    print("\n" + "#" * 60)
    print("DORINA V3 (180.6M) SHARD 1 İLK CANLI TESTLERİ")
    print("#" * 60)
    
    for p in test_prompts:
        generate_text(model, tokenizer, p, max_new_tokens=80, temperature=0.6, top_k=30, device=device)

if __name__ == "__main__":
    main()
