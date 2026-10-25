import torch
import torch.nn.functional as F
from kaggle_v4_shard2_kernel.main import DorinaDenseV4, DenseConfig
from tokenizers import Tokenizer

print("Modeller Yükleniyor...")
cfg = DenseConfig(max_seq_len=2048)

# Shard 2 Yükle
model_s2 = DorinaDenseV4(cfg)
sd_s2 = torch.load("models/dorina_v4/dorina_v4_shard2_final.pt", map_location="cpu", weights_only=False)
model_s2.load_state_dict({k.replace("module.", ""): v for k, v in sd_s2.items()})
model_s2.eval()

# Shard 3 Yükle
model_s3 = DorinaDenseV4(cfg)
sd_s3 = torch.load("models/dorina_v4/dorina_v4_shard3_final.pt", map_location="cpu", weights_only=False)
model_s3.load_state_dict({k.replace("module.", ""): v for k, v in sd_s3.items()})
model_s3.eval()

tok = Tokenizer.from_file("tokenizers/dorina_tokenizer_v4_32k.json")

test_prompts = [
    "The primary function of mitochondria is",
    "Water boils at 100 degrees Celsius because",
    "In computer programming, a loop is used to",
    "The Industrial Revolution changed society by",
    "The chemical formula for water is"
]

print("\n" + "="*85)
print("DORINA-V4 KAFA KAFAYA KIYASLAMA: SHARD 2 (1.0B Token) vs SHARD 3 (1.5B Token)")
print("="*85)

for p in test_prompts:
    print(f"\n[TEST SORUSU / GİRİŞ]: \"{p}\"")
    enc = tok.encode(p)
    inp = torch.tensor([enc.ids], dtype=torch.long)
    
    # 1. Shard 2 Üretimi (Greedy)
    inp_s2 = inp.clone()
    with torch.no_grad():
        for _ in range(40):
            l, _ = model_s2(inp_s2)
            nxt = torch.argmax(l[:, -1, :], dim=-1, keepdim=True)
            inp_s2 = torch.cat([inp_s2, nxt], dim=1)
    out_s2 = tok.decode(inp_s2[0].tolist())
    
    # 2. Shard 3 Üretimi (Greedy)
    inp_s3 = inp.clone()
    with torch.no_grad():
        for _ in range(40):
            l, _ = model_s3(inp_s3)
            nxt = torch.argmax(l[:, -1, :], dim=-1, keepdim=True)
            inp_s3 = torch.cat([inp_s3, nxt], dim=1)
    out_s3 = tok.decode(inp_s3[0].tolist())
    
    print("--- [SHARD 2 - 1.0B]:")
    print(out_s2)
    print("--- [SHARD 3 - 1.5B]:")
    print(out_s3)
