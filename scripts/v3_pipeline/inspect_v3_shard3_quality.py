import os
import sys
import torch
from tokenizers import Tokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from model_moe_v3 import DorinaMoEV3, MoEConfig

tok = Tokenizer.from_file(os.path.join(ROOT_DIR, "tokenizers/dorina_tokenizer_v3_32k.json"))
cfg = MoEConfig(vocab_size=32768, n_layer=12, d_model=512, n_heads=8, n_kv_heads=2, max_seq_len=1024, n_experts=32, top_k=2, expert_hidden=256, shared_hidden=256)
model = DorinaMoEV3(cfg)
sd = torch.load(os.path.join(ROOT_DIR, "models/dorina_v3_180m/shard3/dorina_v3_shard3_final.pt"), map_location="cuda", weights_only=False)
model.load_state_dict(sd)
model.cuda().eval()

prompts = [
    "<|im_start|>user\nTürkiye'nin başkenti neresidir?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\n2 + 2 kaç eder?<|im_end|>\n<|im_start|>assistant\n",
    "def fibonacci(n):\n    \"\"\"Return nth fibonacci number.\"\"\"\n",
    "<|im_start|>user\nBana Python ile bir liste sıralama örneği yaz.<|im_end|>\n<|im_start|>assistant\n"
]

print("=== DORINA V3 SHARD 3 (600M TOKEN) CANLI CEVAP ÖRNEKLERİ ===")
for p in prompts:
    inp = torch.tensor([tok.encode(p).ids], dtype=torch.long, device="cuda")
    out = inp
    with torch.no_grad():
        for _ in range(40):
            logits, _ = model(out[:, -1024:])
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            if nxt.item() == 3:
                break
    resp = tok.decode(out[0].tolist()[len(inp[0]):])
    print(f"\n[Girdi]: {p.strip()}")
    print(f"[Dorina V3 Shard 3]: {resp.strip()}")
    print("-" * 60)
