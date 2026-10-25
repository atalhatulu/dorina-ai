"""
================================================================================
DORINA-AI: CANLI ETKİLEŞİMLİ SOHBET ARAYÜZÜ (chat_dorina.py)
================================================================================
Kullanım:
  python chat_dorina.py
Çıkmak için:
  'q', 'quit' veya 'exit' yazabilirsin.
================================================================================
"""

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

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.model_moe import DorinaMoEGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Model ve Tokenizer Yükleme
tok_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
tokenizer = Tokenizer.from_file(tok_path)

model = DorinaMoEGPT(
    vocab_size=32768,
    context_length=512,
    n_embd=384,
    n_head=6,
    n_layer=6,
    num_experts=4,
    top_k=2,
    dropout=0.0
).to(device)

model_path = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "dorina_44m_sft_broad", "model.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "dorina_44m_sft_rehearsal", "model.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "run12", "model.pt")

ckpt = torch.load(model_path, map_location=device)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.eval()

print("=" * 70)
print("🤖 DORINA-AI CANLI SOHBET ARAYÜZÜ")
print("⚡ Model: Dorina 2. Nesil (44.5M Sparse MoE - 1 Milyar Token + SFT)")
print("💡 Çıkmak için 'q', 'exit' veya 'quit' yazabilirsin.")
print("=" * 70 + "\n")

chat_history = ""

while True:
    try:
        user_input = input("Talha > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nGörüşmek üzere!")
        break

    if not user_input:
        continue
    if user_input.lower() in ['q', 'exit', 'quit']:
        print("\nDorina > Görüşmek üzere Talha! İyi çalışmalar.")
        break

    # ChatML Şablonu
    prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.4,
            top_k=40
        )[0].tolist()

    full_reply = tokenizer.decode(out_ids, skip_special_tokens=False)
    # Sadece asistan cevabını ve <|im_end|>'e kadar olan kısmı al
    resp = full_reply[len(prompt):].split("<|im_end|>")[0].strip()

    print(f"\nDorina > {resp}\n")
    print("-" * 70)
