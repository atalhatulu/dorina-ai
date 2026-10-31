import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

tok = Tokenizer.from_file('tokenizers/dorina_nano_tr_12k.json')
cfg = NanoTRConfig(vocab_size=12288, max_seq_len=256, d_model=256, n_layer=10, n_heads=8, n_kv_heads=4, head_dim=32, intermediate_size=688, tied_embeddings=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DorinaNanoTR(cfg).to(device)
model.load_state_dict(torch.load('models/dorina_nano_tr/dorina_nano_tr_10m.pt', map_location=device, weights_only=True))
model.eval()

def raw_fix(chunk):
    prompt = f"<fix> {chunk} =>"
    inp = torch.tensor([tok.encode(prompt).ids], device=device)
    gen = []
    with torch.no_grad():
        for _ in range(25):
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            nid = torch.argmax(logits[0, -1]).item()
            if nid == tok.token_to_id('<eos>') or '\n' in tok.decode([nid]):
                break
            gen.append(nid)
            inp = torch.cat([inp, torch.tensor([[nid]], device=device)], dim=1)
    res = tok.decode(gen).strip()
    return res if res else chunk

# Akıllı N-Gram Kayan Pencere (Sliding Window Chunking)
def smart_keyboard_space_pressed(full_text):
    words = full_text.strip().split()
    if not words:
        return full_text
    
    # Eğer tek kelimeyse tek kelimeye bak
    if len(words) == 1:
        return raw_fix(words[0])
    
    # 2 veya 3 kelimelik yerel bağlam penceresi (Son n-gram)
    # Örn: "bunu yapabilirmiyiz" -> son 2 kelime penceresi: "bunu yapabilirmiyiz"
    window_size = min(3, len(words))
    target_chunk = " ".join(words[-window_size:])
    
    # Modeli tüm cümleyle yormak yerine sadece son odak alanına veriyoruz!
    fixed_chunk = raw_fix(target_chunk)
    
    # Geri kalan baştaki kelimelerle birleştir
    prefix = " ".join(words[:-window_size])
    if prefix:
        result = prefix + " " + fixed_chunk
    else:
        result = fixed_chunk
        
    return result

test_inputs = [
    "bunu yapabilirmiyiz",
    "bence sende gelmelisin",
    "yarin aksam kesin geliyom",
    "sen beni anladinmi",
    "cok guzel bir gun ama evde kimse yok",
    "hava sogukmu disarida",
    "ne yapcam simdi ben"
]

print("="*70)
print("🎯 AKILLI KLAVYE KAYAN PENCERE (SLIDING WINDOW) TESTİ")
print("="*70)
for inp in test_inputs:
    out = smart_keyboard_space_pressed(inp)
    print(f"Yazılan:   \"{inp}\"")
    print(f"Düzeltilen:\"{out}\"")
    print("-" * 50)
