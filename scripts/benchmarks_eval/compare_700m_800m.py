import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

tok = Tokenizer.from_file('tokenizers/dorina_nano_tr_12k.json')
cfg = NanoTRConfig(vocab_size=12288, max_seq_len=256, d_model=256, n_layer=10, n_heads=8, n_kv_heads=4, head_dim=32, intermediate_size=688, tied_embeddings=True)

# Load 700M
m700 = DorinaNanoTR(cfg).cuda()
m700.load_state_dict(torch.load('models/dorina_nano_tr/dorina_nano_tr_10m_700m.pt', map_location='cuda', weights_only=True))
m700.eval()

# Load 800M
m800 = DorinaNanoTR(cfg).cuda()
m800.load_state_dict(torch.load('models/dorina_nano_tr/dorina_nano_tr_10m_800m.pt', map_location='cuda', weights_only=True))
m800.eval()

prompts_fix = [
    "Ben ev gidiyrum",
    "arkadaslar bugun toplanti saat kacta",
    "hic bir sey anlamadim",
    "bende gelicem bekleyin",
    "turkiyenin baskenti ankaradir",
    "bu aksam mac varmi acaba"
]

prompts_complete = [
    "Bugün dışarı",
    "Yapay zeka modelleri",
    "İstanbul'da trafik",
    "Yarın sabah erken kalkıp"
]

def generate_fix(model, text):
    prompt = f"<fix> {text} =>"
    enc = tok.encode(prompt).ids
    inp = torch.tensor([enc], device='cuda')
    gen = []
    with torch.no_grad():
        for _ in range(35):
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            nid = torch.argmax(logits[0, -1]).item()
            if nid == tok.token_to_id('<eos>') or '\n' in tok.decode([nid]):
                break
            gen.append(nid)
            inp = torch.cat([inp, torch.tensor([[nid]], device='cuda')], dim=1)
    return tok.decode(gen).strip()

def generate_chained(model, prompt, turns=4, step_tokens=4):
    curr = prompt
    seen_tokens = set()
    for _ in range(turns):
        enc = tok.encode(f"<complete> {curr}").ids
        inp = torch.tensor([enc], device='cuda')
        turn_added = []
        for _ in range(step_tokens):
            with torch.no_grad():
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                logits = logits[0, -1]
                # rep penalty
                for t_id in seen_tokens:
                    if logits[t_id] > 0:
                        logits[t_id] /= 1.3
                    else:
                        logits[t_id] *= 1.3
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_ids = torch.topk(probs, 5)
                nid = top_ids[0].item()
                if nid == tok.token_to_id('<eos>'):
                    break
                seen_tokens.add(nid)
                turn_added.append(nid)
                inp = torch.cat([inp, torch.tensor([[nid]], device='cuda')], dim=1)
        curr += tok.decode(turn_added)
        if '<eos>' in curr:
            break
    return curr

print("="*70)
print("🔍 700M vs 800M BİREBİR KARŞILAŞTIRMALI GELİŞİM TESTİ")
print("="*70)

print("\n1️⃣ İMLA / DE-DA / BOŞLUK DÜZELTME TESTLERİ (<fix>):")
print("-" * 70)
for p in prompts_fix:
    f700 = generate_fix(m700, p)
    f800 = generate_fix(m800, p)
    print(f"Girdi:       {p}")
    print(f"  700M Çıktı: {f700}")
    print(f"  800M Çıktı: {f800}")
    print("-" * 50)

print("\n2️⃣ CÜMLE TAMAMLAMA / ZİNCİRLEME ÜRETİM TESTLERİ (<complete>):")
print("-" * 70)
for p in prompts_complete:
    c700 = generate_chained(m700, p)
    c800 = generate_chained(m800, p)
    print(f"[Başlangıç]: \"{p}\"")
    print(f"  700M: {c700}")
    print(f"  800M: {c800}")
    print("-" * 50)
