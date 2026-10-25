import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

tok = Tokenizer.from_file('tokenizers/dorina_nano_tr_12k.json')
cfg = NanoTRConfig(vocab_size=12288, max_seq_len=256, d_model=256, n_layer=10, n_heads=8, n_kv_heads=4, head_dim=32, intermediate_size=688, tied_embeddings=True)
m800 = DorinaNanoTR(cfg).cuda()
m800.load_state_dict(torch.load('models/dorina_nano_tr/dorina_nano_tr_10m_800m.pt', map_location='cuda', weights_only=True))
m800.eval()

def test_fix(text):
    prompt = f"<fix> {text} =>"
    enc = tok.encode(prompt).ids
    inp = torch.tensor([enc], device='cuda')
    gen = []
    with torch.no_grad():
        for _ in range(40):
            out = m800(inp)
            logits = out[0] if isinstance(out, tuple) else out
            nid = torch.argmax(logits[0, -1]).item()
            if nid == tok.token_to_id('<eos>') or '\n' in tok.decode([nid]):
                break
            gen.append(nid)
            inp = torch.cat([inp, torch.tensor([[nid]], device='cuda')], dim=1)
    res = tok.decode(gen).strip()
    print(f"Girdi:       '{text}'\n800M Çıktı:  '{res}'\n")

print("--- DE / DA / Kİ TESTLERİ ---")
test_fix("sen de geliyormusun")
test_fix("sende bir seyler var")
test_fix("evdeki hesap carsıya uymadı")
test_fix("bende gelicem")

print("--- SORU EKİ Mİ / MI TESTLERİ ---")
test_fix("geldinmi acaba")
test_fix("bu dogrumu sence")

print("--- BÜYÜK / KÜÇÜK HARF & KESME İŞARETİ TESTLERİ ---")
test_fix("ahmetin arabasi ankarada kaldi")
test_fix("turkiyenin en guzel sehri istanbuldur")
