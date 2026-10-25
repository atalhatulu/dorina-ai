import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

tok = Tokenizer.from_file('tokenizers/dorina_nano_tr_12k.json')
cfg = NanoTRConfig(
    vocab_size=12288,
    max_seq_len=256,
    d_model=256,
    n_layer=10,
    n_heads=8,
    n_kv_heads=4,
    head_dim=32,
    intermediate_size=688,
    tied_embeddings=True
)
ckpt = torch.load('models/dorina_nano_tr/dorina_nano_tr_10m_800m.pt', map_location='cuda', weights_only=True)
model = DorinaNanoTR(cfg)
model.load_state_dict(ckpt)
model.cuda().eval()

print("="*60)
print(f"DORINA-NANO-TR (800M Token Checkpoint Testi)")
print("="*60)

def predict_next(text, top_k=5):
    enc = tok.encode(text).ids
    inp = torch.tensor([enc], device='cuda')
    with torch.no_grad():
        out = model(inp)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k)
    res = [(tok.decode([i.item()]), f"{p.item()*100:.1f}%") for i, p in zip(top_ids, top_probs)]
    print(f"\n[Girdi]: \"{text}\"")
    for word, pct in res:
        print(f"   -> '{word}': {pct}")

predict_next("Bugün hava çok ")
predict_next("Ben eve ")
predict_next("Yarın sabah saat ")
predict_next("Kahve içmek ")

def test_fix(text):
    prompt = f"<fix> {text} =>"
    enc = tok.encode(prompt).ids
    inp = torch.tensor([enc], device='cuda')
    out_ids = []
    with torch.no_grad():
        for _ in range(30):
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[0, -1]
            nid = torch.argmax(logits).item()
            if nid == tok.token_to_id('<eos>') or '\n' in tok.decode([nid]):
                break
            out_ids.append(nid)
            inp = torch.cat([inp, torch.tensor([[nid]], device='cuda')], dim=1)
    fixed = tok.decode(out_ids).strip()
    print(f"  [Hatalı]: {text:<26} -> [Düzeltilmiş]: {fixed}")

print("\n" + "="*60)
print("🛠️ OTO-DÜZELTME (<fix>) TESTLERİ:")
print("="*60)
test_fix("Ben ev gidiyrum")
test_fix("geliyorum ams yoldayim")
test_fix("yarin gorusuruz insallah")
test_fix("seni cok seviyorum canim")
test_fix("nerde kaldin bekliyorum")
