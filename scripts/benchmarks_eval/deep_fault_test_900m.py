import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

tok = Tokenizer.from_file('tokenizers/dorina_nano_tr_12k.json')
cfg = NanoTRConfig(vocab_size=12288, max_seq_len=256, d_model=256, n_layer=10, n_heads=8, n_kv_heads=4, head_dim=32, intermediate_size=688, tied_embeddings=True)

# Load 900M
m900 = DorinaNanoTR(cfg).cuda()
m900.load_state_dict(torch.load('models/dorina_nano_tr/dorina_nano_tr_10m_900m.pt', map_location='cuda', weights_only=True))
m900.eval()

def fix(text):
    prompt = f"<fix> {text} =>"
    enc = tok.encode(prompt).ids
    inp = torch.tensor([enc], device='cuda')
    gen = []
    with torch.no_grad():
        for _ in range(35):
            out = m900(inp)
            logits = out[0] if isinstance(out, tuple) else out
            nid = torch.argmax(logits[0, -1]).item()
            if nid == tok.token_to_id('<eos>') or '\n' in tok.decode([nid]):
                break
            gen.append(nid)
            inp = torch.cat([inp, torch.tensor([[nid]], device='cuda')], dim=1)
    return tok.decode(gen).strip()

def complete(prompt, max_tok=15):
    enc = tok.encode(f"<complete> {prompt}").ids
    inp = torch.tensor([enc], device='cuda')
    gen = []
    with torch.no_grad():
        for _ in range(max_tok):
            out = m900(inp)
            logits = out[0] if isinstance(out, tuple) else out
            nid = torch.argmax(logits[0, -1]).item()
            if nid == tok.token_to_id('<eos>'):
                break
            gen.append(nid)
            inp = torch.cat([inp, torch.tensor([[nid]], device='cuda')], dim=1)
    return prompt + tok.decode(gen)

print("="*70)
print("🔎 DORINA-NANO-TR (900M) DERİN HATA ARAMA RAPORU")
print("="*70)

# Kategori 1: de / da bağlacı
print("\n[TEST 1: de/da Bağlacı ve Ekleri]")
tests_deda = [
    ("bende gelecegim", "ben de geleceğim"),
    ("sende haklısın kardesim", "sen de haklısın kardeşim"),
    ("evde kimse yok", "evde kimse yok"),
    ("araba da kaldım", "arabada kaldım")
]
for raw, exp in tests_deda:
    out = fix(raw)
    status = "OK" if out.lower() == exp.lower() else "HATA"
    print(f"Girdi: {raw:<25} -> Cikti: {out:<25} | Beklenen: {exp:<22} [{status}]")

# Kategori 2: Soru Eki mi/mı
print("\n[TEST 2: Soru Eki (mi/mı/mu/mü)]")
tests_mi = [
    ("geldinmi acaba", "geldin mi acaba"),
    ("bu dogrumu sence", "bu doğru mu sence"),
    ("biliyormusun bunu", "biliyor musun bunu"),
    ("gordunmu onu", "gördün mü onu")
]
for raw, exp in tests_mi:
    out = fix(raw)
    status = "OK" if out.lower() == exp.lower() else "HATA"
    print(f"Girdi: {raw:<25} -> Cikti: {out:<25} | Beklenen: {exp:<22} [{status}]")

# Kategori 3: Sokak Dili -> Yazı Dili
print("\n[TEST 3: Sokak Dili (gelcem, yapcam)]")
tests_slang = [
    ("yarin gelcem", "yarın geleceğim"),
    ("ne yapcam simdi", "ne yapacağım şimdi"),
    ("nereye gidiyon", "nereye gidiyorsun")
]
for raw, exp in tests_slang:
    out = fix(raw)
    status = "OK" if out.lower() == exp.lower() else "HATA"
    print(f"Girdi: {raw:<25} -> Cikti: {out:<25} | Beklenen: {exp:<22} [{status}]")

# Kategori 4: Özel İsimler & Kesme İşareti
print("\n[TEST 4: Özel İsimler & Kesme İşareti]")
tests_names = [
    ("turkiyenin baskenti", "Türkiye'nin başkenti"),
    ("istanbulda hava guzel", "İstanbul'da hava güzel"),
    ("ahmetin telefonu", "Ahmet'in telefonu")
]
for raw, exp in tests_names:
    out = fix(raw)
    status = "OK" if out == exp else "HATA"
    print(f"Girdi: {raw:<25} -> Cikti: {out:<25} | Beklenen: {exp:<22} [{status}]")

# Kategori 5: Cümle Tamamlama (Döngü & Saçmalama Kontrolü)
print("\n[TEST 5: Cümle Tamamlama Akıcılığı & Döngü Testi]")
tests_gen = [
    "Bugün işe giderken",
    "Türkiye'nin en büyük gölü",
    "Yapay zeka teknolojisi",
    "Kahvaltıda çay ve"
]
for prompt in tests_gen:
    comp = complete(prompt, 18)
    print(f"Girdi: \"{prompt}\" -> Devamı: \"{comp}\"")
