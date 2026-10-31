import sys
import io
import time
from pathlib import Path

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure paths
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

print("="*75)
print("🏋️ DORINA-NANO-TR (1.0B TOKEN FİNAL) - KAPSAMLI VE ZORLU STRES TESTİ")
print("="*75)

tok = Tokenizer.from_file('tokenizers/dorina_nano_tr_12k.json')
cfg = NanoTRConfig(
    vocab_size=12288, max_seq_len=256, d_model=256,
    n_layer=10, n_heads=8, n_kv_heads=4, head_dim=32,
    intermediate_size=688, tied_embeddings=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DorinaNanoTR(cfg).to(device)

model_path = 'models/dorina_nano_tr/dorina_nano_tr_10m.pt'
ckpt = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.eval()

print(f"Cihaz: {device.upper()} | Model Ağırlığı: {model_path} Yüklendi!\n")

def run_fix(prompt_text, max_tokens=40):
    inp_str = f"<fix> {prompt_text} =>"
    input_ids = tok.encode(inp_str).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    gen_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            next_id = torch.argmax(logits[0, -1, :]).item()
            if next_id == tok.token_to_id("<eos>") or '\n' in tok.decode([next_id]):
                break
            gen_tokens.append(next_id)
            x = torch.cat((x, torch.tensor([[next_id]], device=device)), dim=1)
            
    latency_ms = (time.perf_counter() - t0) * 1000
    res = tok.decode(gen_tokens).strip()
    return res, latency_ms

def run_autocomplete(prefix, next_k=5):
    inp_str = f"<complete> {prefix}"
    input_ids = tok.encode(inp_str).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_ids = torch.topk(probs, next_k)
    candidates = [(tok.decode([idx.item()]), round(p.item() * 100, 1)) for p, idx in zip(top_probs, top_ids)]
    return candidates

def run_chained_gen(prefix, turns=4, tokens_per_turn=4, rep_penalty=1.35):
    curr = prefix
    seen_ids = set()
    for _ in range(turns):
        inp_str = f"<complete> {curr}"
        input_ids = tok.encode(inp_str).ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        turn_added = []
        for _ in range(tokens_per_turn):
            with torch.no_grad():
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                l = logits[0, -1, :].clone()
                for sid in seen_ids:
                    if l[sid] > 0:
                        l[sid] /= rep_penalty
                    else:
                        l[sid] *= rep_penalty
                probs = F.softmax(l, dim=-1)
                top_p, top_i = torch.topk(probs, 5)
                nid = top_i[0].item()
                if nid == tok.token_to_id("<eos>") or '\n' in tok.decode([nid]):
                    break
                seen_ids.add(nid)
                turn_added.append(nid)
                x = torch.cat((x, torch.tensor([[nid]], device=device)), dim=1)
        if not turn_added:
            break
        curr += tok.decode(turn_added)
    return curr

# -------------------------------------------------------------
# TEST KATEGORİLERİ
# -------------------------------------------------------------

TEST_SUITE = {
    "1. AGRESİF KLAVYE TYPO VE BİLEŞİK BOZUKLUK": [
        "bgn cok yorgunm eve gdicem",
        "hadi arkadslar toplntiya geln",
        "yazilkm motoru mukemml calisyr",
        "bu aksm disari cikalm mi"
    ],
    "2. DE/DA VE Kİ BAĞLAÇ / EK AYRIMI (BAĞLAM SINAVI)": [
        "bende sizinle gelecegim",
        "evde kimse kalmadi",
        "sende bir gariplik var gibi",
        "araba da unutmusum cantayi",
        "seninki evde mi bekliyor"
    ],
    "3. SORU EKİ (-mı / -mi / -mu / -mü) VE BİTİŞİK YAZIM": [
        "sen beni anladinmi gercekten",
        "bunu yapabilirmiyiz acaba",
        "hava sogukmu disarida",
        "geldin mi gelmedinmi"
    ],
    "4. SOKAK DİLİ / YAZIM STANDARDI (İSTANBUL TÜRKÇESİ)": [
        "yarin kesin geliyom",
        "napiyon su an neredesin",
        "ne yapcam bilmiyom",
        "gidiyom ben gorusuruz"
    ],
    "5. ÖZEL İSİM, BÜYÜK HARF VE KESME İŞARETİ": [
        "turkiyenin baskenti ankaradir",
        "istanbulda bogazici koprusu cok kalabalik",
        "ahmetin arabasi bozuldu",
        "ataturk turkiyenin kurucusudur"
    ]
}

for cat_name, test_cases in TEST_SUITE.items():
    print(f"\n📁 {cat_name}")
    print("-" * 75)
    for t in test_cases:
        res, lat = run_fix(t)
        print(f"  [Girdi]:  \"{t}\"")
        print(f"  [Çıktı]:  \"{res}\"  (⏱️ {lat:.1f}ms)")
        print()

print("\n" + "="*75)
print("📊 6. CANLI KLAVYE ANLIK TAHMİN (NEXT-TOKEN PROBABILITIES)")
print("="*75)
autocomplete_tests = [
    "Bugün hava oldukça ",
    "Türkiye Büyük Millet ",
    "Yarın sabah erken ",
    "Ben bu akşam eve "
]
for p in autocomplete_tests:
    preds = run_autocomplete(p, 4)
    print(f"  [Prefix]: \"{p}\"")
    p_str = " | ".join([f"'{w}': %{pct}" for w, pct in preds])
    print(f"   -> Adaylar: {p_str}\n")

print("\n" + "="*75)
print("⛓️ 7. ZİNCİRLEME CÜMLE ÜRETİMİ (REPETITION PENALTY & DÖNGÜ TESTİ)")
print("="*75)
gen_tests = [
    "Bugün dışarı çıkıp",
    "Yapay zeka modelleri günümüzde",
    "İstanbul'da sabah saatlerinde trafik"
]
for g in gen_tests:
    chained = run_chained_gen(g, turns=4, tokens_per_turn=5)
    print(f"  [Tohum]: \"{g}\"")
    print(f"  [Nihai Metin]: \"{chained}\"\n")
