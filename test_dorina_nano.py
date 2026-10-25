import os
import sys
import torch
import torch.nn.functional as F
from scripts.dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

# Force UTF-8 terminal encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# Checkpoint resolution: prioritize latest milestone
MODEL_PATHS = [
    "models/dorina_nano_tr/dorina_nano_tr_10m_700m.pt",
    "models/dorina_nano_tr/dorina_nano_tr_10m_600m.pt",
    "models/dorina_nano_tr/dorina_nano_tr_10m_500m.pt",
    "models/dorina_nano_tr/dorina_nano_tr_10m.pt"
]

MODEL_PATH = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

TOKENIZER_PATH = "tokenizers/dorina_nano_tr_12k.json"

print("=" * 70)
print("     DORINA-NANO-TR (10.4M) İNTERAKTİF TEST & ANALİZ PANELİ")
print("=" * 70)
print(f"Yüklenen Model Checkpoint'i: {MODEL_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

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

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model = DorinaNanoTR(cfg).to(device)
model.load_state_dict(state_dict)
model.eval()

print(f"Model Hazır! Çalışma Ortamı: {device} | Kelime Dağarcığı: 12,288 BPE")
print("=" * 70)

def predict_next_tokens(text_prompt, top_k=5):
    """Metnin sonrasına gelebilecek aday tokenleri ve % olasılıklarını hesaplar."""
    full_prompt = f"<complete> {text_prompt}"
    input_ids = tokenizer.encode(full_prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        res = model(x)
        logits = res[0] if isinstance(res, tuple) else res
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, top_k)
        
    candidates = []
    for prob, idx in zip(top_probs, top_indices):
        tok_str = tokenizer.decode([idx.item()])
        percentage = prob.item() * 100
        candidates.append((tok_str, percentage))
    return candidates

def fix_sentence(corrupted_text, max_tokens=40):
    """Bozuk / ASCII / İmla / ek hatalarını düzeltir."""
    fix_prompt = f"<fix> {corrupted_text} =>"
    input_ids = tokenizer.encode(fix_prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated_ids = []
    for _ in range(max_tokens):
        x_cond = x if x.size(1) <= cfg.max_seq_len else x[:, -cfg.max_seq_len:]
        with torch.no_grad():
            res = model(x_cond)
            logits = res[0] if isinstance(res, tuple) else res
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat((x, next_id), dim=1)
            tok_id = next_id.item()
            if tok_id == tokenizer.token_to_id("<eos>") or tok_id == 3:
                break
            decoded_char = tokenizer.decode([tok_id])
            if "\n" in decoded_char:
                break
            generated_ids.append(tok_id)
            
    return tokenizer.decode(generated_ids).strip()

def loop_sentence_expansion(start_text, rounds=4, words_per_round=5, temp=0.5, rep_penalty=1.25):
    """
    Zincirleme / Döngülü Cümle Genişletme (Chained Loop Expansion).
    Her turda 4-5 kelime üretir, üretilen parçayı ana metne ekler,
    tekrar cezası (repetition penalty) uygulayarak papağan döngülerini engeller.
    """
    current_text = start_text
    print(f"\n[Başlangıç]: \"{current_text}\"")
    print("-" * 55)

    seen_token_ids = set()
    initial_ids = tokenizer.encode(current_text).ids
    for t_id in initial_ids:
        seen_token_ids.add(t_id)

    for r in range(1, rounds + 1):
        full_prompt = f"<complete> {current_text}"
        input_ids = tokenizer.encode(full_prompt).ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        round_token_ids = []
        for _ in range(words_per_round):
            x_cond = x if x.size(1) <= cfg.max_seq_len else x[:, -cfg.max_seq_len:]
            with torch.no_grad():
                res = model(x_cond)
                logits = (res[0] if isinstance(res, tuple) else res)[:, -1, :].clone()
                
                # Repetition Penalty: Daha önce geçen tokenlerin logitlerini baskıla
                for sid in seen_token_ids:
                    if logits[0, sid] > 0:
                        logits[0, sid] /= rep_penalty
                    else:
                        logits[0, sid] *= rep_penalty
                
                logits = logits / temp
                probs = F.softmax(logits, dim=-1)
                
                # Top-k kısıtlaması (en mantıklı 30 aday arasından seç)
                v, _ = torch.topk(probs, 30)
                probs[probs < v[:, [-1]]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_id = torch.multinomial(probs, num_samples=1)
                tok_id = next_id.item()
                
                if tok_id == tokenizer.token_to_id("<eos>") or tok_id == 3:
                    break
                    
                round_token_ids.append(tok_id)
                seen_token_ids.add(tok_id)
                x = torch.cat((x, next_id), dim=1)

        round_chunk = tokenizer.decode(round_token_ids)
        current_text = (current_text + round_chunk).strip()
        print(f"  Tur {r} ➔ Eklenen: {repr(round_chunk)} | Metin: \"{current_text}\"")

    print("-" * 55)
    return current_text

def interactive_loop():
    while True:
        print("\n" + "=" * 50)
        print(" [1] ➔ Sonraki Kelime Tahmini (% Olasılık Grafiği)")
        print(" [2] ➔ Cümle / İmla / ASCII / de-da Düzeltme (<fix>)")
        print(" [3] ➔ ZİNCİRLEME (LOOP) Cümle Genişletme (Tekrarsız)")
        print(" [q] ➔ Çıkış")
        print("=" * 50)
        
        secim = input("Seçiminiz (1/2/3/q): ").strip()
        
        if secim.lower() == 'q':
            print("Çıkılıyor...")
            break
            
        elif secim == '1':
            metin = input("\nMetin girin (Örn: 'Bugün hava çok'): ").strip()
            if not metin:
                continue
            candidates = predict_next_tokens(metin, top_k=6)
            print(f"\n'{metin}' sonrasına gelebilecek en olası adaylar:")
            print("-" * 55)
            for rank, (tok, pct) in enumerate(candidates, 1):
                tok_disp = repr(tok) if tok.strip() == "" else f"'{tok}'"
                bar = "█" * int(pct / 4)
                print(f"  {rank}. {tok_disp:<18} %{pct:5.2f}  {bar}")
            print("-" * 55)
            
        elif secim == '2':
            metin = input("\nDüzeltilecek metni girin: ").strip()
            if not metin:
                continue
            duzeltilmis = fix_sentence(metin)
            print("-" * 55)
            print(f"Girdi:       {metin}")
            print(f"Düzeltilmiş: {duzeltilmis}")
            print("-" * 55)
            
        elif secim == '3':
            metin = input("\nBaşlangıç cümlesi girin (Örn: 'Bugün dışarı'): ").strip()
            if not metin:
                continue
            final_cümle = loop_sentence_expansion(metin, rounds=4, words_per_round=5, temp=0.5, rep_penalty=1.25)
            print(f"\n🎯 NİHAİ ZİNCİRLEME CÜMLE:\n\"{final_cümle}\"")
            print("=" * 55)
            
        else:
            print("Geçersiz seçim! Lütfen 1, 2, 3 veya q girin.")

if __name__ == "__main__":
    interactive_loop()
