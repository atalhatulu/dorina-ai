import os
import sys
import torch
import torch.nn.functional as F
import tiktoken

# Proje kök dizinini Python yoluna ekle ve asıl modeli import et
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Windows terminal utf-8 uyumluluğu
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.model import DorinaGPT

# -----------------------------------------------------------------------------
# Model Dosyası Seçici Menü
# -----------------------------------------------------------------------------

def select_model_file():
    print("\n" + "=" * 65)
    print("📁 MODEL ARAMA VE SEÇİM MENÜSÜ")
    print("=" * 65)
    
    candidates = []
    for root, dirs, files in os.walk("."):
        if ".venv" in root or ".git" in root:
            continue
        for file in files:
            if file.endswith(".pt"):
                full_path = os.path.normpath(os.path.join(root, file))
                candidates.append(full_path)
                
    candidates = sorted(list(set(candidates)))

    if not candidates:
        print("❌ Dizinde hiçbir .pt uzantılı model ağırlığı bulunamadı!")
        return None

    print("Bulunan kayıtlı model ağırlıkları:\n")
    for idx, path in enumerate(candidates, 1):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f" [{idx}] ➔ {path} ({size_mb:.2f} MB)")
        
    print("\n [q] ➔ Çıkış")
    print("-" * 65)
    
    while True:
        choice = input(f"Yüklemek istediğiniz model numarasını girin [1-{len(candidates)}] (veya q): ").strip().lower()
        if choice == "q":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print("⚠️ Geçersiz seçim! Lütfen listedeki bir numarayı girin.")

# -----------------------------------------------------------------------------
# Ana Yürütme ve Model Yükleme
# -----------------------------------------------------------------------------

selected_ckpt = select_model_file()
if not selected_ckpt:
    print("İşlem iptal edildi. Çıkılıyor.")
    sys.exit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

print(f"\n📦 Seçilen model yükleniyor: {selected_ckpt}")
ckpt = torch.load(selected_ckpt, map_location=device)
state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

# DorinaGPT mimari değişkenlerini doğrudan state_dict üzerinden otomatik oku
tok_emb = state_dict.get("token_embedding.weight", None)
pos_emb = state_dict.get("position_embedding.weight", None)

if tok_emb is not None and pos_emb is not None:
    vocab_size, n_embd = tok_emb.shape
    context_length, _ = pos_emb.shape
    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("blocks."):
            idx = int(k.split(".")[1])
            layer_indices.add(idx)
    n_layer = len(layer_indices) if layer_indices else 8
    n_head = 8 if n_embd >= 256 else 4
else:
    vocab_size, context_length, n_layer, n_embd, n_head = enc.n_vocab, 256, 8, 256, 8

print(f"🔎 Tespit edilen mimari: {n_layer} Katman, {n_embd} Boyut, {n_head} Kafa, {context_length} Kontekst")

# Asıl DorinaGPT örneğini oluştur ve ağırlıkları yükle
model = DorinaGPT(
    vocab_size=vocab_size,
    context_length=context_length,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=0.0
).to(device)

model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"✓ Model başarıyla {device.upper()} üzerine yüklendi!\n")

# -----------------------------------------------------------------------------
# Üretim Fonksiyonları
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_text(prompt, max_tokens=80, temperature=0.7, top_k=40, repetition_penalty=1.15):
    input_ids = enc.encode_ordinary(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_tokens):
        x_cond = x if x.size(1) <= context_length else x[:, -context_length:]
        out = model(x_cond)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :]  # Şekil: (1, vocab_size)
        
        # --- REPETITION PENALTY (Tekrar Cezası) ---
        if repetition_penalty != 1.0:
            # Üretilmiş olan benzersiz token'ları topla
            used_tokens = set(x[0].tolist())
            for token_id in used_tokens:
                # Logit pozitifse cezalandırmak için böleriz, negatifse çarparız
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
        # -----------------------------------------
        
        logits = logits / max(temperature, 1e-5)
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_id), dim=1)
        
        # Standart EOT (<|endoftext|>) kontrolü
        if next_id.item() == enc.eot_token:
            break
            
        # ChatML <|im_end|> durma kontrolü (oluşan son metin üzerinden)
        curr_decoded = enc.decode(x[0].tolist())
        if "<|im_end|>" in curr_decoded:
            break
            
    full_output = enc.decode(x[0].tolist())
    # Eğer çıktı içinde <|im_end|> varsa o noktada temizce kes
    if "<|im_end|>" in full_output:
        full_output = full_output.split("<|im_end|>")[0].strip()
    return full_output

@torch.no_grad()
def predict_next_words(prompt, top_n=5):
    if not prompt:
        return
        
    input_ids = enc.encode_ordinary(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    x_cond = x if x.size(1) <= context_length else x[:, -context_length:]
    
    out = model(x_cond)
    logits = out[0] if isinstance(out, tuple) else out
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)[0]
    
    # Yeni satır (\n), satır başı (\r) ve saf boşlukları olasılık havuzunda sıfırla
    # Böylece model biçimlendirme yerine bir sonraki KELİMEYİ göstermek zorunda kalır
    newline_tokens = [enc.encode_ordinary(s)[0] for s in ["\n", "\r", "\n\n", " ", "  "] if len(enc.encode_ordinary(s)) == 1]
    for tid in newline_tokens:
        probs[tid] = 0.0
        
    # Kalan gerçek kelimelerden en yüksek 50'sini çek
    top_probs, top_indices = torch.topk(probs, 50)
    
    print(f"\n🔎 Girdi: \"{prompt}\"")
    print("Sıradaki en olası kelimeler:")
    
    shown = 0
    for i in range(len(top_indices)):
        token_id = top_indices[i].item()
        word = enc.decode([token_id])
        pct = top_probs[i].item() * 100
        
        # Görünmez veya boş string kontrolü
        if not word.strip():
            continue
            
        print(f"  {shown+1}. '{word}' ➔ %{pct:.2f}")
        shown += 1
        
        if shown == top_n:
            break
    print()

def main():
    while True:
        print("\n" + "=" * 40)
        print("🎮 DORINA LAB - ÇALIŞMA MODU")
        print("=" * 40)
        print(" [1] ➔ Cümle / Hikaye Tamamlama (Serbest Üretim)")
        print(" [2] ➔ Sonraki Kelime Tahmin Testi (Olasılık Dağılımı)")
        print(" [3] ➔ 💬 Dorina Asistan ile Sohbet (ChatML)")
        print(" [q] ➔ Çıkış")
        
        mode = input("Seçiminiz (1/2/3/q): ").strip().lower()
        
        if mode == "1":
            prompt = input("\nBaşlangıç metni girin: ").strip()
            if not prompt: 
                continue
            t_input = input("Sıcaklık (0.2: kesin/katı, 0.7: dengeli, 1.0: yaratıcı) [0.7]: ").strip()
            temp = float(t_input) if t_input else 0.7
            tok_input = input("Üretilecek token sayısı [80]: ").strip()
            tokens = int(tok_input) if tok_input else 80
            
            print("\n⏳ Üretiliyor...\n" + "-" * 50)
            output = generate_text(prompt, max_tokens=tokens, temperature=temp)
            print(output)
            print("-" * 50)
            
        elif mode == "2":
            prompt = input("\nCümleyi yazın: ")
            if not prompt: 
                continue
            predict_next_words(prompt)

        elif mode == "3":
            print("\n" + "=" * 50)
            print("💬 DORINA ASİSTAN SOHBET MODU (Çıkış için 'exit' yazın)")
            print("=" * 50)
            while True:
                user_msg = input("\n👤 Sen: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    break
                chat_prompt = f"<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n"
                print("\n🤖 Dorina: ", end="", flush=True)
                full_reply = generate_text(chat_prompt, max_tokens=120, temperature=0.6, top_k=40)
                # Sadece asistanın yanıt kısmını göster
                if chat_prompt in full_reply:
                    reply_clean = full_reply[len(chat_prompt):].strip()
                else:
                    reply_clean = full_reply.replace(chat_prompt, "").strip()
                if "<|im_end|>" in reply_clean:
                    reply_clean = reply_clean.split("<|im_end|>")[0].strip()
                print(reply_clean)
            
        elif mode == "q":
            print("Görüşmek üzere!")
            break

if __name__ == "__main__":
    main()