import os
import sys
import glob
import torch
import torch.nn.functional as F
import tiktoken

# 1. Kök dizini (dorina-ai) sys.path'e bağla
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.model import DorinaGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# Model Seçimi (En son eğitilen run'ı otomatik yakalar)
# -----------------------------------------------------------------------------
def get_best_model():
    candidates = []
    runs_dir = os.path.join(ROOT_DIR, "runs")
    for root, _, files in os.walk(runs_dir):
        for f in files:
            if f in ["best.pt", "model.pt"]:
                candidates.append(os.path.normpath(os.path.join(root, f)))
    candidates = sorted(list(set(candidates)))
    if not candidates:
        print(f"❌ '{runs_dir}' klasöründe test edilecek model bulunamadı!")
        sys.exit(1)
    
    print("\n📦 Bulunan Modeller:")
    for idx, c in enumerate(candidates, 1):
        rel_c = os.path.relpath(c, ROOT_DIR)
        print(f" [{idx}] ➔ {rel_c}")
    choice = input(f"\nTest edilecek model [Varsayılan sonuncu: {len(candidates)}]: ").strip()
    idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(candidates) else len(candidates) - 1
    return candidates[idx]

model_path = get_best_model()
print(f"\n🚀 Yükleniyor: {model_path}")

raw_dict = torch.load(model_path, map_location=device)
state_dict = raw_dict.get("model_state_dict", raw_dict.get("model", raw_dict))
saved_cfg = raw_dict.get("config", {})

# Hiperparametreleri checkpoint'ten veya ağırlık tensörlerinden dinamik belirle
if saved_cfg:
    vocab_size = saved_cfg.get("vocab_size", enc.n_vocab)
    context_length = saved_cfg.get("context_length", 256)
    n_embd = saved_cfg.get("n_embd", 512)
    n_head = saved_cfg.get("n_head", 8)
    n_layer = saved_cfg.get("n_layer", 10)
else:
    # Ağırlık tensörlerinden çıkarım yap
    wte = state_dict.get("transformer.wte.weight", state_dict.get("token_embedding.weight"))
    wpe = state_dict.get("transformer.wpe.weight", state_dict.get("position_embedding.weight"))
    vocab_size, n_embd = wte.shape if wte is not None else (enc.n_vocab, 512)
    context_length = wpe.shape[0] if wpe is not None else 256
    layer_indices = {int(k.split(".")[2]) for k in state_dict.keys() if k.startswith("transformer.h.")}
    n_layer = len(layer_indices) if layer_indices else 10
    n_head = 8 if n_embd >= 256 else 4

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

# -----------------------------------------------------------------------------
# Çıkarım Fonksiyonları
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample(prompt, max_tokens=60, temp=0.7, rep_pen=1.15):
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_tokens):
        x_cond = x if x.size(1) <= context_length else x[:, -context_length:]
        out = model(x_cond)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :]
        
        # Repetition penalty
        for tid in set(x[0].tolist()):
            logits[0, tid] = logits[0, tid] / rep_pen if logits[0, tid] > 0 else logits[0, tid] * rep_pen
            
        logits = logits / max(temp, 1e-5)
        v, _ = torch.topk(logits, min(35, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
        
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat((x, next_id), dim=1)
        
        if next_id.item() == enc.eot_token:
            break
            
    return enc.decode(x[0].tolist())

@torch.no_grad()
def get_top_k_words(prompt, k=3):
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    x_cond = x if x.size(1) <= context_length else x[:, -context_length:]
    out = model(x_cond)
    logits = out[0] if isinstance(out, tuple) else out
    probs = F.softmax(logits[:, -1, :], dim=-1)[0]
    
    control_tokens = [enc.encode_ordinary(s)[0] for s in ["\n", "\r", "\n\n"] if len(enc.encode_ordinary(s)) == 1]
    for tid in control_tokens:
        probs[tid] = 0.0
        
    top_p, top_i = torch.topk(probs, 50)
    results = []
    for i in range(len(top_i)):
        word = enc.decode([top_i[i].item()])
        clean_word = word.strip()
        if clean_word:
            results.append((clean_word, top_p[i].item() * 100))
        if len(results) == k:
            break
    return results

# -----------------------------------------------------------------------------
# Hazır Kalıp Test Bataryası (Benchmark Suite)
# -----------------------------------------------------------------------------
BENCHMARK_PROMPTS = [
    {
        "kategori": "🌍 Fen & Doğa Mantığı",
        "aciklama": "Suyun donma durumunu kavrıyor mu?",
        "prompt": "When water gets extremely cold, it turns into",
        "kelime_testi": True
    },
    {
        "kategori": "🚀 Uzay & Bilim",
        "aciklama": "Güneşin bir yıldız olduğunu biliyor mu?",
        "prompt": "In the solar system, the Sun is a bright",
        "kelime_testi": True
    },
    {
        "kategori": "📖 Masal & Hikaye Akışı",
        "aciklama": "Olay örgüsünü sürdürüyor mu?",
        "prompt": "Once upon a time, a small rabbit found a golden key. The rabbit",
        "kelime_testi": False
    },
    {
        "kategori": "🧠 Neden-Sonuç İlişkisi",
        "aciklama": "Açlık durumunda ne yapacağını biliyor mu?",
        "prompt": "The puppy was very hungry, so it started to eat some",
        "kelime_testi": True
    }
]

def run_suite():
    print("\n" + "=" * 75)
    print("🎯 DORINA AI - HAZIR KALIP DOĞRULAMA VE TEST PANELİ")
    print("=" * 75)
    
    total_tests = len(BENCHMARK_PROMPTS)
    for idx, test in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"\n[{idx}/{total_tests}] Kategori: {test['kategori']}")
        print(f"📌 Hedef: {test['aciklama']}")
        print(f"👉 Fitil Metin: \"{test['prompt']}\"")
        
        if test.get("kelime_testi"):
            top3 = get_top_k_words(test['prompt'], k=3)
            sonuc_str = ", ".join([f"'{w}' (%{p:.1f})" for w, p in top3])
            print(f"📊 En Olası Kelimeler: {sonuc_str}")
        
        output = sample(test['prompt'], max_tokens=50, temp=0.7)
        print("📝 Model Çıktısı:")
        print(f"   {output.strip()}")
        print("-" * 75)

if __name__ == "__main__":
    run_suite()