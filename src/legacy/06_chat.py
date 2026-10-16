import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# =====================================================================
# 1. PARAMETRELER VE SÖZLÜK YAPISI (TINYSTORIES UYUMLU)
# =====================================================================
context_length = 64
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TinyStories veri setinden veya data klasöründen vocab boyutunu eşitleyelim
data_path = "data/TinyStoriesV2-GPT4-train.txt"
if os.path.exists(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        sample_text = f.read(2_000_000)
    chars = sorted(list(set(sample_text)))
else:
    # Yedek olarak data klasöründeki tüm txt'leri okuyalım
    dosya_listesi = glob.glob("data/*.txt")
    text = ""
    for dosya in dosya_listesi:
        with open(dosya, "r", encoding="utf-8", errors='ignore') as f:
            text += f.read() + "\n"
    chars = sorted(list(set(text)))

vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l if i < len(itos)])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DorinaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =====================================================================
# 2. MODELİ YÜKLEME (dorina_tinystories.pt)
# =====================================================================
model = DorinaGPT().to(device)
model_file = "dorina_tinystories.pt"

if not os.path.exists(model_file):
    model_file = "dorina_v1.pt" # Yedek olarak eski model

try:
    model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
    model.eval()
    print(f"✅ Model başarıyla yüklendi: {model_file} (Vocab Size: {vocab_size}, Device: {device})")
except Exception as e:
    print(f"❌ Model yüklenirken hata oluştu: {e}")
    exit(1)

# =====================================================================
# 3. ANALİZ VE HİKAYE ÜRETME FONKSİYONLARI
# =====================================================================
def analyze_next_token(prompt_str, top_k=5):
    tokens = encode(prompt_str)
    if not tokens:
        print("⚠️ Uyarı: Girdiğiniz metindeki karakterler sözlükte bulunamadı!")
        return
        
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    if idx.shape[1] > context_length:
        idx = idx[:, -context_length:]

    with torch.no_grad():
        logits, _ = model(idx)
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
    print(f"\n📝 Girdi Metin: \"{prompt_str}\"")
    print(f"🔍 Dorina'nın Sonraki Karakter Tahminleri (Top {top_k}):")
    print("-" * 45)
    print(f"{'Karakter':<10} | {'Olasılık (%)':<12} | Görsel Bar")
    print("-" * 45)
    
    for p, idx_val in zip(top_probs, top_indices):
        char_val = itos[idx_val.item()]
        display_char = repr(char_val) if char_val != ' ' else "' '"
        percentage = p.item() * 100
        bar = "█" * int(percentage / 2)
        print(f"{display_char:<10} | %{percentage:5.2f}       | {bar}")
    print("-" * 45)

def generate_story(prompt_str, max_tokens=300):
    tokens = encode(prompt_str)
    if not tokens:
        tokens = [0] # Boşsa sıfırdan başla
        
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"\n📖 '{prompt_str}' ile hikaye üretiliyor...\n" + "="*50)
    with torch.no_grad():
        output_idx = model.generate(idx, max_new_tokens=max_tokens)
        story_text = decode(output_idx[0].tolist())
        print(story_text)
    print("="*50 + "\n")

# =====================================================================
# 4. İNTERAKTİF MENÜ
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🔮 DORINA TINYSTORIES - İNTERAKTİF HİKAYE VE ANALİZ PANELİ")
    print("=" * 65)
    print("1 -> Bir kelime/cümle yazıp sonraki harf olasılıklarını gör")
    print("2 -> Bir başlangıç metni verip Dorina'ya uzun hikaye yazdır")
    print("q -> Çıkış\n")

    while True:
        secim = input("Seçiminiz (1 / 2 / q) > ").strip().lower()
        if secim == 'q':
            print("Görüşmek üzere, Talha!")
            break
        elif secim == '1':
            user_input = input("Analiz edilecek metin > ")
            if user_input:
                analyze_next_token(user_input, top_k=5)
        elif secim == '2':
            user_input = input("Hikaye başlangıcı (Örn: 'Once upon a time'): ")
            if not user_input:
                user_input = "Once upon a time"
            generate_story(user_input, max_tokens=300)
        else:
            print("Geçersiz seçim, lütfen 1, 2 veya q girin.")