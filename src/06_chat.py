import glob
import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================================
# 1. PARAMETRELER VE ÇOKLU VERİDEN SÖZLÜK OLUŞTURMA
# =====================================================================
context_length = 64
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data klasöründeki tüm txt dosyalarını okuyarak aynı sözlüğü (vocab) oluşturalım
dosya_listesi = glob.glob("data/*.txt")
text = ""
for dosya in dosya_listesi:
    with open(dosya, "r", encoding="utf-8") as f:
        text += f.read() + "\n"

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] # Sözlükte olmayan karakterleri ele
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

# =====================================================================
# 2. MODELİ YÜKLEME
# =====================================================================
model = DorinaGPT().to(device)
try:
    model.load_state_dict(torch.load("dorina_v1.pt", map_location=device))
    model.eval()
    print(f"✅ Dorina v2 ağırlıkları başarıyla yüklendi! (Vocab Size: {vocab_size})")
except FileNotFoundError:
    print("❌ Hata: 'dorina_v1.pt' bulunamadı!")
    exit(1)

# =====================================================================
# 3. İNCELEME VE OLASILIK ANALİZİ FONKSİYONU
# =====================================================================
def analyze_next_token(prompt_str, top_k=5):
    tokens = encode(prompt_str)
    if not tokens:
        print("⚠️ Uyarı: Girdiğiniz metindeki karakterler sözlükte bulunamadı!")
        return []

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

    predictions = []
    for p, idx_val in zip(top_probs, top_indices):
        char_val = itos[idx_val.item()]
        display_char = repr(char_val) if char_val != ' ' else "' '"
        percentage = p.item() * 100
        bar = "█" * int(percentage / 2)
        print(f"{display_char:<10} | %{percentage:5.2f}       | {bar}")
        predictions.append((char_val, p.item()))

    print("-" * 45)
    return predictions

# =====================================================================
# 4. İNTERAKTİF DÖNGÜ
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🔮 DORINA v2 - İNTERAKTİF OLASILIK VE KELİME TAMAMLAMA PANELİ")
    print("=" * 65)
    print("İstediğin bir İngilizce kelime veya cümle yaz (Örn: 'King', 'The', 'Once').")
    print("Çıkmak için 'q' yazıp Enter'a bas.\n")

    while True:
        user_input = input("Metin gir > ")
        if user_input.lower() == 'q':
            print("Görüşmek üzere, Talha!")
            break
        if not user_input:
            continue

        preds = analyze_next_token(user_input, top_k=5)
        print("\n" + "~" * 65 + "\n")
