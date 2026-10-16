import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================================
# 1. HİPERPARAMETRELER (DEVAM EĞİTİMİ)
# =====================================================================
batch_size = 32
context_length = 64
max_iters = 1500       # Devam eğitimi için 1500 adım yeterli
eval_interval = 300
learning_rate = 5e-4   # Mevcut ağırlıkları bozmamak için öğrenme oranını biraz düşürebiliriz!
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

torch.manual_seed(1337)

# =====================================================================
# 2. VERİ SETİ (İster aynı veriyi genişletelim, ister yeni bir metin verelim)
# =====================================================================
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - context_length, (batch_size,))
    x = torch.stack([data_source[i:i+context_length] for i in ix])
    y = torch.stack([data_source[i+1:i+context_length+1] for i in ix])
    return x.to(device), y.to(device)

# =====================================================================
# 3. MİMARİ TANIMLA (05 ve 06 ile aynı)
# =====================================================================
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

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

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
# 4. MODELİ OLUŞTUR VE ÖNCEKİ AĞIRLIKLARI (dorina_v1.pt) YÜKLE!
# =====================================================================
model = DorinaGPT().to(device)

try:
    model.load_state_dict(torch.load("dorina_v1.pt", map_location=device))
    print("✅ Önceki 'dorina_v1.pt' ağırlıkları başarıyla yüklendi!")
    print("🧠 Dorina sıfırdan değil, eski hafızasının üstüne koyarak eğitime başlayacak.")
except FileNotFoundError:
    print("❌ Hata: 'dorina_v1.pt' bulunamadı! Önce 05_dorina_gpt.py ile temel modeli eğitmelisin.")
    exit(1)

# =====================================================================
# 5. DEVAM EĞİTİMİ DÖNGÜSÜ (CONTINUAL TRAINING)
# =====================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n🚀 DORINA İÇİN DEVAM EĞİTİMİ (FINE-TUNING) BAŞLIYOR...")
baslangic_zamani = time.time()

for iter in range(max_iters + 1):
    if iter % eval_interval == 0:
        model.eval()
        losses = {}
        with torch.no_grad():
            for split in ['train', 'val']:
                loss_list = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    xb, yb = get_batch(split)
                    _, loss = model(xb, yb)
                    loss_list[k] = loss.item()
                losses[split] = loss_list.mean()
        model.train()
        print(f"Adım {iter:4d}/{max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# =====================================================================
# 6. YENİDEN KAYDETME VE TEST
# =====================================================================
torch.save(model.state_dict(), "dorina_v1.pt")
print("\n💾 Güncellenen model ağırlıkları tekrar 'dorina_v1.pt' dosyasına kaydedildi!")

print("\n--- 🤖 GÜNCELLENMİŞ DORINA'NIN YENİ SÖZLERİ ---")
model.eval()
baslangic_tokeni = torch.zeros((1, 1), dtype=torch.long, device=device)
cikti = decode(model.generate(baslangic_tokeni, max_new_tokens=300)[0].tolist())
print(cikti)
