import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================================
# 1. HİPERPARAMETRELER (DORINA v1 KONFİGÜRASYONU)
# Ryzen 5 3600 CPU üzerinde ~1-2 dakikada eğitilecek tatlı nokta!
# =====================================================================
batch_size = 32        # Aynı anda işlenecek metin parçası sayısı
context_length = 64    # Dorina bir sonraki harfi tahmin ederken 64 karakter geriye bakacak
max_iters = 3000       # Toplam eğitim adımı (iterasyon)
eval_interval = 500    # Kaç adımda bir Loss (Hata) raporu verilsin?
learning_rate = 1e-3   # AdamW için ideal başlangıç öğrenme oranı (0.001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100       # Doğrulama (Validation) testi için ortalama alınacak adım sayısı
n_embd = 64            # Embedding vektör boyutu (Her token 64 sayı ile temsil edilecek)
n_head = 4             # Paralel çalışacak Attention başlığı sayısı (64 / 4 = her başlık 16 boyutlu)
n_layer = 4            # Üst üste dizilecek Transformer Bloğu (Katman) sayısı
dropout = 0.1          # Ezberlemeyi (Overfitting) önlemek için %10 nöron kapatma oranı

torch.manual_seed(1337)

# =====================================================================
# 2. VERİ SETİNİ HAZIRLAMA
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

@torch.no_grad() # Değerlendirme sırasında türev (gradient) hesaplama, hız kazan!
def estimate_loss(model):
    out = {}
    model.eval() # Modeli test moduna al (Dropout devre dışı kalır)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Modeli tekrar eğitim moduna al
    return out

# =====================================================================
# 3. DORINA GPT MİMARİSİ (TRANSFORMER BLOKLARI)
# =====================================================================

class Head(nn.Module):
    """Tek bir Causal Self-Attention Başlığı"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal Mask (tril) modelin parametresi değildir, sabit bir tampondur (buffer)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Dikkat skorlarını hesapla (Scaled Dot-Product Attention)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Geleceği maskele
        wei = F.softmax(wei, dim=-1) # Olasılıklara çevir
        wei = self.dropout(wei)

        # Value vektörleriyle çarp
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Birden fazla Attention başlığını paralel çalıştırıp sonuçları birleştirir"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Tüm başlıkların çıktılarını yan yana (son boyutta) birleştir
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Attention ile toplanan bilgileri nöronlarda işleyen (düşünen) katman"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Boyutu 4 katına çıkar (64 -> 256)
            nn.GELU(),                     # Modern LLM aktivasyon fonksiyonu!
            nn.Linear(4 * n_embd, n_embd), # Tekrar 64 boyuta indir
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Tam bir Transformer Bloğu: İletişim (Attention) + Düşünme (FeedForward)"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # Katman Normalizasyonu 1
        self.ln2 = nn.LayerNorm(n_embd) # Katman Normalizasyonu 2

    def forward(self, x):
        # 'x = x + ...' yapısı Residual Connection (Artık Bağlantı) olarak adlandırılır.
        # Türevlerin kaybolmadan ilk katmanlara kadar akmasını sağlar!
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DorinaGPT(nn.Module):
    """Dorina Dil Modelinin Ana Sınıfı"""
    def __init__(self):
        super().__init__()
        # 1. Token ve Pozisyon Embedding Tabloları
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)

        # 2. Üst üste dizilmiş Transformer Blokları
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Son katman normalizasyonu

        # 3. Çıktı Başlığı (LM Head): 64 boyutlu vektörü sözlükteki 65 karakterin puanına çevirir
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token ve Pozisyon vektörlerini topla!
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)

        # Transformer bloklarından geçir
        x = self.blocks(x)
        x = self.ln_f(x)

        # Sonraki karakter tahmin puanlarını (logits) üret
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Cross-Entropy Loss hesabı için matrisleri düzleştiriyoruz
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Dorina'nın metin üretmesini (konuşmasını) sağlayan fonksiyon"""
        for _ in range(max_new_tokens):
            # Girdiyi maksimum context_length (64) olacak şekilde kırp
            idx_cond = idx[:, -context_length:]
            # Tahminleri al
            logits, _ = self(idx_cond)
            # Sadece en son adımın (son karakterin) tahminlerine odaklan
            logits = logits[:, -1, :] # (B, C)
            # Softmax ile olasılıklara çevir
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Olasılık dağılımından bir sonraki karakteri seç
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Seçilen karakteri mevcut metnin sonuna ekle ve devam et!
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# =====================================================================
# 4. MODELİ OLUŞTURMA VE EĞİTİM ÖNCESİ TEST (BEBEK DORİNA)
# =====================================================================
model = DorinaGPT().to(device)
toplam_parametre = sum(p.numel() for p in model.parameters())

print("=" * 65)
print(f"🚀 DORINA GPT v1 BAŞLATILDI! (Cihaz: {device.upper()})")
print(f"🧠 Toplam Eğitilebilir Parametre (Ağırlık) Sayısı: {toplam_parametre:,}")
print("=" * 65)

print("\n--- 👶 EĞİTİM ÖNCESİ DORİNA'NIN İLK SÖZLERİ (TAMAMEN RASTGELE) ---")
baslangic_tokeni = torch.zeros((1, 1), dtype=torch.long, device=device)
rastgele_cikti = decode(model.generate(baslangic_tokeni, max_new_tokens=200)[0].tolist())
print(rastgele_cikti)
print("-" * 65)

# =====================================================================
# 5. EĞİTİM DÖNGÜSÜ (ADAMW OPTIMIZER İLE)
# =====================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n🔥 EĞİTİM BAŞLIYOR (3000 Adım)...")
baslangic_zamani = time.time()

for iter in range(max_iters + 1):

    # Belirli aralıklarla Train ve Validation Loss değerlerini raporla
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        gecen_sure = time.time() - baslangic_zamani
        print(f"Adım {iter:4d}/{max_iters} | Train Loss: {losses['train']:.4f} | "
              f"Val Loss: {losses['val']:.4f} | Süre: {gecen_sure:.1f} sn")

    # 1. Veri paketini (batch) çek
    xb, yb = get_batch('train')

    # 2. İleri Yayılım (Forward Pass) ve Loss hesabı
    logits, loss = model(xb, yb)

    # 3. Önceki adımın türevlerini sıfırla
    optimizer.zero_grad(set_to_none=True)

    # 4. Geri Yayılım (Backpropagation - Türevleri hesapla!)
    loss.backward()

    # 5. AdamW ile ağırlıkları güncelle!
    optimizer.step()

# =====================================================================
# 6. EĞİTİM SONRASI TEST VE MODELİ KAYDETME
# =====================================================================
print("\n" + "=" * 65)
print("🎓 EĞİTİM TAMAMLANDI! İŞTE EĞİTİLMİŞ DORINA GPT v1 KONUŞUYOR:")
print("=" * 65 + "\n")

model.eval()
egitilmis_cikti = decode(model.generate(baslangic_tokeni, max_new_tokens=400)[0].tolist())
print(egitilmis_cikti)

# Eğitilen modeli diske kaydet
torch.save(model.state_dict(), "dorina_v1.pt")
print("\n" + "=" * 65)
print("💾 Model ağırlıkları 'dorina_v1.pt' dosyasına başarıyla kaydedildi!")
