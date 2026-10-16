import torch

# =====================================================================
# 1. VERİYİ OKUMA VE SÖZLÜK (VOCABULARY) OLUŞTURMA
# =====================================================================
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Veri seti yüklendi! Toplam karakter sayısı: {len(text):,}")

# Hızlı eğitim için önce karakter/sembol bazlı kelime dağarcığını çıkaralım
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Benzersiz token (vocab_size) sayısı: {vocab_size}")
print(f"Sözlükteki karakterler: {''.join(chars)}\n")

# String-to-Index (stoi) ve Index-to-String (itos) tabloları
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]          # Metni sayı dizisine çevirir
decode = lambda l: ''.join([itos[i] for i in l]) # Sayı dizisini metne çevirir

# Tüm veri setini PyTorch Tensor'una (64-bit tam sayı dizisine) çevirelim
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Veri Tensor Boyutu: {data.shape}, Veri Tipi: {data.dtype}")

# =====================================================================
# 2. EĞİTİM (TRAIN) VE DOĞRULAMA (VALIDATION) AYRIMI (%90 - %10)
# Dorina'nın ezber yapıp yapmadığını (Overfitting) anlamak için şart!
# =====================================================================
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# =====================================================================
# 3. BATCH (VERİ PAKETİ) OLUŞTURUCU
# =====================================================================
context_length = 8  # Dorina tek seferde maksimum 8 token geriye baksın (şimdilik)
batch_size = 4      # Aynı anda (paralel) 4 farklı metin parçası işlesin

torch.manual_seed(1337)

def get_batch(split):
    """Rastgele metin parçaları seçip Girdi (x) ve Hedef (y) matrisleri üretir."""
    secilen_veri = train_data if split == "train" else val_data
    # Veri setinin içinde rastgele 4 başlangıç noktası seç
    ix = torch.randint(len(secilen_veri) - context_length, (batch_size,))

    # Girdiler (x) ve 1 adım sağa kaydırılmış hedefler (y)
    x = torch.stack([secilen_veri[i : i + context_length] for i in ix])
    y = torch.stack([secilen_veri[i + 1 : i + context_length + 1] for i in ix])
    return x, y

xb, yb = get_batch("train")

print("\n--- GİRDİ MATRİSİ (xb) [Boyut: 4x8] ---")
print(xb)
print("\n--- HEDEF MATRİSİ (yb) [Boyut: 4x8 - 1 Sağa Kaydırılmış!] ---")
print(yb)

print("\n--- İLK PAKETTE DORİNA'NIN ÖĞRENECEĞİ ÖRNEKLER ---")
for t in range(context_length):
    baglam = xb[0, :t+1].tolist()
    hedef = yb[0, t].item()
    print(f"Girdi: {baglam} ({repr(decode(baglam))}) ---> Tahmin Edilecek Hedef: {hedef} ({repr(decode([hedef]))})")
