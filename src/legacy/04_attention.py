import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# =====================================================================
# BOYUTLARIMIZ:
# B (Batch Size)     = 1 (Örnek olarak tek bir cümle inceleyelim)
# T (Time / Context) = 8 (Cümlede 8 token var: "Let's he")
# C (Channels / Emb) = 16 (Her token 16 sayıdan oluşan bir Embedding vektörü)
# =====================================================================
B, T, C = 1, 8, 16
x = torch.randn(B, T, C) # Rastgele 8 token'lık girdi vektörümüz

# Tek bir Attention Başlığı (Head) boyutu
head_size = 8

# Query, Key ve Value üreten doğrusal katmanlarımız (Bias yok, saf matris çarpımı)
key_layer   = nn.Linear(C, head_size, bias=False)
query_layer = nn.Linear(C, head_size, bias=False)
value_layer = nn.Linear(C, head_size, bias=False)

# 1. Her token kendi Q, K ve V vektörlerini üretir
k = key_layer(x)   # Boyut: (B, T, head_size) -> (1, 8, 8)
q = query_layer(x) # Boyut: (B, T, head_size) -> (1, 8, 8)
v = value_layer(x) # Boyut: (B, T, head_size) -> (1, 8, 8)

# 2. Query ve Key'leri çarpıp ham Dikkat Skorlarını (Attention Scores / wei) bulalım
# (1, 8, 8) @ (1, 8, 8)^T ---> (1, 8, 8) boyutunda bir tablo çıkar!
wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)

print("--- 1. HAM ATTENTION SKORLARI (Geleceği Görebiliyor - Hileli!) ---")
print(wei[0].detach().numpy().round(2))

# 3. CAUSAL MASK (Kopya Çekmeyi Önleyen Alt-Üçgen Maske)
# torch.tril -> Sadece alt üçgeni 1 olan, üst üçgeni 0 olan bir matris üretir
tril = torch.tril(torch.ones(T, T))

# Üst üçgendeki 0 olan yerleri -sonsuz (-inf) yapıyoruz!
wei = wei.masked_fill(tril == 0, float('-inf'))

print("\n--- 2. MASKELENMİŞ SKORLAR (Gelecek -inf ile Kapatıldı) ---")
print(wei[0].detach().numpy().round(2))

# 4. SOFTMAX: Skorları toplamı 1.0 (%100) olan olasılıklara çevir!
wei = F.softmax(wei, dim=-1)

print("\n--- 3. SOFTMAX SONRASI DİKKAT DAĞILIMI (Olasılıklar) ---")
print("Her satır bir token'dır. Sadece kendisi ve solundaki geçmiş token'lara dikkat (ağırlık) verebilir:")
print(wei[0].detach().numpy().round(2))

# 5. Son olarak bu olasılıklarla Value (V) vektörlerini çarpıp çıktıyı üretiyoruz
out = wei @ v
print(f"\nAttention Çıktı Boyutu: {out.shape} (Başarıyla bağlam bilgisi yüklendi!)")
