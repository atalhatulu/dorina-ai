import numpy as np

# Rastgeleliği sabitleyelim ki her çalıştırdığımızda aynı başlangıç değerlerini görelim
np.random.seed(42)

# =====================================================================
# 1. VERİ SETİ (DATASET)
# Gizli Kuralımız: y = (2 * x1) + (3 * x2) - 1
# Ama Dorina bu formülü BİLMİYOR! Sadece aşağıdaki örnekleri görecek.
# =====================================================================
X = np.array([
    [1.0, 2.0],  # Örnek 1: (2*1) + (3*2) - 1 = 7.0
    [2.0, 1.0],  # Örnek 2: (2*2) + (3*1) - 1 = 6.0
    [3.0, 3.0],  # Örnek 3: (2*3) + (3*3) - 1 = 14.0
    [0.0, 4.0],  # Örnek 4: (2*0) + (3*4) - 1 = 11.0
])

# Gerçek hedefler (Target Outputs)
Y_gercek = np.array([[7.0], [6.0], [14.0], [11.0]])

# =====================================================================
# 2. MODEL PARAMETRELERİ (WEIGHTS & BIAS)
# 2 girdimiz olduğu için 2 adet Ağırlık (W) ve 1 adet Sapma (b) oluşturuyoruz.
# Başlangıçta tamamen rastgele sayılar!
# =====================================================================
W = np.random.randn(2, 1)  # 2x1 boyutunda rastgele ağırlık matrisi
b = np.zeros((1, 1))       # Başlangıçta 0 olan bias

print("--- EĞİTİM ÖNCESİ RASTGELE PARAMETRELER ---")
print(f"Başlangıç Ağırlıkları (W): {W.flatten()} (Hedefimiz: [2.0, 3.0])")
print(f"Başlangıç Bias (b)       : {b.flatten()[0]:.4f} (Hedefimiz: -1.0)\n")

# =====================================================================
# 3. HİPERPARAMETRELER (HYPERPARAMETERS)
# =====================================================================
learning_rate = 0.05  # Öğrenme oranı (Adım büyüklüğü)
epochs = 500          # Veri setini kaç kez baştan sona döneceğiz?
n_samples = X.shape[0]

# =====================================================================
# 4. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# =====================================================================
for epoch in range(epochs):

    # ADIM A: İleri Yayılım (Forward Pass)
    # Formül: Y_tahmin = X . W + b (Matris çarpımı)
    Y_tahmin = np.dot(X, W) + b

    # ADIM B: Kayıp / Hata Hesabı (Mean Squared Error Loss)
    # Formül: Loss = Ortalama( (Y_tahmin - Y_gercek)^2 )
    hata_farki = Y_tahmin - Y_gercek
    loss = np.mean(hata_farki ** 2)

    # ADIM C: Geri Yayılım (Backpropagation - Türev / Gradient Hesabı)
    # Loss fonksiyonunun W ve b'ye göre kısmi türevini (Zincir Kuralı) alıyoruz:
    # dLoss / dY_tahmin = 2 * (Y_tahmin - Y_gercek) / n_samples
    dLoss_dTahmin = (2.0 / n_samples) * hata_farki

    # dLoss / dW = X^T . dLoss_dTahmin
    dW = np.dot(X.T, dLoss_dTahmin)

    # dLoss / db = Toplam(dLoss_dTahmin)
    db = np.sum(dLoss_dTahmin, axis=0, keepdims=True)

    # ADIM D: Ağırlık Güncellemesi (Optimizer / Gradient Descent)
    # Türevin TERSİ yönünde küçük bir adım atıyoruz!
    W = W - (learning_rate * dW)
    b = b - (learning_rate * db)

    # Her 50 adımda bir durumu ekrana yazdıralım
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss (Hata): {loss:.6f} | W: [{W[0][0]:.3f}, {W[1][0]:.3f}] | b: {b[0][0]:.3f}")

print("\n--- EĞİTİM TAMAMLANDI! ---")
print(f"Öğrenilen Ağırlıklar (W): [{W[0][0]:.4f}, {W[1][0]:.4f}] (Gerçek Kural: [2.0, 3.0])")
print(f"Öğrenilen Bias (b)      : {b[0][0]:.4f} (Gerçek Kural: -1.0)")

# =====================================================================
# 5. TEST / ÇIKARIM (INFERENCE)
# Şimdi nöronumuza eğitimde HİÇ GÖRMEDİĞİ yeni bir veri verelim: [5.0, 2.0]
# Kurala göre cevap: (2 * 5) + (3 * 2) - 1 = 15.0 olmalı!
# =====================================================================
yeni_veri = np.array([[5.0, 2.0]])
tahmin = np.dot(yeni_veri, W) + b
print(f"\nTest Girdisi [5, 2] için Dorina Hücresinin Tahmini: {tahmin[0][0]:.4f} (Gerçek Cevap: 15.0)")
