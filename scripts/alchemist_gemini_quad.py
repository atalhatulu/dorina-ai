import time
import pyautogui
import pyperclip

# Fail-safe ekran köşesine gidince hata vermesini kapatıyoruz
pyautogui.FAILSAFE = False

# Koordinatlar (1920x1080 ekran için 4 dikey pencere)
# Ekran 4'e bölündüğünde orta noktalar: ~240, ~720, ~1200, ~1680
WINDOWS = [
    {"name": "Pencere 1 (Python Algoritmaları)", "x": 240, "y": 900},
    {"name": "Pencere 2 (Matematik & Mantık)", "x": 720, "y": 900},
    {"name": "Pencere 3 (Türkçe Derin Muhakeme)", "x": 1200, "y": 900},
    {"name": "Pencere 4 (Sistem Mimarisi & SQL)", "x": 1680, "y": 900},
]

PROMPTS = [
    # Pencere 1
    "Bana Dorina V3 model eğitimi için Python Algoritmaları (ağaçlar, graflar, dinamik programlama, asenkron yapılar) üzerine tam 10 adet yüksek kaliteli soru-cevap çifti üret. Yanıtı SADECE ve SADECE JSON formatında şu yapıda ver:\n[{\"instruction\": \"...\", \"response\": \"...\"}]\nHiçbir giriş veya çıkış metni yazma, sadece saf JSON listesi ver.",
    
    # Pencere 2
    "Bana Dorina V3 model eğitimi için Matematiksel Muhakeme ve Mantık problemleri (adım adım ispat, kombinatorik, olasılık, denklem çözümü) üzerine tam 10 adet yüksek kaliteli soru-cevap çifti üret. Yanıtı SADECE ve SADECE JSON formatında şu yapıda ver:\n[{\"instruction\": \"...\", \"response\": \"...\"}]\nHiçbir giriş veya çıkış metni yazma, sadece saf JSON listesi ver.",
    
    # Pencere 3
    "Bana Dorina V3 model eğitimi için İleri Düzey Türkçe Dil ve Felsefi Muhakeme (kavram analizleri, mantık çıkarımları, atasözü felsefesi) üzerine tam 10 adet yüksek kaliteli soru-cevap çifti üret. Yanıtı SADECE ve SADECE JSON formatında şu yapıda ver:\n[{\"instruction\": \"...\", \"response\": \"...\"}]\nHiçbir giriş veya çıkış metni yazma, sadece saf JSON listesi ver.",
    
    # Pencere 4
    "Bana Dorina V3 model eğitimi için Sistem Mimarisi, Linux, Docker ve SQL Sorgu Optimizasyonu üzerine tam 10 adet yüksek kaliteli soru-cevap çifti üret. Yanıtı SADECE ve SADECE JSON formatında şu yapıda ver:\n[{\"instruction\": \"...\", \"response\": \"...\"}]\nHiçbir giriş veya çıkış metni yazma, sadece saf JSON listesi ver.",
]

print("=" * 60)
print("Dorina-Alchemist: 4-Pencere Otomasyonu Başlıyor!")
print("=" * 60)
print("Lütfen fareye dokunma, 3 saniye içinde başlıyoruz...")
time.sleep(3)

for idx, (win, prompt) in enumerate(zip(WINDOWS, PROMPTS)):
    print(f"\n[{idx+1}/4] {win['name']} penceresine tıklanıyor (X={win['x']}, Y={win['y']})...")
    
    # Metin kutusuna tıkla
    pyautogui.click(win['x'], win['y'])
    time.sleep(0.5)
    
    # Panoya promptu kopyala ve Ctrl+V yap (Türkçe karakterleri hatasız aktarmak için)
    pyperclip.copy(prompt)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.5)
    
    # Gönder (Enter)
    pyautogui.press('enter')
    print(f"  -> Prompt gönderildi!")
    time.sleep(1.0)

print("\n" + "=" * 60)
print("4 pencereye de promptlar başarıyla dağıtıldı! Modeller üretiyor...")
print("=" * 60)
