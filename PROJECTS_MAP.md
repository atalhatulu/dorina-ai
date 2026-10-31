# Dorina AI - Repository Architecture & Project Index 🏛️

Bu doküman, projelerimizin düzenli mimarisini, hangi kodun nerede olduğunu ve görev dağılımını gösterir.

---

## 📂 Aktif Projeler (`projects/`)

### 1. `projects/dorinandroagi/` (Mobil OS Ajanı 📱)
- **Açıklama:** Samsung Galaxy A70 ve Android cihazlar için donanım kontrolü (Parlaklık, ses, alarm, fener, uygulama açıcı, batarya).
- **İçerik:** `DorinaAssistant/` (Tam Android APK kaynak kodları), `dorina_bridge_server.py`, `eval_android_agent.py`.

### 2. `projects/dorina_v4/` (144.7M Dense Ana Temel Model 🧠)
- **Açıklama:** 3.0B token'ı tamamlamış kristal temel modelimiz. İleride 1:50 ve 1:100 için saklanan amiral gemisi omurgamız.
- **Ağırlıklar:** `models/dorina_v4/` altında (`dorina_v4_3b_stable_momentum.pt` ve kristal checkpointler).

### 3. `projects/dorina_nano_keyboard/` (Akıllı Klavye & İmla Motoru ⌨️)
- **Açıklama:** Gboard/SwiftKey benzeri kelime tamamlama, imla ve soru eki (`-mi`) ayıran hafif klavye motoru.
- **İçerik:** `dorina_v4_keyboard_app.py`, `web_keyboard_app.py`, `dorina_nano_tr_model.py`.

### 4. `projects/dorina_voice_tts/` (Sesli Asistan & TTS 🎙️)
- **Açıklama:** FastPitch & HiFi-GAN tabanlı yerel Türkçe ses sentezleme ve vokal motoru.
- **Ağırlıklar:** `models/dorina_hifigan_local_ckpt.pt`, `models/dorina_tts_real.pt`.

### 5. `projects/dorina_data_purifier/` (Web Kazıyıcı & Puanlayıcı 🧹✨)
- **Açıklama:** Yeni nesil veri fabrikamız! İnternetten ham HTML çeken, Nano modelle reklam/çerez pisliğini süpüren ve büyük modelle (1-5 puan) filtreleyen altın veri üretim hattı.

---

## 🗄️ Arşiv & Depo (`archive/`)
- **`archive/root_scripts/`**: Eski indirme, analiz ve yardımcı scriptler.
- **`archive/root_logs/`**: Eski Shard eğitim ve doğrulama logları.
- **`archive/root_tests/`**: Eski karşılaştırma ve ara test scriptleri.
- **`archive/legacy/`**: V1-V3 arası eski modellerin kodları.
