# DorinAndroAGI (Dorina-Android-AGI) 🤖📱

**DorinAndroAGI**, Dorina-V4 (144.7M Dense) tabanlı, Android işletim sistemini yerel (on-device) seviyede kontrol edebilen, sesli (STT / TTS) ve fonksiyon çağrılı (Tool Calling) özel mobil otonom ajan projesidir.

---

## 🎯 Proje Hedefleri
1. **Sıfır Bulut Bağımlılığı (Tamamen Offline):** Model doğrudan cihaz üzerinde veya yerel köprüyle çalışır; internet gerektirmeden telefon donanımını kontrol eder.
2. **Derin Android Entegrasyonu:**
   - 💡 **Ekran Parlaklığı:** `Settings.System` üzerinden %0 - %100 doğrudan kontrol (Otomatik parlaklık geçersiz kılma).
   - ⏰ **Alarm & Zamanlayıcı:** `AlarmClock.ACTION_SET_ALARM` ile doğrudan alarm kurma.
   - 🔊 **Ses Kanalları:** `AudioManager` ile medya/zil sesi yönetimi.
   - 🔦 **El Feneri:** `CameraManager` ile anlık fener açma/kapama.
   - 🔋 **Batarya Durumu:** `BatteryManager` ile anlık pil seviyesi ve şarj durumu okuma.
   - 🚀 **Uygulama Başlatıcı:** YouTube, TikTok, WhatsApp, Kamera, Galeri vb. doğrudan intent tetikleme.
3. **Doğal Dil & Yazım Toleransı:**
   - Türkçe karakter esnekliği (`i`/`ı`, `c`/`ç`, `s`/`ş`, `g`/`ğ`, `u`/`ü`, `o`/`ö`).
   - Casing (büyük/küçük harf) ve yazım hatası toleransı (`TurkishNormalizer`).
4. **Çift Yönlü Ses Desteği:**
   - **STT (Speech-to-Text):** Türkçe sesli komut alma.
   - **TTS (Text-to-Speech):** Eylemleri Türkçe sesli yanıtla bildirme.

---

## 📂 Proje Bileşenleri
- **`DorinaAssistant/`**: Tam derlenebilir Android Studio / Gradle projesi (`com.dorina.assistant`).
  - `MainActivity.java`: UI, ses tanıma (STT), sesli okuma (TTS) ve komut işleme.
  - `AndroidDeviceActions.java`: Yerel Android API eylemleri.
  - `TurkishNormalizer.java`: Türkçe yazım ve fonetik dengeleyici.
- **`dorina_v4_android_agent.pt`**: Dorina-V4 Android Ajan SFT ağırlıkları (32,832 vocab).
- **`convert_dorina_to_gguf.py`**: Modeli mobilde saf C++ hızıyla koşturmak için GGUF dönüştürücü.
- **`dorina_bridge_server.py`**: Cihazla bilgisayar arasında canlı model köprüsü.
- **`dorina_android_agent_dataset.json`**: 5,924 örnekli Android OS Agent altın SFT veri seti.

---

## 📌 Değerlendirme & Sonraki Adımlar
- Genel Türkçe kimlik/sohbet (Identity SFT) ağırlıkları eklenerek saf komut modunun yanı sıra sohbet yeteneği genişletilecek.
- GGUF modeli doğrudan Android NDK / JNI içine gömülerek tek parça bağımsız APK haline getirilecek.
