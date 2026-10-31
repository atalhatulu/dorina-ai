# Dorina AI 🏛️

**Dorina AI**, sıfırdan (*from scratch*) saf PyTorch tensör mimarisiyle inşa edilmiş bağımsız bir Yapay Zeka ekosistemi ve Büyük Dil Modeli (LLM / SLM) araştırma-geliştirme projesidir.

Harici veya soyutlayıcı yüksek seviye kütüphaneler (HuggingFace Transformers vb.) kullanılmadan; Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU aktivasyonu, RMSNorm, ChatML şablonlama ve BPE tokenizasyon mekanizmaları doğrudan sıfırdan yazılmıştır.

---

## 🏛️ Proje Mimarisi ve Ekosistem

Dorina AI bünyesinde 5 ana odak alanı bulunmaktadır:

1. **Dorina-V4 (`projects/dorina_v4/` & `models/dorina_v4/`):** 144.7M Dense Transformer amiral gemisi temel ve SFT dil modelleri (3.0B token müfredatı, Dual T4 Kaggle DDP).
2. **Dorinandro AGI (`projects/dorinandroagi/`):** Samsung Galaxy A70 ve Android cihazlar için saf yerel Cihaz İçi İşletim Sistemi Ajanı (GGUF f16 / PyTorch, HTTP/Bridge, tam Android APK kaynak kodu).
3. **Dorina Nano Keyboard (`projects/dorina_nano_keyboard/`):** Akıllı klavye tamamlama, imla ve soru eki (`-mi`) ayıran hafif uç cihaz dil motoru.
4. **Dorina Voice & TTS (`projects/dorina_voice_tts/`):** FastPitch ve HiFi-GAN tabanlı uçtan uca Türkçe vokal sentezleme motoru.
5. **Dorina Data Purifier (`projects/dorina_data_purifier/`):** Web kazıma, Nano temizleyici ve LLM kalite filtreli altın veri üretim fabrikası.

---

## 🧠 Model Ailesi ve Envanteri

Dorina projesi kapsamında eğitilmiş, optimize edilmiş ve kullanımda olan model envanteri:

### 1. Dorina-V4 Amiral Gemisi (144.7M Dense SLM)
> **Mimari:** 16 Layers, 768 Hidden Dim, 12 Q Heads, 4 KV Heads (3:1 GQA), 2560 SwiGLU MLP, RoPE, RMSNorm, 32.768 / 32.832 Vocab, 2048 Context.

| Model / Checkpoint | Tür / Aşama | Token / Boyut | Açıklama & Yetenek |
| :--- | :--- | :--- | :--- |
| `dorina_v4_shard1_final.pt` | Pre-training Shard 1 | 0.5B Tokens (578 MB) | FineWeb-Edu temel İngilizce gramer ve kelime dizilimi. |
| `dorina_v4_shard2_final.pt` | Pre-training Shard 2 | 1.0B Tokens (578 MB) | Olgusal tutarlılık ve cümle yapılarının oturması. |
| `dorina_v4_shard3_final.pt` | Pre-training Shard 3 | 1.5B Tokens (578 MB) | Temel biyoloji, kimya ve genel dünya bilgisi. |
| `dorina_v4_shard4_final.pt` | Pre-training Shard 4 | 2.0B Tokens (578 MB) | %35 FineWeb-Edu, %25 Python Edu, %25 TR-HQ, %15 FineMath. Kod ve Türkçe sıçraması. |
| `dorina_v4_shard4_decayed.pt`| WSD Decay Shard 4 | 2.0B Tokens (578 MB) | Erken kristalleşme ve LR soğutma kontrol noktası. |
| `dorina_v4_shard5_final.pt` | Pre-training Shard 5 | 2.5B Tokens (578 MB) | Yüksek kaliteli Türkçe ve mantıksal akıl yürütme sentezi. |
| `dorina_v4_shard6_1to20_crystal_final.pt` | Pre-training Shard 6 | 3.0B Tokens (578 MB) | **3.0 Milyar token kristal taban.** Final Loss: ~1.46. 1:20 Chinchilla optimal omurga. |
| `dorina_v4_3b_stable_momentum.pt` | Pre-training Full State | 3.0B Tokens (1.73 GB) | Model ağırlıkları + AdamW momentum & optimizer durumları (continual pre-train için). |
| `dorina_v4_gold_sft_final.pt` | SFT Chat / Talimat | 32.832 Vocab (579 MB) | ChatML formatında Türkçe/İngilizce talimat ve çok turlu diyalog modeli. |
| `dorina_v4_crystal_sft_best.pt` | SFT Ara Değerlendirme | 578 MB | Kristal taban üzerine en düşük doğrulama kaybı veren SFT kontrol noktası. |
| `dorina_v4_android_agent.pt` | Mobil OS Ajanı (PyTorch) | 579 MB | Android sistem eylemlerini (ses, parlaklık, wifi, batarya, fener) çağıran JSON tool-call modeli. |
| `dorina_v4_clean_agent.pt` | Mobil OS Ajanı (Sade) | 579 MB | Temizlenmiş ve optimize edilmiş Android asistan modeli. |
| `dorina_v4_android_agent_f16.gguf` | Cihaz İçi Mobil Motor | 340 MB | llama.cpp / Termux / Android cihazlarda NPU/CPU üzerinde doğrudan koşan GGUF formatı. |

---

### 2. Dorina Voice & Ses Modelleri
| Model / Ağırlık | Mimari | Boyut | Açıklama |
| :--- | :--- | :--- | :--- |
| `dorina_tts_real.pt` / `prototype.pt` | FastPitch Mel Akustik Model | ~25.5 MB | Fonem ve metin dizgilerini mel-spektrograma dönüştüren vokal motoru. |
| `dorina_hifigan_local_ckpt.pt` | HiFi-GAN Vocoder | 660 MB | Mel-spektrogramı doğrudan 22.05kHz doğal dalga formuna (.wav) çeviren nöral ses üretici. |

---

### 3. Dorina Nano & Hafif Uç Modeller
| Model | Mimari / Parametre | Tokenizer | Görev |
| :--- | :--- | :--- | :--- |
| **Dorina Nano TR** | Hafif Dense (~10M-30M) | BPE 8K / 12K TR | Mobil klavye tahmini, Türkçe morfoloji ve ek denetimi. |

---

### 4. Tarihsel Arşiv & Prototipler
| Sürüm / Model | Parametre | Detay |
| :--- | :--- | :--- |
| `dorina_44m_moe` (`releases/`) | 44.5M Sparse MoE | Erken dönem MoE araştırma modeli (4 Uzman Top-1, Switch/GShard). |
| `v0.10 - v0.14` | 240K - 278M | Projenin ilk aşama araştırma prototipleri. |

---

## 💻 Canlı Etkileşim ve Çalıştırma

### 1. Dorina-V4 ile Sohbet Etme

Sohbet arayüzünü çalıştırmak için:

```powershell
python chat_dorina.py
```

### 2. Android OS Ajan Sunucusunu Başlatma

Mobil asistan ile WiFi köprüsünü kurmak için:

```powershell
python projects/dorinandroagi/dorina_bridge_server.py
```

### 3. Akıllı Klavye Testi

```powershell
python projects/dorina_nano_keyboard/dorina_v4_keyboard_app.py
```

---

## 📚 Dokümantasyon

- [AI Training Handbook](AI_TRAINING_HANDBOOK.md): Büyük ölçekli model eğitimi, kristalleşme ve loss dinamikleri kılavuzu.
- [AI Research Handbook](docs/AI_RESEARCH_HANDBOOK.md): Araştırma ve mimari tasarım notları.
- [Projects Architecture Map](PROJECTS_MAP.md): Ekosistem dizin ve modül kılavuzu.
- [Dorina-V4 Runs Log](models/dorina_v4/TRAINING_RUNS.md): Shard 1'den Shard 6'ya detaylı eğitim seyir defteri.
