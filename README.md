# Dorina AI

PyTorch kullanılarak sıfırdan (*from scratch*) geliştirilen, özgün Transformer tabanlı bir Büyük Dil Modeli (LLM) araştırma ve geliştirme projesidir. Hazır ağırlıklar ya da harici yüksek seviye kütüphaneler (HuggingFace Transformers vb.) kullanılmadan; tokenizasyon, çok kafalı dikkat (Multi-Head Self-Attention), artık bağlantılar (Residual Connections) ve katman normalizasyonu doğrudan tensör matematiğiyle inşa edilmiştir.

---

## 🎯 Proje Vizyonu ve Yol Haritası

Temel amaç, model mimarisini küçük ölçekten başlayarak kademeli olarak büyütmek ve 1 Milyar (1B) parametre hedefine ulaştırmaktır.

* **Aşama 1 (Tamamlandı):** 240K prototipten 70M ve 278M ölçeğine geçiş. Cosmopedia v2 üzerinde sentetik akıl yürütme ve dil kurgusu eğitimi tamamlandı.
* **Aşama 2 (Aktif - Gece Eğitimi):** FineWeb-Edu + Cosmopedia Data-Mix (%65 / %35) ile genel dünya bilgisi ve akademik dil yetenekleri kazandırma.
* **Aşama 3 (Planlanan):** Kod & Mantık (Python / Reasoning) continual pre-training.
* **Aşama 4 (Gelecek Vizyonu):** Özgün Rust tabanlı Byte-Level BPE Tokenizer, RoPE, SwiGLU mimarisi ve 1B parametreli Dorina 2.0.

---

## 📊 Model Sürüm ve Karşılaştırma Matrisi

| Sürüm | Model Dosyası | Tokenizer | Katman (`n_layer`) | Boyut (`n_embd`) | Kafa (`n_head`) | Bağlam (`ctx`) | Parametre | Durum / Başarı |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v0.10** | `dorina_v0.10_240k.pt` | Karakter (~295) | 4 | 64 | 4 | 64 | ~240K | Arşivlendi |
| **v0.11** | `dorina_v0.11_4m.pt` | GPT-2 BPE (50.257) | 4 | 128 | 4 | 128 | ~4.0M | Arşivlendi |
| **v0.12** | `dorina_v0.12_32m.pt` | GPT-2 BPE (50.257) | 8 | 256 | 8 | 256 | ~32.1M | Arşivlendi |
| **v0.13** | `dorina_v0.13_70m/model.pt` | GPT-2 BPE (50.257) | 10 | 512 | 8 | 512 | ~70.0M | Val Loss: 3.12 (Releases'ta) |
| **v0.13-SFT**| `dorina_v0.13_70m_sft/model.pt` | GPT-2 BPE (50.257) | 10 | 512 | 8 | 512 | ~70.0M | SFT Talimat Modeli |
| **v0.14 (P1)** | `dorina_v0.14_278m/model.pt` | GPT-2 BPE (50.257) | 16 | 1024 | 16 | 1024 | **278,293,504** | **Val Loss: 4.6830 (Donduruldu)** |
| **v0.14 (P2)** | `runs/dorina_v0.14_278m/run2/` | GPT-2 BPE (50.257) | 16 | 1024 | 16 | 1024 | **278,293,504** | **Aktif Gece Eğitimi (Data-Mix)** |

---

## 🏗️ Dizin Mimarisi

```text
dorina-ai/
├── data/                               # Ham ve binary (.bin) tokenize edilmiş veri setleri
│   ├── dorina300m_phase1_train.bin     # Cosmopedia v2 eğitim seti (429.2 MB)
│   ├── dorina278m_phase2_train.bin     # FineWeb-Edu eğitim seti (85.8 MB)
│   └── TinyStoriesV2-GPT4-train.txt    # Erken modeller için 2.1 GB masal seti
│
├── releases/                           # Başarıyla tamamlanmış dondurulmuş sürümler
│   ├── dorina_v0.13_70m/               # 70M Temel Model
│   ├── dorina_v0.13_70m_sft/           # 70M SFT Sohbet/Talimat Modeli
│   └── dorina_v0.14_278m/              # 278M Aşama 1 Modeli (Cosmopedia v2)
│
├── runs/                               # Aktif eğitim ve kontrol noktaları (checkpoints)
│   └── dorina_v0.14_278m/run2/         # Aşama 2 (FineWeb-Edu + Cosmopedia Data-Mix)
│
├── scripts/                            # Veri hazırlama, değerlendirme ve izleme betikleri
│   ├── prepare_278m_phase2.py          # FineWeb-Edu veri seti çekici ve BPE dönüştürücü
│   ├── prepare_300m_phase1.py          # Cosmopedia v2 veri seti hazırlayıcı
│   ├── train_custom_tokenizer.py       # Rust tabanlı Byte-Level BPE tokenizer eğiticisi
│   ├── generate.py                     # Kayıtlı modellerle metin üretme arayüzü
│   └── eval_suite.py                   # Otomatik değerlendirme test seti
│
├── src/                                # Çekirdek kütüphaneler
│   ├── model.py                        # Saf PyTorch tensör Transformer mimarisi
│   └── utils/                          # Logger ve Plotter araçları
│
├── web/                                # Gerçek zamanlı Cyberpunk telemetri Dashboard
├── train_278m_phase2.py                # Aşama 2 Data-Mix eğitim betiği
├── requirements.txt                    # Python bağımlılıkları
└── README.md                           # Proje dokümantasyonu
```

---

## ⚙️ Donanım & Çalışma Ortamı

* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM)
* **Hesaplama Mimarisi:** Ada Lovelace (Compute Capability 8.9)
* **Hızlandırma:** FP16 Mixed Precision (`torch.cuda.amp`), Gradient Accumulation, RAM-Cached Binary Data Loader
* **Kütüphaneler:** PyTorch (`torch`), OpenAI `tiktoken`, `tokenizers`, `matplotlib`, `numpy`

---

## 🚀 Hızlı Başlangıç

### 1. Ortamın Hazırlanması

```powershell
# Sanal ortamı aktive edin
.venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Metin Üretimi (Inference)

```powershell
# 278M Aşama 1 modelinden metin üretmek için:
python scripts/generate.py --model releases/dorina_v0.14_278m/model.pt --prompt "The universe is full of mysteries because"
```

### 3. Gerçek Zamanlı Telemetri Dashboard'u

```powershell
# Dashboard yerel sunucusunu başlatın:
python web/serve_dashboard.py
```
Tarayıcınızdan `http://localhost:8000` adresine giderek eğitim kaybını, perplexity ve GPU sıcaklık/kullanım grafiklerini canlı takip edebilirsiniz.