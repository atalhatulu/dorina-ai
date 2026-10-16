# Dorina AI

PyTorch kullanılarak sıfırdan (*from scratch*) geliştirilen, özgün Transformer tabanlı bir Büyük Dil Modeli (LLM) araştırma ve geliştirme projesidir. Hazır ağırlıklar ya da harici yüksek seviye kütüphaneler (HuggingFace Transformers vb.) kullanılmadan; tokenizasyon, çok kafalı dikkat (Multi-Head Self-Attention), artık bağlantılar (Residual Connections) ve katman normalizasyonu doğrudan tensör matematiğiyle inşa edilmiştir.

---

## 🎯 Proje Vizyonu ve Yol Haritası

Temel amaç, model mimarisini küçük ölçekten başlayarak kademeli olarak büyütmek ve 1 Milyar (1B) parametre hedefine ulaştırmaktır.

* **Aşama 1 (Tamamlanan / Devam Eden):** Pre-training — TinyStories veri seti üzerinde dilin sözdizimi, gramer yapısı ve temel kurgu bağlamını öğrenme.
* **Aşama 2:** Instruction Fine-Tuning — Önceden eğitilmiş modelin soru-cevap ve talimat takip eden bir asistan formatına dönüştürülmesi.
* **Aşama 3:** Ölçekleme & Hizalama (Alignment) — Bağlam penceresini ve katman sayısını artırarak sohbet yeteneklerini pekiştirme.

---

## 📊 Model Sürüm ve Karşılaştırma Matrisi

| Sürüm | Model Dosyası | Tokenizer | Katman (`n_layer`) | Boyut (`n_embd`) | Kafa (`n_head`) | Bağlam (`ctx`) | Parametre | Durum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **v0.1** | `dorina_v0.1_char_tiny_240k.pt` | Karakter (~295) | 4 | 64 | 4 | 64 | ~240K | Arşivlendi |
| **v0.11** | `dorina_v0.11_bpe_small_4m.pt` | GPT-2 BPE (50.257) | 4 | 128 | 4 | 128 | ~4.0M | Arşivlendi |
| **v0.12** | `dorina_v0.12_bpe_large_32m.pt` | GPT-2 BPE (50.257) | 8 | 256 | 8 | 256 | ~32.1M | Aktif Eğitim |

---

## 🏗️ Dizin Mimarisi

```text
dorina-ai/
├── checkpoints/                        # Model ağırlıkları ve mimari yapılandırmaları (.pt)
│   ├── dorina_v0.1_char_tiny_240k.pt
│   ├── dorina_v0.11_bpe_small_4m.pt
│   └── dorina_v0.12_bpe_large_32m.pt
│
├── data/                               # Eğitim ve doğrulama metinleri
│   └── TinyStoriesV2-GPT4-train.txt
│
├── logs/                               # Deney takibi ve metrikler (Adım, Loss, Süre)
│   ├── dorina_v0.1_char_tiny_240k.json
│   ├── dorina_v0.11_bpe_small_4m.json
│   ├── dorina_v0.12_bpe_large_32m.json
│   └── plots/                          # Kayıp (Loss) eğrisi görselleri (.png)
│
├── src/                                # Laboratuvar kodları ve eğitim boru hatları
│   ├── 01_data_download.py             # Veri seti indirme
│   ├── 02_data_inspection.py           # Veri yapısı ve karakter dağılımı analizi
│   ├── 03_tokenizer.py                 # Karakter seviyesi tokenizer prototipi
│   ├── 04_dataloader.py                # Batching ve tensör yükleme pipeline'ı
│   ├── 05_baseline_train.py            # v0.1 karakter tabanlı model eğitimi
│   ├── 06_chat.py                      # İnteraktif test ve üretim arayüzü
│   ├── 07_attention_mechanism.py       # Scaled Dot-Product Attention uygulaması
│   ├── 08_transformer_block.py         # Multi-Head Attention & FFN blok mimarisi
│   ├── 09_bpe_tokenizer.py             # tiktoken (GPT-2 BPE) entegrasyonu
│   ├── 10_tokenized_train.py           # v0.11 BPE tabanlı ara model eğitimi
│   ├── 11_large_model_train.py         # v0.12 32M parametreli model eğitimi
│   │
│   └── utils/                          # Deney ve analiz araçları
│       ├── logger.py                   # Anlık JSON metrik kayıt modülü
│       ├── plotter.py                  # Matplotlib kayıp eğrisi görselleştirici
│       └── parse_current_log.py        # Terminal çıktısı ayrıştırıcı
│
├── requirements.txt                    # Python bağımlılıkları
└── README.md                           # Proje dokümantasyonu

```

---

## ⚙️ Donanım & Çalışma Ortamı

* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM)
* **Hızlandırma:** CUDA Toolkit & cuDNN
* **Temel Kütüphaneler:** PyTorch (`torch`), OpenAI `tiktoken`, `matplotlib`, `numpy`

---

## 🚀 Hızlı Başlangıç

### 1. Ortamın Hazırlanması

```powershell
# Sanal ortamı aktive edin
.venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

```

### 2. Modellerin Eğitilmesi

```powershell
# 32M parametreli v0.12 modelini çalıştırmak için:
python src/11_large_model_train.py

```

### 3. Metrik ve Kayıp (Loss) Görselleştirme

```powershell
# 'logs/' klasöründeki tüm modellerin kayıp eğrilerini karşılaştıran grafiği üretir:
python src/utils/plotter.py

```

Çıktı görseli doğrudan `logs/plots/loss_comparison.png` konumuna kaydedilir.