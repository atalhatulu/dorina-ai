# AI Engineering Research & NotebookLM Handbook 📚🔬

Bu el kitabı, küçük dil modelleri (SLM), kodlama uzmanı yapay zekalar (Code-LLMs), sentetik veri mühendisliği ve veri-verimli eğitim stratejileri üzerine en kritik teknik makaleleri, ArXiv linklerini ve NotebookLM çalışma yönergelerini içerir.

---

## 🎯 1. NotebookLM İçin İndirilecek "Altın" 6 Makale

### A. Küçük Dil Modelleri & Sentetik Veri Devrimi
1. **SmolLM: A family of state-of-the-art small language models (HuggingFace, 2024)**
   - **Kapsam:** 135M, 360M ve 1.7B modellerin veri dağılımı (%40 Cosmo sentetik ders kitabı, %35 StarCoder kodu, %25 FineWeb-Edu), LR ve hiperparametreleri.
   - **ArXiv / Blog:** [HuggingFace SmolLM Blog](https://huggingface.co/blog/smollm) | [Paper ArXiv:2502.02737](https://arxiv.org/abs/2502.02737)
2. **Textbooks Are All You Need (Microsoft Phi-1 & Phi-1.5, 2023)**
   - **Kapsam:** Küçük modelleri devleştiren "Ders Kitabı Kalitesinde Sentetik Veri" devrimi. Temiz kod ve sentetik anlatımlarla 1.3B modelin devleri geride bırakması.
   - **ArXiv:** [ArXiv:2306.11644](https://arxiv.org/abs/2306.11644)
3. **LIMA: Less Is More for Alignment (Meta AI, 2023)**
   - **Kapsam:** Zekanın pre-training'de kazanıldığını, SFT'nin sadece 1.000 (bin) adet kristal veriyle bile GPT-4 seviyesine ulaşabileceğini kanıtlayan mihenk taşı.
   - **ArXiv:** [ArXiv:2305.11206](https://arxiv.org/abs/2305.11206)

### B. Kod Modelleri & Repo-Level Mimariler
4. **DeepSeek-Coder: When the Large Language Model Meets Programming (DeepSeek, 2024)**
   - **Kapsam:** %87 kod, %10 kod dokümantasyonu, %3 matematik veri karması. FIM (Fill-In-the-Middle) mimarisi ve Repo-Level bağlam ilişkileri.
   - **ArXiv:** [ArXiv:2401.14196](https://arxiv.org/abs/2401.14196)
5. **StarCoder: may the source be with you! (BigCode / HuggingFace & ServiceNow, 2023)**
   - **Kapsam:** 80+ programlama dilinde açık kaynak kod filtreleme, lisans temizliği, kod tokenizer optimizasyonu ve çoklu dil kod tamamlama.
   - **ArXiv:** [ArXiv:2305.06161](https://arxiv.org/abs/2305.06161)

### C. Büyük Ölçekli Eğitim ve Aşırı Eğitim (Over-training)
6. **The Llama 3 Herd of Models (Meta AI, 2024)**
   - **Kapsam:** 15 Trilyon token'lık eğitim aşamaları, veri kalite filtreleri, WSD / Cosine Decay soğuma (annealing) evresi ve modern Transformer optimizasyonları.
   - **ArXiv:** [ArXiv:2407.21783](https://arxiv.org/abs/2407.21783)

---

## 🔍 2. Nasıl Takip Edilir ve Araştırılır?
- **HuggingFace Papers:** [huggingface.co/papers](https://huggingface.co/papers) (Topluluk oylamalı günlük en iyi AI yayınları)
- **ArXiv Computer Science (Computation and Language):** [arxiv.org/list/cs.CL/recent](https://arxiv.org/list/cs.CL/recent)
- **NotebookLM Prompt Önerisi:** *"Bu kaynaklardaki 100M-500M arası küçük modellerin veri dağılım oranlarını, sentetik veri üretim yöntemlerini ve kodlama yeteneği kazandırma adımlarını karşılaştırmalı bir tablo olarak hazırla."*

---

## ⚡ 3. Bir "Coder" Model İçin Minimum Parametre Sınırları

| Parametre Boyutu | Yapabildikleri | Yapamadıkları / Sınırları | Gerçek Dünya Örnekleri |
| :--- | :--- | :--- | :--- |
| **~10M - 30M** | Tek satır sözdizimi tamamlama, değişken/fonksiyon adı tahmin etme. | Çok satırlı mantık kuramaz, algoritma tasarlayamaz. | Tabnine Legacy, klasik autocomplete |
| **~60M - 80M** | Küçük fonksiyonlar yazma (toplama, döngü kurma, liste filtreleme), basit syntax hatası düzeltme. | Karmaşık mimariler, çok dosyalı bağlamlar, derin mantık kurma. | Küçük Transformer prototipleri |
| **~120M - 150M** *(Bizim Alanımız)* | **Altın Eşik (Sweet Spot):** Tam fonksiyon yazma, algoritma mantığı (`if/else/while`), FIM ile kod aralarını doldurma, birim test (Unit Test) üretme. | 500+ satırlık devasa yazılım mimarisi tasarlama. | **Dorina-V4 (144M)**, SmolLM-135M, StarCoder-160M |
| **~500M - 1B** | Çok dosyalı proje anlayışı, gelişmiş hata ayıklama (debugging), API tasarımı, karmaşık algoritmalar. | Çok yüksek bellek gerektirir, mobil cihazlarda pil tüketir. | DeepSeek-Coder-1.3B, Qwen2.5-Coder-0.5B/1.5B |
