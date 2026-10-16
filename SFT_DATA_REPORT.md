# 📋 Dorina-70M SFT (Instruction Tuning) Veri Hazırlama Raporu

**Tarih:** 16 Ekim 2026  
**Hedef Model:** Dorina-70M (10 Layer, 512 Embd, 8 Head, Context: 256)  
**Toplam Örnek Sayısı:** 12.000 Örnek (%100 Tamamlandı)  
**Toplam Token Sayısı:** 1.599.149 Token (~1.60M Token)  
**Ortalama Uzunluk:** ~133 token / örnek (256 context sınırına ideal uyum)  

---

## 1. Veri Dağılımı ve Kaynaklar

| Kaynak / Veri Seti | Hedef Örnek | Üretilen Örnek | Oran | Kazandırılan Yetenek & Görev |
| :--- | :---: | :---: | :---: | :--- |
| **`teknium/OpenHermes-2.5`** (+ Dolly-15k) | 6.000 | **6.000** | **%50** | **Genel Akıl Yürütme & CoT:** Çok yönlü günlük sorular, adım adım mantık ve genel bilgi. |
| **`HuggingFaceH4/ultrachat_200k`** | 3.000 | **3.000** | **%25** | **Sohbet & Asistan Refleksleri:** Doğrudan yanıta odaklanan, kibar ve yardımsever diyaloglar. |
| **`sahil2801/CodeAlpaca-20k`** | 1.800 | **1.800** | **%15** | **Kodlama & Sentaks:** Kısa Python fonksiyonları, algoritma mantığı, JSON ve yapılandırılmış çıktılar. |
| **`Open-Orca/1million-gpt-4`** | 1.200 | **1.200** | **%10** | **Talimat Takibi & Bilimsel QA:** "Madde madde yaz", "özetle", "yalnızca cevabı ver" kuralları ve mantık testleri. |
| **TOPLAM** | **12.000** | **12.000** | **%100** | **Tam Dengeli SFT Veri Seti** |

---

## 2. Filtreleme ve Format Standartları

1. **Context Sınırı:** $35 \le \text{tokens} \le 220$ kuralı ile 256 kontekst penceresine tam oturan kompakt örnekler seçildi.
2. **ChatML Standardı:**
   ```text
   <|im_start|>user
   {kullanıcı sorusu / talimatı}
   <|im_end|>
   <|im_start|>assistant
   {doğrudan, net asistan yanıtı}
   <|im_end|>
   ```
3. **Özel Token Ayrımı:** `tiktoken` (`gpt2`) üzerinde `encode_ordinary` metodu kullanılarak ChatML etiketleri kararlı ve hatasız şekilde tensörleştirildi.

---

## 3. Üretilen Veri Dosyaları (`data/`)

| Dosya Adı | Format | Boyut | Açıklama |
| :--- | :---: | :---: | :--- |
| `data/sft_train.jsonl` | JSON Lines | ~5.15 MB | 10.800 adet doğrulanmış eğitim örneği (şeffaf inceleme). |
| `data/sft_val.jsonl` | JSON Lines | ~0.56 MB | 1.200 adet (%10) değerlendirme örneği. |
| `data/sft_train.bin` | uint16 Binary | ~2.75 MB | **Yol A:** Standart Causal Pre-training tarzı sürekli dizi (`np.memmap`). |
| `data/sft_val.bin` | uint16 Binary | ~0.30 MB | **Yol A:** Doğrulama sürekli ikili dizisi. |
| `data/sft_train_masked.npz` | NumPy NPZ | ~3.24 MB | **Yol B:** 256 uzunluklu matris (`inputs` & `targets`). Prompt kısımları `-100` ile maskelenmiştir. |
| `data/sft_val_masked.npz` | NumPy NPZ | ~0.35 MB | **Yol B:** Maskelenmiş değerlendirme tensörleri. |

---

## 4. Eğitim Seçenekleri

* **Yol A (Klasik Causal SFT):** Mevcut `train.py` yapısıyla doğrudan çalışır.
* **Yol B (Masked SFT - Önerilen):** Prompt tokenları (`<|im_start|>user ... <|im_end|><|im_start|>assistant\n`) loss hesaplamasından dışlanır (`target = -100`). Model yalnızca asistan yanıtını üretmeyi öğrenir; ezberleme riski minimuma iner.
