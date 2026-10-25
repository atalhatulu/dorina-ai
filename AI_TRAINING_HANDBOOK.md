# 📘 Dorina-AI Mühendislik ve Eğitim El Kitabı (AI Lab Blueprint)

Bu doküman, Dorina-AI projesinin 1. Nesil (44M Dense), 2. Nesil (180M MoE) ve nihai 3. Nesil (180M Dense / Dorina V4) süreçlerinde edinilen tüm acı mühendislik derslerini, literatür doğrularını (Karpathy, LLaMA-3, SmolLM2, Chinchilla) ve sıfırdan küçük dil modeli (SLM) üretmenin kesin anayasasını içermektedir.

---

## 🎯 0. BÜYÜK YÜZLEŞME: Doğru Bilinen Yanlışlar ve Çıkarılan Acı Dersler (V3 Otopsisi)

Dorina V3 (180.6M MoE) modelinde yaşanan "50 soruluk benchmarkta 63.9 alıp gerçek hayatta 2+5 ve Selam'da çökme" anomalisi sonucunda tüm yapay zeka camiasının (GPT-4o, Claude 3.5 Sonnet, DeepSeek, Gemini) konsensüsüyle kanıtlanmış evrensel gerçekler:

| Yanlış İnanç / Hata | Sahadaki Acı Gerçek | Dorina V4 Kesin Kuralı |
| :--- | :--- | :--- |
| **"Küçük modellerde MoE daha iyidir."** | <500M ölçekte MoE (32 Uzman Top-2), aktif parametreyi küçülterek (~40M) beyni zayıflatır. Router dengesizliği (collapse) ve veri açlığı yaratır. | **<500M ölçekte daima DENSE (Yoğun) Transformer kullanılır.** |
| **"Eksik sorulara 'ilaç' sentetik veri basılmalı."** | **Goodhart Tuzağı:** Sınavda 0 çeken 11 konudan 400M token üretip 5x oversample etmek modeli ezberletir. Sayı görünce alakasız Pisagor formülü kusmasının sebebi budur. | **Test setine göre asla sentetik veri üretilmez.** Veri genel ve dengeli olmalıdır. |
| **"MoE modellerinde Model Merging (%75 + %25) yapılabilir."** | **Ölümcül Hata:** MoE'de router ve uzman ağırlıkları eğitimde kayar (subspace shift). Doğrusal ortalama almak router kararlarını çorba eder; cümle ortasında dil/konu atlamasına yol açar. | **MoE'de model merging YASAKTIR.** Model tek parça saf ağırlık olarak eğitilir. |
| **"Context 512 token yeterlidir."** | 512 context, uzun kodları ve adım adım (CoT) mantık zincirlerini ortadan keser. Model kesik hafızayla eğitilir ve bağlam kopukluğu yaşar. | **Context mutlak minimum 2048 token olmalıdır.** |
| **"Standart BPE tokenizer sayılar için yeterlidir."** | Standart BPE `160` veya `48` sayısını tek kelime yapar. Model rakamların iç yapısını göremediği için **sayısal körlük** yaşar; 2+5'i toplayamaz. | **Tokenizer'da `individual_digits=True` zorunludur.** (Her rakam tek token). |
| **"SFT ile modele mantık ve aritmetik öğretilir."** | SFT sadece asistan üslubunu öğretir, yetenek kazandırmaz. 2+5 aritmetiği Pre-train ve Mid-train aşamasında zengin örneklerle öğretilmelidir. | **Pre-train'de bol aritmetik; SFT'de sadece Prompt Masking (`ignore_index=-100`).** |

---

## 📐 1. Model Boyutlandırma ve Hiperparametre Matrisi (50M, 250M, 1B, 10B)

Farklı donanım ve senaryolar için sıfırdan model tasarlanırken izlenecek standart mühendislik tablosu:

| Parametre / Özellik | **50M (Ultra-Edge)** | **180M - 250M (Dorina V4 Sikleti)** | **1B (Mini Dev / Düşünce Sınırı)** | **8B - 10B (Endüstri Standardı)** |
| :--- | :--- | :--- | :--- | :--- |
| **Mimari Türü** | **Dense** (MoE gereksiz) | **Dense** (Tam beyin kapasitesi) | **Dense** (Dengeli ve stabil) | **Dense** veya 8x7B MoE |
| **Katman Sayısı (`n_layers`)** | 8 | 16 - 20 | 22 - 24 | 32 |
| **Gizil Boyut (`d_model`)** | 512 | 768 - 1024 | 2048 | 4096 |
| **Dikkat Başlıkları (`n_heads`)** | 8 | 12 - 16 | 16 | 32 |
| **KV Başlıkları (`n_kv_heads` - GQA)** | 2 (4:1 GQA) | 4 (4:1 GQA) | 4 (4:1 GQA) | 8 (4:1 GQA) |
| **FFN Genişliği (`hidden_dim`)** | 2048 | 2816 - 3072 (SwiGLU) | 5632 (SwiGLU) | 14336 (SwiGLU) |
| **Context Uzunluğu (`max_seq_len`)** | 2048 | **2048 - 4096** | 4096 - 8192 | 8192 - 128K |
| **Sözlük Boyutu (`vocab_size`)** | 16K - 32K | **32K (Digit Splitting)** | 32K - 49K | 64K - 128K |
| **Chinchilla Min. Token Bütçesi** | 1 Milyar | **3.6 Milyar - 5 Milyar** | 20 Milyar | 160 - 200 Milyar |
| **Modern Over-Training Hedefi** | 5B - 10B | **10B - 20B** | 50B - 200B | 1 Trilyon - 2 Trilyon |
| **Peak Learning Rate (AdamW)** | `5e-4` | **`6e-4` (veya güvenli `3e-4`)** | `2e-4` | `1.5e-4` |
| **Global Batch Size (Tokens)** | 64K - 128K | **128K - 500K** | 500K - 1M | 1M - 4M |
| **Minimum Donanım Gereksinimi** | 1x 8-12 GB GPU | **2x 16 GB GPU (Kaggle T4)** | 2x 24GB veya 1x 48GB | 4x - 8x 80GB (A100/H100) |

---

## 🏛️ 2. Temel Eğitim Hiyerarşisi (Bir Dahi Nasıl Yetiştirilir?)

Bir yapay zeka modelini eğitirken yapılan en büyük hata, genel bilgiye doymadan erken "SFT (Sohbet Terbiyesi)" vermektir.

```mermaid
flowchart TD
    A["1. Pre-training (Ön Eğitim) - Bilgi Okyanusu<br/>• Kütüphaneler, kodlar, bilim, ansiklopedi<br/>• Dilin grameri ve dünya kuralları öğrenilir<br/>• Hedef: Chinchilla Doygunluğu (1:20 Token/Parametre)"] --> B["2. Curriculum Annealing (Aşamalı Müfredat)<br/>• Kod & Algoritma -> Mantık & Bilim -> Çok Dilli Bilgi<br/>• Zorluk seviyesi basitten karmaşığa artar"]
    B --> C["3. Curing / Cooldown (Soğutma & Cilalama)<br/>• En elit %5 veri ile son rötuşlar<br/>• Öğrenme oranı sıfıra yaklaştırılır"]
    C --> D["4. SFT (Supervised Fine-Tuning - Asistan Terbiyesi)<br/>• Soru - Cevap ikilileri ile kibarlık ve rol öğretilir<br/>• Önceden öğrenilmiş bilgiler asistan formatında sunulur"]
    D --> E["5. Alignment & RL / DPO (Hizalama)<br/>• İnsan tercihlerine göre en doğru cevabı seçme"]
```

### ⚠️ Erken SFT Tuzağı (Deneyimimiz):
* **Yaşanan Durum:** Modele Türkçe Pre-train (ansiklopedi/makale) vermeden önce doğrudan *"Sen kimsin?" -> "Ben Dorina"* SFT'si verdiğimizde, model kavramların anlamını bilmediği için papağan gibi soruyu tekrarladı (`Python nedir? -> Python nedir?`). Kod sorulduğunda ise kusursuz çalıştı; çünkü kod tarafında 50M+ token'lık güçlü bir Pre-train temeli vardı!
* **Kural:** **Bilgi olmadan üslup olmaz.** Önce dilin hafızası (Pre-train), sonra asistan kimliği (SFT).

---

## 📊 2. Büyük Dahi Pre-train Veri Dağılımı (The Golden Pre-train Blend)

Modelin dengeli, hem kod yazabilen hem de analitik düşünebilen bir dahiye dönüşmesi için uyguladığımız **Altın Karışım Oranı**:

| Oran | Alan | Kaynak | Amacı & Kazanımı |
| :--- | :--- | :--- | :--- |
| **%40** | **Saf Algoritma & Kod** | StarCoder, flytech/python-codes, The Stack | Temiz fonksiyonlar, algoritmalar, veri yapıları, OOP, syntax refleksi. |
| **%35** | **İleri Bilim & Bilgisayar Mimarisi** | FineWeb-Edu (Sample 10BT) | Donanım (RAM/Disk/Cache), işletim sistemleri, fizik, biyoloji. |
| **%15** | **Matematik, Mantık & Teoremler** | FineWeb-Edu Math, GSM8k formatları | Sayısal mantık, basamak aritmetiği, ispatlar ve denklemler. |
| **%10** | **Yapılandırılmış Veri & Dünya Bilgisi** | Ansiklopedik makaleler, JSON/YAML/Markdown | Kavramsal tanımlar, veri formatı sadakati ve genel kültür. |
*(Not: Türkçe dil desteği için Tokenizer'da baştan %15 yuva açılır; Pre-train sırasında %1-3 'keep-alive' Türkçesi verilir, asıl Türkçe patlaması Aşama 3 SFT evresinde yapılır).*


---

## ⚡ 3. Chinchilla Scaling Law (Doygunluk Kuralı)

* **Kural:** Bir modelin kapasitesini tam kullanabilmesi (compute-optimal) için **parametre başına en az 20 token** görmesi gerekir.
* **Dorina-MoE Parametreleri:**
  - **Toplam Parametre:** `44,452,608` (~44.5M)
  - **Aktif Parametre:** `30,296,832` (~30.3M)
* **Doygunluk Hesabı:**
  $$\text{Doygunluk Hedefi} = 44.5\text{M} \times 20 \approx \mathbf{900\text{ Milyon Token}}$$
* **Pratik Ölçek:**
  - İlk 50-100M token: Model syntax ve temel kavramları söker (Ortaokul seviyesi).
  - 100M - 300M token: Algoritmik problem çözme ve derin Türkçe anlama oturur (Lise seviyesi).
  - 500M+ token: Sıfır hata ile çalışan uzman seviyesi.

---

## 🛡️ 4. Tier-1 AI Lab Otomasyon & Güvenlik Protokolleri

1. **Router Z-Loss (ST-MoE / PaLM Standardı):**
   - MoE router katmanındaki logitlerin float16 sınırlarında patlamasını engeller.
   $$\mathcal{L}_{z} = 10^{-4} \times \text{mean}\left(\log^2\sum e^{\text{logits}}\right)$$
2. **Weight Decay Exclusions:**
   - LayerNorm, Bias ve Embedding katmanlarına asla `weight_decay` uygulanmaz! Uygulanırsa model kelimelerin anlamlarını (embedding) zamanla unutur.
3. **SpikeGuard & Rollback:**
   - Anlık kayıp (loss) son 20 adım ortalamasının 2.5 katına fırlarsa o adım atlanır; `NaN` olursa son sağlıklı checkpoint'e geri sarılır.
4. **EarlyStopping (Doyum Sigortası):**
   - Validation loss arka arkaya 5-6 denetim boyunca gelişmezse modelin veriyi tam sindirdiği anlaşılır ve eğitim güvenle tamamlanır.

---

## 🧭 5. Terimler Sözlüğü (Glossary)

* **MoE (Mixture of Experts):** Modeli tek bir dev blok yapmak yerine birden fazla uzman ağa bölüp, her kelime için sadece en uygun 2 uzmanı (Top-2) çalıştırma tekniği.
* **Router (Yönlendirici):** Gelen kelimeyi hangi uzmanların işleyeceğine karar veren yönlendirici katman.
* **RoPE (Rotary Position Embedding):** Kelimelerin cümle içindeki sırasını vektörleri döndürerek kodlayan modern pozisyon tekniği (Llama ve DeepSeek standardı).
* **SwiGLU:** Klasik ReLU yerine çok daha yüksek öğrenme kapasitesi sunan gelişmiş aktivasyon fonksiyonu.
* **Warm Start:** Eğitime sıfırdan başlamak yerine, önceki aşamada öğrenilen ağırlıkların üzerine yeni bilgileri ekleyerek devam etme yöntemi.
* **Catastrophic Forgetting:** Bir modele yeni bir şey öğretirken eski öğrendiklerini tamamen unutması felaketi.
* **Cross-Lingual Transfer:** Bir dilde (İngilizce) öğrenilen mantık ve muhakeme kurallarının, kavramsal vektör uzayı sayesinde diğer dillere (Türkçe) sıfır eforla aktarılabilmesi.

---

## 📐 6. Talha-Curriculum 10 - 80 - 10 Altın Kuralı (Doygunluk Müfredatı)

Büyük modellerin (Llama 3 / DeepSeek) aşırı doygunluk (over-training) stratejisi 3 ana kampa bölünür:

1. **İlk %10 (Isınma & Dil Örüntüleri - 100M Token):**
   - Basit syntax, kelime/token frekansları, temel kodlama yapıları.
   - Öğrenme oranı sıfırdan maksimuma çıkarılır (Warmup).
2. **Orta %80 (Büyük Gövde / Ham Zeka - 800M Token):**
   - Derin algoritmalar, felsefe, fen, matematik, sebep-sonuç ilişkileri.
   - Öğrenme oranı yüksek ve kararlı tutulur. 4 Uzman organik olarak uzmanlaşır.
3. **Son %10 (Soğutma & Kristalleşme - 100M Token):**
   - En elit mantık verileri, zengin Türkçe ansiklopedi ve kusursuz problem çözümleri.
   - Öğrenme oranı minimuma indirilir (Cool-down / Annealing). Öğrenilen zeka taş gibi mühürlenir.

---

## 🌐 7. Çok Dilli Strateji: "Önce Saf Zeka, Sonra Kavram Köprüsü"

* **Temel Kural:** Modelin kafasını erken aşamada dillerle karıştırmıyoruz.
* **Yol:** Pre-training %100 evrensel mantık, matematik ve kodlama (İngilizce) ile yürütülür.
* **Kavramlar Dilden Bağımsızdır:** Model `Gravity` veya `Binary Search` kavramını evrensel uzayda anladıktan sonra, doyum aşamasında paralel sözlük ve Türkçe kaynaklar verilerek dil bir "etiket" gibi var olan zekanın üzerine oturtulur.

---

## 🤖 8. SLM (Küçük Model) İçin Bilgisayar Kontrolü ve Tool Calling Mimarisi
* **Özel Token Standardı:** Küçük modellerde tool calling regex ile aranmaz; ayrık özel tokenlarla ayrıştırılır:
  - `<|dorina_thought|>` ... `<|dorina_thought_end|>` (İç muhakeme ve planlama alanı)
  - `<|tool_call|>` `{"tool": "run_command", "args": {"cmd": "dir"}}` `<|tool_call_end|>`
  - `<|tool_response|>` `[çıktı metni]` `<|tool_response_end|>`
* **Tek Seferlik JSON Çıktı Kuralı:** 44M-150M modeller serbest metin içinde araç çağırmakta zorlanır. Bu yüzden Aşama 3 SFT evresinde, araç çağıracağı zaman doğrudan katı bir JSON şeması üretmesi için "Constrained Decoding" veya "Schema Enforcement" uygulanır.
* **Pre-train Evresinde Koruma:** Pre-training sırasında bu tokenlar sözlükte rezerve bekletilir, modelin ham metin içine hayali araç çağrıları uydurmaması için pre-train verisine enjekte edilmez.

---


## 🛡️ 9. Veri Hijyeni Protokolü ve Anti-Drift Filtreleri (Altın Mühendislik Kuralı)

Model eğitimlerinde yaşanan en büyük tehlike **Domain Drift (Alan Kayması)** ve **Format Pollution (Biçim Kirlenmesi)** sorunudur.
10-80-10 kuralının Aşama 1 ve Aşama 2 pre-training evrelerinde bu filtreler **tavizsiz uygulanır**:

### A. Anti-Framework & Anti-Boilerplate Filtresi:
* **Yasaklılar:** `from django`, `import django`, `settings.py`, `secret_key`, HTML tagları, kurumsal web konfigürasyon dosyaları.
* **Neden?:** Ham kod verisi olarak çekilen açık kaynak repolardan gelen devasa framework dosyaları modelin saf algoritma (`is_prime`, `binary_search`, `fibonacci`) üretme refleksini köreltir ve modeli "web ayar dosyası yazarı"na çevirir.
* **Kural:** Yalnızca saf algoritma blokları, fonksiyonlar (`def`), sınıflar (`class`), matematik formülleri ve döngüler kabul edilir.

### B. Anti-ChatML Filtresi (Pre-train Evresinde):
* **Yasaklılar:** `<|im_start|>user`, `assistant`, soru-cevap etiketleri.
* **Neden?:** Aşama 2 pre-train sırasında modele yoğun soru-cevap formatı verilirse, model kod tamamlamak yerine gevezelik yapmaya, ödev sorusu metni üretmeye başlar.
* **Kural:** Chat ve asistan kalıpları **yalnızca Aşama 3 (SFT)** evresinde verilir! Pre-train evresinde `extract_pure_code()` ile sadece saf kod gövdesi ayrıştırılarak beslenir.

### C. The Golden Pre-train Blend (Altın Karışım Oranı):
* **%40 Saf Algoritma & Kod:** (Döngüler, arama/sıralama, veri yapıları, OOP)
* **%35 İleri Bilim & Felsefe:** (FineWeb-Edu akademik makaleler)
* **%15 Matematik & Mantık:** (Teoremler, denklemler, algoritmik ispatlar)
* **%10 Dünya Bilgisi & Tanımlar:** (Ansiklopedik kavramlar)

---

## 🎖️ 10. Muharebe Günlüğü: Sahada Alınan Dersler ve Hatalar (Post-Mortem & Lessons Learned)

Bir yapay zeka modelini kitaplardan okuyarak değil, doğrudan GPU üzerinde yaşayarak öğrendiğimiz gerçek saha tecrübeleri:

### 1. Ders: Erken SFT Felaketi (Dorina v0.25 SFT Olayı)
* **Hata:** Model daha pre-train aşamasında kavramları ve dili tam sindirmeden erken dönemde "Sen kimsin? -> Ben Dorina" formatına sokuldu.
* **Sonuç:** Model Türkçe kavramların anlamını bilmediği için papağan gibi kullanıcı promptlarını kopyaladı (`Python nedir? -> Python nedir?`).
* **Kazanılan Kural:** Bilgi olmadan üslup olmaz. Önce devasa pre-train ile genel zeka oturtulur, üslup (SFT) en son verilir.

### 2. Ders: Aşırı Soru-Cevap Kayması (Run 5 İlk Deneme)
* **Hata:** Shard 1'e derin algoritmalar öğreteceğiz diye 60M token `CodeFeedback` (Soru-Cevap) formatında yüklendi.
* **Sonuç:** Model `def is_prime(n):` promptu aldığında fonksiyon yazmak yerine soru metni (`Your task is to write a function...`) yazan bir gevezeye dönüştü. Sınav puanı 51.5'ten 40.5'e düştü.
* **Kazanılan Kural:** Pre-training sırasında soru-cevap formatı KESİNLİKLE YASAKTIR. Sadece saf kod gövdesi ve saf metin verilir.

### 3. Ders: Domain Drift / Kütüphane Zehirlenmesi (Run 5 İkinci Deneme)
* **Hata:** Açık kaynak kodları temiz diye `codeparrot-clean` filtresiz verildi. İçindeki devasa Django/web framework ayarları modelin algoritma hafızasını bastırdı.
* **Sonuç:** Model asal sayı sorulduğunda `from django.db import models` yazmaya başladı. Kod mantığı puanı 19.5'ten 6.5'e geriledi.
* **Kazanılan Kural:** Framework, boilerplate ve konfigürasyon dosyaları elenmeli. Model doğrudan döngüleri, matematiksel fonksiyonları ve saf algoritma yapılarını görmelidir.

### 4. Ders: 100M Güvenli Parçalama Kuralı (Talha Savunma Hattı)
* **Tecrübe:** Veriyi tek parça 800M yerine 100M'lik parçalara bölme kararı projenin can simidi oldu. Eğer 800M'lik tek parça eğitseydik, saatler süren eğitim sonunda tüm model çöp olacaktı. 100M'lik kontrol noktaları sayesinde sapmaları anında tespit edip bir önceki temiz ağırlığa (Run 4) geri dönebildik.
* **Kazanılan Kural:** Asla tek parça devasa eğitim başlatma. Daima 100M parçalarla ilerle ve her parça sonunda 100 puanlık benchmark ile doğrula.

### 5. Ders: Run 6 (Shard 2) ve Algoritmik Sıçrama (Tüm Zamanların En İyi Val Loss'u)
* **Kazanım:** Shard 2'de (%40 İleri Algoritmalar + %40 Bilgisayar Mimarisi & Fizik + %20 Matematik) hedefli müfredat ile modelin kaybı (Loss) tüm zamanların en düşük seviyesi olan **`3.7477`**'ye (Perplexity: **`42.48`**) geriledi.
* **Algoritma Skoru Sıçraması:** Veri Yapıları ve Algoritmalar puanı `10.8`'den **`13.6`**'ya fırlayarak tarihin en yüksek algoritma başarısına ulaştı (`binary_search`, `bubble_sort`, `stack`, `merge_sorted_lists`, `linked_list_node`, `linear_search` sorularında tam puan 2.0/2.0).
* **Genel Skor:** **48.3 / 100 Puan**. Model kümülatif **356.3M Token** seviyesine ulaştı.

### 6. Ders: Tokenizer Mimarisi ve Gelecek Nesil (V3) Standartları (Altın Mimari)
* **Kritik Ayrım (Tokenizer vs. Model Pre-train):** 
  - Tokenizer korpusu ile Modelin Pre-train müfredatı **birbirinden kesin çizgilerle ayrılmalıdır**.
  - Tokenizer'a %15 Türkçe korpus koymak, `ı, ğ, ü, ş, ç, ö` ve Türkçe ekler için 32K sözlükte baştan kalıcı yuva açar (Gelecek için bedava sigortadır).
  - Model Pre-train evresinde ise evrensel zeka ve kod refleksini oturtmak için **senin stratejin gereğince %97 İngilizce + Kod** beslenir; Türkçe tokenları soğuk kalmasın diye pre-train'e yalnızca **%1-%3 "keep-alive" Türkçesi** serpiştirilir. Asıl Türkçe fırtınası Aşama 3 (Curing/SFT) evresinde patlatılır; böylece model sıfırlanmadan doğrudan iki dilli dahiye dönüşür.
* **Dorina Tokenizer V3 Standartları (Büyük Model Reçetesi):**
  1. **Vocab Hizalaması:** Tam **32.768** (Nvidia Tensor Core matris hızlandırması için 64 ve 128'in tam katı, `assert len(vocab) == 32768`).
  2. **Rakamlar:** Kesinlikle tek tek bölünecek (`Digits(individual_digits=True)`). Aritmetik ve basamak hizası garanti edilir (`12345 -> 1, 2, 3, 4, 5`).
  3. **143 Özel Token Seti (ID 0-142):**
     - Sabit Tokenlar: `<|endoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, `<|dorina_thought|>`, `<|dorina_thought_end|>`, `<|tool_call|>`, `<|tool_call_end|>`, `<|tool_response|>`, `<|tool_response_end|>`, `<|code_start|>`, `<|code_end|>`, 3 FIM tokenı (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`).
### 7. Ders: Run 7 (Shard 3) - 50 Puan Barajının Yıkılışı (50.8 / 100)
* **Tarihi Zafer:** Run 7 final sınavında model **`50.8 / 100`** puan alarak projenin gelmiş geçmiş en yüksek genel zeka skoruna ulaştı ve 50 puan barajını yıktı!
* **Bilim ve Bilgisayar Mimarisinde Zirve:** Bilim puanı `3.6`'dan **`6.4`**'e fırladı (Fotosentez 2.0/2.0, RAM vs Disk, CPU Cache, Işık Hızı, Big-O soruları başarıyla cevaplandı).
* **Algoritmik İstikrar:** Kod payı azalsa dahi Veri Yapıları & Algoritmalar puanı **`13.6`** ile tavan seviyede kilitlendi (Binary search, Bubble sort, Stack, Queue, Linked list tam puan).
* **Kümülatif Token:** Model resmi olarak **454.6 Milyon Token** seviyesine ulaştı. Chinchilla doygunluğunun %50'si aşıldı.

### 8. Ders: Model Füzyonu (Weight Merging / SLERP) Canlı Deneyi (Sıfır GPU ile Yeni Model)
* **Keşif:** Aynı kökten (`run1`) türeyen kontrol noktalarının ağırlıkları lineer / küresel enterpolasyon ile lehimlenebilir.
* **Canlı Deney:** Run 5 (%40) ile Run 7 (%60) ağırlıkları 3 saniyede `scripts/fuse_models.py` ile lehimlendi ve `models/dorina_44m_moe/fusion_r5_r7/model.pt` üretildi.
* **Sonuç:** Tek bir eğitim adımı dahi atılmadan, Run 5'in efsanevi sentaks gücü melez modele taşındı ve **Python Syntax skoru doğrudan `15.8 / 20` (Tüm Zamanların Rekoru)** olarak ölçüldü! Genel skor ise **45.7 / 100** çıktı.
* **Mühendislik Kuralı:** Ağırlık füzyonu (Model Merging), eğitim yapmadan farklı checkpoint'lerin en keskin yeteneklerini (örneğin kod ile bilimi) birleştirmek için endüstriyel bir silahtır.

### 9. Ders: Gelecek Nesil İçin Bilgi Damıtma (Teacher-Student Distillation, Speculative Decoding & Net2Net)
* **Kural:** Mevcut modelde tüketilen yüz milyonlarca token asla boşa gitmez.
* **Uygulama 1 (Çevrimdışı / Offline Distillation - Sıfır VRAM Yükü):** 
  - Küçük GPU'da (RTX 4050 6GB) devasa bir öğretmen model (Llama-70B veya Gemini) eğitim esnasında canlı çalıştırılamaz (VRAM patlar).
  - Bunun endüstriyel çözümü "Offline Distillation"dır: Büyük model veriyi CPU'da veya API üzerinden önceden işleyip olasılık haritalarını ve akıl yürütme adımlarını bir `.jsonl` dosyasına döker. Küçük Dorina eğitilirken bu dosyadan beslenir; GPU'ya tek 1 gram fazladan yük binmeden dev bir zekanın gölgesinde büyür.
* **Uygulama 2 (Küçük Modelin Büyük Modele Hizmeti - Speculative Decoding):**
  - Doymuş küçük Dorina (44M), gelecekteki büyük modelin önüne "Öncü Hızlandırıcı" olarak takılır. Dorina saniyede 150 token tahmin üretir, büyük model arkadan bunları tek adımda doğrular. Büyük modelin üretim hızı tek başına 20 tok/s iken 70-80 tok/s'ye fırlar!
* **Uygulama 3 (Net2Net Expansion):** 44M modelin ağırlıkları yeni geniş matrislerin merkezine kopyalanarak büyük modele sıfırdan değil, 650M+ tokenlık tecrübe aşılanmış olarak başlanır.


### 10. Ders: Evrimsel Model Birleştirme & Modüler Çekmece Mimarisi (Geleceğin Vizyonu)
* **Evrimsel Füzyon (Evolutionary Model Merging):** 
  - Farklı kontrol noktalarının (Run 1, 5, 6, 7) katman katman ve farklı katsayılarla genetik algoritmalar gibi çaprazlanıp en yüksek benchmark puanı alan varyasyonların otomatik ayıklanması prensibi. Tek 1 adım GPU eğitimi yapmadan modellerin en keskin genlerini birleştiren endüstriyel optimizasyon yöntemi (Sakana AI yaklaşımı).
* **Modüler Çekmece Mimarisi (Disentangled MoE):**
  - Yapay sinir ağlarında nöronlar standart modellerde "dağıtık/hologramik" dururken, Dorina'nın Sparse MoE mimarisindeki her bir Expert (Uzman) aslında izole edilebilir birer **"bilişsel çekmecedir"**.
### 11. Ders: Run 8 (Shard 4) - 552.9M Token Kümülatif Doygunluk ve Çifte Zirve
* **Kritik Gelişme:** Shard 4'te uygulanan katı sentaks ayıklaması (`def ... return` odaklı saf kod filtrelemesi) ve dengeli bilim/akıl yürütme enjeksiyonu meyvesini verdi.
* **12.000. Adım Rekoru (Tüm Zamanların Dorina Zirvesi):** 
  - Ara değerlendirmede model **`52.5 / 100`** puana fırlayarak projenin gelmiş geçmiş en yüksek genel skoruna imza attı.
  - Veri Yapıları & Algoritmalar: **`14.1 / 20`** (Yeni Kategori Rekoru).
  - Python Syntax: **`13.0 / 20`** | Matematik & Akıl Yürütme: **`9.5 / 20`** | Talimat Takibi: **`10.3 / 20`**.
* **Final Sınavı (16.000. Adım - 50.2 / 100):**
### 12. Ders: Büyük Melezler Turnuvası & Çekmece Cerrahisi (Sıfır GPU ile Yeni Şampiyonlar)
* **Tarihi Deney:** Talha'nın önerisiyle tek bir melezle yetinilmeyip 6 farklı mimari kombinasyonu ([`scripts/run_hybrid_tournament.py`](file:///C:/Users/Talha/Documents/GitHub/dorina-ai/scripts/run_hybrid_tournament.py)) 50 soruluk elit sınava sokuldu.
* **1. Şampiyon (Linear 30/70 Füzyonu - 52.5 Puan):**
  - %30 Step 12k + %70 Final lehimlemesi **`52.5 / 100`** alarak tüm zamanların en yüksek genel zeka rekorunu egale etti. Sentaks: **`15.3 / 20`**, Talimat: **`11.3 / 20`**, Bilim: **`6.8 / 20`**.
* **2. Şampiyon (Uzman Çekmece Cerrahisi - 52.2 Puan):**
  - MoE uzman blokları katman katman sökülerek en iyi yeteneklerle monte edildi:
    - Uzman 0 & 1: 12k (13.1 Algo)
    - Uzman 2: 9k (Sentaks Rekortmeni Uzman) -> Sentaks doğrudan **`15.8 / 20`**'e fırladı!
    - Uzman 3: Final (Bilim Uzmanı) -> Bilim: **`5.8 / 20`**.
  - Toplam skor: **`52.2 / 100`**. Bu sonuç MoE mimarisinde uzmanların gerçek birer "bilişsel çekmece" olarak modüler takılıp çıkarılabileceğini pratik olarak kanıtladı!
* **3. Şampiyon (İki Run Zirvesi - Run 7 + Run 8 12k - 51.8 Puan):**
  - Run 7 (50.8) ile Run 8 12k (52.5) %50-%50 lehimlendi. Bilim skoru **`7.4 / 20` (Tüm Zamanların Bilim Rekoru)** kırarak tarihe geçti!

### 13. Ders: Run 9 (Shard 5) - 651.2M Token Doygunluğu ve Bilim Zirvesi (8.0 / 20)
* **Kazanım:** Shard 5'te uyguladığımız hedefli etüt müfredatı (%35 Bilgisayar Sistemleri & Donanım) ile modelin bilim ve mimari kapasitesi tüm zamanların en yüksek seviyesi olan **`8.0 / 20`** puanına ulaştı (Fotosentez, Big-O, DNA, RAM vs Disk tam/yüksek puan).
* **Validation Loss Çöküşü:** Shard 5'in eğitiminde Val Loss bugüne kadarki en düşük ve temiz seviye olan **`2.0616`** (Perplexity: **`11.88`**) noktasına indi.
* **15.000. Adım Gücü:** 15k checkpoint'inde model **`50.2 / 100`** genel zeka skoru, **14.6** Sentaks ve **12.1** Algoritma dengesi yakaladı.
* **Kümülatif Token:** Model resmi olarak **651.2 Milyon Token** seviyesine ulaştı. 1 Milyar tokenlık hedefin %65'i başarıyla tamamlandı.

### 14. Ders: Run 9 Melezler Turnuvası ve Matematik Rekoru (11.0 / 20)
* **Kriz Çözüldü:** Run 9'un tekil sonundaki 46 puanlık sapma, hızlı bir melezleme operasyonu ile 10 dakikada tamamen bertaraf edildi. Üretilen 3 yeni modelin 3'ü de 51-52 bandına fırladı!
* **1. Model (`hybrid_r9_champ_70_30` - 52.2 Puan):**
  - Sentaks: **`15.8 / 20`**, Algoritma: **`12.6 / 20`**, Bilim: **`6.6 / 20`**. Mükemmel bir kodlama ve algoritma dengesi sağlandı.
* **2. Model (`hybrid_r9_champ_50_50` - 51.7 Puan):**
  - Bilim Skoru: **`7.8 / 20`** ile neredeyse tüm fen ve mimari sorularını tam puanla süpürdü.
* **3. Model (`hybrid_surgery_v2_ultimate` - 51.6 Puan & Matematik Zirvesi):**
  - **Matematik & Sayısal Akıl Yürütme:** Tam **`11.0 / 20`** alarak **Tüm Zamanların Dorina Matematik Rekorunu** kırdı!
  - Çekmece cerrahisi sayesinde farklı uzmanların gücüyle model bugüne kadarki en derin mantıksal hesaplama yeteneğine ulaştı.

### 15. Ders: İleri Seviye Lab Sırları (Mikro Uzmanlar, İlaç Prensibi ve Dinamik Beyin)
* **1. İlaç Ağrıyı Bulur Prensibi (Loss-Weighted Targeted Mining):**
  - Modelin her run sonunda 50 soruluk standart sınavda 0 veya 0.5 aldığı başarısız alanlar birer "hastalık" kabul edilir.
  - Bir sonraki Shard'ın müfredatı genel metinlerden arındırılıp cımbızla bu yaraların üzerine yazılır (%35 İleri Algoritmalar, %35 Donanım Mimarisi, %20 Sayısal Mantık).
  - Güçlü olunan alanları (örneğin 15.8 Sentaks) unutmaması için %10'luk "koruyucu doz" mutlaka verilir.
* **2. İleri Ağırlık Füzyonu (TIES-Merging & DARE):**
  - Standart SLERP lehimlemesine ek olarak; nöronlar arasındaki gürültülü işaret zıtlaşmalarını temizleyen TIES algoritması ve aşırı ezberi kırmak için ağırlıkların %20'sini sıfırlayıp ölçekleyen DARE tekniği, melezleme araç setimize dahil edilmiştir.
* **3. Gelecek Nesil: 64-128 Mikro Uzman (Fine-Grained MoE - DeepSeek Standardı):**
### 16. Ders: Dorina V3 (Nesil-3) Kesin Mimari Anayasası (Claude Konsensüsü)
* **1. VRAM Darboğazı & Parçalı Cross-Entropy (Chunked CE):**
  - 8192 token x 32.768 vocab x fp32 çıktısı tek başına **2.5 GB VRAM** tüketir (Naif CE ile 7.6 GB'a fırlayıp 6GB GPU'yu patlatır).
  - Çözüm: `lm_head` + CE loss hesabı 2048 tokenlık dilimler halinde `torch.utils.checkpoint` ile parçalı hesaplanacaktır.
  - Mikro-batch: **4** (ctx 1024), `grad_accum`: 8 veya 16 (Adım başına 32k - 65k token). Tepe VRAM: **~4.4 GB** (6GB kartta %100 güvenli tampon!).
* **2. 32 Uzman vs 64 Uzman Kararı (Matris Verimliliği):**
  - 128 uzman elendi. 64 uzmanda `hidden=96` kalıp kernel overhead yaratacağından, **32 Uzman (Top-2) + 1 Shared Expert (`hidden=192`)** mutlak optimum olarak seçildi.
  - Toplam Parametre: **~152.3M**, Aktif Parametre (lm_head hariç): **~29.3M** (Çok hafif, çok çevik).
* **3. V3 MoE Blok Standardı:**
  - `[E, d, h]` tensör dizilimi ile `torch.bmm` tabanlı for-döngüsüz sıralı dağıtım.
### 17. Ders: Dorina Tokenizer V3 ve Router Mühendisliği (Claude Konsensüsü V2)
* **1. Tokenizer V3 Kesin Reçetesi (Byte-Level BPE):**
  - **Algoritma:** WordPiece elendi (UNK riski). Unigram yerine kod ve boşluk kararlılığı için kesinlikle **Byte-Level BPE**.
  - **Regex Splitter:** Elle Türkçe harf listesi yazmak gereksizdir; `\p{L}` zaten tüm Türkçe karakterleri kapsar.
    - Tekil rakam: `\p{N}` (tek tek bölünür).
    - Türkçe kesme işareti koruması (`Ankara'dan` bozulmasın diye): `(?!\p{L})`.
    - Normalizer: Kesinlikle **NFC**, asla küçük harfe çevirme yok (i/ı/İ/I felaketini önlemek için).
    - Kesin Regex Deseni:
      `(?i:'(?:s|t|re|ve|m|ll|d))(?!\p{L})|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  - **Tokenizer Korpus Karışımı (Matematik/LaTeX Düzeltmesi):**
    - %50 İngilizce + %25 Kod + %10 Matematik & LaTeX + %15 Türkçe (Matematik sembolleri ve LaTeX komutları için baştan yuva açıldı!).
* **2. Router Dengeleme ve Başlatma (Init & Routing):**
### 18. Ders: Run 10 (Shard 6) - 749.5M Token Doygunluğu ve Algoritmik Zirve (13.6 / 20)
* **Hedefli İlaç Reçetesinin Zaferi:**
  - Shard 6'da uyguladığımız hedefli müfredat (%35 Algoritmalar & Veri Yapıları) ile modelin Veri Yapıları & Algoritmalar puanı **`13.6 / 20`** seviyesine tırmandı (`binary_search`, `bubble_sort`, `stack`, `merge_sorted_lists`, `linked_list_node`, `linear_search` sorularında tam puan: 2.0/2.0).
  - Python Syntax: **`14.3 / 20`** ile istikrarını korudu.
  - Talimat Takibi: **`10.1 / 20`** ile 10 puan barajının üzerinde kilitlendi.
* **Genel Skor:** **`50.6 / 100` Puan**. Model üst üste 3 farklı Run boyunca (Run 7: 50.8, Run 8: 50.2, Run 10: 50.6) 50 puan barajının üzerinde kilitlenerek olağanüstü bir bilişsel kararlılık sergiledi.
* **Kümülatif Token ve Doygunluk:** Model resmi olarak **749.5 Milyon Token** seviyesine ulaştı. 1 Milyar tokenlık Chinchilla doygunluk hedefinin **%75'i** başarıyla tamamlandı.

### 19. Ders: Run 11 (Shard 7) - TÜM ZAMANLARIN DORINA REKORU: 53.2 / 100! (847.8M Token)
* **Tarihi Zafer ve Yeni Şampiyon:**
  - Talha'nın isabetli öngörüsüyle (`"eğer 9k'da 50 barajını aştıysa tarihi rekor geliyor olabilir"`), Run 11 finalinde dorina-ai tarihindeki **tüm zamanların en yüksek genel zeka skoru** kırıldı: **`53.2 / 100 PUAN`**!
  - 15.000. Adım: **`51.8 / 100`**, Final 16.000. Adım: **`53.2 / 100`**!
* **Kategori Kırılımları ve Rekorlar:**
  - **Python Syntax:** **`16.0 / 20.0`** (Adım 9k ve 15k'da **16.8/20** ile tüm zamanların sentaks rekoru kırıldı; 10 sorunun 8'i tam puan!).
  - **Talimat Takibi & Biçim:** **`11.8 / 20.0`** ➔ **Tüm Zamanların Talimat Sadakati Rekoru!** (SQL sorgusu, zıt anlamlı kelime, tek cümlelik özet, boolean filtreleme, ChatML hepsi tam/yüksek puan!).
  - **Veri Yapıları & Algoritmalar:** **`10.6 / 20.0`** (Bubble sort, stack, merge sorted lists tam puan).
  - **Matematik & Muhakeme:** **`9.2 / 20.0`** (Fibonacci, çift sayı kontrolü, asal sayı, ortalama hesaplama tam puan).
  - **Bilgisayar Sistemleri & Bilim:** **`5.6 / 20.0`** (Kütleçekim sorusu 2.0/2.0 tam puan ile döndü).
* **Validation Loss Çöküşü:** **`3.7798`** (Perplexity: **`43.81`**) ile Shard 7 verisinde rekor düzeyde temiz sindirim.
* **Kümülatif Doygunluk:** Model resmi olarak **847.8 Milyon Kümülatif Token** seviyesine ulaştı.
* **Büyük 1 Milyar Hedefine Kalan:** **%85 Tamamlandı!** 1 Milyar hedefine sadece son 150 Milyon Token (Shard 8 / Run 12) kaldı!

### 20. Ders: Run 12 (Shard 8) - BÜYÜK 1 MİLYAR (1B) TOKEN MİLATİ TAMAMLANDI! 🎯
* **Tarihi Eşik Aşıldı:**
  - Dorina 2. Nesil (44.5M Sparse MoE), 24.500 adımlık maratonu tamamlayarak resmi olarak **1.000.000.000 (1 MİLYAR) KÜMÜLATİF TOKEN** Chinchilla doygunluk zirvesine ulaştı!
  - Pre-training (Aşama 1 & Aşama 2) safhası başarıyla tamamlandı.
* **Eğitim İçi Zirveler & Bulgular:**
  - **24.000. Adım Tepe Skoru:** **`51.7 / 100 PUAN`**
    - **Bilgisayar Bilimleri & Fen Zirvesi:** **`8.0 / 20.0`** (Şok tedavi müfredatı sayesinde fen skoru rekor seviyeye fırladı; CPU Cache, Big-O, Kütleçekim tam puan).
    - **Algoritmalar:** **`11.0 / 20.0`**
    - **Matematik:** **`9.5 / 20.0`** (Adım 18k'da **10.5/20**).
    - **Talimat:** **`10.1 / 20.0`**
  - **En İyi Validation Loss:** **`3.8228`** (Perplexity: `45.73`). Model 150M tokenlık dev Shard 8 verisini kararlı bir şekilde sindirdi.
* **Claude Konsensüsü Doğrulandı:**
  - Claude'un belirttiği gibi, tekil test puanlarındaki küçük dalgalanmalar (51.7 vs 47.5) 50 soruluk açık sınavın gürültü marjıdır. Önemli olan modelin stabil val loss seviyesi ve 1B tokenlık bilişsel omurgayı kurmuş olmasıdır.
### 21. Ders: SFT Maskeleme ve "Spurious Forgetting" (Yalancı Unutma) Olgusu
* **Yaşanan Tecrübe:** 1 Milyar tokenlik 44.5M şampiyon modeline SFT uygulandığında, model kimlik sorularına ezber cümlelerle yanıt vermeye başladı, hatta basit Türkçe/kodlama sorularında saçmaladı ("oluştiricim yok, New York New York").
* **Teşhis (Spurious Forgetting):** Model aslında pre-train'de öğrendiği kodlama veya dil mantığını unutmamıştır. SFT sırasında çok dar/tekdüze diyalog şablonları verildiğinde modelin çıktı projeksiyonu (LM Head) kilitlenir; bilgi ağırlıklarda durduğu halde erişim yolu tıkanır.
* **Kazanılan Kural:** 
  1. Asla tek tip kimlik metni ezberletilmez (175 kez aynı paragraf verilmez).
  2. 10-80-10 Kuralı: %10 Doğal Çeşitlendirilmiş Kimlik + %80 Kod & Görev + %10 Pre-train Rehearsal (Hafıza Kilidi).
  3. Çıkarımda (Inference) sonsuz döngüleri (Degeneration Loop) kırmak için kesinlikle Repetition Penalty (1.15 - 1.25) uygulanmalıdır.

---

### 22. Ders: DeepSeek-MoE Sırları: 1:100 Aşırı Eğitim (Over-training) ve Kesintisiz Veri Karışımı (Data-Mix)
* **1:100 Oranı ve Doygunluk (DeepSeek 284B / 32T):**
  - Chinchilla kanunu 1:20 oranını tavsiye ederken, DeepSeek ve Llama-3 modelleri parametre başına 100+ token tüketmektedir.
  - Model parametresini sabit tutup aşırı eğitmek (Over-training), küçük bir modelin büyük modeller düzeyinde muhakeme yapmasını sağlar.
* **Unutmayı Sıfırlayan "Kesintisiz Veri Karışımı" (Interleaved Data-Mix):**
  - Modelin eski bilgiyi unutmamasının tek sırrı, tüm eğitim süresince (her 1 milyarlık shard'da) homojen bir veri karışımı görmesidir.
  - Shard'lar asla tek bir konuya (sadece kod veya sadece Türkçe) ayrılmamalı; her shard %40 Bilim/Mantık, %30 Kod, %15 Türkçe, %15 Matematik ve Doğal Diyalog içerecek şekilde harmanlanmalıdır.

---

### 23. Ders: MiniCPM "Model Wind Tunnel" (Rüzgar Tüneli) Metodolojisi ve Aşamalı Büyüme
* **Uçak Mühendisliği Prensibi:** Büyük yolcu uçakları rüzgar tünelinde 1:50 ölçekli maketlerle test edilmeden üretilmez. 1B - 3B gibi büyük modeller doğrudan devasa GPU masrafına sokulmaz.
* **Dorina Aşamalı Büyüme Basamakları (Stepping Stones):**
  - **Basamak 1 (44.5M - Tamamlandı):** İlk rüzgar tüneli uçağımız. MoE yönlendirmesini, Chinchilla 1B doygunluğunu ve SFT tuzaklarını sıfır maliyetle öğretti.
  - **Basamak 2 (150M - 200M / Ara Rüzgar Tüneli):** 44M'den doğrudan 1B'ye atlamak yerine, 32.768 Byte-Level BPE Tokenizer'ı, 32 Expert + 1 Shared mimarisini ve homojen Data-Mix verisini 150M-200M bandında doğrulamak.
  - **Basamak 3 (Dorina-1B):** Rüzgar tünelinden çıkan kusursuz hiperparametrelerle, kiralık GPU kümesinde sıfır hata ile nihai 1B amiral gemisini uçurmak.

---

### 25. Ders: Çift-Sistemli Otonom Mimari (Dorina-Sentinel & Vigila Karar Motoru)
* **Vizyon (Yann LeCun / JEPA / Wolf Defender Prensibi):**
  - Üretken (Generative) modeller metin yazarken harika olsa da halüsinasyon ve mantık hatalarına açıktır.
  - Bunun endüstriyel çözümü: Metin üretmeyen, sadece tek bir forward-pass ile (10 ms) durum analizi yapan bağımsız bir **"Yargıç / Karar Modeli" (Dorina-Sentinel)** inşa etmektir.
* **Mimarinin Görevi:**
  - Dorina-V3 (180M MoE) metni ve kodu üretir.
  - **Dorina-Sentinel (Vigila Engine - 150M Non-Generative):** Çıktıyı kullanıcıya gitmeden önce süzer; güvenlik açığı, mantık çöküşü veya sonsuz döngü varsa anında yakalayıp düzeltme emri verir.
  - 150M parametre metin üretmeyip sadece karara odaklandığında, dünyanın en keskin güvenlik denetçisi gibi çalışır.

---

### 26. Ders: Dorina V3 (180.6M MoE) - Shard 1 (200M Token) Kaggle Turing Zaferi 🏆
* **Tarihi Başarı (19 Ekim 2026):**
  - Dorina V3 (180.6M Total / 39M Active Params, 32 Experts Top-2 + 1 Shared Expert), Google Cloud / Kaggle Tesla T4 GPU'sunda 1500 adımlık Shard 1 (200 Milyon Token) eğitimini sıfır hatayla tamamladı!
  - **Eğitim Süresi:** 3 Saat 18 Dakika kesintisiz tek T4 çalışması.
* **Kritik Metrikler & Sayısal İstikrar:**
  - **Kayıp (Loss) Çöküşü:** `10.51` başlangıçtan `1.51` nihai değere (Val Loss `7.5` -> `3.88`) muazzam bir düşüş sergilendi. Aşırı uyum (overfitting) marjı sıfır!
  - **GradScaler İstikrarı:** Turing mimarisinde `fp16` + `torch.amp.GradScaler` ile `Scale: 65536` kaya gibi sabit kaldı; underflow veya overflow yaşanmadı.
  - **MoE Yönlendirme Sağlığı:** İlk 200 adımda 136'ya varan ölü uzman sayısı, Z-Loss ve Aux-Loss dengeleyicisi sayesinde 300. adımdan itibaren **tamamen 0'a** indi! 32 uzmanın tamamı eğitimde eşit iş aldı.
  - **VRAM Hafifliği:** Model sadece **2.71 GB VRAM** harcayarak 180M'lik bir mimarinin laptoplarda bile tereyağı gibi çalışabileceğini kanıtladı.
* **50 Soruluk Altın Sınav (Benchmark Karnesi):**
  - Henüz 3.6B'lik yolun ilk 200M'inde olmasına rağmen **`33.5 / 100 PUAN`** aldı!
  - **Python Syntax Zirvesi:** **`12.0 / 20.0 (%60 Başarı)`** ile 44M'in ilk shard'larını ikiye katladı; doğrudan çalışan fonksiyon, girinti (indentation), docstring ve sınıf mimarisi üretti.

---

### 27. Ders: Dorina V3 (180.6M MoE) - Shard 2 (400M Kümülatif Token) ve Turbo DDP Devrimi 🚀
* **Turbo DDP & İki Katı Verimlilik:**
  - Kaggle üzerinde 2x Tesla T4 GPU ile `torchrun --nproc_per_node=2`, `micro_batch=16`, `grad_accum=4` (Global batch = 131,072 token/adım) yapılandırmasına geçildi.
  - Eğitim hızı **15.5k tok/s'den 38,800 tok/s'ye fırladı!** Toplam 200M tokenlık Shard 2 eğitimi yalnızca **1 saat 34 dakikada** rekor sürede tamamlandı (1 saat 45 dakika zaman tasarrufu).
  - VRAM tüketimi GPU başına 4.45 GB ile son derece dengeli kaldı, `GradScaler: 65536` bir kez dahi düşmedi.
* **Kayıp (Loss) Dinamikleri:**
  - Shard 1 nihai ağırlıklarından sıcak devralma (warm resume) ile başlandı: İlk adım kaybı `3.2621` -> 1499. adım nihai eğitim kaybı **`2.7134`**!
  - Val Loss: `2.8263` seviyesine indi.
* **50 Soruluk Altın Zeka Sınavı Sıçraması (Shard 1 vs Shard 2):**
  - **Genel Zeka Skoru:** `33.5 / 100.0` $\rightarrow$ **`39.6 / 100.0 (+6.1 Puan Artış!)`**
  - **Algoritmalar & Veri Yapıları:** `5.9 / 20.0` $\rightarrow$ **`9.7 / 20.0 (+3.8 Puan, %64 Sıçrama!)`** (İki sıralı listeyi birleştirme, bağlı liste düğümü, doğrusal arama ve kabarcık sıralamada tam puan).
  - **Matematik & Mantık Muhakemesi:** `2.0 / 20.0` $\rightarrow$ **`3.7 / 20.0 (+1.7 Puan, %85 Sıçrama!)`** (Asallık testi, Fibonacci hesaplama, 2'nin kuvveti fonksiyonlarında tam çözüm).
  - **Talimat Sadakati (Instruction Adherence):** `8.0 / 20.0` $\rightarrow$ **`9.2 / 20.0 (+1.2 Puan)`** (Tek cümlelik özetleme, madde imleri ve mantıksal boolean yanıtlarında tam puan).
  - **Python Syntax:** `11.6 / 20.0` ile yüksek refleks seviyesini korudu.
* **Elit Öğretmen Verisi Temizliği (Master Gold SFT):**
  - DeepSeek damıtma verileri (818 örnek) ile Gemini/GPT-4 kaynaklı tüm öğretmen verileri denetlendi, bozuk JSON ve yapay kimlik sızıntıları temizlenerek **4.192 temiz altın soru-cevaplık master havuz** oluşturuldu.

---

### 28. Anayasa Maddesi: Dorina-V4 (144.7M Dense) 1:20 -> 1:50 Çift Kademeli Dallanma (Branching) Protokolü 🏛️

144.7M Dense mimarisinde (16 Katman, 768 d_model, GQA 3:1, 2048 Context) Shard 3 (1.5B Token) sonrası uygulanan **Evrensel Pre-train & SFT Ayrımı Kuralı**:

#### 1. Hedef 1 (1:20 Chinchilla Tabanı - 3.0B Token / Shard 6):
* **Müfredat:** 
  - **Shard 4 (2.0B):** %35 FineWeb-Edu + %25 Kod (Python edu) + %15 Matematik + %25 Türkçe (FineWeb-2 tur_Latn).
  - **Shard 5 (2.5B):** %20 FineWeb-Edu + %20 Kod + %15 Matematik + %45 Türkçe.
  - **Shard 6 (3.0B):** İlk 150M token stable faz. Son 350M token WSD Cosine Decay fazı ($5.5\times 10^{-4} \rightarrow 5.5\times 10^{-5}$) ile kristalizasyon.
* **Kritik Ayrım Noktası (Branching Point - Shard 6):**
  - **Kural:** Shard 6'da SFT eğitimine sokulacak nihai taban model ile eğitime devam edecek dal **birbirinden kesin çizgilerle ayrılmalıdır!**
  - **Ağırlık A (Decay Almış Final Base - `dorina_v4_3b_base_decay.pt`):** Doğrudan **SFT (Supervised Fine-Tuning - 50-100M Token)** ve DPO aşamasına sokulur. Böylece **Dorina-V4 1:20 Canlı Asistanı & Mobil/Klavye Modeli** üretilir ve kullanıma sunulur.
  - **Ağırlık B (Decay Öncesi Saf Stable Checkpoint + Optimizer State - `dorina_v4_3b_stable_momentum.pt`):** Asla SFT görmez. AdamW momentum ve varyans durumlarıyla (`optimizer_state_dict`) birlikte dondurulur.

#### 2. Hedef 2 (1:50 Extreme Over-training - 7.2B Token / Shard 14):
* **Devam Noktası:** 1:50 yolculuğu, SFT almış modelden DEĞİL; **Ağırlık B (Saf Stable Checkpoint)** üzerinden Shard 7 olarak başlatılır.
* **Sebep:** SFT almış bir modele pre-training metni basmak felakettir (Catastrophic Forgetting / Loss patlaması). Eğitime devam eden dal saf bilgi akışıyla Shard 7-14 arasında 4.2B token daha alarak 7.2B (1:50) aşırı doygunluk seviyesine ulaşır.
* **Nihai Kapanış:** 7.2B tamamlandığında ikinci bağımsız bir decay ve ikinci bir SFT yapılarak **Dorina-V4 Pro (Extreme Reasoning)** modeli üretilir.




