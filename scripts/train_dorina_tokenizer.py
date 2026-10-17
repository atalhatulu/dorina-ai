import os
import sys
import time
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "tokenizer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 70)
print("🦀 DORINA-32K BYTE-LEVEL BPE TOKENIZER EĞİTİMİ (RUST ENGINE)")
print("=" * 70)

# 1. Özel Token'lar (Dorina ekosisteminin kalıcı anahtarları)
SPECIAL_TOKENS = [
    "<|endoftext|>",      # Standart metin sonu (EOS)
    "<|pad|>",            # Dolgu (Padding)
    "<|im_start|>",       # ChatML Sohbet Başlangıcı
    "<|im_end|>",         # ChatML Sohbet Bitişi
    "<|dorina_thought|>", # Düşünme / Akıl Yürütme (Reasoning Modu)
    "<|code_start|>",     # Kod Bloğu Başlangıcı
    "<|code_end|>",       # Kod Bloğu Bitişi
    "<|user|>",           # Kullanıcı Mesajı
    "<|assistant|>",      # Asistan Cevabı
    "<|system|>"          # Sistem Yönergesi
]

VOCAB_SIZE = 32768  # 32k Altın Standart (2^15)

print(f"📦 Hedef Vocab Boyutu: {VOCAB_SIZE:,}")
print(f"🔑 Özel Token Sayısı:  {len(SPECIAL_TOKENS)}")
for t in SPECIAL_TOKENS:
    print(f"   • {t}")

# 2. Tokenizer Modelini Tanımlama
tokenizer = Tokenizer(BPE(unk_token=None))  # Byte-Level olduğu için UNK token gerekmez!
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=ByteLevel.alphabet(),
    show_progress=True,
    min_frequency=2
)

# 3. Dengeli Eğitim Verisi Akışı (Python Kod + İngilizce Bilim / Mantık)
print("\n🌐 Tokenizer eğitimi için veri akışı başlatılıyor...")
print("   • Kaynak 1: Hugging Face CodeFeedback & Python Instruction (Kod & Mantık)")
print("   • Kaynak 2: Yerel Bilim ve Genel Kültür Metinleri (Wikitext / Metinler)")

def text_generator():
    total_samples = 0
    
    # 1. Kod Veri Setleri
    for ds_name in ["iamtarun/python_code_instructions_18k_alpaca", "sahil2801/CodeAlpaca-20k"]:
        try:
            ds = load_dataset(ds_name, split="train", streaming=True)
            for row in ds:
                text = row.get("prompt") or row.get("output") or row.get("instruction") or ""
                if len(text.strip()) > 30:
                    yield text
                    total_samples += 1
                    if total_samples >= 25000:
                        break
        except Exception as e:
            print(f"⚠️ {ds_name} atlandı: {e}")
            
    # 2. Yerel Genel Kültür & Bilim Metinleri
    wiki_path = os.path.join(ROOT_DIR, "data", "wikitext-2.txt")
    if os.path.exists(wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(line.strip()) > 40:
                    yield line.strip()
                    total_samples += 1
                    if total_samples >= 45000:
                        break
                        
    print(f"\n   ✓ Toplam {total_samples:,} örnek metin Rust BPE motoruna aktarıldı.")

# 4. Tokenizer'ı Eğit (Rust Motoru)
t0 = time.time()
print("\n⚙️ Rust tabanlı BPE sözlük eğitimi başladı (Multi-threaded)...")
tokenizer.train_from_iterator(text_generator(), trainer=trainer)
train_time = time.time() - t0
print(f"✓ Tokenizer eğitimi başarıyla bitti! (Süre: {train_time:.2f} saniye)")

# 5. Kaydetme
save_path = os.path.join(OUTPUT_DIR, "dorina_tokenizer_32k.json")
tokenizer.save(save_path)
print(f"💾 Tokenizer kaydedildi ➔ {save_path}")

# 6. Hızlı Doğrulama ve Test
print("\n" + "=" * 70)
print("🧪 TOKENIZER DOĞRULAMA TESTLERİ:")
print("=" * 70)

test_sentences = [
    "def calculate_area(width, height):\n    return width * height",
    "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n<|dorina_thought|>\nI am Dorina AI.<|im_end|>",
    "Yapay zeka modelleri kod yazmayı ve Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) öğreniyor.",
    "2026 yılında uzay araştırmaları için 1024 yeni uydu fırlatıldı."
]

for s in test_sentences:
    encoded = tokenizer.encode(s)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\n[Girdi   ]: {repr(s)}")
    print(f"[Tokenler]: {encoded.tokens[:10]} ... (Toplam: {len(encoded.ids)} token)")
    print(f"[ID'ler  ]: {encoded.ids[:10]} ...")
    print(f"[Çözülen ]: {repr(decoded)}")
    assert decoded == s, "Kritik Hata: Tokenizer çözülen metin girdiyle birebir eşleşmedi!"

print("\n" + "=" * 70)
print("🎉 DORINA-32K TOKENIZER %100 BAŞARIYLA TAMAMLANDI VE DOĞRULANDI!")
print("=" * 70)
