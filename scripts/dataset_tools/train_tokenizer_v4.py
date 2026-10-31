import os
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Digits, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def build_dorina_v4_tokenizer():
    print("=" * 60)
    print("Building Dorina V4 BPE Tokenizer (32K English + Clean Code)")
    print("Individual Digit Splitting: ACTIVE (No arithmetic blindness)")
    print("=" * 60)

    # ByteLevel + Individual Digits pre-tokenizer
    # Rakamlar tek tek ayrılır: '160' -> ['1', '6', '0']
    tokenizer = Tokenizer(BPE(unk_token=None))
    
    tokenizer.pre_tokenizer = Sequence([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    ])
    
    tokenizer.decoder = ByteLevelDecoder()

    # Özel kontrol tokenları
    special_tokens = [
        "<|endoftext|>",   # 0: Pre-train döküman ayırıcı
        "<|im_start|>",     # 1: ChatML başlangıç
        "<|im_end|>",       # 2: ChatML bitiş
    ]

    trainer = BpeTrainer(
        vocab_size=32768,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet()
    )

    out_dir = "tokenizers"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dorina_tokenizer_v4_32k.json")

    # Geçici eğitim korpusu: İngilizce cümleler, Python kodu ve aritmetik
    print("\n[Tokenizer Eğitimi] Örnek metinler toplanıyor...")
    sample_texts = [
        # Aritmetik ve sayılar
        "The sum of 2 and 5 is 7. If I have 3 apples and I eat 1, I have 2 apples left.",
        "Calculate 7 times 8 = 56. What is 123 + 456? The answer is 579.",
        "Mathematics, algebra, calculus, geometry, matrix multiplication, neural networks.",
        # Python ve Kod sözdizimi
        "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "class Node:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next",
        "import os\nimport sys\nimport torch\nimport numpy as np\nfrom dataclasses import dataclass",
        # ChatML formatı
        "<|im_start|>system\nYou are Dorina, a helpful and concise AI assistant developed by Talha.<|im_end|>\n<|im_start|>user\nHello! How are you?<|im_end|>\n<|im_start|>assistant\nHello! I am doing well, ready to help you with coding and mathematics.<|im_end|>"
    ]

    # HuggingFace streaming'den birkaç bin satırlık temiz FineWeb ve Python örneği ekleyelim
    try:
        from datasets import load_dataset
        print("[Streaming] HuggingFace FineWeb-Edu'dan temiz İngilizce metinler çekiliyor...")
        ds_web = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        count = 0
        for item in ds_web:
            sample_texts.append(item["text"][:1000])
            count += 1
            if count >= 3000:
                break
        print(f"[Streaming OK] {count} adet FineWeb-Edu belgesi eklendi.")
    except Exception as e:
        print(f"[Streaming Uyarısı] FineWeb çekilemedi ({e}), yerel zengin örneklerle devam ediliyor.")

    # Tokenizer eğit
    print(f"\n[Eğitiliyor] Toplam {len(sample_texts)} metin parçası üzerinden BPE eğitiliyor...")
    tokenizer.train_from_iterator(sample_texts, trainer=trainer)
    tokenizer.save(out_path)
    print(f"[BAŞARILI] Tokenizer kaydedildi: {out_path}")

    # Test doğrulama
    print("\n" + "=" * 50)
    print("Tokenizer Doğrulama Testleri:")
    print("=" * 50)
    
    # 1. Rakam testi
    test_math = "2 + 5 = 7 and 160 / 4"
    enc_math = tokenizer.encode(test_math)
    print(f"Girdi: '{test_math}'")
    print(f"Tokenler: {enc_math.tokens}")
    print(f"ID'ler: {enc_math.ids}")
    
    # 2. ChatML testi
    test_chat = "<|im_start|>user\nHello Dorina<|im_end|>"
    enc_chat = tokenizer.encode(test_chat)
    print(f"\nChatML Girdi: '{test_chat}'")
    print(f"Tokenler: {enc_chat.tokens}")
    print(f"ID'ler: {enc_chat.ids}")
    
    print("\nDorina V4 Tokenizer Kullanıma Hazır!")

if __name__ == "__main__":
    build_dorina_v4_tokenizer()
