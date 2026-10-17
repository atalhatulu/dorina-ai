import os
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Proje dizinleri
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "tokenizer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_custom_tokenizer(vocab_size=32768):
    print("=" * 70)
    print("🦀 DORINA TOKENIZER (RUST BYTE-LEVEL BPE)")
    print(f"🎯 Hedef Kelime Haznesi (Vocab): {vocab_size:,}")
    print("=" * 70)

    # 1. Byte-Level BPE Model Başlat (GPT-4 / Llama 3 Stili)
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Özel Sistem ve Kontrol Etiketleri (Native Special Tokens)
    special_tokens = [
        "<|pad|>",
        "<|unk|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|dorina_thought|>",
        "<|code_start|>",
        "<|code_end|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>"
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True
    )

    print("\n📦 Tokenizer mimarisi ve özel etiketler tanımlandı.")
    print("   ✓ Special Tokens:", special_tokens)
    return tokenizer, trainer

if __name__ == "__main__":
    train_custom_tokenizer()
