"""
Dorina TTS - Türkçe Karakter Tokenizer ve Metin Normalizasyonu
"""
import re

# Türkçe karakter tablosu + noktalama + özel tokenler
# 0: <PAD>, 1: <UNK>, 2: <BOS>, 3: <EOS>, 4: <SIL>
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SIL>"]

TURKISH_CHARS = list("abcçdefgğhıijklmnoöprsştuüvyz")
PUNCTUATIONS = list(".,!? -:;")

VOCAB = SPECIAL_TOKENS + TURKISH_CHARS + PUNCTUATIONS
CHAR_TO_ID = {ch: idx for idx, ch in enumerate(VOCAB)}
ID_TO_CHAR = {idx: ch for idx, ch in enumerate(VOCAB)}

PAD_ID = CHAR_TO_ID["<PAD>"]
UNK_ID = CHAR_TO_ID["<UNK>"]
BOS_ID = CHAR_TO_ID["<BOS>"]
EOS_ID = CHAR_TO_ID["<EOS>"]
SIL_ID = CHAR_TO_ID["<SIL>"]


def normalize_turkish_text(text: str) -> str:
    """Türkçe metni standartlaştırır, büyük harfleri dönüştürür."""
    # Türkçe İ/I düzeltmesi
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    
    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_sequence(text: str, add_special_tokens: bool = True) -> list[int]:
    """Metni modelin anlayacağı ID dizisine dönüştürür."""
    text = normalize_turkish_text(text)
    seq = []
    
    if add_special_tokens:
        seq.append(BOS_ID)
        
    for ch in text:
        if ch in CHAR_TO_ID:
            seq.append(CHAR_TO_ID[ch])
        elif ch == " ":
            seq.append(CHAR_TO_ID.get(" ", SIL_ID))
        else:
            seq.append(UNK_ID)
            
    if add_special_tokens:
        seq.append(EOS_ID)
        
    return seq


def sequence_to_text(seq: list[int]) -> str:
    """ID dizisini tekrar okunabilir metne çevirir."""
    chars = []
    for token_id in seq:
        if token_id in ID_TO_CHAR:
            ch = ID_TO_CHAR[token_id]
            if ch not in SPECIAL_TOKENS:
                chars.append(ch)
    return "".join(chars)


if __name__ == "__main__":
    sample = "Merhaba! Ben Dorina, nasılsın?"
    norm = normalize_turkish_text(sample)
    seq = text_to_sequence(sample)
    print("Normal:", norm)
    print("Vocab Boyutu:", len(VOCAB))
    print("Sequence:", seq)
    print("Geri Çevrilen:", sequence_to_text(seq))
