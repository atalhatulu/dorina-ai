"""
Dorina TTS - Gelişmiş Türkçe Metin Normalizasyonu & Tokenizer
Sayıları, noktalama işaretlerini ve Türkçe harf dönüşümlerini (I/ı, İ/i) hatasız yönetir.
"""
import re

# Özel Tokenler ve Türkçe Karakter Kümesi
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SIL>"]
TURKISH_CHARS = list("abcçdefgğhıijklmnoöprsştuüvyz")
EXTENDED_CHARS = list("âîû") # Claude'un belirttiği düzeltme/şapka harfleri
PUNCTUATIONS = list(".,!? -:;")

VOCAB = SPECIAL_TOKENS + TURKISH_CHARS + EXTENDED_CHARS + PUNCTUATIONS
CHAR_TO_ID = {ch: idx for idx, ch in enumerate(VOCAB)}
ID_TO_CHAR = {idx: ch for idx, ch in enumerate(VOCAB)}

PAD_ID = CHAR_TO_ID["<PAD>"]
UNK_ID = CHAR_TO_ID["<UNK>"]
BOS_ID = CHAR_TO_ID["<BOS>"]
EOS_ID = CHAR_TO_ID["<EOS>"]
SIL_ID = CHAR_TO_ID["<SIL>"]


# Türkçe Sayı Okuyucu
ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
THOUSANDS = ["", "bin", "milyon", "milyar"]


def _num_to_turkish(n: int) -> str:
    if n == 0:
        return "sıfır"
    parts = []
    
    # 3'lü basamak gruplama
    def parse_group(val):
        h = val // 100
        t = (val % 100) // 10
        o = val % 10
        res = []
        if h == 1:
            res.append("yüz")
        elif h > 1:
            res.extend([ONES[h], "yüz"])
        if t > 0:
            res.append(TENS[t])
        if o > 0:
            res.append(ONES[o])
        return " ".join(res)

    group_idx = 0
    while n > 0:
        val = n % 1000
        if val > 0:
            group_str = parse_group(val)
            # 1000 için "bir bin" denmez, "bin" denir
            if group_idx == 1 and val == 1:
                parts.append("bin")
            else:
                s = f"{group_str} {THOUSANDS[group_idx]}".strip()
                parts.append(s)
        n //= 1000
        group_idx += 1
        
    return " ".join(reversed(parts))


def normalize_turkish_text(text: str) -> str:
    """Türkçe metin normalizasyonu: sayıları ve harfleri düzeltir."""
    # 1. Elle Türkçe Lowercase (Python varsayılanı 'I' -> 'i' yapıp bozar)
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    
    # 2. Sayıları kelimelere dönüştür (örn: "1885" -> "bin sekiz yüz seksen beş")
    def replace_num(match):
        val = int(match.group(0))
        return " " + _num_to_turkish(val) + " "
        
    text = re.sub(r"\d+", replace_num, text)
    
    # 3. Fazla boşlukları ve yabancı işaretleri temizle
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_sequence(text: str) -> list[int]:
    norm_text = normalize_turkish_text(text)
    seq = [BOS_ID]
    for ch in norm_text:
        if ch in CHAR_TO_ID:
            seq.append(CHAR_TO_ID[ch])
        elif ch == " ":
            seq.append(CHAR_TO_ID.get(" ", SIL_ID))
        else:
            seq.append(UNK_ID)
    seq.append(EOS_ID)
    return seq


if __name__ == "__main__":
    t = "1885 yılında Thomas B. Ward 250 adet bizon gördü!"
    print("Normal:", normalize_turkish_text(t))
    print("Tokens:", text_to_sequence(t)[:15])
