"""
Dorina Quality & Heuristic Filters (Calibrated for Academic & Scientific texts)
Prevents false-positive rejection of valid technical terminology repetitions.
"""

import re
from typing import Tuple, Dict

TURKISH_STOPWORDS = {
    "ve", "bir", "bu", "da", "de", "için", "ile", "olan", "olarak", "gibi",
    "en", "çok", "daha", "kadar", "ise", "var", "yok", "ama", "fakat", "ancak",
    "bu", "şu", "o", "biz", "siz", "onlar", "tarafından", "sonra", "önce", "kendi",
    "her", "bütün", "hiç", "çünkü", "nasıl", "neden", "niçin", "veya", "ya", "hem"
}

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")

class QualityFilter:
    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 40000,
        min_tr_stopword_ratio: float = 0.03,
        min_alphanumeric_ratio: float = 0.60,
        # Bilimsel metinler için kalibre edildi (0.20 -> 0.40)
        max_repetitive_ngram_ratio: float = 0.40
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_tr_stopword_ratio = min_tr_stopword_ratio
        self.min_alphanumeric_ratio = min_alphanumeric_ratio
        self.max_repetitive_ngram_ratio = max_repetitive_ngram_ratio

    def evaluate(self, text: str) -> Tuple[bool, str, Dict[str, float]]:
        if not text:
            return False, "EMPTY_TEXT", {}

        words = text.split()
        num_words = len(words)

        if num_words < self.min_words:
            return False, f"TOO_SHORT ({num_words} < {self.min_words})", {"word_count": num_words}
        if num_words > self.max_words:
            return False, f"TOO_LONG ({num_words} > {self.max_words})", {"word_count": num_words}

        # 1. Alfanümerik oran
        chars = list(text)
        total_chars = len(chars)
        alnum_chars = sum(1 for c in chars if c.isalnum() or c.isspace())
        alnum_ratio = alnum_chars / max(1, total_chars)

        if alnum_ratio < self.min_alphanumeric_ratio:
            return False, f"LOW_ALNUM_RATIO ({alnum_ratio:.2f})", {"alnum_ratio": alnum_ratio}

        # 2. Türkçe Stop-word ve Karakter Kontrolü
        low_words = [w.lower().strip(".,!?:;\"'()[]{}") for w in words]
        tr_stop_hits = sum(1 for w in low_words if w in TURKISH_STOPWORDS)
        tr_stop_ratio = tr_stop_hits / max(1, num_words)

        has_tr_chars = any(c in TURKISH_CHARS for c in text)

        if tr_stop_ratio < self.min_tr_stopword_ratio and not has_tr_chars:
            return False, f"NOT_TURKISH (tr_stop_ratio: {tr_stop_ratio:.3f})", {"tr_stop_ratio": tr_stop_ratio}

        # 3. Cümle Yapısı
        punct_count = sum(1 for c in text if c in ".!?")
        sentence_density = punct_count / max(1, num_words)
        if sentence_density < 0.005:
            return False, f"NO_PUNCTUATION_OR_SENTENCES ({sentence_density:.3f})", {"sentence_density": sentence_density}

        # 4. 3-gram Tekrar Kontrolü (Gerçek spam'i yakalar, bilimsel terim ikililerini affeder)
        trigrams = [f"{low_words[i]} {low_words[i+1]} {low_words[i+2]}" for i in range(len(low_words)-2)]
        if trigrams:
            unique_trigrams = set(trigrams)
            repeat_ratio = 1.0 - (len(unique_trigrams) / len(trigrams))
            if repeat_ratio > self.max_repetitive_ngram_ratio:
                return False, f"HIGH_REPETITION_SPAM ({repeat_ratio:.2f})", {"repeat_ratio": repeat_ratio}
        else:
            repeat_ratio = 0.0

        metrics = {
            "word_count": num_words,
            "alnum_ratio": round(alnum_ratio, 3),
            "tr_stop_ratio": round(tr_stop_ratio, 3),
            "sentence_density": round(sentence_density, 3),
            "repeat_ratio": round(repeat_ratio, 3)
        }

        return True, "PASSED_GOLD", metrics
