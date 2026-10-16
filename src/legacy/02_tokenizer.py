from collections import Counter
import json

class DorinaTokenizer:
    def __init__(self):
        # 0-255 arası temel byte'lar (her karakter başlangıçta kendi byte değeridir)
        self.merges = {}        # (token1, token2) -> yeni_token_id
        self.vocab = {i: bytes([i]) for i in range(256)} # id -> byte karşılığı

    def _get_pair_stats(self, ids):
        """Yan yana gelen token çiftlerinin kaç kez geçtiğini sayar."""
        counts = Counter()
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def _merge_pair(self, ids, pair, new_id):
        """Dizideki tüm (pair[0], pair[1]) ikililerini new_id ile değiştirir."""
        new_ids = []
        i = 0
        while i < len(ids):
            # Eğer aradığımız ikiliyi bulduysak, ikisini birleştirip tek ID yaz
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size, verbose=True):
        """
        Metin üzerinde BPE algoritmasını eğitir.
        vocab_size: Hedeflenen toplam kelime dağarcığı boyutu (256'dan büyük olmalı).
        """
        assert vocab_size >= 256, "Vocab size en az 256 olmalıdır."
        num_merges = vocab_size - 256

        # Metni ham UTF-8 byte dizisine (0-255 arası tam sayılara) çevir
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)
        baslangic_uzunlugu = len(ids)

        print(f"--- DORINA TOKENIZER EĞİTİMİ BAŞLIYOR ---")
        print(f"Ham metin uzunluğu (byte): {baslangic_uzunlugu}")
        print(f"Hedeflenen birleştirme (merge) sayısı: {num_merges}\n")

        for i in range(num_merges):
            stats = self._get_pair_stats(ids)
            if not stats:
                break

            # En sık yan yana gelen ikiliyi bul
            best_pair = max(stats, key=stats.get)
            count = stats[best_pair]

            # Eğer en sık geçen ikili bile sadece 1 kez geçiyorsa durabiliriz
            if count < 2:
                break

            new_id = 256 + i
            ids = self._merge_pair(ids, best_pair, new_id)

            # Kuralı ve kelime dağarcığını kaydet
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                token_str = self.vocab[new_id].decode('utf-8', errors='replace')
                print(f"Merge {i+1:2d}/{num_merges}: {best_pair} -> Yeni Token ID: {new_id} | "
                      f"Metin: {repr(token_str):10s} (Frekans: {count})")

        bitis_uzunlugu = len(ids)
        sikistirma_orani = baslangic_uzunlugu / bitis_uzunlugu
        print(f"\n--- TOKENIZER EĞİTİMİ TAMAMLANDI ---")
        print(f"Eski uzunluk: {baslangic_uzunlugu} token -> Yeni uzunluk: {bitis_uzunlugu} token")
        print(f"Sıkıştırma Oranı (Compression Ratio): {sikistirma_orani:.2f}X\n")

    def encode(self, text):
        """String metni Dorina'nın anlayacağı Token ID listesine çevirir."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_pair_stats(ids)
            # Mevcut çiftler içinde daha önce öğrendiğimiz en düşük indeksli merge kuralını bul
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # Artık birleştirilecek tanıdık bir çift kalmadı
            new_id = self.merges[pair]
            ids = self._merge_pair(ids, pair, new_id)
        return ids

    def decode(self, ids):
        """Token ID listesini tekrar okunabilir İngilizce metne çevirir."""
        byte_dizisi = b"".join(self.vocab[idx] for idx in ids)
        return byte_dizisi.decode("utf-8", errors="replace")


# =====================================================================
# TEST EDELİM!
# =====================================================================
if __name__ == "__main__":
    # Örnek İngilizce eğitim metnimiz (Tekrar eden kelimeler ve ekler içeriyor)
    sample_text = (
        "Dorina is an artificial intelligence model built from scratch. "
        "Training an intelligence from scratch is the best way to understand how intelligence works. "
        "The model learns patterns, the model predicts the next token, and the model becomes smarter. "
        "Learning from scratch is super cool and building Dorina is fun!"
    )

    tokenizer = DorinaTokenizer()

    # 256 temel karakterin üzerine 30 yeni birleştirme (merge) öğrenmesini isteyelim (Toplam 286 vocab)
    tokenizer.train(sample_text, vocab_size=286)

    # Şimdi yeni bir cümleyi Encode ve Decode edelim
    test_cumlesi = "Dorina is learning from scratch!"
    encoded_ids = tokenizer.encode(test_cumlesi)
    decoded_text = tokenizer.decode(encoded_ids)

    print("--- TEST SONUÇLARI ---")
    print(f"Orijinal Cümle : {test_cumlesi}")
    print(f"Karakter Sayısı: {len(test_cumlesi)}")
    print(f"Token ID'leri  : {encoded_ids}")
    print(f"Token Sayısı   : {len(encoded_ids)}")

    # Her bir Token ID'nin hangi harf/kelime parçasına denk geldiğini görelim:
    parcalar = [tokenizer.decode([tid]) for tid in encoded_ids]
    print(f"Token Parçaları: {parcalar}")
    print(f"Decode Edilmiş : {decoded_text}")
