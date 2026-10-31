"""
Dorina Corpus Inspector
Streams Turkish Web Corpus from HuggingFace (mC4 / FineWeb / OSCAR)
Searches for specific high-value domains (e.g. evrimagaci.org) to inspect raw boilerplate & noise.
"""

from datasets import load_dataset
import sys

print("[1/3] Hugging Face'ten Türkçe mC4 (Multilingual C4) akışı (streaming) başlatılıyor...")

try:
    # mC4 Türkçe dilimi (streaming modda diske devasa dosya indirmez, canlı okur)
    dataset = load_dataset("allenai/c4", "tr", split="train", streaming=True)
    
    print("[2/3] Canlı akış bağlandı! İçinde 'evrimagaci.org' veya kaliteli bilim siteleri taranıyor...")
    
    found_count = 0
    scanned_count = 0
    
    for sample in dataset:
        scanned_count += 1
        url = sample.get("url", "")
        text = sample.get("text", "")
        
        if "evrimagaci.org" in url or "evrimagaci" in text.lower()[:300]:
            found_count += 1
            print("\n" + "="*70)
            print(f"🎯 [BULUNDU #{found_count}] URL: {url}")
            print(f"Toplam Karakter: {len(text)} | Kelime: {len(text.split())}")
            print("-" * 70)
            print("--- İLK 500 KARAKTER (BAŞLANGIÇ GÜRÜLTÜSÜ) ---")
            print(text[:500])
            print("\n--- SON 400 KARAKTER (BİTİŞ GÜRÜLTÜSÜ) ---")
            print(text[-400:])
            print("="*70 + "\n")
            
            if found_count >= 2:
                break
                
        if scanned_count % 1000 == 0:
            print(f"-> {scanned_count} adet genel Türkçe web sayfası tarandı...")
            
        if scanned_count >= 15000:
            print("15.000 sayfa tarandı, örnek için durduruluyor.")
            break

except Exception as e:
    print(f"Hata oluştu: {e}")
