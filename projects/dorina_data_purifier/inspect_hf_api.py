"""
Dorina HF Web Sample Inspector (UTF-8 safe)
"""

import urllib.request
import json
import sys

# Windows konsolunda UTF-8 çıktısı
sys.stdout.reconfigure(encoding='utf-8')

url = "https://datasets-server.huggingface.co/rows?dataset=allenai%2Fc4&config=tr&split=train&offset=0&limit=10"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        rows = data.get("rows", [])
        
        for idx, item in enumerate(rows[:4], 1):
            row_data = item.get("row", {})
            text = row_data.get("text", "")
            doc_url = row_data.get("url", "Bilinmeyen URL")
            print("="*70)
            print(f"[AÇIK WEB DOKÜMANI #{idx}] URL: {doc_url}")
            print(f"Karakter: {len(text)} | Kelime: {len(text.split())}")
            print("-" * 70)
            print("--- METNİN BAŞI (İLK 300 KARAKTER) ---")
            print(text[:300])
            print("\n--- METNİN SONU (SON 200 KARAKTER) ---")
            print(text[-200:])
            print("="*70 + "\n")

except Exception as e:
    print(f"Hata: {e}")
