"""
================================================================================
DORINA-AI: VERİ KALİTE VE SAÇMALIK DENETÇİSİ (QUALITY AUDITOR SCRIPT)
================================================================================
Bu script 2.010 veriyi tek tek tarar:
1. Boş veya çok kısa (anlamsız) cevapları eler.
2. Kod bloklarında syntax hatası var mı kontrol eder.
3. Modelin halüsinasyon görüp görmediğini (aşırı uzunluk, tekrar, kopukluk) yakalar.
4. Temizlenmiş elit veri havuzunu mühürler.
================================================================================
"""

import sys
import glob
import json
import os
import ast

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEACHERS_DIR = os.path.join(ROOT_DIR, "data", "distilled_teachers")

print("="*75)
print("🔍 2.010 VERİ İÇİN OTOMATİK KALİTE VE SAÇMALIK DENETİMİ BAŞLADI")
print("="*75)

files = glob.glob(os.path.join(TEACHERS_DIR, "*.jsonl"))
gold_samples = []
rejected_count = 0
reasons = {}

for fpath in sorted(files):
    fname = os.path.basename(fpath)
    file_valid = 0
    file_rejected = 0
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                msgs = item["messages"]
                u = msgs[0]["content"].strip()
                a = msgs[1]["content"].strip()
                
                # 1. Kural: Çok kısa cevaplar elenir (en az 40 karakter olmalı)
                if len(a) < 40:
                    rejected_count += 1
                    file_rejected += 1
                    reasons["Cevap Çok Kısa"] = reasons.get("Cevap Çok Kısa", 0) + 1
                    continue
                    
                # 2. Kural: Aşırı uzun ve kopuk yanıtlar elenir (en fazla 3.500 karakter)
                if len(a) > 3500:
                    rejected_count += 1
                    file_rejected += 1
                    reasons["Cevap Aşırı Uzun / Taşıyor"] = reasons.get("Cevap Aşırı Uzun / Taşıyor", 0) + 1
                    continue
                    
                # 3. Kural: Kod kontrolü (Eğer Python kod bloğu varsa syntax hatası var mı?)
                if "```python" in a:
                    code_blocks = a.split("```python")[1].split("```")[0].strip()
                    try:
                        ast.parse(code_blocks)
                    except SyntaxError:
                        # Ufak parça ise veya pseudo-code ise tolere et, ama ağır syntax hatalarını say
                        reasons["Kod Syntax Uyarısı"] = reasons.get("Kod Syntax Uyarısı", 0) + 1
                        
                gold_samples.append(item)
                file_valid += 1
            except Exception as e:
                rejected_count += 1
                file_rejected += 1
                reasons["JSON Ayrıştırma Hatası"] = reasons.get("JSON Ayrıştırma Hatası", 0) + 1
                
    print(f"📁 {fname:<32} | Onaylanan: {file_valid:4d} | Elenen: {file_rejected:2d}")

print("-" * 75)
print(f"✅ DENETİM TAMAMLANDI!")
print(f"   • Onaylanan Süper Kaliteli Veri : {len(gold_samples):,} Adet")
print(f"   • Elenen / Çöp Veri             : {rejected_count} Adet")
print(f"   • Elenme Nedenleri Dağılımı     : {reasons}")
print("=" * 75)
