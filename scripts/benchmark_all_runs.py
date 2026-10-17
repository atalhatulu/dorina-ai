"""
================================================================================
DORINA-MoE: TÜM RUN'LAR İÇİN GENEL BENCHMARK VE KARŞILAŞTIRMA (scripts/benchmark_all_runs.py)
================================================================================
Amaç:
  Bu script, models/dorina_44m_moe/ dizinindeki tüm aktif run modellerini (run1, run2,
  run3, run4, run5...) sırasıyla 100 puanlık teste sokar, modeller arasındaki puan
  artışını / gerilemesini hesaplar ve tek bir özet gelişim tablosu olarak raporlar.

Çıktı:
  - Her run klasörüne: benchmark_score.json
  - Genel özet: models/dorina_44m_moe/benchmark_comparison.json
================================================================================
"""

import os
import sys
import json
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.benchmark_suite import run_benchmark

def main():
    runs = ["run1", "run2", "run3", "run4"]
    results = {}
    
    print("\n" + "=" * 80)
    print("🏛️  DORINA-MoE: TÜM RUN'LAR İÇİN GENEL BENCHMARK KARŞILAŞTIRMASI")
    print("=" * 80)
    
    for r in runs:
        model_path = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", r, "model.pt")
        out_json = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", r, "benchmark_score.json")
        if not os.path.exists(model_path):
            print(f"⚠️  {r} modeli bulunamadı: {model_path}")
            continue
            
        print(f"\n🚀 {r.upper()} İÇİN BENCHMARK BAŞLIYOR...")
        rep = run_benchmark(model_path, out_json)
        results[r] = rep
        
    print("\n" + "=" * 90)
    print("📊 DORINA-MoE BÜYÜK GELİŞİM TABLOSU (RUN1 -> RUN4)")
    print("=" * 90)
    print(f"{'Run':<8} | {'Val Loss':<10} | {'Kod (25)':<10} | {'Mantık (25)':<12} | {'Bilim (25)':<12} | {'Format (25)':<12} | {'TOPLAM (100)':<12}")
    print("-" * 90)
    
    for r, rep in results.items():
        cats = rep["category_scores"]
        val_loss = f"{rep['val_loss']:.4f}" if rep.get("val_loss") else "N/A"
        print(f"{r:<8} | {val_loss:<10} | {cats.get('Code Logic', 0.0):<10.1f} | {cats.get('Math & Reasoning', 0.0):<12.1f} | {cats.get('Science Knowledge', 0.0):<12.1f} | {cats.get('Instruction Adherence', 0.0):<12.1f} | {rep['total_score']:<12.1f}")
        
    print("=" * 90)
    
    # Genel karşılaştırma dosyasını kaydet
    summary_path = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "benchmark_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Genel karşılaştırma tablosu kaydedildi: {summary_path}")

if __name__ == "__main__":
    main()
