"""
================================================================================
DORINA-MoE: TÜM RUN'LAR İÇİN 50 SORULUK BÜYÜK SINAV (scripts/benchmark_all_runs.py)
================================================================================
Amaç:
  Bu script, models/dorina_44m_moe/ dizinindeki tüm aktif modelleri (run1, run2,
  run3, run4, run5...) yeni 50 Soruluk Grand Benchmark sınavına sokar ve
  karşılaştırmalı büyük gelişim tablosunu üretir.
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
    runs = ["run1", "run2", "run3", "run4", "run5"]
    results = {}
    
    print("\n" + "=" * 90)
    print("🏛️  DORINA-MoE: 50 SORULUK GRAND BENCHMARK İLE TÜM MODELLERİN SINAVI (RUN 1 -> RUN 5)")
    print("=" * 90)
    
    for r in runs:
        model_path = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", r, "model.pt")
        out_json = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", r, "benchmark_50q.json")
        if not os.path.exists(model_path):
            print(f"⚠️  {r} modeli bulunamadı: {model_path}")
            continue
            
        print(f"\n🚀 [{r.upper()}] İÇİN 50 SORULUK BÜYÜK SINAV BAŞLIYOR...")
        rep = run_benchmark(model_path, out_json)
        results[r] = rep
        
    print("\n" + "=" * 105)
    print("📊 DORINA-MoE BÜYÜK GELİŞİM MATRİSİ (50 SORU / 100 PUAN)")
    print("=" * 105)
    print(f"{'Model':<8} | {'Syntax(20)':<12} | {'Algo(20)':<10} | {'Mat&Mantık(20)':<15} | {'Bilim(20)':<11} | {'Talimat(20)':<13} | {'TOPLAM(100)':<12}")
    print("-" * 105)
    
    for r, rep in results.items():
        cats = rep["category_scores"]
        print(f"{r:<8} | {cats.get('Python Syntax', 0.0):<12.1f} | {cats.get('Algorithms', 0.0):<10.1f} | {cats.get('Math & Reasoning', 0.0):<15.1f} | {cats.get('Science Knowledge', 0.0):<11.1f} | {cats.get('Instruction Adherence', 0.0):<13.1f} | {rep['total_score']:<12.1f}")
        
    print("=" * 105)
    
    summary_path = os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "benchmark_50q_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 50 Soruluk Büyük Karşılaştırma Raporu Kaydedildi: {summary_path}")

if __name__ == "__main__":
    main()
