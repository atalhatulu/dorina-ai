"""
================================================================================
DORINA-MoE: OTOMATİK MELEZ TURNUVASI (scripts/run_hybrid_tournament.py)
================================================================================
Bu script sırayla tüm melez modelleri standart 50 soruluk sınava sokar,
sonuçları toplar ve bir şampiyonluk tablosu yazdırır.
================================================================================
"""

import os
import sys
import json
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models", "dorina_44m_moe")
PY_BIN = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")
BENCH_SCRIPT = os.path.join(ROOT_DIR, "scripts", "benchmark_suite.py")

CANDIDATES = [
    ("hybrid_slerp_12k_final", "Model A (%50 Step 12k + %50 Final)"),
    ("hybrid_12k_final_70_30", "%70 Step 12k + %30 Final"),
    ("hybrid_12k_final_30_70", "%30 Step 12k + %70 Final"),
    ("hybrid_r7_r8_12k_50_50", "%50 Run 7 + %50 Run 8 12k (İki Zirve Füzyonu)"),
    ("hybrid_expert_surgery", "Model B (Çekmece: Exp0,1=12k, Exp2=9k, Exp3=Final)"),
    ("hybrid_expert_r7_r8", "Model C (Run 7 Taban + 9k Sentaks + Final Bilim)")
]

def main():
    print("=" * 80)
    print("🏆 DORINA-MoE: BÜYÜK MELEZ MODELLER TURNUVASI BAŞLIYOR")
    print("=" * 80)

    results = []

    for model_name, desc in CANDIDATES:
        model_path = os.path.join(MODELS_DIR, model_name, "model.pt")
        score_path = os.path.join(MODELS_DIR, model_name, "benchmark_score.json")

        if not os.path.exists(model_path):
            print(f"⚠️ Model dosyası bulunamadı: {model_path}")
            continue

        # Eğer daha önce test edilmişse tekrar çalıştırma, direkt oku
        if os.path.exists(score_path):
            print(f"\n⚡ Zaten test edilmiş: {model_name} ({desc})")
            with open(score_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append((model_name, desc, data["total_score"], data.get("category_scores", {})))
            continue

        print(f"\n🚀 Sınav Başlatılıyor: {model_name}")
        print(f"   Açıklama: {desc}")
        cmd = [PY_BIN, BENCH_SCRIPT, model_path, score_path]
        res = subprocess.run(cmd, cwd=ROOT_DIR)

        if res.returncode == 0 and os.path.exists(score_path):
            with open(score_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append((model_name, desc, data["total_score"], data.get("category_scores", {})))
            print(f"   ✓ Tamamlandı! Skor: {data['total_score']} / 100")
        else:
            print(f"   ❌ Sınavda hata oluştu: {model_name}")

    print("\n" + "=" * 90)
    print("🏆 BÜYÜK MELEZ TURNUVASI LİDERLİK TABLOSU (LEADERBOARD)")
    print("=" * 90)
    results.sort(key=lambda x: x[2], reverse=True)

    header = f"{'Sıra':<4} | {'Model':<25} | {'Skor':<6} | {'Syntax':<7} | {'Algo':<7} | {'Math':<7} | {'Sci':<7} | {'Inst':<7}"
    print(header)
    print("-" * len(header))

    for idx, (m_name, desc, score, cats) in enumerate(results, 1):
        py_s = cats.get("Python Syntax", 0)
        alg_s = cats.get("Algorithms", 0)
        math_s = cats.get("Math & Reasoning", 0)
        sci_s = cats.get("Science Knowledge", 0)
        inst_s = cats.get("Instruction Adherence", 0)
        print(f"{idx:<4} | {m_name:<25} | {score:<6.1f} | {py_s:<7.1f} | {alg_s:<7.1f} | {math_s:<7.1f} | {sci_s:<7.1f} | {inst_s:<7.1f}")

    print("=" * 90)

if __name__ == "__main__":
    main()
