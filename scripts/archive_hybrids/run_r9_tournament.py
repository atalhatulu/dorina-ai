"""
================================================================================
DORINA-MoE: RUN 9 MELEZLER TURNUVASI (scripts/run_r9_tournament.py)
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
    ("hybrid_surgery_v2_ultimate", "Süper Çekmece V2 (Exp0,1=12k, Exp2=9k, Exp3=R9 15k)"),
    ("hybrid_r9_champ_70_30", "%70 Şampiyon + %30 Run 9 15k (Hafif Bilim Aşısı)"),
    ("hybrid_r9_champ_50_50", "%50 Şampiyon + %50 Run 9 15k (Dengeli Lehim)")
]

def main():
    print("=" * 80)
    print("🏆 RUN 9 YENİ MELEZLER TURNUVASI BAŞLIYOR")
    print("=" * 80)

    results = []

    for model_name, desc in CANDIDATES:
        model_path = os.path.join(MODELS_DIR, model_name, "model.pt")
        score_path = os.path.join(MODELS_DIR, model_name, "benchmark_score.json")

        if not os.path.exists(model_path):
            continue

        if os.path.exists(score_path):
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

    print("\n" + "=" * 90)
    print("🏆 YENİ MELEZ TURNUVASI LİDERLİK TABLOSU")
    print("=" * 90)
    results.sort(key=lambda x: x[2], reverse=True)

    header = f"{'Sıra':<4} | {'Model':<28} | {'Skor':<6} | {'Syntax':<7} | {'Algo':<7} | {'Math':<7} | {'Sci':<7} | {'Inst':<7}"
    print(header)
    print("-" * len(header))

    for idx, (m_name, desc, score, cats) in enumerate(results, 1):
        py_s = cats.get("Python Syntax", 0)
        alg_s = cats.get("Algorithms", 0)
        math_s = cats.get("Math & Reasoning", 0)
        sci_s = cats.get("Science Knowledge", 0)
        inst_s = cats.get("Instruction Adherence", 0)
        print(f"{idx:<4} | {m_name:<28} | {score:<6.1f} | {py_s:<7.1f} | {alg_s:<7.1f} | {math_s:<7.1f} | {sci_s:<7.1f} | {inst_s:<7.1f}")

    print("=" * 90)

if __name__ == "__main__":
    main()
