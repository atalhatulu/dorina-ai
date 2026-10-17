"""
================================================================================
DORINA-MoE: 100 PUANLIK STANDART BENCHMARK SÜİTİ (scripts/benchmark_suite.py)
================================================================================
Amaç:
  Bu script, eğitilen her Dorina modelini 4 ana kategoride 100 puan üzerinden
  sınava sokan, puanlayan ve ayrıntılı karne (JSON) üreten standart değerlendirme motorudur.

Kategoriler & Puan Dağılımı (Toplam: 100 Puan):
  1. Python Syntax & Kod Mantığı (25 Puan) -> Girinti, döngüler, taban durumlar, algoritmalar
  2. Algoritmik & Matematiksel Akıl Yürütme (25 Puan) -> Fibonacci, faktöriyel, modüler hesap
  3. Bilim & Dünya Bilgisi (25 Puan) -> Fizik, biyoloji, bilgisayar mimarisi kavramları
  4. Talimat Sadakati & Biçimlendirme (25 Puan) -> JSON, ChatML formatı, liste yapıları

Kullanım:
  python scripts/benchmark_suite.py <model_path> <output_json_path>
================================================================================
"""

import os
import sys
import json
import time
import re
import torch
from tokenizers import Tokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.model_moe import DorinaMoEGPT

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# DORINA 100-POINT STANDARDIZED BENCHMARK SUITE
# 4 Kategoride 25'er puan = 100 Puan
# 1. Python Syntax & Code Logic (25 Puan)
# 2. Algorithmic & Mathematical Reasoning (25 Puan)
# 3. Science & World Knowledge (25 Puan)
# 4. Instruction Adherence & Structure (25 Puan)
# =============================================================================

BENCHMARK_TASKS = [
    # ---------------------------------------------------------
    # KATEGORI 1: Python Syntax & Code Logic (25 Puan)
    # ---------------------------------------------------------
    {
        "id": "code_1_prime",
        "category": "Code Logic",
        "weight": 5.0,
        "prompt": "def is_prime(n):\n    \"\"\"Check if an integer is a prime number.\"\"\"\n",
        "criteria": [
            ("indentation", r"    \w+", 1.0, "Indent/girinti doğruluğu"),
            ("base_case", r"(n\s*<=?\s*1|n\s*<\s*2|return\s+False)", 1.5, "Negatif ve 1 taban kontrolü"),
            ("loop_or_range", r"(for\s+\w+\s+in\s+range|while)", 1.5, "Bölünebilirlik döngüsü"),
            ("return_stmt", r"return\s+(True|False)", 1.0, "Boolean dönüş ifadesi")
        ]
    },
    {
        "id": "code_2_max",
        "category": "Code Logic",
        "weight": 5.0,
        "prompt": "def find_max(numbers):\n    \"\"\"Find and return the maximum number in a list.\"\"\"\n",
        "criteria": [
            ("base_case", r"(if\s+not\s+numbers|len\(numbers\)\s*==\s*0|return\s+None)", 1.5, "Boş liste kontrolü"),
            ("max_init", r"(max_val|largest|m)\s*=\s*(numbers\[0\]|-float|float)", 1.5, "Başlangıç değeri ataması"),
            ("compare", r"(if\s+\w+\s*>\s*\w+|max\(|return\s+max)", 2.0, "Karşılaştırma veya max kullanımı")
        ]
    },
    {
        "id": "code_3_binary_search",
        "category": "Code Logic",
        "weight": 5.0,
        "prompt": "def binary_search(arr, target):\n    \"\"\"Binary search algorithm.\"\"\"\n    low = 0\n    high = len(arr) - 1\n",
        "criteria": [
            ("loop", r"while\s+low\s*<=\s*high:", 1.5, "While döngü koşulu"),
            ("mid", r"mid\s*=\s*\(?low\s*\+\s*high\)?\s*//\s*2", 1.5, "Orta nokta hesabı"),
            ("conditions", r"(arr\[mid\]\s*==\s*target|arr\[mid\]\s*<\s*target)", 2.0, "Hedef karşılaştırmaları")
        ]
    },
    {
        "id": "code_4_reverse",
        "category": "Code Logic",
        "weight": 5.0,
        "prompt": "def reverse_string(s):\n    \"\"\"Return the reversed string.\"\"\"\n",
        "criteria": [
            ("slice_or_loop", r"(\[::-1\]|reversed\(|for\s+|join)", 3.0, "Tersine çevirme mantığı"),
            ("return_stmt", r"return\s+", 2.0, "Sonuç döndürme")
        ]
    },
    {
        "id": "code_5_oop",
        "category": "Code Logic",
        "weight": 5.0,
        "prompt": "class BankAccount:\n    def __init__(self, owner, balance=0):\n        self.owner = owner\n        self.balance = balance\n\n    def deposit(self, amount):\n",
        "criteria": [
            ("check_amount", r"(if\s+amount\s*>\s*0|self\.balance\s*\+=\s*amount)", 2.5, "Bakiye artırma mantığı"),
            ("return_or_attr", r"(return\s+self\.balance|self\.balance)", 2.5, "Bakiye güncelleme/döndürme")
        ]
    },

    # ---------------------------------------------------------
    # KATEGORI 2: Algorithmic & Mathematical Reasoning (25 Puan)
    # ---------------------------------------------------------
    {
        "id": "math_1_fib",
        "category": "Math & Reasoning",
        "weight": 6.0,
        "prompt": "def fibonacci(n):\n    \"\"\"Generate the nth Fibonacci number.\"\"\"\n",
        "criteria": [
            ("base_case", r"(if\s+n\s*<=?\s*1|if\s+n\s*==\s*0|return\s+n)", 2.5, "Fibonacci taban koşulu"),
            ("step", r"(fibonacci\(n\s*-\s*1\)|a,\s*b\s*=\s*b|b,\s*a\s*\+\s*b)", 3.5, "Adım ilerletme mantığı")
        ]
    },
    {
        "id": "math_2_factorial",
        "category": "Math & Reasoning",
        "weight": 6.0,
        "prompt": "def factorial(n):\n    \"\"\"Calculate n factorial (n!).\"\"\"\n",
        "criteria": [
            ("base_case", r"(if\s+n\s*<=?\s*1|if\s+n\s*==\s*0|return\s+1)", 2.5, "Faktöriyel 0/1 taban koşulu"),
            ("multiplication", r"(n\s*\*\s*factorial|result\s*\*=|math\.prod)", 3.5, "Çarpım hesabı")
        ]
    },
    {
        "id": "math_3_arithmetic",
        "category": "Math & Reasoning",
        "weight": 7.0,
        "prompt": "Question: What is 15 multiplied by 4, and what is half of that result?\nAnswer: 15 multiplied by 4 is",
        "criteria": [
            ("mult_answer", r"(60)", 4.0, "15 * 4 = 60 tespiti"),
            ("half_answer", r"(30|half\s+is\s+30)", 3.0, "Yarısı 30 tespiti")
        ]
    },
    {
        "id": "math_4_even_odd",
        "category": "Math & Reasoning",
        "weight": 6.0,
        "prompt": "def is_even(number):\n    \"\"\"Check if number is even.\"\"\"\n",
        "criteria": [
            ("modulo", r"(number\s*%\s*2\s*==\s*0|% 2)", 3.5, "Mod 2 kontrolü"),
            ("return_bool", r"return\s+", 2.5, "Sonuç dönüşü")
        ]
    },

    # ---------------------------------------------------------
    # KATEGORI 3: Science & World Knowledge (25 Puan)
    # ---------------------------------------------------------
    {
        "id": "sci_1_gravity",
        "category": "Science Knowledge",
        "weight": 6.0,
        "prompt": "Question: What is gravity and how does it work?\nAnswer: Gravity is",
        "criteria": [
            ("force_mass", r"(force|attract|mass|spacetime|curvature|Einstein|Newton)", 3.0, "Kütleçekim ve kütle ilişkisi"),
            ("physics_coherence", r"(pulls|objects|matter|earth|downward|accelerat)", 3.0, "Fiziksel tutarlılık")
        ]
    },
    {
        "id": "sci_2_photosynthesis",
        "category": "Science Knowledge",
        "weight": 6.0,
        "prompt": "Question: Explain photosynthesis in plants.\nAnswer: Photosynthesis is the process where plants convert",
        "criteria": [
            ("inputs", r"(sunlight|light|carbon\s+dioxide|water|co2)", 3.0, "Girdiler (ışık, su, CO2)"),
            ("outputs", r"(glucose|sugar|energy|oxygen|o2)", 3.0, "Çıktılar (glikoz, oksijen)")
        ]
    },
    {
        "id": "sci_3_ram_vs_disk",
        "category": "Science Knowledge",
        "weight": 6.0,
        "prompt": "Question: What is the difference between RAM and a hard drive (SSD/HDD)?\nAnswer: RAM is",
        "criteria": [
            ("ram_volatile", r"(volatile|temporary|faster|fast|memory|lost)", 3.0, "RAM geçici/hızlı hafıza"),
            ("storage_permanent", r"(permanent|storage|saved|drive|disk|persist|long-term)", 3.0, "SSD/Disk kalıcı depolama")
        ]
    },
    {
        "id": "sci_4_dna",
        "category": "Science Knowledge",
        "weight": 7.0,
        "prompt": "Question: What does DNA stand for and what is its role in living organisms?\nAnswer: DNA stands for",
        "criteria": [
            ("dna_name", r"(deoxyribonucleic\s+acid|genetic)", 4.0, "Deoxyribonucleic acid / genetik"),
            ("dna_role", r"(instructions|heredity|code|genes|organism|protein|biological)", 3.0, "Biyolojik kod / bilgi rolü")
        ]
    },

    # ---------------------------------------------------------
    # KATEGORI 4: Instruction Adherence & Structure (25 Puan)
    # ---------------------------------------------------------
    {
        "id": "instruct_1_json",
        "category": "Instruction Adherence",
        "weight": 6.0,
        "prompt": "Generate a valid JSON object with keys 'status' (value 'success') and 'code' (value 200):\n{\n",
        "criteria": [
            ("key_status", r"\"status\"\s*:\s*\"success\"", 3.0, "JSON 'status': 'success'"),
            ("key_code", r"\"code\"\s*:\s*200", 2.0, "JSON 'code': 200"),
            ("json_close", r"\}", 1.0, "JSON kapanış parantezi")
        ]
    },
    {
        "id": "instruct_2_chatml",
        "category": "Instruction Adherence",
        "weight": 7.0,
        "prompt": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "criteria": [
            ("capital_paris", r"Paris", 4.0, "Paris cevabı"),
            ("assistant_structure", r"(Paris|France|capital)", 3.0, "Mantıksal cevap bağlamı")
        ]
    },
    {
        "id": "instruct_3_summary",
        "category": "Instruction Adherence",
        "weight": 6.0,
        "prompt": "Summarize in one sentence: 'Artificial neural networks are computing systems inspired by the biological neural networks that constitute animal brains.'\nSummary:",
        "criteria": [
            ("brain_or_neural", r"(neural|network|brain|comput|model|system)", 3.0, "Anahtar kavramların varlığı"),
            ("conciseness", r"(\.\s*$|\n|[A-Za-z0-9\s]{10,120})", 3.0, "Kısa ve tek cümlelik özet")
        ]
    },
    {
        "id": "instruct_4_bullet_points",
        "category": "Instruction Adherence",
        "weight": 6.0,
        "prompt": "List 3 advantages of using Python:\n1.",
        "criteria": [
            ("bullet_1", r"1\..+", 2.0, "1. Madde formatı"),
            ("bullet_2", r"(2\..+|\n2\.)", 2.0, "2. Madde devamı"),
            ("python_traits", r"(easy|readable|simple|librar|community|syntax)", 2.0, "Python avantaj terimleri")
        ]
    }
]


def load_model_and_tokenizer(model_path):
    tokenizer_path = os.path.join(ROOT_DIR, "tokenizer", "dorina_tokenizer_32k.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer bulunamadı: {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint bulunamadı: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        cfg = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "tok_emb.weight" in checkpoint:
        # Doğrudan kaydedilmiş state_dict
        cfg = {}
        state_dict = checkpoint
    else:
        cfg = {}
        state_dict = checkpoint
    
    model = DorinaMoEGPT(
        vocab_size=cfg.get("vocab_size", 32768),
        context_length=cfg.get("context_length", 512),
        n_embd=cfg.get("n_embd", 384),
        n_head=cfg.get("n_head", 6),
        n_layer=cfg.get("n_layer", 6),
        num_experts=cfg.get("num_experts", 4),
        top_k=cfg.get("top_k", 2),
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, tokenizer, checkpoint


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.6, top_k=30):
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    gen_tokens = out[0].tolist()
    full_text = tokenizer.decode(gen_tokens)
    # Sadece üretilen kısmı da ayıralım
    new_tokens = gen_tokens[len(input_ids):]
    generated_completion = tokenizer.decode(new_tokens)
    
    return full_text, generated_completion


def evaluate_task(task, full_text, completion):
    task_score = 0.0
    rubric_results = []
    
    # regex aramasını hem tam metinde hem completion'da yapalım
    eval_target = full_text
    
    for crit_name, pattern, max_points, desc in task["criteria"]:
        match = re.search(pattern, eval_target, re.IGNORECASE | re.MULTILINE)
        if match:
            task_score += max_points
            rubric_results.append({
                "criterion": crit_name,
                "desc": desc,
                "passed": True,
                "points": max_points,
                "max_points": max_points
            })
        else:
            rubric_results.append({
                "criterion": crit_name,
                "desc": desc,
                "passed": False,
                "points": 0.0,
                "max_points": max_points
            })
            
    # Görev ağırlığını aşmasın
    task_score = min(task_score, task["weight"])
    return task_score, rubric_results


def run_benchmark(model_path, output_json=None):
    print("=" * 80)
    print(f"🏛️  DORINA STANDARDIZED BENCHMARK SUITE (100 PUANLIK SINAV)")
    print(f"📦 Model: {model_path}")
    print("=" * 80)
    
    start_time = time.time()
    model, tokenizer, checkpoint = load_model_and_tokenizer(model_path)
    
    category_scores = {
        "Code Logic": 0.0,
        "Math & Reasoning": 0.0,
        "Science Knowledge": 0.0,
        "Instruction Adherence": 0.0
    }
    
    detailed_results = []
    
    print("\n--- TESTLER BAŞLATILIYOR ---\n")
    
    for idx, task in enumerate(BENCHMARK_TASKS, 1):
        full_text, completion = generate_text(
            model, tokenizer, task["prompt"],
            max_new_tokens=100, temperature=0.5, top_k=25
        )
        
        task_score, rubrics = evaluate_task(task, full_text, completion)
        category_scores[task["category"]] += task_score
        
        print(f"[{idx:02d}/{len(BENCHMARK_TASKS):02d}] ({task['category']}) {task['id']}: {task_score:.1f}/{task['weight']:.1f} Puan")
        print(f"   Prompt: {repr(task['prompt'][:45])}...")
        print(f"   Çıktı : {repr(completion.strip()[:65])}...")
        
        detailed_results.append({
            "id": task["id"],
            "category": task["category"],
            "score": task_score,
            "max_score": task["weight"],
            "prompt": task["prompt"],
            "completion": completion,
            "rubrics": rubrics
        })
        
    total_score = sum(category_scores.values())
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"🏆 SINAV SONUÇ KARNESİ (MODEL: {os.path.basename(os.path.dirname(model_path))})")
    print("=" * 80)
    print(f"1. Python Syntax & Code Logic        : {category_scores['Code Logic']:5.1f} / 25.0 Puan")
    print(f"2. Algorithmic & Math Reasoning      : {category_scores['Math & Reasoning']:5.1f} / 25.0 Puan")
    print(f"3. Science & World Knowledge         : {category_scores['Science Knowledge']:5.1f} / 25.0 Puan")
    print(f"4. Instruction Adherence & Structure : {category_scores['Instruction Adherence']:5.1f} / 25.0 Puan")
    print("-" * 80)
    print(f"🎯 TOPLAM GENEL ZEKA PUANI           : {total_score:5.1f} / 100.0 PUAN")
    print(f"⏱️  Test Süresi                       : {elapsed:.1f} saniye")
    print("=" * 80)
    
    benchmark_report = {
        "model_path": model_path,
        "run_name": os.path.basename(os.path.dirname(model_path)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_score": round(total_score, 2),
        "max_score": 100.0,
        "category_scores": {k: round(v, 2) for k, v in category_scores.items()},
        "val_loss": checkpoint.get("val_loss", None),
        "step": checkpoint.get("step", None),
        "detailed_results": detailed_results
    }
    
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
        print(f"✓ Detaylı benchmark raporu kaydedildi: {output_json}")
        
    return benchmark_report


if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "run4", "model.pt")
    target_out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(target_model), "benchmark_score.json")
    
    run_benchmark(target_model, target_out)
