"""
================================================================================
DORINA-MoE: 50 SORULUK BÜYÜK STANDART BENCHMARK SÜİTİ (scripts/benchmark_suite.py)
================================================================================
Amaç:
  Bu script, Dorina modellerinin genel zekasını (General Intelligence),
  kodlama kabiliyetini, algoritmik derinliğini, fen/bilgisayar bilimleri
  bilgisini ve biçimsel talimat sadakatini 50 kapsamlı soruyla ve
  100 puan üzerinden en hassas şekilde ölçen altın standart sınav motorudur.

Kategoriler & Puan Dağılımı (5 Kategori x 20 Puan = 100 Puan):
  1. Python Syntax & Temel Kodlama (20 Puan - 10 Soru)
  2. Veri Yapıları & Algoritmalar   (20 Puan - 10 Soru)
  3. Matematik & Sayısal Muhakeme  (20 Puan - 10 Soru)
  4. Bilim & Bilgisayar Mimarisi   (20 Puan - 10 Soru)
  5. Talimat Takibi & Biçimlendirme(20 Puan - 10 Soru)

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
# 50 ADET STANDARTLAŞTIRILMIŞ BENCHMARK GÖREVİ (Her Biri 2.0 Puan = 100 Puan)
# =============================================================================

BENCHMARK_TASKS = [
    # -------------------------------------------------------------------------
    # KATEGORI 1: PYTHON SYNTAX & TEMEL KODLAMA (10 Soru x 2.0 Puan = 20 Puan)
    # -------------------------------------------------------------------------
    {
        "id": "py_01_reverse_str",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def reverse_string(s):\n    \"\"\"Return the reversed string.\"\"\"\n",
        "criteria": [
            ("slice_or_rev", r"(\[::-1\]|reversed\(|for\s+|join)", 1.2, "Ters çevirme ifadesi"),
            ("return_val", r"return\s+", 0.8, "Sonuç döndürme")
        ]
    },
    {
        "id": "py_02_find_max",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def find_max(numbers):\n    \"\"\"Find maximum number in list.\"\"\"\n",
        "criteria": [
            ("max_logic", r"(max\(|for\s+|if\s+\w+\s*>\s*\w+|return\s+max)", 1.2, "Maksimum arama mantığı"),
            ("return_val", r"return\s+", 0.8, "Dönüş ifadesi")
        ]
    },
    {
        "id": "py_03_count_vowels",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def count_vowels(text):\n    \"\"\"Count vowels in text.\"\"\"\n    vowels = 'aeiou'\n",
        "criteria": [
            ("loop_or_sum", r"(for\s+\w+\s+in\s+text|sum\(|in\s+vowels)", 1.2, "Sesli harf sayma döngüsü"),
            ("return_val", r"return\s+", 0.8, "Sayaç dönüşü")
        ]
    },
    {
        "id": "py_04_bank_account_oop",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "class BankAccount:\n    def __init__(self, owner, balance=0):\n        self.owner = owner\n        self.balance = balance\n\n    def deposit(self, amount):\n",
        "criteria": [
            ("balance_add", r"self\.balance\s*\+=\s*amount", 1.2, "Bakiye artırımı"),
            ("return_val", r"(return\s+self\.balance|return\s+amount|self\.balance)", 0.8, "Bakiye güncelleme/döndürme")
        ]
    },
    {
        "id": "py_05_list_comprehension",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def get_squares(n):\n    \"\"\"Return list of squares from 1 to n using list comprehension.\"\"\"\n    return [",
        "criteria": [
            ("comp_logic", r"x\s*\*\*\s*2|i\s*\*\s*i|x\s*\*\s*x", 1.2, "Kare alma ifadesi"),
            ("range_iter", r"for\s+\w+\s+in\s+range", 0.8, "Range döngüsü")
        ]
    },
    {
        "id": "py_06_try_except",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def safe_divide(a, b):\n    \"\"\"Safely divide a by b, return None on ZeroDivisionError.\"\"\"\n    try:\n",
        "criteria": [
            ("div_logic", r"return\s+a\s*/\s*b", 1.0, "Bölme işlemi"),
            ("except_block", r"except\s+ZeroDivisionError:\s*\n\s*return\s+None", 1.0, "ZeroDivisionError yakalama")
        ]
    },
    {
        "id": "py_07_dict_counter",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def char_frequency(s):\n    \"\"\"Return dictionary of character frequencies.\"\"\"\n    counts = {}\n    for char in s:\n",
        "criteria": [
            ("dict_inc", r"(counts\[char\]\s*=\s*counts\.get|counts\[char\]\s*\+=|if\s+char\s+in\s+counts)", 1.2, "Sözlük frekans sayımı"),
            ("return_dict", r"return\s+counts", 0.8, "Sözlük dönüşü")
        ]
    },
    {
        "id": "py_08_file_io",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def read_file_lines(filename):\n    \"\"\"Read all lines from file using with statement.\"\"\"\n    with open(",
        "criteria": [
            ("with_open", r"with\s+open\(filename,\s*['\"]r['\"]\)\s*as\s+\w+:", 1.0, "Dosya okuma bloğu"),
            ("readlines", r"(\.readlines\(|\.read\(|for\s+line\s+in)", 1.0, "Satır okuma")
        ]
    },
    {
        "id": "py_09_lambda_sort",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def sort_by_length(words):\n    \"\"\"Sort list of strings by their length.\"\"\"\n    return sorted(words, key=",
        "criteria": [
            ("len_key", r"(len|lambda\s+\w+:\s*len\(\w+\))", 1.5, "Uzunluk anahtarı"),
            ("close_call", r"\)", 0.5, "Fonksiyon kapanışı")
        ]
    },
    {
        "id": "py_10_remove_duplicates",
        "category": "Python Syntax",
        "weight": 2.0,
        "prompt": "def remove_duplicates(items):\n    \"\"\"Remove duplicates while preserving or using set.\"\"\"\n",
        "criteria": [
            ("set_or_loop", r"(return\s+list\(set\(items\)\)|set\(|seen\s*=|for\s+item\s+in)", 1.5, "Tekrarları temizleme"),
            ("return_res", r"return\s+", 0.5, "Dönüş ifadesi")
        ]
    },

    # -------------------------------------------------------------------------
    # KATEGORI 2: VERİ YAPILARI & ALGORİTMALAR (10 Soru x 2.0 Puan = 20 Puan)
    # -------------------------------------------------------------------------
    {
        "id": "algo_01_binary_search",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "def binary_search(arr, target):\n    \"\"\"Binary search algorithm.\"\"\"\n    low = 0\n    high = len(arr) - 1\n",
        "criteria": [
            ("while_loop", r"while\s+low\s*<=\s*high:", 0.8, "While döngüsü"),
            ("mid_calc", r"mid\s*=\s*\(?low\s*\+\s*high\)?\s*//\s*2", 0.6, "Orta nokta hesabı"),
            ("branching", r"(arr\[mid\]\s*==\s*target|low\s*=\s*mid\s*\+\s*1|high\s*=\s*mid\s*-\s*1)", 0.6, "Arama dallanması")
        ]
    },
    {
        "id": "algo_02_bubble_sort",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "def bubble_sort(arr):\n    \"\"\"Bubble sort algorithm.\"\"\"\n    n = len(arr)\n    for i in range(n):\n",
        "criteria": [
            ("inner_loop", r"for\s+j\s+in\s+range\(", 1.0, "İç döngü"),
            ("swap_cond", r"(if\s+arr\[j\]\s*>\s*arr\[j\s*\+\s*1\]|arr\[j\],\s*arr\[j\+1\]\s*=)", 1.0, "Takas mantığı")
        ]
    },
    {
        "id": "algo_03_quick_sort",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "def quick_sort(arr):\n    \"\"\"Quick sort algorithm.\"\"\"\n    if len(arr) <= 1:\n        return arr\n",
        "criteria": [
            ("pivot_pick", r"pivot\s*=\s*arr\[", 0.8, "Pivot seçimi"),
            ("partitions", r"(left|right|less|greater|\[x\s+for\s+x\s+in\s+arr)", 0.6, "Bölüntüleme"),
            ("recursive_call", r"quick_sort\(", 0.6, "Özyineleme")
        ]
    },
    {
        "id": "algo_04_palindrome_check",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "def is_palindrome(s):\n    \"\"\"Check if string is a palindrome.\"\"\"\n",
        "criteria": [
            ("clean_or_compare", r"(s\s*==\s*s\[::-1\]|while\s+left\s*<\s*right|return\s+s\s*==)", 1.5, "Palindrom karşılaştırması"),
            ("return_bool", r"return\s+(True|False|s\s*==)", 0.5, "Boolean dönüşü")
        ]
    },
    {
        "id": "algo_05_stack_data_structure",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n",
        "criteria": [
            ("pop_logic", r"(return\s+self\.items\.pop\(\)|self\.items\.pop\(\))", 1.2, "Pop işlemi"),
            ("empty_check", r"(if\s+not\s+self\.items|len\(self\.items\)\s*==\s*0|return\s+None|return\s+self\.items\.pop)", 0.8, "Boşluk kontrolü / pop")
        ]
    },
    {
        "id": "algo_06_queue_data_structure",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, item):\n        self.items.append(item)\n    def dequeue(self):\n",
        "criteria": [
            ("dequeue_logic", r"(return\s+self\.items\.pop\(0\)|self\.items\.pop\(0\))", 1.5, "Baştan çıkarma mantığı (pop(0))"),
            ("return_val", r"return\s+", 0.5, "Sonuç döndürme")
        ]
    },
    {
        "id": "algo_07_two_sum",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "def two_sum(nums, target):\n    \"\"\"Find indices of two numbers that add up to target.\"\"\"\n    seen = {}\n    for i, num in enumerate(nums):\n",
        "criteria": [
            ("complement", r"(target\s*-\s*num|diff\s*=|complement\s*=)", 1.0, "Fark/tümleyen hesabı"),
            ("dict_lookup", r"(if\s+(target\s*-\s*num|diff|complement)\s+in\s+seen|seen\[num\]\s*=\s*i)", 1.0, "Hash tablosu kontrolü")
        ]
    },
    {
        "id": "algo_08_merge_sorted_lists",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "def merge_sorted(l1, l2):\n    \"\"\"Merge two sorted lists into one.\"\"\"\n    result = []\n    i = j = 0\n    while i < len(l1) and j < len(l2):\n",
        "criteria": [
            ("compare_append", r"(if\s+l1\[i\]\s*<=?\s*l2\[j\]|result\.append)", 1.2, "Karşılaştırma ve ekleme"),
            ("extend_remain", r"(result\.extend|result\s*\+=|while\s+i\s*<|while\s+j\s*<)", 0.8, "Kalanları birleştirme")
        ]
    },
    {
        "id": "algo_09_linked_list_node",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n",
        "criteria": [
            ("node_init", r"self\.val\s*=\s*val", 1.0, "Node değer ataması"),
            ("next_pointer", r"self\.next\s*=\s*next", 1.0, "Next gösterici ataması")
        ]
    },
    {
        "id": "algo_10_linear_search",
        "category": "Algorithms",
        "weight": 2.0,
        "prompt": "def linear_search(arr, target):\n    \"\"\"Return index of target or -1 if not found.\"\"\"\n",
        "criteria": [
            ("loop_enum", r"(for\s+i,\s*\w+\s+in\s+enumerate|for\s+i\s+in\s+range)", 1.0, "Doğrusal arama döngüsü"),
            ("return_idx_or_neg", r"(return\s+i|return\s+-1)", 1.0, "İndeks veya -1 dönüşü")
        ]
    },

    # -------------------------------------------------------------------------
    # KATEGORI 3: MATEMATİK & SAYISAL AKIL YÜRÜTME (10 Soru x 2.0 Puan = 20 Puan)
    # -------------------------------------------------------------------------
    {
        "id": "math_01_is_prime",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def is_prime(n):\n    \"\"\"Check if an integer is a prime number.\"\"\"\n",
        "criteria": [
            ("base_case", r"(n\s*<=?\s*1|n\s*<\s*2|return\s+False)", 0.8, "1 ve altı taban kontrolü"),
            ("div_loop", r"(for\s+\w+\s+in\s+range|while)", 0.8, "Bölünebilirlik döngüsü"),
            ("return_bool", r"return\s+(True|False)", 0.4, "Boolean dönüş")
        ]
    },
    {
        "id": "math_02_factorial",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def factorial(n):\n    \"\"\"Calculate n factorial (n!).\"\"\"\n",
        "criteria": [
            ("base_case", r"(if\s+n\s*<=?\s*1|if\s+n\s*==\s*0|return\s+1)", 1.0, "Faktöriyel 0/1 tabanı"),
            ("multiplication", r"(n\s*\*\s*factorial|result\s*\*=|math\.prod)", 1.0, "Çarpım hesabı")
        ]
    },
    {
        "id": "math_03_fibonacci",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def fibonacci(n):\n    \"\"\"Generate the nth Fibonacci number.\"\"\"\n",
        "criteria": [
            ("base_case", r"(if\s+n\s*<=?\s*1|if\s+n\s*==\s*0|return\s+n)", 1.0, "Fibonacci tabanı"),
            ("step_iter", r"(fibonacci\(n\s*-\s*1\)|a,\s*b\s*=\s*b|b,\s*a\s*\+\s*b)", 1.0, "Adım ilerletme")
        ]
    },
    {
        "id": "math_04_is_even",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def is_even(number):\n    \"\"\"Check if number is even.\"\"\"\n",
        "criteria": [
            ("modulo_2", r"(number\s*%\s*2\s*==\s*0|% 2)", 1.5, "Mod 2 kontrolü"),
            ("return_bool", r"return\s+", 0.5, "Sonuç dönüşü")
        ]
    },
    {
        "id": "math_05_gcd_euclidean",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def gcd(a, b):\n    \"\"\"Calculate Greatest Common Divisor using Euclidean algorithm.\"\"\"\n    while b != 0:\n",
        "criteria": [
            ("euclid_step", r"(a,\s*b\s*=\s*b,\s*a\s*%\s*b|temp\s*=\s*b)", 1.5, "Öklid mod takası"),
            ("return_a", r"return\s+a", 0.5, "EBOB dönüşü")
        ]
    },
    {
        "id": "math_06_arithmetic_mult",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "Question: What is 15 multiplied by 4, and what is half of that result?\nAnswer: 15 multiplied by 4 is",
        "criteria": [
            ("sixty", r"(60)", 1.2, "60 tespiti"),
            ("thirty", r"(30)", 0.8, "Yarısının 30 tespiti")
        ]
    },
    {
        "id": "math_07_power_func",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def power(base, exp):\n    \"\"\"Calculate base raised to exp.\"\"\"\n",
        "criteria": [
            ("power_op", r"(base\s*\*\*\s*exp|pow\(base|result\s*\*=\s*base)", 1.5, "Üs alma işlemi"),
            ("return_res", r"return\s+", 0.5, "Dönüş ifadesi")
        ]
    },
    {
        "id": "math_08_is_power_of_two",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def is_power_of_two(n):\n    \"\"\"Check if n is a power of 2 using bitwise operation.\"\"\"\n    return",
        "criteria": [
            ("bitwise_check", r"(n\s*&\s*\(n\s*-\s*1\)\s*==\s*0|n\s*>=\s*1)", 1.5, "Bitwise n & (n-1) == 0 kontrolü"),
            ("positive_check", r"(n\s*>\s*0|and)", 0.5, "Pozitiflik şartı")
        ]
    },
    {
        "id": "math_09_sum_of_digits",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def sum_of_digits(n):\n    \"\"\"Calculate sum of digits of an integer.\"\"\"\n    return sum(",
        "criteria": [
            ("digit_sum", r"(int\(d\)|int\(c\)|int\(x\)|int\(char\))", 1.2, "Rakam int dönüşümü"),
            ("str_iter", r"for\s+\w+\s+in\s+str\(n\)", 0.8, "Stringe çevirip yineleme")
        ]
    },
    {
        "id": "math_10_average_calc",
        "category": "Math & Reasoning",
        "weight": 2.0,
        "prompt": "def calculate_average(numbers):\n    \"\"\"Return average of numbers list.\"\"\"\n    if not numbers:\n        return 0\n    return",
        "criteria": [
            ("sum_len", r"sum\(numbers\)\s*/\s*len\(numbers\)", 1.5, "Toplam / Eleman sayısı"),
            ("float_div", r"/", 0.5, "Bölme operatörü")
        ]
    },

    # -------------------------------------------------------------------------
    # KATEGORI 4: BİLİM & BİLGİSAYAR MİMARİSİ (10 Soru x 2.0 Puan = 20 Puan)
    # -------------------------------------------------------------------------
    {
        "id": "sci_01_gravity",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: What is gravity and how does it work?\nAnswer: Gravity is",
        "criteria": [
            ("force_attract", r"(force|attract|mass|spacetime|curvature|Einstein|Newton)", 1.2, "Kütleçekim ve kütle"),
            ("pull_motion", r"(pulls|objects|matter|earth|downward|accelerat)", 0.8, "Çekim etkisi")
        ]
    },
    {
        "id": "sci_02_photosynthesis",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: Explain photosynthesis in plants.\nAnswer: Photosynthesis is the process where plants convert",
        "criteria": [
            ("inputs", r"(sunlight|light|carbon\s+dioxide|water|co2)", 1.0, "Işık, su, CO2"),
            ("outputs", r"(glucose|sugar|energy|oxygen|o2)", 1.0, "Glikoz, oksijen")
        ]
    },
    {
        "id": "sci_03_ram_vs_disk",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: What is the difference between RAM and a hard drive (SSD/HDD)?\nAnswer: RAM is",
        "criteria": [
            ("ram_volatile", r"(volatile|temporary|faster|fast|memory|lost)", 1.0, "RAM geçici/hızlı hafıza"),
            ("storage_perm", r"(permanent|storage|saved|drive|disk|persist|long-term)", 1.0, "Kalıcı depolama")
        ]
    },
    {
        "id": "sci_04_dna",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: What does DNA stand for and what is its role in living organisms?\nAnswer: DNA stands for",
        "criteria": [
            ("dna_name", r"(deoxyribonucleic\s+acid|genetic)", 1.2, "Deoxyribonucleic acid"),
            ("dna_role", r"(instructions|heredity|code|genes|organism|protein|biological)", 0.8, "Genetik bilgi rolü")
        ]
    },
    {
        "id": "sci_05_stack_vs_heap",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: In computer memory management, what is the difference between Stack and Heap?\nAnswer: The Stack is used for",
        "criteria": [
            ("stack_usage", r"(local|static|automatic|function\s+calls|fast|fixed|frames)", 1.0, "Stack yerel/fonksiyon hafızası"),
            ("heap_usage", r"(dynamic|objects|global|manual|allocated|pointers|larger)", 1.0, "Heap dinamik bellek")
        ]
    },
    {
        "id": "sci_06_cpu_cache",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: What is CPU Cache (L1/L2/L3) and why is it needed?\nAnswer: CPU cache is a",
        "criteria": [
            ("cache_speed", r"(small|fast|high-speed|temporary|sram|latency)", 1.0, "Yüksek hızlı önbellek"),
            ("cache_purpose", r"(reduce|bottleneck|ram|fetch|processor|instructions)", 1.0, "RAM gecikmesini azaltma")
        ]
    },
    {
        "id": "sci_07_big_o",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: What is Big O notation used for in computer science?\nAnswer: Big O notation describes the",
        "criteria": [
            ("complexity", r"(complexity|time|space|performance|growth|worst-case)", 1.2, "Zaman/alan karmaşıklığı"),
            ("input_growth", r"(input|size|scale|n|asymptotic|algorithm)", 0.8, "Girdi büyüdükçe ölçeklenme")
        ]
    },
    {
        "id": "sci_08_speed_of_light",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: What is the speed of light in a vacuum?\nAnswer: The speed of light in a vacuum is approximately",
        "criteria": [
            ("speed_val", r"(300,?000|3\s*x\s*10\^?8|299,?792)", 1.2, "300.000 km/s veya 3x10^8 m/s"),
            ("speed_unit", r"(km/s|m/s|meters\s+per\s+second|kilometers)", 0.8, "Birim doğruluğu")
        ]
    },
    {
        "id": "sci_09_water_formula",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: What is the chemical formula for water and what elements does it contain?\nAnswer: Water is",
        "criteria": [
            ("h2o", r"(H2O|H_2O)", 1.2, "H2O formülü"),
            ("elements", r"(hydrogen|oxygen|hidrojen|oksijen)", 0.8, "Hidrojen ve oksijen")
        ]
    },
    {
        "id": "sci_10_compiler_vs_interpreter",
        "category": "Science Knowledge",
        "weight": 2.0,
        "prompt": "Question: What is the difference between a Compiler and an Interpreter?\nAnswer: A compiler translates",
        "criteria": [
            ("compiler_action", r"(entire|whole|all\s+at\s+once|machine\s+code|binary|executable)", 1.0, "Tüm kodu makine koduna derleme"),
            ("interpreter_action", r"(line\s+by\s+line|statement|runtime|directly|executes)", 1.0, "Satır satır yorumlama")
        ]
    },

    # -------------------------------------------------------------------------
    # KATEGORI 5: TALİMAT SADAKATİ & YAPISAL ÇIKTI (10 Soru x 2.0 Puan = 20 Puan)
    # -------------------------------------------------------------------------
    {
        "id": "inst_01_json_status",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "Generate a valid JSON object with keys 'status' (value 'success') and 'code' (value 200):\n{\n",
        "criteria": [
            ("status_val", r"\"status\"\s*:\s*\"success\"", 1.0, "'status': 'success'"),
            ("code_val", r"\"code\"\s*:\s*200", 1.0, "'code': 200")
        ]
    },
    {
        "id": "inst_02_chatml_capital",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "criteria": [
            ("paris_answer", r"Paris", 1.2, "Paris cevabı"),
            ("no_endless_loop", r"(Paris|capital\s+of\s+France)", 0.8, "Bağlam tutarlılığı")
        ]
    },
    {
        "id": "inst_03_summary_one_sentence",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "Summarize in one sentence: 'Artificial neural networks are computing systems inspired by biological neural networks in animal brains.'\nSummary:",
        "criteria": [
            ("neural_kw", r"(neural|brain|network|comput)", 1.2, "Anahtar kavramlar"),
            ("sentence_fmt", r"(\.\s*$|\n|[A-Za-z0-9\s]{10,120})", 0.8, "Cümle formu")
        ]
    },
    {
        "id": "inst_04_bullet_points",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "List 3 advantages of using Python:\n1.",
        "criteria": [
            ("item_1", r"1\..+", 0.8, "1. Madde"),
            ("item_2", r"(2\..+|\n2\.)", 0.6, "2. Madde"),
            ("py_traits", r"(easy|readable|simple|librar|community|syntax)", 0.6, "Python avantajları")
        ]
    },
    {
        "id": "inst_05_boolean_only",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "Is the sun a star? Answer with only True or False.\nAnswer:",
        "criteria": [
            ("true_ans", r"\bTrue\b", 1.5, "True yanıtı"),
            ("concise", r"(^[\s\n]*True[\s\n\.]*$|True)", 0.5, "Net cevap")
        ]
    },
    {
        "id": "inst_06_markdown_code_block",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "Write a python hello world inside markdown code blocks:\n```python\n",
        "criteria": [
            ("print_hello", r"print\s*\(\s*['\"]Hello[,\s]*World!?['\"]\s*\)", 1.2, "print('Hello World')"),
            ("close_ticks", r"```", 0.8, "Kod bloğu kapanışı (```)")
        ]
    },
    {
        "id": "inst_07_type_hints",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "def add_numbers(x: int, y: int) -> int:\n    \"\"\"Add two integers with type hints.\"\"\"\n",
        "criteria": [
            ("typed_return", r"return\s+x\s*\+\s*y", 1.5, "Toplama dönüşü"),
            ("indent", r"    \w+", 0.5, "Girinti")
        ]
    },
    {
        "id": "inst_08_sql_query",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "Write a SQL query to select all users older than 18 from users table:\nSELECT",
        "criteria": [
            ("select_from", r"(SELECT\s+\*|\*)\s+FROM\s+users", 1.2, "SELECT * FROM users"),
            ("where_clause", r"WHERE\s+age\s*>\s*18", 0.8, "WHERE age > 18")
        ]
    },
    {
        "id": "inst_09_csv_format",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "Create a 2-line CSV of name and age:\nname,age\n",
        "criteria": [
            ("csv_row", r"\w+,\s*\d+", 1.5, "İsim,Yaş formatı"),
            ("delimiter", r",", 0.5, "Virgül ayrımı")
        ]
    },
    {
        "id": "inst_10_opposite_word",
        "category": "Instruction Adherence",
        "weight": 2.0,
        "prompt": "What is the opposite of 'hot'? Answer with one word:\n",
        "criteria": [
            ("cold_ans", r"\b(cold|Cold)\b", 1.5, "Cold cevabı"),
            ("single_word", r"(cold|Cold)", 0.5, "Tek kelime")
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


def generate_text(model, tokenizer, prompt, max_new_tokens=90, temperature=0.5, top_k=25):
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    gen_tokens = out[0].tolist()
    full_text = tokenizer.decode(gen_tokens)
    new_tokens = gen_tokens[len(input_ids):]
    generated_completion = tokenizer.decode(new_tokens)
    
    return full_text, generated_completion


def evaluate_task(task, full_text, completion):
    task_score = 0.0
    rubric_results = []
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
            
    task_score = min(task_score, task["weight"])
    return task_score, rubric_results


def run_benchmark(model_path, output_json=None):
    print("=" * 80)
    print(f"🏛️  DORINA GRAND STANDARDIZED BENCHMARK SUITE (50 SORULUK ELİT SINAV)")
    print(f"📦 Model: {model_path}")
    print("=" * 80)
    
    start_time = time.time()
    model, tokenizer, checkpoint = load_model_and_tokenizer(model_path)
    
    category_scores = {
        "Python Syntax": 0.0,
        "Algorithms": 0.0,
        "Math & Reasoning": 0.0,
        "Science Knowledge": 0.0,
        "Instruction Adherence": 0.0
    }
    
    detailed_results = []
    
    for idx, task in enumerate(BENCHMARK_TASKS, 1):
        full_text, completion = generate_text(
            model, tokenizer, task["prompt"],
            max_new_tokens=90, temperature=0.5, top_k=25
        )
        
        task_score, rubrics = evaluate_task(task, full_text, completion)
        category_scores[task["category"]] += task_score
        
        # Konsolda temiz tek satır ilerleme
        print(f"[{idx:02d}/50] ({task['category'][:12]:<12}) {task['id']:<20} -> {task_score:.1f}/2.0 P")
        
        detailed_results.append({
            "id": task["id"],
            "category": task["category"],
            "score": round(task_score, 2),
            "max_score": task["weight"],
            "prompt": task["prompt"],
            "completion": completion.strip()[:120],
            "rubrics": rubrics
        })
        
    total_score = sum(category_scores.values())
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"🏆 50 SORULUK BÜYÜK SINAV KARNESİ (MODEL: {os.path.basename(os.path.dirname(model_path))})")
    print("=" * 80)
    print(f"1. Python Syntax & Temel Kodlama : {category_scores['Python Syntax']:5.1f} / 20.0 Puan")
    print(f"2. Veri Yapıları & Algoritmalar  : {category_scores['Algorithms']:5.1f} / 20.0 Puan")
    print(f"3. Matematik & Sayısal Muhakeme : {category_scores['Math & Reasoning']:5.1f} / 20.0 Puan")
    print(f"4. Bilim & Bilgisayar Mimarisi  : {category_scores['Science Knowledge']:5.1f} / 20.0 Puan")
    print(f"5. Talimat Takibi & Biçim       : {category_scores['Instruction Adherence']:5.1f} / 20.0 Puan")
    print("-" * 80)
    print(f"🎯 TOPLAM GENEL ZEKA PUANI      : {total_score:5.1f} / 100.0 PUAN")
    print(f"⏱️  Test Süresi                  : {elapsed:.1f} saniye")
    print("=" * 80)
    
    val_loss = None
    step = None
    if isinstance(checkpoint, dict):
        val_loss = checkpoint.get("val_loss", None)
        step = checkpoint.get("step", None)
        
    benchmark_report = {
        "model_path": model_path,
        "run_name": os.path.basename(os.path.dirname(model_path)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_score": round(total_score, 2),
        "max_score": 100.0,
        "category_scores": {k: round(v, 2) for k, v in category_scores.items()},
        "val_loss": val_loss,
        "step": step,
        "detailed_results": detailed_results
    }
    
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
        print(f"✓ Detaylı 50 soruluk rapor kaydedildi: {output_json}")
        
    return benchmark_report


if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT_DIR, "models", "dorina_44m_moe", "run5", "model.pt")
    target_out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(target_model), "benchmark_score_50q.json")
    
    run_benchmark(target_model, target_out)
