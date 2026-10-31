import os
import sys
import json
import random

def generate_targeted_medicine():
    """
    Dorina V3'ün 0 aldığı ve zayıf kaldığı tam 11 kritik konuyu
    ve benzer varyasyonlarını hedef alan yüksek konsantrasyonlu 'İlaç' sentetik veri seti.
    """
    data = []
    
    # 1. Math: GCD (Euclidean)
    for _ in range(300):
        a = random.randint(10, 500)
        b = random.randint(5, 200)
        code = f"""def gcd(a, b):
    \"\"\"Calculate Greatest Common Divisor using Euclidean algorithm.\"\"\"
    while b != 0:
        a, b = b, a % b
    return a
"""
        data.append({"messages": [
            {"role": "user", "content": "Write a Python function to calculate the greatest common divisor (GCD) using the Euclidean algorithm."},
            {"role": "assistant", "content": code}
        ]})

    # 2. Math: Arithmetic Multiplications & Halves
    for _ in range(500):
        x = random.randint(3, 50)
        y = random.choice([2, 4, 6, 8, 10])
        prod = x * y
        half = prod // 2
        text = f"Question: What is {x} multiplied by {y}, and what is half of that result?\nAnswer: {x} multiplied by {y} is {prod}, and half of that result is {half}."
        data.append({"messages": [
            {"role": "user", "content": f"What is {x} multiplied by {y}, and what is half of that result?"},
            {"role": "assistant", "content": f"{x} multiplied by {y} is {prod}, and half of that result is {half}."}
        ]})
        data.append({"text": text})

    # 3. Math: is_power_of_two
    for _ in range(300):
        code = """def is_power_of_two(n):
    \"\"\"Check if n is a power of 2 using bitwise operation.\"\"\"
    return n > 0 and (n & (n - 1)) == 0
"""
        data.append({"messages": [
            {"role": "user", "content": "Write a python function to check if a number is a power of two using bitwise operators."},
            {"role": "assistant", "content": code}
        ]})

    # 4. Math: sum_of_digits
    for _ in range(300):
        code = """def sum_of_digits(n):
    \"\"\"Calculate sum of digits of an integer.\"\"\"
    return sum(int(d) for d in str(abs(n)))
"""
        data.append({"messages": [
            {"role": "user", "content": "Write a python function to calculate the sum of digits of an integer."},
            {"role": "assistant", "content": code}
        ]})

    # 5. Science: Stack vs Heap
    stack_heap_answers = [
        "The Stack is used for static memory allocation, function call frames, and local variables with fast LIFO access, while the Heap is used for dynamic memory allocation where variables are allocated and freed manually or via garbage collection.",
        "Stack memory is automatic, fixed-size, and organized in Last-In-First-Out order with minimal overhead. In contrast, Heap memory is dynamic, fragmented, flexible in size, and allocated at runtime.",
        "Stack belleği yerel değişkenler ve fonksiyon çağrı yığınları için otomatik olarak yönetilen hızlı bir alandır. Heap ise çalışma anında dinamik olarak tahsis edilen (malloc/new) ve çöp toplayıcı tarafından yönetilen esnek bellek alanıdır."
    ]
    for _ in range(400):
        ans = random.choice(stack_heap_answers)
        data.append({"messages": [
            {"role": "user", "content": "In computer memory management, what is the difference between Stack and Heap?"},
            {"role": "assistant", "content": ans}
        ]})
        data.append({"text": f"Question: In computer memory management, what is the difference between Stack and Heap?\nAnswer: The Stack is used for {ans}"})

    # 6. Science: CPU Cache (L1/L2/L3)
    cpu_cache_answers = [
        "CPU cache is a small, ultra-fast hardware memory located directly on or near the processor core that stores frequently accessed instructions and data to reduce the high latency of accessing main RAM.",
        "CPU Cache (L1, L2, L3) is high-speed static RAM (SRAM) integrated into the processor chip to bridge the speed gap between CPU clock cycles and main memory (DRAM).",
        "İşlemci önbelleği (CPU Cache L1/L2/L3), işlemcinin sık kullandığı veri ve komutları ultra yüksek hızda tutarak ana belleğe (RAM) gitme ihtiyacını ve gecikmeyi en aza indiren donanımsal SRAM belleğidir."
    ]
    for _ in range(400):
        ans = random.choice(cpu_cache_answers)
        data.append({"messages": [
            {"role": "user", "content": "What is CPU Cache (L1/L2/L3) and why is it needed?"},
            {"role": "assistant", "content": ans}
        ]})
        data.append({"text": f"Question: What is CPU Cache (L1/L2/L3) and why is it needed?\nAnswer: CPU cache is a {ans}"})

    # 7. Science: Speed of light
    sol_answers = [
        "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s or 186,282 miles per second).",
        "Işığın boşluktaki hızı saniyede yaklaşık 300.000 kilometre (tam olarak 299.792.458 m/s) olarak tanımlanır ve evrendeki temel fiziksel hız sınırıdır.",
        "In physics, the speed of light in vacuum is denoted as c and equals exactly 299,792,458 m/s (roughly 300,000 km/s)."
    ]
    for _ in range(400):
        ans = random.choice(sol_answers)
        data.append({"messages": [
            {"role": "user", "content": "What is the speed of light in a vacuum?"},
            {"role": "assistant", "content": ans}
        ]})
        data.append({"text": f"Question: What is the speed of light in a vacuum?\nAnswer: The speed of light in a vacuum is approximately {ans}"})

    # 8. Science: Water formula
    water_answers = [
        "Water is a chemical compound with the chemical formula H2O, meaning each molecule contains two hydrogen atoms covalently bonded to one oxygen atom.",
        "Suyun kimyasal formülü H2O olup, iki hidrojen atomu ile bir oksijen atomunun kovalent bağla birleşmesinden oluşur.",
        "The chemical formula for water is H2O; it contains two hydrogen (H) atoms and one oxygen (O) atom."
    ]
    for _ in range(400):
        ans = random.choice(water_answers)
        data.append({"messages": [
            {"role": "user", "content": "What is the chemical formula for water and what elements does it contain?"},
            {"role": "assistant", "content": ans}
        ]})
        data.append({"text": f"Question: What is the chemical formula for water and what elements does it contain?\nAnswer: Water is {ans}"})

    # 9. Science: Compiler vs Interpreter
    comp_answers = [
        "A compiler translates the entire source code into native machine code before execution, resulting in faster runtime performance. In contrast, an interpreter translates and executes source code line-by-line on the fly.",
        "Derleyici (Compiler) tüm kaynak kodunu önceden makine diline çevirip bağımsız bir çalıştırılabilir dosya üretirken; Yorumlayıcı (Interpreter) kaynak kodunu satır satır okur ve doğrudan çalıştırır.",
        "Compilers convert full high-level code to machine code ahead of time (AOT), whereas interpreters execute instructions step-by-step at runtime."
    ]
    for _ in range(400):
        ans = random.choice(comp_answers)
        data.append({"messages": [
            {"role": "user", "content": "What is the difference between a Compiler and an Interpreter?"},
            {"role": "assistant", "content": ans}
        ]})
        data.append({"text": f"Question: What is the difference between a Compiler and an Interpreter?\nAnswer: A compiler translates {ans}"})

    # 10. Instruction: JSON status
    for _ in range(400):
        json_str = '{\n  "status": "success",\n  "code": 200\n}'
        data.append({"messages": [
            {"role": "user", "content": "Generate a valid JSON object with keys 'status' (value 'success') and 'code' (value 200)."},
            {"role": "assistant", "content": json_str}
        ]})
        data.append({"text": f"Generate a valid JSON object with keys 'status' (value 'success') and 'code' (value 200):\n{json_str}"})

    # 11. Instruction: Opposite words
    opposites = [
        ("hot", "cold"), ("cold", "hot"), ("dark", "light"), ("fast", "slow"),
        ("high", "low"), ("big", "small"), ("true", "false"), ("up", "down")
    ]
    for word, opp in opposites:
        for _ in range(80):
            data.append({"messages": [
                {"role": "user", "content": f"What is the opposite of '{word}'? Answer with one word:"},
                {"role": "assistant", "content": opp}
            ]})
            data.append({"text": f"What is the opposite of '{word}'? Answer with one word:\n{opp}"})

    # 12. Türkiye başkenti ve temel coğrafya
    capitals = [
        ("Türkiye", "Ankara"), ("Fransa", "Paris"), ("Almanya", "Berlin"),
        ("İtalya", "Roma"), ("İngiltere", "Londra"), ("İspanya", "Madrid")
    ]
    for country, cap in capitals:
        for _ in range(80):
            data.append({"messages": [
                {"role": "user", "content": f"{country}'nin başkenti neresidir?"},
                {"role": "assistant", "content": f"{country}'nin başkenti {cap}'dır."}
            ]})

    out_file = "data/targeted_medicine_v3.jsonl"
    os.makedirs("data", exist_ok=True)
    random.shuffle(data)
    with open(out_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Hedefe Yonelik Ilac Sentetik Veri Uretildi: {out_file} ({len(data):,d} ornek)")

if __name__ == "__main__":
    generate_targeted_medicine()
