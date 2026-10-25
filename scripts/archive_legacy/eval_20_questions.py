import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import tiktoken
from src.model import DorinaGPT
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Model ve Tokenizer Yükleme
print("=" * 70)
print("🤖 DORINA-278M KAPSAMLI DEĞERLENDİRME VE 20 SORULUK TEST SÜİTİ")
print("=" * 70)

enc = tiktoken.get_encoding("gpt2")
checkpoint_path = os.path.join(ROOT_DIR, "runs", "dorina_v0.14_278m", "run3", "best.pt")
if not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(ROOT_DIR, "releases", "dorina_v0.14_278m", "model.pt")

print(f"📦 Model Yükleniyor: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=device)
cfg = ckpt.get("config", {})

model = DorinaGPT(
    vocab_size=cfg.get("vocab_size", 50257),
    context_length=cfg.get("context_length", 256),
    n_embd=cfg.get("n_embd", 1024),
    n_head=cfg.get("n_head", 16),
    n_layer=cfg.get("n_layer", 18),
    dropout=0.0
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"✓ Model başarıyla yüklendi! Parametre: ~278M | En İyi Val Loss: {ckpt.get('val_loss', 'N/A')}\n")

# 20 Adet Test Sorusu / Promptu:
# - 7 Adet Python & Algoritma
# - 7 Adet Dünya Bilgisi, Bilim & Mantık
# - 6 Adet Sohbet, Diyalog & Akıl Yürütme
TEST_PROMPTS = [
    # --- Python & Kodlama ---
    {"cat": "Kod (Fonksiyon)", "prompt": "def reverse_string(s):\n    \"\"\"Return the reversed version of string s.\"\"\"\n"},
    {"cat": "Kod (Algoritma)", "prompt": "def bubble_sort(arr):\n    # Sort array in ascending order\n"},
    {"cat": "Kod (Matematik)", "prompt": "def fibonacci(n):\n    # Return the nth fibonacci number\n"},
    {"cat": "Kod (Sınıf/OOP)", "prompt": "class Rectangle:\n    def __init__(self, width, height):\n"},
    {"cat": "Kod (Dosya/IO)", "prompt": "def read_file_lines(filename):\n    with open(filename, 'r') as f:\n"},
    {"cat": "Kod (Problem Çözme)", "prompt": "### Instruction:\nWrite a python function to find the maximum number in a list.\n### Response:\n"},
    {"cat": "Kod (Veri Yapısı)", "prompt": "class Queue:\n    def __init__(self):\n        self.items = []\n"},
    
    # --- Bilim, Dünya Bilgisi & Mantık ---
    {"cat": "Bilim (Fizik)", "prompt": "Gravity is the fundamental force that"},
    {"cat": "Bilim (Kimya)", "prompt": "Water is composed of hydrogen and oxygen molecules which"},
    {"cat": "Tarih (Medeniyet)", "prompt": "The Industrial Revolution dramatically transformed human society by"},
    {"cat": "Biyoloji (Hücre)", "prompt": "DNA contains the genetic instructions used in"},
    {"cat": "Coğrafya (Gezegen)", "prompt": "The solar system consists of the Sun and"},
    {"cat": "Felsefe & Mantık", "prompt": "Artificial intelligence will shape the future of work because"},
    {"cat": "Bilgisayar Bilimi", "prompt": "An algorithm is defined as a step-by-step procedure to"},

    # --- Sohbet, Talimat & İletişim ---
    {"cat": "Sohbet (Selamlama)", "prompt": "User: Hello! Who are you?\nDorina:"},
    {"cat": "Sohbet (Yardım)", "prompt": "User: Can you explain how a computer works in simple terms?\nDorina:"},
    {"cat": "Sohbet (Tavsiye)", "prompt": "User: What is the best way to learn programming?\nDorina:"},
    {"cat": "Hikaye (Kurgu)", "prompt": "Once upon a time in a futuristic city powered by clean energy,"},
    {"cat": "Açıklama (Özet)", "prompt": "Explain the difference between a compiler and an interpreter:\n"},
    {"cat": "Mantık (Soru-Cevap)", "prompt": "Question: Why do airplanes fly?\nAnswer: Airplanes fly because their wings generate lift by"}
]

print("-" * 70)
print(f"🚀 20 TEST ÇALIŞTIRILIYOR (Temperature: 0.65, Top-k: 40)...")
print("-" * 70 + "\n")

results = []

for idx, item in enumerate(TEST_PROMPTS, 1):
    category = item["cat"]
    prompt = item["prompt"]
    
    tokens = enc.encode_ordinary(prompt)
    ctx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        out_tokens = model.generate(ctx, max_new_tokens=45, temperature=0.65)[0].tolist()
    
    generated_text = enc.decode(out_tokens)
    
    print(f"[{idx}/20] Kategori: {category}")
    print(f"PROMPT:\n{prompt}")
    print(f"DORINA ÇIKTISI:\n{generated_text}")
    print("-" * 70 + "\n")
    
    results.append({
        "index": idx,
        "category": category,
        "prompt": prompt,
        "output": generated_text
    })

# Raporu kaydet
report_path = os.path.join(ROOT_DIR, "eval_20_questions_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("DORINA-278M: 20 SORULUK KAPSAMLI PERFORMANS DEĞERLENDİRME RAPORU\n")
    f.write(f"Model: dorina_v0.14_278m (Run 3 Final)\n")
    f.write("=" * 70 + "\n\n")
    for r in results:
        f.write(f"[{r['index']}/20] Kategori: {r['category']}\n")
        f.write(f"PROMPT:\n{r['prompt']}\n")
        f.write(f"ÇIKTI:\n{r['output']}\n")
        f.write("-" * 70 + "\n\n")

print(f"📄 Tüm test sonuçları kaydedildi ➔ {report_path}")
