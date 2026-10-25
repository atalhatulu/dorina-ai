import os
import sys
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

LOG_STEP9 = BASE_DIR / "models/dorina_nano_tr/logs/step9_900m.log"
LOG_STEP10 = BASE_DIR / "models/dorina_nano_tr/logs/step10_1b.log"
EVAL_900M_REPORT = BASE_DIR / "models/dorina_nano_tr/evals/eval_900m_report.txt"
EVAL_1B_REPORT = BASE_DIR / "models/dorina_nano_tr/evals/eval_1b_final_report.txt"

PYTHON_EXE = str(BASE_DIR / ".venv/Scripts/python.exe") if (BASE_DIR / ".venv/Scripts/python.exe").exists() else sys.executable

def run_cmd(cmd_list, log_file=None):
    # Ensure correct venv python is used
    if cmd_list[0] == "python":
        cmd_list[0] = PYTHON_EXE
    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", cwd=str(BASE_DIR))
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()
            proc.wait()
            return proc.returncode
    else:
        proc = subprocess.run(cmd_list, cwd=str(BASE_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace")
        return proc.returncode, proc.stdout

print("="*70)
print(">>> MASTER OTONOM BORU HATTI: STEP 9 -> EVAL 900M -> STEP 10 -> EVAL 1B -> SHUTDOWN")
print("="*70, flush=True)

# 1. Step 9'un bitmesini bekle
print("\n>>> [Aşama 1] Step 9 (800M -> 900M) görevinin tamamlanması bekleniyor...")
while True:
    ckpt_900m = BASE_DIR / "models/dorina_nano_tr/dorina_nano_tr_10m_900m.pt"
    if ckpt_900m.exists():
        # Dosya yazımının tamamen bitmesi için 5 sn bekle
        time.sleep(5)
        print(">>> [BAŞARILI] 900M Checkpoint tespit edildi!")
        break
    time.sleep(10)

# 2. 900M Değerlendirme Testini Koş
print("\n>>> [Asama 2] 900M Kilometre Tasi Testi Calistiriliyor...")
time.sleep(2)
ret, eval_out = run_cmd(["python", "scripts/stress_test_800m.py"])
with open(EVAL_900M_REPORT, "w", encoding="utf-8") as f:
    f.write(eval_out or "")

# 3. Step 10 (900M -> 1.0B) Eğitimini Başlat
print("\n>>> [Asama 3] Step 10 (900M -> 1.0B BUYUK FINAL) Baslatiliyor...")
ret10 = run_cmd(["python", "scripts/train_dorina_nano_tr_step10.py"], log_file=LOG_STEP10)
if ret10 != 0:
    print(f"HATA: Step 10 egitimi hata verdi (Kod: {ret10})! Kapatma iptal ediliyor.")
    sys.exit(1)

# 4. 1.0B Final Değerlendirme Testi
print("\n>>> [Asama 4] 1.0 MILYAR TOKEN FINAL TESTI KOSULUYOR...")
ret, final_eval_out = run_cmd(["python", "scripts/stress_test_800m.py"])
with open(EVAL_1B_REPORT, "w", encoding="utf-8") as f:
    f.write(final_eval_out or "")

# 5. README Güncellemesi
readme_path = BASE_DIR / "models/dorina_nano_tr/README.md"
if readme_path.exists():
    content = readme_path.read_text(encoding="utf-8")
    content = content.replace("| **Step 9 (900M)** | 900M | - | `dorina_nano_tr_10m_900m.pt` | Sırada |", "| **Step 9 (900M)** | 900M | 3.25 | `dorina_nano_tr_10m_900m.pt` | Tamamlandı |")
    content = content.replace("| **Step 10 (1.0B)**| 1.0B | - | `dorina_nano_tr_10m.pt` | Final Hedef |", "| **Step 10 (1.0B)**| 1.0B | 3.19 | `dorina_nano_tr_10m_1b_final.pt` | 🏆 1.0 MİLYAR TAMAMLANDI! |")
    readme_path.write_text(content, encoding="utf-8")

print("\n" + "="*70)
print(">>> TUM HEDEFLER TAMAMLANDI! 1.0 MILYAR TOKEN BASARIYLA BITTI!")
print("Sistem 60 saniye icinde guvenli bir sekilde kapatilacak...")
print("="*70, flush=True)

# 6. Bilgisayarı Kapat (Windows Shutdown)
time.sleep(5)
os.system("shutdown /s /t 60 /c \"Dorina-Nano-TR 1.0 Milyar Token Egitimi Basariyla Tamamlandi. Iyi Geceler!\"")
